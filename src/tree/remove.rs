//! Deletion operations for `MassTree`.
//!
//! This module implements the `remove()` operation following the C++
//! reference in `reference/masstree_remove.hh`.
//!
//! # Algorithm Overview
//!
//! 1. Navigate to the target leaf using optimistic traversal
//! 2. Search for the key within the leaf
//! 3. Lock the leaf and verify the key still exists
//! 4. Remove the slot from the permutation
//! 5. Retire the value via seize
//! 6. If leaf is now empty, trigger leaf removal

use std::hint as StdHint;
use std::ptr as StdPtr;
use std::sync::atomic::Ordering as AtomicOrdering;

use seize::{Guard, LocalGuard};

use crate::ksearch::upper_bound_internode_generic;
use crate::{
    TreeInternode,
    alloc_trait::NodeAllocatorGeneric,
    key::Key,
    leaf_trait::{LayerCapableLeaf, TreePermutation},
    leaf24::{KSUF_KEYLENX, LAYER_KEYLENX},
    nodeversion::{LockGuard, NodeVersion},
    slot::ValueSlot,
    tree::MassTreeGeneric,
};

// ============================================================================
//  Error Types
// ============================================================================

/// Errors that can occur during removal.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RemoveError {
    /// Retry limit exceeded during optimistic concurrency.
    ///
    /// This should be extremely rare. It indicates severe contention
    /// on the target leaf node.
    RetryLimitExceeded,
}

impl std::fmt::Display for RemoveError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::RetryLimitExceeded => write!(f, "retry limit exceeded"),
        }
    }
}

impl std::error::Error for RemoveError {}

// ============================================================================
//  Search Result Types
// ============================================================================

/// Result of searching for a key to remove.
#[derive(Debug)]
enum RemoveSearchResult {
    /// Key not found in this leaf.
    NotFound,

    /// Key found at logical position `ki`, physical slot `kp`.
    ///
    /// Note: `kp` is captured here but discarded after locking because
    /// the slot position must be re-verified under lock.
    Found {
        /// Logical position in permutation (0..size).
        ki: usize,

        /// Physical slot index (0..WIDTH).
        /// Discarded after locking - position is re-verified.
        #[allow(dead_code, reason = "Captured for debugging, verified under lock")]
        kp: usize,
    },

    /// Key might be in sublayer; descend and retry.
    DescendLayer {
        /// Pointer to the layer root.
        layer_ptr: *mut u8,

        /// Physical slot index containing the layer pointer.
        slot: usize,
    },
}

/// Result of attempting to find, lock, and verify a key for removal.
///
/// This enum separates the search/lock phase from the actual removal,
/// enabling cleaner control flow in the main remove loop.
enum RemoveLockResult<'t, 'g, S, L, A>
where
    S: ValueSlot,
    S::Value: Send + Sync + 'static,
    S::Output: Send + Sync,
    L: LayerCapableLeaf<S>,
    A: NodeAllocatorGeneric<S, L>,
{
    /// Key not found in this layer.
    NotFound,

    /// Key found and cursor is ready for removal.
    Ready(RemoveCursor<'t, 'g, S, L, A>),

    /// Need to descend into sublayer.
    DescendLayer {
        /// Pointer to the sublayer root.
        layer_ptr: *mut u8,

        /// Parent leaf containing the layer slot.
        parent_leaf: *mut u8,

        /// Physical slot in parent containing the layer pointer.
        parent_slot: usize,
    },

    /// Version changed or slot moved, retry from `reach_leaf`.
    Retry,

    /// Leaf is part of a gc'd sublayer, restart from tree root.
    ///
    /// This happens when another thread called `gc_layer` between our
    /// search and lock. The key must be unshifted before retry.
    RestartFromRoot,
}

// ============================================================================
//  Layer Context
// ============================================================================

/// Context for tracking layer descent during remove operations.
///
/// When descending into a sublayer, we track the parent leaf and slot
/// so that `gc_layer` can clear the parent's layer slot if the sublayer
/// becomes empty.
///
/// # Why This Is Needed
///
/// Sublayer roots have NULL parent pointers (they're marked as roots).
/// We cannot use `sublayer_leaf.parent()` to find the parent leaf.
/// Instead, we track this information during the layer descent.
#[allow(dead_code, reason = "Deferred: inline structural removal")]
#[derive(Debug, Clone, Copy)]
struct LayerContext {
    /// Pointer to the parent leaf that contains the layer slot.
    parent_leaf: *mut u8,

    /// Physical slot index in the parent leaf containing the layer pointer.
    parent_slot: usize,
}

// ============================================================================
//  Constants
// ============================================================================

/// Maximum retries before giving up.
const MAX_RETRIES: usize = 1000;

/// Maximum retries when locking parent during tree walk.
const MAX_PARENT_RETRIES: usize = 100;

// ============================================================================
//  RemoveCursor — Stateful cursor matching C++ tcursor pattern
// ============================================================================

/// A cursor for remove operations that holds the lock as persistent state.
///
/// This matches the C++ `tcursor` pattern where the lock is part of the cursor
/// rather than passed between functions. The cursor owns the lock and methods
/// consume `self` when they transfer lock ownership up the tree.
///
/// # C++ Reference
///
/// `masstree_tcursor.hh` defines the cursor with:
/// - `n_`: current leaf
/// - `v_`: version/lock state (persistent in cursor)
/// - `kx_.i`, `kx_.p`: position state (ki, kp)
///
/// # Lifetime
///
/// The cursor borrows from:
/// - The tree (`'t`)
/// - The guard (`'g`)
/// - The leaf is accessed through the lock
#[derive(Debug)]
pub struct RemoveCursor<'t, 'g, S, L, A>
where
    S: ValueSlot,
    S::Value: Send + Sync + 'static,
    S::Output: Send + Sync,
    L: LayerCapableLeaf<S>,
    A: NodeAllocatorGeneric<S, L>,
{
    /// Reference to the tree for allocation and root access.
    tree: &'t MassTreeGeneric<S, L, A>,

    /// The locked leaf node. Lock is held for the lifetime of the cursor.
    leaf: *mut L,

    /// The lock guard — this is the persistent state matching C++ `v_`.
    lock: LockGuard<'t>,

    /// Logical position in permutation (0..size).
    ki: usize,

    /// Physical slot index (0..WIDTH).
    kp: usize,

    /// Context for sublayer cleanup (parent leaf + slot).
    #[allow(dead_code, reason = "Deferred: inline structural removal")]
    layer_context: Option<LayerContext>,

    /// Guard for memory reclamation.
    guard: &'g LocalGuard<'g>,
}

impl<'t, 'g, S, L, A> RemoveCursor<'t, 'g, S, L, A>
where
    S: ValueSlot,
    S::Value: Send + Sync + 'static,
    S::Output: Send + Sync,
    L: LayerCapableLeaf<S>,
    A: NodeAllocatorGeneric<S, L>,
{
    /// Create a new remove cursor with the lock already held.
    ///
    /// # Arguments
    ///
    /// * `tree` - The tree being modified
    /// * `leaf` - Pointer to the locked leaf
    /// * `lock` - The lock guard (takes ownership)
    /// * `ki` - Logical position in permutation
    /// * `kp` - Physical slot index
    /// * `layer_context` - Parent info for sublayer cleanup
    /// * `guard` - Guard for memory reclamation
    #[inline(always)]
    #[expect(clippy::too_many_arguments, reason = "Complex state management")]
    const fn new(
        tree: &'t MassTreeGeneric<S, L, A>,
        leaf: *mut L,
        lock: LockGuard<'t>,
        ki: usize,
        kp: usize,
        layer_context: Option<LayerContext>,
        guard: &'g LocalGuard<'g>,
    ) -> Self {
        Self {
            tree,
            leaf,
            lock,
            ki,
            kp,
            layer_context,
            guard,
        }
    }

    /// Get a reference to the locked leaf.
    #[inline(always)]
    fn leaf(&self) -> &L {
        // SAFETY: leaf is valid and locked by us
        unsafe { &*self.leaf }
    }

    /// Complete the removal of a key from the locked leaf.
    ///
    /// This is the main entry point after finding the key. It:
    /// 1. Extracts the value for return
    /// 2. Schedules value retirement via seize
    /// 3. Clears suffix if present
    /// 4. Updates permutation
    /// 5. If leaf is now empty, triggers leaf removal
    ///
    /// # Returns
    ///
    /// The removed value (if any).
    ///
    /// # C++ Reference
    ///
    /// `masstree_remove.hh:162-176` - `finish_remove()`
    #[must_use]
    #[inline(always)]
    pub fn finish_remove(mut self) -> Option<S::Output> {
        let leaf = self.leaf();

        // Step 1: Extract the value pointer
        let value_ptr: *mut u8 = leaf.leaf_value_ptr(self.kp);

        // Step 2: Clone the value for return (before retirement)
        let value: Option<S::Output> = if value_ptr.is_null() {
            None
        } else {
            leaf.try_clone_output(self.kp)
        };

        // Step 3: Schedule value retirement
        if !value_ptr.is_null() {
            // SAFETY: value_ptr is valid (from leaf.leaf_value_ptr()), guard protects reclamation.
            unsafe {
                self.guard.defer_retire(value_ptr, |ptr, _| {
                    S::cleanup_value_ptr(ptr);
                });
            }
        }

        // Step 4: Clear suffix if present
        let slot_keylenx: u8 = leaf.keylenx(self.kp);
        if slot_keylenx == KSUF_KEYLENX {
            // SAFETY: We hold the lock on this leaf (self.lock), and self.kp is a
            // valid slot index obtained during search. The guard protects memory.
            unsafe { leaf.clear_ksuf(self.kp, self.guard) };
        }

        // Step 5: Update permutation - remove slot at logical position
        let mut new_perm: L::Perm = leaf.permutation();
        new_perm.remove(self.ki);
        leaf.set_permutation(new_perm);

        // Step 6: Clear the slot value pointer
        leaf.set_leaf_value_ptr(self.kp, StdPtr::null_mut());

        // Step 7: Decrement entry count
        self.tree.dec_count();

        // Step 8: Lazy coalescing for empty leaves
        //
        // When a leaf becomes empty, we:
        // 1. Mark it as empty (so insert can reuse it)
        // 2. Schedule it for background cleanup (if not leftmost)
        //
        // This avoids the lock-coupling approach from C++ that caused infinite loops.
        // See TODO.md "Background: Why C++ Port Failed" for details.
        if new_perm.size() == 0 {
            leaf.mark_empty();

            // Schedule for cleanup if not leftmost (leftmost is sentinel)
            if !leaf.prev().is_null() {
                let ikey_bound = leaf.ikey_bound();
                self.tree.coalesce_queue.schedule(self.leaf, ikey_bound);
            }
        }

        // Mark insert for version increment (lock drops at end of scope)
        self.lock.mark_insert();

        value
    }

    /// Try to remove an empty leaf from the tree structure.
    ///
    /// Called when a leaf becomes empty after key removal. This method:
    /// 1. Marks the leaf as deleted
    /// 2. Unlinks it from the B-link chain
    /// 3. Removes it from the parent internode
    /// 4. Schedules the leaf for retirement
    ///
    /// # Returns
    ///
    /// - `None` if the leaf was removed (lock consumed)
    /// - `Some(self)` if the leaf is leftmost and cannot be removed (lock retained)
    ///
    /// # C++ Reference
    ///
    /// `masstree_remove.hh:178-255` - `remove_leaf()`
    #[allow(dead_code, reason = "Deferred: inline structural removal")]
    #[cold]
    #[inline(never)]
    fn try_remove_leaf(mut self) -> Option<Self> {
        // Get leaf info before any borrows
        let prev_is_null: bool = self.leaf().prev().is_null();
        let next_is_null: bool = self.leaf().safe_next().is_null();

        // Case 1: Leftmost leaf (prev == null) - special handling
        if prev_is_null {
            if next_is_null {
                // Only leaf in sublayer (or main tree root) - call gc_layer
                // gc_layer always consumes self. It returns false for main tree root
                // (leaf stays empty), but we still return None because lock was released.
                let _ = self.gc_layer();
                return None;
            }
            // Leftmost but not only leaf - cannot remove directly
            return Some(self);
        }

        // Case 2: Non-leftmost leaf - we can remove it

        // Capture values we need after consuming self
        let tree = self.tree;
        let leaf_ptr = self.leaf;
        let guard = self.guard;
        let ikey_bound: u64 = self.leaf().ikey_bound();

        // Step 1: Mark as deleted
        self.lock.mark_deleted();

        // Step 2: Unlink from B-link chain
        // SAFETY: leaf is locked, prev is non-null (checked above)
        unsafe { self.leaf().unlink_from_chain() };

        // Step 3: Remove from parent internode (consumes lock via self)
        self.remove_from_parent(leaf_ptr.cast::<u8>(), ikey_bound, None);

        // Step 4: Schedule leaf retirement
        // SAFETY: leaf was unlinked and removed from parent
        unsafe { tree.allocator.retire_leaf(leaf_ptr, guard) };

        None
    }

    /// Garbage collect an empty sublayer.
    ///
    /// Called when the only leaf in a sublayer becomes empty.
    /// Clears the parent's layer slot and retires the sublayer.
    ///
    /// # Returns
    ///
    /// - `true` if the sublayer was cleaned up (lock consumed)
    /// - `false` if this is the main tree root (stays empty, but lock IS consumed - caller must recreate)
    ///
    /// Note: This always consumes `self`. For the main tree root case, we return false
    /// but the lock is still released (main root stays as empty leaf).
    ///
    /// # C++ Reference
    ///
    /// `masstree_remove.hh:24-115` - `gc_layer()`
    #[allow(dead_code, reason = "Deferred: inline structural removal")]
    #[cold]
    #[inline(never)]
    fn gc_layer(self) -> bool {
        // Capture values we need
        let tree = self.tree;
        let leaf_ptr = self.leaf;
        let guard = self.guard;
        let layer_context = self.layer_context;

        let Some(ctx) = layer_context else {
            // No layer context = this is the main tree root
            // Leave as empty leaf - don't mark deleted!
            // Just drop the lock (mark_insert will bump version)
            // Lock drops here, incrementing version
            drop(self.lock);
            return false;
        };

        // Mark sublayer as deleted_layer BEFORE releasing lock
        // This is critical for the lock ordering fix (Phase 3 in TODO.md)
        // SAFETY: leaf is valid and locked
        unsafe { &*leaf_ptr }.mark_deleted_layer();

        // Get sublayer pointer before releasing lock
        let sublayer_ptr: *mut u8 = leaf_ptr.cast::<u8>();

        // Release sublayer lock BEFORE acquiring parent lock
        // This prevents deadlock (always lock in parent -> child order)
        drop(self.lock);

        // Now safe to lock parent
        // SAFETY: ctx.parent_leaf is a valid leaf pointer obtained during traversal
        // and protected by the guard. We released sublayer lock before acquiring parent.
        let parent_leaf: &L = unsafe { &*ctx.parent_leaf.cast::<L>() };
        let mut parent_lock: LockGuard<'_> = parent_leaf.version().lock();

        // Verify slot still points to sublayer (may have changed concurrently)
        let current_slot_ptr: *mut u8 = parent_leaf.leaf_value_ptr(ctx.parent_slot);
        let current_keylenx: u8 = parent_leaf.keylenx(ctx.parent_slot);

        if current_slot_ptr != sublayer_ptr || current_keylenx < LAYER_KEYLENX {
            // Slot has changed - just retire sublayer
            drop(parent_lock);

            // SAFETY: leaf_ptr is a valid leaf allocated via allocator. The leaf is
            // marked deleted_layer and no longer reachable. Guard ensures safe reclamation.
            unsafe { tree.allocator.retire_leaf(leaf_ptr, guard) };
            return true;
        }

        // Clear the layer slot in the parent
        parent_leaf.clear_slot_and_permutation(ctx.parent_slot);
        parent_lock.mark_insert();
        drop(parent_lock);

        // Schedule sublayer leaf for retirement
        // SAFETY: leaf_ptr is valid, marked deleted_layer, and unlinked from parent.
        // Guard ensures deferred reclamation after all readers finish.
        unsafe { tree.allocator.retire_leaf(leaf_ptr, guard) };

        true
    }

    /// Remove a child from its parent internode, potentially collapsing empty chains.
    ///
    /// This method walks up the tree from the removed child, updating parent
    /// pointers and collapsing single-child internodes.
    ///
    /// The cursor's lock is used for the initial child, then transferred to
    /// each parent as we walk up the tree (lock coupling protocol).
    ///
    /// # C++ Reference
    ///
    /// `masstree_remove.hh:217-255` - the main loop in `remove_leaf()`
    #[allow(dead_code, reason = "Deferred: inline structural removal")]
    #[cold]
    #[inline(never)]
    fn remove_from_parent(self, child: *mut u8, ikey_bound: u64, replacement: Option<*mut u8>) {
        // Capture values we need before consuming self
        let tree = self.tree;
        let guard = self.guard;

        let mut current: *mut u8 = child;
        let mut current_ikey: u64 = ikey_bound;
        let mut current_replacement: Option<*mut u8> = replacement;

        // We own the lock from the cursor
        let mut current_lock: LockGuard<'_> = self.lock;

        loop {
            // Step 1: Lock parent while current is locked
            // SAFETY: current is valid and locked; locked_parent traverses valid parent pointer.
            let (parent_lock_opt, parent_ptr): (Option<LockGuard<'_>>, *mut u8) =
                unsafe { Self::locked_parent::<S, L>(current) };

            let Some(mut parent_lock) = parent_lock_opt else {
                // No parent - we're at the layer root
                drop(current_lock);
                return;
            };

            // SAFETY: parent_ptr is a valid internode pointer returned by locked_parent.
            // We hold parent_lock which guarantees exclusive access.
            let parent: &L::Internode = unsafe { &*parent_ptr.cast::<L::Internode>() };

            // Step 2: Mark parent as being modified
            parent_lock.mark_insert();

            debug_assert!(
                !parent.version().is_deleted(),
                "remove_from_parent: parent should not be deleted"
            );

            // Step 3: Find child position using upper_bound
            let kp: usize = upper_bound_internode_generic(current_ikey, parent);

            // Step 4: Validate child is at expected position
            if parent.child(kp) != current {
                // Child not at expected position - search for it
                let nkeys: usize = parent.nkeys();
                let mut found_kp: Option<usize> = None;
                for i in 0..=nkeys {
                    if parent.child(i) == current {
                        found_kp = Some(i);
                        break;
                    }
                }

                if found_kp.is_some() {
                    // Found at different position - restart needed
                    drop(current_lock);
                    drop(parent_lock);
                    return;
                }

                // Not found - already removed by another thread
                drop(current_lock);
                drop(parent_lock);
                return;
            }

            // Step 5: Update child pointer
            let new_child: *mut u8 = current_replacement.unwrap_or(StdPtr::null_mut());
            parent.set_child(kp, new_child);

            // Step 6: Handle replacement or shift
            if let Some(repl) = current_replacement {
                if !repl.is_null() {
                    // SAFETY: repl is a valid node pointer (the replacement child).
                    // parent_ptr is valid and we hold parent_lock.
                    unsafe {
                        NodeCleaner::set_parent_erased::<S, L>(repl, parent_ptr);
                    }
                } else if kp > 0 {
                    NodeCleaner::shift_internode_down_generic::<S, L::Internode>(parent, kp);
                }
            } else if kp > 0 {
                NodeCleaner::shift_internode_down_generic::<S, L::Internode>(parent, kp);
            }

            // Step 7: Handle redirect if leftmost child removed
            if (kp <= 1) && (parent.nkeys() > 0) && parent.child(0).is_null() {
                let new_ikey: u64 = parent.ikey(0);
                NodeCleaner::redirect_ikey_bounds_generic::<S, L>(
                    parent_ptr,
                    &mut parent_lock,
                    current_ikey,
                    new_ikey,
                );
                current_ikey = new_ikey;
            }

            // Step 8: Drop current lock AFTER parent update is complete
            drop(current_lock);

            // Step 9: Check if parent should be collapsed
            if parent.nkeys() > 0 || parent.version().is_root() {
                drop(parent_lock);
                return;
            }

            let child0: *mut u8 = parent.child(0);
            if child0.is_null() {
                drop(parent_lock);
                return;
            }

            // Step 10: Collapse empty parent
            parent_lock.mark_deleted();

            // SAFETY: parent_ptr is a valid internode that we hold locked (parent_lock).
            // We just marked it deleted. Guard ensures deferred reclamation.
            unsafe {
                tree.allocator.retire_internode_erased(parent_ptr, guard);
            }

            parent.set_child(0, StdPtr::null_mut());

            current = parent_ptr;
            current_replacement = Some(child0);
            current_lock = parent_lock; // Transfer lock ownership
        }
    }

    /// Lock the parent of a node, handling concurrent modifications.
    ///
    /// Returns `(Some(lock), parent_ptr)` if parent exists, or `(None, null)` if at root.
    ///
    /// # Safety
    ///
    /// `current_ptr` must point to a valid, locked node.
    #[allow(dead_code, reason = "Deferred: inline structural removal")]
    unsafe fn locked_parent<SS, LL>(current_ptr: *mut u8) -> (Option<LockGuard<'static>>, *mut u8)
    where
        SS: ValueSlot,
        SS::Value: Send + Sync + 'static,
        SS::Output: Send + Sync,
        LL: LayerCapableLeaf<SS>,
    {
        for _ in 0..MAX_PARENT_RETRIES {
            // SAFETY: caller guarantees current_ptr is valid
            let parent_ptr: *mut u8 =
                unsafe { NodeCleaner::get_parent_erased::<SS, LL>(current_ptr) };

            if parent_ptr.is_null() {
                return (None, StdPtr::null_mut());
            }

            // SAFETY: parent_ptr is non-null and valid
            let parent: &LL::Internode = unsafe { &*(parent_ptr.cast::<LL::Internode>()) };
            let parent_lock: LockGuard<'static> = parent.version().lock();

            // Verify parent hasn't changed
            // SAFETY: current_ptr still valid
            let current_parent: *mut u8 =
                unsafe { NodeCleaner::get_parent_erased::<SS, LL>(current_ptr) };
            if current_parent == parent_ptr {
                return (Some(parent_lock), parent_ptr);
            }

            drop(parent_lock);
            StdHint::spin_loop();
        }

        (None, StdPtr::null_mut())
    }
}

/// Unit struct providing stateless utility methods for node removal from the [`crate::MassTree`].
#[derive(Debug)]
pub struct NodeCleaner;

impl NodeCleaner {
    /// Remove a leaf from its parent internode(s) during coalesce.
    ///
    /// This function makes a leaf unreachable from the tree by removing all
    /// parent child pointers that reference it. After this function returns,
    /// the leaf can be safely retired.
    ///
    /// # Algorithm (Lock Coupling)
    /// 1. Start with locked leaf
    /// 2. Acquire parent lock (while leaf still locked)
    /// 3. Find leaf position in parent using `upper_bound` search
    /// 4. Verify leaf is still at expected position
    /// 5. Set child pointer to null
    /// 6. Shift remaining entries to fill gap (if not leftmost)
    /// 7. Release leaf lock (keep parent lock)
    /// 8. If parent is empty and not root, mark deleted and continue up
    /// 9. Otherwise, release parent lock and return
    ///
    /// # Safety
    /// - `leaf_ptr` must point to a valid, locked, deleted leaf
    /// - `lock` must be the lock guard for the leaf at `leaf_ptr`
    /// - The leaf must already be marked deleted and unlinked from B-link chain
    ///
    /// # Returns
    /// `true` if parent cleanup succeeded (safe to retire), `false` if it failed
    /// (parent not found or lock coupling failed - do not retire the leaf).
    #[cold]
    #[inline(never)]
    #[expect(clippy::too_many_lines)]
    pub fn remove_leaf_from_parent_for_coalesce<S, L, A>(
        allocator: &A,
        guard: &LocalGuard<'_>,
        leaf_ptr: *mut L,
        leaf_lock: LockGuard<'_>, // Consumed - ownership transferred to lock coupling
        ikey_bound: u64,
    ) -> bool
    where
        S: ValueSlot,
        S::Value: Send + Sync + 'static,
        S::Output: Send + Sync,
        L: LayerCapableLeaf<S>,
        A: NodeAllocatorGeneric<S, L>,
    {
        let mut current: *mut u8 = leaf_ptr.cast();
        let mut current_ikey: u64 = ikey_bound;
        let mut current_replacement: Option<*mut u8> = None;
        let mut current_lock: LockGuard<'_> = leaf_lock;

        loop {
            // Step 1: Lock parent while current is locked
            // SAFETY: current is valid and locked; we hold current_lock.
            let (parent_lock_opt, parent_ptr): (Option<LockGuard<'_>>, *mut u8) =
                unsafe { Self::locked_parent_generic::<S, L>(current) };

            let Some(mut parent_lock) = parent_lock_opt else {
                // No parent - we're at the layer root
                // The leaf is the only node in this layer (root leaf), nothing to remove from.
                // This is a valid success case - the leaf has no parent pointer to clean up.
                drop(current_lock);
                return true;
            };

            // SAFETY: parent_ptr is a valid internode pointer returned by locked_parent.
            // We hold parent_lock which guarantees exclusive access.
            let parent: &L::Internode = unsafe { &*parent_ptr.cast::<L::Internode>() };

            // Step 2: Mark parent as being modified
            parent_lock.mark_insert();

            debug_assert!(
                !parent.version().is_deleted(),
                "remove_leaf_from_parent: parent should not be deleted"
            );

            // Step 3: Find child position using upper_bound
            let mut kp: usize = upper_bound_internode_generic(current_ikey, parent);

            // Step 4: Validate child is at expected position
            if parent.child(kp) != current {
                // Child not at expected position - search for it linearly.
                // This can happen if concurrent operations changed the parent structure.
                let nkeys: usize = parent.nkeys();
                let mut found_kp: Option<usize> = None;
                for i in 0..=nkeys {
                    if parent.child(i) == current {
                        found_kp = Some(i);
                        break;
                    }
                }

                if let Some(actual_kp) = found_kp {
                    // Found at different position - use the actual position
                    kp = actual_kp;
                } else {
                    // Not found - already removed by another thread (concurrent coalesce)
                    // This is a success case: the parent cleanup is already done.
                    drop(current_lock);
                    drop(parent_lock);
                    return true;
                }
            }

            // Step 5: Update child pointer
            let new_child: *mut u8 = current_replacement.unwrap_or(StdPtr::null_mut());
            parent.set_child(kp, new_child);

            // Step 6: Handle replacement or shift
            if let Some(repl) = current_replacement {
                if !repl.is_null() {
                    // SAFETY: repl is a valid node pointer (the replacement child).
                    // parent_ptr is valid and we hold parent_lock.
                    unsafe {
                        Self::set_parent_erased::<S, L>(repl, parent_ptr);
                    }
                } else if kp > 0 {
                    Self::shift_internode_down_generic::<S, L::Internode>(parent, kp);
                }
            } else if kp > 0 {
                Self::shift_internode_down_generic::<S, L::Internode>(parent, kp);
            }

            // Step 7: Handle redirect if leftmost child removed
            //
            // When we remove a child at position 0 or 1, and after shifting child[0]
            // becomes null (because the original child[0] was the one removed), we
            // need to update ancestor separator keys that used the old ikey_bound.
            //
            // This happens when:
            // - kp == 0: We removed the leftmost child directly
            // - kp == 1 and after shift child[0] is null: Edge case from prior collapse
            //
            // The redirect walks up ancestors updating separator keys from old_ikey
            // to new_ikey (the first key of the new leftmost child).
            if (kp <= 1) && (parent.nkeys() > 0) && parent.child(0).is_null() {
                let new_ikey: u64 = parent.ikey(0);
                Self::redirect_ikey_bounds_generic::<S, L>(
                    parent_ptr,
                    &mut parent_lock,
                    current_ikey,
                    new_ikey,
                );
                current_ikey = new_ikey;
            }

            // Step 8: Drop current lock AFTER parent update is complete
            // This is the lock coupling protocol - never release current before parent is updated
            drop(current_lock);

            // Step 9: Check if parent should be collapsed
            if parent.nkeys() > 0 || parent.version().is_root() {
                // Parent still has children or is root - we're done successfully
                drop(parent_lock);
                return true;
            }

            // Parent is empty (nkeys == 0) and not root
            let child0: *mut u8 = parent.child(0);
            if child0.is_null() {
                // No remaining child - this can happen if we removed the last child.
                // The parent is now an empty non-root internode. We still mark it
                // deleted but there's nothing to propagate up as replacement.
                parent_lock.mark_deleted();
                unsafe {
                    allocator.retire_internode_erased(parent_ptr, guard);
                }
                drop(parent_lock);
                return true;
            }

            // Step 10: Collapse empty parent
            // The parent has exactly one child (child[0]). We'll mark the parent
            // deleted and continue walking up, installing child[0] as the replacement
            // in the grandparent.
            parent_lock.mark_deleted();

            // SAFETY: parent_ptr is a valid internode that we hold locked (parent_lock).
            // We just marked it deleted. Guard ensures deferred reclamation.
            unsafe {
                allocator.retire_internode_erased(parent_ptr, guard);
            }

            // Clear child pointer (the child will become the replacement)
            parent.set_child(0, StdPtr::null_mut());

            // Continue walking up with the remaining child as replacement
            current = parent_ptr;
            current_replacement = Some(child0);
            current_lock = parent_lock; // Transfer lock ownership
        }
    }

    // ============================================================================
    //  Public Entry Point
    // ============================================================================

    /// Main entry point for concurrent deletion.
    ///
    /// # Algorithm
    /// 1. Navigate to the target leaf using optimistic traversal
    /// 2. Search for the key within the leaf
    /// 3. Lock the leaf and verify the key still exists
    /// 4. Remove the slot from the permutation
    /// 5. Retire the value via seize
    /// 6. If leaf is now empty, trigger leaf removal
    ///
    /// # Reference
    /// C++ `masstree_remove.hh:162-176` - `finish_remove()`
    ///
    /// # Errors
    /// If fails to properly remove
    pub fn remove_concurrent_generic<S, L, A>(
        tree: &MassTreeGeneric<S, L, A>,
        key_bytes: &[u8],
        guard: &LocalGuard<'_>,
    ) -> Result<Option<S::Output>, RemoveError>
    where
        S: ValueSlot,
        S::Value: Send + Sync + 'static,
        S::Output: Send + Sync,
        L: LayerCapableLeaf<S>,
        A: NodeAllocatorGeneric<S, L>,
    {
        let mut key = Key::new(key_bytes);
        let mut retry_count: usize = 0;

        // Track layer descent for multi-layer keys
        let mut layer_root: *mut u8 = tree.root_ptr.load(AtomicOrdering::Acquire);

        // Track parent layer context for gc_layer cleanup
        let mut layer_context: Option<LayerContext> = None;

        'layer_loop: loop {
            'retry_loop: loop {
                if retry_count >= MAX_RETRIES {
                    return Err(RemoveError::RetryLimitExceeded);
                }
                retry_count += 1;

                // Step 1: Navigate to target leaf
                let leaf_ptr: *mut L =
                    tree.reach_leaf_concurrent_generic(layer_root, &key, false, guard);
                // SAFETY: reach_leaf_concurrent_generic returns a valid leaf pointer
                let leaf: &L = unsafe { &*leaf_ptr };

                // Step 2: Get stable version and search for slot
                let version: u32 = leaf.version().stable();
                let perm: L::Perm = leaf.permutation();

                let search_result: RemoveSearchResult =
                    Self::search_for_remove_generic::<S, L>(leaf, &key, &perm);

                // Step 3: Version validation before locking
                if leaf.version().has_changed(version) {
                    continue 'retry_loop;
                }

                // Step 4: Handle search result
                match search_result {
                    RemoveSearchResult::NotFound => {
                        return Ok(None);
                    }

                    RemoveSearchResult::Found { ki, kp: _ } => {
                        // Lock, verify, and get cursor or control flow instruction
                        let lock_result = Self::lock_and_verify_for_remove(
                            tree,
                            leaf_ptr,
                            ki,
                            &key,
                            layer_context,
                            guard,
                        );

                        match lock_result {
                            RemoveLockResult::NotFound => {
                                return Ok(None);
                            }

                            RemoveLockResult::Ready(cursor) => {
                                return Ok(cursor.finish_remove());
                            }

                            RemoveLockResult::DescendLayer {
                                layer_ptr: lp,
                                parent_leaf,
                                parent_slot,
                            } => {
                                layer_context = Some(LayerContext {
                                    parent_leaf,
                                    parent_slot,
                                });
                                layer_root = lp;
                                key.shift();
                                continue 'layer_loop;
                            }

                            RemoveLockResult::Retry => {}

                            RemoveLockResult::RestartFromRoot => {
                                key.unshift_all();
                                layer_root = tree.root_ptr.load(AtomicOrdering::Acquire);
                                layer_context = None;
                                continue 'layer_loop;
                            }
                        }
                    }

                    RemoveSearchResult::DescendLayer { layer_ptr, slot } => {
                        // Check if sublayer is deleted before descending
                        if !Self::is_sublayer_valid(layer_ptr) {
                            return Ok(None);
                        }

                        layer_context = Some(LayerContext {
                            parent_leaf: leaf_ptr.cast::<u8>(),
                            parent_slot: slot,
                        });
                        layer_root = layer_ptr;
                        key.shift();
                        continue 'layer_loop;
                    }
                }
            }
        }
    }

    // ============================================================================
    //  Search for Remove
    // ============================================================================

    /// Search for a key within a leaf for removal.
    ///
    /// Unlike `search_for_insert`, we need to find an exact match.
    ///
    /// # Algorithm
    ///
    /// 1. Linear scan through permutation slots
    /// 2. Compare ikey values
    /// 3. If ikey matches, check keylenx and suffix
    /// 4. Return position if exact match found
    #[inline(always)]
    fn search_for_remove_generic<S, L>(
        leaf: &L,
        key: &Key<'_>,
        perm: &L::Perm,
    ) -> RemoveSearchResult
    where
        S: ValueSlot,
        S::Value: Send + Sync + 'static,
        S::Output: Send + Sync,
        L: LayerCapableLeaf<S>,
    {
        let target_ikey: u64 = key.ikey();
        let size: usize = perm.size();

        for ki in 0..size {
            let kp: usize = perm.get(ki);
            let slot_ikey: u64 = leaf.ikey(kp);

            if slot_ikey < target_ikey {
                continue;
            }

            if slot_ikey > target_ikey {
                // Past the target - key not found
                return RemoveSearchResult::NotFound;
            }

            // ikey matches - check key length/type
            let slot_keylenx: u8 = leaf.keylenx(kp);

            if slot_keylenx >= LAYER_KEYLENX {
                // This is a layer pointer
                if key.has_suffix() {
                    // Key continues - need to descend
                    let layer_ptr: *mut u8 = leaf.leaf_value_ptr(kp);
                    return RemoveSearchResult::DescendLayer {
                        layer_ptr,
                        slot: kp,
                    };
                }
                // Short key can't match layer pointer
                return RemoveSearchResult::NotFound;
            }

            // Check inline key length
            #[expect(clippy::cast_possible_truncation, reason = "key.current_len() <= 8")]
            let key_len: u8 = key.current_len() as u8;

            if slot_keylenx == KSUF_KEYLENX {
                // Has suffix - compare suffix
                if !key.has_suffix() {
                    continue; // Key too short
                }

                let suffix: &[u8] = key.suffix();
                if leaf.ksuf_equals(kp, suffix) {
                    return RemoveSearchResult::Found { ki, kp };
                }
                continue;
            }

            // Inline key (no suffix)
            if key_len <= 8 && slot_keylenx == key_len {
                // Exact match for short key
                return RemoveSearchResult::Found { ki, kp };
            }
        }

        RemoveSearchResult::NotFound
    }

    /// Lock the leaf and verify the key is still present for removal.
    ///
    /// This is the hot path after finding a key. It:
    /// 1. Locks the leaf
    /// 2. Checks for `deleted_layer` (gc'd sublayer)
    /// 3. Re-verifies the key position after locking
    /// 4. Returns appropriate result for caller to act on
    ///
    /// # Returns
    ///
    /// - `NotFound`: Key was not found (sublayer deleted)
    /// - `Ready(cursor)`: Cursor is ready for removal
    /// - `DescendLayer{...}`: Need to descend into sublayer
    /// - `Retry`: Version changed, retry from `reach_leaf`
    /// - `RestartFromRoot`: Leaf is `deleted_layer`, restart with unshifted key
    #[inline]
    fn lock_and_verify_for_remove<'t, 'g, S, L, A>(
        tree: &'t MassTreeGeneric<S, L, A>,
        leaf_ptr: *mut L,
        ki: usize,
        key: &Key<'_>,
        layer_context: Option<LayerContext>,
        guard: &'g LocalGuard<'g>,
    ) -> RemoveLockResult<'t, 'g, S, L, A>
    where
        S: ValueSlot,
        S::Value: Send + Sync + 'static,
        S::Output: Send + Sync,
        L: LayerCapableLeaf<S>,
        A: NodeAllocatorGeneric<S, L>,
    {
        // SAFETY: leaf_ptr is valid from reach_leaf_concurrent_generic
        let leaf: &L = unsafe { &*leaf_ptr };

        // Step 1: Lock the leaf
        let lock: LockGuard<'_> = leaf.version().lock();

        // Step 2: Check if this leaf is part of a gc'd sublayer
        if leaf.deleted_layer() {
            drop(lock);
            return RemoveLockResult::RestartFromRoot;
        }

        // Step 3: Re-verify after lock (key might have moved)
        let new_perm: L::Perm = leaf.permutation();
        if new_perm.size() <= ki {
            // Slot was removed by concurrent delete
            drop(lock);
            return RemoveLockResult::Retry;
        }

        let new_kp: usize = new_perm.get(ki);
        let slot_ikey: u64 = leaf.ikey(new_kp);
        let slot_keylenx: u8 = leaf.keylenx(new_kp);

        // Verify this is still our key
        if slot_ikey != key.ikey() {
            drop(lock);
            return RemoveLockResult::Retry;
        }

        // Step 4: Handle based on key type
        if slot_keylenx >= LAYER_KEYLENX {
            // This is a layer pointer, not a value - need to descend
            drop(lock);
            let lp: *mut u8 = leaf.leaf_value_ptr(new_kp);

            // Check if sublayer is deleted before descending
            if !Self::is_sublayer_valid(lp) {
                return RemoveLockResult::NotFound;
            }

            return RemoveLockResult::DescendLayer {
                layer_ptr: lp,
                parent_leaf: leaf_ptr.cast::<u8>(),
                parent_slot: new_kp,
            };
        }

        // Step 5: Key found, create cursor for removal
        let cursor = RemoveCursor::new(tree, leaf_ptr, lock, ki, new_kp, layer_context, guard);

        RemoveLockResult::Ready(cursor)
    }

    // ============================================================================
    //  Sublayer Helpers
    // ============================================================================

    /// Check if a sublayer is valid (not deleted) before descending.
    ///
    /// Returns `true` if the sublayer is valid and can be descended into,
    /// `false` if the sublayer has been garbage collected.
    ///
    /// This check runs on every layer descent (hot path). Only finding a deleted
    /// sublayer is rare. The function is tiny (pointer cast + load), so inlining
    /// is always beneficial.
    ///
    /// # Safety
    ///
    /// `layer_ptr` must point to a valid node protected by a guard.
    #[inline(always)]
    fn is_sublayer_valid(layer_ptr: *mut u8) -> bool {
        // SAFETY: layer_ptr came from a valid slot, protected by guard
        #[expect(clippy::cast_ptr_alignment, reason = "Checked")]
        let sublayer_version: &NodeVersion = unsafe { &*layer_ptr.cast::<NodeVersion>() };

        !sublayer_version.is_deleted()
    }

    // ============================================================================
    //  Internode Restructuring (Cold Paths)
    // ============================================================================

    /// Shift internode keys and children down after removal.
    ///
    /// When a non-leftmost child is removed at position `kp` (`child[kp]` is set to null),
    /// we need to fill the gap by shifting subsequent entries down:
    /// - Keys: `ikey[kp-1..nkeys-1) = ikey[kp..nkeys)`
    /// - Children: `child[kp..nkeys) = child[kp+1..nkeys+1)`
    /// - Decrement nkeys by 1
    ///
    /// # Example
    ///
    /// Before (nkeys=3, kp=2):
    /// ```text
    /// keys:     [k0, k1, k2]
    /// children: [c0, c1, NULL, c3]  <- c2 was removed
    ///
    /// After shift_down(2):
    /// ```text
    /// keys:     [k0, k2, _]      <- k1 removed (was separator for c1/c2)
    /// children: [c0, c1, c3, _]  <- gap filled
    /// nkeys = 2
    ///
    /// # C++ Reference
    ///
    /// `masstree_remove.hh:231`: `p->shift_down(kp - 1, kp, p->nkeys_ - kp)`
    #[cold]
    #[inline(never)]
    fn shift_internode_down_generic<S, I>(inode: &I, removed_pos: usize)
    where
        S: ValueSlot,
        I: TreeInternode<S>,
    {
        let nkeys: usize = inode.nkeys();

        debug_assert!(removed_pos > 0, "shift_down: removed_pos must be > 0");
        debug_assert!(
            removed_pos <= nkeys,
            "shift_down: removed_pos out of bounds"
        );

        let count: usize = nkeys - removed_pos;

        // Shift keys: ikey[removed_pos - 1 + i] = ikey[removed_pos + i] for i in 0..count
        for i in 0..count {
            let key: u64 = inode.ikey(removed_pos + i);
            inode.set_ikey(removed_pos - 1 + i, key);
        }

        // Shift children: child[removed_pos + i] = child[removed_pos + 1 + i] for i in 0..count
        for i in 0..count {
            let child: *mut u8 = inode.child(removed_pos + 1 + i);
            inode.set_child(removed_pos + i, child);
        }

        // Decrement nkeys
        #[expect(clippy::cast_possible_truncation, reason = "nkeys <= WIDTH")]
        inode.set_nkeys((nkeys - 1) as u8);
    }

    /// Redirect ikey bounds in ancestor internodes after leftmost child removal.
    ///
    /// When the leftmost child of an internode is removed, the separator keys
    /// stored in ancestor internodes may reference the old ikey bound,. This
    /// function walks up the tree updating these bounds.
    ///
    /// # Lock Coupling
    /// This function uses hand-over-hand locking matching C++ pattern:
    /// - Start with `start_internode` locked (passed as `current_lock`)
    /// - Lock parent, then unlock current
    /// - Continue until we reach a position where updates are no longer needed
    ///
    /// # C++ Reference
    ///
    /// `masstree_remove.hh:257-276`:
    /// ```cpp
    /// void tcursor<P>::redirect(internode_type* n, ikey_type ikey,
    ///                           ikey_type replacement_ikey, threadinfo& ti) {
    ///     int kp = -1;
    ///
    ///     do {
    ///         internode_type* p = n->locked_parent(ti);
    ///
    ///         if (kp >= 0) {
    ///             n->unlock();
    ///         }
    ///
    ///         kp = internode_type::bound_type::upper(ikey, *p);
    ///         masstree_invariant(p->child_[kp] == n);
    ///
    ///         if (kp > 0) {
    ///             p->ikey0_[kp - 1] = replacement_ikey;
    ///         }
    ///
    ///         n = p;
    ///     } while (kp == 0 || (kp == 1 && !n->child_[0]));
    ///
    ///     n->unlock();
    /// }
    /// ```
    #[cold]
    #[inline(never)]
    #[expect(clippy::cast_sign_loss)]
    fn redirect_ikey_bounds_generic<S, L>(
        start_internode: *mut u8,
        _start_lock: &mut LockGuard<'_>,
        old_ikey: u64,
        new_ikey: u64,
    ) where
        S: ValueSlot,
        S::Value: Send + Sync + 'static,
        S::Output: Send + Sync,
        L: LayerCapableLeaf<S>,
    {
        let mut current: *mut u8 = start_internode;

        // kp starts at -1 to indicate first iteration (don't unlock current yet)
        let mut kp: i32 = -1;

        // Using Option to track whether we own a lock that needs dropping
        // On first iter, caller owns the lock (passed in start_lock)
        // On subsequent iters, we own the lock
        let mut owned_lock: Option<LockGuard<'_>> = None;

        loop {
            // Step 1: Lock parent (current is still locked)
            // SAFETY: current is a valid internode pointer
            let (parent_lock_opt, parent_ptr) =
                unsafe { Self::locked_parent_generic::<S, L>(current) };

            // Check if we reached root
            let Some(parent_lock) = parent_lock_opt else {
                // No parent, we're at layer root
                // Drop our owned lock if we have one (but not the caller's lock)
                drop(owned_lock);
                return;
            };

            // Step 2: unlock previous node if this isn't the first iteration
            // C++ `if (kp >= 0) { n->unlock(); }`
            if kp >= 0 {
                // Drop our owned lock from previous iteration
                drop(owned_lock.take());
            }

            // SAFETY: parent_ptr is valid and point to an internode
            let parent: &L::Internode = unsafe { &*(parent_ptr.cast::<L::Internode>()) };

            // Step 3: Find position of current node in parent
            #[expect(clippy::cast_possible_wrap, clippy::cast_possible_truncation)]
            {
                kp = upper_bound_internode_generic(old_ikey, parent) as i32;
            }

            // Validate current is at expected position
            debug_assert_eq!(
                parent.child(kp as usize),
                current,
                "redirect: current not found at expected position kp={kp} for old_ikey={old_ikey:#x}"
            );

            if kp > 0 {
                // NOTE: The C++ comment sayas 'p->ikey0_[kp - 1] might not equal ikey'
                // This is because we're looking up by the bound, not exact match
                parent.set_ikey((kp - 1) as usize, new_ikey);
            }

            // Step 5: Move up: parent becomes current
            current = parent_ptr;
            owned_lock = Some(parent_lock);

            // Step 6: Check termination condition
            // C++ `while (kp == 0 || (kp == 1 && !n->child_[0]))`
            // Continue if:
            // - kp == 0: we're at the leftmost position, may need to update higher
            // - kp == 1 && child[0] is null: leftmost child was removed, may need to update higher
            let should_continue: bool = (kp == 0) || ((kp == 1) && parent.child(0).is_null());

            if !should_continue {
                // Done - drop the lock and return
                drop(owned_lock);
                return;
            }
        }
    }

    /// Get the parent internode pointer from a node (leaf or internode).
    ///
    /// # Safety
    /// `node_ptr` must point to a valid leaf or internode.
    unsafe fn get_parent_erased<S, L>(node_ptr: *mut u8) -> *mut u8
    where
        S: ValueSlot,
        S::Value: Send + Sync + 'static,
        S::Output: Send + Sync,
        L: LayerCapableLeaf<S>,
    {
        // SAFETY: Caller guarantees node_ptr points to valid leaf or internode.
        #[expect(clippy::cast_ptr_alignment)]
        let version: &NodeVersion = unsafe { &*(node_ptr.cast::<NodeVersion>()) };

        if version.is_leaf() {
            // SAFETY: version.is_leaf() confirmed node is a leaf.
            let leaf: &L = unsafe { &*(node_ptr.cast::<L>()) };
            leaf.parent()
        } else {
            // SAFETY: !version.is_leaf() confirmed node is an internode.
            let inode: &L::Internode = unsafe { &*(node_ptr.cast::<L::Internode>()) };
            inode.parent()
        }
    }

    /// This matches the C++ `node_base<P>::locked_parent()` from `masstree_struct.hh`.
    ///
    /// # Algorithm
    /// 1. Read parent pointer from current node
    /// 2. If null, return (None, null) - we've reached layer root
    /// 3. Lock the parent
    /// 4. Validate that `current.parent() == parent` still holds
    /// 5. If validation fails (parent changed due to split), unlock and retry
    ///
    /// # Returns
    /// `(Some(LockGuard), parent_ptr)` if parent exists and is locked,
    /// `(None, null)` if current is a root (no parent)
    ///
    /// # Safety
    /// - `current_ptr` must ppint to a valid, locked node (leaf or internode)
    /// - The caller must hold a lock on the current code
    ///
    /// ```cpp
    /// template <typename P>
    ///
    /// internode<P>* node_base<P>::locked_parent(threadinfo& ti) const {
    ///     node_base<P>* p;
    ///     masstree_precondition(!this->concurrent || this->locked());
    ///
    ///     while (true) {
    ///         p = this->parent();
    ///
    ///         if (!this->parent_exists(p)) {
    ///             break;
    ///         }
    ///
    ///         nodeversion_type pv = p->lock(*p, ti.lock_fence(tc_internode_lock));
    ///
    ///         if (p == this->parent()) {
    ///             masstree_invariant(!p->isleaf());
    ///             break;
    ///         }
    ///
    ///         p->unlock(pv);
    ///         relax_fence();
    ///     }
    ///
    ///     return static_cast<internode<P>*>(p);
    /// }
    /// ```
    unsafe fn locked_parent_generic<'a, S, L>(
        current_ptr: *mut u8,
    ) -> (Option<LockGuard<'a>>, *mut u8)
    where
        S: ValueSlot,
        S::Value: Send + Sync + 'static,
        S::Output: Send + Sync,
        L: LayerCapableLeaf<S>,
    {
        for _ in 0..MAX_PARENT_RETRIES {
            // Step 1: Read parent pointer
            // SAFETY: current_ptr is valid (guaranteed by caller of unsafe fn).
            let parent_ptr: *mut u8 = unsafe { Self::get_parent_erased::<S, L>(current_ptr) };

            // Step 2: Check if we've reached root
            if parent_ptr.is_null() {
                return (None, StdPtr::null_mut());
            }

            // Step 3: Lock the parent (must be an internode)
            // SAFETY: parent_ptr is non-null and points to an internode.
            let parent: &L::Internode = unsafe { &*(parent_ptr.cast::<L::Internode>()) };
            let parent_lock: LockGuard<'_> = parent.version().lock();

            // Step 4: Validate parent hasn't changed (could change due to split/collapse)
            // SAFETY: current_ptr is still valid, re-reading parent to validate.
            let current_parent: *mut u8 = unsafe { Self::get_parent_erased::<S, L>(current_ptr) };

            if current_parent == parent_ptr {
                // Parent is still valid and locked
                debug_assert!(
                    !parent.version().is_leaf(),
                    "locked_parent: parent must be an internode"
                );
                return (Some(parent_lock), parent_ptr);
            }

            // Step 5: Parent changed - unlock and retry
            drop(parent_lock);

            // Relax fence like C++ relax_fence()
            StdHint::spin_loop();
        }

        // Retry limit exceeded - return as if no parent (caller handles gracefully)
        (None, StdPtr::null_mut())
    }

    /// Set the parent pointer on a node (leaf or internode).
    ///
    /// This dispatches based on [`NodeVersion::is_leaf`] to call the appropriate
    /// `set_parent` method.
    ///
    /// # Safety
    /// - `node_ptr` must point to a valid leaf or internode
    /// - The node's type must match what [`NodeVersion::is_leaf`] reports
    #[inline(always)]
    unsafe fn set_parent_erased<S, L>(node_ptr: *mut u8, new_parent: *mut u8)
    where
        S: ValueSlot,
        S::Value: Send + Sync + 'static,
        S::Output: Send + Sync,
        L: LayerCapableLeaf<S>,
    {
        // SAFETY: Caller guarantees node_ptr points to valid leaf or internode.
        #[expect(clippy::cast_ptr_alignment, reason = "Checked by caller")]
        let version: &NodeVersion = unsafe { &*(node_ptr.cast::<NodeVersion>()) };

        if version.is_leaf() {
            // SAFETY: version.is_leaf() confirmed node is a leaf.
            let leaf: &L = unsafe { &*(node_ptr.cast::<L>()) };
            leaf.set_parent(new_parent);
        } else {
            // SAFETY: !version.is_leaf() confirmed node is an internode.
            let inode: &L::Internode = unsafe { &*(node_ptr.cast::<L::Internode>()) };
            inode.set_parent(new_parent);
        }
    }
}

// ============================================================================
//  Tests
// ============================================================================

#[cfg(test)]
#[expect(clippy::unwrap_used, reason = "Tests")]
mod unit_tests;
