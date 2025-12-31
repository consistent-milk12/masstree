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
    Found {
        /// Logical position in permutation (0..size).
        ki: usize,

        /// Physical slot index (0..WIDTH).
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

// ============================================================================
//  Layer Context
// ============================================================================

/// Context for tracking layer descent during remove operations.
///
/// When descending into a sublayer, we track the parent leaf and slot
/// so that gc_layer can clear the parent's layer slot if the sublayer
/// becomes empty.
///
/// # Why This Is Needed
///
/// Sublayer roots have NULL parent pointers (they're marked as roots).
/// We cannot use `sublayer_leaf.parent()` to find the parent leaf.
/// Instead, we track this information during the layer descent.
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

/// Unit struct providing stateless utility methods for node removal from the [`MassTree`].
#[derive(Debug)]
pub struct NodeCleaner;

impl NodeCleaner {
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

                match search_result {
                    RemoveSearchResult::NotFound => {
                        // Key doesn't exist
                        return Ok(None);
                    }

                    RemoveSearchResult::Found { ki, kp } => {
                        // Step 4: Lock the leaf
                        let mut lock: LockGuard<'_> = leaf.version().lock();

                        // Step 5: Re-verify after lock (key might have moved)
                        let new_perm: L::Perm = leaf.permutation();
                        if new_perm.size() <= ki {
                            // Slot was removed by concurrent delete
                            drop(lock);
                            continue 'retry_loop;
                        }

                        let new_kp: usize = new_perm.get(ki);
                        let slot_ikey: u64 = leaf.ikey(new_kp);
                        let slot_keylenx: u8 = leaf.keylenx(new_kp);

                        // Verify this is still our key
                        if slot_ikey != key.ikey() {
                            drop(lock);
                            continue 'retry_loop;
                        }

                        // Handle based on key type
                        if slot_keylenx >= LAYER_KEYLENX {
                            // This is a layer pointer, not a value
                            // Need to descend into layer
                            drop(lock);
                            let lp: *mut u8 = leaf.leaf_value_ptr(new_kp);

                            // Check if sublayer is deleted before descending
                            // SAFETY: lp is non-null, protected by guard
                            let sublayer_version: &NodeVersion = unsafe { &*lp.cast::<NodeVersion>() };
                            if sublayer_version.is_deleted() {
                                // Sublayer was garbage collected - key no longer exists
                                #[cfg(feature = "tracing")]
                                tracing::debug!(
                                    layer_ptr = ?lp,
                                    "remove: sublayer deleted during descent"
                                );
                                return Ok(None);
                            }

                            // Track this layer descent for gc_layer cleanup
                            layer_context = Some(LayerContext {
                                parent_leaf: leaf_ptr.cast::<u8>(),
                                parent_slot: new_kp,
                            });

                            layer_root = lp;
                            key.shift();
                            continue 'layer_loop;
                        }

                        // Step 6: Finish the removal
                        let removed_value: Option<S::Output> = Self::finish_remove_generic::<S, L, A>(
                            tree, leaf, &mut lock, ki, kp, layer_context, guard,
                        );

                        // Step 7: Check if leaf is now empty
                        // NOTE: We intentionally do NOT mark_deleted() here.
                        // Marking a leaf as deleted without updating the tree structure
                        // (parent pointers, B-link chain) causes infinite retry loops
                        // because get/insert see the deleted flag and retry, but the
                        // root still points to the deleted leaf.
                        //
                        // Full leaf removal requires:
                        // 1. Unlinking from B-link chain (btree_leaflink::unlink)
                        // 2. Updating parent internode child pointers
                        // 3. Potentially collapsing empty internodes
                        // 4. gc_layer for empty sublayers
                        //
                        // For now, empty leaves stay in the tree but have size=0,
                        // so searches correctly return not-found.
                        // This is a known limitation documented in KNOWN_BUGS.md.

                        // Lock automatically released on drop
                        return Ok(removed_value);
                    }

                    RemoveSearchResult::DescendLayer { layer_ptr, slot } => {
                        // Key continues in sublayer - check if deleted before descending
                        // SAFETY: layer_ptr came from valid slot, protected by guard
                        let sublayer_version: &NodeVersion =
                            unsafe { &*layer_ptr.cast::<NodeVersion>() };
                        if sublayer_version.is_deleted() {
                            // Sublayer was garbage collected - key doesn't exist
                            #[cfg(feature = "tracing")]
                            tracing::debug!(
                                layer_ptr = ?layer_ptr,
                                "remove: sublayer deleted, key not found"
                            );
                            return Ok(None);
                        }

                        // Track this layer descent for gc_layer cleanup
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

    /// Remove a child from its parent internode, potentially collapsing empty chains.
    ///
    /// This function walks up the tree from the removed child, updating parent
    /// pointers and collapsing single-child internodes.
    ///
    /// # Algorithm (from C++ `masstree_remove.hh:211-255`)
    ///
    /// 1. Lock the parent internode (while current is still locked)
    /// 2. Find the child's position using `upper_bound`
    /// 3. Validate `parent.child[kp] == current`
    /// 4. Update the child pointer (to replacement or null)
    /// 5. If replacing: set replacement's parent pointer
    /// 6. If nulling and not leftmost: shift keys/children down
    /// 7. If removing leftmost child and parent has keys: call `redirect()`
    /// 8. Unlock current, keep parent locked
    /// 9. If parent is now empty and not root: mark deleted, continue up with child[0] as replacement
    ///
    /// # Lock Coupling Protocol
    ///
    /// The C++ reference uses "hand-over-hand" locking:
    /// - Lock parent while current is still locked
    /// - Unlock current after parent updates are complete
    /// - Continue upward with parent as the new "current"
    ///
    /// This prevents races where the parent structure changes while we're modifying it.
    ///
    /// # C++ Reference
    ///
    /// `masstree_remove.hh:217-255`:
    /// ```cpp
    /// while (true) {
    ///     internode_type *p = n->locked_parent(ti);  // Locks parent, current still locked
    ///
    ///     p->mark_insert();
    ///
    ///     // ... update p->child_[kp] ...
    ///
    ///     n->unlock();  // Unlock current AFTER parent update
    ///     n = p;        // Parent becomes new current
    ///
    ///     if (p->nkeys_ || p->is_root()) break;
    ///
    ///     // ... collapse empty parent ...
    /// }
    ///
    /// n->unlock();  // Final unlock
    /// ```
    fn remove_from_parent_generic<S, L, A>(
        tree: &MassTreeGeneric<S, L, A>,
        child: *mut u8,
        ikey_bound: u64,
        replacement: Option<*mut u8>,
        guard: &LocalGuard<'_>,
    ) where
        S: ValueSlot,
        S::Value: Send + Sync + 'static,
        S::Output: Send + Sync,
        L: LayerCapableLeaf<S>,
        A: NodeAllocatorGeneric<S, L>,
    {
        // Current node and its lock (the leaf we're removing, which is already locked by caller)
        //
        // NOTE: The caller (remove_leaf_generic) holds the lock on child.
        // We don't have the lock guard here, but the C++ protocol assumes current is locked.
        //
        // CRITICAL: The caller MUST keep the leaf locked until this function returns,
        // or pass the lock guard to us. For now, we match the C++ pattern where the
        // caller's lock remains held during this call.

        let mut current: *mut u8 = child;
        let mut current_ikey: u64 = ikey_bound;
        let mut current_replacement: Option<*mut u8> = replacement;

        // For the first iteration, we don't own a lock, the caller does.
        // After the first iteration, we'll own the lck on `current`.
        let mut current_lock: Option<LockGuard<'_>> = None;

        loop {
            // Step 1: Lock parent while current is locked
            // SAFETY: current is a valid node pointer, locked by caller or us
            let (parent_lock_opt, parent_ptr): (Option<LockGuard<'_>>, *mut u8) =
                unsafe { Self::locked_parent_generic::<S, L>(current) };

            // Check if we reached the layer root
            let Some(mut parent_lock) = parent_lock_opt else {
                // No parent, we're at the layer root
                // Clean up: drop current_lock if we have one
                drop(current_lock);

                // The layer pointer in the grandparent will be handled by gc_layer
                return;
            };

            // SAFETY: parent_ptr is valid and points to an internode
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
            // This check matches C++ `masstree_invariant(p->child_[kp] == n)`
            //
            // If child is not at expected position, the tree structure changed
            // (possibly due to concurrent operations). We need to re-find the child.
            if parent.child(kp) != current {
                // Child not at expected position - search for it
                // This can happen if concurrent removes shifted the internode
                let nkeys: usize = parent.nkeys();
                let mut found_kp: Option<usize> = None;
                for i in 0..=nkeys {
                    if parent.child(i) == current {
                        found_kp = Some(i);
                        break;
                    }
                }

                if let Some(actual_kp) = found_kp {
                    // Found child at different position - but we need to restart
                    // because the ikey_bound may have changed too
                    drop(current_lock);
                    drop(parent_lock);
                    return; // Exit and let the caller retry if needed
                } else {
                    // Child not found in parent at all - already removed by another thread
                    drop(current_lock);
                    drop(parent_lock);
                    return;
                }
            }

            // Step 5: Update child pointer
            let new_child: *mut u8 = current_replacement.unwrap_or(StdPtr::null_mut());
            parent.set_child(kp, new_child);

            // Step 6: Handle replacement or shift
            if let Some(replacement) = current_replacement {
                if !replacement.is_null() {
                    // Set replacement's parent pointer
                    //
                    // CRITICAL: Must dispatch based on node type (leaf or internode)
                    //
                    // SAFETY: `replacement` is a valid non-null node
                    unsafe {
                        Self::set_parent_erased::<S, L>(replacement, parent_ptr);
                    }
                } else if kp > 0 {
                    // Non-leftmost child removed with no replacement: shift down to fill gap
                    // C++ `p->shift_down(kp -1, kp, p->nkeys_ - kp)`
                    Self::shift_internode_down_generic::<S, L::Internode>(parent, kp);
                }
            } else if kp > 0 {
                // No replacement provided (None) - still need to shift down if not leftmost
                // This happens when removing a leaf with no sibling to replace it
                Self::shift_internode_down_generic::<S, L::Internode>(parent, kp);
            }

            // Step 7: Check if leftmost child was removed and parent still has keys
            // In this case, we need to redirect ikey bounds in ancestors
            //
            // C++ cond: `kp <= 1 && p->nkeys_ > 0 && !p->child_[0]`
            // This means that we removed at position 0 or 1, parent has keys left,
            // but child[0] is now null (the leftmost child is gone).
            if (kp <= 1) && (parent.nkeys() > 0) && parent.child(0).is_null() {
                // The new leftmost key is at ikey(0)
                let new_ikey: u64 = parent.ikey(0);

                // NOTE: redirect needs to be called with parent still locked
                // we pass parent_ptr because redirect will lock-couple upward
                Self::redirect_ikey_bounds_generic::<S, L>(
                    parent_ptr,
                    &mut parent_lock,
                    current_ikey,
                    new_ikey,
                );

                current_ikey = new_ikey;
            }

            // Step 8: Unlock current (the child we just removed from parent)
            // On first iteration, the caller holds the lock (we have None).
            // On subsequent iterations, we hold the lock.
            drop(current_lock);

            // Step 9: Check if parent is empty and should be collapsed
            if parent.nkeys() > 0 || parent.version().is_root() {
                // Parent still has children or is the layer root - we're done
                drop(parent_lock);
                return;
            }

            // Check if we can collapse - need a valid child[0] as replacement
            let child0: *mut u8 = parent.child(0);
            if child0.is_null() {
                // No valid replacement - can't collapse
                // This shouldn't normally happen, but handle it gracefully
                drop(parent_lock);
                return;
            }

            // Step 10: Parent is empty and not root - mark deleted and collapse
            parent_lock.mark_deleted();

            // Step 11: Schedule parent retirement via allocator
            // SAFETY: parent_ptr will be unreachable after we update its parent's child pointer
            unsafe {
                tree.allocator.retire_internode_erased(parent_ptr, guard);
            }

            // Continue up the tree: parent becomes current, child[0] becomes replacement
            // Clear child[0] to prevent double-free when parent is reclaimed
            parent.set_child(0, std::ptr::null_mut());

            current = parent_ptr;
            current_replacement = Some(child0);
            current_lock = Some(parent_lock);

            // NOTE: current_ikey stays the same for the collapse operation
            // The ikey bound we're looking for in the grandparent is still the same
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
                    return RemoveSearchResult::DescendLayer { layer_ptr, slot: kp };
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

    // ============================================================================
    //  Finish Remove
    // ============================================================================

    /// Complete the removal of a key from a locked leaf.
    ///
    /// # Preconditions
    ///
    /// - Leaf is locked (caller holds `LockGuard`)
    /// - Key exists at logical position `ki`, physical slot `kp`
    ///
    /// # Algorithm
    ///
    /// 1. Extract value for return
    /// 2. Schedule value retirement via seize
    /// 3. Clear suffix if present
    /// 4. Update permutation using `perm.remove(ki)`
    /// 5. Store updated permutation
    /// 6. Decrement entry count
    fn finish_remove_generic<S, L, A>(
        tree: &MassTreeGeneric<S, L, A>,
        leaf: &L,
        lock: &mut LockGuard<'_>,
        ki: usize,
        kp: usize,
        layer_context: Option<LayerContext>,
        guard: &LocalGuard<'_>,
    ) -> Option<S::Output>
    where
        S: ValueSlot,
        S::Value: Send + Sync + 'static,
        S::Output: Send + Sync,
        L: LayerCapableLeaf<S>,
        A: NodeAllocatorGeneric<S, L>,
    {
        // Step 1: Extract the value pointer
        let value_ptr: *mut u8 = leaf.leaf_value_ptr(kp);

        // Step 2: Clone the value for return (before retirement)
        let value: Option<S::Output> = if value_ptr.is_null() {
            None
        } else {
            // SAFETY: value_ptr points to valid value created during insert
            // We use try_clone_output which handles Arc cloning properly
            leaf.try_clone_output(kp)
        };

        // Step 3: Schedule value retirement
        // The old value pointer needs to be freed after all readers are done
        if !value_ptr.is_null() {
            // SAFETY: value_ptr was created by insert and will be valid until retirement
            unsafe {
                guard.defer_retire(value_ptr, |ptr, _| {
                    S::cleanup_value_ptr(ptr);
                });
            }
        }

        // Step 4: Clear suffix if present
        let slot_keylenx: u8 = leaf.keylenx(kp);
        if slot_keylenx == KSUF_KEYLENX {
            // Clear the suffix slot
            // SAFETY: We hold the lock and kp is valid
            unsafe { leaf.clear_ksuf(kp, guard) };
        }

        // Step 5: Update permutation - remove slot at logical position `ki`
        let mut new_perm: L::Perm = leaf.permutation();
        new_perm.remove(ki);
        leaf.set_permutation(new_perm);

        // Step 6: Clear the slot value pointer
        // This prevents accidental access to retired value
        leaf.set_leaf_value_ptr(kp, std::ptr::null_mut());

        // Step 7: Decrement entry count
        tree.dec_count();

        // Step 8: Check if leaf is now empty and should be removed
        //
        // Leaf coalescing removes empty leaves from the tree structure:
        // - Empty leaves are unlinked from the B-link chain
        // - Removed from parent internodes
        // - Scheduled for deferred reclamation
        //
        // NOTE: Leaf coalescing is DISABLED due to known bugs that cause hangs.
        // See KNOWN_BUGS.md section "Leaf Coalescing Disabled" for details.
        // Re-enable after fixing:
        // - locked_parent_generic unbounded retry loop (remove.rs:1208-1242)
        // - Traversal retry on null children (generic.rs:1051-1055)
        // - Lock coupling protocol timing differences vs C++
        if false && new_perm.size() == 0 {
            // Try to remove the empty leaf from the tree
            let removed: bool =
                Self::remove_leaf_generic::<S, L, A>(tree, leaf, lock, layer_context, guard);
            if removed {
                // Leaf was marked deleted and unlinked - don't call mark_insert
                // The version was already bumped by mark_deleted
                return value;
            }
            // Leaf is leftmost and cannot be removed - fall through to mark_insert
        }

        // Mark insert in lock guard for version increment
        // This ensures readers see the removal
        lock.mark_insert();

        value
    }

    // ============================================================================
    //  Leaf Removal (Coalescing)
    // ============================================================================

    /// Remove an empty leaf from the tree structure.
    ///
    /// Called when a leaf becomes empty after key removal. This function:
    /// 1. Marks the leaf as deleted
    /// 2. Unlinks it from the B-link chain
    /// 3. Removes it from the parent internode
    /// 4. Schedules the leaf for retirement via seize
    ///
    /// # Algorithm (from C++ `masstree_remove.hh:178-255`)
    ///
    /// Leftmost leaves (`prev == null`) are never removed because:
    /// - They may be the only leaf in a sublayer
    /// - Removing them requires special `gc_layer` handling
    ///
    /// Non-leftmost leaves can be safely unlinked and retired.
    ///
    /// # Returns
    ///
    /// - `true` if the leaf was removed and scheduled for retirement
    /// - `false` if the leaf is leftmost and cannot be removed
    ///
    /// # Preconditions
    ///
    /// - `leaf` is locked (caller holds `LockGuard`)
    /// - `leaf.permutation().size() == 0` (leaf is empty)
    fn remove_leaf_generic<S, L, A>(
        tree: &MassTreeGeneric<S, L, A>,
        leaf: &L,
        lock: &mut LockGuard<'_>,
        layer_context: Option<LayerContext>,
        guard: &LocalGuard<'_>,
    ) -> bool
    where
        S: ValueSlot,
        S::Value: Send + Sync + 'static,
        S::Output: Send + Sync,
        L: LayerCapableLeaf<S>,
        A: NodeAllocatorGeneric<S, L>,
    {
        // Case 1: Leftmost leaf (prev == null) - special handling
        //
        // If this is the ONLY leaf in a sublayer (prev=null AND next=null),
        // we need to clean up the sublayer by clearing the parent's layer slot.
        // This prevents infinite retry loops where readers descend into the
        // deleted sublayer and keep retrying.
        if leaf.prev().is_null() {
            if leaf.safe_next().is_null() {
                // Only leaf in sublayer (or main tree root) - call gc_layer
                return Self::gc_layer_sync_generic::<S, L, A>(
                    tree, leaf, lock, layer_context, guard,
                );
            }
            // Leftmost but not only leaf - cannot remove directly
            return false;
        }

        // Case 2: Non-leftmost leaf - we can remove it

        // Step 1: Mark as deleted
        // This tells concurrent readers to retry from a higher level
        lock.mark_deleted();

        // Step 2: Unlink from B-link chain
        // SAFETY: leaf is locked, prev is non-null (checked above)
        unsafe { leaf.unlink_from_chain() };

        // Step 3: Remove from parent internode
        // This updates the parent's child pointer and potentially collapses empty internodes
        let ikey_bound: u64 = leaf.ikey_bound();
        let leaf_ptr: *mut u8 = std::ptr::from_ref::<L>(leaf).cast_mut().cast::<u8>();

        Self::remove_from_parent_generic::<S, L, A>(tree, leaf_ptr, ikey_bound, None, guard);

        // Step 4: Schedule leaf retirement via allocator
        //
        // The leaf has been:
        // - Marked deleted (readers will retry)
        // - Unlinked from B-link chain (range scans skip it)
        // - Removed from parent (traversal won't reach it)
        //
        // After a grace period (all current readers finish), the leaf can be freed.
        let leaf_ptr_for_retire: *mut L = std::ptr::from_ref::<L>(leaf).cast_mut();

        // SAFETY: leaf_ptr_for_retire points to a valid leaf that we just unlinked
        // and removed from the parent. After seize's grace period, no readers can
        // reach this leaf.
        unsafe { tree.allocator.retire_leaf(leaf_ptr_for_retire, guard) };

        #[cfg(feature = "tracing")]
        tracing::debug!(
            leaf_ptr = ?leaf_ptr_for_retire,
            ikey_bound = format_args!("{:016x}", ikey_bound),
            "remove_leaf: leaf retired"
        );

        true
    }

    // ============================================================================
    //  gc_layer - Sublayer Cleanup
    // ============================================================================

    /// Garbage collect an empty sublayer synchronously using tracked context.
    ///
    /// This is called when the only leaf in a sublayer becomes empty.
    /// Uses the `LayerContext` to find and clear the parent's layer slot.
    ///
    /// # Algorithm
    ///
    /// 1. Check if this is a sublayer (has `layer_context`) or main root
    /// 2. Lock the parent leaf
    /// 3. Verify the layer slot still points to this sublayer
    /// 4. Clear the layer slot
    /// 5. Mark sublayer leaf as deleted
    /// 6. Schedule sublayer leaf for retirement
    ///
    /// # Returns
    ///
    /// - `true` if the sublayer was successfully cleaned up or marked deleted
    /// - `false` if this is the main tree root (stays allocated but empty)
    ///
    /// # Preconditions
    ///
    /// - `sublayer_leaf` is the ONLY leaf in its sublayer (prev=null, next=null)
    /// - `sublayer_leaf` is locked by the caller via `sublayer_lock`
    /// - The sublayer leaf is empty (permutation size == 0)
    fn gc_layer_sync_generic<S, L, A>(
        tree: &MassTreeGeneric<S, L, A>,
        sublayer_leaf: &L,
        sublayer_lock: &mut LockGuard<'_>,
        layer_context: Option<LayerContext>,
        guard: &LocalGuard<'_>,
    ) -> bool
    where
        S: ValueSlot,
        S::Value: Send + Sync + 'static,
        S::Output: Send + Sync,
        L: LayerCapableLeaf<S>,
        A: NodeAllocatorGeneric<S, L>,
    {
        // Step 1: Check if we have layer context (are we a sublayer?)
        let Some(ctx) = layer_context else {
            // No layer context = this is the main tree root
            //
            // CRITICAL: Do NOT mark the main tree root as deleted!
            // The tree's root_ptr still points to this leaf, so if we mark it
            // deleted, all operations will see is_deleted() and retry forever.
            //
            // Instead, leave it as an empty leaf. Searches will correctly
            // return NotFound from the empty permutation. Inserts will
            // add to the empty leaf.

            #[cfg(feature = "tracing")]
            tracing::debug!(
                leaf_ptr = ?std::ptr::from_ref(sublayer_leaf),
                "gc_layer: main tree root is empty, keeping as empty leaf"
            );

            // Return false because we didn't actually remove/delete the leaf
            // This will cause finish_remove_generic to call mark_insert()
            // which bumps the version so readers see the change.
            return false;
        };

        // Step 2: Lock the parent leaf
        //
        // SAFETY: ctx.parent_leaf came from a valid leaf during layer descent
        let parent_leaf: &L = unsafe { &*ctx.parent_leaf.cast::<L>() };
        let mut parent_lock: LockGuard<'_> = parent_leaf.version().lock();

        // Step 3: Verify the layer slot still points to this sublayer
        //
        // Race check: another thread might have modified the parent
        let sublayer_ptr: *mut u8 = std::ptr::from_ref(sublayer_leaf).cast_mut().cast();
        let current_slot_ptr: *mut u8 = parent_leaf.leaf_value_ptr(ctx.parent_slot);
        let current_keylenx: u8 = parent_leaf.keylenx(ctx.parent_slot);

        if current_slot_ptr != sublayer_ptr || current_keylenx < LAYER_KEYLENX {
            // Slot has changed - concurrent modification
            // Just mark deleted and let eventual cleanup handle it
            drop(parent_lock);
            sublayer_lock.mark_deleted();

            #[cfg(feature = "tracing")]
            tracing::warn!(
                sublayer_ptr = ?sublayer_ptr,
                parent_slot = ctx.parent_slot,
                current_ptr = ?current_slot_ptr,
                "gc_layer: parent slot changed, marking deleted only"
            );

            // Still retire the sublayer leaf via allocator
            // SAFETY: sublayer_ptr is a valid leaf that we're removing
            let sublayer_leaf_ptr: *mut L = sublayer_ptr.cast();
            unsafe { tree.allocator.retire_leaf(sublayer_leaf_ptr, guard) };

            return true;
        }

        // Step 4: Clear the layer slot in the parent
        parent_leaf.clear_slot_and_permutation(ctx.parent_slot);

        // Mark parent as modified so readers retry
        parent_lock.mark_insert();

        // Step 5: Mark sublayer leaf as deleted
        sublayer_lock.mark_deleted();

        #[cfg(feature = "tracing")]
        tracing::debug!(
            sublayer_ptr = ?sublayer_ptr,
            parent_ptr = ?ctx.parent_leaf,
            slot = ctx.parent_slot,
            "gc_layer: cleared parent slot, sublayer marked deleted"
        );

        // Step 6: Unlock parent (before retire to reduce lock hold time)
        drop(parent_lock);

        // Step 7: Schedule sublayer leaf for retirement via allocator
        // SAFETY: sublayer_ptr is a valid leaf that we're removing
        let sublayer_leaf_ptr: *mut L = sublayer_ptr.cast();
        unsafe { tree.allocator.retire_leaf(sublayer_leaf_ptr, guard) };

        true
    }

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
        unsafe {
            // Read the NodeVersion at the start of the node to check if it's a leaf
            // SAFETY: Ensured by caller that node_ptr actually points to valid NodeVersion
            #[expect(clippy::cast_ptr_alignment)]
            let version: &NodeVersion = &*(node_ptr.cast::<NodeVersion>());

            if version.is_leaf() {
                let leaf: &L = &*(node_ptr.cast::<L>());
                leaf.parent()
            } else {
                let inode: &L::Internode = &*(node_ptr.cast::<L::Internode>());
                inode.parent()
            }
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
        loop {
            // Step 1: Read parent pointer
            let parent_ptr: *mut u8 = unsafe { Self::get_parent_erased::<S, L>(current_ptr) };

            // Step 2: Check if we've reached root
            if parent_ptr.is_null() {
                return (None, StdPtr::null_mut());
            }

            // Step 3: Lock the parent (must be an internode)
            // SAFETY: parent_ptr is non-null and points to an internode
            let parent: &L::Internode = unsafe { &*(parent_ptr.cast::<L::Internode>()) };
            let parent_lock: LockGuard<'_> = parent.version().lock();

            // Step 4: Validate parent hasn' changed
            // The parent pointer could have changed if:
            // - A concurrent split moved current to a new parent
            // - A concurrent collapse changed the parent chain
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
        // Read the NodeVersion at the start of the node
        #[expect(clippy::cast_ptr_alignment, reason = "Checked by caller")]
        let version: &NodeVersion = unsafe { &*(node_ptr.cast::<NodeVersion>()) };

        if version.is_leaf() {
            // It's a leaf
            let leaf: &L = unsafe { &*(node_ptr.cast::<L>()) };
            leaf.set_parent(new_parent);
        } else {
            // It's an internode
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
