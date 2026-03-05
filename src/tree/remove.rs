//! Deletion operations for `MassTree`.

use std::fmt::{self as StdFmt, Display, Formatter};
use std::hint as StdHint;
use std::ptr as StdPtr;
use std::sync::atomic::Ordering as AtomicOrdering;

use seize::LocalGuard;

use crate::ksearch::upper_bound_internode_generic;
use crate::leaf15::LeafNode15;
use crate::tree::coalesce::Route;
use crate::{
    TreeInternode, TreeLeafNode,
    alloc_trait::TreeAllocator,
    internode::InternodeNode,
    key::Key,
    leaf15::{KSUF_KEYLENX, LAYER_KEYLENX},
    nodeversion::{LockGuard, NodeVersion},
    policy::LeafPolicy,
    tree::MassTreeGeneric,
};

// ============================================================================
//  Error Types
// ============================================================================

/// Errors that can occur during removal.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RemoveError {
    /// Retry limit exceeded during optimistic concurrency.
    RetryLimitExceeded,
}

impl Display for RemoveError {
    fn fmt(&self, f: &mut Formatter<'_>) -> StdFmt::Result {
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
        #[allow(dead_code, reason = "Captured for debugging, verified under lock")]
        kp: usize,
    },

    /// Key might be in sublayer; descend and retry.
    DescendLayer {
        /// Pointer to the layer root.
        layer_ptr: *mut u8,
    },
}

/// Result of attempting to find, lock, and verify a key for removal.
enum RemoveLockResult<'t, 'g, P, A>
where
    P: LeafPolicy,
    A: TreeAllocator<P>,
{
    /// Key not found in this layer.
    NotFound,

    /// Key found and cursor is ready for removal.
    Ready(RemoveCursor<'t, 'g, P, A>),

    /// Need to descend into sublayer.
    DescendLayer {
        /// Pointer to the sublayer root.
        layer_ptr: *mut u8,
    },

    /// Version changed or slot moved, retry from `reach_leaf`.
    Retry,

    /// Leaf is part of a gc'd sublayer, restart from tree root.
    RestartFromRoot,
}

// ============================================================================
//  Constants
// ============================================================================

/// Maximum retries before giving up.
const MAX_RETRIES: usize = 1000;

/// Maximum retries when locking parent during tree walk.
const MAX_PARENT_RETRIES: usize = 100;

/// Size of the inline key (ikey) in bytes.
const IKEY_SIZE: u8 = 8;

// ============================================================================
//  Locked Parent Result
// ============================================================================

/// Result of attempting to lock a node's parent.
///
/// Distinguishes between "no parent exists" (safe) and "retry exhaustion" (unsafe).
enum LockedParentResult<'a> {
    /// Successfully locked the parent. Contains guard and pointer.
    Locked(LockGuard<'a>, *mut u8),

    /// Node has no parent (it's a layer root). This is a valid success case.
    NoParent,

    /// Failed to lock parent after `MAX_PARENT_RETRIES` attempts.
    RetryExhausted,
}

// ============================================================================
//  RemoveCursor
// ============================================================================

/// A cursor for remove operations that holds the lock as persistent state.
///
/// # Lifetime
///
/// The cursor borrows from:
/// - The tree (`'t`)
/// - The guard (`'g`)
/// - The leaf is accessed through the lock
#[derive(Debug)]
pub struct RemoveCursor<'t, 'g, P, A>
where
    P: LeafPolicy,
    A: TreeAllocator<P>,
{
    /// Reference to the tree for allocation and root access.
    tree: &'t MassTreeGeneric<P, A>,

    /// The locked leaf node. Lock is held for the lifetime of the cursor.
    leaf: *mut LeafNode15<P>,

    /// The lock guard.
    lock: LockGuard<'t>,

    /// Logical position in permutation (0..size).
    ki: usize,

    /// Physical slot index (0..WIDTH).
    kp: usize,

    /// Route from tree root to this leaf's sublayer parent (ikey per layer).
    /// Empty for top-layer leaves.
    route: Route,

    /// Guard for memory reclamation.
    guard: &'g LocalGuard<'g>,
}

impl<'t, 'g, P, A> RemoveCursor<'t, 'g, P, A>
where
    P: LeafPolicy,
    A: TreeAllocator<P>,
{
    /// Create a new remove cursor with the lock already held.
    #[inline(always)]
    #[expect(clippy::too_many_arguments, reason = "Complex state management")]
    const fn new(
        tree: &'t MassTreeGeneric<P, A>,
        leaf: *mut LeafNode15<P>,
        lock: LockGuard<'t>,
        ki: usize,
        kp: usize,
        route: Route,
        guard: &'g LocalGuard<'g>,
    ) -> Self {
        Self {
            tree,
            leaf,
            lock,
            ki,
            kp,
            route,
            guard,
        }
    }

    /// Complete the removal of a key from the locked leaf.
    #[must_use]
    pub fn finish_remove(mut self) -> Option<P::Output> {
        // SAFETY: leaf is valid and locked by us
        let leaf: &LeafNode15<P> = unsafe { &*self.leaf };
        let slot_keylenx: u8 = leaf.keylenx(self.kp);
        self.lock.mark_insert();

        let value: Option<P::Output> = leaf.take_value(self.kp);

        if P::NEEDS_RETIREMENT
            && let Some(ref v) = value
        {
            // SAFETY: The value was just taken from the slot.
            unsafe { P::retire_output(P::clone_output(v), self.guard) };
        }

        leaf.set_keylenx(self.kp, 0);

        if slot_keylenx == KSUF_KEYLENX {
            // SAFETY: We hold the lock on this leaf (self.lock), and self.kp is a
            // valid slot index obtained during search.
            unsafe { leaf.clear_ksuf(self.kp, self.guard) };
        }

        let mut new_perm: <LeafNode15<P> as TreeLeafNode<P>>::Perm = leaf.permutation();
        new_perm.remove(self.ki);
        // Release is required: concurrent insert threads perform optimistic
        // searches that read the permutation without holding the lock. The
        // compiler may reorder a Relaxed store before the preceding suffix and
        // value clears, making removed-slot data inconsistent to readers that
        // still see the old permutation. Release ensures all slot clears are
        // visible before the permutation update.
        leaf.set_permutation(new_perm);

        self.tree.dec_count();

        if new_perm.size() == 0 {
            self.schedule_coalesce_cold(leaf);
        }

        value
    }

    /// Handle empty-leaf coalesce scheduling. Separated from the hot path
    /// to keep `finish_remove` compact for the common case (size > 0).
    #[cold]
    #[inline(never)]
    fn schedule_coalesce_cold(&mut self, leaf: &LeafNode15<P>) {
        leaf.mark_empty();

        // Atomic dedup: try_mark_queued() sets the queued bit and returns
        // true only if it was not already set. This eliminates TOCTOU races
        // between concurrent finish_remove calls on the same leaf.
        if leaf.try_mark_queued() {
            if self.route.is_empty() {
                // Non-sublayer: schedule chain coalesce (pointer-based).
                let ikey_bound: u64 = leaf.ikey_bound();

                self.tree
                    .coalesce_queue
                    .schedule_chain(self.leaf.cast::<u8>(), ikey_bound);
            } else {
                // Sublayer: schedule route-based gc (no pointers stored).
                self.tree
                    .coalesce_queue
                    .schedule_sublayer(std::mem::take(&mut self.route));
            }
        }
    }
}

/// Unit struct providing stateless utility methods for node removal from the [`crate::MassTree`].
#[derive(Debug)]
pub struct NodeCleaner;

impl NodeCleaner {
    /// Remove a leaf from its parent internode(s) during coalesce.
    #[cold]
    #[inline(never)]
    pub fn remove_leaf_from_parent_for_coalesce<P, A>(
        allocator: &A,
        guard: &LocalGuard<'_>,
        leaf_ptr: *mut LeafNode15<P>,
        leaf_lock: LockGuard<'_>, // Consumed - ownership transferred to lock coupling
        ikey_bound: u64,
    ) -> bool
    where
        P: LeafPolicy,
        A: TreeAllocator<P>,
    {
        let mut current: *mut u8 = leaf_ptr.cast();
        let mut current_ikey: u64 = ikey_bound;
        let mut current_replacement: Option<*mut u8> = None;
        let mut current_lock: LockGuard<'_> = leaf_lock;

        loop {
            // SAFETY: current is valid and locked; we hold current_lock.
            let parent_result: LockedParentResult<'_> =
                unsafe { Self::locked_parent_generic::<P>(current) };

            let (mut parent_lock, parent_ptr) = match parent_result {
                LockedParentResult::Locked(lock, ptr) => (lock, ptr),

                LockedParentResult::NoParent => {
                    drop(current_lock);
                    return true;
                }

                LockedParentResult::RetryExhausted => {
                    drop(current_lock);
                    return false;
                }
            };

            // SAFETY: parent_ptr is a valid internode pointer returned by locked_parent_generic.
            let parent: &InternodeNode = unsafe { &*parent_ptr.cast::<InternodeNode>() };

            parent_lock.mark_insert();

            debug_assert!(
                !parent.version().is_deleted(),
                "remove_leaf_from_parent: parent should not be deleted"
            );

            let mut kp: usize = upper_bound_internode_generic(current_ikey, parent);

            if TreeInternode::child(parent, kp) != current {
                let nkeys: usize = parent.nkeys();
                let mut found_kp: Option<usize> = None;

                for i in 0..=nkeys {
                    if TreeInternode::child(parent, i) == current {
                        found_kp = Some(i);
                        break;
                    }
                }

                if let Some(actual_kp) = found_kp {
                    kp = actual_kp;
                } else {
                    drop(current_lock);
                    drop(parent_lock);

                    return true;
                }
            }

            let new_child: *mut u8 = current_replacement.unwrap_or(StdPtr::null_mut());
            parent.set_child(kp, new_child);

            let should_shift: bool = match current_replacement {
                Some(repl) if !repl.is_null() => {
                    // SAFETY: repl is a valid node pointer (the replacement child).
                    // parent_ptr is valid and we hold parent_lock.
                    unsafe {
                        Self::set_parent_erased::<P>(repl, parent_ptr);
                    }

                    false
                }

                _ => kp > 0,
            };

            if should_shift {
                Self::shift_internode_down_generic::<InternodeNode>(parent, kp);
            }

            if (kp <= 1) && (parent.nkeys() > 0) && TreeInternode::child(parent, 0).is_null() {
                let new_ikey: u64 = parent.ikey(0);
                Self::redirect_ikey_bounds_generic::<P>(parent_ptr, current_ikey, new_ikey);

                current_ikey = new_ikey;
            }

            drop(current_lock);

            if parent.nkeys() > 0 || parent.version().is_root() {
                drop(parent_lock);
                return true;
            }

            // Parent is empty (nkeys == 0) and not root
            let child0: *mut u8 = TreeInternode::child(parent, 0);

            // Step 10: Collapse empty parent
            parent_lock.mark_deleted();

            // Clear child pointer before retirement so no write occurs on
            // memory that has been scheduled for reclamation.
            parent.set_child(0, StdPtr::null_mut());

            // SAFETY: parent_ptr is a valid internode that we hold locked (parent_lock).
            unsafe {
                allocator.retire_internode_erased(parent_ptr, guard);
            }

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
    /// # Errors
    ///
    /// If fails to properly remove
    pub fn remove_concurrent_generic<P, A>(
        tree: &MassTreeGeneric<P, A>,
        key_bytes: &[u8],
        guard: &LocalGuard<'_>,
    ) -> Result<Option<P::Output>, RemoveError>
    where
        P: LeafPolicy,
        A: TreeAllocator<P>,
    {
        let mut key: Key<'_> = Key::new(key_bytes);
        let single_layer_mode: bool = !key.has_suffix();
        let mut retry_count: usize = 0;

        // Track layer descent for multi-layer keys
        let mut layer_root: *mut u8 = tree.root_ptr.load(AtomicOrdering::Acquire);

        // Route from tree root: ikey at each layer level, for sublayer GC.
        let mut route: Route = Vec::new();

        'layer_loop: loop {
            if retry_count >= MAX_RETRIES {
                return Err(RemoveError::RetryLimitExceeded);
            }
            retry_count += 1;

            let leaf_ptr: *mut LeafNode15<P> =
                tree.reach_leaf_concurrent_generic(layer_root, &key, false, guard);

            // SAFETY: reach_leaf_concurrent_generic returns a valid leaf pointer
            let leaf: &LeafNode15<P> = unsafe { &*leaf_ptr };

            'forward: loop {
                let version: u32 = leaf.version().stable();
                let perm: <LeafNode15<P> as TreeLeafNode<P>>::Perm = leaf.permutation();

                let search_result: RemoveSearchResult = if single_layer_mode {
                    Self::search_for_remove_single_layer::<P>(leaf, &key, &perm)
                } else {
                    Self::search_for_remove_generic::<P>(leaf, &key, &perm)
                };

                if leaf.version().has_changed(version) {
                    continue 'forward;
                }

                match search_result {
                    RemoveSearchResult::NotFound => {
                        return Ok(None);
                    }

                    RemoveSearchResult::Found { ki, kp: _ } => {
                        // Lock, verify, and get cursor or control flow instruction
                        let lock_result = Self::lock_and_verify_for_remove(
                            tree, leaf_ptr, ki, &key, &mut route, guard,
                        );

                        match lock_result {
                            RemoveLockResult::NotFound => {
                                return Ok(None);
                            }

                            RemoveLockResult::Ready(cursor) => {
                                return Ok(cursor.finish_remove());
                            }

                            RemoveLockResult::DescendLayer { layer_ptr: lp } => {
                                route.push(key.ikey());
                                layer_root = lp;
                                key.shift();
                                continue 'layer_loop;
                            }

                            RemoveLockResult::Retry => {
                                // Re-scan same leaf: permutation likely changed
                                // but leaf itself is still valid.
                            }

                            RemoveLockResult::RestartFromRoot => {
                                key.unshift_all();
                                layer_root = tree.root_ptr.load(AtomicOrdering::Acquire);
                                route.clear();
                                continue 'layer_loop;
                            }
                        }
                    }

                    RemoveSearchResult::DescendLayer { layer_ptr } => {
                        debug_assert!(
                            !single_layer_mode,
                            "DescendLayer unreachable in single-layer mode"
                        );
                        if !Self::is_sublayer_valid(layer_ptr) {
                            return Ok(None);
                        }

                        route.push(key.ikey());
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
    #[inline(always)]
    fn search_for_remove_generic<P>(
        leaf: &LeafNode15<P>,
        key: &Key<'_>,
        perm: &<LeafNode15<P> as TreeLeafNode<P>>::Perm,
    ) -> RemoveSearchResult
    where
        P: LeafPolicy,
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
                return RemoveSearchResult::NotFound;
            }

            // ikey matches - prefetch value to hide cache miss during keylenx check
            leaf.prefetch_value(kp);

            let slot_keylenx: u8 = leaf.keylenx(kp);

            if slot_keylenx >= LAYER_KEYLENX {
                // This is a layer pointer
                if key.has_suffix() {
                    // Key continues - need to descend
                    let layer_ptr: *mut u8 = leaf.load_layer_raw(kp);
                    return RemoveSearchResult::DescendLayer { layer_ptr };
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

                leaf.prefetch_suffix();
                let suffix: &[u8] = key.suffix();
                if leaf.ksuf_equals(kp, suffix) {
                    return RemoveSearchResult::Found { ki, kp };
                }
                continue;
            }

            // Inline key (no suffix)
            if key_len <= IKEY_SIZE && slot_keylenx == key_len {
                // Exact match for short key
                return RemoveSearchResult::Found { ki, kp };
            }
        }

        RemoveSearchResult::NotFound
    }

    /// Single-layer fast path: key has no suffix, so skip layer/suffix branches.
    #[inline(always)]
    fn search_for_remove_single_layer<P>(
        leaf: &LeafNode15<P>,
        key: &Key<'_>,
        perm: &<LeafNode15<P> as TreeLeafNode<P>>::Perm,
    ) -> RemoveSearchResult
    where
        P: LeafPolicy,
    {
        let target_ikey: u64 = key.ikey();
        #[expect(clippy::cast_possible_truncation, reason = "current_len() <= 8")]
        let search_keylenx: u8 = key.current_len() as u8;
        let size: usize = perm.size();

        for ki in 0..size {
            let kp: usize = perm.get(ki);
            let slot_ikey: u64 = leaf.ikey_relaxed(kp);

            if slot_ikey < target_ikey {
                continue;
            }

            if slot_ikey > target_ikey {
                return RemoveSearchResult::NotFound;
            }

            // ikey matches, prefetch value while checking keylenx
            leaf.prefetch_value(kp);
            let slot_keylenx: u8 = leaf.keylenx_relaxed(kp);

            if slot_keylenx == search_keylenx {
                return RemoveSearchResult::Found { ki, kp };
            }
        }

        RemoveSearchResult::NotFound
    }

    /// Lock the leaf and verify the key is still present for removal.
    #[inline]
    fn lock_and_verify_for_remove<'t, 'g, P, A>(
        tree: &'t MassTreeGeneric<P, A>,
        leaf_ptr: *mut LeafNode15<P>,
        ki: usize,
        key: &Key<'_>,
        route: &mut Route,
        guard: &'g LocalGuard<'g>,
    ) -> RemoveLockResult<'t, 'g, P, A>
    where
        P: LeafPolicy,
        A: TreeAllocator<P>,
    {
        // SAFETY: leaf_ptr is valid from reach_leaf_concurrent_generic
        let leaf: &LeafNode15<P> = unsafe { &*leaf_ptr };

        // Prefetch leaf data (permutation, ikeys, values) before entering the
        // lock spin loop. By the time the lock is acquired, the cache lines we
        // need for post-lock verification are likely hot in L1.
        leaf.prefetch_for_search();
        let lock: LockGuard<'_> = leaf.version().lock_bounded();

        if leaf.deleted_layer() {
            drop(lock);
            return RemoveLockResult::RestartFromRoot;
        }

        let new_perm: <LeafNode15<P> as TreeLeafNode<P>>::Perm = leaf.permutation();

        if new_perm.size() <= ki {
            drop(lock);
            return RemoveLockResult::Retry;
        }

        let new_kp: usize = new_perm.get(ki);
        // Prefetch the value at this slot while we verify ikey/keylenx below.
        leaf.prefetch_value(new_kp);
        let slot_ikey: u64 = leaf.ikey(new_kp);
        let slot_keylenx: u8 = leaf.keylenx(new_kp);

        // Verify this is still our key
        if slot_ikey != key.ikey() {
            drop(lock);
            return RemoveLockResult::Retry;
        }

        if slot_keylenx >= LAYER_KEYLENX {
            let lp: *mut u8 = leaf.load_layer_raw(new_kp);
            drop(lock);

            // Check if sublayer is deleted before descending
            if !Self::is_sublayer_valid(lp) {
                return RemoveLockResult::NotFound;
            }

            return RemoveLockResult::DescendLayer { layer_ptr: lp };
        }

        let cursor: RemoveCursor<'_, '_, P, A> = RemoveCursor::new(
            tree,
            leaf_ptr,
            lock,
            ki,
            new_kp,
            std::mem::take(route),
            guard,
        );

        RemoveLockResult::Ready(cursor)
    }

    // ============================================================================
    //  Sublayer Helpers
    // ============================================================================

    /// Check if a sublayer is valid (not deleted) before descending.
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
    #[cold]
    #[inline(never)]
    fn shift_internode_down_generic<I>(inode: &I, removed_pos: usize)
    where
        I: TreeInternode,
    {
        let nkeys: usize = inode.nkeys();

        debug_assert!(removed_pos > 0, "shift_down: removed_pos must be > 0");
        debug_assert!(
            removed_pos <= nkeys,
            "shift_down: removed_pos out of bounds"
        );

        let count: usize = nkeys - removed_pos;

        for i in 0..count {
            let key: u64 = inode.ikey(removed_pos + i);
            inode.set_ikey(removed_pos - 1 + i, key);
        }

        for i in 0..count {
            let child: *mut u8 = inode.child(removed_pos + 1 + i);
            inode.set_child(removed_pos + i, child);
        }

        // Decrement nkeys
        #[expect(clippy::cast_possible_truncation, reason = "nkeys <= WIDTH")]
        inode.set_nkeys((nkeys - 1) as u8);
    }

    /// Redirect ikey bounds in ancestor internodes after leftmost child removal.
    #[cold]
    #[inline(never)]
    #[expect(clippy::cast_sign_loss)]
    fn redirect_ikey_bounds_generic<P>(start_internode: *mut u8, old_ikey: u64, new_ikey: u64)
    where
        P: LeafPolicy,
    {
        let mut current: *mut u8 = start_internode;

        let mut kp: i32 = -1;

        let mut owned_lock: Option<LockGuard<'_>> = None;

        loop {
            // SAFETY: current is a valid internode pointer
            let parent_result: LockedParentResult<'_> =
                unsafe { Self::locked_parent_generic::<P>(current) };

            let (parent_lock, parent_ptr) = match parent_result {
                LockedParentResult::Locked(lock, ptr) => (lock, ptr),

                LockedParentResult::NoParent | LockedParentResult::RetryExhausted => {
                    drop(owned_lock);
                    return;
                }
            };

            if kp >= 0 {
                drop(owned_lock.take());
            }

            // SAFETY: parent_ptr is valid and point to an internode
            let parent: &InternodeNode = unsafe { &*(parent_ptr.cast::<InternodeNode>()) };

            #[expect(clippy::cast_possible_wrap, clippy::cast_possible_truncation)]
            {
                kp = upper_bound_internode_generic(old_ikey, parent) as i32;
            }

            // Fallback linear scan if binary search landed on the wrong child
            // (can happen if concurrent splits moved keys between reading
            // old_ikey and locking the parent).
            if TreeInternode::child(parent, kp as usize) != current {
                let nkeys: usize = parent.nkeys();
                let mut found: bool = false;

                for i in 0..=nkeys {
                    #[expect(clippy::cast_possible_wrap, clippy::cast_possible_truncation)]
                    if TreeInternode::child(parent, i) == current {
                        kp = i as i32;
                        found = true;
                        break;
                    }
                }

                if !found {
                    // Child already removed from parent by a concurrent coalesce.
                    drop(owned_lock);
                    drop(parent_lock);
                    return;
                }
            }

            if kp > 0 {
                parent.set_ikey((kp - 1) as usize, new_ikey);
            }

            current = parent_ptr;
            owned_lock = Some(parent_lock);

            let should_continue: bool =
                (kp == 0) || ((kp == 1) && TreeInternode::child(parent, 0).is_null());

            if !should_continue {
                drop(owned_lock);
                return;
            }
        }
    }

    /// Get the parent internode pointer from a node (leaf or internode).
    ///
    /// # Safety
    /// `node_ptr` must point to a valid leaf or internode.
    unsafe fn get_parent_erased<P: LeafPolicy>(node_ptr: *mut u8) -> *mut u8 {
        // SAFETY: Caller guarantees node_ptr points to valid leaf or internode.
        #[expect(clippy::cast_ptr_alignment)]
        let version: &NodeVersion = unsafe { &*(node_ptr.cast::<NodeVersion>()) };

        if version.is_leaf() {
            // SAFETY: version.is_leaf() confirmed node is a leaf.
            let leaf: &LeafNode15<P> = unsafe { &*(node_ptr.cast::<LeafNode15<P>>()) };
            TreeLeafNode::parent(leaf)
        } else {
            // SAFETY: !version.is_leaf() confirmed node is an internode.
            let inode: &InternodeNode = unsafe { &*(node_ptr.cast::<InternodeNode>()) };
            TreeInternode::parent(inode)
        }
    }

    unsafe fn locked_parent_generic<'a, P: LeafPolicy>(
        current_ptr: *mut u8,
    ) -> LockedParentResult<'a> {
        for _ in 0..MAX_PARENT_RETRIES {
            // SAFETY: current_ptr is valid (guaranteed by caller of unsafe fn).
            let parent_ptr: *mut u8 = unsafe { Self::get_parent_erased::<P>(current_ptr) };

            if parent_ptr.is_null() {
                return LockedParentResult::NoParent;
            }

            // SAFETY: parent_ptr is non-null and points to an internode.
            let parent: &InternodeNode = unsafe { &*(parent_ptr.cast::<InternodeNode>()) };
            let parent_lock: LockGuard<'_> = parent.version().lock();

            // SAFETY: current_ptr is still valid, re-reading parent to validate.
            let current_parent: *mut u8 = unsafe { Self::get_parent_erased::<P>(current_ptr) };

            if current_parent == parent_ptr {
                debug_assert!(
                    !parent.version().is_leaf(),
                    "locked_parent: parent must be an internode"
                );

                return LockedParentResult::Locked(parent_lock, parent_ptr);
            }

            drop(parent_lock);

            StdHint::spin_loop();
        }

        LockedParentResult::RetryExhausted
    }

    /// Set the parent pointer on a node (leaf or internode).
    #[inline(always)]
    unsafe fn set_parent_erased<P: LeafPolicy>(node_ptr: *mut u8, new_parent: *mut u8) {
        // SAFETY: Caller guarantees node_ptr points to valid leaf or internode.
        #[expect(clippy::cast_ptr_alignment, reason = "Checked by caller")]
        let version: &NodeVersion = unsafe { &*(node_ptr.cast::<NodeVersion>()) };

        if version.is_leaf() {
            // SAFETY: version.is_leaf() confirmed node is a leaf.
            let leaf: &LeafNode15<P> = unsafe { &*(node_ptr.cast::<LeafNode15<P>>()) };
            leaf.set_parent(new_parent);
        } else {
            // SAFETY: !version.is_leaf() confirmed node is an internode.
            let inode: &InternodeNode = unsafe { &*(node_ptr.cast::<InternodeNode>()) };
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
