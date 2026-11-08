//! Deletion operations for `MassTree`.

use seize::LocalGuard;
use std::fmt::{self as StdFmt, Display, Formatter};
use std::hint as StdHint;
use std::ptr as StdPtr;

use crate::ksearch::upper_bound_internode_generic;
use crate::leaf15::LeafNode15;
use crate::tree::coalesce::Route;
use crate::tree::split::Propagation;
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
//  Immediate Parent Result
// ============================================================================

/// Result of removing a leaf from its immediate parent internode.
///
/// Used by the parent-before-delete coalesce ordering: the leaf lock is
/// borrowed (not consumed), so the caller retains it for `mark_deleted`
/// and `unlink_from_chain` after a successful parent update.
#[allow(clippy::redundant_pub_crate)]
pub(crate) enum ImmediateParentResult<'a> {
    /// Leaf removed from parent. Parent has remaining keys or is root.
    Success,

    /// Leaf removed from parent. Parent became empty and is not root.
    /// Caller should attempt cascade with the returned parent lock.
    SuccessNeedsCascade {
        parent_ptr: *mut u8,
        parent_lock: LockGuard<'a>,
    },

    /// Failed to lock the leaf's immediate parent (`RetryExhausted`).
    /// No modifications were made. Safe to retry.
    Failure,
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

        // SAFETY: The remove cursor holds the leaf lock and takes the
        // terminal value exactly once before clearing the slot.
        let value: Option<P::Output> = unsafe { leaf.take_value(self.kp) };

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

        // NOTE: Release is required. Concurrent insert threads perform optimistic
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

                self.tree.coalesce_queue.schedule_chain(ikey_bound);
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
    // ========================================================================
    //  Parent-Before-Delete Coalesce
    // ========================================================================

    /// Remove a leaf from its immediate parent internode during coalesce.
    ///
    /// Unlike `remove_leaf_from_parent_for_coalesce`, this function borrows the
    /// leaf lock instead of consuming it. It handles only the imediate parent
    /// (no cascade). The caller retains the leaf lock for `mark_deleted` and
    /// `unlink_from_chain` after a successful parent update.
    #[cold]
    pub(crate) fn remove_leaf_from_immediate_parent<'a, P, A>(
        _allocator: &A,
        _guard: &LocalGuard<'_>,
        leaf_ptr: *mut LeafNode15<P>,
        _leaf_lock: &LockGuard<'_>,
        ikey_bound: u64,
    ) -> ImmediateParentResult<'a>
    where
        P: LeafPolicy,
        A: TreeAllocator<P>,
    {
        let current: *mut u8 = leaf_ptr.cast();

        // SAFETY: current is valid and locked (leaf_lock proves this).
        let parent_result: LockedParentResult<'_> =
            unsafe { Self::locked_parent_generic::<P>(current) };

        let (mut parent_lock, parent_ptr) = match parent_result {
            LockedParentResult::Locked(lock, ptr) => (lock, ptr),

            LockedParentResult::NoParent => {
                // Layer root, no parent to clean. Caller proceeds with delete.
                return ImmediateParentResult::Success;
            }

            LockedParentResult::RetryExhausted => {
                return ImmediateParentResult::Failure;
            }
        };

        // SAFETY: parent_ptr is a valid internode pointer returned by locked_parent_generic.
        let parent: &InternodeNode = unsafe { &*parent_ptr.cast::<InternodeNode>() };

        parent_lock.mark_insert();

        debug_assert!(
            !parent.version().is_deleted(),
            "remove_leaf_from_immediate_parent: parent should not be deleted"
        );

        // Find child slot pointing to our leaf.
        let mut kp: usize = upper_bound_internode_generic(ikey_bound, parent);

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
                // Child already removed by concurrent coalesce.
                drop(parent_lock);
                return ImmediateParentResult::Success;
            }
        }

        // Remove child from parent.
        parent.set_child(kp, StdPtr::null_mut());

        if kp > 0 {
            Self::shift_internode_down_generic::<InternodeNode>(parent, kp);
        }

        // Handle ikey_bound redirect if the removed child was at position 0 or 1.
        if (kp <= 1) && (parent.nkeys() > 0) && TreeInternode::child(parent, 0).is_null() {
            let new_ikey: u64 = parent.ikey(0);
            Self::redirect_ikey_bounds_generic::<P>(parent_ptr, ikey_bound, new_ikey);
        }

        if parent.nkeys() > 0 || parent.version().is_root() {
            drop(parent_lock);
            ImmediateParentResult::Success
        } else {
            // Parent became empty and is not root. Return lock for cascade.
            ImmediateParentResult::SuccessNeedsCascade {
                parent_ptr,
                parent_lock,
            }
        }
    }

    /// Attempt to collapse truly empty internodes upward from the given parent.
    ///
    /// A non-root internode with `nkeys == 0` but a live `child[0]` is a unary
    /// routing level. Collapsing it into the grandparent would shorten only
    /// that branch while the ancestor's immutable height stays unchanged
    /// (GC-04), so it is preserved. Only a node with no surviving child is
    /// truly empty and is unlinked.
    ///
    /// `ikey_bound` is the routing key used to locate the start internode
    /// in its parent. It propagates upward and is updated when a redirect
    /// changes the effective left-edge boundary.
    #[cold]
    pub(crate) fn try_cascade_internodes<P, A>(
        allocator: &A,
        guard: &LocalGuard<'_>,
        start_ptr: *mut u8,
        start_lock: LockGuard<'_>,
        ikey_bound: u64,
    ) where
        P: LeafPolicy,
        A: TreeAllocator<P>,
    {
        let mut current: *mut u8 = start_ptr;
        let mut current_lock: LockGuard<'_> = start_lock;
        let mut current_ikey: u64 = ikey_bound;

        loop {
            // SAFETY: current is valid and locked (current_lock proves this).
            let parent: &InternodeNode = unsafe { &*current.cast::<InternodeNode>() };

            // Capture the remaining child before collapsing.
            let child0: *mut u8 = TreeInternode::child(parent, 0);

            if !child0.is_null() {
                // GC-04: keep the unary routing level. Its child covers the
                // same range, and removing the level here would make the
                // grandparent's height inconsistent with this branch.
                drop(current_lock);
                return;
            }

            // The node has no surviving child. Unlink it from the grandparent
            // first; only then publish the deletion and retire it (GC-03).
            // Retry acquisition instead of returning on exhaustion: a retired
            // node must never remain reachable from the tree.

            let (mut grandparent_lock, grandparent_ptr) = loop {
                // SAFETY: current is valid (locked) and protected by the guard.
                match unsafe { Self::locked_parent_generic::<P>(current) } {
                    LockedParentResult::Locked(lock, ptr) => break (lock, ptr),

                    LockedParentResult::NoParent => {
                        // Current is the root. Keep it; a root contraction is
                        // a separate publication protocol.
                        drop(current_lock);
                        return;
                    }

                    LockedParentResult::RetryExhausted => {
                        // The parent pointer kept moving. Stay blocked until it
                        // stabilizes; the alternative is a linked node with no
                        // child and no retirement.
                        StdHint::spin_loop();
                    }
                }
            };

            // SAFETY: grandparent_ptr is valid, returned by locked_parent_generic.
            let grandparent: &InternodeNode = unsafe { &*grandparent_ptr.cast::<InternodeNode>() };

            grandparent_lock.mark_insert();

            // Find the child slot pointing to current.
            let nkeys: usize = grandparent.nkeys();
            let mut kp: Option<usize> = None;

            for i in 0..=nkeys {
                if TreeInternode::child(grandparent, i) == current {
                    kp = Some(i);
                    break;
                }
            }

            let Some(kp) = kp else {
                // Already removed by concurrent operation.
                drop(current_lock);
                drop(grandparent_lock);
                return;
            };

            // Publish the deletion before unlinking: a reader that still sees
            // the old child pointer must observe DELETED and restart instead
            // of descending into a node with no child.
            current_lock.mark_deleted();

            // Unlink current: there is no surviving child to reparent.
            grandparent.set_child(kp, StdPtr::null_mut());

            if kp > 0 {
                Self::shift_internode_down_generic::<InternodeNode>(grandparent, kp);
            }

            // Redirect ancestor ikey bounds if the replacement created a
            // null-slot-0 shape. Mirrors the check in
            // remove_leaf_from_immediate_parent.
            if (kp <= 1)
                && (grandparent.nkeys() > 0)
                && TreeInternode::child(grandparent, 0).is_null()
            {
                let new_ikey: u64 = grandparent.ikey(0);
                Self::redirect_ikey_bounds_generic::<P>(grandparent_ptr, current_ikey, new_ikey);
                current_ikey = new_ikey;
            }

            drop(current_lock);

            // SAFETY: current is unreachable from the tree (unlinked above)
            // and marked deleted; the guard prevents premature reclamation of
            // any pointer still held by a concurrent reader.
            unsafe { allocator.retire_internode_erased(current, guard) };

            if grandparent.nkeys() > 0 || grandparent.version().is_root() {
                drop(grandparent_lock);
                return;
            }

            // Grandparent also became empty. Continue cascade.
            current = grandparent_ptr;
            current_lock = grandparent_lock;
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
    #[expect(
        clippy::too_many_lines,
        reason = "Locked remove state machine requires the full protocol inline"
    )]
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
        let mut retry_count: usize = 0;

        // Track layer descent for multi-layer keys
        let mut layer_root: *mut u8 = tree.load_root_ptr_generic(guard).cast_mut();

        // Route from tree root: ikey at each layer level, for sublayer GC.
        let mut route: Route = Vec::new();

        'layer_loop: loop {
            // A layer slot may still point at the original left leaf after
            // promotion; climb to the true root before descending.
            layer_root = tree
                .maybe_parent_generic(layer_root.cast_const(), guard)
                .cast_mut();

            let in_sublayer: bool = !route.is_empty();
            let mut leaf_ptr: *mut LeafNode15<P> =
                tree.reach_leaf_concurrent_generic(layer_root, &key, in_sublayer, guard);

            // REM-02: a concurrent split may have moved the target into the
            // next leaf while we were descending. Follow the B-link chain
            // before searching, exactly like insert/get.
            let (advanced_ptr, exceeded): (*mut LeafNode15<P>, bool) =
                tree.advance_to_key_by_bound_generic(leaf_ptr, &key, guard);

            if exceeded {
                if !Self::bump_retry(&mut retry_count) {
                    return Err(RemoveError::RetryLimitExceeded);
                }

                key.unshift_all();
                layer_root = tree.load_root_ptr_generic(guard).cast_mut();
                route.clear();
                continue 'layer_loop;
            }

            leaf_ptr = advanced_ptr;

            // SAFETY: reach/advance return a valid leaf pointer.
            let mut leaf: &LeafNode15<P> = unsafe { &*leaf_ptr };

            'forward: loop {
                // A B-link advance inside this loop may land to the right of
                // the target; restart from the layer root rather than
                // searching the wrong leaf.
                if !leaf.prev(guard).is_null() && key.ikey() < leaf.ikey_bound() {
                    if !Self::bump_retry(&mut retry_count) {
                        return Err(RemoveError::RetryLimitExceeded);
                    }

                    continue 'layer_loop;
                }

                if leaf.version().is_deleted() || leaf.deleted_layer() {
                    if !Self::bump_retry(&mut retry_count) {
                        return Err(RemoveError::RetryLimitExceeded);
                    }

                    // A coalesced leaf may remain linked until unlink finishes.
                    // Restart from the main root so multi-layer state is reset.
                    key.unshift_all();
                    layer_root = tree.load_root_ptr_generic(guard).cast_mut();
                    route.clear();
                    continue 'layer_loop;
                }

                let version: u32 = leaf.version().stable();
                let perm: <LeafNode15<P> as TreeLeafNode<P>>::Perm = leaf.permutation();

                let search_result: RemoveSearchResult = if key.has_suffix() {
                    Self::search_for_remove::<false, P>(leaf, &key, &perm)
                } else {
                    Self::search_for_remove::<true, P>(leaf, &key, &perm)
                };

                if leaf.version().has_changed(version) {
                    if !Self::bump_retry(&mut retry_count) {
                        return Err(RemoveError::RetryLimitExceeded);
                    }

                    let (next_ptr, exceeded): (*mut LeafNode15<P>, bool) =
                        tree.advance_to_key_by_bound_generic(leaf_ptr, &key, guard);

                    if exceeded {
                        key.unshift_all();
                        layer_root = tree.load_root_ptr_generic(guard).cast_mut();
                        route.clear();
                        continue 'layer_loop;
                    }

                    if next_ptr != leaf_ptr {
                        leaf_ptr = next_ptr;
                        // SAFETY: next_ptr is a valid leaf.
                        leaf = unsafe { &*leaf_ptr };
                    }

                    continue 'forward;
                }

                match search_result {
                    RemoveSearchResult::NotFound => {
                        // The key may have moved to a later leaf between the
                        // search and the version check.
                        let (next_ptr, exceeded): (*mut LeafNode15<P>, bool) =
                            tree.advance_to_key_by_bound_generic(leaf_ptr, &key, guard);

                        if exceeded {
                            if !Self::bump_retry(&mut retry_count) {
                                return Err(RemoveError::RetryLimitExceeded);
                            }

                            key.unshift_all();
                            layer_root = tree.load_root_ptr_generic(guard).cast_mut();
                            route.clear();
                            continue 'layer_loop;
                        }

                        if next_ptr == leaf_ptr {
                            return Ok(None);
                        }

                        leaf_ptr = next_ptr;
                        // SAFETY: next_ptr is a valid leaf.
                        leaf = unsafe { &*leaf_ptr };
                    }

                    RemoveSearchResult::Found { ki, kp } => {
                        // Lock, revalidate full key identity, and get a cursor.
                        let lock_result = Self::lock_and_verify_for_remove(
                            tree, leaf_ptr, &key, &mut route, guard, version, perm, ki, kp,
                        );

                        match lock_result {
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
                                if !Self::bump_retry(&mut retry_count) {
                                    return Err(RemoveError::RetryLimitExceeded);
                                }

                                let (next_ptr, exceeded): (*mut LeafNode15<P>, bool) =
                                    tree.advance_to_key_by_bound_generic(leaf_ptr, &key, guard);

                                if exceeded {
                                    key.unshift_all();
                                    layer_root = tree.load_root_ptr_generic(guard).cast_mut();
                                    route.clear();
                                    continue 'layer_loop;
                                }

                                if next_ptr != leaf_ptr {
                                    leaf_ptr = next_ptr;
                                    // SAFETY: next_ptr is a valid leaf.
                                    leaf = unsafe { &*leaf_ptr };
                                }
                            }

                            RemoveLockResult::RestartFromRoot => {
                                if !Self::bump_retry(&mut retry_count) {
                                    return Err(RemoveError::RetryLimitExceeded);
                                }

                                key.unshift_all();
                                layer_root = tree.load_root_ptr_generic(guard).cast_mut();
                                route.clear();
                                continue 'layer_loop;
                            }
                        }
                    }

                    RemoveSearchResult::DescendLayer { layer_ptr } => {
                        if !Self::is_sublayer_valid(layer_ptr) {
                            // The sublayer was GC'd between the search and
                            // the version check. Treat it as contention, not
                            // as a missing key.
                            if !Self::bump_retry(&mut retry_count) {
                                return Err(RemoveError::RetryLimitExceeded);
                            }

                            if leaf.version().has_changed(version) {
                                continue 'forward;
                            }

                            continue 'layer_loop;
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
    /// When `SINGLE_LAYER` is true, layer/suffix branches are skipped since
    /// the key fits in a single ikey. The compiler eliminates dead branches.
    #[inline(always)]
    fn search_for_remove<const SINGLE_LAYER: bool, P>(
        leaf: &LeafNode15<P>,
        key: &Key<'_>,
        perm: &<LeafNode15<P> as TreeLeafNode<P>>::Perm,
    ) -> RemoveSearchResult
    where
        P: LeafPolicy,
    {
        let target_ikey: u64 = key.ikey();
        #[expect(clippy::cast_possible_truncation, reason = "key.current_len() <= 8")]
        let search_keylenx: u8 = key.current_len() as u8;
        let size: usize = perm.size();

        for ki in 0..size {
            let kp: usize = perm.get(ki);
            // Relaxed: OCC-validated read, permutation Acquire provides ordering.
            let slot_ikey: u64 = leaf.ikey_relaxed(kp);

            if slot_ikey < target_ikey {
                continue;
            }

            if slot_ikey > target_ikey {
                return RemoveSearchResult::NotFound;
            }

            // ikey matches - prefetch value to hide cache miss during keylenx check
            leaf.prefetch_value(kp);

            // Relaxed: OCC-validated read, permutation Acquire provides ordering.
            let slot_keylenx: u8 = leaf.keylenx_relaxed(kp);

            if SINGLE_LAYER {
                if slot_keylenx == search_keylenx {
                    return RemoveSearchResult::Found { ki, kp };
                }
            } else {
                if slot_keylenx >= LAYER_KEYLENX {
                    if key.has_suffix() {
                        let layer_ptr: *mut u8 = leaf.load_layer_raw(kp);
                        return RemoveSearchResult::DescendLayer { layer_ptr };
                    }

                    return RemoveSearchResult::NotFound;
                }

                if slot_keylenx == KSUF_KEYLENX {
                    if !key.has_suffix() {
                        continue;
                    }

                    leaf.prefetch_suffix();
                    let suffix: &[u8] = key.suffix();
                    if leaf.ksuf_equals(kp, suffix) {
                        return RemoveSearchResult::Found { ki, kp };
                    }
                    continue;
                }

                if search_keylenx <= IKEY_SIZE && slot_keylenx == search_keylenx {
                    return RemoveSearchResult::Found { ki, kp };
                }
            }
        }

        RemoveSearchResult::NotFound
    }

    /// Lock the leaf and verify the key is still present for removal.
    #[inline]
    #[expect(
        clippy::too_many_arguments,
        reason = "Pre-lock state is needed for the unchanged-leaf fast path"
    )]
    fn lock_and_verify_for_remove<'t, 'g, P, A>(
        tree: &'t MassTreeGeneric<P, A>,
        leaf_ptr: *mut LeafNode15<P>,
        key: &Key<'_>,
        route: &mut Route,
        guard: &'g LocalGuard<'g>,
        pre_version: u32,
        pre_perm: <LeafNode15<P> as TreeLeafNode<P>>::Perm,
        pre_ki: usize,
        pre_slot: usize,
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

        if leaf.deleted_layer() || leaf.version().is_deleted() {
            drop(lock);
            return RemoveLockResult::RestartFromRoot;
        }

        let new_perm: <LeafNode15<P> as TreeLeafNode<P>>::Perm = leaf.permutation();

        if new_perm.size() == 0 {
            drop(lock);
            return RemoveLockResult::Retry;
        }

        // REM-01: if the leaf did not change between the search and the
        // lock, the pre-lock logical position still identifies the same key.
        // This is the common case and avoids a redundant full-key search.
        let unchanged: bool = !leaf.version().has_changed(pre_version) && new_perm == pre_perm;

        let (new_ki, new_slot): (usize, usize) = if unchanged
            && pre_ki < new_perm.size()
            && new_perm.get(pre_ki) == pre_slot
            && !leaf.is_value_empty(pre_slot)
        {
            (pre_ki, pre_slot)
        } else {
            // Remediation: re-run the full-key search under the lock. The
            // pre-lock position can refer to a different key after a
            // same-ikey insert/remove shifted the permutation.
            // `search_for_remove` compares ikey, keylenx, and suffix bytes.
            let search_result: RemoveSearchResult = if key.has_suffix() {
                Self::search_for_remove::<false, P>(leaf, key, &new_perm)
            } else {
                Self::search_for_remove::<true, P>(leaf, key, &new_perm)
            };

            match search_result {
                RemoveSearchResult::Found { ki, kp } => (ki, kp),

                RemoveSearchResult::DescendLayer { layer_ptr: lp } => {
                    drop(lock);

                    if !Self::is_sublayer_valid(lp) {
                        return RemoveLockResult::Retry;
                    }

                    return RemoveLockResult::DescendLayer { layer_ptr: lp };
                }

                RemoveSearchResult::NotFound => {
                    drop(lock);
                    return RemoveLockResult::Retry;
                }
            }
        };

        let slot_keylenx: u8 = leaf.keylenx(new_slot);

        if slot_keylenx >= LAYER_KEYLENX {
            let lp: *mut u8 = leaf.load_layer_raw(new_slot);
            drop(lock);

            if !Self::is_sublayer_valid(lp) {
                return RemoveLockResult::Retry;
            }

            return RemoveLockResult::DescendLayer { layer_ptr: lp };
        }

        let cursor: RemoveCursor<'_, '_, P, A> = RemoveCursor::new(
            tree,
            leaf_ptr,
            lock,
            new_ki,
            new_slot,
            std::mem::take(route),
            guard,
        );

        RemoveLockResult::Ready(cursor)
    }

    // ============================================================================
    //  Sublayer Helpers
    // ============================================================================

    /// Charge one retry. Returns `false` when the budget is exhausted.
    #[inline(always)]
    const fn bump_retry(retry_count: &mut usize) -> bool {
        *retry_count += 1;
        *retry_count < MAX_RETRIES
    }

    /// Check if a sublayer is valid (not deleted) before descending.
    #[inline(always)]
    fn is_sublayer_valid(layer_ptr: *mut u8) -> bool {
        if layer_ptr.is_null() {
            return false;
        }

        // SAFETY: layer_ptr came from a valid slot, protected by guard.
        unsafe { NodeVersion::is_valid_sublayer(layer_ptr) }
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

            let (mut parent_lock, parent_ptr) = match parent_result {
                LockedParentResult::Locked(lock, ptr) => (lock, ptr),

                LockedParentResult::NoParent => {
                    drop(owned_lock);
                    return;
                }

                // GC-03: `child[0]` was already cleared from the previous
                // level. Abandoning the redirect here leaves keys in
                // [old_ikey, new_ikey) routing to a null child, which spins
                // forever. Retry acquisition instead of returning.
                LockedParentResult::RetryExhausted => {
                    StdHint::spin_loop();
                    continue;
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
                // OCC-05: the separator change is a mutation and must be
                // published through the version protocol, not just the lock.
                parent_lock.mark_insert();
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
    /// Reads `NodeVersion::is_leaf()` to dispatch, then delegates to
    /// [`Propagation::get_parent`].
    ///
    /// # Safety
    /// `node_ptr` must point to a valid leaf or internode.
    unsafe fn get_parent_erased<P: LeafPolicy>(node_ptr: *mut u8) -> *mut u8 {
        // SAFETY: Caller guarantees node_ptr points to valid leaf or internode.
        #[expect(clippy::cast_ptr_alignment)]
        let version: &NodeVersion = unsafe { &*(node_ptr.cast::<NodeVersion>()) };

        Propagation::get_parent::<P>(node_ptr, version.is_leaf())
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
    ///
    /// Reads `NodeVersion::is_leaf()` to dispatch, then delegates to
    /// [`Propagation::set_parent`]. Used by unit tests; production cascade no
    /// longer reparents a surviving child (it preserves unary levels).
    #[cfg(test)]
    #[inline(always)]
    unsafe fn set_parent_erased<P: LeafPolicy>(node_ptr: *mut u8, new_parent: *mut u8) {
        // SAFETY: Caller guarantees node_ptr points to valid leaf or internode.
        #[expect(clippy::cast_ptr_alignment, reason = "Checked by caller")]
        let version: &NodeVersion = unsafe { &*(node_ptr.cast::<NodeVersion>()) };

        Propagation::set_parent::<P>(node_ptr, new_parent, version.is_leaf());
    }
}

// ============================================================================
//  Tests
// ============================================================================

#[cfg(test)]
#[expect(clippy::unwrap_used, reason = "Tests")]
mod unit_tests;
