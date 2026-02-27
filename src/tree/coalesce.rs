//! Deferred cleanup queue for empty leaves.
//!
//! Two queue types: chain entries (pointer-based, for non-sublayer leaves) and
//! sublayer entries (route-based, re-traverse from root). The queued bit in
//! modstate deduplicates enqueue at the source.

use std::fmt::{self as StdFmt, Debug, Formatter};
use std::sync::atomic::Ordering as AtomicOrdering;

use crossbeam_queue::SegQueue;
use seize::LocalGuard;

use crate::alloc_trait::TreeAllocator;
use crate::key::Key;
use crate::leaf15::{LAYER_KEYLENX, LeafNode15};
use crate::link::Linker;
use crate::nodeversion::NodeVersion;
use crate::policy::LeafPolicy;
use crate::tree::MassTreeGeneric;
use crate::tree::remove::NodeCleaner;

// ============================================================================
//  Route type
// ============================================================================

/// Compact route from tree root to a sublayer's parent.
/// Each element is the ikey (u64) at that layer level.
pub type Route = Vec<u64>;

// ============================================================================
//  Entry types
// ============================================================================

/// Maximum number of times an entry can be re-queued before being dropped.
const MAX_REQUEUE_COUNT: u8 = 10;

/// Maximum B-link chain hops during GC re-traversal.
const MAX_GC_BLINK_HOPS: usize = 64;

/// Non-sublayer empty leaf pending chain unlink.
#[derive(Debug, Clone)]
struct ChainEntry {
    /// Type-erased pointer to the empty leaf.
    leaf_ptr: *mut u8,
    ikey_bound: u64,
    requeue_count: u8,
}

/// Sublayer empty leaf pending route-based gc.
#[derive(Debug, Clone)]
struct SublayerEntry {
    /// Per-layer ikey segments, root to parent.
    route: Route,
    requeue_count: u8,
}

// ============================================================================
//  CoalesceQueue
// ============================================================================

/// Lock-free queue of empty leaves pending cleanup.
pub struct CoalesceQueue {
    chains: SegQueue<ChainEntry>,
    sublayers: SegQueue<SublayerEntry>,
}

// SAFETY: CoalesceQueue requires unsafe Send/Sync because the `chains` field
// contains ChainEntry with *mut u8 (which is !Send). The pointer is only
// dereferenced under proper synchronization (seize guard + leaf lock).
// The `sublayers` field contains SublayerEntry (Vec<u64> + u8), which is
// trivially Send+Sync.
#[allow(clippy::non_send_fields_in_send_ty)]
unsafe impl Send for CoalesceQueue {}
unsafe impl Sync for CoalesceQueue {}

impl Default for CoalesceQueue {
    fn default() -> Self {
        Self::new()
    }
}

impl CoalesceQueue {
    #[must_use]
    pub const fn new() -> Self {
        Self {
            chains: SegQueue::new(),
            sublayers: SegQueue::new(),
        }
    }

    /// Schedule a non-sublayer empty leaf for chain coalesce.
    #[inline(always)]
    pub fn schedule_chain(&self, leaf_ptr: *mut u8, ikey_bound: u64) {
        self.chains.push(ChainEntry {
            leaf_ptr,
            ikey_bound,
            requeue_count: 0,
        });
    }

    /// Schedule a sublayer for route-based gc.
    #[inline(always)]
    pub fn schedule_sublayer(&self, route: Route) {
        self.sublayers.push(SublayerEntry {
            route,
            requeue_count: 0,
        });
    }

    #[must_use]
    #[inline]
    #[allow(dead_code)]
    pub fn is_empty(&self) -> bool {
        self.chains.is_empty() && self.sublayers.is_empty()
    }

    #[must_use]
    #[inline]
    pub fn len(&self) -> usize {
        self.chains.len() + self.sublayers.len()
    }

    /// Clear all pending entries without processing.
    pub fn clear(&self) {
        while self.chains.pop().is_some() {}
        while self.sublayers.pop().is_some() {}
    }
}

impl Debug for CoalesceQueue {
    fn fmt(&self, f: &mut Formatter<'_>) -> StdFmt::Result {
        f.debug_struct("CoalesceQueue")
            .field("chains", &self.chains.len())
            .field("sublayers", &self.sublayers.len())
            .finish()
    }
}

// ============================================================================
//  Route lookup result
// ============================================================================

/// Result of a GC route lookup operation.
enum RouteLookupResult<T> {
    /// Target found successfully.
    Found(T),

    /// OCC validation failed or lock contention. Requeue, keep queued bit.
    Retry,

    /// Route is truly stale (slot absent, leaf version stable). Drop entry.
    /// The queued bit cannot be cleared because the sublayer is unreachable
    /// without the route.
    NotFound,

    /// Exceeded B-link chain walk budget. Requeue, keep queued bit.
    HopLimit,
}

/// Result of re-traversal: only the parent leaf, not the slot.
/// The slot must be re-found under lock to avoid stale-index bugs.
struct FoundParent<P: LeafPolicy> {
    parent_ptr: *mut LeafNode15<P>,
    last_ikey: u64,
}

// ============================================================================
//  Coalesce processor
// ============================================================================

pub struct Coalesce;

impl Coalesce {
    /// Process all queued removals.
    pub fn process_all<P, A>(tree: &MassTreeGeneric<P, A>, guard: &LocalGuard<'_>) -> usize
    where
        P: LeafPolicy,
        A: TreeAllocator<P>,
    {
        let mut processed: usize = 0;

        while Self::try_remove_one::<P, A>(tree, guard) {
            processed += 1;
        }

        processed
    }

    /// Process up to `limit` queued removals.
    pub fn process_batch<P, A>(
        tree: &MassTreeGeneric<P, A>,
        guard: &LocalGuard<'_>,
        limit: usize,
    ) -> usize
    where
        P: LeafPolicy,
        A: TreeAllocator<P>,
    {
        let mut processed: usize = 0;

        while processed < limit && Self::try_remove_one::<P, A>(tree, guard) {
            processed += 1;
        }

        processed
    }

    /// Try to remove one empty leaf from the queue.
    /// Drains sublayer queue first (higher priority).
    fn try_remove_one<P, A>(tree: &MassTreeGeneric<P, A>, guard: &LocalGuard<'_>) -> bool
    where
        P: LeafPolicy,
        A: TreeAllocator<P>,
    {
        let queue: &CoalesceQueue = tree.coalesce_queue();

        // Sublayer priority: dead sublayer trees consume memory without
        // serving any lookup. Chain leaves are still part of the B-link
        // chain and serve as routing nodes, so delayed cleanup is less harmful.
        if let Some(entry) = queue.sublayers.pop() {
            return Self::try_remove_sublayer::<P, A>(
                tree,
                queue,
                entry.route,
                entry.requeue_count,
                guard,
            );
        }

        if let Some(entry) = queue.chains.pop() {
            return Self::try_remove_chain::<P, A>(
                tree,
                queue,
                entry.leaf_ptr,
                entry.ikey_bound,
                entry.requeue_count,
                guard,
            );
        }

        false
    }

    // ========================================================================
    //  Chain coalesce (non-sublayer leaves)
    // ========================================================================

    /// Process a chain entry: non-sublayer empty leaf pending unlink.
    fn try_remove_chain<P, A>(
        tree: &MassTreeGeneric<P, A>,
        queue: &CoalesceQueue,
        leaf_ptr_erased: *mut u8,
        ikey_bound: u64,
        requeue_count: u8,
        guard: &LocalGuard<'_>,
    ) -> bool
    where
        P: LeafPolicy,
        A: TreeAllocator<P>,
    {
        let leaf_ptr: *mut LeafNode15<P> = leaf_ptr_erased.cast();

        // SAFETY: leaf_ptr is valid, protected by guard.
        let leaf: &LeafNode15<P> = unsafe { &*leaf_ptr };

        let Some(mut lock) = leaf.version().try_lock() else {
            Self::requeue_chain(queue, leaf_ptr_erased, ikey_bound, requeue_count);
            return true;
        };

        if leaf.size() > 0 {
            drop(lock);
            return true;
        }

        if leaf.deleted_layer() {
            drop(lock);
            return true;
        }

        if leaf.prev(guard).is_null() {
            // Leftmost leaf: cannot unlink. If isolated (no next), nothing to do.
            drop(lock);
            return true;
        }

        // Non-leftmost empty leaf: unlink from chain.
        lock.mark_deleted();

        // SAFETY: We hold the lock, and prev is non-null (checked above).
        unsafe { leaf.unlink_from_chain() };

        let leaf_ikey_bound: u64 = leaf.ikey_bound();

        let parent_cleanup_succeeded: bool =
            NodeCleaner::remove_leaf_from_parent_for_coalesce::<P, A>(
                tree.allocator(),
                guard,
                leaf_ptr,
                lock,
                leaf_ikey_bound,
            );

        if parent_cleanup_succeeded {
            // SAFETY: Leaf is now unreachable from tree (marked deleted, unlinked,
            // removed from parent). Guard ensures deferred reclamation.
            unsafe { tree.allocator().retire_leaf(leaf_ptr, guard) };
        }

        true
    }

    fn requeue_chain(queue: &CoalesceQueue, leaf_ptr: *mut u8, ikey_bound: u64, count: u8) {
        if count < MAX_REQUEUE_COUNT {
            queue.chains.push(ChainEntry {
                leaf_ptr,
                ikey_bound,
                requeue_count: count + 1,
            });
        }
    }

    // ========================================================================
    //  Sublayer GC (route-based re-traversal)
    // ========================================================================

    /// Process a sublayer gc entry by re-traversing from root.
    #[cold]
    #[inline(never)]
    fn try_remove_sublayer<P, A>(
        tree: &MassTreeGeneric<P, A>,
        queue: &CoalesceQueue,
        route: Route,
        requeue_count: u8,
        guard: &LocalGuard<'_>,
    ) -> bool
    where
        P: LeafPolicy,
        A: TreeAllocator<P>,
    {
        // Step 1: Find parent via optimistic re-traversal (B-link walk + OCC).
        let found = match Self::find_sublayer_parent::<P, A>(tree, &route, guard) {
            RouteLookupResult::Found(f) => f,

            RouteLookupResult::Retry | RouteLookupResult::HopLimit => {
                Self::requeue_sublayer(queue, route, requeue_count);
                return true;
            }

            RouteLookupResult::NotFound => {
                // Route truly stale. Cannot clear queued bit (unreachable).
                return true;
            }
        };

        // Step 2: Lock parent.
        // SAFETY: found.parent_ptr is valid, protected by guard.
        let parent: &LeafNode15<P> = unsafe { &*found.parent_ptr };
        let Some(mut parent_lock) = parent.version().try_lock() else {
            Self::requeue_sublayer(queue, route, requeue_count);
            return true;
        };

        // Step 3: Re-search for layer slot under parent lock.
        // The parent could have been split between re-traversal and lock acquisition.
        let Some(parent_slot) = Self::find_layer_slot(parent, found.last_ikey) else {
            drop(parent_lock);
            Self::requeue_sublayer(queue, route, requeue_count);

            return true;
        };

        let layer_ptr: *mut u8 = parent.load_layer_raw(parent_slot);
        if layer_ptr.is_null() {
            drop(parent_lock);
            return true;
        }

        // Step 4: Lock sublayer (parent-first ordering, no deadlock).
        // SAFETY: layer_ptr is protected by guard, validated under parent lock.
        let sublayer: &LeafNode15<P> = unsafe { &*layer_ptr.cast::<LeafNode15<P>>() };
        let Some(sublayer_lock) = sublayer.version().try_lock() else {
            drop(parent_lock);
            Self::requeue_sublayer(queue, route, requeue_count);

            return true;
        };

        // Step 5a: Obsolete (sublayer non-empty or already deleted).
        if sublayer.deleted_layer() || sublayer.size() > 0 {
            sublayer.clear_queued();
            drop(sublayer_lock);
            drop(parent_lock);
            return true;
        }

        // Step 5b: Not isolated (empty but has siblings). Keep queued bit set.
        if !sublayer.prev(guard).is_null() || !sublayer.safe_next(guard).is_null() {
            drop(sublayer_lock);
            drop(parent_lock);
            Self::requeue_sublayer(queue, route, requeue_count);

            return true;
        }

        // Step 6: Clear parent slot (OCC dirty bit BEFORE mutation).
        parent_lock.mark_insert();
        parent.clear_slot_and_permutation(parent_slot);

        let parent_now_empty: bool = parent.size() == 0;
        let parent_newly_queued: bool = if parent_now_empty {
            parent.mark_empty();
            parent.try_mark_queued()
        } else {
            false
        };

        // Capture ikey_bound BEFORE dropping locks (required for cascade).
        let parent_ikey_bound: u64 = parent.ikey_bound();

        // Step 7: Mark sublayer deleted and retire.
        sublayer.mark_deleted_layer();
        drop(sublayer_lock);
        drop(parent_lock);

        // SAFETY: sublayer is unreachable from tree (parent slot cleared, sublayer
        // marked deleted). Guard protects against premature reclamation.
        unsafe {
            tree.allocator()
                .retire_leaf(layer_ptr.cast::<LeafNode15<P>>(), guard);
        }

        // Step 8: Cascade if parent became empty AND was newly queued.
        if parent_newly_queued {
            if route.len() > 1 {
                let mut parent_route: Route = route;
                parent_route.pop();

                queue.schedule_sublayer(parent_route);
            } else {
                queue.schedule_chain(found.parent_ptr.cast::<u8>(), parent_ikey_bound);
            }
        }

        true
    }

    fn requeue_sublayer(queue: &CoalesceQueue, route: Route, requeue_count: u8) {
        if requeue_count < MAX_REQUEUE_COUNT {
            queue.sublayers.push(SublayerEntry {
                route,
                requeue_count: requeue_count + 1,
            });
        }

        // If max retries exceded, drop the entry. The queued bit may remain
        // set on the sublayer leaf. This does not cause unsafety but can delay
        // collection until insert reuses the leaf (clear_empty_state stores 0,
        // clearing the bit) or the tree is dropped.
    }

    // ========================================================================
    //  Re-traversal helpers
    // ========================================================================

    /// Find a slot containing a layer pointer with the given ikey.
    fn find_layer_slot<P: LeafPolicy>(leaf: &LeafNode15<P>, target_ikey: u64) -> Option<usize> {
        let perm = leaf.permutation();
        let size: usize = perm.size();

        for ki in 0..size {
            let kp: usize = perm.get(ki);
            if leaf.ikey_relaxed(kp) == target_ikey && leaf.keylenx(kp) >= LAYER_KEYLENX {
                return Some(kp);
            }
        }

        None
    }

    /// Re-traverse from tree root to find the parent leaf for a sublayer.
    fn find_sublayer_parent<P, A>(
        tree: &MassTreeGeneric<P, A>,
        route: &Route,
        guard: &LocalGuard<'_>,
    ) -> RouteLookupResult<FoundParent<P>>
    where
        P: LeafPolicy,
        A: TreeAllocator<P>,
    {
        debug_assert!(!route.is_empty());

        let mut current_root: *const u8 = tree.root_ptr.load(AtomicOrdering::Acquire);

        // Navigate intermediate layers: route[0..n-1]
        for &ikey in &route[..route.len() - 1] {
            let key: Key<'_> = Key::from_ikey(ikey);
            let leaf_ptr: *mut LeafNode15<P> =
                tree.reach_leaf_concurrent_generic(current_root, &key, true, guard);

            match Self::find_layer_in_chain(leaf_ptr, ikey, guard) {
                RouteLookupResult::Found(ptr) => current_root = ptr.cast_const(),

                RouteLookupResult::Retry => return RouteLookupResult::Retry,

                RouteLookupResult::NotFound => return RouteLookupResult::NotFound,

                RouteLookupResult::HopLimit => return RouteLookupResult::HopLimit,
            }
        }

        // Final layer: find parent leaf containing the last route ikey.
        let last_ikey: u64 = route[route.len() - 1];
        let key: Key<'_> = Key::from_ikey(last_ikey);
        let leaf_ptr: *mut LeafNode15<P> =
            tree.reach_leaf_concurrent_generic(current_root, &key, true, guard);

        match Self::advance_to_ikey(leaf_ptr, last_ikey, guard) {
            RouteLookupResult::Found(parent_ptr) => RouteLookupResult::Found(FoundParent {
                parent_ptr,
                last_ikey,
            }),

            RouteLookupResult::Retry => RouteLookupResult::Retry,

            RouteLookupResult::NotFound => RouteLookupResult::NotFound,

            RouteLookupResult::HopLimit => RouteLookupResult::HopLimit,
        }
    }

    /// Walk B-link chain from `start` to find the leaf whose key range contains
    /// `target_ikey`, then search for a layer slot matching that ikey.
    fn find_layer_in_chain<P: LeafPolicy>(
        start: *mut LeafNode15<P>,
        target_ikey: u64,
        guard: &LocalGuard<'_>,
    ) -> RouteLookupResult<*mut u8> {
        let mut leaf_ptr: *mut LeafNode15<P> = start;
        let mut hops: usize = 0;

        loop {
            if hops >= MAX_GC_BLINK_HOPS {
                return RouteLookupResult::HopLimit;
            }

            // SAFETY: leaf_ptr is valid, protected by guard.
            let leaf: &LeafNode15<P> = unsafe { &*leaf_ptr };

            // Skip deleted leaves in the chain (version deleted bit).
            if leaf.version().is_deleted() {
                let next_raw: *mut LeafNode15<P> = leaf.next_raw(guard);
                let next_ptr: *mut LeafNode15<P> = Linker::unmark_ptr(next_raw);

                if next_ptr.is_null() {
                    return RouteLookupResult::NotFound;
                }

                leaf_ptr = next_ptr;
                hops += 1;
                continue;
            }

            // A deleted_layer() leaf was already collected. Treat as Retry.
            if leaf.deleted_layer() {
                return RouteLookupResult::Retry;
            }

            // Handle split-marked next pointer before reading successor's
            // ikey_bound. Matches advance_to_key_by_bound_generic.
            let next_raw: *mut LeafNode15<P> = leaf.next_raw(guard);
            if Linker::is_marked(next_raw) {
                leaf.wait_for_split();
                continue;
            }

            let next_ptr: *mut LeafNode15<P> = Linker::unmark_ptr(next_raw);

            if !next_ptr.is_null() {
                // SAFETY: next_ptr is non-null and protected by guard.
                let next_leaf: &LeafNode15<P> = unsafe { &*next_ptr };
                let next_bound: u64 = next_leaf.ikey_bound();

                if target_ikey >= next_bound {
                    leaf_ptr = next_ptr;
                    hops += 1;
                    continue;
                }
            }

            // This leaf's range should contain target_ikey. Take OCC snapshot.
            let version: u32 = match leaf.version().try_stable() {
                Some(v) => v,

                None => return RouteLookupResult::Retry,
            };

            let slot: Option<usize> = Self::find_layer_slot(leaf, target_ikey);

            if leaf.version().has_changed_or_locked(version) {
                return RouteLookupResult::Retry;
            }

            let Some(kp) = slot else {
                return RouteLookupResult::NotFound;
            };

            let layer_ptr: *mut u8 = leaf.load_layer_raw(kp);

            // Re-check version after loading the pointer.
            if leaf.version().has_changed_or_locked(version) {
                return RouteLookupResult::Retry;
            }

            if layer_ptr.is_null() {
                return RouteLookupResult::NotFound;
            }

            // Verify the layer root is not already deleted.
            // SAFETY: layer_ptr points to a valid node, protected by guard.
            #[expect(clippy::cast_ptr_alignment, reason = "NodeVersion is first field")]
            let layer_version: &NodeVersion = unsafe { &*layer_ptr.cast::<NodeVersion>() };

            if layer_version.is_deleted() {
                return RouteLookupResult::Retry;
            }

            return RouteLookupResult::Found(layer_ptr);
        }
    }

    /// Walk B-link chain until we find the leaf whose range contains `target_ikey`.
    fn advance_to_ikey<P: LeafPolicy>(
        start: *mut LeafNode15<P>,
        target_ikey: u64,
        guard: &LocalGuard<'_>,
    ) -> RouteLookupResult<*mut LeafNode15<P>> {
        let mut leaf_ptr: *mut LeafNode15<P> = start;
        let mut hops: usize = 0;

        loop {
            if hops >= MAX_GC_BLINK_HOPS {
                return RouteLookupResult::HopLimit;
            }

            // SAFETY: leaf_ptr valid, protected by guard.
            let leaf: &LeafNode15<P> = unsafe { &*leaf_ptr };

            if leaf.version().is_deleted() {
                let next_raw: *mut LeafNode15<P> = leaf.next_raw(guard);
                let next_ptr: *mut LeafNode15<P> = Linker::unmark_ptr(next_raw);

                if next_ptr.is_null() {
                    return RouteLookupResult::NotFound;
                }

                leaf_ptr = next_ptr;
                hops += 1;
                continue;
            }

            if leaf.deleted_layer() {
                return RouteLookupResult::Retry;
            }

            // Handle split-marked next pointer before reading successor's ikey_bound.
            let next_raw: *mut LeafNode15<P> = leaf.next_raw(guard);

            if Linker::is_marked(next_raw) {
                leaf.wait_for_split();
                continue;
            }

            let next_ptr: *mut LeafNode15<P> = Linker::unmark_ptr(next_raw);

            if !next_ptr.is_null() {
                // SAFETY: next_ptr non-null, protected by guard.
                let next_leaf: &LeafNode15<P> = unsafe { &*next_ptr };

                if target_ikey >= next_leaf.ikey_bound() {
                    leaf_ptr = next_ptr;
                    hops += 1;
                    continue;
                }
            }

            return RouteLookupResult::Found(leaf_ptr);
        }
    }
}

// ============================================================================
//  Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    use std::ptr;

    #[test]
    fn test_queue_basic_operations() {
        let queue: CoalesceQueue = CoalesceQueue::new();

        assert!(queue.is_empty());
        assert_eq!(queue.len(), 0);

        queue.schedule_chain(ptr::null_mut(), 100);
        queue.schedule_chain(ptr::null_mut(), 200);

        assert!(!queue.is_empty());
        assert_eq!(queue.len(), 2);

        queue.clear();
        assert!(queue.is_empty());
    }

    #[test]
    fn test_debug_impl() {
        let queue = CoalesceQueue::new();
        queue.schedule_chain(ptr::null_mut(), 42);

        let debug_str = format!("{queue:?}");
        assert!(debug_str.contains("CoalesceQueue"));
        assert!(debug_str.contains("chains"));
    }

    #[test]
    fn test_schedule_sublayer() {
        let queue = CoalesceQueue::new();

        queue.schedule_sublayer(vec![0x1234, 0x5678]);
        assert_eq!(queue.len(), 1);

        queue.schedule_chain(ptr::null_mut(), 300);
        assert_eq!(queue.len(), 2);

        queue.clear();
        assert!(queue.is_empty());
    }

    #[test]
    fn test_requeue_count_limit() {
        let entry: ChainEntry = ChainEntry {
            leaf_ptr: ptr::null_mut(),
            ikey_bound: 42,
            requeue_count: 0,
        };

        assert_eq!(entry.requeue_count, 0);

        let sub_entry: SublayerEntry = SublayerEntry {
            route: vec![1, 2, 3],
            requeue_count: 0,
        };
        assert_eq!(sub_entry.requeue_count, 0);

        const {
            assert!(
                MAX_REQUEUE_COUNT >= 5,
                "MAX_REQUEUE_COUNT should be at least 5"
            );
        }
        const {
            assert!(
                MAX_REQUEUE_COUNT <= 20,
                "MAX_REQUEUE_COUNT should be at most 20"
            );
        }
    }
}
