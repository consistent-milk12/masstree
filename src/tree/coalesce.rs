//! Deferred cleanup queue for empty leaves.
//!
//! This module implements lazy coalescing for the Masstree. When leaves become
//! empty after key removal, they are queued for background cleanup rather than
//! being removed inline. This avoids the lock-coupling issues that caused
//! infinite loops.

use std::fmt::{self as StdFmt, Debug, Formatter};

use crossbeam_queue::SegQueue;
use seize::LocalGuard;

use crate::alloc_trait::TreeAllocator;
use crate::leaf15::LeafNode15;
use crate::policy::LeafPolicy;
use crate::tree::remove::NodeCleaner;

/// Context for sublayer cleanup.
#[derive(Debug, Clone, Copy)]
pub struct SublayerContext {
    /// Pointer to the parent leaf (type-erased for generics).
    pub parent_leaf: *const u8,

    /// Physical slot index in the parent leaf containing the layer pointer.
    pub parent_slot: usize,
}

// SAFETY: SublayerContext is Send/Sync because:
// - parent_leaf is a raw pointer to a node protected by the tree's concurrency protocol
// - The pointer is only dereferenced while holding proper locks
// - The guard ensures the node is not deallocated during use
unsafe impl Send for SublayerContext {}
unsafe impl Sync for SublayerContext {}

/// Maximum number of times an entry can be re-queued before being dropped.
const MAX_REQUEUE_COUNT: u8 = 10;

/// Entry in the coalesce queue: pointer to empty leaf and its `ikey_bound`.
#[derive(Debug, Clone)]
struct CoalesceEntry<L> {
    /// Pointer to the empty leaf.
    leaf_ptr: *mut L,

    /// The `ikey_bound` of the leaf (for debugging/logging).
    ikey_bound: u64,

    /// Context chain for sublayer cleanup.
    layer_contexts: Vec<SublayerContext>,

    /// Number of times this entry has been re-queued due to lock contention.
    requeue_count: u8,
}

// SAFETY: CoalesceEntry contains raw pointers but is only accessed under
// proper synchronization (Mutex for the queue, guard for leaf access).
unsafe impl<L: Send> Send for CoalesceEntry<L> {}
unsafe impl<L: Sync> Sync for CoalesceEntry<L> {}

/// Lock-free queue of empty leaves pending cleanup.
pub struct CoalesceQueue<L> {
    /// Lock-free pending cleanup entries.
    pending: SegQueue<CoalesceEntry<L>>,
}

impl<L> Default for CoalesceQueue<L> {
    fn default() -> Self {
        Self::new()
    }
}

impl<L> CoalesceQueue<L> {
    /// Create a new empty coalesce queue.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            pending: SegQueue::new(),
        }
    }

    /// Schedule an empty leaf for cleanup.
    #[inline(always)]
    pub fn schedule(
        &self,
        leaf_ptr: *mut L,
        ikey_bound: u64,
        layer_contexts: Vec<SublayerContext>,
    ) {
        let entry = CoalesceEntry {
            leaf_ptr,
            ikey_bound,
            layer_contexts,
            requeue_count: 0,
        };
        // Lock-free push - never blocks!
        self.pending.push(entry);
    }

    /// Check if the queue is empty.
    #[must_use]
    #[inline]
    #[allow(dead_code)]
    pub fn is_empty(&self) -> bool {
        self.pending.is_empty()
    }

    /// Get approximate queue length.
    #[must_use]
    #[inline]
    pub fn len(&self) -> usize {
        self.pending.len()
    }

    /// Clear all pending entries without processing.
    pub fn clear(&self) {
        while self.pending.pop().is_some() {}
    }
}

impl<L> Debug for CoalesceQueue<L> {
    fn fmt(&self, f: &mut Formatter<'_>) -> StdFmt::Result {
        f.debug_struct("CoalesceQueue")
            .field("pending_count", &self.pending.len())
            .finish()
    }
}

// SAFETY: CoalesceQueue is Send if L is Send because:
// - SegQueue<T> is Send when T: Send
// - CoalesceEntry<L> contains *mut LeafNode15<P>, but L: Send means the pointer can cross threads
unsafe impl<L: Send> Send for CoalesceQueue<L> {}

// SAFETY: CoalesceQueue is Sync if L is Send because:
// - SegQueue<T> is Sync when T: Send (allows shared access that moves T between threads)
// - Leaf pointers are only dereferenced under proper synchronization (seize guard)
// - Note: L: Send (not L: Sync) because SegQueue may move entries between threads
unsafe impl<L: Send> Sync for CoalesceQueue<L> {}

pub struct Coalesce;

impl Coalesce {
    /// Process all queued removals.
    pub fn process_all<P, A>(
        queue: &CoalesceQueue<LeafNode15<P>>,
        allocator: &A,
        guard: &LocalGuard<'_>,
    ) -> usize
    where
        P: LeafPolicy,
        A: TreeAllocator<P>,
    {
        let mut processed = 0;

        while Self::try_remove_one::<P, A>(queue, allocator, guard) {
            processed += 1;
        }

        processed
    }

    /// Process up to `limit` queued removals.
    ///
    /// Useful for bounded cleanup during normal operations.
    pub fn process_batch<P, A>(
        queue: &CoalesceQueue<LeafNode15<P>>,
        allocator: &A,
        guard: &LocalGuard<'_>,
        limit: usize,
    ) -> usize
    where
        P: LeafPolicy,
        A: TreeAllocator<P>,
    {
        let mut processed = 0;

        while processed < limit && Self::try_remove_one::<P, A>(queue, allocator, guard) {
            processed += 1;
        }

        processed
    }

    /// Try to remove one empty leaf from the queue.
    fn try_remove_one<P, A>(
        queue: &CoalesceQueue<LeafNode15<P>>,
        allocator: &A,
        guard: &LocalGuard<'_>,
    ) -> bool
    where
        P: LeafPolicy,
        A: TreeAllocator<P>,
    {
        // Lock-free pop from the queue
        let Some(entry) = queue.pending.pop() else {
            return false;
        };

        let leaf_ptr: *mut LeafNode15<P> = entry.leaf_ptr;
        let entry_ikey_bound: u64 = entry.ikey_bound;

        let leaf: &LeafNode15<P> = unsafe { &*leaf_ptr };

        let Some(mut lock) = leaf.version().try_lock() else {
            let new_count = entry.requeue_count.saturating_add(1);

            if new_count <= MAX_REQUEUE_COUNT {
                let requeue_entry = CoalesceEntry {
                    leaf_ptr,
                    ikey_bound: entry_ikey_bound,
                    layer_contexts: entry.layer_contexts,
                    requeue_count: new_count,
                };

                queue.pending.push(requeue_entry);
            }

            return true;
        };

        if leaf.size() > 0 {
            drop(lock);
            return true;
        }

        if leaf.prev(guard).is_null() {
            if leaf.safe_next(guard).is_null() {
                if !entry.layer_contexts.is_empty() {
                    return Self::gc_layer::<P, A>(
                        allocator,
                        guard,
                        queue,
                        leaf_ptr,
                        lock,
                        entry.layer_contexts,
                    );
                }

                drop(lock);

                return true;
            }

            drop(lock);
            return true;
        }

        lock.mark_deleted();

        // SAFETY: We hold the lock, and prev is non-null (checked above).
        unsafe { leaf.unlink_from_chain() };

        let ikey_bound: u64 = leaf.ikey_bound();

        let parent_cleanup_succeeded = NodeCleaner::remove_leaf_from_parent_for_coalesce::<P, A>(
            allocator, guard, leaf_ptr, lock, ikey_bound,
        );

        if parent_cleanup_succeeded {
            // SAFETY: Leaf is now unreachable from tree (marked deleted, unlinked,
            // removed from parent). Guard ensures deferred reclamation.
            unsafe { allocator.retire_leaf(leaf_ptr, guard) };
        }

        true
    }

    /// Garbage collect an empty sublayer, cascading up twig chains.
    #[cold]
    #[inline(never)]
    fn gc_layer<P, A>(
        allocator: &A,
        guard: &LocalGuard<'_>,
        queue: &CoalesceQueue<LeafNode15<P>>,
        leaf_ptr: *mut LeafNode15<P>,
        mut lock: crate::nodeversion::LockGuard<'_>,
        mut contexts: Vec<SublayerContext>,
    ) -> bool
    where
        P: LeafPolicy,
        A: TreeAllocator<P>,
    {
        use crate::leaf15::LAYER_KEYLENX;

        const MAX_LOCK_RETRIES: u32 = 1000;

        // Pop the immediate parent context (last element = deepest ancestor).
        let ctx: SublayerContext = contexts.pop().expect("gc_layer called with empty contexts");

        // SAFETY: leaf_ptr is valid and locked (we hold `lock`).
        let leaf: &LeafNode15<P> = unsafe { &*leaf_ptr };

        leaf.mark_deleted_layer();
        lock.mark_deleted();

        // Get sublayer pointer before releasing lock
        let sublayer_ptr: *mut u8 = leaf_ptr.cast::<u8>();

        drop(lock);

        // SAFETY: ctx.parent_leaf was a valid leaf pointer obtained during traversal
        // and protected by the guard.
        let parent_leaf: &LeafNode15<P> = unsafe { &*(ctx.parent_leaf.cast::<LeafNode15<P>>()) };

        let mut parent_lock = None;

        for _ in 0..MAX_LOCK_RETRIES {
            if let Some(lock) = parent_leaf.version().try_lock() {
                parent_lock = Some(lock);
                break;
            }

            std::hint::spin_loop();
        }

        let Some(mut parent_lock) = parent_lock else {
            return true;
        };

        let current_keylenx: u8 = parent_leaf.keylenx(ctx.parent_slot);

        if current_keylenx < LAYER_KEYLENX {
            drop(parent_lock);

            // SAFETY: leaf_ptr is a valid leaf allocated via allocator.
            unsafe { allocator.retire_leaf(leaf_ptr, guard) };

            return true;
        }

        let current_layer_ptr: *mut u8 = parent_leaf.load_layer_raw(ctx.parent_slot);

        if current_layer_ptr != sublayer_ptr {
            drop(parent_lock);
            unsafe { allocator.retire_leaf(leaf_ptr, guard) };
            return true;
        }

        parent_leaf.clear_slot_and_permutation(ctx.parent_slot);

        let parent_now_empty: bool = parent_leaf.size() == 0;

        if parent_now_empty {
            parent_leaf.mark_empty();
        }
        let parent_ikey_bound: u64 = parent_leaf.ikey_bound();

        parent_lock.mark_insert();
        drop(parent_lock);

        // SAFETY: leaf_ptr is valid, marked deleted_layer, and unlinked from parent.
        unsafe { allocator.retire_leaf(leaf_ptr, guard) };

        if parent_now_empty && !contexts.is_empty() {
            // SAFETY: ctx.parent_leaf was originally *mut u8 in LayerContext, cast to
            // *const in SublayerContext.
            let parent_leaf_mut: *mut LeafNode15<P> =
                ctx.parent_leaf.cast_mut().cast::<LeafNode15<P>>();
            queue.schedule(parent_leaf_mut, parent_ikey_bound, contexts);
        }

        true
    }
}

// ============================================================================
//  Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    use std::ptr;

    // Test with a dummy type to verify queue operations
    #[test]
    fn test_queue_basic_operations() {
        let queue: CoalesceQueue<u8> = CoalesceQueue::new();

        assert!(queue.is_empty());
        assert_eq!(queue.len(), 0);

        // Schedule some entries (no layer context)
        queue.schedule(ptr::null_mut(), 100, Vec::new());
        queue.schedule(ptr::null_mut(), 200, Vec::new());

        assert!(!queue.is_empty());
        assert_eq!(queue.len(), 2);

        // Clear
        queue.clear();
        assert!(queue.is_empty());
    }

    #[test]
    fn test_debug_impl() {
        let queue: CoalesceQueue<u8> = CoalesceQueue::new();
        queue.schedule(ptr::null_mut(), 42, Vec::new());
        let debug_str = format!("{queue:?}");
        assert!(debug_str.contains("CoalesceQueue"));
        assert!(debug_str.contains("pending_count"));
    }

    #[test]
    fn test_schedule_with_layer_context() {
        let queue: CoalesceQueue<u8> = CoalesceQueue::new();

        // Schedule with sublayer context chain
        let ctx = SublayerContext {
            parent_leaf: ptr::without_provenance(0x1234),
            parent_slot: 5,
        };
        queue.schedule(ptr::null_mut(), 300, vec![ctx]);

        assert_eq!(queue.len(), 1);
    }

    #[test]
    fn test_requeue_count_limit() {
        // Verify that CoalesceEntry has the requeue_count field
        let entry: CoalesceEntry<u8> = CoalesceEntry {
            leaf_ptr: ptr::null_mut(),
            ikey_bound: 42,
            layer_contexts: Vec::new(),
            requeue_count: 0,
        };
        assert_eq!(entry.requeue_count, 0);

        // Verify max requeue constant is reasonable
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
