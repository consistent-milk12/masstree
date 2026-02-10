//! Deferred cleanup queue for empty leaves.
//!
//! This module implements lazy coalescing for the Masstree. When leaves become
//! empty after key removal, they are queued for background cleanup rather than
//! being removed inline. This avoids the lock-coupling issues that caused
//! infinite loops in the C++ port.
//!
//! # Design
//!
//! The key insight is that empty leaves don't need immediate removal:
//! 1. **Reuse**: Insert operations can reuse empty leaves (Phase 8)
//! 2. **Deferred cleanup**: Background processing removes truly stale leaves
//! 3. **Safe traversal**: Traversal already handles deleted nodes via retry
//!
//! # Safety
//!
//! Leaf retirement is now safe because we properly implement parent cleanup:
//! 1. Mark leaf as deleted (version bit)
//! 2. Unlink from B-link chain (prev/next)
//! 3. Remove from parent internode (lock coupling walk)
//! 4. Retire leaf (epoch-based reclamation)
//!
//! After step 3, the leaf is unreachable from the tree. Existing references
//! are protected by seize guards and will be reclaimed safely.
//!
//! # Thread Safety
//!
//! The queue uses `crossbeam_queue::SegQueue` for lock-free operations.
//! This provides:
//! - Zero contention on push (producers never block)
//! - Lock-free pop for consumers
//! - Excellent scalability under high remove rates

use std::fmt::{self as StdFmt, Debug, Formatter};

use crossbeam_queue::SegQueue;
use seize::LocalGuard;

use crate::alloc_trait::TreeAllocator;
use crate::leaf15::LeafNode15;
use crate::policy::LeafPolicy;
use crate::tree::remove::NodeCleaner;

/// Context for sublayer cleanup - tracks the parent leaf and slot.
///
/// When a sublayer's only leaf becomes empty, we need to clear the parent
/// leaf's layer slot. This struct stores that information.
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
///
/// If a leaf stays locked for this many attempts (e.g., due to a crashed thread
/// holding the lock), we give up and drop the entry. The leaf remains allocated
/// but logically deleted - a bounded memory leak is preferable to infinite loops.
const MAX_REQUEUE_COUNT: u8 = 10;

/// Entry in the coalesce queue: pointer to empty leaf and its `ikey_bound`.
#[derive(Debug, Clone)]
struct CoalesceEntry<L> {
    /// Pointer to the empty leaf.
    leaf_ptr: *mut L,

    /// The `ikey_bound` of the leaf (for debugging/logging).
    ikey_bound: u64,

    /// Context chain for sublayer cleanup.
    /// Contains one entry per layer descended. When the leaf is inside a sublayer,
    /// the last element is the immediate parent; earlier elements are higher ancestors.
    /// Used by `gc_layer` to cascade cleanup up twig chains.
    layer_contexts: Vec<SublayerContext>,

    /// Number of times this entry has been re-queued due to lock contention.
    /// When this exceeds `MAX_REQUEUE_COUNT`, the entry is dropped.
    requeue_count: u8,
}

// SAFETY: CoalesceEntry contains raw pointers but is only accessed under
// proper synchronization (Mutex for the queue, guard for leaf access).
unsafe impl<L: Send> Send for CoalesceEntry<L> {}
unsafe impl<L: Sync> Sync for CoalesceEntry<L> {}

/// Lock-free queue of empty leaves pending cleanup.
///
/// Uses `crossbeam_queue::SegQueue` for wait-free push operations.
/// Leaves are added when they become empty after key removal. The queue
/// is processed during low-contention periods via `process_all()`.
///
/// # Memory Management
///
/// Leaves are only retired after:
/// 1. Successfully locking the leaf
/// 2. Verifying it's still empty (not reused by insert)
/// 3. Unlinking from the B-link chain
/// 4. Marking as deleted
///
/// # Performance
///
/// - `schedule()`: Lock-free push, never blocks
/// - `is_empty()`: Lock-free read
/// - `len()`: O(n) - walks segments (use sparingly)
/// - `clear()`: O(n) - pops all entries
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
    ///
    /// **Lock-free**: This never blocks, even under heavy contention.
    /// Called from `finish_remove()` when a leaf becomes empty.
    /// The leaf must already be marked with `mark_empty()`.
    ///
    /// # Arguments
    ///
    /// * `leaf_ptr` - Pointer to the empty leaf
    /// * `ikey_bound` - The `ikey_bound` of the leaf (for correctness checks)
    /// * `layer_contexts` - Chain of sublayer contexts for cascading `gc_layer` cleanup
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
    ///
    /// **Lock-free**: This is a fast, non-blocking check.
    #[must_use]
    #[inline]
    #[allow(dead_code)] // Public API, may be used by external code
    pub fn is_empty(&self) -> bool {
        self.pending.is_empty()
    }

    /// Get approximate queue length.
    ///
    /// # Performance Warning
    ///
    /// **O(n) complexity** - This walks all segments to count entries.
    /// Do NOT use on hot paths or in tight loops.
    /// Prefer `is_empty()` for emptiness checks.
    #[must_use]
    #[inline]
    pub fn len(&self) -> usize {
        self.pending.len()
    }

    /// Clear all pending entries without processing.
    ///
    /// Used during tree teardown when leaves will be freed anyway.
    ///
    /// # Performance Note
    ///
    /// **O(n) complexity** - `SegQueue` doesn't have a fast clear, so this
    /// pops entries one by one. Acceptable for teardown (infrequent).
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
    ///
    /// Call this during low-contention periods (e.g., between batch operations
    /// or in a background thread).
    ///
    /// # Returns
    ///
    /// The number of entries processed (including skipped/re-queued).
    ///
    /// # Arguments
    ///
    /// * `queue` - The coalesce queue to process
    /// * `allocator` - Node allocator for retirement
    /// * `guard` - Reclamation guard for safe memory reclamation
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
    ///
    /// # Returns
    ///
    /// The number of entries processed.
    ///
    /// # Arguments
    ///
    /// * `queue` - The coalesce queue to process
    /// * `allocator` - Node allocator for retirement
    /// * `guard` - Reclamation guard for safe memory reclamation
    /// * `limit` - Maximum number of entries to process
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
    ///
    /// Returns `true` if an entry was processed (removed or re-queued).
    ///
    /// # Algorithm
    ///
    /// 1. Pop an entry from the queue (lock-free)
    /// 2. Try to lock the leaf
    /// 3. Verify still empty
    /// 4. Verify not leftmost (cannot remove leftmost)
    /// 5. Mark as deleted
    /// 6. Unlink from B-link chain
    /// 7. **Remove from parent internode** (enables safe retirement)
    /// 8. Retire the leaf (only if parent cleanup succeeded)
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

        // SAFETY: leaf_ptr was valid when queued, and seize protects it from reclamation
        // while any thread might hold a reference.
        let leaf: &LeafNode15<P> = unsafe { &*leaf_ptr };

        // Step 1: Try to lock the leaf
        let Some(mut lock) = leaf.version().try_lock() else {
            // Leaf is locked by another thread - re-queue for later if within limit
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
            // If exceeded MAX_REQUEUE_COUNT, silently drop the entry.
            // The leaf remains allocated but logically deleted - bounded memory leak
            // is preferable to infinite re-queuing (e.g., if holder thread crashed).
            return true;
        };

        // Step 2: Verify leaf is still empty
        if leaf.size() > 0 {
            // Leaf was reused for new inserts - done
            drop(lock);
            return true;
        }

        // Step 3: Handle leftmost leaf cases
        // The leftmost leaf (prev == null) requires special handling.
        if leaf.prev(guard).is_null() {
            // Check if this is the only leaf in a sublayer (gc_layer case)
            if leaf.safe_next(guard).is_null() {
                // Both prev and next are null - this might be the only leaf in a sublayer
                if !entry.layer_contexts.is_empty() {
                    // We have layer context - this IS a sublayer. Do gc_layer cleanup.
                    return Self::gc_layer::<P, A>(
                        allocator,
                        guard,
                        queue,
                        leaf_ptr,
                        lock,
                        entry.layer_contexts,
                    );
                }
                // No layer context - this is the main tree root. Leave it empty.
                drop(lock);
                return true;
            }

            // Leftmost but not only leaf - cannot remove directly.
            // Note: We do NOT re-queue because the leftmost status won't change
            // (no key can be inserted before the leftmost leaf's ikey_bound).
            // Re-queuing would cause infinite loops in process_all().
            drop(lock);
            return true;
        }

        // Step 4: Mark as deleted
        // This sets the DELETED bit in the version, signaling to readers
        // that this leaf is being removed.
        //
        // IMPORTANT: We must mark deleted BEFORE unlinking or removing from parent,
        // because concurrent readers check is_deleted() to know to follow B-links.
        lock.mark_deleted();

        // Step 5: Unlink from B-link chain
        // This removes the leaf from the prev<->next chain so B-link traversal
        // will skip over it. The next pointer is left marked to signal deletion.
        //
        // SAFETY: We hold the lock, and prev is non-null (checked above).
        unsafe { leaf.unlink_from_chain() };

        // Step 6: Get ikey_bound for parent lookup
        // The ikey_bound identifies this leaf's position in the parent internode.
        let ikey_bound: u64 = leaf.ikey_bound();

        // Step 7: Remove from parent internode (enables safe retirement)
        //
        // This uses lock coupling to walk up the tree:
        // 1. Lock parent while leaf is still locked
        // 2. Remove child pointer from parent
        // 3. Shift remaining entries
        // 4. If parent becomes empty, continue walking up
        //
        // After this call returns true, the leaf is unreachable from the tree
        // and can be safely retired.
        //
        // Note: The lock is consumed by this function.
        let parent_cleanup_succeeded = NodeCleaner::remove_leaf_from_parent_for_coalesce::<P, A>(
            allocator, guard, leaf_ptr, lock, ikey_bound,
        );

        // Step 8: Retire the leaf (only if parent cleanup succeeded)
        //
        // SAFETY: Only retire if parent_cleanup_succeeded is true, meaning:
        // - Marked deleted (version has DELETED bit)
        // - Unlinked from B-link chain (prev/next pointers)
        // - Removed from parent internode (no child pointer references it)
        //
        // If parent cleanup failed (e.g., lock coupling exceeded retry limit),
        // the leaf is still marked deleted and unlinked from B-link chain,
        // but still reachable from parent. In this case we do NOT retire -
        // the leaf will be "leaked" but safe. This is a rare edge case.
        //
        // Existing references are protected by seize guards and will be handled
        // by epoch-based reclamation.
        if parent_cleanup_succeeded {
            // SAFETY: Leaf is now unreachable from tree (marked deleted, unlinked,
            // removed from parent). Guard ensures deferred reclamation.
            unsafe { allocator.retire_leaf(leaf_ptr, guard) };
        }
        // Note: If parent cleanup failed, the leaf remains allocated but
        // logically deleted. This is a memory leak but prevents UAF.
        // Future improvement: re-queue for retry or track leaked nodes.

        true
    }

    /// Garbage collect an empty sublayer, cascading up twig chains.
    ///
    /// Called when the only leaf in a sublayer becomes empty.
    /// Clears the parent's layer slot and retires the sublayer leaf.
    /// If the parent leaf also becomes empty (twig node), schedules it
    /// for coalesce with the remaining ancestor contexts.
    ///
    /// # Algorithm
    ///
    /// 1. Mark sublayer leaf as `deleted_layer` (special version bit)
    /// 2. Release sublayer lock BEFORE acquiring parent lock (deadlock prevention)
    /// 3. Lock the parent leaf
    /// 4. Verify the slot still points to this sublayer
    /// 5. Clear the slot and update permutation
    /// 6. Retire the sublayer leaf
    /// 7. If parent became empty, schedule it for coalesce with remaining contexts
    ///
    /// # C++ Reference
    ///
    /// `masstree_remove.hh:24-115` - `gc_layer()`
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

        // Use bounded retries instead of blocking lock() to prevent potential hangs.
        // If parent is persistently locked (e.g., concurrent heavy operations), we
        // leave the sublayer marked deleted_layer but don't retire it. This is a
        // bounded memory leak but prevents hangs and ensures safety (no dangling ptr).
        const MAX_LOCK_RETRIES: u32 = 1000;

        // Pop the immediate parent context (last element = deepest ancestor).
        let ctx: SublayerContext = contexts.pop().expect("gc_layer called with empty contexts");

        // SAFETY: leaf_ptr is valid and locked (we hold `lock`).
        let leaf: &LeafNode15<P> = unsafe { &*leaf_ptr };

        // Step 1: Mark sublayer as deleted_layer BEFORE releasing lock
        // This signals to readers that this sublayer is being cleaned up.
        leaf.mark_deleted_layer();
        lock.mark_deleted();

        // Get sublayer pointer before releasing lock
        let sublayer_ptr: *mut u8 = leaf_ptr.cast::<u8>();

        // Step 2: Release sublayer lock BEFORE acquiring parent lock
        // This prevents deadlock (always lock in parent -> child order)
        drop(lock);

        // Step 3: Try to lock the parent leaf (bounded retries to avoid hangs)
        // SAFETY: ctx.parent_leaf was a valid leaf pointer obtained during traversal
        // and protected by the guard. We released sublayer lock before acquiring parent.
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
            // Could not acquire parent lock after retries.
            // Sublayer is marked deleted_layer so readers handle it correctly.
            // We don't retire it to avoid dangling pointer in parent slot.
            // This is a bounded memory leak but prevents hangs.
            return true;
        };

        // Step 4: Verify slot still points to sublayer (may have changed concurrently)
        let current_keylenx: u8 = parent_leaf.keylenx(ctx.parent_slot);

        // Check keylenx first (cheaper) — if no longer a layer, slot changed.
        if current_keylenx < LAYER_KEYLENX {
            drop(parent_lock);
            // SAFETY: leaf_ptr is a valid leaf allocated via allocator. The leaf is
            // marked deleted_layer and no longer reachable. Guard ensures safe reclamation.
            unsafe { allocator.retire_leaf(leaf_ptr, guard) };
            return true;
        }

        let current_layer_ptr: *mut u8 = parent_leaf.load_layer_raw(ctx.parent_slot);
        if current_layer_ptr != sublayer_ptr {
            // Layer pointer changed — different sublayer installed.
            drop(parent_lock);
            unsafe { allocator.retire_leaf(leaf_ptr, guard) };
            return true;
        }

        // Step 5: Clear the layer slot in the parent
        parent_leaf.clear_slot_and_permutation(ctx.parent_slot);

        // Step 6: Check if parent became empty (twig chain cascade)
        let parent_now_empty: bool = parent_leaf.size() == 0;
        if parent_now_empty {
            parent_leaf.mark_empty();
        }
        let parent_ikey_bound: u64 = parent_leaf.ikey_bound();

        parent_lock.mark_insert();
        drop(parent_lock);

        // Step 7: Retire sublayer leaf
        // SAFETY: leaf_ptr is valid, marked deleted_layer, and unlinked from parent.
        // Guard ensures deferred reclamation after all readers finish.
        unsafe { allocator.retire_leaf(leaf_ptr, guard) };

        // Step 8: Cascade — if parent became empty and has remaining ancestor contexts,
        // schedule it for coalesce. try_remove_one will handle all checks (leftmost,
        // still empty, solitary, requeue-on-contention).
        if parent_now_empty && !contexts.is_empty() {
            // SAFETY: ctx.parent_leaf was originally *mut u8 in LayerContext, cast to
            // *const in SublayerContext. Casting back to *mut is sound — the pointer
            // identity is preserved and the node is protected by the guard.
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
