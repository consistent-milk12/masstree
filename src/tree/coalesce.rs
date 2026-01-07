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
//! The queue uses `parking_lot::Mutex` for interior mutability. This is
//! acceptable because:
//! - Cleanup is not on the hot path
//! - Contention is low (many producers, one consumer)
//! - Lock hold time is minimal (just push/pop)

use std::collections::VecDeque;

use parking_lot::Mutex;
use seize::LocalGuard;

use crate::alloc_trait::NodeAllocatorGeneric;
use crate::leaf_trait::LayerCapableLeaf;
use crate::slot::ValueSlot;
use crate::tree::remove::NodeCleaner;

/// Entry in the coalesce queue: pointer to empty leaf and its `ikey_bound`.
#[derive(Debug, Clone, Copy)]
struct CoalesceEntry<L> {
    /// Pointer to the empty leaf.
    leaf_ptr: *mut L,

    /// The `ikey_bound` of the leaf (for debugging/logging).
    ikey_bound: u64,
}

// SAFETY: CoalesceEntry contains raw pointers but is only accessed under
// proper synchronization (Mutex for the queue, guard for leaf access).
unsafe impl<L: Send> Send for CoalesceEntry<L> {}
unsafe impl<L: Sync> Sync for CoalesceEntry<L> {}

/// Queue of empty leaves pending cleanup.
///
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
pub struct CoalesceQueue<L> {
    /// Pending cleanup entries.
    pending: Mutex<VecDeque<CoalesceEntry<L>>>,
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
            pending: Mutex::new(VecDeque::new()),
        }
    }

    /// Schedule an empty leaf for cleanup.
    ///
    /// This is called from `finish_remove()` when a leaf becomes empty.
    /// The leaf must already be marked with `mark_empty()`.
    ///
    /// # Arguments
    ///
    /// * `leaf_ptr` - Pointer to the empty leaf
    /// * `ikey_bound` - The `ikey_bound` of the leaf (for correctness checks)
    #[inline]
    pub fn schedule(&self, leaf_ptr: *mut L, ikey_bound: u64) {
        let entry = CoalesceEntry {
            leaf_ptr,
            ikey_bound,
        };
        self.pending.lock().push_back(entry);
    }

    /// Check if the queue is empty.
    #[must_use]
    #[inline] // Not #[inline(always)] - takes mutex lock, not hot path
    #[allow(dead_code)]
    pub fn is_empty(&self) -> bool {
        self.pending.lock().is_empty()
    }

    /// Get the number of pending entries.
    #[must_use]
    #[inline]
    pub fn len(&self) -> usize {
        self.pending.lock().len()
    }

    /// Clear the queue without processing.
    ///
    /// Used during tree teardown when leaves will be freed anyway.
    #[inline]
    pub fn clear(&self) {
        self.pending.lock().clear();
    }
}

impl<L> std::fmt::Debug for CoalesceQueue<L> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let len = self.pending.lock().len();
        f.debug_struct("CoalesceQueue")
            .field("pending_count", &len)
            .finish()
    }
}

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
    pub fn process_all<S, L, A>(
        queue: &CoalesceQueue<L>,
        allocator: &A,
        guard: &LocalGuard<'_>,
    ) -> usize
    where
        S: ValueSlot,
        S::Value: Send + Sync + 'static,
        S::Output: Send + Sync,
        L: LayerCapableLeaf<S>,
        A: NodeAllocatorGeneric<S, L>,
    {
        let mut processed = 0;

        while Self::try_remove_one::<S, L, A>(queue, allocator, guard) {
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
    pub fn process_batch<S, L, A>(
        queue: &CoalesceQueue<L>,
        allocator: &A,
        guard: &LocalGuard<'_>,
        limit: usize,
    ) -> usize
    where
        S: ValueSlot,
        S::Value: Send + Sync + 'static,
        S::Output: Send + Sync,
        L: LayerCapableLeaf<S>,
        A: NodeAllocatorGeneric<S, L>,
    {
        let mut processed = 0;

        while processed < limit && Self::try_remove_one::<S, L, A>(queue, allocator, guard) {
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
    /// 1. Pop an entry from the queue
    /// 2. Try to lock the leaf
    /// 3. Verify still empty
    /// 4. Verify not leftmost (cannot remove leftmost)
    /// 5. Mark as deleted
    /// 6. Unlink from B-link chain
    /// 7. **Remove from parent internode** (enables safe retirement)
    /// 8. Retire the leaf (only if parent cleanup succeeded)
    fn try_remove_one<S, L, A>(
        queue: &CoalesceQueue<L>,
        allocator: &A,
        guard: &LocalGuard<'_>,
    ) -> bool
    where
        S: ValueSlot,
        S::Value: Send + Sync + 'static,
        S::Output: Send + Sync,
        L: LayerCapableLeaf<S>,
        A: NodeAllocatorGeneric<S, L>,
    {
        let entry: CoalesceEntry<L> = {
            let mut pending = queue.pending.lock();
            match pending.pop_front() {
                Some(e) => e,
                None => return false,
            }
        };

        let leaf_ptr: *mut L = entry.leaf_ptr;
        let entry_ikey_bound: u64 = entry.ikey_bound;

        // SAFETY: leaf_ptr was valid when queued, and seize protects it from reclamation
        // while any thread might hold a reference.
        let leaf: &L = unsafe { &*leaf_ptr };

        // Step 1: Try to lock the leaf
        let Some(mut lock) = leaf.version().try_lock() else {
            // Leaf is locked by another thread - re-queue for later
            queue.schedule(leaf_ptr, entry_ikey_bound);
            return true;
        };

        // Step 2: Verify leaf is still empty
        if leaf.size() > 0 {
            // Leaf was reused for new inserts - done
            drop(lock);
            return true;
        }

        // Step 3: Verify leaf is not leftmost
        // The leftmost leaf must be preserved for B-link traversal invariants
        if leaf.prev().is_null() {
            // Leftmost leaf - cannot remove, re-queue
            queue.schedule(leaf_ptr, entry_ikey_bound);
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
        let parent_cleanup_succeeded = NodeCleaner::remove_leaf_from_parent_for_coalesce::<S, L, A>(
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

        // Schedule some entries
        queue.schedule(ptr::null_mut(), 100);
        queue.schedule(ptr::null_mut(), 200);

        assert!(!queue.is_empty());
        assert_eq!(queue.len(), 2);

        // Clear
        queue.clear();
        assert!(queue.is_empty());
    }

    #[test]
    fn test_debug_impl() {
        let queue: CoalesceQueue<u8> = CoalesceQueue::new();
        queue.schedule(ptr::null_mut(), 42);
        let debug_str = format!("{queue:?}");
        assert!(debug_str.contains("CoalesceQueue"));
        assert!(debug_str.contains("pending_count"));
    }
}
