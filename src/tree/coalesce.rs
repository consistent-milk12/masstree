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

/// Entry in the coalesce queue: pointer to empty leaf and its `ikey_bound`.
#[derive(Debug, Clone, Copy)]
struct CoalesceEntry<L> {
    /// Pointer to the empty leaf.
    leaf_ptr: *mut L,

    /// The `ikey_bound` of the leaf (for debugging/logging).
    ikey_bound: u64,

    /// Optional context for sublayer cleanup.
    /// Present when this leaf is inside a sublayer (descended via layer pointer).
    layer_context: Option<SublayerContext>,
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
    /// * `layer_context` - Optional sublayer context for `gc_layer` cleanup
    #[inline]
    pub fn schedule(
        &self,
        leaf_ptr: *mut L,
        ikey_bound: u64,
        layer_context: Option<SublayerContext>,
    ) {
        let entry = CoalesceEntry {
            leaf_ptr,
            ikey_bound,
            layer_context,
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
            queue.schedule(leaf_ptr, entry_ikey_bound, entry.layer_context);
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
        if leaf.prev().is_null() {
            // Check if this is the only leaf in a sublayer (gc_layer case)
            if leaf.safe_next().is_null() {
                // Both prev and next are null - this might be the only leaf in a sublayer
                if let Some(ctx) = entry.layer_context {
                    // We have layer context - this IS a sublayer. Do gc_layer cleanup.
                    return Self::gc_layer::<S, L, A>(
                        allocator, guard, leaf_ptr, lock, ctx,
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

    /// Garbage collect an empty sublayer.
    ///
    /// Called when the only leaf in a sublayer becomes empty.
    /// Clears the parent's layer slot and retires the sublayer leaf.
    ///
    /// # Algorithm
    ///
    /// 1. Mark sublayer leaf as `deleted_layer` (special version bit)
    /// 2. Release sublayer lock BEFORE acquiring parent lock (deadlock prevention)
    /// 3. Lock the parent leaf
    /// 4. Verify the slot still points to this sublayer
    /// 5. Clear the slot and update permutation
    /// 6. Retire the sublayer leaf
    ///
    /// # C++ Reference
    ///
    /// `masstree_remove.hh:24-115` - `gc_layer()`
    #[cold]
    #[inline(never)]
    fn gc_layer<S, L, A>(
        allocator: &A,
        guard: &LocalGuard<'_>,
        leaf_ptr: *mut L,
        mut lock: crate::nodeversion::LockGuard<'_>,
        ctx: SublayerContext,
    ) -> bool
    where
        S: ValueSlot,
        S::Value: Send + Sync + 'static,
        S::Output: Send + Sync,
        L: LayerCapableLeaf<S>,
        A: NodeAllocatorGeneric<S, L>,
    {
        use crate::leaf24::LAYER_KEYLENX;

        // SAFETY: leaf_ptr is valid and locked (we hold `lock`).
        let leaf: &L = unsafe { &*leaf_ptr };

        // Step 1: Mark sublayer as deleted_layer BEFORE releasing lock
        // This signals to readers that this sublayer is being cleaned up.
        leaf.mark_deleted_layer();
        lock.mark_deleted();

        // Get sublayer pointer before releasing lock
        let sublayer_ptr: *mut u8 = leaf_ptr.cast::<u8>();

        // Step 2: Release sublayer lock BEFORE acquiring parent lock
        // This prevents deadlock (always lock in parent -> child order)
        drop(lock);

        // Step 3: Lock the parent leaf
        // SAFETY: ctx.parent_leaf was a valid leaf pointer obtained during traversal
        // and protected by the guard. We released sublayer lock before acquiring parent.
        let parent_leaf: &L = unsafe { &*(ctx.parent_leaf.cast::<L>()) };
        let mut parent_lock = parent_leaf.version().lock();

        // Step 4: Verify slot still points to sublayer (may have changed concurrently)
        let current_slot_ptr: *mut u8 = parent_leaf.leaf_value_ptr(ctx.parent_slot);
        let current_keylenx: u8 = parent_leaf.keylenx(ctx.parent_slot);

        if current_slot_ptr != sublayer_ptr || current_keylenx < LAYER_KEYLENX {
            // Slot has changed - just retire sublayer without parent update
            drop(parent_lock);

            // SAFETY: leaf_ptr is a valid leaf allocated via allocator. The leaf is
            // marked deleted_layer and no longer reachable. Guard ensures safe reclamation.
            unsafe { allocator.retire_leaf(leaf_ptr, guard) };
            return true;
        }

        // Step 5: Clear the layer slot in the parent
        parent_leaf.clear_slot_and_permutation(ctx.parent_slot);
        parent_lock.mark_insert();
        drop(parent_lock);

        // Step 6: Schedule sublayer leaf for retirement
        // SAFETY: leaf_ptr is valid, marked deleted_layer, and unlinked from parent.
        // Guard ensures deferred reclamation after all readers finish.
        unsafe { allocator.retire_leaf(leaf_ptr, guard) };

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
        queue.schedule(ptr::null_mut(), 100, None);
        queue.schedule(ptr::null_mut(), 200, None);

        assert!(!queue.is_empty());
        assert_eq!(queue.len(), 2);

        // Clear
        queue.clear();
        assert!(queue.is_empty());
    }

    #[test]
    fn test_debug_impl() {
        let queue: CoalesceQueue<u8> = CoalesceQueue::new();
        queue.schedule(ptr::null_mut(), 42, None);
        let debug_str = format!("{queue:?}");
        assert!(debug_str.contains("CoalesceQueue"));
        assert!(debug_str.contains("pending_count"));
    }

    #[test]
    fn test_schedule_with_layer_context() {
        let queue: CoalesceQueue<u8> = CoalesceQueue::new();

        // Schedule with sublayer context
        let ctx = SublayerContext {
            parent_leaf: ptr::without_provenance(0x1234),
            parent_slot: 5,
        };
        queue.schedule(ptr::null_mut(), 300, Some(ctx));

        assert_eq!(queue.len(), 1);
    }
}
