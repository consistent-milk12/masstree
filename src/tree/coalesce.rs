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
use crate::leaf_trait::{LayerCapableLeaf, TreePermutation};
use crate::slot::ValueSlot;

/// Entry in the coalesce queue: pointer to empty leaf and its `ikey_bound`.
#[derive(Debug, Clone, Copy)]
struct CoalesceEntry<L> {
    /// Pointer to the empty leaf.
    leaf_ptr: *mut L,
    /// The `ikey_bound` of the leaf (for debugging/logging).
    #[allow(dead_code)]
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

/// Try to remove a single queued leaf.
///
/// Returns `true` if a leaf was processed (removed or skipped),
/// `false` if the queue was empty.
///
/// # Leaf Processing
///
/// For each dequeued leaf:
/// 1. Try to lock - if contention, re-queue for later
/// 2. Check if still empty - if not (reused by insert), skip
/// 3. Check if leftmost - leftmost leaves cannot be removed
/// 4. Remove: mark deleted, unlink, retire
///
/// # Arguments
///
/// * `queue` - The coalesce queue to process
/// * `allocator` - Node allocator for retirement
/// * `guard` - Epoch guard for safe memory reclamation
pub fn try_remove_one<S, L, A>(
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
    // Pop an entry from the queue
    let entry = {
        let mut q = queue.pending.lock();
        q.pop_front()
    };

    let Some(entry) = entry else {
        return false; // Queue empty
    };

    let leaf_ptr = entry.leaf_ptr;

    // SAFETY: leaf_ptr came from a valid allocation and is protected by guard.
    let leaf: &L = unsafe { &*leaf_ptr };

    // Step 1: Try to lock the leaf
    let Some(mut lock) = leaf.version().try_lock() else {
        // Contention - re-queue for later
        queue.pending.lock().push_back(entry);
        return true; // Still "processed" an entry
    };

    // Step 2: Check if still empty (might have been reused by insert)
    if leaf.permutation().size() > 0 {
        // Leaf was reused - nothing to do
        drop(lock);
        return true;
    }

    // Step 3: Check if leftmost (cannot remove leftmost leaf)
    if leaf.prev().is_null() {
        // Leftmost leaf - stays as sentinel
        drop(lock);
        return true;
    }

    // Step 4: Safe to remove - mark deleted, unlink, retire
    // Mark as deleted (sets splitting flag in version)
    lock.mark_deleted();

    // Unlink from B-link chain
    // SAFETY: We hold the lock, and prev is non-null (checked above)
    unsafe { leaf.unlink_from_chain() };

    // Retire the leaf via seize
    // SAFETY: The leaf is marked deleted and unlinked, no new references
    // can be acquired. Existing references are protected by guards.
    unsafe { allocator.retire_leaf(leaf_ptr, guard) };

    // Note: Parent cleanup is skipped for now (Phase 11 in TODO.md).
    // Traversal already handles stale/deleted child pointers via retry.
    true
}

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
/// * `guard` - Epoch guard for safe memory reclamation
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
    while try_remove_one::<S, L, A>(queue, allocator, guard) {
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
/// * `guard` - Epoch guard for safe memory reclamation
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
    while processed < limit && try_remove_one::<S, L, A>(queue, allocator, guard) {
        processed += 1;
    }
    processed
}

impl<L> std::fmt::Debug for CoalesceQueue<L> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let len = self.pending.lock().len();
        f.debug_struct("CoalesceQueue")
            .field("pending_count", &len)
            .finish()
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
