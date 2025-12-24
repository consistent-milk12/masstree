//! Node allocation abstraction for `MassTree`.
//!
//! This module provides the [`NodeAllocator`] trait that abstracts how nodes
//! are allocated and (eventually) deallocated.
//!
//! ## Allocators
//!
//! - [`SeizeAllocator`] (default): Miri-compliant allocator using `Box::into_raw()`
//!   for clean pointer provenance. Stores raw pointers for cleanup, ready for
//!   integration with seize's deferred reclamation. Uses interior mutability
//!   via `parking_lot::Mutex` to support concurrent allocation tracking.
//!
//! - [`ArenaAllocator`] (deprecated): Original Phase 1/2 allocator that stores
//!   `Box<T>` in a Vec. Causes Stacked Borrows violations under Miri.
//!
//! ## Memory Reclamation
//!
//! The [`NodeAllocator`] trait provides retirement methods for seize-based
//! memory reclamation:
//! - `retire_leaf` / `retire_internode`: Retire individual nodes
//! - `retire_subtree_root`: Retire an entire subtree (for layer teardown)
//! - `teardown_tree`: Synchronous tree teardown at `MassTree::drop`

mod reclaim;

use parking_lot::Mutex;
use seize::{Guard, LocalGuard};

use crate::internode::InternodeNode;
use crate::leaf::LeafNode;
use crate::slot::ValueSlot;

use reclaim::{
    reclaim_internode_boxed, reclaim_leaf_boxed, reclaim_subtree_impl, reclaim_subtree_root,
};

/// Trait for allocating and deallocating tree nodes.
///
/// Implementations must guarantee:
///
/// 1. **Pointer stability**: Returned pointers remain valid until explicitly
///    deallocated or the allocator is dropped.
///
/// 2. **Provenance**: Returned pointers have valid provenance for the
///    allocated memory (Stacked Borrows compliant).
///
/// 3. **Thread safety**: For concurrent allocators (Phase 3), allocation
///    must be thread-safe.
///
/// # Type Parameters
///
/// * `S` - The slot type implementing [`ValueSlot`]
/// * `WIDTH` - The node width (number of slots)
///
/// # Safety
///
/// Implementors must ensure that pointers returned by `alloc_*` methods
/// remain valid until the corresponding `dealloc_*` is called (or the
/// allocator is dropped for arena-style allocators).
pub trait NodeAllocator<S: ValueSlot, const WIDTH: usize> {
    /// Allocate a leaf node and return a stable raw pointer.
    ///
    /// The returned pointer is valid for reads and writes until:
    /// - `dealloc_leaf` is called with this pointer, OR
    /// - The allocator is dropped
    ///
    /// # Arguments
    ///
    /// * `node` - The leaf node to allocate (takes ownership)
    ///
    /// # Returns
    ///
    /// A raw mutable pointer to the allocated node with valid provenance.
    ///
    /// # Note
    ///
    /// Uses interior mutability (`parking_lot::Mutex`) so this can be called
    /// from concurrent code paths with only `&self`.
    fn alloc_leaf(&self, node: Box<LeafNode<S, WIDTH>>) -> *mut LeafNode<S, WIDTH>;

    /// Allocate an internode and return a stable raw pointer.
    ///
    /// The returned pointer is valid for reads and writes until:
    /// - `dealloc_internode` is called with this pointer, OR
    /// - The allocator is dropped
    ///
    /// # Arguments
    ///
    /// * `node` - The internode to allocate (takes ownership)
    ///
    /// # Returns
    ///
    /// A raw mutable pointer to the allocated node with valid provenance.
    ///
    /// # Note
    ///
    /// Uses interior mutability (`parking_lot::Mutex`) so this can be called
    /// from concurrent code paths with only `&self`.
    fn alloc_internode(&self, node: Box<InternodeNode<S, WIDTH>>) -> *mut InternodeNode<S, WIDTH>;

    /// Deallocate a leaf node.
    ///
    /// For arena-style allocators, this is a no-op (nodes are freed when
    /// the allocator drops). For epoch-based allocators, this schedules
    /// the node for deferred reclamation.
    ///
    /// # Safety
    ///
    /// The pointer must have been returned by a previous call to `alloc_leaf`
    /// on this allocator, and must not have been deallocated already.
    ///
    /// After this call, the pointer is invalid and must not be dereferenced.
    #[inline(always)]
    #[expect(unused_variables, reason = "by default it's no op")]
    unsafe fn dealloc_leaf(&self, ptr: *mut LeafNode<S, WIDTH>) {
        // Default: no-op for arena-style allocators
    }

    /// Deallocate an internode.
    ///
    /// For arena-style allocators, this is a no-op (nodes are freed when
    /// the allocator drops). For epoch-based allocators, this schedules
    /// the node for deferred reclamation.
    ///
    /// # Safety
    ///
    /// The pointer must have been returned by a previous call to `alloc_internode`
    /// on this allocator, and must not have been deallocated already.
    ///
    /// After this call, the pointer is invalid and must not be dereferenced.
    #[inline(always)]
    #[expect(unused_variables, reason = "by default it's no op")]
    unsafe fn dealloc_internode(&self, ptr: *mut InternodeNode<S, WIDTH>) {
        // Default: no-op for arena-style allocators
    }

    /// Track a leaf pointer for cleanup (concurrent-safe via `&self`).
    ///
    /// This method allows concurrent code paths (which only have `&self`)
    /// to register allocations made via `Box::into_raw()` for cleanup.
    ///
    /// For allocators without interior mutability, this is a no-op
    /// (they don't support concurrent allocation tracking).
    ///
    /// # Arguments
    ///
    /// * `ptr` - Raw pointer to a leaf node allocated via `Box::into_raw`
    #[inline(always)]
    #[expect(unused_variables, reason = "by default it's no op")]
    fn track_leaf(&self, ptr: *mut LeafNode<S, WIDTH>) {
        // Default: no-op for allocators without interior mutability
    }

    /// Track an internode pointer for cleanup (concurrent-safe via `&self`).
    ///
    /// This method allows concurrent code paths (which only have `&self`)
    /// to register allocations made via `Box::into_raw()` for cleanup.
    ///
    /// For allocators without interior mutability, this is a no-op
    /// (they don't support concurrent allocation tracking).
    ///
    /// # Arguments
    ///
    /// * `ptr` - Raw pointer to an internode allocated via `Box::into_raw`
    #[inline(always)]
    #[expect(unused_variables, reason = "by default it's no op")]
    fn track_internode(&self, ptr: *mut InternodeNode<S, WIDTH>) {
        // Default: no-op for allocators without interior mutability
    }

    // ========================================================================
    //  Memory Reclamation
    // ========================================================================

    /// Retire a leaf node that has been unlinked from the tree.
    ///
    /// Schedules the node for deferred reclamation via seize. The node will
    /// be freed once no readers can hold references to it.
    ///
    /// # Safety
    ///
    /// - `ptr` must have been allocated by this allocator.
    /// - `ptr` must be unreachable from the tree by any new traversal.
    /// - In-flight traversals must detect deletion/retry via the OCC protocol.
    #[inline(always)]
    #[expect(unused_variables, reason = "by default it's no op")]
    unsafe fn retire_leaf(&self, ptr: *mut LeafNode<S, WIDTH>, guard: &LocalGuard<'_>) {
        // Default: no-op for arena-style allocators
    }

    /// Retire an internode that has been unlinked from the tree.
    ///
    /// Schedules the node for deferred reclamation via seize.
    ///
    /// # Safety
    ///
    /// - `ptr` must have been allocated by this allocator.
    /// - `ptr` must be unreachable from the tree by any new traversal.
    /// - In-flight traversals must detect deletion/retry via the OCC protocol.
    #[inline(always)]
    #[expect(unused_variables, reason = "by default it's no op")]
    unsafe fn retire_internode(&self, ptr: *mut InternodeNode<S, WIDTH>, guard: &LocalGuard<'_>) {
        // Default: no-op for arena-style allocators
    }

    /// Retire an entire subtree that has been unlinked from the tree.
    ///
    /// Typically used for reclaiming a whole layer when a layer pointer is removed.
    /// The subtree will be traversed and all nodes freed once safe.
    ///
    /// # Safety
    ///
    /// - The subtree must be fully unlinked from the main tree before calling.
    /// - No other shared pointers may reference nodes exclusively through this subtree.
    #[inline(always)]
    #[expect(unused_variables, reason = "by default it's no op")]
    unsafe fn retire_subtree_root(&self, root_ptr: *mut u8, guard: &LocalGuard<'_>) {
        // Default: no-op for arena-style allocators
    }

    /// Teardown reachable nodes at `MassTree::drop`.
    ///
    /// Called when the tree is being destroyed and no concurrent access is possible.
    /// This traverses and frees all nodes reachable from the root.
    ///
    /// Arena allocators can no-op (they free via their own Drop).
    /// Seize-based allocators must implement this to free nodes by traversal.
    #[inline(always)]
    #[expect(unused_variables, reason = "by default it's no op")]
    fn teardown_tree(&self, root_ptr: *mut u8) {
        // Default: no-op for arena-style allocators
    }
}

// ============================================================================
//  SeizeAllocator
// ============================================================================

/// Miri-compliant allocator designed for concurrent use with seize reclamation.
///
/// Uses `Box::into_raw()` to obtain raw pointers with clean provenance,
/// avoiding the Stacked Borrows issues of [`ArenaAllocator`]. Raw pointers
/// are stored for cleanup when the allocator is dropped.
///
/// # Miri Compliance
///
/// This allocator passes Miri with `-Zmiri-strict-provenance` because:
/// - `Box::into_raw()` transfers full ownership to the raw pointer
/// - The Vec stores raw pointers, not Boxes, so no aliasing occurs
/// - Writes through the raw pointer don't conflict with Vec access
///
/// # Interior Mutability
///
/// Uses `parking_lot::Mutex` for interior mutability, allowing concurrent
/// code paths (which only have `&self`) to track allocations. This is
/// necessary for the concurrent insert path in `locked.rs`.
///
/// # Memory Management
///
/// Currently, nodes are freed when the allocator drops (arena-style).
/// Future: integrate with seize's `guard.defer_retire()` for concurrent
/// reclamation when nodes are unlinked during splits/removes.
///
/// # Type Parameters
///
/// * `S` - The slot type implementing [`ValueSlot`]
/// * `WIDTH` - The node width (number of slots)
pub struct SeizeAllocator<S: ValueSlot, const WIDTH: usize> {
    /// Raw pointers to allocated leaf nodes (for cleanup on drop).
    leaf_ptrs: Mutex<Vec<*mut LeafNode<S, WIDTH>>>,

    /// Raw pointers to allocated internode nodes (for cleanup on drop).
    internode_ptrs: Mutex<Vec<*mut InternodeNode<S, WIDTH>>>,
}

// SAFETY: The raw pointers are owned by this allocator and only accessed
// through the tree's synchronization protocol (version locks, seize guards).
// The Mutex provides interior mutability for concurrent allocation tracking.
unsafe impl<S: ValueSlot + Send, const WIDTH: usize> Send for SeizeAllocator<S, WIDTH> {}
unsafe impl<S: ValueSlot + Sync, const WIDTH: usize> Sync for SeizeAllocator<S, WIDTH> {}

impl<S: ValueSlot, const WIDTH: usize> std::fmt::Debug for SeizeAllocator<S, WIDTH> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SeizeAllocator")
            .field("leaf_count", &self.leaf_ptrs.lock().len())
            .field("internode_count", &self.internode_ptrs.lock().len())
            .finish()
    }
}

impl<S: ValueSlot, const WIDTH: usize> SeizeAllocator<S, WIDTH> {
    /// Create a new seize-compatible allocator.
    #[must_use]
    #[inline(always)]
    pub const fn new() -> Self {
        Self {
            leaf_ptrs: Mutex::new(Vec::new()),
            internode_ptrs: Mutex::new(Vec::new()),
        }
    }

    /// Create a new allocator with pre-allocated capacity.
    ///
    /// # Arguments
    ///
    /// * `leaf_capacity` - Initial capacity for leaf pointer storage
    /// * `internode_capacity` - Initial capacity for internode pointer storage
    #[must_use]
    #[inline(always)]
    pub fn with_capacity(leaf_capacity: usize, internode_capacity: usize) -> Self {
        Self {
            leaf_ptrs: Mutex::new(Vec::with_capacity(leaf_capacity)),
            internode_ptrs: Mutex::new(Vec::with_capacity(internode_capacity)),
        }
    }

    /// Return the number of allocated leaf nodes.
    #[must_use]
    #[inline(always)]
    pub fn leaf_count(&self) -> usize {
        self.leaf_ptrs.lock().len()
    }

    /// Return the number of allocated internodes.
    #[must_use]
    #[inline(always)]
    pub fn internode_count(&self) -> usize {
        self.internode_ptrs.lock().len()
    }

    /// Return the total number of allocated nodes.
    #[must_use]
    #[inline(always)]
    pub fn total_count(&self) -> usize {
        self.leaf_ptrs.lock().len() + self.internode_ptrs.lock().len()
    }

    /// Track a leaf pointer for cleanup (concurrent-safe via `&self`).
    ///
    /// This method allows concurrent code paths to register allocations
    /// for cleanup without requiring `&mut self`.
    ///
    /// # Arguments
    ///
    /// * `ptr` - Raw pointer to a leaf node allocated via `Box::into_raw`
    #[inline(always)]
    pub fn track_leaf(&self, ptr: *mut LeafNode<S, WIDTH>) {
        self.leaf_ptrs.lock().push(ptr);
    }

    /// Track an internode pointer for cleanup (concurrent-safe via `&self`).
    ///
    /// This method allows concurrent code paths to register allocations
    /// for cleanup without requiring `&mut self`.
    ///
    /// # Arguments
    ///
    /// * `ptr` - Raw pointer to an internode allocated via `Box::into_raw`
    #[inline(always)]
    pub fn track_internode(&self, ptr: *mut InternodeNode<S, WIDTH>) {
        self.internode_ptrs.lock().push(ptr);
    }
}

impl<S: ValueSlot, const WIDTH: usize> Default for SeizeAllocator<S, WIDTH> {
    fn default() -> Self {
        Self::new()
    }
}

impl<S: ValueSlot, const WIDTH: usize> NodeAllocator<S, WIDTH> for SeizeAllocator<S, WIDTH> {
    #[inline]
    fn alloc_leaf(&self, leaf: Box<LeafNode<S, WIDTH>>) -> *mut LeafNode<S, WIDTH> {
        // Convert Box to raw pointer immediately.
        // This gives us a pointer with clean provenance — no intermediate borrows.
        let ptr: *mut LeafNode<S, WIDTH> = Box::into_raw(leaf);

        // Store the raw pointer for cleanup on drop.
        // The Vec only holds the pointer value, not ownership of the node.
        self.leaf_ptrs.lock().push(ptr);

        ptr
    }

    #[inline(always)]
    fn alloc_internode(&self, node: Box<InternodeNode<S, WIDTH>>) -> *mut InternodeNode<S, WIDTH> {
        let ptr: *mut InternodeNode<S, WIDTH> = Box::into_raw(node);
        self.internode_ptrs.lock().push(ptr);
        ptr
    }

    #[inline]
    unsafe fn dealloc_leaf(&self, ptr: *mut LeafNode<S, WIDTH>) {
        // For now, just remove from tracking and free immediately.
        // Future: integrate with seize guard.defer_retire() for concurrent reclamation.
        let found = {
            let mut ptrs = self.leaf_ptrs.lock();
            ptrs.iter().position(|&p| p == ptr).is_some_and(|pos| {
                ptrs.swap_remove(pos);
                true
            })
        };
        if found {
            // SAFETY: ptr was created by alloc_leaf via Box::into_raw
            // (caller guarantees via trait contract)
            unsafe { drop(Box::from_raw(ptr)) };
        }
    }

    #[inline]
    unsafe fn dealloc_internode(&self, ptr: *mut InternodeNode<S, WIDTH>) {
        let found = {
            let mut ptrs = self.internode_ptrs.lock();
            ptrs.iter().position(|&p| p == ptr).is_some_and(|pos| {
                ptrs.swap_remove(pos);
                true
            })
        };
        if found {
            // SAFETY: ptr was created by alloc_internode via Box::into_raw
            // (caller guarantees via trait contract)
            unsafe { drop(Box::from_raw(ptr)) };
        }
    }

    #[inline(always)]
    fn track_leaf(&self, ptr: *mut LeafNode<S, WIDTH>) {
        self.leaf_ptrs.lock().push(ptr);
    }

    #[inline(always)]
    fn track_internode(&self, ptr: *mut InternodeNode<S, WIDTH>) {
        self.internode_ptrs.lock().push(ptr);
    }

    // ========================================================================
    //  Memory Reclamation
    // ========================================================================

    unsafe fn retire_leaf(&self, ptr: *mut LeafNode<S, WIDTH>, guard: &LocalGuard<'_>) {
        // Remove from tracking to keep counts accurate and prevent double-free.
        {
            let mut ptrs = self.leaf_ptrs.lock();
            if let Some(pos) = ptrs.iter().position(|&p| p == ptr) {
                ptrs.swap_remove(pos);
            }
        }
        // Schedule deferred reclamation via seize.
        // SAFETY: Caller guarantees ptr validity and unlink discipline.
        unsafe { guard.defer_retire(ptr, reclaim_leaf_boxed::<S, WIDTH>) };
    }

    unsafe fn retire_internode(&self, ptr: *mut InternodeNode<S, WIDTH>, guard: &LocalGuard<'_>) {
        // Remove from tracking to keep counts accurate and prevent double-free.
        {
            let mut ptrs = self.internode_ptrs.lock();
            if let Some(pos) = ptrs.iter().position(|&p| p == ptr) {
                ptrs.swap_remove(pos);
            }
        }
        // Schedule deferred reclamation via seize.
        // SAFETY: Caller guarantees ptr validity and unlink discipline.
        unsafe { guard.defer_retire(ptr, reclaim_internode_boxed::<S, WIDTH>) };
    }

    unsafe fn retire_subtree_root(&self, root_ptr: *mut u8, guard: &LocalGuard<'_>) {
        // Note: We don't remove individual nodes from tracking here.
        // The subtree reclaimer will drop them, and our Drop impl will
        // skip any that are no longer in the tracking vecs.
        //
        // SAFETY: Caller guarantees subtree is fully unlinked.
        unsafe { guard.defer_retire(root_ptr, reclaim_subtree_root::<S, WIDTH>) };
    }

    #[expect(
        clippy::not_unsafe_ptr_arg_deref,
        reason = "root_ptr is a valid tree root from MassTree::drop"
    )]
    fn teardown_tree(&self, root_ptr: *mut u8) {
        // Clear tracking to prevent double-free in Drop.
        // The subtree reclaimer will drop all nodes.
        self.leaf_ptrs.lock().clear();
        self.internode_ptrs.lock().clear();

        // Reclaim synchronously via DFS traversal.
        // SAFETY: Tree drop is exclusive access, no concurrent readers.
        unsafe { reclaim_subtree_impl::<S, WIDTH>(root_ptr) };
    }
}

impl<S: ValueSlot, const WIDTH: usize> Drop for SeizeAllocator<S, WIDTH> {
    fn drop(&mut self) {
        // Free all remaining leaf nodes.
        // SAFETY: We have &mut self, so we can use get_mut() to avoid locking.
        for ptr in self.leaf_ptrs.get_mut().drain(..) {
            // SAFETY: Each ptr was created by Box::into_raw in alloc_leaf
            // and has not been freed (would have been removed from vec).
            unsafe {
                drop(Box::from_raw(ptr));
            }
        }

        // Free all remaining internodes.
        for ptr in self.internode_ptrs.get_mut().drain(..) {
            // SAFETY: Each ptr was created by Box::into_raw in alloc_internode
            // and has not been freed (would have been removed from vec).
            unsafe {
                drop(Box::from_raw(ptr));
            }
        }
    }
}

// ============================================================================
//  NodeAllocatorGeneric Implementation
// ============================================================================

impl<S, const WIDTH: usize> crate::alloc_trait::NodeAllocatorGeneric<S, LeafNode<S, WIDTH>>
    for SeizeAllocator<S, WIDTH>
where
    S: ValueSlot + Send + Sync + 'static,
{
    #[inline(always)]
    fn alloc_leaf(&self, node: Box<LeafNode<S, WIDTH>>) -> *mut LeafNode<S, WIDTH> {
        NodeAllocator::alloc_leaf(self, node)
    }

    #[inline(always)]
    fn track_leaf(&self, ptr: *mut LeafNode<S, WIDTH>) {
        NodeAllocator::track_leaf(self, ptr);
    }

    #[inline(always)]
    unsafe fn retire_leaf(&self, ptr: *mut LeafNode<S, WIDTH>, guard: &LocalGuard<'_>) {
        // SAFETY: Caller guarantees ptr is valid and unreachable
        unsafe { NodeAllocator::retire_leaf(self, ptr, guard) }
    }

    #[inline(always)]
    fn alloc_internode_erased(&self, node_ptr: *mut u8) -> *mut u8 {
        // SAFETY: Caller passes a valid Box<InternodeNode<S, WIDTH>> as *mut u8
        let node: Box<InternodeNode<S, WIDTH>> =
            unsafe { Box::from_raw(node_ptr.cast::<InternodeNode<S, WIDTH>>()) };
        NodeAllocator::alloc_internode(self, node).cast()
    }

    #[inline(always)]
    fn track_internode_erased(&self, ptr: *mut u8) {
        NodeAllocator::track_internode(self, ptr.cast());
    }

    #[inline(always)]
    unsafe fn retire_internode_erased(&self, ptr: *mut u8, guard: &LocalGuard<'_>) {
        // SAFETY: Caller guarantees ptr is valid InternodeNode<S, WIDTH>
        unsafe { NodeAllocator::retire_internode(self, ptr.cast(), guard) }
    }

    #[inline(always)]
    fn teardown_tree(&self, root_ptr: *mut u8) {
        NodeAllocator::teardown_tree(self, root_ptr);
    }

    #[inline(always)]
    unsafe fn retire_subtree_root(&self, root_ptr: *mut u8, guard: &LocalGuard<'_>) {
        // SAFETY: Caller guarantees subtree is unlinked
        unsafe { NodeAllocator::retire_subtree_root(self, root_ptr, guard) }
    }
}

// ============================================================================
//  Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::leaf::LeafValue;

    // ========================================================================
    //  SeizeAllocator Tests
    // ========================================================================

    /// Type alias for seize-based allocator.
    type SeizeArcAllocator<V, const WIDTH: usize> = SeizeAllocator<LeafValue<V>, WIDTH>;

    #[test]
    fn test_seize_allocator_new() {
        let alloc: SeizeArcAllocator<u64, 15> = SeizeAllocator::new();

        assert_eq!(alloc.leaf_count(), 0);
        assert_eq!(alloc.internode_count(), 0);
        assert_eq!(alloc.total_count(), 0);
    }

    #[test]
    fn test_seize_allocator_with_capacity() {
        let alloc: SeizeArcAllocator<u64, 15> = SeizeAllocator::with_capacity(100, 50);

        assert_eq!(alloc.leaf_count(), 0);
        assert!(alloc.leaf_ptrs.lock().capacity() >= 100);
        assert!(alloc.internode_ptrs.lock().capacity() >= 50);
    }

    #[test]
    fn test_seize_alloc_leaf_returns_valid_pointer() {
        let alloc: SeizeArcAllocator<u64, 15> = SeizeAllocator::new();
        let leaf = LeafNode::new();
        let ptr = alloc.alloc_leaf(leaf);

        assert!(!ptr.is_null());
        assert_eq!(alloc.leaf_count(), 1);

        // Pointer should be readable
        unsafe {
            let _ = (*ptr).size();
        }
    }

    #[test]
    fn test_seize_multiple_allocations_distinct_pointers() {
        let alloc: SeizeArcAllocator<u64, 15> = SeizeAllocator::new();

        let ptr1 = alloc.alloc_leaf(LeafNode::new());
        let ptr2 = alloc.alloc_leaf(LeafNode::new());
        let ptr3 = alloc.alloc_leaf(LeafNode::new());

        assert_ne!(ptr1, ptr2);
        assert_ne!(ptr2, ptr3);
        assert_ne!(ptr1, ptr3);
        assert_eq!(alloc.leaf_count(), 3);
    }

    #[test]
    fn test_seize_dealloc_removes_and_frees() {
        let alloc: SeizeArcAllocator<u64, 15> = SeizeAllocator::new();

        let ptr1 = alloc.alloc_leaf(LeafNode::new());
        let ptr2 = alloc.alloc_leaf(LeafNode::new());
        assert_eq!(alloc.leaf_count(), 2);

        // SAFETY: ptr1 was just allocated above
        unsafe { alloc.dealloc_leaf(ptr1) };
        assert_eq!(alloc.leaf_count(), 1);

        // SAFETY: ptr2 was just allocated above
        unsafe { alloc.dealloc_leaf(ptr2) };
        assert_eq!(alloc.leaf_count(), 0);
    }

    #[test]
    fn test_seize_drop_frees_all_nodes() {
        let alloc: SeizeArcAllocator<u64, 15> = SeizeAllocator::new();

        for _ in 0..100 {
            alloc.alloc_leaf(LeafNode::new());
            alloc.alloc_internode(InternodeNode::new(0));
        }

        assert_eq!(alloc.total_count(), 200);

        // Drop should free all nodes without leaking
        drop(alloc);
        // No assertion needed — Miri will catch leaks
    }

    #[test]
    fn test_seize_write_through_pointer_valid() {
        let alloc: SeizeArcAllocator<u64, 15> = SeizeAllocator::new();
        let ptr = alloc.alloc_leaf(LeafNode::new());

        // This is the critical test — writing through the pointer
        // should not cause Stacked Borrows violations
        unsafe {
            (*ptr).set_ikey(0, 0x1234_5678_90AB_CDEF);
            (*ptr).set_keylenx(0, 8);
            assert_eq!((*ptr).ikey(0), 0x1234_5678_90AB_CDEF);
        }
    }

    #[test]
    fn test_seize_allocator_default() {
        let alloc: SeizeArcAllocator<u64, 15> = SeizeAllocator::default();
        assert_eq!(alloc.total_count(), 0);
    }

    #[test]
    fn test_seize_mixed_allocations() {
        let alloc: SeizeArcAllocator<u64, 15> = SeizeAllocator::new();

        // Interleave leaf and internode allocations
        let leaf_ptr = alloc.alloc_leaf(LeafNode::new());
        let inode_ptr = alloc.alloc_internode(InternodeNode::new(0));
        let leaf_ptr2 = alloc.alloc_leaf(LeafNode::new());
        let inode_ptr2 = alloc.alloc_internode(InternodeNode::new(1));

        assert_eq!(alloc.leaf_count(), 2);
        assert_eq!(alloc.internode_count(), 2);
        assert_eq!(alloc.total_count(), 4);

        // All pointers should be valid and distinct
        unsafe {
            let _ = (*leaf_ptr).size();
            let _ = (*leaf_ptr2).size();
            assert_eq!((*inode_ptr).height(), 0);
            assert_eq!((*inode_ptr2).height(), 1);
        }
    }

    #[test]
    fn test_seize_internode_alloc_and_dealloc() {
        let alloc: SeizeArcAllocator<u64, 15> = SeizeAllocator::new();

        let ptr1 = alloc.alloc_internode(InternodeNode::new(0));
        let ptr2 = alloc.alloc_internode(InternodeNode::new(1));
        assert_eq!(alloc.internode_count(), 2);

        // SAFETY: ptr1 was just allocated above
        unsafe { alloc.dealloc_internode(ptr1) };
        assert_eq!(alloc.internode_count(), 1);

        // Verify remaining pointer is still valid
        unsafe {
            assert_eq!((*ptr2).height(), 1);
        }

        // SAFETY: ptr2 was just allocated above
        unsafe { alloc.dealloc_internode(ptr2) };
        assert_eq!(alloc.internode_count(), 0);
    }

    // ========================================================================
    //  Memory Reclamation Tests
    // ========================================================================

    #[test]
    fn test_seize_teardown_tree_clears_tracking() {
        let alloc: SeizeArcAllocator<u64, 15> = SeizeAllocator::new();

        // Allocate a single leaf node
        let ptr = alloc.alloc_leaf(LeafNode::new());
        assert_eq!(alloc.leaf_count(), 1);

        // Teardown should clear tracking and free the node
        alloc.teardown_tree(ptr.cast());

        // Tracking should be cleared
        assert_eq!(alloc.leaf_count(), 0);
        assert_eq!(alloc.internode_count(), 0);
    }

    #[test]
    fn test_seize_teardown_tree_null_is_noop() {
        let alloc: SeizeArcAllocator<u64, 15> = SeizeAllocator::new();

        // Teardown with null should not crash (no nodes allocated)
        alloc.teardown_tree(std::ptr::null_mut());

        // Tracking should remain at zero
        assert_eq!(alloc.leaf_count(), 0);
        assert_eq!(alloc.internode_count(), 0);
    }

    #[test]
    fn test_seize_teardown_then_drop_no_double_free() {
        let alloc: SeizeArcAllocator<u64, 15> = SeizeAllocator::new();

        // Allocate a leaf node
        let ptr = alloc.alloc_leaf(LeafNode::new());
        assert_eq!(alloc.leaf_count(), 1);

        // Teardown should clear tracking and free
        alloc.teardown_tree(ptr.cast());
        assert_eq!(alloc.leaf_count(), 0);

        // Drop should be a no-op (no double-free)
        drop(alloc);
        // Miri will catch any double-free
    }
}
