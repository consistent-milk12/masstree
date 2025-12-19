//! Node allocation abstraction for `MassTree`.
//!
//! This module provides the [`NodeAllocator`] trait that abstracts how nodes
//! are allocated and (eventually) deallocated. Currently uses [`ArenaAllocator`]
//! which keeps nodes alive until the tree is dropped. Phase 3.4 will add
//! `SeizeAllocator` using hyaline reclamation (`seize` crate) for concurrent access.

use std::ptr as StdPtr;

use crate::internode::InternodeNode;
use crate::leaf::LeafNode;
use crate::slot::ValueSlot;

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
    fn alloc_leaf(&mut self, node: Box<LeafNode<S, WIDTH>>) -> *mut LeafNode<S, WIDTH>;

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
    fn alloc_internode(
        &mut self,
        node: Box<InternodeNode<S, WIDTH>>,
    ) -> *mut InternodeNode<S, WIDTH>;

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
    #[allow(unused_variables)]
    fn dealloc_leaf(&mut self, ptr: *mut LeafNode<S, WIDTH>) {
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
    #[allow(unused_variables)]
    fn dealloc_internode(&mut self, ptr: *mut InternodeNode<S, WIDTH>) {
        // Default: no-op for arena-style allocators
    }
}

/// Arena-based node allocator for single-threaded use.
///
/// Nodes are stored in `Vec<Box<T>>` arenas. The `Box` provides a stable
/// heap address for the node contents, while the `Vec` tracks ownership.
/// All nodes are freed when the allocator (and thus the tree) is dropped.
///
/// # Pointer Stability
///
/// `Vec<Box<T>>` provides stable pointers because:
/// - `Box<T>` allocates `T` on the heap at a fixed address
/// - When `Vec` reallocates, only the `Box` pointers (8 bytes) move
/// - The heap-allocated node contents never move
///
/// # Type Parameters
///
/// * `S` - The slot type implementing [`ValueSlot`]
/// * `WIDTH` - The node width (number of slots)
///
/// # Example
///
/// ```ignore
/// use masstree::alloc::ArenaAllocator;
/// use masstree::leaf::LeafValue;
/// use masstree::MassTree;
///
/// // Explicit allocator type (usually inferred)
/// let tree: MassTree<u64, 15, ArenaAllocator<LeafValue<u64>, 15>> = MassTree::new();
/// ```
#[derive(Debug)]
pub struct ArenaAllocator<S: ValueSlot, const WIDTH: usize> {
    /// Arena for leaf nodes.
    leaf_arena: Vec<Box<LeafNode<S, WIDTH>>>,

    /// Arena for internode nodes.
    internode_arena: Vec<Box<InternodeNode<S, WIDTH>>>,
}

impl<S: ValueSlot, const WIDTH: usize> ArenaAllocator<S, WIDTH> {
    /// Create a new arena allocator with default capacity.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            leaf_arena: Vec::new(),
            internode_arena: Vec::new(),
        }
    }

    /// Create a new arena allocator with pre-allocated capacity.
    ///
    /// # Arguments
    ///
    /// * `leaf_capacity` - Initial capacity for leaf arena
    /// * `internode_capacity` - Initial capacity for internode arena
    #[must_use]
    pub fn with_capacity(leaf_capacity: usize, internode_capacity: usize) -> Self {
        Self {
            leaf_arena: Vec::with_capacity(leaf_capacity),
            internode_arena: Vec::with_capacity(internode_capacity),
        }
    }

    /// Return the number of leaf nodes in the arena.
    #[inline]
    #[must_use]
    pub const fn leaf_count(&self) -> usize {
        self.leaf_arena.len()
    }

    /// Return the number of internodes in the arena.
    #[inline]
    #[must_use]
    pub const fn internode_count(&self) -> usize {
        self.internode_arena.len()
    }

    /// Return the total number of nodes in the arena.
    #[inline]
    #[must_use]
    pub const fn total_count(&self) -> usize {
        self.leaf_arena.len() + self.internode_arena.len()
    }
}

impl<S: ValueSlot, const WIDTH: usize> Default for ArenaAllocator<S, WIDTH> {
    fn default() -> Self {
        Self::new()
    }
}

impl<S: ValueSlot, const WIDTH: usize> NodeAllocator<S, WIDTH> for ArenaAllocator<S, WIDTH> {
    fn alloc_leaf(&mut self, leaf: Box<LeafNode<S, WIDTH>>) -> *mut LeafNode<S, WIDTH> {
        self.leaf_arena.push(leaf);
        let idx: usize = self.leaf_arena.len() - 1;

        // SAFETY: We just pushed, so idx is valid. We derive the pointer AFTER
        // storing to maintain Stacked Borrows provenance. The Box provides a
        // stable heap address that won't move even if the Vec reallocates.
        #[allow(clippy::indexing_slicing)]
        unsafe {
            StdPtr::from_mut::<LeafNode<S, WIDTH>>(self.leaf_arena.get_unchecked_mut(idx).as_mut())
        }
    }

    fn alloc_internode(
        &mut self,
        node: Box<InternodeNode<S, WIDTH>>,
    ) -> *mut InternodeNode<S, WIDTH> {
        self.internode_arena.push(node);
        let idx: usize = self.internode_arena.len() - 1;

        // SAFETY: Same reasoning as alloc_leaf.
        #[allow(clippy::indexing_slicing)]
        unsafe {
            StdPtr::from_mut::<InternodeNode<S, WIDTH>>(
                self.internode_arena.get_unchecked_mut(idx).as_mut(),
            )
        }
    }

    // dealloc_leaf and dealloc_internode use default no-op implementations
}

// ============================================================================
//  Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::leaf::LeafValue;

    /// Type alias for Arc-based allocator (default mode).
    type ArcAllocator<V, const WIDTH: usize> = ArenaAllocator<LeafValue<V>, WIDTH>;

    #[test]
    fn test_arena_allocator_new() {
        let alloc: ArcAllocator<u64, 15> = ArenaAllocator::new();

        assert_eq!(alloc.leaf_count(), 0);
        assert_eq!(alloc.internode_count(), 0);
        assert_eq!(alloc.total_count(), 0);
    }

    #[test]
    fn test_arena_allocator_with_capacity() {
        let alloc: ArcAllocator<u64, 15> = ArenaAllocator::with_capacity(100, 50);

        assert_eq!(alloc.leaf_count(), 0);
        assert!(alloc.leaf_arena.capacity() >= 100);
        assert!(alloc.internode_arena.capacity() >= 50);
    }

    #[test]
    fn test_alloc_leaf_returns_stable_pointer() {
        let mut alloc: ArcAllocator<u64, 15> = ArenaAllocator::new();

        // Allocate several leaves
        let leaf1 = LeafNode::new();
        let ptr1 = alloc.alloc_leaf(leaf1);

        let leaf2 = LeafNode::new();
        let ptr2 = alloc.alloc_leaf(leaf2);

        let leaf3 = LeafNode::new();
        let ptr3 = alloc.alloc_leaf(leaf3);

        // All pointers should be distinct
        assert_ne!(ptr1, ptr2);
        assert_ne!(ptr2, ptr3);
        assert_ne!(ptr1, ptr3);

        // Pointers should remain valid after more allocations
        // (Vec may reallocate, but Box contents don't move)
        for _ in 0..100 {
            let leaf = LeafNode::new();
            let _ = alloc.alloc_leaf(leaf);
        }

        // Original pointers should still be dereferenceable
        // SAFETY: Pointers came from alloc_leaf and allocator hasn't been dropped
        unsafe {
            // Just verify we can read without crashing
            let _ = (*ptr1).size();
            let _ = (*ptr2).size();
            let _ = (*ptr3).size();
        }

        assert_eq!(alloc.leaf_count(), 103);
    }

    #[test]
    fn test_alloc_internode_returns_stable_pointer() {
        let mut alloc: ArcAllocator<u64, 15> = ArenaAllocator::new();

        let node1 = InternodeNode::new(0);
        let ptr1 = alloc.alloc_internode(node1);

        let node2 = InternodeNode::new(1);
        let ptr2 = alloc.alloc_internode(node2);

        assert_ne!(ptr1, ptr2);

        // Verify pointers remain valid
        unsafe {
            assert_eq!((*ptr1).height(), 0);
            assert_eq!((*ptr2).height(), 1);
        }

        assert_eq!(alloc.internode_count(), 2);
    }

    #[test]
    fn test_default_impl() {
        let alloc: ArcAllocator<u64, 15> = ArenaAllocator::default();
        assert_eq!(alloc.total_count(), 0);
    }

    #[test]
    fn test_mixed_allocations() {
        let mut alloc: ArcAllocator<u64, 15> = ArenaAllocator::new();

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
            let _ = (*inode_ptr).height();
            let _ = (*inode_ptr2).height();
        }
    }

    #[test]
    fn test_dealloc_is_noop() {
        let mut alloc: ArcAllocator<u64, 15> = ArenaAllocator::new();

        let ptr = alloc.alloc_leaf(LeafNode::new());
        let count_before = alloc.leaf_count();

        // Dealloc is a no-op for arena allocator
        alloc.dealloc_leaf(ptr);

        // Count unchanged (arena doesn't actually free)
        assert_eq!(alloc.leaf_count(), count_before);
    }
}
