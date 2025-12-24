//! Generic allocator trait for tree nodes.
//!
//! This module defines [`NodeAllocatorGeneric`] that abstracts over allocators
//! for different leaf node types (`LeafNode<S, WIDTH>` and `LeafNode24<S>`).
//!
//! # Design
//!
//! The trait uses static dispatch (generics) for zero-cost abstraction.
//! Internode pointers use `*mut u8` for type erasure since Rust doesn't
//! support const generics from associated constants in type position.
//!
//! # Implementors
//!
//! - `SeizeAllocator<S, WIDTH>` for `LeafNode<S, WIDTH>`
//! - `SeizeAllocator24<S>` for `LeafNode24<S>`

use seize::LocalGuard;

use crate::leaf_trait::TreeLeafNode;
use crate::slot::ValueSlot;

/// Trait for allocating and deallocating tree nodes generically.
///
/// Abstracts over `SeizeAllocator<S, WIDTH>` and `SeizeAllocator24<S>`, enabling
/// tree operations to work with any leaf type implementing [`TreeLeafNode`].
///
/// # Type Parameters
///
/// - `S`: The slot type implementing [`ValueSlot`]
/// - `L`: The leaf node type implementing [`TreeLeafNode<S>`]
///
/// # Internode Handling
///
/// Internode pointers use `*mut u8` for type erasure. Implementations must ensure
/// internodes have the same WIDTH as leaves (invariant enforced by construction).
///
/// # Thread Safety
///
/// Implementations must be `Send + Sync` to support concurrent tree operations.
///
/// # Implementors
///
/// - `SeizeAllocator<S, WIDTH>` for `L = LeafNode<S, WIDTH>`
/// - `SeizeAllocator24<S>` for `L = LeafNode24<S>`
pub trait NodeAllocatorGeneric<S: ValueSlot, L: TreeLeafNode<S>>: Send + Sync {
    // ========================================================================
    // Leaf Allocation
    // ========================================================================

    /// Allocate a leaf node and return a stable raw pointer.
    ///
    /// The returned pointer is valid until explicitly retired or the allocator drops.
    ///
    /// # Arguments
    ///
    /// * `node` - The leaf node to allocate (takes ownership)
    ///
    /// # Returns
    ///
    /// A raw mutable pointer to the allocated node with valid provenance.
    fn alloc_leaf(&mut self, node: Box<L>) -> *mut L;

    /// Track a leaf pointer for cleanup (concurrent-safe via `&self`).
    ///
    /// Used by concurrent code paths that allocate via `Box::into_raw()`.
    /// For allocators without interior mutability, this is a no-op.
    ///
    /// # Arguments
    ///
    /// * `ptr` - Raw pointer to a leaf node allocated via `Box::into_raw`
    fn track_leaf(&self, ptr: *mut L);

    /// Retire a leaf node for deferred reclamation.
    ///
    /// Schedules the node for deferred reclamation via seize. The node will
    /// be freed once no readers can hold references to it.
    ///
    /// # Safety
    ///
    /// - `ptr` must have been allocated by this allocator
    /// - `ptr` must be unreachable from the tree by any new traversal
    /// - In-flight traversals must detect deletion/retry via the OCC protocol
    unsafe fn retire_leaf(&self, ptr: *mut L, guard: &LocalGuard<'_>);

    // ========================================================================
    // Internode Allocation (type-erased)
    // ========================================================================

    /// Allocate an internode and return a type-erased pointer.
    ///
    /// The concrete type is `InternodeNode<S, L::WIDTH>` but represented as
    /// `*mut u8` to avoid const generic limitations.
    ///
    /// # Arguments
    ///
    /// * `node_ptr` - A `Box<InternodeNode<S, WIDTH>>` cast to `*mut u8` via `Box::into_raw().cast()`
    ///
    /// # Returns
    ///
    /// A type-erased pointer to the allocated internode.
    ///
    /// # Safety Note
    ///
    /// - The caller must pass a valid `Box::into_raw().cast()` pointer
    /// - The caller must cast the result back to the correct internode type
    fn alloc_internode_erased(&mut self, node_ptr: *mut u8) -> *mut u8;

    /// Track an internode pointer for cleanup (concurrent-safe).
    ///
    /// # Arguments
    ///
    /// * `ptr` - Raw pointer to an internode allocated via `Box::into_raw`
    fn track_internode_erased(&self, ptr: *mut u8);

    /// Retire an internode for deferred reclamation.
    ///
    /// # Safety
    ///
    /// - `ptr` must point to a valid `InternodeNode<S, L::WIDTH>`
    /// - `ptr` must be unreachable from the tree by any new traversal
    unsafe fn retire_internode_erased(&self, ptr: *mut u8, guard: &LocalGuard<'_>);

    // ========================================================================
    // Tree Lifecycle
    // ========================================================================

    /// Teardown all reachable nodes at tree drop.
    ///
    /// Called when the tree is destroyed and no concurrent access is possible.
    /// This traverses and frees all nodes reachable from the root.
    ///
    /// Arena allocators can no-op (they free via their own Drop).
    /// Seize-based allocators must implement this to free nodes by traversal.
    ///
    /// # Arguments
    ///
    /// * `root_ptr` - Pointer to the tree root (leaf or internode)
    fn teardown_tree(&mut self, root_ptr: *mut u8);

    /// Retire an entire subtree rooted at `root_ptr`.
    ///
    /// Typically used for reclaiming a whole layer when a layer pointer is removed.
    /// The subtree will be traversed and all nodes freed once safe.
    ///
    /// # Safety
    ///
    /// - The subtree must be fully unlinked from the main tree
    /// - `root_ptr` must point to a valid leaf or internode
    /// - No other shared pointers may reference nodes exclusively through this subtree
    unsafe fn retire_subtree_root(&self, root_ptr: *mut u8, guard: &LocalGuard<'_>);
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::leaf::{LeafNode, LeafValue};
    use crate::leaf24::LeafNode24;

    // ========================================================================
    // Generic Test Helpers
    // ========================================================================

    /// Test that we can allocate a leaf via the generic trait.
    fn test_generic_alloc_leaf<S, L, A>(alloc: &mut A)
    where
        S: ValueSlot + Send + Sync + 'static,
        L: TreeLeafNode<S>,
        A: NodeAllocatorGeneric<S, L>,
    {
        let leaf = L::new_boxed();
        let ptr = alloc.alloc_leaf(leaf);
        assert!(!ptr.is_null());

        // Verify leaf is accessible
        unsafe {
            let leaf_ref = &*ptr;
            assert!(leaf_ref.is_empty());
        }
    }

    /// Test that tracking a leaf works via the generic trait.
    fn test_generic_track_leaf<S, L, A>(alloc: &A)
    where
        S: ValueSlot + Send + Sync + 'static,
        L: TreeLeafNode<S>,
        A: NodeAllocatorGeneric<S, L>,
    {
        let leaf = L::new_boxed();
        let ptr = Box::into_raw(leaf);
        alloc.track_leaf(ptr);
        // Just verify it doesn't panic - actual cleanup happens at drop
    }

    // ========================================================================
    // SeizeAllocator Tests
    // ========================================================================

    #[test]
    fn test_seize_allocator_generic_alloc() {
        use crate::alloc::SeizeAllocator;

        let mut alloc: SeizeAllocator<LeafValue<u64>, 15> = SeizeAllocator::new();
        test_generic_alloc_leaf::<LeafValue<u64>, LeafNode<LeafValue<u64>, 15>, _>(&mut alloc);
    }

    #[test]
    fn test_seize_allocator_generic_track() {
        use crate::alloc::SeizeAllocator;

        let alloc: SeizeAllocator<LeafValue<u64>, 15> = SeizeAllocator::new();
        test_generic_track_leaf::<LeafValue<u64>, LeafNode<LeafValue<u64>, 15>, _>(&alloc);
    }

    // ========================================================================
    // SeizeAllocator24 Tests
    // ========================================================================

    #[test]
    fn test_seize_allocator24_generic_alloc() {
        use crate::alloc24::SeizeAllocator24;

        let mut alloc: SeizeAllocator24<LeafValue<u64>> = SeizeAllocator24::new();
        test_generic_alloc_leaf::<LeafValue<u64>, LeafNode24<LeafValue<u64>>, _>(&mut alloc);
    }

    #[test]
    fn test_seize_allocator24_generic_track() {
        use crate::alloc24::SeizeAllocator24;

        let alloc: SeizeAllocator24<LeafValue<u64>> = SeizeAllocator24::new();
        test_generic_track_leaf::<LeafValue<u64>, LeafNode24<LeafValue<u64>>, _>(&alloc);
    }

    // ========================================================================
    // Trait Object (dyn) Not Required Tests
    // ========================================================================

    /// Verify that the trait enables fully generic code.
    fn generic_tree_setup<S, L, A>(_alloc: &mut A) -> bool
    where
        S: ValueSlot + Send + Sync + 'static,
        L: TreeLeafNode<S>,
        A: NodeAllocatorGeneric<S, L>,
    {
        // This compiles, proving generic code can use the trait
        true
    }

    #[test]
    fn test_generic_code_compiles() {
        use crate::alloc::SeizeAllocator;
        use crate::alloc24::SeizeAllocator24;

        let mut alloc15: SeizeAllocator<LeafValue<u64>, 15> = SeizeAllocator::new();
        let mut alloc24: SeizeAllocator24<LeafValue<u64>> = SeizeAllocator24::new();

        assert!(generic_tree_setup::<
            LeafValue<u64>,
            LeafNode<LeafValue<u64>, 15>,
            _,
        >(&mut alloc15));
        assert!(generic_tree_setup::<LeafValue<u64>, LeafNode24<LeafValue<u64>>, _>(
            &mut alloc24
        ));
    }
}
