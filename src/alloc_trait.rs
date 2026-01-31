//! Generic allocator trait for tree nodes.
//!
//! This module defines [`NodeAllocatorGeneric`] that abstracts over allocators
//! for different leaf node types.
//!
//! # Design
//!
//! The trait uses static dispatch (generics) for zero-cost abstraction.
//! Internode pointers use `*mut u8` for type erasure since Rust doesn't
//! support const generics from associated constants in type position.
//!
//! # Implementors
//!
//! - [`SeizeAllocator15<S>`](crate::alloc15::SeizeAllocator15) for `LeafNode15<S>`

use seize::LocalGuard;

use crate::leaf_trait::{TreeInternode, TreeLeafNode};
use crate::nodeversion::NodeVersion;
use crate::slot::ValueSlot;

/// Trait for allocating and deallocating tree nodes generically.
///
/// Enables tree operations to work with any leaf type implementing [`TreeLeafNode`].
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
/// - [`SeizeAllocator15<S>`](crate::alloc15::SeizeAllocator15) for `L = LeafNode15<S>`
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
    ///
    /// # Note
    ///
    /// Uses interior mutability (`parking_lot::Mutex`) so this can be called
    /// from concurrent code paths with only `&self`.
    fn alloc_leaf(&self, node: Box<L>) -> *mut L;

    /// Allocate a leaf node directly without going through Box.
    ///
    /// This is an optimization for pool allocators that can write directly
    /// to pool memory, avoiding the intermediate Box allocation.
    ///
    /// # Arguments
    ///
    /// * `is_root` - Whether this is a tree root node
    /// * `is_layer_root` - Whether this is a layer root node
    ///
    /// # Returns
    ///
    /// A raw mutable pointer to the initialized node.
    ///
    /// # Default Implementation
    ///
    /// Falls back to creating a Box and calling `alloc_leaf`.
    #[inline]
    fn alloc_leaf_direct(&self, is_root: bool, is_layer_root: bool) -> *mut L {
        let node = if is_layer_root {
            L::new_layer_root_boxed()
        } else if is_root {
            L::new_root_boxed()
        } else {
            L::new_boxed()
        };
        self.alloc_leaf(node)
    }

    /// Track a leaf pointer for cleanup (no-op for traversal-based allocators).
    ///
    /// With tree traversal teardown, tracking is unnecessary - nodes are freed
    /// by walking the tree structure. This method exists for API compatibility
    /// but is typically a no-op.
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
    /// This is O(1) - direct `guard.defer_retire()` call with no tracking overhead.
    ///
    /// # Safety
    ///
    /// - `ptr` must point to a valid leaf allocated by this allocator
    /// - `ptr` must be unreachable from the tree (unlinked)
    /// - `ptr` must not be retired multiple times
    unsafe fn retire_leaf(&self, ptr: *mut L, guard: &LocalGuard<'_>);

    // ========================================================================
    // Internode Allocation (type-erased)
    // ========================================================================

    /// Allocate an internode and return a type-erased pointer.
    ///
    /// The concrete type is `InternodeNode` but represented as
    /// `*mut u8` for uniformity with leaf node pointers.
    ///
    /// # Arguments
    ///
    /// * `node_ptr` - A `Box<InternodeNode>` cast to `*mut u8` via `Box::into_raw().cast()`
    ///
    /// # Returns
    ///
    /// A type-erased pointer to the allocated internode.
    ///
    /// # Safety Note
    ///
    /// - The caller must pass a valid `Box::into_raw().cast()` pointer
    /// - The caller must cast the result back to the correct internode type
    ///
    /// # Note
    ///
    /// Uses interior mutability (`parking_lot::Mutex`) so this can be called
    /// from concurrent code paths with only `&self`.
    fn alloc_internode_erased(&self, node_ptr: *mut u8) -> *mut u8;

    /// Allocate an internode directly without going through Box.
    ///
    /// This is an optimization for pool allocators that can write directly
    /// to pool memory, avoiding the intermediate Box allocation.
    ///
    /// # Arguments
    ///
    /// * `height` - Tree height for the internode (0 = children are leaves)
    ///
    /// # Returns
    ///
    /// A type-erased pointer to the initialized internode.
    ///
    /// # Default Implementation
    ///
    /// Falls back to creating a Box and calling `alloc_internode_erased`.
    #[inline]
    fn alloc_internode_direct(&self, height: u32) -> *mut u8 {
        let node: Box<L::Internode> = L::Internode::new_boxed(height);
        self.alloc_internode_erased(Box::into_raw(node).cast())
    }

    /// Allocate an internode as a root node directly without Box.
    ///
    /// # Arguments
    ///
    /// * `height` - Tree height for the internode
    ///
    /// # Default Implementation
    ///
    /// Falls back to creating a Box and calling `alloc_internode_erased`.
    #[inline]
    fn alloc_internode_direct_root(&self, height: u32) -> *mut u8 {
        let node: Box<L::Internode> = L::Internode::new_boxed(height);
        node.version().mark_root();
        self.alloc_internode_erased(Box::into_raw(node).cast())
    }

    /// Allocate an internode for a split operation directly without Box.
    ///
    /// Creates an internode with a split-locked version copied from the parent.
    ///
    /// # Arguments
    ///
    /// * `parent_version` - Version from the parent being split (must be locked)
    /// * `height` - Tree height for the internode
    ///
    /// # Default Implementation
    ///
    /// Falls back to creating a Box and calling `alloc_internode_erased`.
    #[inline]
    fn alloc_internode_direct_for_split(
        &self,
        parent_version: &NodeVersion,
        height: u32,
    ) -> *mut u8 {
        let node: Box<L::Internode> = L::Internode::new_boxed_for_split(parent_version, height);
        self.alloc_internode_erased(Box::into_raw(node).cast())
    }

    /// Track an internode pointer for cleanup (no-op for traversal-based allocators).
    ///
    /// With tree traversal teardown, tracking is unnecessary - nodes are freed
    /// by walking the tree structure. This method exists for API compatibility
    /// but is typically a no-op.
    ///
    /// # Arguments
    ///
    /// * `ptr` - Raw pointer to an internode allocated via `Box::into_raw`
    fn track_internode_erased(&self, ptr: *mut u8);

    /// Retire an internode for deferred reclamation.
    ///
    /// The internode will be freed once all guards that might reference it are dropped.
    /// This is O(1) - direct `guard.defer_retire()` call with no tracking overhead.
    ///
    /// # Safety
    ///
    /// - `ptr` must point to a valid internode allocated by this allocator
    /// - `ptr` must be unreachable from the tree (unlinked)
    /// - `ptr` must not be retired multiple times
    unsafe fn retire_internode_erased(&self, ptr: *mut u8, guard: &LocalGuard<'_>);

    // ========================================================================
    // Tree Lifecycle
    // ========================================================================

    /// Teardown all reachable nodes at tree drop.
    ///
    /// Called when the tree is destroyed and no concurrent access is possible.
    /// This traverses and frees all nodes reachable from the root, including
    /// sublayer trees reached via layer pointers.
    ///
    /// # Traversal
    ///
    /// - Uses `NodeVersion::is_leaf()` at offset 0 to distinguish node types
    /// - For leaves: recurses into layer pointers (where `is_layer(slot)` is true)
    /// - For internodes: recurses into all child pointers
    /// - Properly handles true-inline leaves (checks `is_layer()` before interpreting pointers)
    ///
    /// # Arguments
    ///
    /// * `root_ptr` - Pointer to the tree root (leaf or internode)
    fn teardown_tree(&self, root_ptr: *mut u8);

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
mod unit_tests;
