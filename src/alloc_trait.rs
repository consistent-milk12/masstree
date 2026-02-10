//! Generic allocator trait for tree nodes.
//!
//! This module defines [`TreeAllocator`] that abstracts over allocators
//! for different leaf policies.
//!
//! # Design
//!
//! The trait uses static dispatch (generics) for zero-cost abstraction.
//! Internode pointers use `*mut u8` for type erasure since Rust doesn't
//! support const generics from associated constants in type position.
//!
//! # Implementors
//!
//! - [`SeizeAllocator<P>`](crate::alloc15::SeizeAllocator) for `LeafNode15<P>`

use seize::LocalGuard;

use crate::internode::InternodeNode;
use crate::leaf_trait::TreeInternode;
use crate::leaf15::LeafNode15;
use crate::nodeversion::NodeVersion;
use crate::policy::LeafPolicy;

/// Trait for allocating and deallocating tree nodes generically.
///
/// Enables tree operations to work with any leaf policy implementing [`LeafPolicy`].
///
/// # Type Parameters
///
/// - `P`: The leaf policy implementing [`LeafPolicy`]
///
/// The leaf type is always `LeafNode15<P>` — no separate leaf type parameter needed.
///
/// # Internode Handling
///
/// Internode pointers use `*mut u8` for type erasure. Implementations must ensure
/// internodes have the same WIDTH as leaves (invariant enforced by construction).
///
/// # Thread Safety
///
/// Implementations must be `Send + Sync` to support concurrent tree operations.
pub trait TreeAllocator<P: LeafPolicy>: Send + Sync {
    // ========================================================================
    // Leaf Allocation
    // ========================================================================

    /// Allocate a leaf node and return a stable raw pointer.
    fn alloc_leaf(&self, node: Box<LeafNode15<P>>) -> *mut LeafNode15<P>;

    /// Allocate a leaf node directly without going through Box.
    #[inline]
    fn alloc_leaf_direct(&self, is_root: bool, is_layer_root: bool) -> *mut LeafNode15<P> {
        let node = if is_layer_root {
            LeafNode15::<P>::new_layer_root_boxed()
        } else if is_root {
            LeafNode15::<P>::new_root_boxed()
        } else {
            LeafNode15::<P>::new_boxed()
        };
        self.alloc_leaf(node)
    }

    /// Track a leaf pointer for cleanup (no-op for traversal-based allocators).
    fn track_leaf(&self, ptr: *mut LeafNode15<P>);

    /// Retire a leaf node for deferred reclamation.
    ///
    /// # Safety
    ///
    /// - `ptr` must point to a valid leaf allocated by this allocator
    /// - `ptr` must be unreachable from the tree (unlinked)
    unsafe fn retire_leaf(&self, ptr: *mut LeafNode15<P>, guard: &LocalGuard<'_>);

    // ========================================================================
    // Internode Allocation (unchanged — type-erased)
    // ========================================================================

    /// Allocate an internode and return a type-erased pointer.
    fn alloc_internode_erased(&self, node_ptr: *mut u8) -> *mut u8;

    /// Allocate an internode directly without going through Box.
    #[inline]
    fn alloc_internode_direct(&self, height: u32) -> *mut u8 {
        let node: Box<InternodeNode> = InternodeNode::new_boxed(height);
        self.alloc_internode_erased(Box::into_raw(node).cast())
    }

    /// Allocate an internode as a root node directly without Box.
    #[inline]
    fn alloc_internode_direct_root(&self, height: u32) -> *mut u8 {
        let node: Box<InternodeNode> = InternodeNode::new_boxed(height);
        node.version().mark_root();
        self.alloc_internode_erased(Box::into_raw(node).cast())
    }

    /// Allocate an internode for a split operation directly without Box.
    #[inline]
    fn alloc_internode_direct_for_split(
        &self,
        parent_version: &NodeVersion,
        height: u32,
    ) -> *mut u8 {
        let node: Box<InternodeNode> = InternodeNode::new_boxed_for_split(parent_version, height);
        self.alloc_internode_erased(Box::into_raw(node).cast())
    }

    /// Track an internode pointer for cleanup (no-op for traversal-based allocators).
    fn track_internode_erased(&self, ptr: *mut u8);

    /// Retire an internode for deferred reclamation.
    ///
    /// # Safety
    ///
    /// - `ptr` must point to a valid internode
    /// - `ptr` must be unreachable from the tree
    unsafe fn retire_internode_erased(&self, ptr: *mut u8, guard: &LocalGuard<'_>);

    // ========================================================================
    // Tree Lifecycle
    // ========================================================================

    /// Teardown all reachable nodes at tree drop.
    fn teardown_tree(&self, root_ptr: *mut u8);

    /// Retire an entire subtree.
    ///
    /// # Safety
    ///
    /// - The subtree must be fully unlinked from the main tree
    unsafe fn retire_subtree_root(&self, root_ptr: *mut u8, guard: &LocalGuard<'_>);
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod unit_tests;
