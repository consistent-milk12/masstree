//! Tree traversal helpers for reaching leaf nodes.
//!
//! This module provides methods for navigating from the root to leaf nodes,
//! supporting both immutable and mutable traversal patterns.

use std::sync::atomic::Ordering;

use crate::alloc::NodeAllocator;
use crate::internode::InternodeNode;
use crate::key::Key;
use crate::ksearch::upper_bound_internode_direct;
use crate::leaf::{LeafNode, LeafValue};
use crate::nodeversion::NodeVersion;
use crate::prefetch::prefetch_read;

use super::MassTree;

impl<V, const WIDTH: usize, A: NodeAllocator<LeafValue<V>, WIDTH>> MassTree<V, WIDTH, A> {
    /// Find the leftmost leaf node in the tree.
    ///
    /// Traverses down the leftmost path from the given internode to find
    /// the first (leftmost) leaf in the tree.
    #[expect(dead_code, reason = "Will be used for iteration/range queries")]
    pub(super) fn find_leftmost_leaf(
        root: &InternodeNode<LeafValue<V>, WIDTH>,
    ) -> *const LeafNode<LeafValue<V>, WIDTH> {
        let mut node: *const u8 = root.child(0);
        let mut height: u32 = root.height();

        // Traverse down the leftmost path
        while height > 0 {
            // SAFETY: Child pointers are valid from arena allocation.
            // height > 0 means children are internodes.
            let internode: &InternodeNode<LeafValue<V>, WIDTH> =
                unsafe { &*node.cast::<InternodeNode<LeafValue<V>, WIDTH>>() };

            node = internode.child(0);
            height -= 1;
        }

        // height == 0 means node points to a leaf
        node.cast::<LeafNode<LeafValue<V>, WIDTH>>()
    }

    /// Reach the leaf node that should contain the given key.
    ///
    /// Traverses from root through internodes to find the target leaf.
    /// Uses `root_ptr` as the single source of truth (no `RootNode` enum).
    ///
    /// # Arguments
    ///
    /// * `key` - The key to search for
    ///
    /// # Returns
    ///
    /// Reference to the leaf node that contains or should contain the key.
    #[allow(dead_code)]
    #[inline(always)]
    pub(super) fn reach_leaf(&self, key: &Key<'_>) -> &LeafNode<LeafValue<V>, WIDTH> {
        let root: *const u8 = self.root_ptr.load(Ordering::Acquire);

        // SAFETY: root_ptr always points to a valid node.
        // NodeVersion is the first field of both LeafNode and InternodeNode.
        #[expect(
            clippy::cast_ptr_alignment,
            reason = "root points to LeafNode or InternodeNode, both properly aligned"
        )]
        let version_ptr: *const NodeVersion = root.cast::<NodeVersion>();
        let is_leaf: bool = unsafe { (*version_ptr).is_leaf() };

        if is_leaf {
            // SAFETY: is_leaf() confirmed this is a LeafNode
            unsafe { &*(root.cast::<LeafNode<LeafValue<V>, WIDTH>>()) }
        } else {
            // SAFETY: !is_leaf() confirmed this is an InternodeNode
            let internode: &InternodeNode<LeafValue<V>, WIDTH> =
                unsafe { &*(root.cast::<InternodeNode<LeafValue<V>, WIDTH>>()) };

            self.reach_leaf_via_internode(internode, key)
        }
    }

    /// Reach the leaf node that should contain the given key (mutable).
    ///
    /// Uses `root_ptr` as the single source of truth (no `RootNode` enum).
    #[inline(always)]
    #[expect(
        clippy::needless_pass_by_ref_mut,
        reason = "Returns &mut LeafNode which requires &mut self for lifetime"
    )]
    pub(super) fn reach_leaf_mut(&mut self, key: &Key<'_>) -> &mut LeafNode<LeafValue<V>, WIDTH> {
        let root: *mut u8 = self.root_ptr.load(Ordering::Acquire);

        // SAFETY: root_ptr always points to a valid node.
        // NodeVersion is the first field of both LeafNode and InternodeNode.
        #[expect(
            clippy::cast_ptr_alignment,
            reason = "root points to LeafNode or InternodeNode, both properly aligned"
        )]
        let version_ptr: *const NodeVersion = root.cast::<NodeVersion>();
        let is_leaf: bool = unsafe { (*version_ptr).is_leaf() };

        if is_leaf {
            // SAFETY: is_leaf() confirmed this is a LeafNode
            unsafe { &mut *(root.cast::<LeafNode<LeafValue<V>, WIDTH>>()) }
        } else {
            // SAFETY: !is_leaf() confirmed this is an InternodeNode
            let internode: &InternodeNode<LeafValue<V>, WIDTH> =
                unsafe { &*(root.cast::<InternodeNode<LeafValue<V>, WIDTH>>()) };

            let ikey: u64 = key.ikey();
            let child_idx: usize = upper_bound_internode_direct(ikey, internode);
            let start_ptr: *mut u8 = internode.child(child_idx);

            // Prefetch child node while checking if it's a leaf
            prefetch_read(start_ptr);

            let children_are_leaves: bool = internode.children_are_leaves();

            if children_are_leaves {
                // SAFETY: children_are_leaves() guarantees child is LeafNode
                unsafe { &mut *start_ptr.cast::<LeafNode<LeafValue<V>, WIDTH>>() }
            } else {
                // Iterative traversal for deeper trees
                // SAFETY: The returned pointer is valid for the tree's lifetime (arena-backed)
                unsafe { &mut *Self::reach_leaf_mut_iterative_static(start_ptr, ikey) }
            }
        }
    }

    /// Iterative leaf reach for deeply nested trees.
    ///
    /// # Safety
    ///
    /// The returned reference is valid for as long as the tree's arenas are not modified.
    /// Tree operations are single-threaded (Phase 3.2-3.3 will add concurrency).
    #[inline(always)]
    fn reach_leaf_mut_iterative_static(
        mut current: *mut u8,
        ikey: u64,
    ) -> *mut LeafNode<LeafValue<V>, WIDTH> {
        loop {
            // SAFETY: current is a valid internode pointer from traversal
            let internode: &InternodeNode<LeafValue<V>, WIDTH> =
                unsafe { &*(current as *const InternodeNode<LeafValue<V>, WIDTH>) };
            let child_idx: usize = upper_bound_internode_direct(ikey, internode);
            let child_ptr: *mut u8 = internode.child(child_idx);

            // Prefetch child node while checking if it's a leaf
            prefetch_read(child_ptr);

            if internode.children_are_leaves() {
                // SAFETY: children_are_leaves() guarantees child is LeafNode
                return child_ptr.cast::<LeafNode<LeafValue<V>, WIDTH>>();
            }

            current = child_ptr;
        }
    }
}
