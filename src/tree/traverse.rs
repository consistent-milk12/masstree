//! Tree traversal helpers for reaching leaf nodes.
//!
//! This module provides methods for navigating from the root to leaf nodes,
//! supporting both immutable and mutable traversal patterns.

use crate::alloc::NodeAllocator;
use crate::internode::InternodeNode;
use crate::key::Key;
use crate::ksearch::upper_bound_internode_direct;
use crate::leaf::{LeafNode, LeafValue};

use super::{MassTree, RootNode};

/// Traverse from an internode to the target leaf (iterative).
///
/// Free function to avoid `self_only_used_in_recursion` lint.
fn reach_leaf_from_internode<V, const WIDTH: usize>(
    mut internode: &InternodeNode<LeafValue<V>, WIDTH>,
    ikey: u64,
) -> &LeafNode<LeafValue<V>, WIDTH> {
    loop {
        let child_idx: usize = upper_bound_internode_direct(ikey, internode);
        let child_ptr: *mut u8 = internode.child(child_idx);

        if internode.children_are_leaves() {
            // Child is a leaf
            // SAFETY: children_are_leaves() guarantees child is LeafNode
            return unsafe { &*(child_ptr as *const LeafNode<LeafValue<V>, WIDTH>) };
        }

        // Child is another internode, continue iteration
        // SAFETY: !children_are_leaves() guarantees child is InternodeNode
        internode = unsafe { &*(child_ptr as *const InternodeNode<LeafValue<V>, WIDTH>) };
    }
}

impl<V, const WIDTH: usize, A: NodeAllocator<LeafValue<V>, WIDTH>> MassTree<V, WIDTH, A> {
    /// Find the leftmost leaf node in the tree.
    ///
    /// Traverses down the leftmost path from the given internode to find
    /// the first (leftmost) leaf in the tree.
    #[expect(dead_code, reason = "Will be used for iteration/range queries")]
    pub(super) fn find_leftmost_leaf(root: &InternodeNode<LeafValue<V>, WIDTH>) -> *const LeafNode<LeafValue<V>, WIDTH> {
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
    /// For single-threaded mode, no version checking is needed.
    ///
    /// # Arguments
    ///
    /// * `key` - The key to search for
    ///
    /// # Returns
    ///
    /// Reference to the leaf node that contains or should contain the key.
    #[inline]
    pub(super) fn reach_leaf(&self, key: &Key<'_>) -> &LeafNode<LeafValue<V>, WIDTH> {
        match &self.root {
            RootNode::Leaf(leaf) => leaf.as_ref(),

            RootNode::Internode(internode) => {
                // Traverse internodes to find leaf
                reach_leaf_from_internode(internode.as_ref(), key.ikey())
            }
        }
    }

    /// Reach the leaf node that should contain the given key (mutable).
    #[inline]
    pub(super) fn reach_leaf_mut(&mut self, key: &Key<'_>) -> &mut LeafNode<LeafValue<V>, WIDTH> {
        // Check if root is a leaf first (immutable borrow to check)
        let is_leaf: bool = self.root.is_leaf();

        if is_leaf {
            // Now we can borrow mutably
            match &mut self.root {
                RootNode::Leaf(leaf) => return leaf.as_mut(),
                RootNode::Internode(_) => unreachable!(),
            }
        }

        // Root is an internode - extract info with immutable borrow first
        let (start_ptr, children_are_leaves, ikey) = match &self.root {
            RootNode::Internode(internode) => {
                let ikey: u64 = key.ikey();
                let child_idx: usize = upper_bound_internode_direct(ikey, internode.as_ref());
                (
                    internode.child(child_idx),
                    internode.children_are_leaves(),
                    ikey,
                )
            }

            RootNode::Leaf(_) => unreachable!(),
        };

        if children_are_leaves {
            // SAFETY: children_are_leaves() guarantees child is LeafNode
            return unsafe { &mut *start_ptr.cast::<LeafNode<LeafValue<V>, WIDTH>>() };
        }

        // Iterative traversal for deeper trees
        // SAFETY: The returned pointer is valid for the tree's lifetime (arena-backed)
        unsafe { &mut *Self::reach_leaf_mut_iterative_static(start_ptr, ikey) }
    }

    /// Iterative leaf reach for deeply nested trees.
    ///
    /// # Safety
    ///
    /// The returned reference is valid for as long as the tree's arenas are not modified.
    /// This is guaranteed by the single-threaded Phase 1 design.
    fn reach_leaf_mut_iterative_static(mut current: *mut u8, ikey: u64) -> *mut LeafNode<LeafValue<V>, WIDTH> {
        loop {
            // SAFETY: current is a valid internode pointer from traversal
            let internode: &InternodeNode<LeafValue<V>, WIDTH> =
                unsafe { &*(current as *const InternodeNode<LeafValue<V>, WIDTH>) };
            let child_idx: usize = upper_bound_internode_direct(ikey, internode);
            let child_ptr: *mut u8 = internode.child(child_idx);

            if internode.children_are_leaves() {
                // SAFETY: children_are_leaves() guarantees child is LeafNode
                return child_ptr.cast::<LeafNode<LeafValue<V>, WIDTH>>();
            }

            current = child_ptr;
        }
    }
}
