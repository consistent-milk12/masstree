//! Leaf iteration for tree-level diagnostics and validation.
//!
//! The `LeafIterator` traverses all leaves in a `MassTree` (including sublayers)
//! in a single pass. This is useful for:
//!
//! - Validating tree invariants across all leaves
//! - Collecting statistics (orphaned slots, size distribution, etc.)
//! - Debugging and analysis
//!
//! # Quiescence Requirements
//!
//! This iterator should only be used when the tree is quiescent:
//! - No concurrent insertions, splits, or deletions in progress
//! - Typically used in test teardown or debugging scenarios
//!
//! Using this iterator on an actively modified tree may yield inconsistent results.

#![allow(dead_code)] // Diagnostic tool may not be used in all builds

use std::collections::VecDeque;

use crate::internode::InternodeNode;
use crate::leaf::{LAYER_KEYLENX, LeafNode, LeafValue};
use crate::nodeversion::NodeVersion;

/// Iterator over all leaves in a `MassTree` (including sublayers).
///
/// Yields raw pointers to leaf nodes. The caller is responsible for ensuring
/// the tree is quiescent and the pointers remain valid during iteration.
pub struct LeafIterator<V, const WIDTH: usize> {
    /// Stack of nodes to visit (type-erased pointers)
    stack: VecDeque<*const u8>,
    /// Marker for value type
    _marker: std::marker::PhantomData<V>,
}

impl<V, const WIDTH: usize> LeafIterator<V, WIDTH> {
    /// Create a new leaf iterator starting from the tree root.
    ///
    /// # Safety
    ///
    /// - `root_ptr` must be a valid pointer to a leaf or internode, or null
    /// - The tree must not be modified during iteration
    pub unsafe fn new(root_ptr: *const u8) -> Self {
        let mut stack = VecDeque::new();
        if !root_ptr.is_null() {
            // Canonicalize through maybe_parent pattern
            // SAFETY: root_ptr is valid per precondition
            let canonical = unsafe { Self::canonicalize_root(root_ptr) };
            stack.push_back(canonical);
        }
        Self {
            stack,
            _marker: std::marker::PhantomData,
        }
    }

    /// Follow parent pointers to find the canonical layer root.
    ///
    /// This handles the `maybe_parent` pattern where layer pointers
    /// may point to stale leaf nodes that have been promoted.
    ///
    /// # Safety
    ///
    /// - `node` must be a valid pointer to a leaf or internode, or null
    unsafe fn canonicalize_root(mut node: *const u8) -> *const u8 {
        // Bound iterations to prevent infinite loops on corruption
        for _ in 0..1024 {
            if node.is_null() {
                return node;
            }

            // SAFETY: node is valid per precondition
            #[expect(
                clippy::cast_ptr_alignment,
                reason = "node was allocated as LeafNode/InternodeNode which have NodeVersion as first field"
            )]
            let version: &NodeVersion = unsafe { &*node.cast::<NodeVersion>() };
            let parent: *mut u8 = if version.is_leaf() {
                // SAFETY: version says it's a leaf
                unsafe { (*node.cast::<LeafNode<LeafValue<V>, WIDTH>>()).parent() }
            } else {
                // SAFETY: version says it's an internode
                unsafe { (*node.cast::<InternodeNode<LeafValue<V>, WIDTH>>()).parent() }
            };

            if parent.is_null() {
                return node;
            }

            node = parent;
        }

        node
    }
}

impl<V, const WIDTH: usize> Iterator for LeafIterator<V, WIDTH> {
    type Item = *const LeafNode<LeafValue<V>, WIDTH>;

    fn next(&mut self) -> Option<Self::Item> {
        while let Some(node) = self.stack.pop_front() {
            if node.is_null() {
                continue;
            }

            // SAFETY: Caller guarantees tree is quiescent
            #[expect(
                clippy::cast_ptr_alignment,
                reason = "node was allocated as LeafNode/InternodeNode which have NodeVersion as first field"
            )]
            let version: &NodeVersion = unsafe { &*node.cast::<NodeVersion>() };

            if version.is_leaf() {
                // This is a leaf - process it
                let leaf: &LeafNode<LeafValue<V>, WIDTH> =
                    unsafe { &*node.cast::<LeafNode<LeafValue<V>, WIDTH>>() };

                // Collect sublayer roots from this leaf
                for slot in 0..WIDTH {
                    let keylenx: u8 = leaf.keylenx(slot);
                    if keylenx >= LAYER_KEYLENX {
                        let layer_ptr: *mut u8 = leaf.leaf_value_ptr(slot);
                        if !layer_ptr.is_null() {
                            // Canonicalize through maybe_parent
                            let canonical =
                                unsafe { Self::canonicalize_root(layer_ptr.cast_const()) };
                            self.stack.push_back(canonical);
                        }
                    }
                }

                return Some(node.cast());
            }

            let inode: &InternodeNode<LeafValue<V>, WIDTH> =
                unsafe { &*node.cast::<InternodeNode<LeafValue<V>, WIDTH>>() };
            let nkeys: usize = inode.nkeys();
            for i in 0..=nkeys {
                let child: *mut u8 = inode.child(i);
                if !child.is_null() {
                    self.stack.push_back(child.cast_const());
                }
            }
        }

        None
    }
}

/// Statistics collected from iterating over tree leaves.
#[derive(Debug, Clone, Default)]
pub struct LeafStats {
    /// Total number of leaves visited
    pub leaf_count: usize,
    /// Total number of entries across all leaves
    pub entry_count: usize,
    /// Number of leaves with orphaned slots
    pub leaves_with_orphans: usize,
    /// Total number of orphaned slots
    pub orphan_count: usize,
    /// Number of leaves that were frozen (skip counting)
    pub frozen_leaves: usize,
}

impl LeafStats {
    /// Collect statistics from a tree by iterating all leaves.
    ///
    /// # Safety
    ///
    /// - `root_ptr` must be a valid tree root (leaf or internode), or null
    /// - The tree must be quiescent (no concurrent modifications)
    pub unsafe fn collect<V, const WIDTH: usize>(root_ptr: *const u8) -> Self {
        let mut stats = Self::default();
        // SAFETY: root_ptr validity guaranteed by caller
        let iter: LeafIterator<V, WIDTH> = unsafe { LeafIterator::new(root_ptr) };

        for leaf_ptr in iter {
            stats.leaf_count += 1;

            // SAFETY: leaf_ptr comes from iterator which only yields valid pointers
            let leaf: &LeafNode<LeafValue<V>, WIDTH> = unsafe { &*leaf_ptr };

            // Check if frozen - skip detailed stats
            // Use permutation_try() which returns Err if frozen
            let Ok(perm) = leaf.permutation_try() else {
                stats.frozen_leaves += 1;
                continue;
            };

            stats.entry_count += perm.size();

            // Check for orphans
            if let Some(has_orphans) = leaf.has_orphaned_slots_if_safe()
                && has_orphans
            {
                stats.leaves_with_orphans += 1;
                stats.orphan_count += leaf.find_orphaned_slots().len();
            }
        }

        stats
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::MassTree;

    #[test]
    fn test_empty_tree_iteration() {
        let tree: MassTree<u64> = MassTree::new();
        let stats = unsafe {
            LeafStats::collect::<u64, 15>(
                tree.root_ptr
                    .load(std::sync::atomic::Ordering::Acquire)
                    .cast_const(),
            )
        };
        // Empty tree has one leaf (the root)
        assert_eq!(stats.leaf_count, 1);
        assert_eq!(stats.entry_count, 0);
        assert_eq!(stats.orphan_count, 0);
    }

    #[test]
    fn test_simple_tree_iteration() {
        let mut tree: MassTree<u64> = MassTree::new();

        // Insert a few keys
        let _ = tree.insert(b"key1", 1);
        let _ = tree.insert(b"key2", 2);
        let _ = tree.insert(b"key3", 3);

        let stats = unsafe {
            LeafStats::collect::<u64, 15>(
                tree.root_ptr
                    .load(std::sync::atomic::Ordering::Acquire)
                    .cast_const(),
            )
        };

        assert!(stats.leaf_count >= 1);
        assert_eq!(stats.entry_count, 3);
        assert_eq!(stats.orphan_count, 0);
    }
}
