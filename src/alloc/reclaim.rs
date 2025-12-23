//! Reclaim helpers for seize-based memory reclamation.
//!
//! This module provides:
//! - Single-node reclaimers for `guard.defer_retire()`
//! - Subtree traversal for tree teardown
//! - Layer root canonicalization for stale layer pointers

// This module is private, so pub(crate) is effectively the same as pub.
// We use pub to satisfy clippy::redundant_pub_crate while keeping intent clear.
#![allow(clippy::redundant_pub_crate)]

// We cast *mut u8 to *mut NodeVersion which has stricter alignment (4 bytes).
// This is safe because all nodes are allocated with proper alignment via Box.
#![allow(clippy::cast_ptr_alignment)]

use std::collections::HashSet;

use seize::Collector;

use crate::internode::InternodeNode;
use crate::leaf::{LAYER_KEYLENX, LeafNode};
use crate::nodeversion::NodeVersion;
use crate::slot::ValueSlot;

// ============================================================================
//  Single-Node Reclaimers (seize callback signatures)
// ============================================================================

/// Reclaim a boxed leaf node (seize callback).
///
/// # Safety
///
/// - `ptr` must point to a valid `LeafNode<S, WIDTH>` allocated via `Box::into_raw`.
/// - Must only be called after seize determines it's safe (no readers).
pub(crate) unsafe fn reclaim_leaf_boxed<S: ValueSlot, const WIDTH: usize>(
    ptr: *mut LeafNode<S, WIDTH>,
    _collector: &Collector,
) {
    // SAFETY: Caller guarantees ptr is valid and from Box::into_raw.
    // Seize ensures no readers remain.
    unsafe { drop(Box::from_raw(ptr)) };
}

/// Reclaim a boxed internode (seize callback).
///
/// # Safety
///
/// - `ptr` must point to a valid `InternodeNode<S, WIDTH>` allocated via `Box::into_raw`.
/// - Must only be called after seize determines it's safe (no readers).
pub(crate) unsafe fn reclaim_internode_boxed<S: ValueSlot, const WIDTH: usize>(
    ptr: *mut InternodeNode<S, WIDTH>,
    _collector: &Collector,
) {
    // SAFETY: Caller guarantees ptr is valid and from Box::into_raw.
    // Seize ensures no readers remain.
    unsafe { drop(Box::from_raw(ptr)) };
}

// ============================================================================
//  Layer Root Canonicalization
// ============================================================================

/// Canonicalize a layer pointer to the layer root using the `maybe_parent` pattern.
///
/// Layer pointers may become stale after layer root promotion (split creates internode).
/// This follows parent pointers to find the true layer root.
///
/// # Safety
///
/// - `node` must point to a valid `LeafNode` or `InternodeNode`.
/// - Only use for layer pointers, not main-tree children.
unsafe fn canonicalize_layer_root<S: ValueSlot, const WIDTH: usize>(mut node: *mut u8) -> *mut u8 {
    // Defensive loop bound: cycles indicate corruption.
    for _ in 0..1024 {
        if node.is_null() {
            return node;
        }

        // SAFETY: Both LeafNode and InternodeNode have NodeVersion as first field.
        let version: &NodeVersion = unsafe { &*node.cast::<NodeVersion>() };

        let parent: *mut u8 = if version.is_leaf() {
            // SAFETY: version.is_leaf() confirms this is a LeafNode.
            unsafe { (*node.cast::<LeafNode<S, WIDTH>>()).parent() }
        } else {
            // SAFETY: !is_leaf() confirms this is an InternodeNode.
            unsafe { (*node.cast::<InternodeNode<S, WIDTH>>()).parent() }
        };

        if parent.is_null() {
            return node;
        }

        node = parent;
    }

    // Reached loop limit - return current node (should not happen in valid tree).
    node
}

// ============================================================================
//  Subtree Reclamation
// ============================================================================

/// Reclaim an entire subtree rooted at a type-erased pointer.
///
/// Performs DFS traversal, dropping all nodes. Does NOT follow B-link
/// pointers (next/prev) - only parent-child relationships.
///
/// # Safety
///
/// - `root_ptr` must point to a valid `LeafNode<S, WIDTH>` or `InternodeNode<S, WIDTH>`.
/// - The subtree must be unreachable by new traversals before calling.
pub(crate) unsafe fn reclaim_subtree_impl<S: ValueSlot, const WIDTH: usize>(root_ptr: *mut u8) {
    if root_ptr.is_null() {
        return;
    }

    let mut stack: Vec<*mut u8> = Vec::with_capacity(64);
    stack.push(root_ptr);

    // Track visited nodes to avoid double-free if tree has corruption.
    // Uses ptr.addr() for identity (strict provenance compliant).
    let mut visited: HashSet<usize> = HashSet::new();

    while let Some(node) = stack.pop() {
        if node.is_null() {
            continue;
        }

        // Use exposed address for identity only (never reconstitute pointers from it).
        if !visited.insert(node.addr()) {
            continue;
        }

        // SAFETY: Both node types have NodeVersion as first field.
        let version: &NodeVersion = unsafe { &*node.cast::<NodeVersion>() };

        if version.is_leaf() {
            // SAFETY: version.is_leaf() confirms LeafNode.
            let leaf: &LeafNode<S, WIDTH> = unsafe { &*node.cast::<LeafNode<S, WIDTH>>() };

            // Collect layer subtrees BEFORE dropping the leaf.
            for slot in 0..WIDTH {
                let keylenx: u8 = leaf.keylenx(slot);
                if keylenx < LAYER_KEYLENX {
                    continue;
                }

                let layer_ptr: *mut u8 = leaf.leaf_value_ptr(slot);
                if layer_ptr.is_null() {
                    continue;
                }

                // Canonicalize layer root (handles maybe_parent pattern).
                // SAFETY: layer_ptr points to valid node (keylenx >= LAYER_KEYLENX).
                let layer_root: *mut u8 = unsafe { canonicalize_layer_root::<S, WIDTH>(layer_ptr) };
                if !layer_root.is_null() {
                    stack.push(layer_root);
                }
            }

            // Drop the leaf (Drop impl handles values and ksuf).
            // SAFETY: node is valid LeafNode from Box::into_raw.
            unsafe { drop(Box::from_raw(node.cast::<LeafNode<S, WIDTH>>())) };
        } else {
            // SAFETY: !is_leaf() confirms InternodeNode.
            let inode: &InternodeNode<S, WIDTH> =
                unsafe { &*node.cast::<InternodeNode<S, WIDTH>>() };

            // Collect children BEFORE dropping the internode.
            let nkeys: usize = inode.nkeys();
            for i in 0..=nkeys {
                let child: *mut u8 = inode.child(i);
                if !child.is_null() {
                    stack.push(child);
                }
            }

            // Drop the internode.
            // SAFETY: node is valid InternodeNode from Box::into_raw.
            unsafe { drop(Box::from_raw(node.cast::<InternodeNode<S, WIDTH>>())) };
        }
    }
}

/// Seize-compatible wrapper for subtree reclamation.
///
/// This is the callback signature required by `guard.defer_retire()`.
///
/// # Safety
///
/// - `root_ptr` must point to a valid `LeafNode<S, WIDTH>` or `InternodeNode<S, WIDTH>`.
/// - The subtree must be unreachable by new traversals before retirement.
pub(crate) unsafe fn reclaim_subtree_root<S: ValueSlot, const WIDTH: usize>(
    root_ptr: *mut u8,
    _collector: &Collector,
) {
    // SAFETY: Caller guarantees root_ptr validity and safety.
    unsafe { reclaim_subtree_impl::<S, WIDTH>(root_ptr) };
}

// ============================================================================
//  Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::leaf::LeafValue;

    #[test]
    fn test_reclaim_single_leaf() {
        // Create a leaf node via Box::into_raw
        let leaf: Box<LeafNode<LeafValue<u64>, 15>> = LeafNode::new();
        let ptr = Box::into_raw(leaf);

        // Reclaim it - should not panic or leak
        // SAFETY: ptr was just created from Box::into_raw
        unsafe {
            let collector = Collector::new();
            reclaim_leaf_boxed::<LeafValue<u64>, 15>(ptr, &collector);
        }
    }

    #[test]
    fn test_reclaim_single_internode() {
        // Create an internode via Box::into_raw
        let inode: Box<InternodeNode<LeafValue<u64>, 15>> = InternodeNode::new(0);
        let ptr = Box::into_raw(inode);

        // Reclaim it - should not panic or leak
        // SAFETY: ptr was just created from Box::into_raw
        unsafe {
            let collector = Collector::new();
            reclaim_internode_boxed::<LeafValue<u64>, 15>(ptr, &collector);
        }
    }

    #[test]
    fn test_reclaim_subtree_null_is_noop() {
        // Null root should be a no-op
        // SAFETY: Null is explicitly handled
        unsafe {
            reclaim_subtree_impl::<LeafValue<u64>, 15>(std::ptr::null_mut());
        }
    }

    #[test]
    fn test_reclaim_subtree_single_leaf() {
        // Create a single leaf node
        let leaf: Box<LeafNode<LeafValue<u64>, 15>> = LeafNode::new();
        let ptr = Box::into_raw(leaf);

        // Reclaim as subtree - should work for single node
        // SAFETY: ptr was just created from Box::into_raw
        unsafe {
            reclaim_subtree_impl::<LeafValue<u64>, 15>(ptr.cast());
        }
    }
}
