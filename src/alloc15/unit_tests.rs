use std::alloc::Layout;
use std::ptr as StdPtr;

use super::{LeafNode15, SeizeAllocator};
use crate::alloc_trait::TreeAllocator;
use crate::internode::InternodeNode;
use crate::node_pool;
use crate::nodeversion::NodeVersion;
use crate::policy::{BoxPolicy, InlinePolicy};

// =============================================================================
// SeizeAllocator Basic Tests (BoxPolicy)
// =============================================================================

#[test]
fn test_seize_allocator_new() {
    // Allocator is stateless - just verify construction works
    let _alloc: SeizeAllocator<BoxPolicy<u64>> = SeizeAllocator::new();
}

#[test]
fn test_seize_allocator_alloc_leaf_direct() {
    let alloc: SeizeAllocator<BoxPolicy<u64>> = SeizeAllocator::new();

    let ptr = alloc.alloc_leaf_direct(false, false);
    assert!(!ptr.is_null());

    // Verify the pointer is valid and initialized
    unsafe {
        assert!((*ptr).is_empty());
    }

    // Clean up - alloc_leaf_direct uses pool, so use pool_dealloc
    unsafe {
        StdPtr::drop_in_place(ptr);
        let layout = Layout::new::<LeafNode15<BoxPolicy<u64>>>();
        node_pool::pool_dealloc(ptr.cast(), layout);
    }
}

#[test]
fn test_seize_allocator_teardown_single_leaf() {
    let alloc: SeizeAllocator<BoxPolicy<u64>> = SeizeAllocator::new();

    let ptr = alloc.alloc_leaf_direct(true, false);
    assert!(!ptr.is_null());

    // teardown_tree should free the root leaf
    alloc.teardown_tree(ptr.cast());
    // If this doesn't leak memory, test passes (checked by miri)
}

// =============================================================================
// SeizeAllocator Deep Tree Tests
// =============================================================================

/// Max internode height is 15.
const INTERNODE_MAX_DEPTH: u32 = 15;

/// Number of keys for wide tree test.
const WIDE_TREE_NUM_KEYS: u8 = 14;

#[test]
fn test_seize_allocator_teardown_deep_internode_tree() {
    // Create a tree with many internode levels to verify no stack overflow
    // with iterative traversal.
    //
    // Structure: chain of internodes, each with one child
    // (depth = 15 levels, which is the max internode height)

    let alloc: SeizeAllocator<BoxPolicy<u64>> = SeizeAllocator::new();

    // Create the leaf at the bottom
    let leaf_ptr = alloc.alloc_leaf_direct(false, false);

    // Build chain of internodes pointing to each other
    let mut child_ptr: *mut u8 = leaf_ptr.cast();

    for height in 0..INTERNODE_MAX_DEPTH {
        let inode_ptr = alloc.alloc_internode_direct(height);
        let inode: &InternodeNode = unsafe { &*inode_ptr.cast::<InternodeNode>() };

        // Set child[0] to point to previous node
        inode.set_child(0, child_ptr);
        // Set nkeys to 0 (only using the leftmost child)
        inode.set_nkeys(0);

        // Update child's parent pointer
        unsafe {
            let version_ptr = child_ptr.cast::<NodeVersion>();
            if (*version_ptr).is_leaf() {
                let leaf: &LeafNode15<BoxPolicy<u64>> = &*child_ptr.cast();
                leaf.set_parent(inode_ptr);
            } else {
                let child_inode: &InternodeNode = &*child_ptr.cast();
                child_inode.set_parent(inode_ptr);
            }
        }

        child_ptr = inode_ptr;
    }

    // The top internode is the root
    let root_ptr = child_ptr;

    // Mark root
    unsafe {
        let inode: &InternodeNode = &*root_ptr.cast();
        inode.version().mark_root();
    }

    // teardown_tree should handle the deep tree without stack overflow
    alloc.teardown_tree(root_ptr);
    // If this completes without crashing, test passes
}

#[test]
fn test_seize_allocator_teardown_wide_internode_tree() {
    // Create a wide internode with many children (all leaves)
    // This tests breadth handling in the iterative traversal.

    let alloc: SeizeAllocator<BoxPolicy<u64>> = SeizeAllocator::new();

    // Create root internode
    let root_ptr = alloc.alloc_internode_direct_root(0);
    let root: &InternodeNode = unsafe { &*root_ptr.cast::<InternodeNode>() };

    // Create 15 leaf children (one for each child slot used with 14 keys)
    root.set_nkeys(WIDE_TREE_NUM_KEYS);

    for i in 0..=usize::from(WIDE_TREE_NUM_KEYS) {
        let leaf_ptr = alloc.alloc_leaf_direct(false, false);
        unsafe {
            root.set_child(i, leaf_ptr.cast());
            (*leaf_ptr).set_parent(root_ptr);
        }
    }

    // Set some keys
    for i in 0..usize::from(WIDE_TREE_NUM_KEYS) {
        root.set_ikey(i, (i as u64 + 1) * 100);
    }

    // teardown_tree should handle the wide tree correctly
    alloc.teardown_tree(root_ptr);
}

// =============================================================================
// SeizeAllocator InlinePolicy Tests
// =============================================================================

#[test]
fn test_seize_allocator_inline_new() {
    // Allocator is stateless - just verify construction works
    let _alloc: SeizeAllocator<InlinePolicy<u64>> = SeizeAllocator::new();
}

#[test]
fn test_seize_allocator_inline_alloc_leaf_direct() {
    let alloc: SeizeAllocator<InlinePolicy<u64>> = SeizeAllocator::new();

    let ptr = alloc.alloc_leaf_direct(false, false);
    assert!(!ptr.is_null());

    // Verify the pointer is valid and initialized
    unsafe {
        assert!((*ptr).is_empty());
    }

    // Clean up - alloc_leaf_direct uses pool, so use pool_dealloc
    unsafe {
        StdPtr::drop_in_place(ptr);
        let layout = Layout::new::<LeafNode15<InlinePolicy<u64>>>();
        node_pool::pool_dealloc(ptr.cast(), layout);
    }
}

#[test]
fn test_seize_allocator_inline_teardown_single_leaf() {
    let alloc: SeizeAllocator<InlinePolicy<u64>> = SeizeAllocator::new();

    let ptr = alloc.alloc_leaf_direct(true, false);
    assert!(!ptr.is_null());

    // teardown_tree should free the root leaf
    alloc.teardown_tree(ptr.cast());
    // If this doesn't leak memory, test passes (checked by miri)
}

#[test]
fn test_seize_allocator_inline_teardown_deep_tree() {
    // Create a tree with many internode levels for inline variant
    let alloc: SeizeAllocator<InlinePolicy<u64>> = SeizeAllocator::new();

    // Create the leaf at the bottom
    let leaf_ptr = alloc.alloc_leaf_direct(false, false);

    // Build chain of internodes
    let mut child_ptr: *mut u8 = leaf_ptr.cast();

    for height in 0..INTERNODE_MAX_DEPTH {
        let inode_ptr = alloc.alloc_internode_direct(height);
        let inode: &InternodeNode = unsafe { &*inode_ptr.cast::<InternodeNode>() };

        inode.set_child(0, child_ptr);
        inode.set_nkeys(0);

        // Update child's parent pointer
        unsafe {
            let version_ptr = child_ptr.cast::<NodeVersion>();
            if (*version_ptr).is_leaf() {
                let leaf: &LeafNode15<InlinePolicy<u64>> = &*child_ptr.cast();
                leaf.set_parent(inode_ptr);
            } else {
                let child_inode: &InternodeNode = &*child_ptr.cast();
                child_inode.set_parent(inode_ptr);
            }
        }

        child_ptr = inode_ptr;
    }

    let root_ptr = child_ptr;
    unsafe {
        let inode: &InternodeNode = &*root_ptr.cast();
        inode.version().mark_root();
    }

    alloc.teardown_tree(root_ptr);
}

// =============================================================================
// Null Pointer Handling
// =============================================================================

#[test]
fn test_seize_allocator_teardown_null() {
    let alloc: SeizeAllocator<BoxPolicy<u64>> = SeizeAllocator::new();

    // Should handle null gracefully (no-op)
    alloc.teardown_tree(StdPtr::null_mut());
}

#[test]
fn test_seize_allocator_inline_teardown_null() {
    let alloc: SeizeAllocator<InlinePolicy<u64>> = SeizeAllocator::new();

    // Should handle null gracefully (no-op)
    alloc.teardown_tree(StdPtr::null_mut());
}
