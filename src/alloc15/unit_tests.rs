use std::alloc::Layout;

use super::{LeafNode15, NodeAllocatorGeneric, SeizeAllocator15};
use crate::node_pool;
use crate::value::LeafValue;

#[test]
fn test_seize_allocator15_new() {
    // Allocator is stateless - just verify construction works
    let _alloc: SeizeAllocator15<LeafValue<u64>> = SeizeAllocator15::new();
}

#[test]
fn test_seize_allocator15_alloc_leaf() {
    let alloc: SeizeAllocator15<LeafValue<u64>> = SeizeAllocator15::new();
    let leaf: Box<LeafNode15<LeafValue<u64>>> = LeafNode15::new();

    let ptr = alloc.alloc_leaf(leaf);
    assert!(!ptr.is_null());

    // Verify the pointer is valid
    unsafe {
        assert!((*ptr).is_empty());
    }

    // Clean up - alloc_leaf uses Box, so Box::from_raw is correct
    unsafe {
        drop(Box::from_raw(ptr));
    }
}

#[test]
fn test_seize_allocator15_track_leaf_is_noop() {
    let alloc: SeizeAllocator15<LeafValue<u64>> = SeizeAllocator15::new();
    let leaf: Box<LeafNode15<LeafValue<u64>>> = LeafNode15::new();
    let ptr: *mut LeafNode15<LeafValue<u64>> = Box::into_raw(leaf);

    // track_leaf is now a no-op (traversal handles cleanup)
    alloc.track_leaf(ptr);

    // Clean up manually - this was Box-allocated
    unsafe {
        drop(Box::from_raw(ptr));
    }
}

#[test]
fn test_seize_allocator15_alloc_leaf_direct() {
    let alloc: SeizeAllocator15<LeafValue<u64>> = SeizeAllocator15::new();

    let ptr = alloc.alloc_leaf_direct(false, false);
    assert!(!ptr.is_null());

    // Verify the pointer is valid and initialized
    unsafe {
        assert!((*ptr).is_empty());
    }

    // Clean up - alloc_leaf_direct uses pool, so use pool_dealloc
    unsafe {
        std::ptr::drop_in_place(ptr);
        let layout = Layout::new::<LeafNode15<LeafValue<u64>>>();
        node_pool::pool_dealloc(ptr.cast(), layout);
    }
}

#[test]
fn test_seize_allocator15_teardown_single_leaf() {
    let alloc: SeizeAllocator15<LeafValue<u64>> = SeizeAllocator15::new();

    let ptr = alloc.alloc_leaf_direct(true, false);
    assert!(!ptr.is_null());

    // teardown_tree should free the root leaf
    alloc.teardown_tree(ptr.cast());
    // If this doesn't leak memory, test passes (checked by miri)
}
