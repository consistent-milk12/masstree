use super::*;
use crate::value::LeafValue;

#[test]
fn test_seize_allocator24_new() {
    let alloc: SeizeAllocator15<LeafValue<u64>> = SeizeAllocator15::new();
    assert_eq!(alloc.leaf_count(), 0);
    assert_eq!(alloc.internode_count(), 0);
}

#[test]
fn test_seize_allocator24_alloc_leaf() {
    let alloc: SeizeAllocator15<LeafValue<u64>> = SeizeAllocator15::new();
    let leaf: Box<LeafNode15<LeafValue<u64>>> = LeafNode15::new();

    let ptr = alloc.alloc_leaf(leaf);
    assert!(!ptr.is_null());
    assert_eq!(alloc.leaf_count(), 1);

    // Verify the pointer is valid
    unsafe {
        assert!((*ptr).is_empty());
    }
}

#[test]
fn test_seize_allocator24_track_leaf() {
    let alloc: SeizeAllocator15<LeafValue<u64>> = SeizeAllocator15::new();
    let leaf: Box<LeafNode15<LeafValue<u64>>> = LeafNode15::new();
    let ptr: *mut LeafNode15<LeafValue<u64>> = Box::into_raw(leaf);

    alloc.track_leaf(ptr);
    assert_eq!(alloc.leaf_count(), 1);
}

#[test]
fn test_seize_allocator24_drop_frees_nodes() {
    let alloc: SeizeAllocator15<LeafValue<u64>> = SeizeAllocator15::new();
    let leaf: Box<LeafNode15<LeafValue<u64>>> = LeafNode15::new();

    let _ = alloc.alloc_leaf(leaf);
    assert_eq!(alloc.leaf_count(), 1);

    // Drop the allocator - nodes should be freed
    drop(alloc);
    // If this doesn't leak memory, test passes (checked by miri)
}
