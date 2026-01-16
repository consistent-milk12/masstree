use super::{NodeAllocatorGeneric, TreeLeafNode, ValueSlot};
use crate::alloc24::SeizeAllocator24;
use crate::leaf24::LeafNode24;
use crate::value::LeafValue;

// ========================================================================
// Generic Test Helpers
// ========================================================================

/// Test that we can allocate a leaf via the generic trait.
fn test_generic_alloc_leaf<S, L, A>(alloc: &A)
where
    S: ValueSlot + Send + Sync + 'static,
    L: TreeLeafNode<S>,
    A: NodeAllocatorGeneric<S, L>,
{
    let leaf: Box<L> = L::new_boxed();
    let ptr: *mut L = alloc.alloc_leaf(leaf);
    assert!(!ptr.is_null());

    // Verify leaf is accessible
    unsafe {
        let leaf_ref: &L = &*ptr;
        assert!(leaf_ref.is_empty());
    }

    // Clean up: the allocator doesn't track standalone allocations,
    // so we must manually deallocate.
    // SAFETY: ptr came from alloc_leaf which uses Box::into_raw
    unsafe { drop(Box::from_raw(ptr)) };
}

/// Test that tracking a leaf works via the generic trait.
fn test_generic_track_leaf<S, L, A>(alloc: &A)
where
    S: ValueSlot + Send + Sync + 'static,
    L: TreeLeafNode<S>,
    A: NodeAllocatorGeneric<S, L>,
{
    let leaf: Box<L> = L::new_boxed();
    let ptr: *mut L = Box::into_raw(leaf);
    alloc.track_leaf(ptr);
    // track_leaf is a no-op for seize allocator (tree traversal handles cleanup).
    // For standalone allocations in tests, we must manually deallocate.
    // SAFETY: ptr came from Box::into_raw
    unsafe { drop(Box::from_raw(ptr)) };
}

// ========================================================================
// SeizeAllocator24 Tests
// ========================================================================

#[test]
fn test_seize_allocator24_generic_alloc() {
    let alloc: SeizeAllocator24<LeafValue<u64>> = SeizeAllocator24::new();
    test_generic_alloc_leaf::<LeafValue<u64>, LeafNode24<LeafValue<u64>>, _>(&alloc);
}

#[test]
fn test_seize_allocator24_generic_track() {
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
    let mut alloc24: SeizeAllocator24<LeafValue<u64>> = SeizeAllocator24::new();

    assert!(generic_tree_setup::<
        LeafValue<u64>,
        LeafNode24<LeafValue<u64>>,
        _,
    >(&mut alloc24));
}
