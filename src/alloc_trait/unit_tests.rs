use super::TreeAllocator;
use crate::alloc15::SeizeAllocator;
use crate::leaf15::LeafNode15;
use crate::policy::{BoxPolicy, LeafPolicy};

// ========================================================================
// Generic Test Helpers
// ========================================================================

/// Test that we can allocate a leaf via the generic trait.
fn test_generic_alloc_leaf<P, A>(alloc: &A)
where
    P: LeafPolicy,
    A: TreeAllocator<P>,
{
    let leaf: Box<LeafNode15<P>> = LeafNode15::<P>::new_boxed();
    let ptr: *mut LeafNode15<P> = alloc.alloc_leaf(leaf);
    assert!(!ptr.is_null());

    // Verify leaf is accessible
    unsafe {
        let leaf_ref: &LeafNode15<P> = &*ptr;
        assert!(leaf_ref.is_empty());
    }

    // Clean up: the allocator doesn't track standalone allocations,
    // so we must manually deallocate.
    // SAFETY: ptr came from alloc_leaf which uses Box::into_raw
    unsafe { drop(Box::from_raw(ptr)) };
}

/// Test that tracking a leaf works via the generic trait.
fn test_generic_track_leaf<P, A>(alloc: &A)
where
    P: LeafPolicy,
    A: TreeAllocator<P>,
{
    let leaf: Box<LeafNode15<P>> = LeafNode15::<P>::new_boxed();
    let ptr: *mut LeafNode15<P> = Box::into_raw(leaf);
    alloc.track_leaf(ptr);
    // track_leaf is a no-op for seize allocator (tree traversal handles cleanup).
    // For standalone allocations in tests, we must manually deallocate.
    // SAFETY: ptr came from Box::into_raw
    unsafe { drop(Box::from_raw(ptr)) };
}

// ========================================================================
// SeizeAllocator Tests
// ========================================================================

#[test]
fn test_seize_allocator15_generic_alloc() {
    let alloc: SeizeAllocator<BoxPolicy<u64>> = SeizeAllocator::new();
    test_generic_alloc_leaf::<BoxPolicy<u64>, _>(&alloc);
}

#[test]
fn test_seize_allocator15_generic_track() {
    let alloc: SeizeAllocator<BoxPolicy<u64>> = SeizeAllocator::new();
    test_generic_track_leaf::<BoxPolicy<u64>, _>(&alloc);
}

// ========================================================================
// Trait Object (dyn) Not Required Tests
// ========================================================================

/// Verify that the trait enables fully generic code.
fn generic_tree_setup<P, A>(_alloc: &mut A) -> bool
where
    P: LeafPolicy,
    A: TreeAllocator<P>,
{
    // This compiles, proving generic code can use the trait
    true
}

#[test]
fn test_generic_code_compiles() {
    let mut alloc: SeizeAllocator<BoxPolicy<u64>> = SeizeAllocator::new();

    assert!(generic_tree_setup::<BoxPolicy<u64>, _>(&mut alloc));
}
