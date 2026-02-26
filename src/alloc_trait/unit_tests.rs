use super::TreeAllocator;
use crate::alloc15::SeizeAllocator;
use crate::policy::BoxPolicy;

// ========================================================================
// Trait Object (dyn) Not Required Tests
// ========================================================================

/// Verify that the trait enables fully generic code.
fn generic_tree_setup<P, A>(_alloc: &mut A) -> bool
where
    P: crate::policy::LeafPolicy,
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
