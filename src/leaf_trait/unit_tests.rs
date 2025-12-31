use super::*;
use crate::leaf24::LeafNode24;
use crate::permuter24::Permuter24;
use crate::value::LeafValue;

// ========================================================================
//  TreePermutation Tests
// ========================================================================

fn test_permutation_empty<P: TreePermutation>() {
    let p = P::empty();
    assert_eq!(p.size(), 0);
    assert_eq!(p.back(), 0);
}

fn test_permutation_insert<P: TreePermutation>() {
    let mut p = P::empty();
    assert_eq!(p.size(), 0);

    let slot = p.insert_from_back(0);
    assert_eq!(slot, 0);
    assert_eq!(p.size(), 1);
    assert_eq!(p.get(0), 0);
}

fn test_permutation_insert_immutable<P: TreePermutation>() {
    let p = P::empty();
    let (new_p, slot) = p.insert_from_back_immutable(0);

    // Original unchanged
    assert_eq!(p.size(), 0);

    // New permuter has insert
    assert_eq!(new_p.size(), 1);
    assert_eq!(slot, 0);
    assert_eq!(new_p.get(0), 0);
}

fn test_permutation_roundtrip<P: TreePermutation>() {
    let p = P::empty();
    let raw = p.value();
    let p2 = P::from_value(raw);
    assert_eq!(p, p2);
}

#[test]
fn test_permuter24_trait_empty() {
    test_permutation_empty::<Permuter24>();
}

#[test]
fn test_permuter24_trait_insert() {
    test_permutation_insert::<Permuter24>();
}

#[test]
fn test_permuter24_trait_insert_immutable() {
    test_permutation_insert_immutable::<Permuter24>();
}

#[test]
fn test_permuter24_trait_roundtrip() {
    test_permutation_roundtrip::<Permuter24>();
}

// ========================================================================
//  TreeLeafNode Tests
// ========================================================================

fn test_leaf_new<L: TreeLeafNode<LeafValue<u64>>>() {
    let leaf: Box<L> = L::new_boxed();
    assert!(leaf.is_empty());
    assert!(!leaf.is_full());
    assert_eq!(leaf.size(), 0);
}

fn test_leaf_permutation<L: TreeLeafNode<LeafValue<u64>>>() {
    let leaf: Box<L> = L::new_boxed();
    let perm = leaf.permutation();
    assert_eq!(perm.size(), 0);

    // Insert via permutation
    let mut new_perm = perm;
    let slot = new_perm.insert_from_back(0);
    leaf.set_permutation(new_perm);

    assert_eq!(leaf.size(), 1);
    assert_eq!(leaf.permutation().get(0), slot);
}

fn test_leaf_ikey<L: TreeLeafNode<LeafValue<u64>>>() {
    let leaf: Box<L> = L::new_boxed();
    leaf.set_ikey(0, 12345);
    assert_eq!(leaf.ikey(0), 12345);
    assert_eq!(leaf.ikey_bound(), 12345);
}

fn test_leaf_keylenx<L: TreeLeafNode<LeafValue<u64>>>() {
    let leaf: Box<L> = L::new_boxed();
    leaf.set_keylenx(1, 8);
    assert_eq!(leaf.keylenx(1), 8);
    assert!(!leaf.is_layer(1));
    assert!(!leaf.has_ksuf(1));

    // Test layer marker
    leaf.set_keylenx(2, 128);
    assert!(leaf.is_layer(2));
}

fn test_leaf_linking<L: TreeLeafNode<LeafValue<u64>>>() {
    let leaf1: Box<L> = L::new_boxed();
    let leaf2: Box<L> = L::new_boxed();
    let leaf2_ptr = Box::into_raw(leaf2);

    leaf1.set_next(leaf2_ptr);
    assert_eq!(leaf1.safe_next(), leaf2_ptr);
    assert!(!leaf1.next_is_marked());

    leaf1.mark_next();
    assert!(leaf1.next_is_marked());
    assert_eq!(leaf1.safe_next(), leaf2_ptr);

    leaf1.unmark_next();
    assert!(!leaf1.next_is_marked());

    // Cleanup
    // SAFETY: leaf2_ptr was created by Box::into_raw above and not yet freed
    let _ = unsafe { Box::from_raw(leaf2_ptr) };
}

fn test_leaf_version<L: TreeLeafNode<LeafValue<u64>>>() {
    let leaf: Box<L> = L::new_boxed();
    let version = leaf.version();

    // Should be unlocked initially
    assert!(!version.is_locked());

    // Can lock (guard unlocks on drop)
    {
        let _guard = version.lock();
        assert!(version.is_locked());
    }
    // Guard dropped, should be unlocked
    assert!(!version.is_locked());
}

#[test]
fn test_leafnode24_trait_new() {
    test_leaf_new::<LeafNode24<LeafValue<u64>>>();
}

#[test]
fn test_leafnode24_trait_permutation() {
    test_leaf_permutation::<LeafNode24<LeafValue<u64>>>();
}

#[test]
fn test_leafnode24_trait_ikey() {
    test_leaf_ikey::<LeafNode24<LeafValue<u64>>>();
}

#[test]
fn test_leafnode24_trait_keylenx() {
    test_leaf_keylenx::<LeafNode24<LeafValue<u64>>>();
}

#[test]
fn test_leafnode24_trait_linking() {
    test_leaf_linking::<LeafNode24<LeafValue<u64>>>();
}

#[test]
fn test_leafnode24_trait_version() {
    test_leaf_version::<LeafNode24<LeafValue<u64>>>();
}

// ========================================================================
//  WIDTH Constant Verification
// ========================================================================

#[test]
fn test_width_constants() {
    // Permutation WIDTH matches leaf WIDTH
    assert_eq!(
        <Permuter24 as TreePermutation>::WIDTH,
        <LeafNode24<LeafValue<u64>> as TreeLeafNode<LeafValue<u64>>>::WIDTH
    );

    // Verify actual values
    assert_eq!(
        <LeafNode24<LeafValue<u64>> as TreeLeafNode<LeafValue<u64>>>::WIDTH,
        24
    );
}

// ========================================================================
//  Generic Function Tests (prove traits enable generic code)
// ========================================================================

/// Generic function that works with any permutation type
fn generic_perm_fill<P: TreePermutation>(count: usize) -> P {
    let mut perm = P::empty();
    for i in 0..count.min(P::WIDTH) {
        perm.insert_from_back(i);
    }
    perm
}

/// Generic function that works with any leaf node type
#[allow(clippy::unnecessary_box_returns)]
fn generic_leaf_setup<L: TreeLeafNode<LeafValue<u64>>>(ikey: u64) -> Box<L> {
    let leaf = L::new_boxed();
    leaf.set_ikey(0, ikey);
    leaf.set_keylenx(0, 8);

    let mut perm = leaf.permutation();
    perm.insert_from_back(0);
    leaf.set_permutation(perm);

    leaf
}

#[test]
fn test_generic_perm_fill_24() {
    let perm: Permuter24 = generic_perm_fill(10);
    assert_eq!(perm.size(), 10);
}

#[test]
fn test_generic_leaf_setup_24() {
    let leaf: Box<LeafNode24<LeafValue<u64>>> = generic_leaf_setup(42);
    assert_eq!(leaf.ikey(0), 42);
    assert_eq!(leaf.size(), 1);
}
