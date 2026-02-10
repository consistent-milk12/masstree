use super::{TreeLeafNode, TreePermutation};
use crate::leaf15::LeafNode15;
use crate::permuter::Permuter15;
use crate::policy::{ArcPolicy, LeafPolicy};

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
fn test_permuter15_trait_empty() {
    test_permutation_empty::<Permuter15>();
}

#[test]
fn test_permuter15_trait_insert() {
    test_permutation_insert::<Permuter15>();
}

#[test]
fn test_permuter15_trait_insert_immutable() {
    test_permutation_insert_immutable::<Permuter15>();
}

#[test]
fn test_permuter15_trait_roundtrip() {
    test_permutation_roundtrip::<Permuter15>();
}

// ========================================================================
//  TreeLeafNode Tests
// ========================================================================

fn test_leaf_new<Po: LeafPolicy, L: TreeLeafNode<Po>>() {
    let leaf: Box<L> = L::new_boxed();
    assert!(leaf.is_empty());
    assert!(!leaf.is_full());
    assert_eq!(leaf.size(), 0);
}

fn test_leaf_permutation<Po: LeafPolicy, L: TreeLeafNode<Po>>() {
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

fn test_leaf_ikey<Po: LeafPolicy, L: TreeLeafNode<Po>>() {
    let leaf: Box<L> = L::new_boxed();
    leaf.set_ikey(0, 12345);
    assert_eq!(leaf.ikey(0), 12345);
    assert_eq!(leaf.ikey_bound(), 12345);
}

fn test_leaf_keylenx<Po: LeafPolicy, L: TreeLeafNode<Po>>() {
    let leaf: Box<L> = L::new_boxed();
    leaf.set_keylenx(1, 8);
    assert_eq!(leaf.keylenx(1), 8);
    assert!(!leaf.is_layer(1));
    assert!(!leaf.has_ksuf(1));

    // Test layer marker
    leaf.set_keylenx(2, 128);
    assert!(leaf.is_layer(2));
}

fn test_leaf_linking<Po: LeafPolicy, L: TreeLeafNode<Po>>() {
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

fn test_leaf_version<Po: LeafPolicy, L: TreeLeafNode<Po>>() {
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
fn test_leafnode15_trait_new() {
    test_leaf_new::<ArcPolicy<u64>, LeafNode15<ArcPolicy<u64>>>();
}

#[test]
fn test_leafnode15_trait_permutation() {
    test_leaf_permutation::<ArcPolicy<u64>, LeafNode15<ArcPolicy<u64>>>();
}

#[test]
fn test_leafnode15_trait_ikey() {
    test_leaf_ikey::<ArcPolicy<u64>, LeafNode15<ArcPolicy<u64>>>();
}

#[test]
fn test_leafnode15_trait_keylenx() {
    test_leaf_keylenx::<ArcPolicy<u64>, LeafNode15<ArcPolicy<u64>>>();
}

#[test]
fn test_leafnode15_trait_linking() {
    test_leaf_linking::<ArcPolicy<u64>, LeafNode15<ArcPolicy<u64>>>();
}

#[test]
fn test_leafnode15_trait_version() {
    test_leaf_version::<ArcPolicy<u64>, LeafNode15<ArcPolicy<u64>>>();
}

// ========================================================================
//  WIDTH Constant Verification
// ========================================================================

#[test]
fn test_width_constants() {
    // Permutation WIDTH matches leaf WIDTH
    assert_eq!(
        <Permuter15 as TreePermutation>::WIDTH,
        <LeafNode15<ArcPolicy<u64>> as TreeLeafNode<ArcPolicy<u64>>>::WIDTH
    );

    // Verify actual values
    assert_eq!(
        <LeafNode15<ArcPolicy<u64>> as TreeLeafNode<ArcPolicy<u64>>>::WIDTH,
        15
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
fn generic_leaf_setup<Po: LeafPolicy, L: TreeLeafNode<Po>>(ikey: u64) -> Box<L> {
    let leaf = L::new_boxed();
    leaf.set_ikey(0, ikey);
    leaf.set_keylenx(0, 8);

    let mut perm = leaf.permutation();
    perm.insert_from_back(0);
    leaf.set_permutation(perm);

    leaf
}

#[test]
fn test_generic_perm_fill_15() {
    let perm: Permuter15 = generic_perm_fill(10);
    assert_eq!(perm.size(), 10);
}

#[test]
fn test_generic_leaf_setup_15() {
    let leaf: Box<LeafNode15<ArcPolicy<u64>>> = generic_leaf_setup(42);
    assert_eq!(leaf.ikey(0), 42);
    assert_eq!(leaf.size(), 1);
}
