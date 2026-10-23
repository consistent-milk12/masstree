// Append these tests at the end of `src/leaf15/unit_tests.rs` in a temporary
// checkout. They use crate-private fields and must run as lib unit tests.
//
//   cargo test --lib audit_ -- --nocapture
//   cargo +nightly miri test --lib audit_move_value_to_same_leaf_double_owns

#[test]
fn audit_value_cleared_under_inserting_is_not_detected_by_has_changed() {
    use crate::leaf15::KSUF_KEYLENX;
    use crate::permuter::Permuter15;
    use crate::policy::LeafPolicy;

    type Leaf = LeafNode15<BoxPolicy<u64>>;
    let leaf: Box<Leaf> = LeafNode15::new_root_boxed();
    leaf.set_ikey(0, 0x6162_6364_6566_6768);
    leaf.store_value(0, &BoxPolicy::<u64>::into_output(100));
    leaf.set_keylenx(0, KSUF_KEYLENX);
    leaf.set_permutation(Permuter15::make_sorted(1));

    let v = leaf.version().stable();
    assert!(!leaf.is_value_empty(0));

    let mut lock = leaf.version().lock_bounded();
    lock.mark_insert();
    let _ = leaf.take_value_for_layer(0);

    assert!(leaf.is_value_empty(0));
    assert!(
        !leaf.version().has_changed(v),
        "has_changed must notice a cleared value slot under an in-flight insert"
    );

    drop(lock);
}

#[test]
fn audit_move_value_to_same_leaf_double_owns() {
    use crate::permuter::Permuter15;
    use crate::policy::LeafPolicy;

    type Leaf = LeafNode15<BoxPolicy<u64>>;
    let leaf: Box<Leaf> = LeafNode15::new();
    let v = BoxPolicy::<u64>::into_output(7);
    leaf.store_value(0, &v);
    leaf.set_permutation(Permuter15::make_sorted(1));

    leaf.move_value_to(&leaf, 0, 1);
    drop(leaf);
}

#[test]
fn audit_new_insert_does_not_bump_version() {
    use crate::key::Key;
    use crate::tree::MassTree15;

    let tree: MassTree15<u64> = MassTree15::new();
    tree.insert(b"a", 1u64);
    let guard = tree.guard();
    let root = tree.load_root_ptr_generic(&guard);
    let leaf_ptr = tree.reach_leaf_concurrent_generic(root, &Key::new(b"a"), false, &guard);
    let version_before = unsafe { &*leaf_ptr }.version().value();
    let perm_before = unsafe { &*leaf_ptr }.permutation_raw();

    tree.insert(b"b", 2u64);

    let version_after = unsafe { &*leaf_ptr }.version().value();
    let perm_after = unsafe { &*leaf_ptr }.permutation_raw();
    assert_ne!(perm_before, perm_after, "insertion should have changed the permutation");
    assert_ne!(
        version_before, version_after,
        "new-key insert did not bump vinsert: readers cannot detect it"
    );
}
