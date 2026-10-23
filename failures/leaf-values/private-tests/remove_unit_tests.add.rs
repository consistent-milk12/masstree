// Append this test at the end of `src/tree/remove/unit_tests.rs` in a
// temporary checkout. It deterministically stages the pre-lock search,
// inserts a same-ikey key that shifts the victim's logical position, then
// calls the real post-lock verifier.
//
//   cargo test --lib audit_remove_stale_ki -- --nocapture

#[test]
fn audit_remove_stale_ki_same_ikey_removes_wrong_key() {
    use crate::key::Key;

    let tree: TestTree = MassTree15::new();
    let k1: &[u8] = b"abcdefghZZZ";
    let k0: &[u8] = b"abcdefgh";

    tree.insert(k1, 100u64);
    let guard = tree.guard();
    let root = tree.load_root_ptr_generic(&guard);
    let leaf_ptr = tree.reach_leaf_concurrent_generic(root, &Key::new(k1), false, &guard);
    let leaf: &TestLeaf = unsafe { &*leaf_ptr };

    let perm = leaf.permutation();
    let (ki, kp) = match NodeCleaner::search_for_remove::<false, _>(leaf, &Key::new(k1), &perm) {
        super::RemoveSearchResult::Found { ki, kp } => (ki, kp),
        other => panic!("expected Found, got {other:?}"),
    };
    assert_eq!(leaf.ikey(kp), Key::new(k1).ikey());

    tree.insert_with_guard(k0, 7u64, &guard);

    let mut route = Vec::new();
    match NodeCleaner::lock_and_verify_for_remove(
        &tree,
        leaf_ptr,
        ki,
        &Key::new(k1),
        &mut route,
        &guard,
    ) {
        super::RemoveLockResult::Ready(cursor) => {
            let removed = cursor.finish_remove().map(|p| *p);
            assert_eq!(removed, Some(7), "removed the wrong key's value");
        }
        _ => panic!("expected Ready for the wrong key"),
    }

    assert_eq!(tree.get_with_guard(k1, &guard).map(|p| *p), Some(100));
    assert_eq!(tree.get_with_guard(k0, &guard).map(|p| *p), None);
}
