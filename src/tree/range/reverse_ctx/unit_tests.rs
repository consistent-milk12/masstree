use super::*;
use crate::{BoxPolicy, MassTree15};

#[test]
fn initial_layer_descent_rejects_stale_version_without_cursor_mutation() {
    let tree = MassTree15::<u64>::new();
    let guard = tree.guard();
    tree.insert_with_guard(b"abcdefghx", 1, &guard);
    tree.insert_with_guard(b"abcdefghy", 2, &guard);
    let root = tree.load_root_ptr_generic(&guard);
    let leaf_ptr = root.cast::<LeafNode15<BoxPolicy<u64>>>().cast_mut();
    // SAFETY: The main root contains one layer slot; guard protects the leaf.
    let leaf = unsafe { &*leaf_ptr };
    let version = leaf.version().stable();
    let perm = leaf.permutation();
    // Change the parent after its scan position was snapshotted.
    tree.insert_with_guard(b"zzzzzzzz", 3, &guard);
    let mut ctx = ReverseScanCtx::new_with_bound(root, CursorKey::from_slice(b"abcdefghz"), true);
    let before = ctx.cursor_key.full_key().to_vec();
    let result =
        ctx.try_initial_position_reverse(leaf, perm, perm.size(), version, root, leaf_ptr, true);
    assert!(matches!(result, InitialPositionResult::Retry));
    assert_eq!(ctx.cursor_key.full_key(), before);
    assert!(ctx.layer_stack.is_empty());
}

#[test]
fn reverse_bound_after_promoted_prefix_includes_whole_layer() {
    use crate::{MassTree, RangeBound};
    use std::collections::BTreeMap;
    let model: BTreeMap<_, _> = (0..40u64)
        .map(|i| ([b"PREFIX00".as_slice(), &i.to_be_bytes()].concat(), i))
        .collect();
    let tree: MassTree<u64> = model.iter().map(|(k, &v)| (k, v)).collect();
    let boxed: MassTree15<u64> = model.iter().map(|(k, &v)| (k, v)).collect();
    let guard = tree.guard();
    let boxed_guard = boxed.guard();
    let expected: Vec<_> = model.iter().rev().map(|(k, &v)| (k.clone(), v)).collect();
    for end in [
        RangeBound::Included(b"ZZZZZZZZ"),
        RangeBound::Excluded(b"ZZZZZZZZ"),
        RangeBound::Included(b"ZZZZZZZZ\0\0\0\0\0\0\0\0"),
    ] {
        let actual: Vec<_> = tree
            .range(RangeBound::Unbounded, end, &guard)
            .rev()
            .map(|e| (e.key, e.value))
            .collect();
        assert_eq!(actual, expected, "iterator: {end:?}");
        let mut actual = Vec::new();
        tree.scan_rev_batch(
            RangeBound::Unbounded,
            end,
            |k, v| {
                actual.push((k.to_vec(), v));
                true
            },
            &guard,
        );
        assert_eq!(actual, expected, "keyed batch: {end:?}");
        let mut actual = Vec::new();
        boxed.scan_rev_batch_ref(
            RangeBound::Unbounded,
            end,
            |k, v| {
                actual.push((k.to_vec(), *v));
                true
            },
            &boxed_guard,
        );
        assert_eq!(actual, expected, "reference batch: {end:?}");
        let mut actual = Vec::new();
        tree.scan_values_rev(
            RangeBound::Unbounded,
            end,
            |v| {
                actual.push(v);
                true
            },
            &guard,
        );
        assert_eq!(
            actual,
            expected.iter().map(|(_, v)| *v).collect::<Vec<_>>(),
            "values batch: {end:?}"
        );
    }
}
