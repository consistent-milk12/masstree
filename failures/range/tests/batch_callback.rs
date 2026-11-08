//! Regression candidates: callbacks must never observe an unvalidated batch.
//! These intentionally fail until RNG-10 is fixed.
use masstree::{MassTree, MassTree15, RangeBound::Unbounded};

fn check_order(seen: &[u64], reverse: bool) {
    assert!(
        seen.windows(2)
            .all(|pair| if reverse { pair[0] > pair[1] } else { pair[0] < pair[1] }),
        "callback observed non-monotonic values: {seen:?}"
    );
}

fn inline_scan(reverse: bool, values_only: bool) {
    let tree = MassTree::<u64>::new();
    for key in 1u64..=15 {
        tree.insert(&key.to_be_bytes(), key);
    }
    let guard = tree.guard();
    let mut seen = Vec::new();
    let mut visit = |value| {
        seen.push(value);
        // Mutate inside the batch, after the initial entry and its callback.
        if seen.len() == 2 {
            let removed = if reverse { 13u64 } else { 3u64 };
            let added = if reverse { 0u64 } else { 16u64 };
            assert_eq!(tree.remove(&removed.to_be_bytes()).unwrap(), Some(removed));
            assert_eq!(tree.insert(&added.to_be_bytes(), added), None);
        }
        // Bound the failure even if retry logic starts replaying entries.
        seen.len() < 40
    };
    let count = match (reverse, values_only) {
        (false, false) => tree.scan_intra_leaf_batch(Unbounded, Unbounded, |_, v| visit(v), &guard),
        (true, false) => tree.scan_rev_batch(Unbounded, Unbounded, |_, v| visit(v), &guard),
        (false, true) => tree.scan_values(Unbounded, Unbounded, &mut visit, &guard),
        (true, true) => tree.scan_values_rev(Unbounded, Unbounded, &mut visit, &guard),
    };
    assert_eq!(count, seen.len());
    check_order(&seen, reverse);
}

#[test]
fn forward_keyed() {
    inline_scan(false, false);
}

#[test]
fn reverse_keyed() {
    inline_scan(true, false);
}

#[test]
fn forward_values() {
    inline_scan(false, true);
}

#[test]
fn reverse_values() {
    inline_scan(true, true);
}

fn ref_scan(reverse: bool) {
    let tree = MassTree15::<u64>::new();
    for key in 1u64..=15 {
        tree.insert(&key.to_be_bytes(), key);
    }
    let guard = tree.guard();
    let mut seen = Vec::new();
    let mut visit = |_: &[u8], value: &u64| {
        seen.push(*value);
        if seen.len() == 2 {
            let removed = if reverse { 13u64 } else { 3u64 };
            let added = if reverse { 0u64 } else { 16u64 };
            assert_eq!(tree.remove(&removed.to_be_bytes()).unwrap(), Some(removed));
            assert_eq!(tree.insert(&added.to_be_bytes(), added), None);
        }
        seen.len() < 40
    };
    let count = if reverse {
        tree.scan_rev_batch_ref(Unbounded, Unbounded, &mut visit, &guard)
    } else {
        tree.scan_intra_leaf_batch_ref(Unbounded, Unbounded, &mut visit, &guard)
    };
    assert_eq!(count, seen.len());
    check_order(&seen, reverse);
}

#[test]
fn forward_ref() {
    ref_scan(false);
}

#[test]
fn reverse_ref() {
    ref_scan(true);
}
