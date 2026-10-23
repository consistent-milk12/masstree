use masstree::MassTree;
use std::panic::{AssertUnwindSafe, catch_unwind};

fn zero_suffix_key(first: u8, len: usize) -> Vec<u8> {
    let mut k = vec![0u8; len];
    k[0] = first;
    k
}

#[test]
fn split_failed_on_equal_ikey_run_at_zero() {
    let tree: MassTree<usize> = MassTree::new();

    // 8 keys sharing ikey 0x41_00.. (lengths 1..=8; trailing NULs don't change the ikey)
    for len in 1..=8usize {
        assert_eq!(tree.insert(&zero_suffix_key(b'A', len), len), None, "A len {len}");
    }
    // 7 keys sharing ikey 0x42_00..
    for len in 1..=7usize {
        assert_eq!(tree.insert(&zero_suffix_key(b'B', len), 100 + len), None, "B len {len}");
    }
    assert_eq!(tree.len(), 15);

    // 16th key: another ikey-B terminal key (length 8).
    let res = catch_unwind(AssertUnwindSafe(|| {
        tree.insert(&zero_suffix_key(b'B', 8), 999)
    }));
    eprintln!("insert result: {:?}", res.as_ref().map(|v| format!("{v:?}")).map_err(|_| "PANIC"));

    let guard = tree.guard();
    let mut count = 0;
    for _ in tree.iter(&guard) {
        count += 1;
    }
    eprintln!("len counter={} scanned={}", tree.len(), count);
    drop(guard);

    assert!(res.is_ok(), "insert panicked (SplitFailed) instead of splitting");
}
