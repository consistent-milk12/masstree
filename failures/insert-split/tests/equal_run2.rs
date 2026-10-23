use masstree::MassTree;

fn z(first: u8, len: usize) -> Vec<u8> {
    let mut k = vec![0u8; len];
    k[0] = first;
    k
}

#[test]
fn split_failed_when_insert_pos_is_mid_and_run_starts_at_zero() {
    let tree: MassTree<usize> = MassTree::new();

    // 7 keys with ikey A (lengths 1..=7)
    for len in 1..=7usize {
        tree.insert(&z(b'A', len), len);
    }
    // 8 keys with ikey B (lengths 1..=8)
    for len in 1..=8usize {
        tree.insert(&z(b'B', len), 100 + len);
    }
    assert_eq!(tree.len(), 15);

    // New key with ikey A and a suffix: sorted position is 7 (right after the A run),
    // which equals size/2; calculate_split_point walks the run down to 0 -> None.
    let mut k = vec![0u8; 9];
    k[0] = b'A';
    k[8] = b'X';
    let res = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| tree.insert(&k, 999)));
    eprintln!("insert A-suffix: {:?}", res.as_ref().map(|v| format!("{v:?}")).map_err(|_| "PANIC"));
    assert!(res.is_ok(), "insert panicked (SplitFailed)");
}
