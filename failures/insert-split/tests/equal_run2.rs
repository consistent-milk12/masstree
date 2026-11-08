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

    // New key with ikey A and a suffix: sorted position is 7 (right after the A
    // run), which equals size/2. The split must route the new key so that all
    // equal-ikey A keys stay reachable.
    let mut k = vec![0u8; 9];
    k[0] = b'A';
    k[8] = b'X';
    let res = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| tree.insert(&k, 999)));
    eprintln!(
        "insert A-suffix: {:?}",
        res.as_ref()
            .map(|v| format!("{v:?}"))
            .map_err(|_| "PANIC")
    );
    assert!(res.is_ok(), "insert panicked (SplitFailed)");

    // Every key must remain reachable through point lookup, not just the
    // newly inserted one. A split at an equal-ikey separator would strand all
    // seven A keys in the left leaf.
    for len in 1..=7usize {
        assert_eq!(tree.get(&z(b'A', len)), Some(len), "lost A key len={len}");
    }
    for len in 1..=8usize {
        assert_eq!(
            tree.get(&z(b'B', len)),
            Some(100 + len),
            "lost B key len={len}"
        );
    }
    assert_eq!(tree.get(&k), Some(999), "lost new A-suffix key");
    assert_eq!(tree.len(), 16);

    // The full-key iteration must match as well.
    let guard = tree.guard();
    let mut keys: Vec<Vec<u8>> = tree.iter(&guard).map(|e| e.key().to_vec()).collect();
    keys.sort();
    let mut expected: Vec<Vec<u8>> = (1..=7).map(|len| z(b'A', len)).collect();
    expected.extend((1..=8).map(|len| z(b'B', len)));
    expected.push(k);
    expected.sort();
    assert_eq!(keys, expected);
}
