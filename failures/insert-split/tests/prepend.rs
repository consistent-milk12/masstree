use masstree::MassTree;

#[test]
fn prepend_split_between_equal_ikeys() {
    let tree: MassTree<usize> = MassTree::new();

    // Same ikey (first 8 bytes), terminal key vs suffix key.
    assert_eq!(tree.insert(b"AAAAAAAA", 1usize), None);
    assert_eq!(tree.insert(b"AAAAAAAAA", 2usize), None);

    // Fill the remaining 13 slots with larger ikeys (B..N).
    let fillers: [&[u8]; 13] = [
        b"B", b"C", b"D", b"E", b"F", b"G", b"H", b"I", b"J", b"K", b"L", b"M", b"N",
    ];
    for (i, k) in fillers.iter().enumerate() {
        assert_eq!(tree.insert(k, 100 + i), None, "insert {k:?}");
    }

    assert_eq!(tree.len(), 15);

    // A key smaller than everything, forcing the prepend split at pos 1.
    assert_eq!(tree.insert(b"AAA", 16usize), None);

    eprintln!("len after split = {}", tree.len());
    let guard = tree.guard();
    for e in tree.iter(&guard) {
        eprintln!("scan key={:?} value={}", e.key, e.value);
    }
    drop(guard);

    for (k, v) in [
        (&b"AAA"[..], 16usize),
        (&b"AAAAAAAA"[..], 1usize),
        (&b"AAAAAAAAA"[..], 2usize),
        (&b"B"[..], 100usize),
        (&b"N"[..], 112usize),
    ] {
        let got = tree.get(k);
        eprintln!("get {:?} -> {:?} (want {})", k, got, v);
    }

    assert_eq!(tree.get(b"AAA"), Some(16), "new left key");
    assert_eq!(tree.get(b"AAAAAAAAA"), Some(2), "suffix key");
    assert_eq!(tree.get(b"AAAAAAAA"), Some(1), "terminal key sharing ikey with separator");
}
