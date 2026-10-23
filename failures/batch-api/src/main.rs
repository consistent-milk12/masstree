use masstree::{BatchEntry, MassTree15, MassTree15Inline};

fn dup_order() {
    // 400 duplicate "a" keys interleaved with 400 distinct "b<i>" keys.
    let mut entries: Vec<(Vec<u8>, u64)> = Vec::new();
    for i in 0..400u64 {
        entries.push((b"a".to_vec(), i));
        entries.push((format!("b{i:04}").into_bytes(), i));
    }
    let expected_last = 399u64;
    let tree: MassTree15<u64> = MassTree15::new();
    let res = tree.insert_batch(entries);
    let got = tree.get(b"a").unwrap();
    println!("dup: inserted={} updated={} final_a={} expected_last={} old_len={}",
        res.inserted, res.updated, got, expected_last, res.old_values.len());
    assert_eq!(res.total(), 800);
    assert_eq!(res.updated + res.inserted, 800);
    if got != expected_last {
        println!("  ** last-wins violated **");
    }
}

fn dup_old_values() {
    // Check old_values for a duplicate sequence: should contain 0..N-1 if last-wins.
    let mut entries: Vec<(Vec<u8>, u64)> = Vec::new();
    for i in 0..400u64 {
        entries.push((b"a".to_vec(), i));
        entries.push((format!("b{i:04}").into_bytes(), i));
    }
    let tree: MassTree15Inline<u64> = MassTree15Inline::new();
    let res = tree.insert_batch(entries);
    let mut olds = res.old_values.clone();
    olds.sort_unstable();
    let expected: Vec<u64> = (0..399).collect();
    println!("old_values match 0..398: {} (first few {:?})", olds == expected, &olds[..olds.len().min(8)]);
}

fn mutate_ikey() {
    // Demonstrate stale cached ikey from public mutable fields corrupts leaf ordering.
    let tree: MassTree15<u64> = MassTree15::new();
    let g = tree.guard();
    for i in 0..16u64 {
        tree.insert_with_guard(&i.to_be_bytes(), i, &g);
    }
    drop(g);

    let e1 = BatchEntry::new(vec![0u8; 8], 1111u64);
    let mut e2 = BatchEntry::new(Vec::new(), 2222u64);
    e2.key = 0x00ff_ffff_ffff_ffffu64.to_be_bytes().to_vec();
    let res = tree.insert_batch_entries(vec![e1, e2], &tree.guard());
    println!("mutate: inserted={} updated={} total={}", res.inserted, res.updated, res.total());
    println!("  get(0x00ff..) = {:?} (expected Some(1111))", tree.get(&0x00ff_ffff_ffff_ffffu64.to_be_bytes()));
    println!("  get([0;8]) = {:?}", tree.get(&[0u8; 8]));
}

fn main() {
    dup_order();
    dup_old_values();
    mutate_ikey();
}
