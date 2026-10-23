use masstree::MassTree15;

fn check<T: Copy + std::fmt::Debug + Eq + Send + Sync + 'static>(tree: &MassTree15<T>, n: usize, expect: impl Fn(usize)->T, label: &str) {
    let mut bad = 0;
    for i in 0..n {
        let k = format!("common_prefix_{i:06}").into_bytes();
        if tree.get(&k) != Some(expect(i)) { bad += 1; }
    }
    println!("{label}: len={} bad={bad}", tree.len());
}

fn main() {
    // long keys sharing first 8 bytes -> layer tree
    let n = 500usize;
    let tree: MassTree15<u64> = MassTree15::new();
    let initial: Vec<(Vec<u8>, u64)> = (0..n).map(|i| (format!("common_prefix_{i:06}").into_bytes(), i as u64)).collect();
    let r0 = tree.insert_batch(initial);
    println!("initial inserted={} updated={} failed={} len={}", r0.inserted, r0.updated, r0.failed, tree.len());

    let updates: Vec<(Vec<u8>, u64)> = (0..n).map(|i| (format!("common_prefix_{i:06}").into_bytes(), (i as u64) + 1000)).collect();
    let r = tree.insert_batch(updates);
    println!("updates inserted={} updated={} old_len={} len={}", r.inserted, r.updated, r.old_values.len(), tree.len());
    check(&tree, n, |i| i as u64 + 1000, "after batch updates");

    // Mixed: 100 existing updates + 100 new keys
    let mixed: Vec<(Vec<u8>, u64)> = (0..100).map(|i| (format!("common_prefix_{i:06}").into_bytes(), 7u64))
        .chain((n..n+100).map(|i| (format!("common_prefix_{i:06}").into_bytes(), 9u64))).collect();
    let r2 = tree.insert_batch(mixed);
    println!("mixed inserted={} updated={} old_len={} len={}", r2.inserted, r2.updated, r2.old_values.len(), tree.len());
    let mut bad = 0;
    for i in 0..100 { let k=format!("common_prefix_{i:06}").into_bytes(); if tree.get(&k)!=Some(7) {bad+=1;} }
    for i in n..n+100 { let k=format!("common_prefix_{i:06}").into_bytes(); if tree.get(&k)!=Some(9) {bad+=1;} }
    println!("mixed bad={bad}");

}
