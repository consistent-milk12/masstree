use masstree::MassTree15Inline;

#[test]
fn promote_sublayer_then_drain() {
    let tree = MassTree15Inline::<u64>::new();
    let prefix: u64 = 1;
    let keys: Vec<Vec<u8>> = (0u16..20).map(|i| {
        let mut k = prefix.to_be_bytes().to_vec();
        k.extend_from_slice(&i.to_be_bytes());
        k
    }).collect();
    for (i, k) in keys.iter().enumerate() {
        tree.insert(k, i as u64);
    }
    assert_eq!(tree.len(), 20);
    for k in &keys { assert!(tree.get(k).is_some()); }
    for k in &keys { tree.remove(k).unwrap(); }
    assert_eq!(tree.len(), 0);
    for _ in 0..8 {
        let mut g = tree.guard();
        tree.maintenance(&mut g);
        drop(g);
    }
    drop(tree);
}
