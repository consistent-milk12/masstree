use masstree::MassTree15Inline;

#[test]
fn inspect_promotion() {
    let tree = MassTree15Inline::<u64>::new();
    let prefix: u64 = 1;
    let keys: Vec<Vec<u8>> = (0u16..20).map(|i| {
        let mut k = prefix.to_be_bytes().to_vec();
        k.extend_from_slice(&i.to_be_bytes());
        k
    }).collect();
    for (i, k) in keys.iter().enumerate() { tree.insert(k, i as u64); }
    println!("AFTER INSERT: len={} pending={} deferred={} abandoned={}", tree.len(), tree.pending_coalesce(), tree.coalesce_deferred(), tree.coalesce_abandoned());
    println!("{}", tree.print_tree());
    for k in &keys { tree.remove(k).unwrap(); }
    println!("AFTER REMOVE: len={} pending={} deferred={} abandoned={}", tree.len(), tree.pending_coalesce(), tree.coalesce_deferred(), tree.coalesce_abandoned());
    for i in 0..8 {
        let mut g = tree.guard();
        tree.maintenance(&mut g);
        drop(g);
        println!("MAINT {} pending={} deferred={} abandoned={}", i, tree.pending_coalesce(), tree.coalesce_deferred(), tree.coalesce_abandoned());
    }
    println!("AFTER MAINT:");
    println!("{}", tree.print_tree());
}
