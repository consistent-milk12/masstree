use masstree::MassTree15Inline;

#[test]
fn stuck_even_with_maintenance_each_remove() {
    let tree = MassTree15Inline::<u64>::new();
    let keys: Vec<Vec<u8>> = (0u16..20).map(|i| {
        let mut k = 1u64.to_be_bytes().to_vec();
        k.extend_from_slice(&i.to_be_bytes());
        k
    }).collect();
    for (i, k) in keys.iter().enumerate() { tree.insert(k, i as u64); }
    for k in &keys {
        tree.remove(k).unwrap();
        for _ in 0..3 {
            let mut g = tree.guard();
            tree.maintenance(&mut g);
            drop(g);
        }
    }
    for _ in 0..50 {
        let mut g = tree.guard();
        tree.maintenance(&mut g);
        drop(g);
    }
    println!("len={} pending={} deferred={} abandoned={}", tree.len(), tree.pending_coalesce(), tree.coalesce_deferred(), tree.coalesce_abandoned());
    println!("{}", tree.print_tree());
}
