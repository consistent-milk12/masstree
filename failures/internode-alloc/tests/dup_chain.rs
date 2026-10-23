use masstree::MassTree15Inline;

#[test]
fn duplicate_chain_entries_hang() {
    let tree = MassTree15Inline::<u64>::new();
    let n = 200u64;
    for round in 0..2 {
        for i in 0..n { tree.insert(&i.to_be_bytes(), i); }
        println!("round {round} inserted len={}", tree.len());
        for i in 0..n { tree.remove(&i.to_be_bytes()).unwrap(); }
        println!("round {round} removed len={} pending={}", tree.len(), tree.pending_coalesce());
    }
    let g = tree.guard();
    let processed = tree.process_coalesce(&g);
    println!("processed={} pending={} abandoned={}", processed, tree.pending_coalesce(), tree.coalesce_abandoned());
}
