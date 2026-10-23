use masstree::MassTree15Inline;

#[test]
fn duplicate_chain_entries_with_maint_between() {
    let tree = MassTree15Inline::<u64>::new();
    let n = 200u64;
    for round in 0..2 {
        for i in 0..n { tree.insert(&i.to_be_bytes(), i); }
        for i in 0..n { tree.remove(&i.to_be_bytes()).unwrap(); }
        let g = tree.guard();
        let p = tree.process_coalesce(&g);
        drop(g);
        println!("round {round} processed={p} pending={} abandoned={}", tree.pending_coalesce(), tree.coalesce_abandoned());
    }
}
