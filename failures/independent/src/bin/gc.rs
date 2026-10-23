use masstree::{MassTree, MassTree15};

fn test(n: usize, boxed: bool) {
    let t: MassTree<u64> = MassTree::new();
    let mut g = t.guard();
    for i in 0..n {
        let mut k = b"shared00".to_vec();
        k.extend_from_slice(format!("{i:04}").as_bytes());
        t.insert_with_guard(&k, i as u64, &g);
    }
    let before = t.print_tree();
    for i in 0..n {
        let mut k = b"shared00".to_vec();
        k.extend_from_slice(format!("{i:04}").as_bytes());
        t.remove_with_guard(&k, &g).unwrap();
    }
    for _ in 0..30 { t.maintenance(&mut g); }
    let after = t.print_tree();
    println!("n={n} boxed={boxed} len={} pending={} abandoned={} before_sublayer={} after_sublayer={} after_has_layer={}",
        t.len(), t.pending_coalesce(), t.coalesce_abandoned(),
        before.contains("SUBLAYER"), after.contains("SUBLAYER"), after.contains("LAYER"));
    if n == 40 { println!("--- before ---\n{before}--- after ---\n{after}"); }
    drop(g);
    drop(t);
    let _ = boxed;
}

fn main() {
    test(10, false);
    test(20, false);
    test(40, false);
    test(100, false);
}
