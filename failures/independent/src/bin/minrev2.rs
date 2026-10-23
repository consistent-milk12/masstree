use masstree::{MassTree, RangeBound};
fn main() {
    let t: MassTree<u64> = MassTree::new();
    let mut g = t.guard();
    for i in 0..20u64 { let k = format!("k{i:04}"); t.insert_with_guard(k.as_bytes(), i, &g); }
    for i in 15..20u64 { let k = format!("k{i:04}"); t.remove_with_guard(k.as_bytes(), &g).unwrap(); }
    println!("before maintenance: rev={:?} pending={}", t.range(RangeBound::Unbounded, RangeBound::Unbounded, &g).rev().map(|e| *e.value()).collect::<Vec<_>>(), t.pending_coalesce());
    t.maintenance(&mut g);
    println!("maintenance pending={}", t.pending_coalesce());
    println!("after maintenance: rev={:?}", t.range(RangeBound::Unbounded, RangeBound::Unbounded, &g).rev().map(|e| *e.value()).collect::<Vec<_>>());
}
