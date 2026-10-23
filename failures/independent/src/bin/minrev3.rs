use masstree::{MassTree, RangeBound};
fn test(remove: std::ops::Range<usize>) {
    let t: MassTree<u64> = MassTree::new();
    let g = t.guard();
    for i in 0..40u64 { let k = format!("k{i:04}"); t.insert_with_guard(k.as_bytes(), i, &g); }
    for i in remove.clone() { let k = format!("k{i:04}"); t.remove_with_guard(k.as_bytes(), &g).unwrap(); }
    let rev: Vec<u64> = t.range(RangeBound::Unbounded, RangeBound::Unbounded, &g).rev().map(|e| *e.value()).collect();
    println!("removed {:?}: rev_len={} first={:?} last={:?} last_api={:?}", remove, rev.len(), rev.first(), rev.last(), t.last().map(|e| e.key));
}
fn main() {
    test(0..5);
    test(5..10);
    test(10..15);
    test(15..20);
    test(20..25);
    test(25..30);
    test(30..35);
    test(35..40);
    test(0..40);
}
