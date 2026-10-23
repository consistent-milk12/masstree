use masstree::MassTree;
fn main() {
    let t: MassTree<u64> = MassTree::new();
    let g = t.guard();
    for i in 0..20u64 { let k = format!("k{i:04}"); t.insert_with_guard(k.as_bytes(), i, &g); }
    for i in 15..20u64 { let k = format!("k{i:04}"); t.remove_with_guard(k.as_bytes(), &g).unwrap(); }
    println!("len={} first={:?} last={:?}", t.len(), t.first().map(|e| e.key), t.last().map(|e| e.key));
    println!("last_with_guard={:?}", t.last_with_guard(&g).map(|e| e.key));
    println!("rev count={}", t.iter(&g).rev().count());
    println!("fwd count={}", t.iter(&g).count());
}
