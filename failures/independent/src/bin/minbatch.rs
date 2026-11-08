use masstree::{MassTree, MassTree15};
fn main() {
    let t: MassTree<u64> = MassTree::new();
    let g = t.guard();
    let r = t.insert_batch_with_guard(vec![(b"k".to_vec(), 1u64), (b"k2".to_vec(), 2), (b"k".to_vec(), 3)], &g);
    println!("inline inserted={} updated={} old={:?} len={} k={:?} k2={:?}", r.inserted, r.updated, r.old_values, t.len(), t.get_with_guard(b"k", &g), t.get_with_guard(b"k2", &g));

    let tb: MassTree15<u64> = MassTree15::new();
    let gb = tb.guard();
    let r = tb.insert_batch_with_guard(vec![(b"k".to_vec(), 1u64), (b"k2".to_vec(), 2), (b"k".to_vec(), 3)], &gb);
    println!("boxed inserted={} updated={} old={:?} len={} k={:?}", r.inserted, r.updated, r.old_values.iter().map(|p| **p).collect::<Vec<_>>(), tb.len(), tb.get_with_guard(b"k", &gb).map(|p| *p));

    // Same ikey, different suffix; batch fallback must create a layer.
    let t2: MassTree<u64> = MassTree::new();
    let g2 = t2.guard();
    let entries: Vec<(Vec<u8>, u64)> = (0..20u8).map(|i| { let mut k = b"shared00".to_vec(); k.push(i); (k, i as u64) }).collect();
    let r = t2.insert_batch_with_guard(entries.clone(), &g2);
    println!("layer batch inserted={} updated={} len={} expected={}", r.inserted, r.updated, t2.len(), entries.len());
    for (k, v) in &entries { assert_eq!(t2.get_with_guard(k, &g2).map(|v| *v), Some(*v), "key {k:?}"); }
}
