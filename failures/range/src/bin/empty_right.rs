use masstree::{MassTree, RangeBound};

fn main() {
    let t: MassTree<u64> = MassTree::new();
    let g = t.guard();
    for i in 0..20u64 {
        let k = format!("k{i:04}");
        t.insert_with_guard(k.as_bytes(), i, &g);
    }
    for i in 15..20u64 {
        let k = format!("k{i:04}");
        t.remove_with_guard(k.as_bytes(), &g).unwrap();
    }
    let forward: Vec<Vec<u8>> = t
        .range(RangeBound::Unbounded, RangeBound::Unbounded, &g)
        .map(|e| e.key().to_vec())
        .collect();
    let reverse: Vec<Vec<u8>> = t
        .range(RangeBound::Unbounded, RangeBound::Unbounded, &g)
        .rev()
        .map(|e| e.key().to_vec())
        .collect();
    println!(
        "len={} forward={} reverse={} last={:?}",
        t.len(),
        forward.len(),
        reverse.len(),
        t.last().map(|e| e.key().to_vec())
    );
    assert_eq!(forward.len(), 15);
    assert_eq!(reverse.len(), 15);
    assert_eq!(t.last().map(|e| e.key().to_vec()), Some(b"k0014".to_vec()));
}
