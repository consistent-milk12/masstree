use masstree::{MassTree, RangeBound};

fn main() {
    let tree: MassTree<u64> = MassTree::new();
    let guard = tree.guard();
    for i in 0..64u64 {
        let mut k = vec![b'p'; 24];
        k.push(i as u8);
        tree.insert_with_guard(&k, i, &guard);
    }
    let start: Vec<u8> = { let mut k = vec![b'p'; 24]; k.push(27); k };
    let fwd: Vec<Vec<u8>> = tree.range(RangeBound::Included(&start), RangeBound::Unbounded, &guard).map(|e| e.key().to_vec()).collect();
    println!("fwd first={:?} len={} has_start={}", fwd.first(), fwd.len(), fwd.contains(&start));
    let all: Vec<Vec<u8>> = tree.range(RangeBound::Unbounded, RangeBound::Unbounded, &guard).map(|e| e.key().to_vec()).collect();
    println!("all len={} pos={:?}", all.len(), all.iter().position(|k| k == &start));
    println!("all around={:?}", &all[25..31]);
    let rev: Vec<Vec<u8>> = tree.range(RangeBound::Unbounded, RangeBound::Included(&start), &guard).rev().map(|e| e.key().to_vec()).collect();
    println!("rev end len={} last={:?}", rev.len(), rev.first());
    let mut prefix = Vec::new();
    tree.scan_prefix(&start, |k, _| { prefix.push(k.to_vec()); true }, &guard);
    println!("prefix={prefix:?}");
}
