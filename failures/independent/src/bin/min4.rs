use masstree::{MassTree, RangeBound};

fn main() {
    for prefix_len in [0usize, 8, 16] {
        let tree: MassTree<u64> = MassTree::new();
        let guard = tree.guard();
        for i in 0..64u64 {
            let mut k = vec![b'p'; prefix_len];
            k.push(i as u8);
            tree.insert_with_guard(&k, i, &guard);
        }
        let mut failures = Vec::new();
        for i in 0..64u8 {
            let start: Vec<u8> = { let mut k = vec![b'p'; prefix_len]; k.push(i); k };
            let fwd: Vec<Vec<u8>> = tree.range(RangeBound::Included(&start), RangeBound::Unbounded, &guard).map(|e| e.key().to_vec()).collect();
            if fwd.first() != Some(&start) {
                failures.push((i, fwd.first().map(|k| k[prefix_len])));
            }
        }
        println!("prefix_len={prefix_len} failures={failures:?}");
    }
}
