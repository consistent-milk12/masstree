use masstree::{MassTree, RangeBound};

fn test(prefix_len: usize) {
    let tree: MassTree<u64> = MassTree::new();
    let guard = tree.guard();
    for i in 0..64u64 {
        let mut k = vec![b'p'; prefix_len];
        k.push(i as u8);
        tree.insert_with_guard(&k, i, &guard);
    }
    for start_i in [0u8, 1, 7, 27, 63] {
        let start: Vec<u8> = { let mut k = vec![b'p'; prefix_len]; k.push(start_i); k };
        let fwd: Vec<Vec<u8>> = tree.range(RangeBound::Included(&start), RangeBound::Unbounded, &guard).map(|e| e.key().to_vec()).collect();
        println!("prefix_len={prefix_len} start={start_i}: first={:?} has={} len={}", fwd.first().map(|k| k[prefix_len]), fwd.contains(&start), fwd.len());
    }
}

fn main() {
    for n in [0, 8, 16, 24, 32, 40] {
        test(n);
    }
}
