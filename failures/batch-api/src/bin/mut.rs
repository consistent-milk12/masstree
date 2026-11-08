use masstree::{BatchEntry, MassTree15};

fn try_n(n: u64) -> bool {
    let tree: MassTree15<u64> = MassTree15::new();
    let g = tree.guard();
    for i in 0..n {
        tree.insert_with_guard(&i.to_be_bytes(), i, &g);
    }
    // Free slots in the leftmost leaf without emptying it.
    for i in 1..10u64.min(n) {
        tree.remove_with_guard(&i.to_be_bytes(), &g).unwrap();
    }
    drop(g);
    let e1 = BatchEntry::new(vec![0u8; 8], 1111u64);
    let e2 = BatchEntry::new(u64::MAX.to_be_bytes().to_vec(), 2222u64);
    let _res = tree.insert_batch_entries(vec![e1, e2], &tree.guard());
    let got = tree.get(&u64::MAX.to_be_bytes());
    let nkeys = tree.iter(&tree.guard()).count();
    let lost = got.is_none();
    if lost || nkeys as u64 != n + 1 {
        println!("n={n}: get(max)={:?} len={} scan_len={} (corruption)", got, tree.len(), nkeys);
        return true;
    }
    false
}
fn main() {
    for n in [16u64, 17, 20, 30, 31, 32, 40, 60, 100, 200] {
        if try_n(n) { break; }
        println!("n={n}: ok");
    }
}
