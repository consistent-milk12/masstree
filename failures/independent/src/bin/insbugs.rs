use masstree::MassTree;
use std::panic::{AssertUnwindSafe, catch_unwind};

fn prepend() {
    let t: MassTree<usize> = MassTree::new();
    assert_eq!(t.insert(b"AAAAAAAA", 1), None);
    assert_eq!(t.insert(b"AAAAAAAAA", 2), None);
    for (i, k) in [&b"B"[..], b"C", b"D", b"E", b"F", b"G", b"H", b"I", b"J", b"K", b"L", b"M", b"N"].iter().enumerate() {
        assert_eq!(t.insert(k, 100 + i), None);
    }
    assert_eq!(t.len(), 15);
    assert_eq!(t.insert(b"AAA", 16), None);
    let keys: Vec<Vec<u8>> = t.iter(&t.guard()).map(|e| e.key().to_vec()).collect();
    println!("prepend: len={} get(A8)={:?} get(A9)={:?} get(AAA)={:?} scan_has_A8={}", t.len(), t.get(b"AAAAAAAA"), t.get(b"AAAAAAAAA"), t.get(b"AAA"), keys.iter().any(|k| k == b"AAAAAAAA"));
    let before = t.len();
    println!("prepend: reinsert A8 -> old={:?} len {} -> {}", t.insert(b"AAAAAAAA", 99), before, t.len());
}

fn split_panic(name: &str, keys: &[Vec<u8>], last: Vec<u8>) {
    let t: MassTree<usize> = MassTree::new();
    for (i, k) in keys.iter().enumerate() {
        assert_eq!(t.insert(k, i), None, "{name}: setup {k:?}");
    }
    assert_eq!(t.len(), 15, "{name}: setup len");
    let res = catch_unwind(AssertUnwindSafe(|| t.insert(&last, 999)));
    println!("{name}: insert {:?} -> {}", last, if res.is_ok() { "ok" } else { "PANIC" });
}

fn main() {
    prepend();
    let mut a: Vec<Vec<u8>> = (1..=8).map(|n| { let mut k = vec![0u8; n]; k[0] = b'A'; k }).collect();
    let mut b: Vec<Vec<u8>> = (1..=7).map(|n| { let mut k = vec![0u8; n]; k[0] = b'B'; k }).collect();
    let mut keys = Vec::new();
    keys.append(&mut a);
    keys.append(&mut b);
    let mut x = vec![0u8; 8];
    x[0] = b'B';
    split_panic("case1", &keys, x);

    let mut a: Vec<Vec<u8>> = (1..=7).map(|n| { let mut k = vec![0u8; n]; k[0] = b'A'; k }).collect();
    let mut b: Vec<Vec<u8>> = (1..=8).map(|n| { let mut k = vec![0u8; n]; k[0] = b'B'; k }).collect();
    let mut keys = Vec::new();
    keys.append(&mut a);
    keys.append(&mut b);
    let mut x = vec![0u8; 9];
    x[0] = b'A';
    x[8] = b'X';
    split_panic("case2", &keys, x);
}
