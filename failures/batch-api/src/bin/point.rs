//! Deterministic point-operation timings; correctness checks are outside timing.
use std::hint::black_box;
use std::time::Instant;

use masstree::{LeafPolicy, MassTree, MassTree15, MassTreeGeneric, SeizeAllocator};

fn measure<P: LeafPolicy<Value = u64>>(
    name: &str,
    keys: &[Vec<u8>],
    mut tree: MassTreeGeneric<P, SeizeAllocator<P>>,
) {
    tree.extend(keys.iter().map(|key| (key, 0)));
    let guard = tree.guard();
    let mut reads = Vec::new();
    let mut updates = Vec::new();
    for round in 0..7u64 {
        let start = Instant::now();
        let mut sum = 0;
        for key in keys {
            sum += *black_box(tree.get_with_guard(black_box(key), &guard).unwrap());
        }
        reads.push(start.elapsed());
        assert_eq!(sum, round * keys.len() as u64);

        let start = Instant::now();
        for key in keys {
            black_box(tree.insert_with_guard(black_box(key), round + 1, &guard));
        }
        updates.push(start.elapsed());
    }
    assert_eq!(tree.len(), keys.len());
    for key in keys {
        assert_eq!(tree.get(key), Some(7));
    }
    reads.sort();
    updates.sort();
    println!(
        "{name} len={}: read={:?} update={:?}",
        keys.len(),
        reads[3],
        updates[3]
    );
}

fn main() {
    for prefix in [b"".as_slice(), b"PREFIX00"] {
        let mut keys: Vec<_> = (0..50_000u64)
            .map(|i| [prefix, &i.to_be_bytes()].concat())
            .collect();
        let mut seed = 0x1234_5678_9abc_def0u64;
        for i in (1..keys.len()).rev() {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            keys.swap(i, seed as usize % (i + 1));
        }
        println!("prefix_len={}", prefix.len());
        measure("boxed", &keys, MassTree15::new());
        measure("inline", &keys, MassTree::new());
    }
}
