//! Repeatable comparison of bulk construction and guarded single-key insertion.
use std::hint::black_box;
use std::time::{Duration, Instant};

use masstree::MassTree;

fn build(keys: &[u64], bulk: bool) -> Duration {
    let mut tree = MassTree::<u64>::new();
    let start = Instant::now();
    if bulk {
        tree.extend(keys.iter().map(|&key| (key.to_be_bytes(), key)));
    } else {
        let guard = tree.guard();
        for &key in keys {
            tree.insert_with_guard(&key.to_be_bytes(), key, &guard);
        }
    }
    let elapsed = start.elapsed();
    assert_eq!(tree.len(), keys.len());
    for &key in keys {
        assert_eq!(tree.get(&key.to_be_bytes()), Some(key));
    }
    black_box(tree);
    elapsed
}

fn main() {
    let mut keys: Vec<_> = (0..20_000u64).collect();
    let mut seed = 0x1234_5678_9abc_def0u64;
    for i in (1..keys.len()).rev() {
        seed ^= seed << 13;
        seed ^= seed >> 7;
        seed ^= seed << 17;
        keys.swap(i, seed as usize % (i + 1));
    }
    for label in ["random", "ascending"] {
        let mut single = Vec::new();
        let mut bulk = Vec::new();
        for round in 0..7 {
            // Alternate order to reduce allocator/cache warmup bias.
            if round % 2 == 0 {
                single.push(build(&keys, false));
                bulk.push(build(&keys, true));
            } else {
                bulk.push(build(&keys, true));
                single.push(build(&keys, false));
            }
        }
        single.sort();
        bulk.sort();
        println!("{label}: single={:?} bulk={:?}", single[3], bulk[3]);
        keys.sort_unstable();
    }
}
