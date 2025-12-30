//! RWSMALL24: Mixed read/write on small key set (24 keys).
//!
//! Port of `kvtest_rwsmall24` from C++ `kvtest.hh`.
//!
//! Pattern:
//! - 24 keys, 7/8 reads, 1/8 writes
//! - Tests hot key performance

#![allow(clippy::unwrap_used)]

use masstree::MassTree24Inline;
use rand::{Rng, SeedableRng, rngs::StdRng};
use std::sync::Arc;
use std::thread;

const SEED: u64 = 31949;
const N: usize = 100_000;
const NUM_KEYS: usize = 24;

#[test]
fn rwsmall24_single_thread() {
    let tree: MassTree24Inline<u64> = MassTree24Inline::new();
    let guard = tree.guard();
    let mut rng = StdRng::seed_from_u64(SEED);

    // Pre-populate keys
    for x in 0..NUM_KEYS as u64 {
        let key = x.to_be_bytes();
        tree.insert_with_guard(&key, x, &guard).unwrap();
    }

    for n in 0..N {
        let x = rng.random_range(0..(NUM_KEYS << 3));
        let key_idx = (x >> 3) as u64;
        let key = key_idx.to_be_bytes();

        if x & 7 != 0 {
            // 7/8 reads
            let _ = tree.get_with_guard(&key, &guard);
        } else {
            // 1/8 writes
            let _ = tree.insert_with_guard(&key, n as u64, &guard);
        }
    }
}

#[test]
fn rwsmall24_concurrent() {
    let tree = Arc::new(MassTree24Inline::<u64>::new());
    let num_threads = 4;
    let per_thread = N / num_threads;

    // Pre-populate
    {
        let guard = tree.guard();
        for x in 0..NUM_KEYS as u64 {
            let key = x.to_be_bytes();
            tree.insert_with_guard(&key, x, &guard).unwrap();
        }
    }

    let handles: Vec<_> = (0..num_threads)
        .map(|tid| {
            let tree = Arc::clone(&tree);
            thread::spawn(move || {
                let guard = tree.guard();
                let mut rng = StdRng::seed_from_u64(SEED + tid as u64);

                for n in 0..per_thread {
                    let x = rng.random_range(0..(NUM_KEYS << 3));
                    let key_idx = (x >> 3) as u64;
                    let key = key_idx.to_be_bytes();

                    if x & 7 != 0 {
                        let _ = tree.get_with_guard(&key, &guard);
                    } else {
                        let _ = tree.insert_with_guard(&key, n as u64, &guard);
                    }
                }
            })
        })
        .collect();

    for h in handles {
        h.join().unwrap();
    }
}
