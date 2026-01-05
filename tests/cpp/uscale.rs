//! USCALE: Update (overwrite) scaling test.
//!
//! Port of `kvtest_uscale` from C++ `kvtest.hh`.
//!
//! Pattern:
//! - Pre-populate tree, then repeatedly update existing keys
//! - Tests update throughput scaling

#![allow(clippy::indexing_slicing, clippy::unwrap_used)]

use masstree::MassTree15Inline as MassTree15Inline;
use rand::{Rng, SeedableRng, rngs::StdRng};
use std::sync::Arc;
use std::thread;

const SEED: u64 = 31949;
const POPULATE: usize = 100_000;
const UPDATES: usize = 100_000;

fn populate_tree(tree: &MassTree15Inline<u64>) -> Vec<u64> {
    let guard = tree.guard();
    let mut rng = StdRng::seed_from_u64(SEED);
    let mut keys = Vec::with_capacity(POPULATE);

    for _ in 0..POPULATE {
        let x: u64 = rng.random();
        let key = x.to_be_bytes();
        let _ = tree.insert_with_guard(&key, x + 1, &guard);
        keys.push(x);
    }
    keys
}

#[test]
fn uscale_single_thread() {
    let tree: MassTree15Inline<u64> = MassTree15Inline::new();
    let keys = populate_tree(&tree);

    let guard = tree.guard();
    let mut rng = StdRng::seed_from_u64(SEED + 1000);

    // Update existing keys
    for _ in 0..UPDATES {
        let idx = rng.random_range(0..keys.len());
        let x = keys[idx];
        let key = x.to_be_bytes();
        let new_val: u64 = rng.random();
        let _ = tree.insert_with_guard(&key, new_val, &guard);
    }
}

#[test]
fn uscale_concurrent_2() {
    uscale_threads(2);
}

#[test]
fn uscale_concurrent_4() {
    uscale_threads(4);
}

#[test]
fn uscale_concurrent_6() {
    uscale_threads(6);
}

fn uscale_threads(num_threads: usize) {
    let tree = Arc::new(MassTree15Inline::<u64>::new());
    let keys = populate_tree(&tree);
    let keys = Arc::new(keys);

    let updates_per_thread = UPDATES / num_threads;

    let handles: Vec<_> = (0..num_threads)
        .map(|tid| {
            let tree = Arc::clone(&tree);
            let keys = Arc::clone(&keys);
            thread::spawn(move || {
                let guard = tree.guard();
                let mut rng = StdRng::seed_from_u64(SEED + 1000 + tid as u64);

                for _ in 0..updates_per_thread {
                    let idx = rng.random_range(0..keys.len());
                    let x = keys[idx];
                    let key = x.to_be_bytes();
                    let new_val: u64 = rng.random();
                    let _ = tree.insert_with_guard(&key, new_val, &guard);
                }
            })
        })
        .collect();

    for h in handles {
        h.join().unwrap();
    }
}
