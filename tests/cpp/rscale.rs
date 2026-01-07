//! Port of `kvtest_rscale` from C++ `kvtest.hh`.
//!
//! Pattern:
//! - Pre-populate tree, then pure random reads
//! - Tests read throughput scaling with threads

#![allow(clippy::unwrap_used)]

use masstree::MassTree15Inline;
use rand::{Rng, SeedableRng, rngs::StdRng};
use std::sync::Arc;
use std::thread;

const SEED: u64 = 31949;
const POPULATE: usize = 100_000;
const READS: usize = 100_000;

fn populate_tree(tree: &MassTree15Inline<u64>) {
    let guard = tree.guard();
    let mut rng = StdRng::seed_from_u64(SEED);

    for _ in 0..POPULATE {
        let x: u64 = rng.random();
        let key = x.to_be_bytes();
        let _ = tree.insert_with_guard(&key, x + 1, &guard);
    }
}

#[test]
fn rscale_single_thread() {
    let tree: MassTree15Inline<u64> = MassTree15Inline::new();
    populate_tree(&tree);

    let guard = tree.guard();
    let mut rng = StdRng::seed_from_u64(SEED);

    // Read the same keys we inserted
    for _ in 0..READS {
        let x: u64 = rng.random();
        let key = x.to_be_bytes();
        let val = tree.get_with_guard(&key, &guard);
        assert_eq!(val, Some(x + 1));
    }
}

#[test]
fn rscale_concurrent_2() {
    rscale_threads(2);
}

#[test]
fn rscale_concurrent_4() {
    rscale_threads(4);
}

#[test]
fn rscale_concurrent_6() {
    rscale_threads(6);
}

fn rscale_threads(num_threads: usize) {
    let tree = Arc::new(MassTree15Inline::<u64>::new());
    populate_tree(&tree);

    let reads_per_thread = READS / num_threads;

    let handles: Vec<_> = (0..num_threads)
        .map(|tid| {
            let tree = Arc::clone(&tree);
            thread::spawn(move || {
                let guard = tree.guard();
                // Each thread re-reads the populated keys
                let mut rng = StdRng::seed_from_u64(SEED);

                for _ in 0..reads_per_thread {
                    let x: u64 = rng.random();
                    let key = x.to_be_bytes();
                    let val = tree.get_with_guard(&key, &guard);
                    assert_eq!(val, Some(x + 1), "thread {tid} key mismatch");
                }
            })
        })
        .collect();

    for h in handles {
        h.join().unwrap();
    }
}
