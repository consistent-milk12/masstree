//! WSCALE: Pure random write scaling test.
//!
//! Port of `kvtest_wscale` from C++ `kvtest.hh`.
//!
//! Pattern:
//! - Pure random key inserts
//! - Tests write throughput scaling with threads

#![allow(clippy::unwrap_used)]

use masstree::MassTree15Inline as MassTree24Inline;
use rand::{Rng, SeedableRng, rngs::StdRng};
use std::sync::Arc;
use std::thread;

const SEED: u64 = 31949;
const N: usize = 100_000;

#[test]
fn wscale_single_thread() {
    let tree: MassTree24Inline<u64> = MassTree24Inline::new();
    let guard = tree.guard();
    let mut rng = StdRng::seed_from_u64(SEED);

    for _ in 0..N {
        let x: u64 = rng.random();
        let key = x.to_be_bytes();
        let _ = tree.insert_with_guard(&key, x + 1, &guard);
    }
}

#[test]
fn wscale_concurrent_2() {
    wscale_threads(2);
}

#[test]
fn wscale_concurrent_4() {
    wscale_threads(4);
}

#[test]
fn wscale_concurrent_6() {
    wscale_threads(6);
}

fn wscale_threads(num_threads: usize) {
    let tree = Arc::new(MassTree24Inline::<u64>::new());
    let per_thread = N / num_threads;

    let handles: Vec<_> = (0..num_threads)
        .map(|tid| {
            let tree = Arc::clone(&tree);
            thread::spawn(move || {
                let guard = tree.guard();
                let mut rng = StdRng::seed_from_u64(SEED + tid as u64);

                for _ in 0..per_thread {
                    let x: u64 = rng.random();
                    let key = x.to_be_bytes();
                    let _ = tree.insert_with_guard(&key, x + 1, &guard);
                }
            })
        })
        .collect();

    for h in handles {
        h.join().unwrap();
    }
}
