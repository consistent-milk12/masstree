//! RWSEP24: Separated key spaces per thread.
//!
//! Port of `kvtest_rwsep24` from C++ `kvtest.hh`.
//!
//! Pattern:
//! - Each thread operates on its own key range
//! - Keys are on different cache lines to avoid false sharing
//! - Tests scaling without contention

#![allow(clippy::unwrap_used, clippy::cast_sign_loss)]

use masstree::MassTree15Inline;
use rand::{Rng, SeedableRng, rngs::StdRng};
use std::sync::Arc;
use std::thread;

const SEED: u64 = 31949;
const NKEYS: usize = 24;
const N_OPS: usize = 50_000;

#[test]
fn rwsep24_single_thread() {
    let tree: MassTree15Inline<u64> = MassTree15Inline::new();
    let guard = tree.guard();

    let base = 1_000_000u64;

    // Pre-populate with separated keys
    for n in 0..(32 + NKEYS) {
        let key = (base + n as u64).to_be_bytes();
        tree.insert_with_guard(&key, n as u64, &guard);
    }

    // Mixed read/write operations
    let mut rng = StdRng::seed_from_u64(SEED);
    for _ in 0..N_OPS {
        // x has bottom 3 bits for operation selection
        let x: usize = rng.random_range(0..(NKEYS << 3));
        let key_idx = x >> 3;
        let key = (base + key_idx as u64).to_be_bytes();

        if x & 7 != 0 {
            // 7/8 chance: get
            let _ = tree.get_with_guard(&key, &guard);
        } else {
            // 1/8 chance: put
            let _ = tree.insert_with_guard(&key, x as u64, &guard);
        }
    }
}

#[test]
fn rwsep24_concurrent() {
    let tree = Arc::new(MassTree15Inline::<u64>::new());
    let num_threads = 4;

    let handles: Vec<_> = (0..num_threads)
        .map(|tid| {
            let tree = Arc::clone(&tree);
            thread::spawn(move || {
                let guard = tree.guard();
                let mut rng = StdRng::seed_from_u64(SEED + tid as u64);

                // Each thread has its own key range
                let base = 1_000_000u64 + (tid as u64 * (32 + NKEYS as u64));

                // Pre-populate this thread's key range
                for n in 0..(32 + NKEYS) {
                    let key = (base + n as u64).to_be_bytes();
                    let _ = tree.insert_with_guard(&key, n as u64, &guard);
                }

                // Mixed operations on this thread's keys
                for _ in 0..N_OPS / num_threads {
                    let x: usize = rng.random_range(0..(NKEYS << 3));
                    let key_idx = x >> 3;
                    let key = (base + key_idx as u64).to_be_bytes();

                    if x & 7 != 0 {
                        let _ = tree.get_with_guard(&key, &guard);
                    } else {
                        let _ = tree.insert_with_guard(&key, x as u64, &guard);
                    }
                }
            })
        })
        .collect();

    for h in handles {
        h.join().unwrap();
    }
}

#[test]
fn rwsep24_no_contention_scaling() {
    // Test that separated key spaces scale well
    let tree = Arc::new(MassTree15Inline::<u64>::new());
    let num_threads = 6;

    let handles: Vec<_> = (0..num_threads)
        .map(|tid| {
            let tree = Arc::clone(&tree);
            thread::spawn(move || {
                let guard = tree.guard();
                let mut rng = StdRng::seed_from_u64(SEED + tid as u64);

                // Completely separate key ranges per thread
                let base = tid as u64 * 1_000_000;

                for i in 0..10_000u64 {
                    let key = (base + i).to_be_bytes();
                    tree.insert_with_guard(&key, i, &guard);
                }

                // Verify our keys
                for _ in 0..10_000 {
                    let i = rng.random_range(0..10_000u64);
                    let key = (base + i).to_be_bytes();
                    let val = tree.get_with_guard(&key, &guard);
                    assert_eq!(val, Some(i));
                }
            })
        })
        .collect();

    for h in handles {
        h.join().unwrap();
    }
}
