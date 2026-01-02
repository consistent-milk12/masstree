//! SAME: Extreme contention on same small set of keys.
//!
//! Port of `kvtest_same` from C++ `kvtest.hh`.
//!
//! Pattern:
//! - All threads repeatedly update the same 10 keys
//! - Tests concurrent update bugs and lock contention

#![allow(clippy::unwrap_used, clippy::cast_sign_loss)]

use masstree::MassTree15Inline as MassTree24Inline;
use rand::{Rng, SeedableRng, rngs::StdRng};
use std::sync::Arc;
use std::thread;

const SEED: u64 = 31949;
const N: usize = 100_000;
const NUM_KEYS: u64 = 10; // Same as C++

#[test]
fn same_single_thread() {
    let tree: MassTree24Inline<u64> = MassTree24Inline::new();
    let guard = tree.guard();
    let mut rng = StdRng::seed_from_u64(SEED);

    for _ in 0..N {
        let x = rng.random_range(0..NUM_KEYS);
        let key = x.to_be_bytes();
        let _ = tree.insert_with_guard(&key, x + 1, &guard);
    }

    // Verify all keys present
    for x in 0..NUM_KEYS {
        let key = x.to_be_bytes();
        let val = tree.get_with_guard(&key, &guard);
        assert_eq!(val, Some(x + 1));
    }
}

#[test]
fn same_concurrent() {
    let tree = Arc::new(MassTree24Inline::<u64>::new());
    let num_threads = 8; // High contention
    let per_thread = N / num_threads;

    let handles: Vec<_> = (0..num_threads)
        .map(|tid| {
            let tree = Arc::clone(&tree);
            thread::spawn(move || {
                let guard = tree.guard();
                let mut rng = StdRng::seed_from_u64(SEED + tid as u64);

                for _ in 0..per_thread {
                    let x = rng.random_range(0..NUM_KEYS);
                    let key = x.to_be_bytes();
                    let _ = tree.insert_with_guard(&key, x + 1, &guard);
                }
            })
        })
        .collect();

    for h in handles {
        h.join().unwrap();
    }

    // Verify all keys present
    let guard = tree.guard();
    for x in 0..NUM_KEYS {
        let key = x.to_be_bytes();
        let val = tree.get_with_guard(&key, &guard);
        assert_eq!(val, Some(x + 1));
    }
}

/// Extreme contention: 32 threads on 10 keys
#[test]
fn same_extreme_contention() {
    let tree = Arc::new(MassTree24Inline::<u64>::new());
    let num_threads = 32;
    let per_thread = 10_000;

    let handles: Vec<_> = (0..num_threads)
        .map(|tid| {
            let tree = Arc::clone(&tree);
            thread::spawn(move || {
                let guard = tree.guard();
                let mut rng = StdRng::seed_from_u64(SEED + tid as u64);

                for _ in 0..per_thread {
                    let x = rng.random_range(0..NUM_KEYS);
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
