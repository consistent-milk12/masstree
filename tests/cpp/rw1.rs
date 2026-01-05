//! RW1: Random put then shuffled get.
//!
//! Port of `kvtest_rw1` from C++ `kvtest.hh`.
//!
//! Pattern:
//! 1. Insert N random keys with value = key + 1
//! 2. Shuffle the keys
//! 3. Get all keys in shuffled order, verify values

#![allow(clippy::unwrap_used)]

use masstree::MassTree15Inline as MassTree15Inline;
use rand::{Rng, SeedableRng, rngs::StdRng, seq::SliceRandom};
use std::sync::Arc;
use std::thread;

const SEED: u64 = 31949;
const N: usize = 100_000;

#[test]
fn rw1_single_thread() {
    let tree: MassTree15Inline<u64> = MassTree15Inline::new();
    let guard = tree.guard();
    let mut rng = StdRng::seed_from_u64(SEED);

    // Put phase: insert random keys
    let mut keys: Vec<u64> = Vec::with_capacity(N);
    for _ in 0..N {
        let x: u64 = rng.random();
        let key = x.to_be_bytes();
        tree.insert_with_guard(&key, x.wrapping_add(1), &guard)
            .unwrap();
        keys.push(x);
    }

    // Shuffle keys
    keys.shuffle(&mut rng);

    // Get phase: verify all keys in shuffled order
    for x in &keys {
        let key = x.to_be_bytes();
        let val = tree.get_with_guard(&key, &guard);
        assert_eq!(val, Some(x.wrapping_add(1)), "key {x} mismatch");
    }
}

#[test]
fn rw1_concurrent() {
    let tree = Arc::new(MassTree15Inline::<u64>::new());
    let num_threads = 4;
    let per_thread = N / num_threads;

    let handles: Vec<_> = (0..num_threads)
        .map(|tid| {
            let tree = Arc::clone(&tree);
            thread::spawn(move || {
                let guard = tree.guard();
                let mut rng = StdRng::seed_from_u64(SEED + tid as u64);

                // Put phase
                let mut keys: Vec<u64> = Vec::with_capacity(per_thread);
                for _ in 0..per_thread {
                    let x: u64 = rng.random();
                    let key = x.to_be_bytes();
                    let _ = tree.insert_with_guard(&key, x.wrapping_add(1), &guard);
                    keys.push(x);
                }

                // Shuffle
                keys.shuffle(&mut rng);

                // Get phase
                for x in &keys {
                    let key = x.to_be_bytes();
                    let val = tree.get_with_guard(&key, &guard);
                    assert_eq!(val, Some(x.wrapping_add(1)));
                }
            })
        })
        .collect();

    for h in handles {
        h.join().unwrap();
    }
}
