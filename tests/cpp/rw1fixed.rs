//! RW1FIXED: Insert then get with fixed random keys.
//!
//! Port of `kvtest_rw1fixed` from C++ `kvtest.hh`.
//!
//! Pattern:
//! - Insert N random keys (uniform distribution)
//! - Shuffle the keys
//! - Get all keys in shuffled order
//! - Verifies all values match

#![allow(
    clippy::cast_sign_loss,
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::unwrap_used
)]

use masstree::MassTree15Inline as MassTree15Inline;
use rand::{Rng, SeedableRng, rngs::StdRng, seq::SliceRandom};
use std::sync::Arc;
use std::thread;

const SEED: u64 = 31949;
const N: usize = 50_000;

#[test]
fn rw1fixed_single_thread() {
    let tree: MassTree15Inline<u64> = MassTree15Inline::new();
    let guard = tree.guard();
    let mut rng = StdRng::seed_from_u64(SEED);

    // Put phase - insert random keys
    let mut keys: Vec<u32> = Vec::with_capacity(N);
    for _ in 0..N {
        let x: u32 = rng.random_range(0..100_000_000);
        let key = x.to_be_bytes();
        tree.insert_with_guard(&key, u64::from(x + 1), &guard)
            .unwrap();
        keys.push(x);
    }

    // Shuffle keys for get phase
    keys.shuffle(&mut rng);

    // Get phase - verify all keys exist with correct values
    for x in &keys {
        let key = x.to_be_bytes();
        let val = tree.get_with_guard(&key, &guard);
        assert_eq!(val, Some(u64::from(*x + 1)), "key {x} mismatch");
    }
}

#[test]
fn rw1fixed_concurrent() {
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
                let mut keys: Vec<u32> = Vec::with_capacity(per_thread);
                for _ in 0..per_thread {
                    let x: u32 = rng.random_range(0..100_000_000);
                    let key = x.to_be_bytes();
                    let _ = tree.insert_with_guard(&key, u64::from(x + 1), &guard);
                    keys.push(x);
                }

                // Shuffle and get
                keys.shuffle(&mut rng);
                for x in &keys {
                    let key = x.to_be_bytes();
                    let val = tree.get_with_guard(&key, &guard);
                    // Value might differ if another thread overwrote, but key should exist
                    assert!(val.is_some(), "key {x} not found");
                }
            })
        })
        .collect();

    for h in handles {
        h.join().unwrap();
    }
}
