//! RW2: Interleaved put/get for random keys.
//!
//! Port of `kvtest_rw2`, `kvtest_rw2g90`, `kvtest_rw2g98` from C++ `kvtest.hh`.
//!
//! Pattern:
//! - Interleave puts and gets based on a get fraction
//! - Gets only access keys that have been put

use masstree::MassTree24Inline;
use rand::{Rng, SeedableRng, rngs::StdRng};
use std::sync::Arc;
use std::thread;

const SEED: u64 = 31949;
const C: u64 = 2654435761; // Golden ratio hash multiplier
const N: usize = 100_000;

fn rw2_seed(get_frac: f64) {
    let tree: MassTree24Inline<u64> = MassTree24Inline::new();
    let guard = tree.guard();
    let mut rng = StdRng::seed_from_u64(SEED);
    let offset: u64 = rng.random();

    let mut puts = 0u64;
    let mut gets = 0u64;

    for _ in 0..N {
        if puts == 0 || rng.random::<f64>() >= get_frac {
            // Insert
            let x = offset.wrapping_add(puts).wrapping_mul(C);
            let key = x.to_be_bytes();
            tree.insert_with_guard(&key, x.wrapping_add(1), &guard)
                .unwrap();
            puts += 1;
        } else {
            // Get
            let idx = rng.random_range(0..puts);
            let x = offset.wrapping_add(idx).wrapping_mul(C);
            let key = x.to_be_bytes();
            let val = tree.get_with_guard(&key, &guard);
            assert_eq!(val, Some(x.wrapping_add(1)));
            gets += 1;
        }
    }

    assert!(puts > 0);
    assert!(gets > 0 || get_frac == 0.0);
}

/// 50% gets
#[test]
fn rw2() {
    rw2_seed(0.5);
}

/// 90% gets
#[test]
fn rw2g90() {
    rw2_seed(0.9);
}

/// 98% gets
#[test]
fn rw2g98() {
    rw2_seed(0.98);
}

/// Concurrent rw2 with 50% gets
#[test]
fn rw2_concurrent() {
    let tree = Arc::new(MassTree24Inline::<u64>::new());
    let num_threads = 4;
    let per_thread = N / num_threads;

    let handles: Vec<_> = (0..num_threads)
        .map(|tid| {
            let tree = Arc::clone(&tree);
            thread::spawn(move || {
                let guard = tree.guard();
                let mut rng = StdRng::seed_from_u64(SEED + tid as u64);
                let offset: u64 = rng.random();

                let mut puts = 0u64;

                for _ in 0..per_thread {
                    if puts == 0 || rng.random::<f64>() >= 0.5 {
                        let x = offset.wrapping_add(puts).wrapping_mul(C);
                        let key = x.to_be_bytes();
                        let _ = tree.insert_with_guard(&key, x.wrapping_add(1), &guard);
                        puts += 1;
                    } else {
                        let idx = rng.random_range(0..puts);
                        let x = offset.wrapping_add(idx).wrapping_mul(C);
                        let key = x.to_be_bytes();
                        let val = tree.get_with_guard(&key, &guard);
                        assert_eq!(val, Some(x.wrapping_add(1)));
                    }
                }
            })
        })
        .collect();

    for h in handles {
        h.join().unwrap();
    }
}
