//! RW2FIXED: Interleaved inserts and gets with fixed key pattern.
//!
//! Port of `kvtest_rw2fixed` variants from C++ `kvtest.hh`.
//!
//! Pattern:
//! - Uses multiplicative hash for key distribution
//! - Gets only retrieve keys that were previously inserted
//! - Tests read/write mix with predictable key pattern

use masstree::MassTree24Inline;
use rand::{Rng, SeedableRng, rngs::StdRng};
use std::sync::Arc;
use std::thread;

const SEED: u64 = 31949;
const N: usize = 50_000;
const MULTIPLIER: u32 = 2654435761;

fn rw2fixed_impl(get_frac: f64) {
    let tree: MassTree24Inline<u64> = MassTree24Inline::new();
    let guard = tree.guard();
    let mut rng = StdRng::seed_from_u64(SEED);

    let offset: u32 = rng.random();
    let mut puts = 0u64;
    let mut gets = 0u64;

    for _ in 0..N {
        let do_get = puts > 0 && rng.random::<f64>() < get_frac;

        if do_get {
            // Get a previously inserted key
            let idx = rng.random_range(0..puts as u32);
            let x = offset.wrapping_add(idx).wrapping_mul(MULTIPLIER) % 100_000_000;
            let key = x.to_be_bytes();
            let val = tree.get_with_guard(&key, &guard);
            assert!(val.is_some(), "key {} should exist", x);
            assert_eq!(val, Some((x + 1) as u64));
            gets += 1;
        } else {
            // Insert
            let x = offset.wrapping_add(puts as u32).wrapping_mul(MULTIPLIER) % 100_000_000;
            let key = x.to_be_bytes();
            tree.insert_with_guard(&key, (x + 1) as u64, &guard)
                .unwrap();
            puts += 1;
        }
    }

    assert!(puts > 0);
    assert!(gets > 0 || get_frac == 0.0);
}

#[test]
fn rw2fixed_50() {
    rw2fixed_impl(0.5);
}

#[test]
fn rw2fixed_g90() {
    rw2fixed_impl(0.9);
}

#[test]
fn rw2fixed_g98() {
    rw2fixed_impl(0.98);
}

#[test]
fn rw2fixed_concurrent() {
    let tree = Arc::new(MassTree24Inline::<u64>::new());
    let num_threads = 4;

    let handles: Vec<_> = (0..num_threads)
        .map(|tid| {
            let tree = Arc::clone(&tree);
            thread::spawn(move || {
                let guard = tree.guard();
                let mut rng = StdRng::seed_from_u64(SEED + tid as u64);
                let offset: u32 = rng.random();
                let mut puts = 0u64;

                for _ in 0..N / num_threads {
                    let do_get = puts > 0 && rng.random::<f64>() < 0.5;

                    if do_get {
                        let idx = rng.random_range(0..puts as u32);
                        let x = offset.wrapping_add(idx).wrapping_mul(MULTIPLIER) % 100_000_000;
                        let key = x.to_be_bytes();
                        let _ = tree.get_with_guard(&key, &guard);
                    } else {
                        let x =
                            offset.wrapping_add(puts as u32).wrapping_mul(MULTIPLIER) % 100_000_000;
                        let key = x.to_be_bytes();
                        let _ = tree.insert_with_guard(&key, (x + 1) as u64, &guard);
                        puts += 1;
                    }
                }
            })
        })
        .collect();

    for h in handles {
        h.join().unwrap();
    }
}
