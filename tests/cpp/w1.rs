//! W1: Write-only benchmark.
//!
//! Port of `kvtest_w1` from C++ `kvtest.hh`.
//!
//! Pattern:
//! - Insert many random keys (10-byte key representation)
//! - No reads, pure write throughput test
//! - May have overwrites with same value

#![allow(clippy::unwrap_used)]

use masstree::MassTree15Inline;
use rand::{RngExt, SeedableRng, rngs::StdRng};
use std::sync::Arc;
use std::thread;

const SEED: u64 = 31949;
const N: usize = 100_000;

fn make_key10(x: u64) -> [u8; 10] {
    let mut key = [0u8; 10];
    // Format as 10-digit decimal string
    let s = format!("{:010}", x % 10_000_000_000);
    key.copy_from_slice(s.as_bytes());
    key
}

#[test]
fn w1_single_thread() {
    let tree: MassTree15Inline<u64> = MassTree15Inline::new();
    let guard = tree.guard();
    let mut rng = StdRng::seed_from_u64(SEED);

    for _ in 0..N {
        let x: u64 = rng.random();
        let key = make_key10(x);
        tree.insert_with_guard(&key, x + 1, &guard);
    }
}

#[test]
fn w1_concurrent_2() {
    w1_concurrent(2);
}

#[test]
fn w1_concurrent_4() {
    w1_concurrent(4);
}

#[test]
fn w1_concurrent_6() {
    w1_concurrent(6);
}

fn w1_concurrent(num_threads: usize) {
    let tree = Arc::new(MassTree15Inline::<u64>::new());
    let per_thread = N / num_threads;

    let handles: Vec<_> = (0..num_threads)
        .map(|tid| {
            let tree = Arc::clone(&tree);
            thread::spawn(move || {
                let guard = tree.guard();
                let mut rng = StdRng::seed_from_u64(SEED + tid as u64);

                for _ in 0..per_thread {
                    let x: u64 = rng.random();
                    let key = make_key10(x);
                    let _ = tree.insert_with_guard(&key, x + 1, &guard);
                }
            })
        })
        .collect();

    for h in handles {
        h.join().unwrap();
    }
}

#[test]
fn w1_sequential_keys() {
    // Write sequential keys (worst case for some data structures)
    let tree: MassTree15Inline<u64> = MassTree15Inline::new();
    let guard = tree.guard();

    for i in 0..N as u64 {
        let key = make_key10(i);
        tree.insert_with_guard(&key, i + 1, &guard);
    }

    // Verify sample
    for i in (0..N as u64).step_by(1000) {
        let key = make_key10(i);
        let val = tree.get_with_guard(&key, &guard);
        assert_eq!(val, Some(i + 1));
    }
}
