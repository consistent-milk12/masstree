//! R1: Read-only benchmark (after pre-population).
//!
//! Port of `kvtest_r1` from C++ `kvtest.hh`.
//!
//! Pattern:
//! - Pre-populate with random keys
//! - Shuffle key order
//! - Read all keys in shuffled order
//! - Pure read throughput test

#![allow(clippy::indexing_slicing)]
#![allow(clippy::unwrap_used)]

use masstree::MassTree15Inline as MassTree24Inline;
use rand::{Rng, SeedableRng, rngs::StdRng, seq::SliceRandom};
use std::sync::Arc;
use std::thread;

const SEED: u64 = 31949;
const N: usize = 100_000;

fn make_key10(x: u64) -> [u8; 10] {
    let mut key = [0u8; 10];
    let s = format!("{:010}", x % 10_000_000_000);
    key.copy_from_slice(s.as_bytes());
    key
}

fn populate_tree(tree: &MassTree24Inline<u64>, seed: u64) -> Vec<u64> {
    let guard = tree.guard();
    let mut rng = StdRng::seed_from_u64(seed);
    let mut keys = Vec::with_capacity(N);

    for _ in 0..N {
        let x: u64 = rng.random();
        let key = make_key10(x);
        let _ = tree.insert_with_guard(&key, x + 1, &guard);
        keys.push(x);
    }
    keys
}

#[test]
fn r1_single_thread() {
    let tree: MassTree24Inline<u64> = MassTree24Inline::new();

    // Populate
    let mut keys = populate_tree(&tree, SEED);

    // Shuffle
    let mut rng = StdRng::seed_from_u64(SEED + 1000);
    keys.shuffle(&mut rng);

    // Read phase
    let guard = tree.guard();
    for x in &keys {
        let key = make_key10(*x);
        let val = tree.get_with_guard(&key, &guard);
        assert_eq!(val, Some(*x + 1), "key {x} mismatch");
    }
}

#[test]
fn r1_concurrent_2() {
    r1_concurrent(2);
}

#[test]
fn r1_concurrent_4() {
    r1_concurrent(4);
}

#[test]
fn r1_concurrent_6() {
    r1_concurrent(6);
}

fn r1_concurrent(num_threads: usize) {
    let tree = Arc::new(MassTree24Inline::<u64>::new());

    // Populate (single-threaded for determinism)
    let keys = populate_tree(&tree, SEED);
    let keys = Arc::new(keys);

    let handles: Vec<_> = (0..num_threads)
        .map(|tid| {
            let tree = Arc::clone(&tree);
            let keys = Arc::clone(&keys);
            thread::spawn(move || {
                let guard = tree.guard();
                let mut rng = StdRng::seed_from_u64(SEED + 1000 + tid as u64);

                // Each thread reads a shuffled subset
                let mut indices: Vec<usize> = (0..keys.len()).collect();
                indices.shuffle(&mut rng);

                let per_thread = keys.len() / num_threads;
                let start = tid * per_thread;
                let end = if tid == num_threads - 1 {
                    keys.len()
                } else {
                    start + per_thread
                };

                (start..end).for_each(|i| {
                    let idx = indices[i];
                    let x = keys[idx];
                    let key = make_key10(x);
                    let val = tree.get_with_guard(&key, &guard);
                    assert_eq!(val, Some(x + 1));
                });
            })
        })
        .collect();

    for h in handles {
        h.join().unwrap();
    }
}

#[test]
fn r1_hot_keys() {
    // Read same keys repeatedly (hot path test)
    let tree: MassTree24Inline<u64> = MassTree24Inline::new();
    let guard = tree.guard();

    // Insert a small set of keys
    let hot_keys: Vec<u64> = (0..100).collect();
    for &x in &hot_keys {
        let key = make_key10(x);
        tree.insert_with_guard(&key, x + 1, &guard).unwrap();
    }

    // Read them many times
    let mut rng = StdRng::seed_from_u64(SEED);
    for _ in 0..10_000 {
        let x = hot_keys[rng.random_range(0..hot_keys.len())];
        let key = make_key10(x);
        let val = tree.get_with_guard(&key, &guard);
        assert_eq!(val, Some(x + 1));
    }
}
