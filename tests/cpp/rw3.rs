//! RW3: Sequential ascending insert then get.
//!
//! Port of `kvtest_rw3` from C++ `kvtest.hh`.
//!
//! Pattern:
//! 1. Insert keys 0, 1, 2, ... N sequentially
//! 2. Get all keys and verify values

#![allow(
    clippy::unwrap_used,
    clippy::cast_sign_loss,
    clippy::cast_possible_truncation
)]

use masstree::MassTree24Inline;
use std::sync::Arc;
use std::thread;

const N: u64 = 100_000;

#[test]
fn rw3_single_thread() {
    let tree: MassTree24Inline<u64> = MassTree24Inline::new();
    let guard = tree.guard();

    // Put phase: sequential ascending
    for n in 0..N {
        let key = n.to_be_bytes();
        tree.insert_with_guard(&key, n + 1, &guard).unwrap();
    }

    // Get phase: verify all
    for n in 0..N {
        let key = n.to_be_bytes();
        let val = tree.get_with_guard(&key, &guard);
        assert_eq!(val, Some(n + 1), "key {n} mismatch");
    }
}

#[test]
fn rw3_concurrent() {
    let tree = Arc::new(MassTree24Inline::<u64>::new());
    let num_threads = 4;

    // Each thread inserts its own sequential range
    let handles: Vec<_> = (0..num_threads)
        .map(|tid| {
            let tree = Arc::clone(&tree);
            thread::spawn(move || {
                let guard = tree.guard();
                let start = tid as u64 * N;
                let end = start + N;

                // Put phase
                for n in start..end {
                    let key = n.to_be_bytes();
                    tree.insert_with_guard(&key, n + 1, &guard).unwrap();
                }

                // Get phase
                for n in start..end {
                    let key = n.to_be_bytes();
                    let val = tree.get_with_guard(&key, &guard);
                    assert_eq!(val, Some(n + 1));
                }
            })
        })
        .collect();

    for h in handles {
        h.join().unwrap();
    }
}

/// Interleaved sequential: threads compete on same key space
#[test]
fn rw3_interleaved() {
    let tree = Arc::new(MassTree24Inline::<u64>::new());
    let num_threads = 4;
    let total = N * num_threads as u64;

    let handles: Vec<_> = (0..num_threads)
        .map(|tid| {
            let tree = Arc::clone(&tree);
            thread::spawn(move || {
                let guard = tree.guard();

                // Each thread inserts every num_threads-th key
                for i in 0..(N as usize) {
                    let n = (i * num_threads + tid) as u64;
                    if n < total {
                        let key = n.to_be_bytes();
                        let _ = tree.insert_with_guard(&key, n + 1, &guard);
                    }
                }
            })
        })
        .collect();

    for h in handles {
        h.join().unwrap();
    }

    // Verify all keys present
    let guard = tree.guard();
    for n in 0..total {
        let key = n.to_be_bytes();
        let val = tree.get_with_guard(&key, &guard);
        assert_eq!(val, Some(n + 1), "key {n} missing");
    }
}
