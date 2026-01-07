//! RW4: Sequential descending insert then get.
//!
//! Port of `kvtest_rw4` from C++ `kvtest.hh`.
//!
//! Pattern:
//! 1. Insert keys TOP, TOP-1, TOP-2, ... descending
//! 2. Get all keys and verify values

#![allow(
    clippy::unwrap_used,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss
)]

use masstree::MassTree15Inline;
use std::sync::Arc;
use std::thread;

const TOP: u64 = 2_147_483_647; // Same as C++ (INT_MAX)
const N: u64 = 100_000;

#[test]
fn rw4_single_thread() {
    let tree: MassTree15Inline<u64> = MassTree15Inline::new();
    let guard = tree.guard();

    // Put phase: sequential descending
    for n in 0..N {
        let key_val = TOP - n;
        let key = key_val.to_be_bytes();
        tree.insert_with_guard(&key, n + 1, &guard).unwrap();
    }

    // Get phase: verify all
    for n in 0..N {
        let key_val = TOP - n;
        let key = key_val.to_be_bytes();
        let val = tree.get_with_guard(&key, &guard);
        assert_eq!(val, Some(n + 1), "key {key_val} mismatch");
    }
}

#[test]
fn rw4_concurrent() {
    let tree = Arc::new(MassTree15Inline::<u64>::new());
    let num_threads = 4;

    let handles: Vec<_> = (0..num_threads)
        .map(|tid| {
            let tree = Arc::clone(&tree);
            thread::spawn(move || {
                let guard = tree.guard();
                // Each thread gets a different starting point
                let offset = tid as u64 * N;

                // Put phase
                for n in 0..N {
                    let key_val = TOP - offset - n;
                    let key = key_val.to_be_bytes();
                    tree.insert_with_guard(&key, n + 1, &guard).unwrap();
                }

                // Get phase
                for n in 0..N {
                    let key_val = TOP - offset - n;
                    let key = key_val.to_be_bytes();
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
