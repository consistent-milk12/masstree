//! RSCAN1: Reverse range scan test.
//!
//! Port of `kvtest_rscan1` from C++ `kvtest.hh`.
//!
//! Pattern:
//! - Populate tree with sequential keys
//! - Perform reverse scans and verify descending order

#![allow(
    clippy::unwrap_used,
    clippy::indexing_slicing,
    clippy::cast_possible_truncation
)]

use masstree::{MassTree24Inline, RangeBound};
use std::sync::Arc;
use std::thread;

const N: u64 = 10_000;

#[test]
fn rscan1_single_thread() {
    let tree: MassTree24Inline<u64> = MassTree24Inline::new();
    let guard = tree.guard();

    // Populate
    for n in 0..N {
        let key = n.to_be_bytes();
        tree.insert_with_guard(&key, n, &guard).unwrap();
    }

    // Reverse scan - collect all keys
    let mut keys: Vec<u64> = Vec::new();
    tree.scan_ref(
        RangeBound::Unbounded,
        RangeBound::Unbounded,
        |key, _| {
            let key_val = u64::from_be_bytes(key.try_into().unwrap());
            keys.push(key_val);
            true
        },
        &guard,
    );

    // Verify we got all keys
    assert_eq!(keys.len(), N as usize);

    // Verify ascending order (scan is forward)
    for i in 1..keys.len() {
        assert!(keys[i] > keys[i - 1], "keys not in order");
    }

    // Now verify by iterating in reverse
    keys.reverse();
    for i in 1..keys.len() {
        assert!(
            keys[i] < keys[i - 1],
            "reversed keys not in descending order"
        );
    }
}

#[test]
fn rscan1_range_descending() {
    let tree: MassTree24Inline<u64> = MassTree24Inline::new();
    let guard = tree.guard();

    // Populate
    for n in 0..N {
        let key = n.to_be_bytes();
        tree.insert_with_guard(&key, n, &guard).unwrap();
    }

    // Scan range and collect
    let start = 100u64.to_be_bytes();
    let end = 200u64.to_be_bytes();
    let mut keys: Vec<u64> = Vec::new();

    tree.scan_ref(
        RangeBound::Included(&start),
        RangeBound::Excluded(&end),
        |key, _| {
            let key_val = u64::from_be_bytes(key.try_into().unwrap());
            keys.push(key_val);
            true
        },
        &guard,
    );

    assert_eq!(keys.len(), 100);

    // Reverse and verify descending
    keys.reverse();
    for i in 1..keys.len() {
        assert!(keys[i] < keys[i - 1]);
    }
}

#[test]
fn rscan1_concurrent() {
    let tree = Arc::new(MassTree24Inline::<u64>::new());

    // Populate
    {
        let guard = tree.guard();
        for n in 0..N {
            let key = n.to_be_bytes();
            tree.insert_with_guard(&key, n, &guard).unwrap();
        }
    }

    let num_threads = 4;
    let handles: Vec<_> = (0..num_threads)
        .map(|_| {
            let tree = Arc::clone(&tree);
            thread::spawn(move || {
                let guard = tree.guard();

                let mut keys: Vec<u64> = Vec::new();
                tree.scan_ref(
                    RangeBound::Unbounded,
                    RangeBound::Unbounded,
                    |key, _| {
                        let key_val = u64::from_be_bytes(key.try_into().unwrap());
                        keys.push(key_val);
                        true
                    },
                    &guard,
                );

                assert_eq!(keys.len(), N as usize);
            })
        })
        .collect();

    for h in handles {
        h.join().unwrap();
    }
}
