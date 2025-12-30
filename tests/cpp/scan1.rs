//! SCAN1: Forward range scan test.
//!
//! Port of `kvtest_scan1` from C++ `kvtest.hh`.
//!
//! Pattern:
//! - Populate tree with sequential keys
//! - Perform range scans and verify order

use masstree::{MassTree24Inline, RangeBound};
use std::sync::Arc;
use std::thread;

const N: u64 = 10_000;

#[test]
fn scan1_single_thread() {
    let tree: MassTree24Inline<u64> = MassTree24Inline::new();
    let guard = tree.guard();

    // Populate with sequential keys
    for n in 0..N {
        let key = n.to_be_bytes();
        tree.insert_with_guard(&key, n, &guard).unwrap();
    }

    // Full scan
    let mut count = 0u64;
    let mut last_key: Option<u64> = None;
    tree.scan_ref(
        RangeBound::Unbounded,
        RangeBound::Unbounded,
        |key, value| {
            let key_val = u64::from_be_bytes(key.try_into().unwrap());
            assert_eq!(key_val, *value, "value mismatch");

            if let Some(last) = last_key {
                assert!(key_val > last, "keys not in order: {last} >= {key_val}");
            }
            last_key = Some(key_val);
            count += 1;
            true
        },
        &guard,
    );

    assert_eq!(count, N, "scan count mismatch");
}

#[test]
fn scan1_range() {
    let tree: MassTree24Inline<u64> = MassTree24Inline::new();
    let guard = tree.guard();

    // Populate
    for n in 0..N {
        let key = n.to_be_bytes();
        tree.insert_with_guard(&key, n, &guard).unwrap();
    }

    // Scan range [100, 200)
    let start = 100u64.to_be_bytes();
    let end = 200u64.to_be_bytes();
    let mut count = 0u64;

    tree.scan_ref(
        RangeBound::Included(&start),
        RangeBound::Excluded(&end),
        |key, value| {
            let key_val = u64::from_be_bytes(key.try_into().unwrap());
            assert!(key_val >= 100 && key_val < 200);
            assert_eq!(key_val, *value);
            count += 1;
            true
        },
        &guard,
    );

    assert_eq!(count, 100, "range scan count mismatch");
}

#[test]
fn scan1_concurrent_read() {
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

                // Each thread does full scan
                let mut count = 0u64;
                tree.scan_ref(
                    RangeBound::Unbounded,
                    RangeBound::Unbounded,
                    |_, _| {
                        count += 1;
                        true
                    },
                    &guard,
                );

                assert_eq!(count, N);
            })
        })
        .collect();

    for h in handles {
        h.join().unwrap();
    }
}

#[test]
fn scan1_concurrent_with_writes() {
    let tree = Arc::new(MassTree24Inline::<u64>::new());

    // Populate initial data
    {
        let guard = tree.guard();
        for n in 0..N {
            let key = n.to_be_bytes();
            tree.insert_with_guard(&key, n, &guard).unwrap();
        }
    }

    let num_readers = 2;
    let num_writers = 2;

    let mut handles = Vec::new();

    // Readers
    for _ in 0..num_readers {
        let tree = Arc::clone(&tree);
        handles.push(thread::spawn(move || {
            let guard = tree.guard();
            for _ in 0..100 {
                let mut count = 0u64;
                tree.scan_ref(
                    RangeBound::Unbounded,
                    RangeBound::Unbounded,
                    |_, _| {
                        count += 1;
                        true
                    },
                    &guard,
                );
                // Count may vary due to concurrent writes
                assert!(count >= N);
            }
        }));
    }

    // Writers
    for tid in 0..num_writers {
        let tree = Arc::clone(&tree);
        handles.push(thread::spawn(move || {
            let guard = tree.guard();
            let base = N + (tid as u64 * 1000);
            for n in 0..1000 {
                let key = (base + n).to_be_bytes();
                let _ = tree.insert_with_guard(&key, base + n, &guard);
            }
        }));
    }

    for h in handles {
        h.join().unwrap();
    }
}
