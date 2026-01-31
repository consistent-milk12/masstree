//! CONFLICTSCAN: Scan with concurrent modifications.
//!
//! Port of `kvtest_conflictscan1` from C++ `kvtest.hh`.
//!
//! Pattern:
//! - Concurrent readers scanning while writers modify
//! - Tests scan stability under concurrent updates

#![allow(clippy::unwrap_used)]

use masstree::{MassTree15Inline, RangeBound};
use rand::{Rng, SeedableRng, rngs::StdRng};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::thread;
use std::time::Duration;

const SEED: u64 = 31949;
const INITIAL_KEYS: u64 = 10_000;

#[test]
fn conflictscan_single_writer() {
    let tree = Arc::new(MassTree15Inline::<u64>::new());
    let done = Arc::new(AtomicBool::new(false));

    // Pre-populate
    {
        let guard = tree.guard();
        for n in 0..INITIAL_KEYS {
            let key = n.to_be_bytes();
            tree.insert_with_guard(&key, n, &guard);
        }
    }

    // Writer thread
    let writer_tree = Arc::clone(&tree);
    let writer_done = Arc::clone(&done);
    let writer = thread::spawn(move || {
        let guard = writer_tree.guard();
        let mut rng = StdRng::seed_from_u64(SEED);
        let mut n = INITIAL_KEYS;

        while !writer_done.load(Ordering::Relaxed) {
            // Insert new keys
            let key = n.to_be_bytes();
            let _ = writer_tree.insert_with_guard(&key, n, &guard);
            n += 1;

            // Also update some existing keys
            let existing = rng.random_range(0..INITIAL_KEYS);
            let key = existing.to_be_bytes();
            let _ = writer_tree.insert_with_guard(&key, existing + 1000, &guard);

            if n.is_multiple_of(1000) {
                thread::yield_now();
            }
        }
    });

    // Reader threads doing scans
    let num_readers = 3;
    let readers: Vec<_> = (0..num_readers)
        .map(|_| {
            let tree = Arc::clone(&tree);
            let done = Arc::clone(&done);
            thread::spawn(move || {
                let guard = tree.guard();
                let mut scan_count = 0;

                while !done.load(Ordering::Relaxed) {
                    let mut count = 0u64;
                    tree.scan(
                        RangeBound::Unbounded,
                        RangeBound::Unbounded,
                        |_, _| {
                            count += 1;
                            true
                        },
                        &guard,
                    );

                    // Should see at least initial keys
                    assert!(
                        count >= INITIAL_KEYS,
                        "scan {scan_count}: only saw {count} keys, expected >= {INITIAL_KEYS}"
                    );
                    scan_count += 1;
                }
            })
        })
        .collect();

    // Let it run for a bit
    thread::sleep(Duration::from_millis(500));
    done.store(true, Ordering::Relaxed);

    writer.join().unwrap();
    for r in readers {
        r.join().unwrap();
    }
}

#[test]
fn conflictscan_multi_writer() {
    let tree = Arc::new(MassTree15Inline::<u64>::new());
    let done = Arc::new(AtomicBool::new(false));

    // Pre-populate
    {
        let guard = tree.guard();
        for n in 0..INITIAL_KEYS {
            let key = n.to_be_bytes();
            tree.insert_with_guard(&key, n, &guard);
        }
    }

    let num_writers = 2;
    let num_readers = 2;

    // Writers
    let writers: Vec<_> = (0..num_writers)
        .map(|tid| {
            let tree = Arc::clone(&tree);
            let done = Arc::clone(&done);
            thread::spawn(move || {
                let guard = tree.guard();

                #[expect(clippy::cast_sign_loss)]
                let mut n = INITIAL_KEYS + (tid as u64 * 100_000);

                while !done.load(Ordering::Relaxed) {
                    let key = n.to_be_bytes();
                    let _ = tree.insert_with_guard(&key, n, &guard);
                    n += 1;

                    if n.is_multiple_of(100) {
                        thread::yield_now();
                    }
                }
            })
        })
        .collect();

    // Readers
    let readers: Vec<_> = (0..num_readers)
        .map(|_| {
            let tree = Arc::clone(&tree);
            let done = Arc::clone(&done);
            thread::spawn(move || {
                let guard = tree.guard();

                while !done.load(Ordering::Relaxed) {
                    let mut count = 0u64;
                    tree.scan(
                        RangeBound::Unbounded,
                        RangeBound::Unbounded,
                        |_, _| {
                            count += 1;
                            true
                        },
                        &guard,
                    );
                    assert!(count >= INITIAL_KEYS);
                }
            })
        })
        .collect();

    thread::sleep(Duration::from_millis(500));
    done.store(true, Ordering::Relaxed);

    for w in writers {
        w.join().unwrap();
    }
    for r in readers {
        r.join().unwrap();
    }
}

#[test]
fn conflictscan_range_scan() {
    let tree = Arc::new(MassTree15Inline::<u64>::new());
    let done = Arc::new(AtomicBool::new(false));

    // Pre-populate range [0, 10000)
    {
        let guard = tree.guard();
        for n in 0..INITIAL_KEYS {
            let key = n.to_be_bytes();
            tree.insert_with_guard(&key, n, &guard);
        }
    }

    // Writer inserts in range [10000, ...)
    let writer_tree = Arc::clone(&tree);
    let writer_done = Arc::clone(&done);
    let writer = thread::spawn(move || {
        let guard = writer_tree.guard();
        let mut n = INITIAL_KEYS;

        while !writer_done.load(Ordering::Relaxed) {
            let key = n.to_be_bytes();
            let _ = writer_tree.insert_with_guard(&key, n, &guard);
            n += 1;
        }
    });

    // Reader scans only [0, 5000) - should not be affected by writer
    let reader_tree = Arc::clone(&tree);
    let reader_done = Arc::clone(&done);
    let reader = thread::spawn(move || {
        let guard = reader_tree.guard();
        let start = 0u64.to_be_bytes();
        let end = 5000u64.to_be_bytes();

        while !reader_done.load(Ordering::Relaxed) {
            let mut count = 0u64;
            reader_tree.scan(
                RangeBound::Included(&start),
                RangeBound::Excluded(&end),
                |_, _| {
                    count += 1;
                    true
                },
                &guard,
            );
            assert_eq!(count, 5000, "range scan count mismatch");
        }
    });

    thread::sleep(Duration::from_millis(300));
    done.store(true, Ordering::Relaxed);

    writer.join().unwrap();
    reader.join().unwrap();
}
