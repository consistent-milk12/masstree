//! SPLITREMOVE: Concurrent split and remove operations.
//!
//! Port of `kvtest_splitremove1` from C++ `kvtest.hh`.
//!
//! Pattern:
//! - Concurrent inserts causing splits
//! - Concurrent removes during splits
//! - Tests split/remove interaction

use masstree::MassTree24Inline;
use rand::{Rng, SeedableRng, rngs::StdRng};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::thread;
use std::time::Duration;

const SEED: u64 = 31949;
const N: u64 = 20_000;

#[test]
fn splitremove_sequential() {
    let tree: MassTree24Inline<u64> = MassTree24Inline::new();
    let guard = tree.guard();

    // Insert sequential keys (will cause many splits)
    for n in 0..N {
        let key = n.to_be_bytes();
        tree.insert_with_guard(&key, n, &guard).unwrap();
    }

    // Remove every other key
    for n in (0..N).step_by(2) {
        let key = n.to_be_bytes();
        let result = tree.remove_with_guard(&key, &guard);
        assert!(result.is_ok());
    }

    // Verify: odd keys present, even keys gone
    for n in 0..N {
        let key = n.to_be_bytes();
        let val = tree.get_with_guard(&key, &guard);
        if n % 2 == 0 {
            assert!(val.is_none(), "key {} should be removed", n);
        } else {
            assert_eq!(val, Some(n), "key {} should exist", n);
        }
    }
}

#[test]
fn splitremove_concurrent_insert_remove() {
    let tree = Arc::new(MassTree24Inline::<u64>::new());
    let insert_done = Arc::new(AtomicBool::new(false));
    let inserted_count = Arc::new(AtomicU64::new(0));

    // Inserter thread - causes splits
    let insert_tree = Arc::clone(&tree);
    let insert_flag = Arc::clone(&insert_done);
    let insert_count = Arc::clone(&inserted_count);
    let inserter = thread::spawn(move || {
        let guard = insert_tree.guard();

        for n in 0..N {
            let key = n.to_be_bytes();
            insert_tree.insert_with_guard(&key, n, &guard).unwrap();
            insert_count.store(n + 1, Ordering::Release);
        }

        insert_flag.store(true, Ordering::Release);
    });

    // Remover thread - chases inserter
    let remove_tree = Arc::clone(&tree);
    let remove_flag = Arc::clone(&insert_done);
    let remove_count = Arc::clone(&inserted_count);
    let remover = thread::spawn(move || {
        let guard = remove_tree.guard();
        let mut removed = 0u64;
        let target = N / 2;

        // Wait for some insertions
        while remove_count.load(Ordering::Acquire) < 100 {
            thread::yield_now();
        }

        while removed < target {
            let current = remove_count.load(Ordering::Acquire);
            let inserter_done = remove_flag.load(Ordering::Acquire);

            // Stay behind inserter, but catch up fully once inserter is done
            let can_remove = if inserter_done {
                removed * 2 < current
            } else {
                removed * 2 + 1 < current.saturating_sub(10)
            };

            if can_remove {
                let key = (removed * 2).to_be_bytes();
                match remove_tree.remove_with_guard(&key, &guard) {
                    Ok(_) => removed += 1,
                    Err(_) => thread::yield_now(),
                }
            } else {
                thread::yield_now();
            }
        }
    });

    inserter.join().unwrap();
    remover.join().unwrap();
}

#[test]
fn splitremove_random() {
    let tree = Arc::new(MassTree24Inline::<u64>::new());
    let done = Arc::new(AtomicBool::new(false));

    // Pre-populate
    {
        let guard = tree.guard();
        for n in 0..N {
            let key = n.to_be_bytes();
            tree.insert_with_guard(&key, n, &guard).unwrap();
        }
    }

    let num_threads = 4;
    let handles: Vec<_> = (0..num_threads)
        .map(|tid| {
            let tree = Arc::clone(&tree);
            let done = Arc::clone(&done);
            thread::spawn(move || {
                let guard = tree.guard();
                let mut rng = StdRng::seed_from_u64(SEED + tid as u64);

                for _ in 0..5000 {
                    if done.load(Ordering::Relaxed) {
                        break;
                    }

                    let key_val: u64 = rng.random_range(0..N * 2);
                    let key = key_val.to_be_bytes();

                    if rng.random::<bool>() {
                        // Insert (may be new or update)
                        let _ = tree.insert_with_guard(&key, key_val, &guard);
                    } else {
                        // Remove (may or may not exist)
                        let _ = tree.remove_with_guard(&key, &guard);
                    }
                }
            })
        })
        .collect();

    // Let it run
    thread::sleep(Duration::from_millis(500));
    done.store(true, Ordering::Relaxed);

    for h in handles {
        h.join().unwrap();
    }
}

#[test]
fn splitremove_reinsert() {
    let tree: MassTree24Inline<u64> = MassTree24Inline::new();
    let guard = tree.guard();

    // Insert, remove, reinsert cycle
    for cycle in 0..3 {
        // Insert
        for n in 0u64..1000 {
            let key = n.to_be_bytes();
            tree.insert_with_guard(&key, n + cycle * 1000, &guard)
                .unwrap();
        }

        // Verify
        for n in 0u64..1000 {
            let key = n.to_be_bytes();
            let val = tree.get_with_guard(&key, &guard);
            assert_eq!(val, Some(n + cycle * 1000));
        }

        // Remove all
        for n in 0u64..1000 {
            let key = n.to_be_bytes();
            let result = tree.remove_with_guard(&key, &guard);
            assert!(result.is_ok());
        }

        // Verify all gone
        for n in 0u64..1000 {
            let key = n.to_be_bytes();
            let val = tree.get_with_guard(&key, &guard);
            assert!(val.is_none(), "cycle {}: key {} should be gone", cycle, n);
        }
    }
}
