//! Concurrent operation regression tests.
//!
//! These tests stress concurrent get/insert operations to catch memory safety
//! issues, race conditions, and correctness bugs.
//!
//! Run with: `cargo test --test concurrent_regression`
//! Run with release: `cargo test --test concurrent_regression --release`
//!
//! ## Tracing
//!
//! Enable tracing to debug race conditions:
//!
//! ```bash
//! # Console output only (debug level for masstree crate)
//! RUST_LOG=masstree=debug cargo test --test concurrent_regression
//!
//! # Trace specific module with JSON file logging
//! RUST_LOG=masstree::tree::locked=trace MASSTREE_LOG_JSON=1 cargo test stress_concurrent
//!
//! # Full trace to file, no console
//! RUST_LOG=trace MASSTREE_LOG_JSON=1 MASSTREE_LOG_CONSOLE=0 cargo test
//! ```
//!
//! Logs are written to `logs/` directory (gitignored).

#![allow(clippy::pedantic)]
#![expect(clippy::unwrap_used)]

mod common;

use masstree::{
    MassTree, get_debug_counters, reset_debug_counters, BLINK_SHOULD_FOLLOW_COUNT,
};
use std::collections::HashSet;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::thread;

// =============================================================================
// Basic Concurrent Insert
// =============================================================================

#[test]
fn concurrent_insert_2_threads_disjoint_keys() {
    let tree = Arc::new(MassTree::<u64>::new());

    let handles: Vec<_> = (0..2)
        .map(|t| {
            let tree = Arc::clone(&tree);
            thread::spawn(move || {
                let guard = tree.guard();
                for i in 0..100 {
                    let key = ((t * 1000 + i) as u64).to_be_bytes();
                    let _ = tree.insert_with_guard(&key, (t * 1000 + i) as u64, &guard);
                }
            })
        })
        .collect();

    for h in handles {
        h.join().unwrap();
    }

    assert_eq!(tree.len(), 200);
}

#[test]
fn concurrent_insert_4_threads_disjoint_keys() {
    let tree = Arc::new(MassTree::<u64>::new());

    let handles: Vec<_> = (0..4)
        .map(|t| {
            let tree = Arc::clone(&tree);
            thread::spawn(move || {
                let guard = tree.guard();
                for i in 0..100 {
                    let key = ((t * 1000 + i) as u64).to_be_bytes();
                    let _ = tree.insert_with_guard(&key, (t * 1000 + i) as u64, &guard);
                }
            })
        })
        .collect();

    for h in handles {
        h.join().unwrap();
    }

    assert_eq!(tree.len(), 400);
}

#[test]
fn concurrent_insert_8_threads_disjoint_keys() {
    let tree = Arc::new(MassTree::<u64>::new());

    let handles: Vec<_> = (0..8)
        .map(|t| {
            let tree = Arc::clone(&tree);
            thread::spawn(move || {
                let guard = tree.guard();
                for i in 0..100 {
                    let key = ((t * 1000 + i) as u64).to_be_bytes();
                    let _ = tree.insert_with_guard(&key, (t * 1000 + i) as u64, &guard);
                }
            })
        })
        .collect();

    for h in handles {
        h.join().unwrap();
    }

    assert_eq!(tree.len(), 800);
}

// =============================================================================
// Concurrent Insert with Contention (Same Key Ranges)
// =============================================================================

#[test]
fn concurrent_insert_2_threads_overlapping_keys() {
    common::init_tracing();

    const NUM_KEYS: usize = 100;
    const NUM_THREADS: usize = 2;

    tracing::info!(
        threads = NUM_THREADS,
        keys = NUM_KEYS,
        "Starting overlapping keys test"
    );

    let tree = Arc::new(MassTree::<u64>::new());
    let success_count = Arc::new(AtomicUsize::new(0));

    let handles: Vec<_> = (0..NUM_THREADS)
        .map(|t| {
            let tree = Arc::clone(&tree);
            let success_count = Arc::clone(&success_count);
            thread::spawn(move || {
                let guard = tree.guard();
                for i in 0..NUM_KEYS {
                    // Overlapping key range
                    let key = (i as u64).to_be_bytes();
                    if tree.insert_with_guard(&key, t as u64, &guard).is_ok() {
                        success_count.fetch_add(1, Ordering::Relaxed);
                    }
                }
            })
        })
        .collect();

    for h in handles {
        h.join().unwrap();
    }

    let actual_len = tree.len();
    tracing::info!(
        expected = NUM_KEYS,
        actual = actual_len,
        success_count = success_count.load(Ordering::Relaxed),
        "All threads joined"
    );

    // Verify all keys are present and find missing ones
    let guard = tree.guard();
    let mut missing_keys = Vec::new();
    for i in 0..NUM_KEYS {
        let key = (i as u64).to_be_bytes();
        if tree.get_with_guard(&key, &guard).is_none() {
            missing_keys.push(i);
            tracing::error!(key = i, "Key missing after concurrent insert");
        }
    }

    if !missing_keys.is_empty() {
        tracing::error!(
            missing_count = missing_keys.len(),
            missing_keys = ?missing_keys,
            tree_len = actual_len,
            "FAILURE: Keys missing after concurrent insert"
        );
        panic!(
            "Missing {} keys: {:?} (tree.len()={}, expected={})",
            missing_keys.len(),
            missing_keys,
            actual_len,
            NUM_KEYS
        );
    }

    assert_eq!(actual_len, NUM_KEYS);
    tracing::info!("Test passed");
}

#[test]
fn concurrent_insert_4_threads_overlapping_keys() {
    common::init_tracing();

    const NUM_KEYS: usize = 50;
    const NUM_THREADS: usize = 4;

    tracing::info!(
        threads = NUM_THREADS,
        keys = NUM_KEYS,
        "Starting 4-thread overlapping keys test"
    );

    let tree = Arc::new(MassTree::<u64>::new());

    let handles: Vec<_> = (0..NUM_THREADS)
        .map(|t| {
            let tree = Arc::clone(&tree);
            thread::spawn(move || {
                let guard = tree.guard();
                for i in 0..NUM_KEYS {
                    let key = (i as u64).to_be_bytes();
                    let _ = tree.insert_with_guard(&key, t as u64, &guard);
                }
            })
        })
        .collect();

    for h in handles {
        h.join().unwrap();
    }

    let actual_len = tree.len();
    tracing::info!(expected = NUM_KEYS, actual = actual_len, "All threads joined");

    // Verify all keys are present
    let guard = tree.guard();
    let mut missing_keys = Vec::new();
    for i in 0..NUM_KEYS {
        let key = (i as u64).to_be_bytes();
        if tree.get_with_guard(&key, &guard).is_none() {
            missing_keys.push(i);
            tracing::error!(key = i, "Key missing after concurrent insert");
        }
    }

    if !missing_keys.is_empty() {
        tracing::error!(
            missing_count = missing_keys.len(),
            missing_keys = ?missing_keys,
            tree_len = actual_len,
            "FAILURE: Keys missing after 4-thread concurrent insert"
        );
        panic!(
            "Missing {} keys: {:?} (tree.len()={}, expected={})",
            missing_keys.len(),
            missing_keys,
            actual_len,
            NUM_KEYS
        );
    }

    assert_eq!(actual_len, NUM_KEYS);
    tracing::info!("Test passed");
}

// =============================================================================
// Concurrent Read + Write
// =============================================================================

#[test]
fn concurrent_read_write_2_threads() {
    common::init_tracing();

    const PRE_POPULATED: usize = 100;
    const WRITER_KEYS: usize = 100;
    const TOTAL_KEYS: usize = PRE_POPULATED + WRITER_KEYS;

    tracing::info!(
        pre_populated = PRE_POPULATED,
        writer_keys = WRITER_KEYS,
        "Starting concurrent read/write test"
    );

    let tree = Arc::new(MassTree::<u64>::new());

    // Pre-populate
    {
        let guard = tree.guard();
        for i in 0..PRE_POPULATED {
            let key = (i as u64).to_be_bytes();
            let _ = tree.insert_with_guard(&key, i as u64, &guard);
        }
    }

    tracing::debug!(len = tree.len(), "Pre-population complete");

    let tree1 = Arc::clone(&tree);
    let writer = thread::spawn(move || {
        let guard = tree1.guard();
        for i in PRE_POPULATED..(PRE_POPULATED + WRITER_KEYS) {
            let key = (i as u64).to_be_bytes();
            let _ = tree1.insert_with_guard(&key, i as u64, &guard);
        }
    });

    let tree2 = Arc::clone(&tree);
    let reader = thread::spawn(move || {
        let guard = tree2.guard();
        let mut found = 0;
        let mut missing = Vec::new();
        for i in 0..PRE_POPULATED {
            let key = (i as u64).to_be_bytes();
            if tree2.get_with_guard(&key, &guard).is_some() {
                found += 1;
            } else {
                missing.push(i);
            }
        }
        (found, missing)
    });

    writer.join().unwrap();
    let (found, missing) = reader.join().unwrap();

    let actual_len = tree.len();
    tracing::info!(
        found,
        expected_found = PRE_POPULATED,
        tree_len = actual_len,
        expected_len = TOTAL_KEYS,
        "Threads joined"
    );

    // Verify all keys are present after both threads complete
    let guard = tree.guard();
    let mut final_missing = Vec::new();
    for i in 0..TOTAL_KEYS {
        let key = (i as u64).to_be_bytes();
        if tree.get_with_guard(&key, &guard).is_none() {
            final_missing.push(i);
            tracing::error!(key = i, "Key missing after concurrent read/write");
        }
    }

    if !final_missing.is_empty() {
        tracing::error!(
            missing_count = final_missing.len(),
            missing_keys = ?final_missing,
            reader_missing = ?missing,
            tree_len = actual_len,
            "FAILURE: Keys missing after concurrent read/write"
        );
        panic!(
            "Missing {} keys: {:?} (tree.len()={}, expected={})",
            final_missing.len(),
            final_missing,
            actual_len,
            TOTAL_KEYS
        );
    }

    // Reader should find all pre-populated keys
    if found != PRE_POPULATED {
        tracing::error!(
            found,
            expected = PRE_POPULATED,
            missing_during_read = ?missing,
            "Reader didn't find all pre-populated keys"
        );
    }
    assert_eq!(found, PRE_POPULATED);
    assert_eq!(actual_len, TOTAL_KEYS);
    tracing::info!("Test passed");
}

#[test]
fn concurrent_read_write_4_threads() {
    let tree = Arc::new(MassTree::<u64>::new());

    // Pre-populate
    {
        let guard = tree.guard();
        for i in 0..1000 {
            let key = (i as u64).to_be_bytes();
            let _ = tree.insert_with_guard(&key, i as u64, &guard);
        }
    }

    let handles: Vec<_> = (0..4)
        .map(|t| {
            let tree = Arc::clone(&tree);
            thread::spawn(move || {
                let guard = tree.guard();
                if t % 2 == 0 {
                    // Writer
                    for i in 0..100 {
                        let key = ((1000 + t * 100 + i) as u64).to_be_bytes();
                        let _ = tree.insert_with_guard(&key, i as u64, &guard);
                    }
                } else {
                    // Reader
                    for i in 0..1000 {
                        let key = (i as u64).to_be_bytes();
                        let _ = tree.get_with_guard(&key, &guard);
                    }
                }
            })
        })
        .collect();

    for h in handles {
        h.join().unwrap();
    }

    // 1000 pre-populated + 2 writers * 100 each
    assert_eq!(tree.len(), 1200);
}

// =============================================================================
// Stress Tests
// =============================================================================

#[test]
fn stress_concurrent_insert_many_keys() {
    // Initialize tracing (safe to call multiple times)
    common::init_tracing();

    // Reset debug counters at start
    reset_debug_counters();

    const NUM_THREADS: usize = 4;
    const KEYS_PER_THREAD: usize = 500;
    const TOTAL_KEYS: usize = NUM_THREADS * KEYS_PER_THREAD;

    let tree = Arc::new(MassTree::<u64>::new());

    tracing::info!(
        threads = NUM_THREADS,
        keys_per_thread = KEYS_PER_THREAD,
        total_keys = TOTAL_KEYS,
        "Starting stress test"
    );

    let insert_failures = Arc::new(AtomicUsize::new(0));
    let verify_failures = Arc::new(AtomicUsize::new(0));

    let handles: Vec<_> = (0..NUM_THREADS)
        .map(|t| {
            let tree = Arc::clone(&tree);
            let insert_failures = Arc::clone(&insert_failures);
            let verify_failures = Arc::clone(&verify_failures);
            thread::spawn(move || {
                let guard = tree.guard();
                tracing::debug!(thread_id = t, "Thread starting inserts");
                for i in 0..KEYS_PER_THREAD {
                    let key_val = t * 10000 + i;
                    let key = (key_val as u64).to_be_bytes();

                    // Insert
                    let result = tree.insert_with_guard(&key, i as u64, &guard);
                    if result.is_err() {
                        insert_failures.fetch_add(1, Ordering::Relaxed);
                        tracing::error!(
                            key = key_val,
                            thread = t,
                            error = ?result.unwrap_err(),
                            "Insert failed"
                        );
                    }

                    // Immediate verification: can we read back what we just inserted?
                    if tree.get_with_guard(&key, &guard).is_none() {
                        verify_failures.fetch_add(1, Ordering::Relaxed);
                        tracing::error!(
                            key = key_val,
                            thread = t,
                            tree_len = tree.len(),
                            "CRITICAL: Key not found immediately after insert"
                        );
                    }
                }
                tracing::debug!(thread_id = t, "Thread completed inserts");
            })
        })
        .collect();

    for h in handles {
        h.join().unwrap();
    }

    let actual_len = tree.len();
    let insert_fail_count = insert_failures.load(Ordering::Relaxed);
    let verify_fail_count = verify_failures.load(Ordering::Relaxed);

    tracing::info!(
        expected = TOTAL_KEYS,
        actual = actual_len,
        insert_failures = insert_fail_count,
        immediate_verify_failures = verify_fail_count,
        "All threads joined, verifying keys"
    );

    // If there were immediate verification failures, that's the smoking gun
    if verify_fail_count > 0 {
        tracing::error!(
            verify_fail_count,
            "Keys not findable immediately after insert - race condition in insert/get"
        );
    }

    // Verify all keys are readable and collect missing ones
    let guard = tree.guard();
    let mut missing_keys = Vec::new();
    for t in 0..NUM_THREADS {
        for i in 0..KEYS_PER_THREAD {
            let key_val = t * 10000 + i;
            let key = (key_val as u64).to_be_bytes();
            if tree.get_with_guard(&key, &guard).is_none() {
                missing_keys.push(key_val);
                tracing::error!(
                    key = key_val,
                    thread = t,
                    index = i,
                    "Key not found after stress test"
                );
            }
        }
    }

    if !missing_keys.is_empty() {
        tracing::error!(
            missing_count = missing_keys.len(),
            missing_keys = ?missing_keys,
            tree_len = actual_len,
            expected = TOTAL_KEYS,
            "FAILURE: Keys missing after stress test"
        );
        panic!(
            "Missing {} keys: {:?} (tree.len()={}, expected={})",
            missing_keys.len(),
            missing_keys,
            actual_len,
            TOTAL_KEYS
        );
    }

    assert_eq!(actual_len, TOTAL_KEYS);

    // Report debug counters - this is the key diagnostic for P0.6-B
    let (blink_should_follow, search_not_found) = get_debug_counters();
    if blink_should_follow > 0 {
        eprintln!(
            "\n*** P0.6-B DIAGNOSTIC ***\n\
             B-link should have been followed: {} times\n\
             Total search NotFound: {} times\n\
             This indicates stale routing after splits.\n",
            blink_should_follow, search_not_found
        );
    }

    tracing::info!(
        blink_should_follow = blink_should_follow,
        search_not_found = search_not_found,
        "Stress test passed"
    );
}

#[test]
fn stress_concurrent_insert_sequential_keys() {
    // Sequential keys cause more splits
    let tree = Arc::new(MassTree::<u64>::new());

    let handles: Vec<_> = (0..4)
        .map(|t| {
            let tree = Arc::clone(&tree);
            thread::spawn(move || {
                let guard = tree.guard();
                // Interleaved sequential: 0,4,8,12... / 1,5,9,13... etc
                for i in 0..250 {
                    let key = ((i * 4 + t) as u64).to_be_bytes();
                    let _ = tree.insert_with_guard(&key, i as u64, &guard);
                }
            })
        })
        .collect();

    for h in handles {
        h.join().unwrap();
    }

    assert_eq!(tree.len(), 1000);
}

#[test]
fn stress_repeated_runs() {
    // Run multiple times to catch intermittent issues
    for run in 0..10 {
        let tree = Arc::new(MassTree::<u64>::new());

        let handles: Vec<_> = (0..4)
            .map(|t| {
                let tree = Arc::clone(&tree);
                thread::spawn(move || {
                    let guard = tree.guard();
                    for i in 0..100 {
                        let key = ((t * 1000 + i) as u64).to_be_bytes();
                        let _ = tree.insert_with_guard(&key, i as u64, &guard);
                    }
                })
            })
            .collect();

        for h in handles {
            h.join().unwrap();
        }

        assert_eq!(tree.len(), 400, "Failed on run {}", run);
    }
}

// =============================================================================
// Split-Heavy Tests (trigger many splits)
// =============================================================================

#[test]
fn concurrent_insert_triggers_splits() {
    // Insert enough keys to trigger multiple splits
    let tree = Arc::new(MassTree::<u64>::new());

    let handles: Vec<_> = (0..2)
        .map(|t| {
            let tree = Arc::clone(&tree);
            thread::spawn(move || {
                let guard = tree.guard();
                // Each thread inserts 200 keys, enough to trigger splits
                for i in 0..200 {
                    let key = ((t * 10000 + i) as u64).to_be_bytes();
                    let _ = tree.insert_with_guard(&key, i as u64, &guard);
                }
            })
        })
        .collect();

    for h in handles {
        h.join().unwrap();
    }

    assert_eq!(tree.len(), 400);

    // Verify all keys
    let guard = tree.guard();
    for t in 0..2 {
        for i in 0..200 {
            let key = ((t * 10000 + i) as u64).to_be_bytes();
            assert!(tree.get_with_guard(&key, &guard).is_some());
        }
    }
}

#[test]
fn concurrent_insert_adjacent_keys_same_leaf() {
    // Keys that likely land in the same leaf
    let tree = Arc::new(MassTree::<u64>::new());

    let handles: Vec<_> = (0..4)
        .map(|t| {
            let tree = Arc::clone(&tree);
            thread::spawn(move || {
                let guard = tree.guard();
                // All threads insert to same key range (0-99)
                // This maximizes contention on the same leaves
                for i in 0..25 {
                    let key = ((t * 25 + i) as u64).to_be_bytes();
                    let _ = tree.insert_with_guard(&key, (t * 25 + i) as u64, &guard);
                }
            })
        })
        .collect();

    for h in handles {
        h.join().unwrap();
    }

    assert_eq!(tree.len(), 100);
}

// =============================================================================
// Long Key Tests (multi-layer)
// =============================================================================

#[test]
fn concurrent_insert_long_keys() {
    let tree = Arc::new(MassTree::<u64>::new());

    let handles: Vec<_> = (0..4)
        .map(|t| {
            let tree = Arc::clone(&tree);
            thread::spawn(move || {
                let guard = tree.guard();
                for i in 0..50 {
                    // 24-byte key spanning multiple layers
                    let key = format!("thread{:02}_key_{:08}", t, i);
                    let _ = tree.insert_with_guard(key.as_bytes(), i as u64, &guard);
                }
            })
        })
        .collect();

    for h in handles {
        h.join().unwrap();
    }

    assert_eq!(tree.len(), 200);
}

// =============================================================================
// Verification Tests
// =============================================================================

#[test]
fn concurrent_insert_all_values_correct() {
    let tree = Arc::new(MassTree::<u64>::new());

    let handles: Vec<_> = (0..4)
        .map(|t| {
            let tree = Arc::clone(&tree);
            thread::spawn(move || {
                let guard = tree.guard();
                for i in 0..100 {
                    let key = ((t * 1000 + i) as u64).to_be_bytes();
                    let value = (t * 1000 + i) as u64;
                    let _ = tree.insert_with_guard(&key, value, &guard);
                }
            })
        })
        .collect();

    for h in handles {
        h.join().unwrap();
    }

    // Verify all values are correct
    let guard = tree.guard();
    for t in 0..4usize {
        for i in 0..100usize {
            let key = ((t * 1000 + i) as u64).to_be_bytes();
            let expected = (t * 1000 + i) as u64;
            let actual = tree.get_with_guard(&key, &guard);
            assert_eq!(
                actual.map(|v| *v),
                Some(expected),
                "Wrong value for key {}",
                t * 1000 + i
            );
        }
    }
}

#[test]
fn concurrent_insert_no_duplicates() {
    let tree = Arc::new(MassTree::<u64>::new());

    let handles: Vec<_> = (0..4)
        .map(|t| {
            let tree = Arc::clone(&tree);
            thread::spawn(move || {
                let guard = tree.guard();
                for i in 0..100 {
                    let key = ((t * 1000 + i) as u64).to_be_bytes();
                    let _ = tree.insert_with_guard(&key, i as u64, &guard);
                }
            })
        })
        .collect();

    for h in handles {
        h.join().unwrap();
    }

    // Collect all keys and check for duplicates
    let guard = tree.guard();
    let mut seen = HashSet::new();
    for t in 0..4 {
        for i in 0..100 {
            let key = (t * 1000 + i) as u64;
            if tree.get_with_guard(&key.to_be_bytes(), &guard).is_some() {
                assert!(seen.insert(key), "Duplicate key found: {}", key);
            }
        }
    }

    assert_eq!(seen.len(), 400);
}

// =============================================================================
// Edge Cases
// =============================================================================

#[test]
fn concurrent_insert_empty_then_read() {
    let tree = Arc::new(MassTree::<u64>::new());

    let tree1 = Arc::clone(&tree);
    let writer = thread::spawn(move || {
        let guard = tree1.guard();
        for i in 0..100 {
            let key = (i as u64).to_be_bytes();
            let _ = tree1.insert_with_guard(&key, i as u64, &guard);
        }
    });

    // Reader starts immediately on empty tree
    let tree2 = Arc::clone(&tree);
    let reader = thread::spawn(move || {
        let guard = tree2.guard();
        let mut attempts = 0;
        let mut found = 0;
        while attempts < 1000 {
            for i in 0..100 {
                let key = (i as u64).to_be_bytes();
                if tree2.get_with_guard(&key, &guard).is_some() {
                    found += 1;
                }
            }
            attempts += 1;
            if found >= 100 {
                break;
            }
        }
        found
    });

    writer.join().unwrap();
    reader.join().unwrap();

    assert_eq!(tree.len(), 100);
}

#[test]
fn single_key_concurrent_updates() {
    // Multiple threads updating the same key
    let tree = Arc::new(MassTree::<u64>::new());
    let update_count = Arc::new(AtomicUsize::new(0));

    // Insert initial value
    {
        let guard = tree.guard();
        let _ = tree.insert_with_guard(&[0u8; 8], 0, &guard);
    }

    let handles: Vec<_> = (0..8)
        .map(|t| {
            let tree = Arc::clone(&tree);
            let update_count = Arc::clone(&update_count);
            thread::spawn(move || {
                let guard = tree.guard();
                for _ in 0..100 {
                    let _ = tree.insert_with_guard(&[0u8; 8], t as u64, &guard);
                    update_count.fetch_add(1, Ordering::Relaxed);
                }
            })
        })
        .collect();

    for h in handles {
        h.join().unwrap();
    }

    // Should still be exactly 1 key
    assert_eq!(tree.len(), 1);
    // All updates should have completed
    assert_eq!(update_count.load(Ordering::Relaxed), 800);
}
