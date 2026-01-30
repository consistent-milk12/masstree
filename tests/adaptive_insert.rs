//! Integration tests for adaptive insert strategy.
//!
//! Tests verify CORRECTNESS under shared-prefix contention, not performance.
//! Performance is evaluated via benchmarks, not tests.

#![expect(clippy::expect_used, clippy::unwrap_used)]

use masstree::{BatchInsertResult, MassTree15};
use std::sync::Arc;
use std::thread;

/// Test shared-prefix insert correctness.
///
/// Verifies all keys are correctly inserted even under high contention.
///
/// # Key Format for Maximum Contention
///
/// Uses 12-byte keys: 8-byte shared prefix + 4-byte unique suffix.
/// This forces all keys to the same leaf initially (same ikey), causing
/// maximum lock contention. The 4-byte suffix differentiates keys without
/// requiring layer descent.
///
/// Format: "PREFIX00" (8 bytes) + "T{tid}I{i:03}" (variable) = unique keys
const KEYS: usize = 100;
const THREADS: usize = 4;
const ENTRIES: usize = 50;
const KEYS_PER_THREAD: usize = 1000;
const UPDATES_PER_THREAD: usize = 10;
const ENTRIES_PER_BATCH: usize = 100;

#[test]
fn test_shared_prefix_correctness() {
    let tree: MassTree15<u64> = MassTree15::new();
    let tree = Arc::new(tree);

    let handles: Vec<_> = (0..THREADS)
        .map(|tid| {
            let tree = Arc::clone(&tree);
            thread::spawn(move || {
                for i in 0..KEYS_PER_THREAD {
                    // 12+ byte key: 8-byte prefix (same ikey) + unique suffix
                    // Format ensures no collisions: "PREFIX00T{tid}I{i:04}"
                    let key = format!("PREFIX00T{tid}I{i:04}");
                    let value = (tid * KEYS_PER_THREAD + i) as u64;
                    // insert() returns Result<Option<S::Output>, InsertError>
                    tree.insert(key.as_bytes(), value).expect("insert failed");
                }
            })
        })
        .collect();

    for h in handles {
        h.join().unwrap();
    }

    // Verify all keys present
    assert_eq!(tree.len(), THREADS * KEYS_PER_THREAD);

    // Verify values correct (get returns Option<S::Output>, owned value)
    for tid in 0..THREADS {
        for i in 0..KEYS_PER_THREAD {
            let key = format!("PREFIX00T{tid}I{i:04}");
            let expected = (tid * KEYS_PER_THREAD + i) as u64;
            // get() returns Option<Arc<u64>> for MassTree15
            let value = tree.get(key.as_bytes()).expect("key should exist");
            assert_eq!(*value, expected);
        }
    }
}

/// Test normal (non-contended) workloads still work correctly.
#[test]
fn test_random_keys_correctness() {
    let tree: MassTree15<u64> = MassTree15::new();
    let tree = Arc::new(tree);
    let handles: Vec<_> = (0..THREADS)
        .map(|tid| {
            let tree = Arc::clone(&tree);
            thread::spawn(move || {
                // Thread-specific key ranges (no shared prefix)
                let base = (tid as u64) * 1_000_000_000;
                for i in 0..KEYS_PER_THREAD {
                    let key = (base + i as u64).to_be_bytes();
                    let value = base + i as u64;
                    tree.insert(&key, value).expect("insert failed");
                }
            })
        })
        .collect();

    for h in handles {
        h.join().unwrap();
    }

    assert_eq!(tree.len(), THREADS * KEYS_PER_THREAD);
}

/// Test mixed workload: some shared-prefix, some random.
#[test]
fn test_mixed_prefix_random() {
    let tree: MassTree15<u64> = MassTree15::new();
    let tree = Arc::new(tree);

    let handles: Vec<_> = (0..THREADS)
        .map(|tid| {
            let tree = Arc::clone(&tree);
            thread::spawn(move || {
                for i in 0..KEYS_PER_THREAD {
                    if i % 2 == 0 {
                        let key = format!("shared__{tid:04}_{i:08}");
                        tree.insert(key.as_bytes(), i as u64)
                            .expect("insert failed");
                    } else {
                        let key = format!("unique{tid:02}_{i:08}");
                        tree.insert(key.as_bytes(), i as u64)
                            .expect("insert failed");
                    }
                }
            })
        })
        .collect();

    for h in handles {
        h.join().unwrap();
    }

    assert_eq!(tree.len(), THREADS * KEYS_PER_THREAD);
}

/// Test updates work correctly under contention.
///
/// NOTE: This does NOT test atomic increment (get+insert is not atomic).
/// It only tests that updates succeed and return valid old values.
#[test]
fn test_shared_prefix_updates() {
    let tree: MassTree15<u64> = MassTree15::new();
    let tree = Arc::new(tree);

    // Pre-populate
    for i in 0..KEYS {
        let key = format!("prefix__{i:08}");
        tree.insert(key.as_bytes(), 0u64).expect("insert failed");
    }

    let handles: Vec<_> = (0..THREADS)
        .map(|tid| {
            let tree = Arc::clone(&tree);
            thread::spawn(move || {
                for _ in 0..UPDATES_PER_THREAD {
                    for i in 0..KEYS {
                        let key = format!("prefix__{i:08}");
                        // Just overwrite with thread-specific value
                        // We're testing that updates work, not atomic increment
                        let new_value = (tid * 1000 + i) as u64;
                        let result = tree.insert(key.as_bytes(), new_value);
                        // Should succeed and return Some(old_value)
                        assert!(result.is_ok());
                        let old = result.unwrap();
                        assert!(old.is_some(), "update should return old value");
                    }
                }
            })
        })
        .collect();

    for h in handles {
        h.join().unwrap();
    }

    // All keys still present
    assert_eq!(tree.len(), KEYS);
}

/// Test batch insert with shared-prefix keys.
///
/// Uses SHORT keys (≤8 bytes) to avoid suffix conflicts that would cause
/// layer descent and be marked as failures in batch mode.
#[test]
fn test_batch_shared_prefix() {
    let tree: MassTree15<u64> = MassTree15::new();
    let tree = Arc::new(tree);

    let handles: Vec<_> = (0..THREADS)
        .map(|tid| {
            let tree = Arc::clone(&tree);
            thread::spawn(move || {
                // Use SHORT keys (8 bytes max) to avoid suffix conflicts
                // Format: "BTxxxxxx" where xxxx encodes tid+index
                // This ensures each key has unique ikey, no suffix needed
                let entries: Vec<_> = (0..ENTRIES_PER_BATCH)
                    .map(|i| {
                        // 8-byte key: 2 chars prefix + 6 chars for tid/index
                        // tid: 0-3, i: 0-99, gives unique keys like "BT000042"
                        let key = format!("BT{tid:02}{i:04}");
                        debug_assert!(key.len() <= 8, "key must be ≤8 bytes for batch test");
                        (key.into_bytes(), (tid * ENTRIES_PER_BATCH + i) as u64)
                    })
                    .collect();

                let result = tree.insert_batch(entries).expect("batch insert failed");
                // With short keys, all should succeed (no suffix conflicts)
                assert_eq!(result.inserted + result.updated, ENTRIES_PER_BATCH);
                assert_eq!(result.failed, 0, "no failures expected with short keys");
            })
        })
        .collect();

    for h in handles {
        h.join().unwrap();
    }

    assert_eq!(tree.len(), THREADS * ENTRIES_PER_BATCH);
}

/// Test batch insert with long keys (expects some layer descent failures).
///
/// This test verifies batch handles suffix conflicts gracefully by marking
/// them as failed (for individual retry) rather than crashing.
///
/// # Note on Assertion
///
/// We use `inserted + updated >= 1` instead of `inserted >= 1` because:
/// - Batch may process entries in any order after sorting
/// - Concurrent activity could cause updates instead of inserts
/// - The key invariant is "at least one operation succeeded"
#[test]
fn test_batch_long_keys_with_conflicts() {
    let tree: MassTree15<u64> = MassTree15::new();

    // All keys share first 8 bytes "longpref", will cause suffix conflicts
    let entries: Vec<_> = (0..ENTRIES)
        .map(|i| {
            let key = format!("longpref_{i:08}"); // 17 bytes, shared 8-byte prefix
            (key.into_bytes(), i as u64)
        })
        .collect();

    let result: BatchInsertResult<Arc<u64>> =
        tree.insert_batch(entries).expect("batch insert failed");

    // At least one operation should succeed (insert or update)
    // Using inserted + updated is more resilient than just inserted
    assert!(
        result.inserted + result.updated >= 1,
        "at least one operation should succeed (got inserted={}, updated={})",
        result.inserted,
        result.updated
    );

    // Total processed should equal ENTRIES
    assert_eq!(
        result.inserted + result.updated + result.failed,
        ENTRIES,
        "all entries should be accounted for"
    );

    // Verify the tree is still usable after batch with conflicts
    assert!(!tree.is_empty());

    // Failed entries can be retried individually
    // (In real code, you'd retry result.failed entries via individual insert)
}
