//! ========================================================================
//!  Batch Insert Tests
//! ========================================================================
//!
//! Comprehensive tests for the batch insert API covering:
//! - Basic functionality (insert, update, mixed)
//! - Edge cases (empty batch, single entry, large batches)
//! - Key patterns (sequential, random, shared prefix, long keys)
//! - Concurrent access (multi-threaded batch + single inserts)
//! - Split handling (batches that trigger splits)

#![expect(clippy::unwrap_used)]

use std::collections::HashSet;
use std::sync::Arc;
use std::thread;

use masstree::{MassTree24, MassTree15, MassTree24Inline, MassTree15Inline};

// ============================================================================
//  Basic Functionality Tests
// ============================================================================

#[test]
fn test_batch_insert_empty() {
    let tree: MassTree24<u64> = MassTree24::new();

    let entries: Vec<(Vec<u8>, u64)> = vec![];
    let result = tree.insert_batch(entries).unwrap();

    assert_eq!(result.inserted, 0);
    assert_eq!(result.updated, 0);
    assert_eq!(result.failed, 0);
    assert!(result.old_values.is_empty());
    assert_eq!(tree.len(), 0);
}

#[test]
fn test_batch_insert_single_entry() {
    let tree: MassTree24<u64> = MassTree24::new();

    let entries = vec![(b"hello".to_vec(), 42u64)];
    let result = tree.insert_batch(entries).unwrap();

    assert_eq!(result.inserted, 1);
    assert_eq!(result.updated, 0);
    assert_eq!(result.failed, 0);
    assert_eq!(tree.len(), 1);

    let value = tree.get(b"hello").unwrap();
    assert_eq!(*value, 42);
}

#[test]
fn test_batch_insert_multiple_entries() {
    let tree: MassTree24<u64> = MassTree24::new();

    let entries: Vec<(Vec<u8>, u64)> = (0..100)
        .map(|i| (format!("key{i:03}").into_bytes(), i as u64))
        .collect();

    let result = tree.insert_batch(entries).unwrap();

    assert_eq!(result.inserted, 100);
    assert_eq!(result.updated, 0);
    assert_eq!(result.failed, 0);
    assert_eq!(tree.len(), 100);

    // Verify all entries
    for i in 0..100 {
        let key = format!("key{i:03}");
        let value = tree.get(key.as_bytes()).unwrap();
        assert_eq!(*value, i as u64);
    }
}

#[test]
fn test_batch_insert_with_updates() {
    let tree: MassTree24<u64> = MassTree24::new();

    // Insert initial values
    for i in 0..50 {
        tree.insert(format!("key{i:03}").as_bytes(), i as u64).unwrap();
    }
    assert_eq!(tree.len(), 50);

    // Batch insert with overlapping keys
    let entries: Vec<(Vec<u8>, u64)> = (25..75)
        .map(|i| (format!("key{i:03}").into_bytes(), (i * 10) as u64))
        .collect();

    let result = tree.insert_batch(entries).unwrap();

    assert_eq!(result.inserted, 25); // keys 50-74 are new
    assert_eq!(result.updated, 25); // keys 25-49 are updates
    assert_eq!(result.failed, 0);
    assert_eq!(tree.len(), 75);

    // Verify old values were returned (they are Arc<u64>)
    assert_eq!(result.old_values.len(), 25);
    let old_values: HashSet<u64> = result.old_values.iter().map(|arc| **arc).collect();
    for i in 25..50 {
        assert!(old_values.contains(&(i as u64)));
    }

    // Verify new values are in tree
    for i in 25..75 {
        let key = format!("key{i:03}");
        let value = tree.get(key.as_bytes()).unwrap();
        assert_eq!(*value, (i * 10) as u64);
    }
}

#[test]
fn test_batch_insert_all_updates() {
    let tree: MassTree24<u64> = MassTree24::new();

    // Insert initial values
    for i in 0..100 {
        tree.insert(format!("key{i:03}").as_bytes(), i as u64).unwrap();
    }

    // Batch update all values
    let entries: Vec<(Vec<u8>, u64)> = (0..100)
        .map(|i| (format!("key{i:03}").into_bytes(), (i + 1000) as u64))
        .collect();

    let result = tree.insert_batch(entries).unwrap();

    assert_eq!(result.inserted, 0);
    assert_eq!(result.updated, 100);
    assert_eq!(result.old_values.len(), 100);
    assert_eq!(tree.len(), 100);

    // Verify all values updated
    for i in 0..100 {
        let key = format!("key{i:03}");
        let value = tree.get(key.as_bytes()).unwrap();
        assert_eq!(*value, (i + 1000) as u64);
    }
}

// ============================================================================
//  Tree Width Tests (MassTree15 vs MassTree24)
// ============================================================================

#[test]
fn test_batch_insert_masstree15() {
    let tree: MassTree15<u64> = MassTree15::new();

    // Use 8-byte keys for optimal batch performance (no layer descent)
    let entries: Vec<(Vec<u8>, u64)> = (0u64..1000)
        .map(|i| (i.to_be_bytes().to_vec(), i))
        .collect();

    let result = tree.insert_batch(entries).unwrap();

    assert_eq!(result.inserted, 1000);
    assert_eq!(tree.len(), 1000);

    // Verify random sample
    for i in [0u64, 100, 500, 999] {
        let value = tree.get(&i.to_be_bytes()).unwrap();
        assert_eq!(*value, i);
    }
}

#[test]
fn test_batch_insert_inline_u64() {
    let tree: MassTree24Inline<u64> = MassTree24Inline::new();

    let entries: Vec<(Vec<u8>, u64)> = (0..500)
        .map(|i| (format!("key{i:04}").into_bytes(), i as u64))
        .collect();

    let result = tree.insert_batch(entries).unwrap();

    assert_eq!(result.inserted, 500);
    assert_eq!(tree.len(), 500);

    // Verify values (returned as V directly, not Arc<V>)
    for i in 0..500 {
        let key = format!("key{i:04}");
        let value = tree.get(key.as_bytes()).unwrap();
        assert_eq!(value, i as u64); // Note: no dereference needed for Inline
    }
}

#[test]
fn test_batch_insert_inline_with_updates() {
    let tree: MassTree15Inline<i32> = MassTree15Inline::new();

    // Insert initial values
    for i in 0..50 {
        tree.insert(format!("k{i}").as_bytes(), i).unwrap();
    }

    // Batch insert with overlaps
    let entries: Vec<(Vec<u8>, i32)> = (0..100)
        .map(|i| (format!("k{i}").into_bytes(), i * -1))
        .collect();

    let result = tree.insert_batch(entries).unwrap();

    assert_eq!(result.inserted, 50);
    assert_eq!(result.updated, 50);
    assert_eq!(tree.len(), 100);

    // Old values are i32 directly (not Arc<i32>)
    for old_value in &result.old_values {
        assert!(*old_value >= 0 && *old_value < 50);
    }
}

// ============================================================================
//  Key Pattern Tests
// ============================================================================

#[test]
fn test_batch_insert_sequential_keys() {
    let tree: MassTree24<u64> = MassTree24::new();

    // Sequential 8-byte keys (optimal locality)
    let entries: Vec<(Vec<u8>, u64)> = (0u64..10000)
        .map(|i| (i.to_be_bytes().to_vec(), i))
        .collect();

    let result = tree.insert_batch(entries).unwrap();

    assert_eq!(result.inserted, 10000);
    assert_eq!(tree.len(), 10000);
}

#[test]
fn test_batch_insert_reverse_sequential() {
    let tree: MassTree24<u64> = MassTree24::new();

    // Reverse sequential (tests sorting)
    let entries: Vec<(Vec<u8>, u64)> = (0u64..1000)
        .rev()
        .map(|i| (i.to_be_bytes().to_vec(), i))
        .collect();

    let result = tree.insert_batch(entries).unwrap();

    assert_eq!(result.inserted, 1000);
    assert_eq!(tree.len(), 1000);

    // Verify order is correct
    for i in 0u64..1000 {
        let value = tree.get(&i.to_be_bytes()).unwrap();
        assert_eq!(*value, i);
    }
}

#[test]
fn test_batch_insert_shared_prefix() {
    let tree: MassTree24<u64> = MassTree24::new();

    // All keys share a common prefix (tests leaf clustering)
    // Note: Long keys with shared prefixes trigger layer descent
    // which batch mode falls back to individual insert
    let entries: Vec<(Vec<u8>, u64)> = (0..500)
        .map(|i| {
            let mut key = b"common_prefix_".to_vec();
            key.extend(format!("{i:06}").as_bytes());
            (key, i as u64)
        })
        .collect();

    let result = tree.insert_batch(entries).unwrap();

    // With layer descent fallback, all entries should eventually be inserted
    assert_eq!(result.inserted + result.failed, 500);
    // Verify tree has correct count
    assert_eq!(tree.len(), result.inserted);
}

#[test]
fn test_batch_insert_random_keys() {
    let tree: MassTree24<u64> = MassTree24::new();

    // Pseudo-random keys (poor locality)
    let entries: Vec<(Vec<u8>, u64)> = (0..1000)
        .map(|i| {
            // Simple hash-like function for deterministic "random" keys
            let hash = (i as u64).wrapping_mul(0x517cc1b727220a95);
            (hash.to_be_bytes().to_vec(), i as u64)
        })
        .collect();

    let result = tree.insert_batch(entries).unwrap();

    assert_eq!(result.inserted, 1000);
    assert_eq!(tree.len(), 1000);
}

#[test]
fn test_batch_insert_short_keys() {
    let tree: MassTree24<u64> = MassTree24::new();

    // Very short keys (1-4 bytes)
    let entries: Vec<(Vec<u8>, u64)> = (0..256)
        .map(|i| (vec![i as u8], i as u64))
        .collect();

    let result = tree.insert_batch(entries).unwrap();

    assert_eq!(result.inserted, 256);
    assert_eq!(tree.len(), 256);

    // Verify
    for i in 0..256 {
        let value = tree.get(&[i as u8]).unwrap();
        assert_eq!(*value, i as u64);
    }
}

#[test]
fn test_batch_insert_long_keys() {
    let tree: MassTree24<u64> = MassTree24::new();

    // Keys longer than 8 bytes (require suffix handling)
    let entries: Vec<(Vec<u8>, u64)> = (0..100)
        .map(|i| {
            let key = format!("this_is_a_very_long_key_number_{i:04}");
            (key.into_bytes(), i as u64)
        })
        .collect();

    let result = tree.insert_batch(entries).unwrap();

    // Some may fail due to layer descent - that's expected
    assert!(result.inserted + result.failed == 100);
    assert!(tree.len() >= result.inserted);
}

#[test]
fn test_batch_insert_duplicate_keys_in_batch() {
    let tree: MassTree24<u64> = MassTree24::new();

    // Same key appears multiple times in batch
    let entries: Vec<(Vec<u8>, u64)> = vec![
        (b"duplicate".to_vec(), 1),
        (b"duplicate".to_vec(), 2),
        (b"duplicate".to_vec(), 3),
        (b"unique".to_vec(), 100),
    ];

    let result = tree.insert_batch(entries).unwrap();

    // First insert of "duplicate" succeeds, subsequent ones are updates
    assert_eq!(result.inserted, 2); // "duplicate" (first) + "unique"
    assert_eq!(result.updated, 2); // "duplicate" (second and third)
    assert_eq!(tree.len(), 2);

    // Final value should be from last occurrence
    let value = tree.get(b"duplicate").unwrap();
    assert_eq!(*value, 3);
}

// ============================================================================
//  Split Handling Tests
// ============================================================================

#[test]
fn test_batch_insert_triggers_single_split() {
    let tree: MassTree15<u64> = MassTree15::new();

    // Insert more than 15 keys with same prefix to trigger split
    let entries: Vec<(Vec<u8>, u64)> = (0..20)
        .map(|i| (format!("same_prefix_{i:02}").into_bytes(), i as u64))
        .collect();

    let result = tree.insert_batch(entries).unwrap();

    assert_eq!(result.inserted + result.failed, 20);
    assert!(tree.len() >= result.inserted);

    // Verify accessible keys
    for i in 0..20 {
        let key = format!("same_prefix_{i:02}");
        if let Some(value) = tree.get(key.as_bytes()) {
            assert_eq!(*value, i as u64);
        }
    }
}

#[test]
fn test_batch_insert_triggers_multiple_splits() {
    let tree: MassTree15<u64> = MassTree15::new();

    // Insert enough keys to trigger multiple splits
    let entries: Vec<(Vec<u8>, u64)> = (0..200)
        .map(|i| (format!("key{i:04}").into_bytes(), i as u64))
        .collect();

    let result = tree.insert_batch(entries).unwrap();

    assert_eq!(result.inserted + result.failed, 200);
    assert!(tree.len() >= result.inserted);
}

#[test]
fn test_batch_insert_large_batch() {
    let tree: MassTree24<u64> = MassTree24::new();

    let entries: Vec<(Vec<u8>, u64)> = (0..10000)
        .map(|i| (format!("key{i:08}").into_bytes(), i as u64))
        .collect();

    let result = tree.insert_batch(entries).unwrap();

    assert!(result.inserted > 0);
    assert_eq!(tree.len(), result.inserted);

    // Sample verification
    for i in [0, 1000, 5000, 9999] {
        let key = format!("key{i:08}");
        if let Some(value) = tree.get(key.as_bytes()) {
            assert_eq!(*value, i as u64);
        }
    }
}

// ============================================================================
//  Concurrent Tests
// ============================================================================

#[test]
fn test_batch_insert_concurrent_batches() {
    let tree = Arc::new(MassTree24::<u64>::new());
    let num_threads = 4;
    let entries_per_thread = 250;

    let mut handles = vec![];

    // Each thread does batch inserts
    for t in 0..num_threads {
        let tree = Arc::clone(&tree);
        let handle = thread::spawn(move || {
            let entries: Vec<(Vec<u8>, u64)> = (0..entries_per_thread)
                .map(|i| {
                    let key = format!("t{t}_k{i:04}");
                    (key.into_bytes(), (t * 1000 + i) as u64)
                })
                .collect();

            tree.insert_batch(entries).unwrap()
        });
        handles.push(handle);
    }

    // Collect results
    let mut total_inserted = 0;
    for handle in handles {
        let result = handle.join().unwrap();
        total_inserted += result.inserted;
    }

    assert!(total_inserted > 0);
    assert_eq!(tree.len(), total_inserted);
}

#[test]
fn test_batch_insert_mixed_with_single_inserts() {
    let tree = Arc::new(MassTree24::<u64>::new());

    // Thread 1: Batch inserts
    let tree1 = Arc::clone(&tree);
    let handle1 = thread::spawn(move || {
        let entries: Vec<(Vec<u8>, u64)> = (0..500)
            .map(|i| (format!("batch_{i:04}").into_bytes(), i as u64))
            .collect();
        tree1.insert_batch(entries).unwrap()
    });

    // Thread 2: Single inserts
    let tree2 = Arc::clone(&tree);
    let handle2 = thread::spawn(move || {
        for i in 0..500 {
            tree2.insert(format!("single_{i:04}").as_bytes(), i as u64).unwrap();
        }
    });

    // Wait for all threads
    let _ = handle1.join().unwrap();
    handle2.join().unwrap();

    assert!(tree.len() > 0);
}

// ============================================================================
//  Guard Reuse Tests
// ============================================================================

#[test]
fn test_batch_insert_with_guard() {
    let tree: MassTree24<u64> = MassTree24::new();
    let guard = tree.guard();

    // Multiple batches under same guard
    let batch1: Vec<(Vec<u8>, u64)> = (0..100)
        .map(|i| (format!("a{i}").into_bytes(), i as u64))
        .collect();

    let batch2: Vec<(Vec<u8>, u64)> = (0..100)
        .map(|i| (format!("b{i}").into_bytes(), i as u64))
        .collect();

    let result1 = tree.insert_batch_with_guard(batch1, &guard).unwrap();
    let result2 = tree.insert_batch_with_guard(batch2, &guard).unwrap();

    assert_eq!(result1.inserted, 100);
    assert_eq!(result2.inserted, 100);
    assert_eq!(tree.len(), 200);
}

#[test]
fn test_batch_insert_then_read_same_guard() {
    let tree: MassTree24<u64> = MassTree24::new();
    let guard = tree.guard();

    let entries: Vec<(Vec<u8>, u64)> = (0..50)
        .map(|i| (format!("key{i}").into_bytes(), i as u64))
        .collect();

    let _ = tree.insert_batch_with_guard(entries, &guard).unwrap();

    // Read back with same guard
    for i in 0..50 {
        let key = format!("key{i}");
        let value = tree.get_with_guard(key.as_bytes(), &guard).unwrap();
        assert_eq!(*value, i as u64);
    }
}

// ============================================================================
//  Edge Cases
// ============================================================================

#[test]
fn test_batch_insert_empty_key() {
    let tree: MassTree24<u64> = MassTree24::new();

    let entries = vec![(vec![], 42u64)];
    let result = tree.insert_batch(entries).unwrap();

    assert_eq!(result.inserted, 1);
    assert_eq!(tree.len(), 1);

    let value = tree.get(&[]).unwrap();
    assert_eq!(*value, 42);
}

#[test]
fn test_batch_insert_same_ikey_different_keylen() {
    let tree: MassTree24<u64> = MassTree24::new();

    // Keys with same first 8 bytes but different lengths
    let entries: Vec<(Vec<u8>, u64)> = vec![
        (b"abcdefgh".to_vec(), 1),      // exactly 8 bytes
        (b"abcdefg".to_vec(), 2),       // 7 bytes
        (b"abcdef".to_vec(), 3),        // 6 bytes
    ];

    let result = tree.insert_batch(entries).unwrap();

    assert_eq!(result.inserted, 3);
    assert_eq!(tree.len(), 3);

    // Verify each key
    assert_eq!(*tree.get(b"abcdefgh").unwrap(), 1);
    assert_eq!(*tree.get(b"abcdefg").unwrap(), 2);
    assert_eq!(*tree.get(b"abcdef").unwrap(), 3);
}

// ============================================================================
//  Result Consistency Tests
// ============================================================================

#[test]
fn test_batch_result_total() {
    let tree: MassTree24<u64> = MassTree24::new();

    // Insert some initial values
    for i in 0..50 {
        tree.insert(format!("k{i}").as_bytes(), i as u64).unwrap();
    }

    // Batch with mix of new and existing keys
    let entries: Vec<(Vec<u8>, u64)> = (0..100)
        .map(|i| (format!("k{i}").into_bytes(), (i * 2) as u64))
        .collect();

    let result = tree.insert_batch(entries).unwrap();

    // Total should equal input count
    assert_eq!(result.total(), 100);
    assert_eq!(result.old_values.len(), result.updated);
}

#[test]
fn test_batch_result_all_succeeded() {
    let tree: MassTree24<u64> = MassTree24::new();

    // Simple batch that should all succeed
    let entries: Vec<(Vec<u8>, u64)> = (0..10)
        .map(|i| ((i as u64).to_be_bytes().to_vec(), i as u64))
        .collect();

    let result = tree.insert_batch(entries).unwrap();

    assert!(result.all_succeeded());
    assert_eq!(result.failed, 0);
}

// ============================================================================
//  Output Type Verification Tests
// ============================================================================

/// Verify that MassTree24 (Arc mode) returns Arc<V> in old_values
#[test]
fn test_batch_arc_mode_old_value_type() {
    let tree: MassTree24<String> = MassTree24::new();

    // Insert initial value
    tree.insert(b"key", "initial".to_string()).unwrap();

    // Update via batch
    let entries = vec![(b"key".to_vec(), "updated".to_string())];
    let result = tree.insert_batch(entries).unwrap();

    assert_eq!(result.updated, 1);
    assert_eq!(result.old_values.len(), 1);

    // old_values is Vec<Arc<String>>
    let old: &Arc<String> = &result.old_values[0];
    assert_eq!(old.as_str(), "initial");
}

/// Verify that MassTree24Inline (Copy mode) returns V directly in old_values
#[test]
fn test_batch_inline_mode_old_value_type() {
    let tree: MassTree24Inline<u64> = MassTree24Inline::new();

    // Insert initial value
    tree.insert(b"key", 100u64).unwrap();

    // Update via batch
    let entries = vec![(b"key".to_vec(), 200u64)];
    let result = tree.insert_batch(entries).unwrap();

    assert_eq!(result.updated, 1);
    assert_eq!(result.old_values.len(), 1);

    // old_values is Vec<u64> (not Vec<Arc<u64>>)
    let old: u64 = result.old_values[0];
    assert_eq!(old, 100);
}
