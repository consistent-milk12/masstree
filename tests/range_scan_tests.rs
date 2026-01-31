//! Range scan integration tests.

#![allow(clippy::unwrap_used)]
#![allow(clippy::indexing_slicing)]
#![allow(clippy::expect_used)]
#![allow(clippy::too_many_lines)]
#![allow(clippy::cast_possible_truncation)]

use std::collections::BTreeMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::thread;

use masstree::{MassTree, MassTree15, RangeBound};

const N: usize = 1000;
const PREFIX_BUCKETS: u64 = 10;

#[test]
fn range_empty_tree() {
    let tree: MassTree<u64> = MassTree::new();
    let guard = tree.guard();
    assert!(tree.iter(&guard).next().is_none());
}

#[test]
fn range_single_element_bounds() {
    let tree: MassTree<u64> = MassTree::new();
    let guard = tree.guard();

    tree.insert(b"k", 1);

    let all: Vec<_> = tree.iter(&guard).collect();
    assert_eq!(all.len(), 1);
    assert_eq!(all[0].key, b"k");
    assert_eq!(all[0].value, 1);

    let inc: Vec<_> = tree
        .range(RangeBound::Included(b"k"), RangeBound::Unbounded, &guard)
        .collect();
    assert_eq!(inc.len(), 1);
    assert_eq!(inc[0].key, b"k");

    let exc: Vec<_> = tree
        .range(RangeBound::Excluded(b"k"), RangeBound::Unbounded, &guard)
        .collect();
    assert!(exc.is_empty());

    let end_excl: Vec<_> = tree
        .range(RangeBound::Unbounded, RangeBound::Excluded(b"k"), &guard)
        .collect();
    assert!(end_excl.is_empty());

    let end_incl: Vec<_> = tree
        .range(RangeBound::Unbounded, RangeBound::Included(b"k"), &guard)
        .collect();
    assert_eq!(end_incl.len(), 1);
    assert_eq!(end_incl[0].key, b"k");
}

#[test]
fn range_end_bound_stops() {
    let tree: MassTree<u64> = MassTree::new();
    let guard = tree.guard();

    for (k, v) in [(b"a", 1), (b"b", 2), (b"c", 3), (b"d", 4)] {
        tree.insert(k, v);
    }

    let entries: Vec<_> = tree
        .range(
            RangeBound::Included(b"b"),
            RangeBound::Excluded(b"d"),
            &guard,
        )
        .collect();
    let keys: Vec<_> = entries.iter().map(|e| e.key.as_slice()).collect();
    assert_eq!(keys, vec![b"b".as_slice(), b"c".as_slice()]);
}

#[test]
fn full_scan_matches_btreemap_sorted() {
    let tree: MassTree<u64> = MassTree::new();
    let guard = tree.guard();

    let mut btree: BTreeMap<Vec<u8>, u64> = BTreeMap::new();

    for i in 0..200u64 {
        // Intentionally scramble insertion order.
        let k = format!("k{:04}", (i * 37) % 200).into_bytes();
        tree.insert(&k, i);
        btree.insert(k, i);
    }

    let scan: Vec<_> = tree.iter(&guard).collect();
    assert_eq!(scan.len(), btree.len());

    for (entry, (k, v)) in scan.iter().zip(btree.iter()) {
        assert_eq!(&entry.key, k);
        assert_eq!(&entry.value, v);
    }
}

#[test]
fn scan_orders_prefix_key_before_layer_contents() {
    let tree: MassTree<u64> = MassTree::new();
    let guard = tree.guard();

    // 8-byte prefix.
    let prefix: &[u8] = b"AAAAAAAA";

    // Insert longer keys (forces suffix conflict -> layer creation).
    let k1 = [prefix, b"BBBBBBBB"].concat();
    let k2 = [prefix, b"CCCCCCCC"].concat();

    tree.insert(&k1, 1);
    tree.insert(&k2, 2);

    // Insert the exact 8-byte prefix key after the layer exists.
    tree.insert(prefix, 0);

    let keys: Vec<Vec<u8>> = tree.keys(&guard).collect();
    assert_eq!(keys.len(), 3);

    assert_eq!(keys[0], prefix);
    assert_eq!(keys[1], k1);
    assert_eq!(keys[2], k2);
}

#[test]
fn scan_multi_layer_conflict_chain_is_sorted() {
    let tree: MassTree<u64> = MassTree::new();
    let guard = tree.guard();

    // Two keys share 16 bytes, differ in the next chunk -> forces a deeper layer chain.
    let k1 = [
        b"AAAAAAAA".as_ref(),
        b"BBBBBBBB".as_ref(),
        b"CCCCCCCC".as_ref(),
    ]
    .concat();
    let k2 = [
        b"AAAAAAAA".as_ref(),
        b"BBBBBBBB".as_ref(),
        b"DDDDDDDD".as_ref(),
    ]
    .concat();

    tree.insert(&k2, 2);
    tree.insert(&k1, 1);

    let keys: Vec<Vec<u8>> = tree.keys(&guard).collect();
    assert_eq!(keys, vec![k1, k2]);
}

#[test]
fn scan_prefix_basic() {
    let tree: MassTree<u64> = MassTree::new();
    let guard = tree.guard();

    tree.insert(b"user:alice", 1);
    tree.insert(b"user:bob", 2);
    tree.insert(b"user:charlie", 3);
    tree.insert(b"admin:root", 4);

    let mut keys: Vec<Vec<u8>> = Vec::new();
    let visited = tree.scan_prefix(
        b"user:",
        |k, _v| {
            keys.push(k.to_vec());
            true
        },
        &guard,
    );

    assert_eq!(visited, 3);
    assert!(keys.iter().all(|k| k.starts_with(b"user:")));
}

/// Stress test: concurrent inserts while scanning.
///
/// Writers insert keys that will cause leaf splits.
/// Readers perform full scans and verify output is sorted.
#[test]
fn scan_concurrent_with_inserts() {
    const NUM_WRITERS: usize = 4;
    const NUM_READERS: usize = 4;
    const KEYS_PER_WRITER: usize = 500;
    const SCANS_PER_READER: usize = 20;

    let tree: Arc<MassTree<u64>> = Arc::new(MassTree::new());
    let stop = Arc::new(AtomicBool::new(false));
    let total_scans = Arc::new(AtomicUsize::new(0));
    let ordering_violations = Arc::new(AtomicUsize::new(0));

    // Spawn writer threads.
    let writers: Vec<_> = (0..NUM_WRITERS)
        .map(|writer_id| {
            let tree = Arc::clone(&tree);
            thread::spawn(move || {
                for i in 0..KEYS_PER_WRITER {
                    // Create keys that will spread across leaves and cause splits.
                    // Mix short and long keys to exercise layer creation.
                    let key = if i % 3 == 0 {
                        // Long key (forces layers).
                        format!("writer{writer_id:02}:key{i:06}:suffix").into_bytes()
                    } else {
                        // Short key.
                        format!("w{writer_id:02}k{i:06}").into_bytes()
                    };
                    let _ = tree.insert(&key, (writer_id * KEYS_PER_WRITER + i) as u64);

                    // Occasionally yield to interleave with readers.
                    if i % 50 == 0 {
                        thread::yield_now();
                    }
                }
            })
        })
        .collect();

    // Spawn reader threads.
    let readers: Vec<_> = (0..NUM_READERS)
        .map(|_| {
            let tree = Arc::clone(&tree);
            let stop = Arc::clone(&stop);
            let total_scans = Arc::clone(&total_scans);
            let ordering_violations = Arc::clone(&ordering_violations);
            thread::spawn(move || {
                let mut scans_done = 0;
                while !stop.load(Ordering::Relaxed) && scans_done < SCANS_PER_READER {
                    let guard = tree.guard();
                    let entries: Vec<_> = tree.iter(&guard).collect();

                    // Verify sorted order.
                    for window in entries.windows(2) {
                        if window[0].key >= window[1].key {
                            ordering_violations.fetch_add(1, Ordering::Relaxed);
                        }
                    }

                    scans_done += 1;
                    total_scans.fetch_add(1, Ordering::Relaxed);

                    // Small delay between scans.
                    thread::yield_now();
                }
            })
        })
        .collect();

    // Wait for writers to finish.
    for w in writers {
        w.join().expect("writer thread panicked");
    }

    // Signal readers to stop and wait.
    stop.store(true, Ordering::Relaxed);
    for r in readers {
        r.join().expect("reader thread panicked");
    }

    // Verify results.
    let violations = ordering_violations.load(Ordering::Relaxed);
    let scans = total_scans.load(Ordering::Relaxed);

    assert_eq!(
        violations, 0,
        "detected {violations} ordering violations across {scans} scans"
    );
    assert!(scans > 0, "no scans were performed");

    // Final verification: full scan should be sorted.
    let guard = tree.guard();
    let final_entries: Vec<_> = tree.iter(&guard).collect();
    for window in final_entries.windows(2) {
        assert!(
            window[0].key < window[1].key,
            "final scan not sorted: {:?} >= {:?}",
            String::from_utf8_lossy(&window[0].key),
            String::from_utf8_lossy(&window[1].key)
        );
    }

    // Verify we inserted the expected number of keys.
    assert_eq!(
        final_entries.len(),
        NUM_WRITERS * KEYS_PER_WRITER,
        "expected {} keys, got {}",
        NUM_WRITERS * KEYS_PER_WRITER,
        final_entries.len()
    );
}

/// Stress test: scan with prefix while concurrent inserts to same prefix.
#[test]
fn scan_prefix_concurrent_with_inserts() {
    const NUM_WRITERS: usize = 2;
    const KEYS_PER_WRITER: usize = 200;
    const SCANS_PER_READER: usize = 30;

    let tree: Arc<MassTree<u64>> = Arc::new(MassTree::new());
    let stop = Arc::new(AtomicBool::new(false));
    let ordering_violations = Arc::new(AtomicUsize::new(0));

    // Pre-populate with some keys outside the scan prefix.
    for i in 0..50u64 {
        let key = format!("other:{i:04}").into_bytes();
        tree.insert(&key, i);
    }

    // Spawn writers that insert keys with the target prefix.
    let writers: Vec<_> = (0..NUM_WRITERS)
        .map(|writer_id| {
            let tree = Arc::clone(&tree);
            thread::spawn(move || {
                for i in 0..KEYS_PER_WRITER {
                    let key = format!("target:{writer_id:02}:{i:06}").into_bytes();
                    let _ = tree.insert(&key, (writer_id * KEYS_PER_WRITER + i) as u64);
                    if i % 20 == 0 {
                        thread::yield_now();
                    }
                }
            })
        })
        .collect();

    // Spawn readers that scan with prefix "target:".
    let readers: Vec<_> = (0..2)
        .map(|_| {
            let tree = Arc::clone(&tree);
            let stop = Arc::clone(&stop);
            let ordering_violations = Arc::clone(&ordering_violations);
            thread::spawn(move || {
                let mut scans_done = 0;
                while !stop.load(Ordering::Relaxed) && scans_done < SCANS_PER_READER {
                    let guard = tree.guard();
                    let mut keys: Vec<Vec<u8>> = Vec::new();
                    tree.scan_prefix(
                        b"target:",
                        |k, _v| {
                            keys.push(k.to_vec());
                            true
                        },
                        &guard,
                    );

                    // Verify all keys have the prefix.
                    for k in &keys {
                        assert!(
                            k.starts_with(b"target:"),
                            "scan_prefix returned key without prefix: {:?}",
                            String::from_utf8_lossy(k)
                        );
                    }

                    // Verify sorted order.
                    for window in keys.windows(2) {
                        if window[0] >= window[1] {
                            ordering_violations.fetch_add(1, Ordering::Relaxed);
                        }
                    }

                    scans_done += 1;
                    thread::yield_now();
                }
            })
        })
        .collect();

    // Wait for writers.
    for w in writers {
        w.join().expect("writer thread panicked");
    }

    stop.store(true, Ordering::Relaxed);
    for r in readers {
        r.join().expect("reader thread panicked");
    }

    let violations = ordering_violations.load(Ordering::Relaxed);
    assert_eq!(violations, 0, "detected {violations} ordering violations");

    // Final check: prefix scan should return exactly the inserted keys.
    let guard = tree.guard();
    let mut final_keys: Vec<Vec<u8>> = Vec::new();
    tree.scan_prefix(
        b"target:",
        |k, _v| {
            final_keys.push(k.to_vec());
            true
        },
        &guard,
    );

    assert_eq!(
        final_keys.len(),
        NUM_WRITERS * KEYS_PER_WRITER,
        "expected {} keys with prefix, got {}",
        NUM_WRITERS * KEYS_PER_WRITER,
        final_keys.len()
    );
}

/// Stress test: deep layer chains under concurrent access.
#[test]
fn scan_deep_layers_concurrent() {
    const NUM_THREADS: usize = 4;
    const KEYS_PER_THREAD: usize = 100;

    let tree: Arc<MassTree<u64>> = Arc::new(MassTree::new());
    let ordering_violations = Arc::new(AtomicUsize::new(0));

    // Create keys that share long prefixes, forcing deep layer chains.
    // 24-byte shared prefix + varying suffix = 3+ layers.
    let shared_prefix = b"AAAAAAAABBBBBBBBCCCCCCCC"; // 24 bytes

    let handles: Vec<_> = (0..NUM_THREADS)
        .map(|thread_id| {
            let tree = Arc::clone(&tree);
            let ordering_violations = Arc::clone(&ordering_violations);
            thread::spawn(move || {
                // Insert phase.
                for i in 0..KEYS_PER_THREAD {
                    let suffix = format!("{thread_id:02}{i:04}");
                    let key = [shared_prefix.as_ref(), suffix.as_bytes()].concat();
                    let _ = tree.insert(&key, (thread_id * KEYS_PER_THREAD + i) as u64);
                }

                // Scan phase (interleaved with other threads' inserts).
                for _ in 0..5 {
                    let guard = tree.guard();
                    let entries: Vec<_> = tree.iter(&guard).collect();

                    for window in entries.windows(2) {
                        if window[0].key >= window[1].key {
                            ordering_violations.fetch_add(1, Ordering::Relaxed);
                        }
                    }

                    thread::yield_now();
                }
            })
        })
        .collect();

    for h in handles {
        h.join().expect("thread panicked");
    }

    let violations = ordering_violations.load(Ordering::Relaxed);
    assert_eq!(violations, 0, "detected {violations} ordering violations");

    // Verify final state.
    let guard = tree.guard();
    let final_entries: Vec<_> = tree.iter(&guard).collect();

    assert_eq!(
        final_entries.len(),
        NUM_THREADS * KEYS_PER_THREAD,
        "missing keys"
    );

    // All keys should share the prefix.
    for entry in &final_entries {
        assert!(
            entry.key.starts_with(shared_prefix),
            "key missing shared prefix: {:?}",
            String::from_utf8_lossy(&entry.key)
        );
    }

    // Sorted order check.
    for window in final_entries.windows(2) {
        assert!(window[0].key < window[1].key, "not sorted");
    }
}

/// Test that prefix scan correctly descends into layer pointers.
///
/// This test mimics the benchmark setup in `08_masstree_prefix_scan`:
/// - Keys are 16 bytes with first 8 bytes as bucketed prefix
/// - Prefix scan should descend into the layer and return all matching entries
///
/// Previously, 8-byte prefix scans would return 0 entries due to a bug
/// where `handle_initial_match` didn't descend into layer pointers when
/// the cursor key had no suffix.
#[test]
fn scan_prefix_layer_descent() {
    let tree: MassTree<u64> = MassTree::new();
    let guard = tree.guard();

    // Generate keys similar to benchmark: 16 bytes with bucketed prefix
    for i in 0..N {
        let mut key = [0u8; 16];
        // First 8 bytes: bucketed prefix
        let prefix = ((i as u64) % PREFIX_BUCKETS).to_be_bytes();
        key[0..8].copy_from_slice(&prefix);
        // Next 8 bytes: unique suffix based on i
        let suffix = (i as u64).wrapping_mul(0x517c_c1b7_2722_0a95).to_be_bytes();
        key[8..16].copy_from_slice(&suffix);

        tree.insert(&key, i as u64);
    }

    // Scan with 8-byte prefix (bucket 0)
    let prefix = 0u64.to_be_bytes();
    let mut count = 0usize;
    let mut keys_found: Vec<Vec<u8>> = Vec::new();

    tree.scan_prefix(
        &prefix,
        |k, _v| {
            count += 1;
            keys_found.push(k.to_vec());
            true
        },
        &guard,
    );

    // With N=1000 and 10 buckets, bucket 0 should have ~100 entries
    let expected = N / PREFIX_BUCKETS as usize;
    assert_eq!(
        count, expected,
        "prefix scan returned {count} entries, expected {expected}"
    );

    // Verify all returned keys have the correct prefix
    for k in &keys_found {
        assert!(
            k.starts_with(&prefix),
            "key {k:?} doesn't start with prefix {prefix:?}"
        );
    }

    // Verify keys are sorted
    for window in keys_found.windows(2) {
        assert!(
            window[0] < window[1],
            "keys not sorted: {:?} >= {:?}",
            window[0],
            window[1]
        );
    }
}

/// Test 8-byte exact prefix scan where prefix key itself exists.
///
/// This tests the edge case where:
/// 1. An exact 8-byte key exists in the tree
/// 2. Longer keys with the same 8-byte prefix also exist (creating a layer)
/// 3. Prefix scan should return both the exact key and all layer contents
#[test]
fn scan_prefix_exact_8byte_with_layer() {
    let tree: MassTree<u64> = MassTree::new();
    let guard = tree.guard();

    let prefix: [u8; 8] = *b"PREFIXXX";

    // Insert longer keys first (creates layer)
    for i in 0..10u64 {
        let mut key = [0u8; 16];
        key[0..8].copy_from_slice(&prefix);
        key[8..16].copy_from_slice(&i.to_be_bytes());
        tree.insert(&key, i);
    }

    // Insert exact 8-byte prefix key
    tree.insert(&prefix, 100);

    // Scan with 8-byte prefix
    let mut keys_found: Vec<Vec<u8>> = Vec::new();
    let count = tree.scan_prefix(
        &prefix,
        |k, _v| {
            keys_found.push(k.to_vec());
            true
        },
        &guard,
    );

    // Should find: 1 exact prefix key + 10 longer keys = 11 total
    assert_eq!(
        count, 11,
        "expected 11 entries (1 exact + 10 longer), got {count}"
    );

    // First key should be the exact 8-byte prefix
    assert_eq!(
        keys_found[0].as_slice(),
        &prefix,
        "first key should be exact prefix"
    );

    // Remaining keys should be the longer ones, sorted
    (1..keys_found.len()).for_each(|i| {
        assert_eq!(keys_found[i].len(), 16, "longer keys should be 16 bytes");
        assert!(
            keys_found[i].starts_with(&prefix),
            "key should start with prefix"
        );
    });
}

// ============================================================================
//  Batch Scan Tests
// ============================================================================

#[test]
fn scan_batch_ref_empty_tree() {
    let tree: MassTree15<u64> = MassTree15::new();
    let guard = tree.guard();

    let count = tree.scan_batch_ref(
        RangeBound::Unbounded,
        RangeBound::Unbounded,
        |_, _| true,
        &guard,
    );

    assert_eq!(count, 0);
}

#[test]
fn scan_batch_ref_single_entry() {
    let tree: MassTree15<u64> = MassTree15::new();
    tree.insert(b"key1", 42);

    let guard = tree.guard();
    let mut entries = Vec::new();
    tree.scan_batch_ref(
        RangeBound::Unbounded,
        RangeBound::Unbounded,
        |key, value| {
            entries.push((key.to_vec(), *value));
            true
        },
        &guard,
    );

    assert_eq!(entries.len(), 1);
    assert_eq!(entries[0], (b"key1".to_vec(), 42));
}

#[test]
fn scan_batch_ref_multiple_entries() {
    let tree: MassTree15<u64> = MassTree15::new();

    for i in 0..100u64 {
        let key = format!("key{i:03}");
        tree.insert(key.as_bytes(), i);
    }

    let guard = tree.guard();
    let mut count = 0usize;
    let mut sum = 0u64;

    tree.scan_batch_ref(
        RangeBound::Unbounded,
        RangeBound::Unbounded,
        |_, value| {
            count += 1;
            sum += *value;
            true
        },
        &guard,
    );

    assert_eq!(count, 100);
    assert_eq!(sum, (0..100).sum::<u64>());
}

#[test]
fn scan_batch_ref_range_bounds() {
    let tree: MassTree15<u64> = MassTree15::new();

    for i in 0..100u64 {
        let key = format!("key{i:03}");
        tree.insert(key.as_bytes(), i);
    }

    let guard = tree.guard();
    let mut entries = Vec::new();
    tree.scan_batch_ref(
        RangeBound::Included(b"key020"),
        RangeBound::Excluded(b"key030"),
        |key, value| {
            entries.push((key.to_vec(), *value));
            true
        },
        &guard,
    );

    assert_eq!(entries.len(), 10);
    assert_eq!(entries[0].1, 20);
    assert_eq!(entries[9].1, 29);
}

#[test]
fn scan_batch_ref_early_stop() {
    let tree: MassTree15<u64> = MassTree15::new();

    for i in 0..100u64 {
        let key = format!("key{i:03}");
        tree.insert(key.as_bytes(), i);
    }

    let guard = tree.guard();
    let mut count = 0usize;
    let result = tree.scan_batch_ref(
        RangeBound::Unbounded,
        RangeBound::Unbounded,
        |_, _| {
            count += 1;
            count < 50 // Stop after 50 entries
        },
        &guard,
    );

    assert_eq!(result, 50);
    assert_eq!(count, 50);
}

/// Critical test: `scan_batch_ref` must produce identical results to `scan_ref`.
#[test]
fn scan_batch_ref_vs_scan_ref_consistency() {
    let tree: MassTree15<u64> = MassTree15::new();

    for i in 0..1000u64 {
        let key = format!("key{i:05}");
        tree.insert(key.as_bytes(), i);
    }

    let guard = tree.guard();

    // Collect with scan_ref
    let mut scan_ref_entries = Vec::new();
    tree.scan_ref(
        RangeBound::Unbounded,
        RangeBound::Unbounded,
        |key, value| {
            scan_ref_entries.push((key.to_vec(), *value));
            true
        },
        &guard,
    );

    // Collect with scan_batch_ref
    let mut batch_entries = Vec::new();
    tree.scan_batch_ref(
        RangeBound::Unbounded,
        RangeBound::Unbounded,
        |key, value| {
            batch_entries.push((key.to_vec(), *value));
            true
        },
        &guard,
    );

    // Must produce identical results
    assert_eq!(scan_ref_entries.len(), batch_entries.len());
    for (a, b) in scan_ref_entries.iter().zip(batch_entries.iter()) {
        assert_eq!(
            a, b,
            "scan_ref and scan_batch_ref produced different results"
        );
    }
}

#[test]
fn scan_batch_ref_single_layer_keys() {
    // Test with short keys (≤ 8 bytes) - uses optimized single-layer path
    let tree: MassTree15<u64> = MassTree15::new();

    for i in 0..100u64 {
        let key = format!("k{i:06}"); // 7 bytes
        tree.insert(key.as_bytes(), i);
    }

    let guard = tree.guard();
    let mut sum = 0u64;
    let count = tree.scan_batch_ref(
        RangeBound::Unbounded,
        RangeBound::Unbounded,
        |_, value| {
            sum += *value;
            true
        },
        &guard,
    );

    assert_eq!(count, 100);
    assert_eq!(sum, (0..100).sum::<u64>());
}

/// Test batch scan with keys > 8 bytes (requires layer navigation).
#[test]
fn scan_batch_ref_long_keys_with_layers() {
    let tree: MassTree15<u64> = MassTree15::new();

    // Create a map of expected keys for verification
    let mut expected: BTreeMap<Vec<u8>, u64> = BTreeMap::new();

    for i in 0..100u64 {
        let key = format!("long_prefix_key_{i:05}"); // >8 bytes = 21 bytes
        tree.insert(key.as_bytes(), i);
        expected.insert(key.into_bytes(), i);
    }

    let guard = tree.guard();

    // Collect with batch scan
    let mut batch_entries = Vec::new();
    tree.scan_batch_ref(
        RangeBound::Unbounded,
        RangeBound::Unbounded,
        |key, value| {
            batch_entries.push((key.to_vec(), *value));
            true
        },
        &guard,
    );

    // Verify count
    assert_eq!(
        batch_entries.len(),
        expected.len(),
        "Wrong number of entries"
    );

    // Verify each entry matches expected
    for (key, value) in &batch_entries {
        assert!(
            expected.get(key) == Some(value),
            "Key {:?} has wrong value or doesn't exist",
            String::from_utf8_lossy(key)
        );
    }

    // Verify sorted order
    for window in batch_entries.windows(2) {
        assert!(
            window[0].0 < window[1].0,
            "Keys not in sorted order: {:?} >= {:?}",
            String::from_utf8_lossy(&window[0].0),
            String::from_utf8_lossy(&window[1].0)
        );
    }
}

/// Concurrent batch scans should be safe.
#[test]
fn scan_batch_ref_concurrent_reads() {
    let tree = Arc::new(MassTree15::<u64>::new());

    // Pre-populate
    for i in 0..10000u64 {
        let key = format!("key{i:08}");
        tree.insert(key.as_bytes(), i);
    }

    // Concurrent batch scans
    let handles: Vec<_> = (0..4)
        .map(|_| {
            let tree = Arc::clone(&tree);
            thread::spawn(move || {
                let guard = tree.guard();
                let mut sum = 0u64;
                let count = tree.scan_batch_ref(
                    RangeBound::Unbounded,
                    RangeBound::Unbounded,
                    |_, value| {
                        sum += *value;
                        true
                    },
                    &guard,
                );
                (count, sum)
            })
        })
        .collect();

    let expected_sum: u64 = (0..10000).sum();

    for handle in handles {
        let (count, sum) = handle.join().unwrap();
        assert_eq!(count, 10000);
        assert_eq!(sum, expected_sum);
    }
}

/// Test `for_each_batch_ref` on [`RangeIter`] directly.
#[test]
fn for_each_batch_ref_basic() {
    let tree: MassTree15<u64> = MassTree15::new();

    for i in 0..100u64 {
        let key = format!("key{i:03}");
        tree.insert(key.as_bytes(), i);
    }

    let guard = tree.guard();
    let mut sum = 0u64;

    let count = tree
        .range(RangeBound::Unbounded, RangeBound::Unbounded, &guard)
        .for_each_batch_ref(|_, value| {
            sum += *value;
            true
        });

    assert_eq!(count, 100);
    assert_eq!(sum, (0..100).sum::<u64>());
}

// ============================================================================
//  scan_ref correctness tests for multi-layer keys
// ============================================================================

/// Test `scan_ref` with multi-layer keys against [`BTreeMap`]
/// This specifically targets the bug where `scan_ref` was returning incorrect keys
/// for multi-layer entries (8 null bytes prepended to keys).
#[test]
fn scan_ref_long_keys_matches_btreemap() {
    let tree: MassTree15<u64> = MassTree15::new();
    let mut expected: BTreeMap<Vec<u8>, u64> = BTreeMap::new();

    // Insert multi-layer keys (> 8 bytes triggers layer creation)
    for i in 0..100u64 {
        let key = format!("long_prefix_key_{i:05}"); // 21 bytes
        tree.insert(key.as_bytes(), i);
        expected.insert(key.into_bytes(), i);
    }

    let guard = tree.guard();

    // Collect with scan_ref
    let mut scan_ref_entries = Vec::new();
    tree.scan_ref(
        RangeBound::Unbounded,
        RangeBound::Unbounded,
        |key, value| {
            scan_ref_entries.push((key.to_vec(), *value));
            true
        },
        &guard,
    );

    // Verify count matches
    assert_eq!(
        scan_ref_entries.len(),
        expected.len(),
        "scan_ref returned wrong number of entries"
    );

    // Verify each entry matches expected
    for (key, value) in &scan_ref_entries {
        let exp_value = expected.get(key);
        assert!(
            exp_value == Some(value),
            "scan_ref: key {:?} (len={}) has wrong value {:?}, expected {:?}",
            String::from_utf8_lossy(key),
            key.len(),
            Some(value),
            exp_value
        );
    }

    // Verify sorted order matches BTreeMap
    let expected_vec: Vec<_> = expected.iter().map(|(k, v)| (k.clone(), *v)).collect();
    for (i, (scan_entry, exp_entry)) in scan_ref_entries.iter().zip(expected_vec.iter()).enumerate()
    {
        assert_eq!(
            scan_entry,
            exp_entry,
            "Mismatch at index {}: scan_ref={:?}, expected={:?}",
            i,
            String::from_utf8_lossy(&scan_entry.0),
            String::from_utf8_lossy(&exp_entry.0)
        );
    }
}

/// Test `scan_ref` with very deep layers (3+ layers).
#[test]
fn scan_ref_deep_layers_matches_btreemap() {
    let tree: MassTree15<u64> = MassTree15::new();
    let mut expected: BTreeMap<Vec<u8>, u64> = BTreeMap::new();

    // 24-byte shared prefix forces 3 layers
    let shared_prefix = b"AAAAAAAABBBBBBBBCCCCCCCC";

    for i in 0..50u64 {
        let suffix = format!("{i:06}");
        let key = [shared_prefix.as_ref(), suffix.as_bytes()].concat(); // 30 bytes
        tree.insert(&key, i);
        expected.insert(key, i);
    }

    let guard = tree.guard();

    // Collect with scan_ref
    let mut scan_ref_entries = Vec::new();
    tree.scan_ref(
        RangeBound::Unbounded,
        RangeBound::Unbounded,
        |key, value| {
            scan_ref_entries.push((key.to_vec(), *value));
            true
        },
        &guard,
    );

    // Verify count
    assert_eq!(scan_ref_entries.len(), expected.len());

    // Verify each key is correct
    for (key, value) in &scan_ref_entries {
        assert!(
            expected.get(key) == Some(value),
            "Key {:?} (len={}) not found or has wrong value in expected",
            String::from_utf8_lossy(key),
            key.len()
        );
    }
}

/// Test `scan_ref` with mixed single-layer and multi-layer keys.
#[test]
fn scan_ref_mixed_layer_keys() {
    let tree: MassTree15<u64> = MassTree15::new();
    let mut expected: BTreeMap<Vec<u8>, u64> = BTreeMap::new();

    // Mix of short (single layer) and long (multi layer) keys
    for i in 0..50u64 {
        // Short key (8 bytes = single layer)
        let short_key = format!("short{i:02}");
        tree.insert(short_key.as_bytes(), i);
        expected.insert(short_key.into_bytes(), i);

        // Long key (20+ bytes = multi layer)
        let long_key = format!("this_is_a_long_key_{i:05}");
        tree.insert(long_key.as_bytes(), i + 1000);
        expected.insert(long_key.into_bytes(), i + 1000);
    }

    let guard = tree.guard();

    let mut scan_ref_entries = Vec::new();
    tree.scan_ref(
        RangeBound::Unbounded,
        RangeBound::Unbounded,
        |key, value| {
            scan_ref_entries.push((key.to_vec(), *value));
            true
        },
        &guard,
    );

    assert_eq!(scan_ref_entries.len(), expected.len());

    for (key, value) in &scan_ref_entries {
        assert!(
            expected.get(key) == Some(value),
            "Key {:?} (len={}) not found or wrong value",
            String::from_utf8_lossy(key),
            key.len()
        );
    }
}
