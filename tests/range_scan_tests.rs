//! Range scan integration tests.

#![allow(clippy::unwrap_used)]
#![allow(clippy::indexing_slicing)]
#![allow(clippy::expect_used)]
#![allow(clippy::too_many_lines)]

use std::collections::BTreeMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::thread;

use masstree::{MassTree, RangeBound};

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

    tree.insert(b"k", 1).unwrap();

    let all: Vec<_> = tree.iter(&guard).collect();
    assert_eq!(all.len(), 1);
    assert_eq!(all[0].key, b"k");
    assert_eq!(*all[0].value, 1);

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
        tree.insert(k, v).unwrap();
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
        tree.insert(&k, i).unwrap();
        btree.insert(k, i);
    }

    let scan: Vec<_> = tree.iter(&guard).collect();
    assert_eq!(scan.len(), btree.len());

    for (entry, (k, v)) in scan.iter().zip(btree.iter()) {
        assert_eq!(&entry.key, k);
        assert_eq!(&*entry.value, v);
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

    tree.insert(&k1, 1).unwrap();
    tree.insert(&k2, 2).unwrap();

    // Insert the exact 8-byte prefix key after the layer exists.
    tree.insert(prefix, 0).unwrap();

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

    tree.insert(&k2, 2).unwrap();
    tree.insert(&k1, 1).unwrap();

    let keys: Vec<Vec<u8>> = tree.keys(&guard).collect();
    assert_eq!(keys, vec![k1, k2]);
}

#[test]
fn scan_prefix_basic() {
    let tree: MassTree<u64> = MassTree::new();
    let guard = tree.guard();

    tree.insert(b"user:alice", 1).unwrap();
    tree.insert(b"user:bob", 2).unwrap();
    tree.insert(b"user:charlie", 3).unwrap();
    tree.insert(b"admin:root", 4).unwrap();

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
        tree.insert(&key, i).unwrap();
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

    const N: usize = 1000;
    const PREFIX_BUCKETS: u64 = 10;

    // Generate keys similar to benchmark: 16 bytes with bucketed prefix
    for i in 0..N {
        let mut key = [0u8; 16];
        // First 8 bytes: bucketed prefix
        let prefix = ((i as u64) % PREFIX_BUCKETS).to_be_bytes();
        key[0..8].copy_from_slice(&prefix);
        // Next 8 bytes: unique suffix based on i
        let suffix = (i as u64).wrapping_mul(0x517c_c1b7_2722_0a95).to_be_bytes();
        key[8..16].copy_from_slice(&suffix);

        tree.insert(&key, i as u64).unwrap();
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
            "key {:?} doesn't start with prefix {:?}",
            k,
            prefix
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
        tree.insert(&key, i).unwrap();
    }

    // Insert exact 8-byte prefix key
    tree.insert(&prefix, 100).unwrap();

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
    for i in 1..keys_found.len() {
        assert_eq!(keys_found[i].len(), 16, "longer keys should be 16 bytes");
        assert!(
            keys_found[i].starts_with(&prefix),
            "key should start with prefix"
        );
    }
}
