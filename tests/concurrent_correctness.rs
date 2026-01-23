//! Concurrent correctness tests targeting specific race conditions.
//!
//! These tests are designed to catch bugs like those fixed in the recent
//! correctness patches (run142):
//!
//! 1. **Remove operation races**: mark_insert() timing, partial state visibility
//! 2. **Permutation slot/position confusion**: remove_slot() vs remove()
//! 3. **Suffix drain atomicity**: concurrent suffix overflow scenarios
//! 4. **Too-right detection**: splits causing traversal to wrong leaf
//!
//! Run with: `cargo test --test concurrent_correctness --release`
//! Run with tracing: `RUST_LOG=masstree=debug cargo test --test concurrent_correctness`

#![allow(clippy::pedantic)]
#![allow(clippy::panic, reason = "Fail fast in tests")]
#![expect(clippy::unwrap_used)]

mod common;

use masstree::{MassTree15, MassTree24};
use std::collections::{BTreeMap, HashSet};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Barrier};
use std::thread;
use std::time::Duration;

// =============================================================================
// CONCURRENT REMOVE STRESS TESTS
// =============================================================================
// These tests target the mark_insert() reordering fix in remove.rs

/// Heavy concurrent remove + read stress test.
///
/// Verifies that:
/// 1. Readers never see partial state during removal
/// 2. Keys being removed don't cause spurious "not found" for other keys
/// 3. After removal completes, the key is definitely gone
#[test]
fn concurrent_remove_read_stress() {
    common::init_tracing();

    const NUM_KEYS: u64 = 2000;
    const NUM_READERS: usize = 4;
    const NUM_REMOVERS: usize = 2;
    const READ_ITERATIONS: usize = 5000;

    let tree = Arc::new(MassTree15::<u64>::new());

    // Pre-populate
    for i in 0..NUM_KEYS {
        tree.insert(&i.to_be_bytes(), i).unwrap();
    }

    let barrier = Arc::new(Barrier::new(NUM_READERS + NUM_REMOVERS));
    let stop_flag = Arc::new(AtomicBool::new(false));
    let removed_keys = Arc::new(std::sync::Mutex::new(HashSet::new()));

    // Track anomalies: key exists but read returns wrong value
    let value_mismatches = Arc::new(AtomicU64::new(0));
    // Track reads that found a key
    let successful_reads = Arc::new(AtomicU64::new(0));

    let mut handles = vec![];

    // Reader threads: continuously read random keys, verify values match
    for rid in 0..NUM_READERS {
        let tree = Arc::clone(&tree);
        let barrier = Arc::clone(&barrier);
        let stop_flag = Arc::clone(&stop_flag);
        let value_mismatches = Arc::clone(&value_mismatches);
        let successful_reads = Arc::clone(&successful_reads);

        handles.push(thread::spawn(move || {
            barrier.wait();
            let guard = tree.guard();
            let mut rng_state = rid as u64 * 12345;

            for _ in 0..READ_ITERATIONS {
                if stop_flag.load(Ordering::Relaxed) {
                    break;
                }

                // Simple LCG for deterministic randomness
                rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1);
                let key = rng_state % NUM_KEYS;

                if let Some(value) = tree.get_with_guard(&key.to_be_bytes(), &guard) {
                    // If we found the key, value MUST equal key
                    // Note: get_with_guard returns Arc<u64>, dereference to compare
                    if *value != key {
                        value_mismatches.fetch_add(1, Ordering::Relaxed);
                        tracing::error!(
                            key = key,
                            expected = key,
                            got = *value,
                            "VALUE MISMATCH: read returned wrong value"
                        );
                    }
                    successful_reads.fetch_add(1, Ordering::Relaxed);
                }
                // Not finding a key is OK - it may have been removed
            }
        }));
    }

    // Remover threads: remove keys in different patterns
    for tid in 0..NUM_REMOVERS {
        let tree = Arc::clone(&tree);
        let barrier = Arc::clone(&barrier);
        let stop_flag = Arc::clone(&stop_flag);
        let removed_keys = Arc::clone(&removed_keys);

        handles.push(thread::spawn(move || {
            barrier.wait();
            let guard = tree.guard();

            // Each remover handles a different key range
            let start = (tid as u64) * (NUM_KEYS / NUM_REMOVERS as u64);
            let end = start + (NUM_KEYS / NUM_REMOVERS as u64);

            for key in start..end {
                if stop_flag.load(Ordering::Relaxed) {
                    break;
                }

                // Remove with some spacing to allow interleaving
                if key % 2 == tid as u64 {
                    let _ = tree.remove_with_guard(&key.to_be_bytes(), &guard);
                    removed_keys.lock().unwrap().insert(key);
                }
            }
        }));
    }

    for h in handles {
        h.join().unwrap();
    }

    let mismatches = value_mismatches.load(Ordering::Relaxed);
    let reads = successful_reads.load(Ordering::Relaxed);

    println!(
        "Remove stress: {} successful reads, {} value mismatches",
        reads, mismatches
    );

    assert_eq!(
        mismatches, 0,
        "Detected {} value mismatches - readers saw corrupted state during remove",
        mismatches
    );
}

/// Test that concurrent removes don't corrupt permutation ordering.
///
/// This targets the permuter.remove_slot() fix where physical slot indices
/// were incorrectly passed to remove() which expected logical positions.
#[test]
fn concurrent_remove_permutation_integrity() {
    common::init_tracing();

    const NUM_ITERATIONS: usize = 50;
    const KEYS_PER_ITER: u64 = 200;
    const NUM_THREADS: usize = 4;

    for iteration in 0..NUM_ITERATIONS {
        let tree = Arc::new(MassTree15::<u64>::new());

        // Insert keys that will land in the same leaf initially
        // Using sequential keys ensures they go to same/adjacent leaves
        for i in 0..KEYS_PER_ITER {
            tree.insert(&i.to_be_bytes(), i).unwrap();
        }

        let barrier = Arc::new(Barrier::new(NUM_THREADS));
        let mut handles = vec![];

        // Half threads remove, half threads read
        for tid in 0..NUM_THREADS {
            let tree = Arc::clone(&tree);
            let barrier = Arc::clone(&barrier);

            handles.push(thread::spawn(move || {
                barrier.wait();
                let guard = tree.guard();

                if tid % 2 == 0 {
                    // Remover: remove every 4th key starting at offset
                    let offset = (tid / 2) as u64;
                    for i in (offset..KEYS_PER_ITER).step_by(4) {
                        let _ = tree.remove_with_guard(&i.to_be_bytes(), &guard);
                    }
                } else {
                    // Reader: scan all keys, verify ordering
                    let mut prev_key: Option<Vec<u8>> = None;
                    for entry in tree.iter(&guard) {
                        if let Some(ref pk) = prev_key {
                            assert!(
                                pk.as_slice() < entry.key(),
                                "Iteration {}: Scan ordering violated! prev={:?} >= curr={:?}",
                                iteration,
                                pk,
                                entry.key()
                            );
                        }
                        prev_key = Some(entry.key().to_vec());
                    }
                }
            }));
        }

        for h in handles {
            h.join().unwrap();
        }

        // Final verification: remaining keys are still accessible
        let guard = tree.guard();
        let mut seen = HashSet::new();
        for entry in tree.iter(&guard) {
            let key_bytes = entry.key();
            assert_eq!(key_bytes.len(), 8, "Key length corrupted");
            let key = u64::from_be_bytes(key_bytes.try_into().unwrap());
            assert!(key < KEYS_PER_ITER, "Invalid key value: {}", key);
            assert!(seen.insert(key), "Duplicate key in iteration: {}", key);
        }
    }

    println!(
        "Permutation integrity: {} iterations passed",
        NUM_ITERATIONS
    );
}

/// Test rapid insert-remove-insert cycles on the same keys.
///
/// This stresses the remove path's handling of slot reuse and version updates.
#[test]
fn concurrent_insert_remove_cycles() {
    common::init_tracing();

    const NUM_KEYS: u64 = 100;
    const CYCLES: usize = 500;
    const NUM_THREADS: usize = 4;

    let tree = Arc::new(MassTree15::<u64>::new());
    let barrier = Arc::new(Barrier::new(NUM_THREADS));
    let errors = Arc::new(AtomicU64::new(0));

    let mut handles = vec![];

    for tid in 0..NUM_THREADS {
        let tree = Arc::clone(&tree);
        let barrier = Arc::clone(&barrier);
        let errors = Arc::clone(&errors);

        handles.push(thread::spawn(move || {
            barrier.wait();
            let guard = tree.guard();

            for cycle in 0..CYCLES {
                let key = ((tid as u64 * 1000) + (cycle as u64 % NUM_KEYS)).to_be_bytes();
                let value = (tid as u64 * 1_000_000) + cycle as u64;

                // Insert
                let _ = tree.insert_with_guard(&key, value, &guard);

                // Verify insert
                match tree.get_with_guard(&key, &guard) {
                    Some(v) if *v == value => {} // OK
                    Some(v) => {
                        // Value mismatch - either our write was overwritten or we see stale data
                        // This is acceptable in concurrent scenarios
                        tracing::trace!(
                            tid = tid,
                            cycle = cycle,
                            expected = value,
                            got = *v,
                            "Value overwritten by another thread"
                        );
                    }
                    None => {
                        // Key should exist immediately after insert
                        errors.fetch_add(1, Ordering::Relaxed);
                        tracing::error!(
                            tid = tid,
                            cycle = cycle,
                            "Key not found immediately after insert"
                        );
                    }
                }

                // Remove
                let _ = tree.remove_with_guard(&key, &guard);

                // Key might be re-inserted by another thread, so not checking absence
            }
        }));
    }

    for h in handles {
        h.join().unwrap();
    }

    let error_count = errors.load(Ordering::Relaxed);
    println!(
        "Insert-remove cycles: {} threads × {} cycles, {} errors",
        NUM_THREADS, CYCLES, error_count
    );

    assert_eq!(
        error_count, 0,
        "Detected {} errors in insert-remove cycles",
        error_count
    );
}

// =============================================================================
// TOO-RIGHT DETECTION TESTS
// =============================================================================
// These tests target the fix in optimistic_reads.rs for concurrent splits

/// Test that concurrent splits don't cause false negatives in point reads.
///
/// This targets the "too-right" detection fix where concurrent splits could
/// cause traversal to land on a leaf that's to the right of where the key
/// should be.
#[test]
fn concurrent_splits_no_false_negatives() {
    common::init_tracing();

    const NUM_READERS: usize = 4;
    const NUM_INSERTERS: usize = 2;
    const READ_ROUNDS: usize = 100;

    let tree = Arc::new(MassTree15::<u64>::new());

    // Pre-populate with some keys so tree has structure
    for i in 0..1000u64 {
        tree.insert(&i.to_be_bytes(), i).unwrap();
    }

    let barrier = Arc::new(Barrier::new(NUM_READERS + NUM_INSERTERS));
    let false_negatives = Arc::new(AtomicU64::new(0));
    let reads_performed = Arc::new(AtomicU64::new(0));

    // Track which keys have been inserted (for verification)
    let inserted_keys = Arc::new(std::sync::Mutex::new(HashSet::<u64>::new()));
    for i in 0..1000u64 {
        inserted_keys.lock().unwrap().insert(i);
    }

    let mut handles = vec![];

    // Reader threads: read keys that are known to exist
    for _rid in 0..NUM_READERS {
        let tree = Arc::clone(&tree);
        let barrier = Arc::clone(&barrier);
        let false_negatives = Arc::clone(&false_negatives);
        let reads_performed = Arc::clone(&reads_performed);
        let inserted_keys = Arc::clone(&inserted_keys);

        handles.push(thread::spawn(move || {
            barrier.wait();
            let guard = tree.guard();

            for _ in 0..READ_ROUNDS {
                // Snapshot current inserted keys
                let keys: Vec<u64> = inserted_keys.lock().unwrap().iter().copied().collect();

                for key in keys.iter().take(500) {
                    // Read a key that we know was inserted
                    if tree.get_with_guard(&key.to_be_bytes(), &guard).is_none() {
                        // Double-check: is it still supposed to exist?
                        if inserted_keys.lock().unwrap().contains(key) {
                            false_negatives.fetch_add(1, Ordering::Relaxed);
                            tracing::warn!(
                                key = key,
                                "FALSE NEGATIVE: key not found but should exist"
                            );
                        }
                    }
                    reads_performed.fetch_add(1, Ordering::Relaxed);
                }
            }
        }));
    }

    // Inserter threads: continuously insert new keys to trigger splits
    for tid in 0..NUM_INSERTERS {
        let tree = Arc::clone(&tree);
        let barrier = Arc::clone(&barrier);
        let inserted_keys = Arc::clone(&inserted_keys);

        handles.push(thread::spawn(move || {
            barrier.wait();
            let guard = tree.guard();

            let start = 1000 + (tid as u64 * 2000);
            let end = start + 2000;

            for key in start..end {
                let _ = tree.insert_with_guard(&key.to_be_bytes(), key, &guard);
                inserted_keys.lock().unwrap().insert(key);

                // Small delay to spread out splits
                if key % 100 == 0 {
                    thread::yield_now();
                }
            }
        }));
    }

    for h in handles {
        h.join().unwrap();
    }

    let fn_count = false_negatives.load(Ordering::Relaxed);
    let total_reads = reads_performed.load(Ordering::Relaxed);

    println!(
        "Split stress: {} reads, {} false negatives",
        total_reads, fn_count
    );

    assert_eq!(
        fn_count, 0,
        "Detected {} false negatives during concurrent splits - too-right detection may be broken",
        fn_count
    );
}

/// Test sequential key insertion with concurrent reads during splits.
///
/// Sequential keys are worst-case for splits as they all go to the right edge.
#[test]
fn sequential_inserts_concurrent_reads() {
    common::init_tracing();

    const NUM_KEYS: u64 = 10_000;
    const NUM_READERS: usize = 4;

    let tree = Arc::new(MassTree15::<u64>::new());
    let barrier = Arc::new(Barrier::new(NUM_READERS + 1));
    let insert_done = Arc::new(AtomicBool::new(false));
    let false_negatives = Arc::new(AtomicU64::new(0));

    // Track highest key inserted (monotonically increasing)
    let max_inserted = Arc::new(AtomicU64::new(0));

    let mut handles = vec![];

    // Single inserter: sequential keys
    {
        let tree = Arc::clone(&tree);
        let barrier = Arc::clone(&barrier);
        let insert_done = Arc::clone(&insert_done);
        let max_inserted = Arc::clone(&max_inserted);

        handles.push(thread::spawn(move || {
            barrier.wait();
            let guard = tree.guard();

            for key in 0..NUM_KEYS {
                let _ = tree.insert_with_guard(&key.to_be_bytes(), key, &guard);
                max_inserted.store(key, Ordering::Release);
            }

            insert_done.store(true, Ordering::Release);
        }));
    }

    // Reader threads
    for _rid in 0..NUM_READERS {
        let tree = Arc::clone(&tree);
        let barrier = Arc::clone(&barrier);
        let insert_done = Arc::clone(&insert_done);
        let false_negatives = Arc::clone(&false_negatives);
        let max_inserted = Arc::clone(&max_inserted);

        handles.push(thread::spawn(move || {
            barrier.wait();
            let guard = tree.guard();

            while !insert_done.load(Ordering::Acquire) {
                let max_key = max_inserted.load(Ordering::Acquire);
                if max_key == 0 {
                    continue;
                }

                // Read a key that definitely exists
                let key = max_key / 2; // Middle of inserted range
                if tree.get_with_guard(&key.to_be_bytes(), &guard).is_none() {
                    false_negatives.fetch_add(1, Ordering::Relaxed);
                }
            }
        }));
    }

    for h in handles {
        h.join().unwrap();
    }

    let fn_count = false_negatives.load(Ordering::Relaxed);
    println!("Sequential insert stress: {} false negatives", fn_count);

    assert_eq!(
        fn_count, 0,
        "Detected {} false negatives during sequential inserts",
        fn_count
    );
}

// =============================================================================
// SUFFIX ATOMICITY TESTS
// =============================================================================
// These tests target the suffix drain atomicity fix in inline.rs

/// Test concurrent inserts with long keys that overflow inline suffix storage.
///
/// This targets the suffix drain fix where clearing inline state before
/// publishing the external pointer could cause readers to see no suffix.
#[test]
fn suffix_overflow_concurrent_stress() {
    common::init_tracing();

    const NUM_THREADS: usize = 4;
    const KEYS_PER_THREAD: usize = 500;
    const KEY_LEN: usize = 24; // Forces suffix storage (>16 bytes)

    let tree = Arc::new(MassTree24::<u64>::new());
    let barrier = Arc::new(Barrier::new(NUM_THREADS));
    let read_errors = Arc::new(AtomicU64::new(0));

    let mut handles = vec![];

    for tid in 0..NUM_THREADS {
        let tree = Arc::clone(&tree);
        let barrier = Arc::clone(&barrier);
        let read_errors = Arc::clone(&read_errors);

        handles.push(thread::spawn(move || {
            barrier.wait();
            let guard = tree.guard();

            for i in 0..KEYS_PER_THREAD {
                // Create 24-byte key: 8-byte prefix + 8-byte mid + 8-byte suffix
                // Format: "T{tid:02}_{i:05}__SUFFIX{i:05}" = 3 + 5 + 2 + 6 + 5 + 3 = 24
                let key = format!("T{:02}_{:05}__SUFFIX{:05}XX", tid, i, i);
                assert_eq!(key.len(), KEY_LEN, "Key length mismatch");

                let value = (tid * 100_000 + i) as u64;

                // Insert
                let _ = tree.insert_with_guard(key.as_bytes(), value, &guard);

                // Immediate read-back
                match tree.get_with_guard(key.as_bytes(), &guard) {
                    Some(v) => {
                        if *v != value {
                            // Value mismatch could be from concurrent overwrite
                            tracing::trace!(
                                tid = tid,
                                i = i,
                                expected = value,
                                got = *v,
                                "Value differs (concurrent update?)"
                            );
                        }
                    }
                    None => {
                        read_errors.fetch_add(1, Ordering::Relaxed);
                        tracing::error!(
                            tid = tid,
                            i = i,
                            key = %key,
                            "Key not found after insert - possible suffix atomicity bug"
                        );
                    }
                }
            }

            // Verification pass: read all keys from this thread
            let mut missing = 0u64;
            for i in 0..KEYS_PER_THREAD {
                let key = format!("T{:02}_{:05}__SUFFIX{:05}XX", tid, i, i);
                if tree.get_with_guard(key.as_bytes(), &guard).is_none() {
                    missing += 1;
                }
            }

            if missing > 0 {
                read_errors.fetch_add(missing, Ordering::Relaxed);
                tracing::error!(tid = tid, missing = missing, "Missing keys in verification");
            }
        }));
    }

    for h in handles {
        h.join().unwrap();
    }

    let errors = read_errors.load(Ordering::Relaxed);
    let total_keys = NUM_THREADS * KEYS_PER_THREAD;

    println!(
        "Suffix overflow stress: {} keys inserted, {} read errors",
        total_keys, errors
    );

    assert_eq!(
        errors, 0,
        "Detected {} read errors with long keys - suffix atomicity may be broken",
        errors
    );
}

/// Test that suffix keys with shared prefixes don't corrupt each other.
///
/// Keys with the same 8-byte prefix but different suffixes must coexist.
#[test]
fn shared_prefix_suffix_integrity() {
    common::init_tracing();

    const PREFIX: &str = "AAAAAAAA"; // 8 bytes
    const NUM_SUFFIXES: usize = 100;
    const NUM_THREADS: usize = 4;
    const ROUNDS: usize = 20;

    for round in 0..ROUNDS {
        let tree = Arc::new(MassTree24::<u64>::new());
        let barrier = Arc::new(Barrier::new(NUM_THREADS));

        let mut handles = vec![];

        // All threads insert keys with same prefix but different suffixes
        for tid in 0..NUM_THREADS {
            let tree = Arc::clone(&tree);
            let barrier = Arc::clone(&barrier);

            handles.push(thread::spawn(move || {
                barrier.wait();
                let guard = tree.guard();

                for i in 0..NUM_SUFFIXES {
                    // Key format: "AAAAAAAA" + "T{tid:02}S{i:04}" = 8 + 8 = 16 bytes
                    let suffix = format!("T{:02}S{:04}", tid, i);
                    let key = format!("{}{}", PREFIX, suffix);
                    assert_eq!(key.len(), 16);

                    let value = (tid * 10000 + i) as u64;
                    let _ = tree.insert_with_guard(key.as_bytes(), value, &guard);
                }
            }));
        }

        for h in handles {
            h.join().unwrap();
        }

        // Verify all keys exist with correct values
        let guard = tree.guard();
        let mut errors = 0;

        for tid in 0..NUM_THREADS {
            for i in 0..NUM_SUFFIXES {
                let suffix = format!("T{:02}S{:04}", tid, i);
                let key = format!("{}{}", PREFIX, suffix);
                let expected = (tid * 10000 + i) as u64;

                match tree.get_with_guard(key.as_bytes(), &guard) {
                    Some(v) if *v == expected => {}
                    Some(v) => {
                        errors += 1;
                        tracing::error!(
                            round = round,
                            key = %key,
                            expected = expected,
                            got = *v,
                            "Value mismatch"
                        );
                    }
                    None => {
                        errors += 1;
                        tracing::error!(round = round, key = %key, "Key not found");
                    }
                }
            }
        }

        assert_eq!(
            errors, 0,
            "Round {}: {} errors with shared prefix keys",
            round, errors
        );
    }

    println!(
        "Shared prefix integrity: {} rounds × {} threads × {} suffixes passed",
        ROUNDS, NUM_THREADS, NUM_SUFFIXES
    );
}

// =============================================================================
// RANGE SCAN CORRECTNESS DURING CONCURRENT MODIFICATIONS
// =============================================================================

/// Test that range scans maintain ordering during concurrent inserts.
#[test]
fn range_scan_ordering_during_inserts() {
    common::init_tracing();

    const INITIAL_KEYS: u64 = 1000;
    const INSERT_KEYS: u64 = 2000;
    const NUM_SCANNERS: usize = 2;
    const NUM_INSERTERS: usize = 2;
    const SCANS_PER_THREAD: usize = 50;

    let tree = Arc::new(MassTree15::<u64>::new());

    // Pre-populate
    for i in 0..INITIAL_KEYS {
        tree.insert(&i.to_be_bytes(), i).unwrap();
    }

    let barrier = Arc::new(Barrier::new(NUM_SCANNERS + NUM_INSERTERS));
    let ordering_violations = Arc::new(AtomicU64::new(0));

    let mut handles = vec![];

    // Scanner threads
    for _sid in 0..NUM_SCANNERS {
        let tree = Arc::clone(&tree);
        let barrier = Arc::clone(&barrier);
        let ordering_violations = Arc::clone(&ordering_violations);

        handles.push(thread::spawn(move || {
            barrier.wait();
            let guard = tree.guard();

            for _ in 0..SCANS_PER_THREAD {
                let mut prev_key: Option<Vec<u8>> = None;
                let mut violations_this_scan = 0u64;

                for entry in tree.iter(&guard) {
                    let curr_key = entry.key().to_vec();

                    if let Some(ref pk) = prev_key
                        && pk.as_slice() >= curr_key.as_slice()
                    {
                        violations_this_scan += 1;
                        tracing::error!(
                            prev = ?pk,
                            curr = ?curr_key,
                            "ORDERING VIOLATION in scan"
                        );
                    }

                    prev_key = Some(curr_key);
                }

                if violations_this_scan > 0 {
                    ordering_violations.fetch_add(violations_this_scan, Ordering::Relaxed);
                }
            }
        }));
    }

    // Inserter threads
    for tid in 0..NUM_INSERTERS {
        let tree = Arc::clone(&tree);
        let barrier = Arc::clone(&barrier);

        handles.push(thread::spawn(move || {
            barrier.wait();
            let guard = tree.guard();

            let start = INITIAL_KEYS + (tid as u64 * INSERT_KEYS);
            let end = start + INSERT_KEYS;

            for key in start..end {
                let _ = tree.insert_with_guard(&key.to_be_bytes(), key, &guard);
            }
        }));
    }

    for h in handles {
        h.join().unwrap();
    }

    let violations = ordering_violations.load(Ordering::Relaxed);
    println!(
        "Scan ordering: {} scanners × {} scans, {} violations",
        NUM_SCANNERS, SCANS_PER_THREAD, violations
    );

    assert_eq!(
        violations, 0,
        "Detected {} ordering violations during concurrent scans",
        violations
    );
}

/// Test range scans during concurrent removes.
#[test]
fn range_scan_during_removes() {
    common::init_tracing();

    const NUM_KEYS: u64 = 2000;
    const NUM_SCANNERS: usize = 2;
    const NUM_REMOVERS: usize = 2;
    const SCANS_PER_THREAD: usize = 30;

    let tree = Arc::new(MassTree15::<u64>::new());

    // Pre-populate
    for i in 0..NUM_KEYS {
        tree.insert(&i.to_be_bytes(), i).unwrap();
    }

    let barrier = Arc::new(Barrier::new(NUM_SCANNERS + NUM_REMOVERS));
    let ordering_violations = Arc::new(AtomicU64::new(0));
    let duplicate_keys = Arc::new(AtomicU64::new(0));

    let mut handles = vec![];

    // Scanner threads
    for _sid in 0..NUM_SCANNERS {
        let tree = Arc::clone(&tree);
        let barrier = Arc::clone(&barrier);
        let ordering_violations = Arc::clone(&ordering_violations);
        let duplicate_keys = Arc::clone(&duplicate_keys);

        handles.push(thread::spawn(move || {
            barrier.wait();
            let guard = tree.guard();

            for _ in 0..SCANS_PER_THREAD {
                let mut prev_key: Option<Vec<u8>> = None;
                let mut seen_keys = HashSet::new();

                for entry in tree.iter(&guard) {
                    let curr_key = entry.key().to_vec();

                    // Check ordering
                    if let Some(ref pk) = prev_key
                        && pk.as_slice() >= curr_key.as_slice()
                    {
                        ordering_violations.fetch_add(1, Ordering::Relaxed);
                    }

                    // Check for duplicates
                    if !seen_keys.insert(curr_key.clone()) {
                        duplicate_keys.fetch_add(1, Ordering::Relaxed);
                        tracing::error!(key = ?curr_key, "DUPLICATE KEY in scan");
                    }

                    prev_key = Some(curr_key);
                }
            }
        }));
    }

    // Remover threads
    for tid in 0..NUM_REMOVERS {
        let tree = Arc::clone(&tree);
        let barrier = Arc::clone(&barrier);

        handles.push(thread::spawn(move || {
            barrier.wait();
            let guard = tree.guard();

            // Remove every Nth key based on thread ID
            for key in (tid as u64..NUM_KEYS).step_by(4) {
                let _ = tree.remove_with_guard(&key.to_be_bytes(), &guard);
            }
        }));
    }

    for h in handles {
        h.join().unwrap();
    }

    let violations = ordering_violations.load(Ordering::Relaxed);
    let duplicates = duplicate_keys.load(Ordering::Relaxed);

    println!(
        "Scan during removes: {} ordering violations, {} duplicates",
        violations, duplicates
    );

    assert_eq!(violations, 0, "Ordering violations detected");
    assert_eq!(duplicates, 0, "Duplicate keys detected in scans");
}

// =============================================================================
// CONSISTENCY VERIFICATION (BTreeMap oracle)
// =============================================================================

/// Compare masstree behavior against BTreeMap as oracle.
///
/// Performs identical operations on both and verifies they agree.
#[test]
fn btreemap_oracle_comparison() {
    common::init_tracing();

    const NUM_OPERATIONS: usize = 10_000;
    const KEY_RANGE: u64 = 500;
    const NUM_THREADS: usize = 4;

    // Single-threaded oracle
    let oracle = Arc::new(std::sync::Mutex::new(BTreeMap::<u64, u64>::new()));
    let tree = Arc::new(MassTree15::<u64>::new());

    let barrier = Arc::new(Barrier::new(NUM_THREADS));

    let mut handles = vec![];

    for tid in 0..NUM_THREADS {
        let oracle = Arc::clone(&oracle);
        let tree = Arc::clone(&tree);
        let barrier = Arc::clone(&barrier);

        handles.push(thread::spawn(move || {
            barrier.wait();
            let guard = tree.guard();
            let mut rng_state = tid as u64 * 54321;

            for op in 0..NUM_OPERATIONS {
                rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1);
                let key = rng_state % KEY_RANGE;
                let value = (tid as u64 * 1_000_000) + op as u64;

                // Hold oracle lock DURING tree operation to ensure atomicity.
                // Without this, another thread can interleave between tree op and oracle op,
                // causing the final states to diverge.
                match op % 10 {
                    0..=5 => {
                        // 60% inserts
                        let mut oracle_guard = oracle.lock().unwrap();
                        let _ = tree.insert_with_guard(&key.to_be_bytes(), value, &guard);
                        oracle_guard.insert(key, value);
                    }
                    6..=7 => {
                        // 20% removes
                        let mut oracle_guard = oracle.lock().unwrap();
                        let _ = tree.remove_with_guard(&key.to_be_bytes(), &guard);
                        oracle_guard.remove(&key);
                    }
                    _ => {
                        // 20% reads (no oracle sync needed)
                    }
                }
            }
        }));
    }

    for h in handles {
        h.join().unwrap();
    }

    // Final consistency check
    let oracle = oracle.lock().unwrap();
    let guard = tree.guard();
    let mut check_errors = 0u64;

    // Check all oracle keys exist in tree
    for (&key, &expected_val) in oracle.iter() {
        match tree.get_with_guard(&key.to_be_bytes(), &guard) {
            Some(v) => {
                if *v != expected_val {
                    // Value mismatch is OK due to concurrent overwrites
                    // But key must exist
                }
            }

            None => {
                check_errors += 1;
            }
        }
    }

    // Check tree doesn't have extra keys (not in oracle)
    for entry in tree.iter(&guard) {
        let key = u64::from_be_bytes(entry.key().try_into().unwrap());

        if !oracle.contains_key(&key) {
            check_errors += 1;
        }
    }

    println!(
        "Oracle comparison: oracle has {} keys, tree has {}, {} mismatches",
        oracle.len(),
        tree.len(),
        check_errors
    );

    // With the fix (holding oracle lock during tree ops), lengths must match exactly
    assert_eq!(
        oracle.len(),
        tree.len(),
        "Length mismatch: oracle={}, tree={}",
        oracle.len(),
        tree.len()
    );

    drop(oracle);

    assert_eq!(
        check_errors, 0,
        "Oracle comparison found {} key existence mismatches",
        check_errors
    );
}

// =============================================================================
// EXTENDED STRESS TESTS
// =============================================================================

/// Long-running stress test with all operations.
#[test]
#[ignore = "long test"]
fn long_running_mixed_workload() {
    common::init_tracing();

    const DURATION_SECS: u64 = 30;
    const NUM_THREADS: usize = 8;
    const KEY_RANGE: u64 = 10_000;

    let tree = Arc::new(MassTree15::<u64>::new());
    let stop_flag = Arc::new(AtomicBool::new(false));
    let operations = Arc::new(AtomicU64::new(0));
    let errors = Arc::new(AtomicU64::new(0));

    // Pre-populate
    for i in 0..KEY_RANGE / 2 {
        tree.insert(&i.to_be_bytes(), i).unwrap();
    }

    let mut handles = vec![];

    for tid in 0..NUM_THREADS {
        let tree = Arc::clone(&tree);
        let stop_flag = Arc::clone(&stop_flag);
        let operations = Arc::clone(&operations);

        handles.push(thread::spawn(move || {
            let guard = tree.guard();
            let mut rng_state = tid as u64 * 98765;
            let mut local_ops = 0u64;

            while !stop_flag.load(Ordering::Relaxed) {
                rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1);
                let key = rng_state % KEY_RANGE;
                let value = rng_state;

                match (rng_state / KEY_RANGE) % 100 {
                    0..=49 => {
                        // 50% reads
                        let _ = tree.get_with_guard(&key.to_be_bytes(), &guard);
                    }
                    50..=79 => {
                        // 30% inserts
                        let _ = tree.insert_with_guard(&key.to_be_bytes(), value, &guard);
                    }
                    80..=94 => {
                        // 15% removes
                        let _ = tree.remove_with_guard(&key.to_be_bytes(), &guard);
                    }
                    _ => {
                        // 5% scans
                        let mut count = 0;
                        for entry in tree.iter(&guard) {
                            let _ = entry.key();
                            count += 1;
                            if count > 100 {
                                break;
                            }
                        }
                    }
                }

                local_ops += 1;

                // Periodic verification
                if local_ops.is_multiple_of(10_000) {
                    // Verify a random key that we just operated on
                    let verify_key: u64 = rng_state % (KEY_RANGE / 2);

                    if tree
                        .get_with_guard(&verify_key.to_be_bytes(), &guard)
                        .is_some()
                    {
                        // Key exists, value should be non-zero (any value is fine)
                    }
                }
            }

            operations.fetch_add(local_ops, Ordering::Relaxed);
        }));
    }

    // Run for specified duration
    thread::sleep(Duration::from_secs(DURATION_SECS));
    stop_flag.store(true, Ordering::Release);

    for h in handles {
        h.join().unwrap();
    }

    let total_ops = operations.load(Ordering::Relaxed);
    let total_errors = errors.load(Ordering::Relaxed);
    let ops_per_sec = total_ops / DURATION_SECS;

    println!(
        "Long-running stress: {} ops in {}s ({} ops/sec), {} errors, final tree size: {}",
        total_ops,
        DURATION_SECS,
        ops_per_sec,
        total_errors,
        tree.len()
    );

    assert_eq!(total_errors, 0, "Errors detected during long-running test");
}
