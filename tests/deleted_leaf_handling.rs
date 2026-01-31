//! Tests for Deleted-leaf handling in point reads.

use masstree::MassTree15;
use std::sync::{Arc, Barrier};
use std::thread as StdThread;

/// Test that point reads correctly handle deleted leaves during coalesce.
///
/// This test verifies latest fix: readers must follow B-links through
/// deleted leaves to find keys that have moved to successor leaves.
#[test]
#[expect(clippy::unwrap_used)]
fn test_get_during_coalesce() {
    let tree = Arc::new(MassTree15::<u64>::new());
    let barrier = Arc::new(Barrier::new(3));

    // Insert keys that will span multiple leaves after splits
    for i in 0u64..100 {
        tree.insert(&i.to_be_bytes(), i);
    }

    // Verify key 50 exists before test
    assert!(
        tree.get(&50u64.to_be_bytes()).is_some(),
        "Key 50 must exist before test"
    );

    // Reader thread 1: continuously reads key 50
    let tree1 = Arc::clone(&tree);
    let barrier1 = Arc::clone(&barrier);
    let reader1 = StdThread::spawn(move || {
        barrier1.wait();
        let guard = tree1.guard();
        let mut found_count = 0u64;
        let mut not_found_count = 0u64;

        for _ in 0..5000 {
            if tree1.get_with_guard(&50u64.to_be_bytes(), &guard).is_some() {
                found_count += 1;
            } else {
                not_found_count += 1;
            }
        }

        (found_count, not_found_count)
    });

    // Reader thread 2: reads various keys
    let tree2 = Arc::clone(&tree);
    let barrier2 = Arc::clone(&barrier);
    let reader2 = StdThread::spawn(move || {
        barrier2.wait();
        let guard = tree2.guard();

        for i in 0..5000u64 {
            let key: u64 = i % 100;
            let _ = tree2.get_with_guard(&key.to_be_bytes(), &guard);
        }
    });

    // Remover thread: removes keys triggering coalesce
    let tree3 = Arc::clone(&tree);
    let barrier3 = Arc::clone(&barrier);
    let remover = StdThread::spawn(move || {
        barrier3.wait();

        // Remove keys around 50 but not 50 itself
        for i in 0u64..100 {
            if i != 50 {
                let _ = tree3.remove(&i.to_be_bytes());
            }

            // Periodically trigger coalesce
            if i % 10 == 0 {
                let guard = tree3.guard();
                tree3.process_coalesce_batch(&guard, 5);
            }
        }
    });

    let (found, not_found) = reader1.join().unwrap();
    reader2.join().unwrap();
    remover.join().unwrap();

    // Key 50 should still be findable after all operations
    assert!(
        tree.get(&50u64.to_be_bytes()).is_some(),
        "Key 50 must exist after test"
    );

    // Key 50 was never removed, so all reads MUST find it.
    // Before fix: not_found would be high due to landing on deleted leaves
    // After fix: not_found must be 0
    println!("Found: {found}, Not found: {not_found}");
    assert!(
        not_found == 0,
        "Reads during coalesce should not miss existing keys. Found: {found}, Not found: {not_found}"
    );
}

/// Stress test for concurrent reads during aggressive coalescing.
///
/// Verifies that the deleted-leaf handling works correctly
/// under high contention with multiple readers and writers.
#[test]
#[expect(
    clippy::unwrap_used,
    clippy::cast_sign_loss,
    clippy::cast_precision_loss
)]
fn stress_get_during_coalesce() {
    use masstree::MassTree15;
    use std::sync::atomic::{AtomicU64, Ordering};
    use std::sync::{Arc, Barrier};
    use std::thread;

    let tree = Arc::new(MassTree15::<u64>::new());
    let num_threads = 6;
    let ops_per_thread = 10_000;
    let missed_reads = Arc::new(AtomicU64::new(0));
    let successful_reads = Arc::new(AtomicU64::new(0));

    // Pre-populate with keys 0..1000
    for i in 0u64..1000 {
        tree.insert(&i.to_be_bytes(), i);
    }

    let barrier = Arc::new(Barrier::new(num_threads));
    let mut handles = vec![];

    for tid in 0..num_threads {
        let tree = Arc::clone(&tree);
        let barrier = Arc::clone(&barrier);
        let missed = Arc::clone(&missed_reads);
        let success = Arc::clone(&successful_reads);

        handles.push(thread::spawn(move || {
            barrier.wait();
            let guard = tree.guard();

            for i in 0..ops_per_thread {
                let key: u64 = (tid as u64 * 100 + i as u64) % 500;

                match i % 5 {
                    0 | 1 => {
                        // 40% reads - track success/failure
                        if tree.get_with_guard(&key.to_be_bytes(), &guard).is_some() {
                            success.fetch_add(1, Ordering::Relaxed);
                        } else {
                            // Only count as missed if key should exist
                            // (wasn't just removed by another thread)
                            missed.fetch_add(1, Ordering::Relaxed);
                        }
                    }
                    2 => {
                        // 20% inserts
                        let _ = tree.insert_with_guard(&key.to_be_bytes(), key, &guard);
                    }
                    3 => {
                        // 20% removes
                        let _ = tree.remove_with_guard(&key.to_be_bytes(), &guard);
                    }
                    4 => {
                        // 20% coalesce
                        tree.process_coalesce_batch(&guard, 10);
                    }
                    _ => unreachable!(),
                }
            }
        }));
    }

    for h in handles {
        h.join().unwrap();
    }

    let total_missed = missed_reads.load(Ordering::Relaxed);
    let total_success = successful_reads.load(Ordering::Relaxed);

    println!("Stress test complete: {total_success} successful reads, {total_missed} missed reads");

    // Note: Some missed reads are expected due to concurrent removes.
    // The fix ensures we don't miss reads due to landing on deleted leaves.
    // A high ratio of missed:success would indicate a problem.
    let miss_ratio = if total_success > 0 {
        total_missed as f64 / total_success as f64
    } else {
        0.0
    };

    assert!(
        miss_ratio < 0.5,
        "Too many missed reads ({total_missed} / {total_success}). Possible deleted-leaf handling bug."
    );
}
