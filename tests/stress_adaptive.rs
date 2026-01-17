//! Stress tests for adaptive insert.
//!
//! These are long-running tests marked `#[ignore]`.
//! Run with: `cargo test --test stress_adaptive -- --ignored`
#![expect(clippy::unwrap_used, clippy::cast_precision_loss)]

use masstree::MassTree24;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::thread;
use std::time::Duration;

const THREADS: usize = 8;
const DURATION_SECS: u64 = 5;
const ITERATIONS: usize = 50;

/// Stress test with extreme contention.
///
/// All threads hit the same key prefix. Tests that:
/// - No deadlock occurs
/// - Progress is made (some operations complete)
///
/// # Miri Compatibility
///
/// When running under Miri (10-100x slower), the threshold is lowered
/// significantly to avoid false failures.
#[test]
#[ignore = "Long-running stress test"]
fn stress_extreme_contention() {
    let tree: MassTree24<u64> = MassTree24::new();
    let tree = Arc::new(tree);
    let stop = Arc::new(AtomicBool::new(false));
    let ops = Arc::new(AtomicUsize::new(0));

    let handles: Vec<_> = (0..THREADS)
        .map(|tid| {
            let tree = Arc::clone(&tree);
            let stop = Arc::clone(&stop);
            let ops = Arc::clone(&ops);

            thread::spawn(move || {
                let mut i = 0u64;
                while !stop.load(Ordering::Relaxed) {
                    // Maximum contention: same prefix, rotating through small key set
                    let key = format!("CONTENTION__{:08}", i % 1000);
                    let _ = tree.insert(key.as_bytes(), tid as u64 * 1_000_000 + i);
                    ops.fetch_add(1, Ordering::Relaxed);
                    i += 1;
                }
            })
        })
        .collect();

    thread::sleep(Duration::from_secs(DURATION_SECS));
    stop.store(true, Ordering::Relaxed);

    for h in handles {
        h.join().unwrap();
    }

    let total_ops = ops.load(Ordering::Relaxed);
    println!(
        "Extreme contention: {} ops in {}s ({:.0} ops/sec)",
        total_ops,
        DURATION_SECS,
        total_ops as f64 / DURATION_SECS as f64
    );

    // Must complete some operations (no deadlock)
    // Threshold is environment-aware:
    // - Miri: 10 ops (extremely slow, 10-100x overhead)
    // - Normal: 100 ops (even slow CI systems should manage this)
    let min_ops = if cfg!(miri) { 10 } else { 100 };
    assert!(
        total_ops > min_ops,
        "Should complete some operations (got {total_ops}, min {min_ops})"
    );
}

/// Test rapid alternation between contended and non-contended patterns.
#[test]
fn test_mode_switching() {
    let tree: MassTree24<u64> = MassTree24::new();
    let tree = Arc::new(tree);

    let handles: Vec<_> = (0..THREADS)
        .map(|tid| {
            let tree = Arc::clone(&tree);
            thread::spawn(move || {
                for iter in 0..ITERATIONS {
                    if iter % 2 == 0 {
                        // Contended
                        for i in 0..10 {
                            let key = format!("shared__{i:08}");
                            let _ = tree.insert(key.as_bytes(), tid as u64);
                        }
                    } else {
                        // Non-contended
                        for i in 0..10 {
                            let key = format!("thread{}_{:08}", tid, iter * 10 + i);
                            let _ = tree.insert(key.as_bytes(), tid as u64);
                        }
                    }
                }
            })
        })
        .collect();

    for h in handles {
        h.join().unwrap();
    }

    // Just verify no crash/deadlock and tree is usable
    assert!(!tree.is_empty());
}
