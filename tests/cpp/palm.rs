//! PALM: Large-scale insert and batched read tests.
//!
//! Port of `kvtest_palma` and `kvtest_palmb` from C++ `kvtest.hh`.
//!
//! Pattern:
//! - palma: Insert many sequential keys
//! - palmb: Batched random reads with sorted access pattern
//! - Tests large dataset performance

#![allow(clippy::unwrap_used)]

use masstree::MassTree15Inline as MassTree24Inline;
use rand::{Rng, SeedableRng, rngs::StdRng};
use std::sync::Arc;
use std::thread;

const SEED: u64 = 31949;
// Reduced from C++ PALMN (128M) for reasonable test time
const PALM_N: u64 = 100_000;
const PALM_BATCH: usize = 341; // 8192 / 24

#[test]
fn palma_sequential_insert() {
    let tree: MassTree24Inline<u64> = MassTree24Inline::new();
    let guard = tree.guard();

    // Sequential inserts
    for i in 0..PALM_N {
        let key = i.to_be_bytes();
        tree.insert_with_guard(&key, i + 1, &guard).unwrap();
    }

    // Verify sample
    for i in (0..PALM_N).step_by(1000) {
        let key = i.to_be_bytes();
        let val = tree.get_with_guard(&key, &guard);
        assert_eq!(val, Some(i + 1));
    }
}

#[test]
fn palmb_batched_reads() {
    let tree: MassTree24Inline<u64> = MassTree24Inline::new();
    let guard = tree.guard();

    // Pre-populate (like palma)
    let read_range = PALM_N / 10; // palmb reads from smaller range
    for i in 0..read_range {
        let key = i.to_be_bytes();
        tree.insert_with_guard(&key, i + 1, &guard).unwrap();
    }

    // Batched reads with sorting (cache-friendly pattern)
    let mut rng = StdRng::seed_from_u64(SEED);
    let mut batch: Vec<u64> = Vec::with_capacity(PALM_BATCH);

    for _ in 0..10 {
        // 10 batches
        batch.clear();

        // Fill batch with random keys
        for _ in 0..PALM_BATCH {
            let x = rng.random_range(0..read_range);
            batch.push(x);
        }

        // Sort for cache-friendly access
        batch.sort_unstable();

        // Read in sorted order
        for &x in &batch {
            let key = x.to_be_bytes();
            let val = tree.get_with_guard(&key, &guard);
            assert_eq!(val, Some(x + 1), "key {x} mismatch");
        }
    }
}

#[test]
#[expect(clippy::cast_sign_loss)]
fn palm_concurrent_insert() {
    let tree = Arc::new(MassTree24Inline::<u64>::new());
    let num_threads = 4;
    let per_thread = PALM_N / num_threads as u64;

    let handles: Vec<_> = (0..num_threads)
        .map(|tid| {
            let tree = Arc::clone(&tree);
            thread::spawn(move || {
                let guard = tree.guard();
                let start = tid as u64 * per_thread;
                let end = start + per_thread;

                for i in start..end {
                    let key = i.to_be_bytes();
                    let _ = tree.insert_with_guard(&key, i + 1, &guard);
                }
            })
        })
        .collect();

    for h in handles {
        h.join().unwrap();
    }

    // Verify all
    let guard = tree.guard();
    for i in 0..PALM_N {
        let key = i.to_be_bytes();
        let val = tree.get_with_guard(&key, &guard);
        assert_eq!(val, Some(i + 1), "key {i} missing");
    }
}

#[test]
fn palm_concurrent_read() {
    let tree = Arc::new(MassTree24Inline::<u64>::new());

    // Pre-populate
    {
        let guard = tree.guard();
        let read_range = PALM_N / 10;
        for i in 0..read_range {
            let key = i.to_be_bytes();
            tree.insert_with_guard(&key, i + 1, &guard).unwrap();
        }
    }

    let num_threads = 4;
    let read_range = PALM_N / 10;

    #[expect(clippy::cast_sign_loss)]
    let handles: Vec<_> = (0..num_threads)
        .map(|tid| {
            let tree = Arc::clone(&tree);

            thread::spawn(move || {
                let guard = tree.guard();
                let mut rng = StdRng::seed_from_u64(SEED + tid as u64);
                let mut batch: Vec<u64> = Vec::with_capacity(PALM_BATCH);

                for _ in 0..5 {
                    batch.clear();

                    for _ in 0..PALM_BATCH {
                        batch.push(rng.random_range(0..read_range));
                    }

                    batch.sort_unstable();

                    for &x in &batch {
                        let key = x.to_be_bytes();
                        let val = tree.get_with_guard(&key, &guard);
                        assert_eq!(val, Some(x + 1));
                    }
                }
            })
        })
        .collect();

    for h in handles {
        h.join().unwrap();
    }
}
