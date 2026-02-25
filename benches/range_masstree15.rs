//! Concurrent range scan stress benchmarks for MassTree15Inline.

#![expect(clippy::unwrap_used)]
#![expect(clippy::pedantic)]
#![expect(clippy::indexing_slicing)]

mod bench_utils;

use bench_utils::{
    keys, keys_adversarial_splits, keys_blink_stress, keys_clustered, keys_hierarchical,
    keys_interleaved_ranges, keys_reverse, keys_sequential, keys_shared_prefix, keys_sparse,
    keys_suffix_only_differ, post_measurement_barrier, pre_measurement_barrier,
};
use divan::{Bencher, black_box};
use masstree::{MassTree15Inline, RangeBound};
use scc::TreeIndex;
use std::sync::Arc;
use std::sync::Barrier;
use std::thread;

fn main() {
    divan::main();
}

// =============================================================================
// Constants
// =============================================================================

const N: usize = 500_000;
const OPS_PER_THREAD: usize = 5_000;
const SCAN_LIMIT: usize = 50; // Early termination for partial scans

/// Warmup iterations per thread before measurement
const WARMUP_OPS: usize = 500;

/// Base seed for RNG (ensures reproducibility across runs)
const BASE_SEED: u64 = 0xDEAD_BEEF_CAFE_BABE;

// =============================================================================
// RNG Helpers (for independent thread randomness)
// =============================================================================

/// Generate divergent seed for thread t to avoid correlation.
/// Uses multiplicative hashing to spread seeds across the space.
const fn thread_seed(base_seed: u64, thread_id: usize) -> u64 {
    let combined = base_seed.wrapping_add(thread_id as u64);
    // Mix with golden ratio hash
    combined.wrapping_mul(0x9e37_79b9_7f4a_7c15)
}

/// Simple LCG step for inline RNG.
#[inline]
#[allow(dead_code)]
const fn lcg_next(state: &mut u64) -> u64 {
    *state = state
        .wrapping_mul(6_364_136_223_846_793_005)
        .wrapping_add(1);
    *state
}

/// Standard thread counts for all benchmarks
const THREAD_COUNTS: [usize; 6] = [1, 2, 4, 6, 8, 12];

// =============================================================================
// Setup Helpers
// =============================================================================

fn setup_masstree15_inline<const K: usize>(keys: &[[u8; K]]) -> MassTree15Inline<u64> {
    let tree = MassTree15Inline::new();
    {
        let guard = tree.guard();
        for (i, key) in keys.iter().enumerate() {
            let _ = tree.insert_with_guard(key, i as u64, &guard);
        }
    }
    tree
}

fn setup_tree_index<const K: usize>(keys: &[[u8; K]]) -> TreeIndex<[u8; K], u64> {
    let tree = TreeIndex::new();
    for (i, key) in keys.iter().enumerate() {
        let _ = tree.insert_sync(*key, i as u64);
    }
    tree
}

/// TreeIndex upsert workaround using remove+insert.
/// NOTE: This adds overhead compared to MassTree's native upsert since TreeIndex
/// lacks a true upsert operation. The extra remove call on collision means
/// TreeIndex benchmarks measure remove+insert rather than a single operation.
/// This is a fairness consideration when comparing results.
fn tree_index_upsert_sync<const K: usize>(
    tree: &TreeIndex<[u8; K], u64>,
    key: [u8; K],
    value: u64,
) {
    let mut key = key;
    let mut value = value;
    for _ in 0..3 {
        match tree.insert_sync(key, value) {
            Ok(()) => return,
            Err((k, v)) => {
                tree.remove_sync(&k);
                key = k;
                value = v;
            }
        }
    }
    let _ = tree.insert_sync(key, value);
}

// =============================================================================
// 00: SNAPSHOT VERIFY (READ-ONLY) - Deterministic prefix scan
// =============================================================================
//
// Verifies that each scan returns the expected key/value sequence for a stable
// (read-only) dataset. This is a "snapshot" check in the sense that the dataset
// is not mutating; it does not attempt to validate linearizable snapshot
// semantics under concurrent writers.
//
// Keep this group small so it remains fast for both implementations.

#[divan::bench_group(name = "00_snapshot_verify_sequential_prefix", sample_count = 200)]
mod snapshot_verify_sequential_prefix {
    use super::*;

    const VERIFY_N: usize = 50_000;
    const VERIFY_OPS_PER_THREAD: usize = 1_000;

    // Verify a fixed, deterministic prefix so we can validate both key order and values
    // without allocating.
    const VERIFY_SCAN_LIMIT: usize = 50;

    fn expected_prefix<const K: usize>(keys: &[[u8; K]]) -> Vec<([u8; K], u64)> {
        keys.iter()
            .take(VERIFY_SCAN_LIMIT)
            .enumerate()
            .map(|(i, k)| (*k, i as u64))
            .collect()
    }

    #[divan::bench(args = THREAD_COUNTS)]
    fn masstree15_inline(bencher: Bencher, threads: usize) {
        let keys = Arc::new(keys_sequential::<8>(VERIFY_N));
        let expected = Arc::new(expected_prefix(keys.as_ref()));
        let tree = Arc::new(setup_masstree15_inline(keys.as_ref()));

        bencher
            .counter(divan::counter::ItemsCount::new(
                threads * VERIFY_OPS_PER_THREAD,
            ))
            .bench_local(|| {
                let warmup_done = Arc::new(Barrier::new(threads));
                let start = Arc::new(Barrier::new(threads));
                let handles: Vec<_> = (0..threads)
                    .map(|_| {
                        let tree = Arc::clone(&tree);
                        let expected = Arc::clone(&expected);
                        let warmup_done = Arc::clone(&warmup_done);
                        let start = Arc::clone(&start);
                        thread::spawn(move || {
                            let guard = tree.guard();

                            // === WARMUP PHASE ===
                            for _ in 0..WARMUP_OPS {
                                let mut seen = 0usize;
                                let visited =
                                    tree.iter(&guard).for_each_intra_leaf_batch(|k, v| {
                                        if seen < VERIFY_SCAN_LIMIT {
                                            let (exp_k, exp_v) = expected[seen];
                                            assert_eq!(k, exp_k.as_slice());
                                            assert_eq!(v, exp_v);
                                            seen += 1;
                                            seen < VERIFY_SCAN_LIMIT
                                        } else {
                                            false
                                        }
                                    });
                                black_box(visited);
                            }
                            warmup_done.wait();

                            // === MEASUREMENT PHASE ===
                            start.wait();
                            pre_measurement_barrier();

                            for _ in 0..VERIFY_OPS_PER_THREAD {
                                let mut seen = 0usize;
                                let visited =
                                    tree.iter(&guard).for_each_intra_leaf_batch(|k, v| {
                                        if seen < VERIFY_SCAN_LIMIT {
                                            let (exp_k, exp_v) = expected[seen];
                                            assert_eq!(k, exp_k.as_slice());
                                            assert_eq!(v, exp_v);
                                            seen += 1;
                                            seen < VERIFY_SCAN_LIMIT
                                        } else {
                                            false
                                        }
                                    });
                                black_box(visited);
                                assert_eq!(seen, VERIFY_SCAN_LIMIT);
                            }

                            post_measurement_barrier();
                        })
                    })
                    .collect();

                for h in handles {
                    h.join().unwrap();
                }
            });
    }

    #[divan::bench(args = THREAD_COUNTS)]
    fn tree_index(bencher: Bencher, threads: usize) {
        let keys = Arc::new(keys_sequential::<8>(VERIFY_N));
        let expected = Arc::new(expected_prefix(keys.as_ref()));
        let tree = Arc::new(setup_tree_index(keys.as_ref()));

        bencher
            .counter(divan::counter::ItemsCount::new(
                threads * VERIFY_OPS_PER_THREAD,
            ))
            .bench_local(|| {
                let warmup_done = Arc::new(Barrier::new(threads));
                let start = Arc::new(Barrier::new(threads));
                let handles: Vec<_> = (0..threads)
                    .map(|_| {
                        let tree = Arc::clone(&tree);
                        let expected = Arc::clone(&expected);
                        let warmup_done = Arc::clone(&warmup_done);
                        let start = Arc::clone(&start);
                        thread::spawn(move || {
                            let guard = sdd::Guard::new();

                            // === WARMUP PHASE ===
                            for _ in 0..WARMUP_OPS {
                                for (i, (k, v)) in
                                    tree.iter(&guard).take(VERIFY_SCAN_LIMIT).enumerate()
                                {
                                    let (exp_k, exp_v) = expected[i];
                                    assert_eq!(k, &exp_k);
                                    assert_eq!(*v, exp_v);
                                }
                            }
                            warmup_done.wait();

                            // === MEASUREMENT PHASE ===
                            start.wait();
                            pre_measurement_barrier();

                            for _ in 0..VERIFY_OPS_PER_THREAD {
                                for (i, (k, v)) in
                                    tree.iter(&guard).take(VERIFY_SCAN_LIMIT).enumerate()
                                {
                                    let (exp_k, exp_v) = expected[i];
                                    assert_eq!(k, &exp_k);
                                    assert_eq!(*v, exp_v);
                                }
                            }

                            post_measurement_barrier();
                        })
                    })
                    .collect();

                for h in handles {
                    h.join().unwrap();
                }
            });
    }
}

// =============================================================================
// 01: SEQUENTIAL KEYS - Best case for range scans
// =============================================================================

#[divan::bench_group(name = "01_sequential_full_scan", sample_count = 200)]
mod sequential_full_scan {
    use super::*;

    // Both implementations scan from the beginning for fair comparison
    // (TreeIndex's range() method hangs under concurrent load)
    #[divan::bench(args = THREAD_COUNTS)]
    fn masstree15_inline(bencher: Bencher, threads: usize) {
        let keys = Arc::new(keys_sequential::<8>(N));
        let tree = Arc::new(setup_masstree15_inline(&keys));

        bencher
            .counter(divan::counter::ItemsCount::new(threads * OPS_PER_THREAD))
            .bench_local(|| {
                let warmup_done = Arc::new(Barrier::new(threads));
                let start = Arc::new(Barrier::new(threads));
                let handles: Vec<_> = (0..threads)
                    .map(|_| {
                        let tree = Arc::clone(&tree);
                        let warmup_done = Arc::clone(&warmup_done);
                        let start = Arc::clone(&start);
                        thread::spawn(move || {
                            let guard = tree.guard();

                            // === WARMUP PHASE ===
                            for _ in 0..WARMUP_OPS {
                                let mut count = 0usize;
                                tree.scan_values(
                                    RangeBound::Unbounded,
                                    RangeBound::Unbounded,
                                    |v| {
                                        black_box(v);
                                        count += 1;
                                        count < SCAN_LIMIT
                                    },
                                    &guard,
                                );
                                black_box(count);
                            }
                            warmup_done.wait();

                            // === MEASUREMENT PHASE ===
                            let mut total = 0usize;
                            start.wait();
                            pre_measurement_barrier();

                            for _ in 0..OPS_PER_THREAD {
                                let mut count = 0usize;
                                tree.scan_values(
                                    RangeBound::Unbounded,
                                    RangeBound::Unbounded,
                                    |v| {
                                        black_box(v);
                                        count += 1;
                                        count < SCAN_LIMIT
                                    },
                                    &guard,
                                );
                                total += count;
                            }

                            post_measurement_barrier();
                            black_box(total);
                        })
                    })
                    .collect();

                for h in handles {
                    h.join().unwrap();
                }
            });
    }

    // NOTE: TreeIndex uses iter() from beginning instead of range() with random starts
    // because TreeIndex's range() method hangs under concurrent load. This means
    // MassTree gets random start positions while TreeIndex always starts from minimum.
    #[divan::bench(args = THREAD_COUNTS)]
    fn tree_index(bencher: Bencher, threads: usize) {
        let keys = Arc::new(keys_sequential::<8>(N));
        let tree = Arc::new(setup_tree_index(&keys));

        bencher
            .counter(divan::counter::ItemsCount::new(threads * OPS_PER_THREAD))
            .bench_local(|| {
                let warmup_done = Arc::new(Barrier::new(threads));
                let start = Arc::new(Barrier::new(threads));
                let handles: Vec<_> = (0..threads)
                    .map(|_| {
                        let tree = Arc::clone(&tree);
                        let warmup_done = Arc::clone(&warmup_done);
                        let start = Arc::clone(&start);
                        thread::spawn(move || {
                            let guard = sdd::Guard::new();

                            // === WARMUP PHASE ===
                            for _ in 0..WARMUP_OPS {
                                let count = tree
                                    .iter(&guard)
                                    .take(SCAN_LIMIT)
                                    .inspect(|(_, v)| {
                                        black_box(*v);
                                    })
                                    .count();
                                black_box(count);
                            }
                            warmup_done.wait();

                            // === MEASUREMENT PHASE ===
                            let mut total = 0usize;
                            start.wait();
                            pre_measurement_barrier();

                            for _ in 0..OPS_PER_THREAD {
                                let count = tree
                                    .iter(&guard)
                                    .take(SCAN_LIMIT)
                                    .inspect(|(_, v)| {
                                        black_box(*v);
                                    })
                                    .count();
                                total += count;
                            }

                            post_measurement_barrier();
                            black_box(total);
                        })
                    })
                    .collect();

                for h in handles {
                    h.join().unwrap();
                }
            });
    }
}

// =============================================================================
// 02: REVERSE KEYS - Insertion stress pattern
// =============================================================================

#[divan::bench_group(name = "02_reverse_scan", sample_count = 200)]
mod reverse_scan {
    use super::*;

    // Both implementations scan from the beginning for fair comparison
    #[divan::bench(args = THREAD_COUNTS)]
    fn masstree15_inline(bencher: Bencher, threads: usize) {
        let keys = Arc::new(keys_reverse::<8>(N));
        let tree = Arc::new(setup_masstree15_inline(&keys));

        bencher
            .counter(divan::counter::ItemsCount::new(threads * OPS_PER_THREAD))
            .bench_local(|| {
                let warmup_done = Arc::new(Barrier::new(threads));
                let start = Arc::new(Barrier::new(threads));
                let handles: Vec<_> = (0..threads)
                    .map(|_| {
                        let tree = Arc::clone(&tree);
                        let warmup_done = Arc::clone(&warmup_done);
                        let start = Arc::clone(&start);

                        thread::spawn(move || {
                            let guard = tree.guard();

                            // === WARMUP PHASE ===
                            for _ in 0..WARMUP_OPS {
                                let mut count = 0usize;
                                tree.scan_values(
                                    RangeBound::Unbounded,
                                    RangeBound::Unbounded,
                                    |v| {
                                        black_box(v);
                                        count += 1;
                                        count < SCAN_LIMIT
                                    },
                                    &guard,
                                );
                                black_box(count);
                            }
                            warmup_done.wait();

                            // === MEASUREMENT PHASE ===
                            let mut total = 0usize;
                            start.wait();
                            pre_measurement_barrier();

                            for _ in 0..OPS_PER_THREAD {
                                let mut count = 0usize;
                                tree.scan_values(
                                    RangeBound::Unbounded,
                                    RangeBound::Unbounded,
                                    |v| {
                                        black_box(v);
                                        count += 1;
                                        count < SCAN_LIMIT
                                    },
                                    &guard,
                                );
                                total += count;
                            }

                            post_measurement_barrier();
                            black_box(total);
                        })
                    })
                    .collect();

                for h in handles {
                    h.join().unwrap();
                }
            });
    }

    // NOTE: TreeIndex uses iter() from beginning instead of range() with random starts
    // because TreeIndex's range() method hangs under concurrent load.
    #[divan::bench(args = THREAD_COUNTS)]
    fn tree_index(bencher: Bencher, threads: usize) {
        let keys = Arc::new(keys_reverse::<8>(N));
        let tree = Arc::new(setup_tree_index(&keys));

        bencher
            .counter(divan::counter::ItemsCount::new(threads * OPS_PER_THREAD))
            .bench_local(|| {
                let warmup_done = Arc::new(Barrier::new(threads));
                let start = Arc::new(Barrier::new(threads));
                let handles: Vec<_> = (0..threads)
                    .map(|_| {
                        let tree = Arc::clone(&tree);
                        let warmup_done = Arc::clone(&warmup_done);
                        let start = Arc::clone(&start);
                        thread::spawn(move || {
                            let guard = sdd::Guard::new();

                            // === WARMUP PHASE ===
                            for _ in 0..WARMUP_OPS {
                                let count = tree
                                    .iter(&guard)
                                    .take(SCAN_LIMIT)
                                    .inspect(|(_, v)| {
                                        black_box(*v);
                                    })
                                    .count();
                                black_box(count);
                            }
                            warmup_done.wait();

                            // === MEASUREMENT PHASE ===
                            let mut total = 0usize;
                            start.wait();
                            pre_measurement_barrier();

                            for _ in 0..OPS_PER_THREAD {
                                let count = tree
                                    .iter(&guard)
                                    .take(SCAN_LIMIT)
                                    .inspect(|(_, v)| {
                                        black_box(*v);
                                    })
                                    .count();
                                total += count;
                            }

                            post_measurement_barrier();
                            black_box(total);
                        })
                    })
                    .collect();

                for h in handles {
                    h.join().unwrap();
                }
            });
    }
}

// =============================================================================
// 03: CLUSTERED KEYS - Hot ranges with gaps
// =============================================================================

#[divan::bench_group(name = "03_clustered_scan", sample_count = 200)]
mod clustered_scan {
    use super::*;

    const CLUSTERS: usize = 500;
    const KEYS_PER_CLUSTER: usize = N / CLUSTERS;
    const GAP_SIZE: u64 = 10_000;

    // Both implementations scan from the beginning for fair comparison
    #[divan::bench(args = THREAD_COUNTS)]
    fn masstree15_inline(bencher: Bencher, threads: usize) {
        let keys = Arc::new(keys_clustered::<8>(CLUSTERS, KEYS_PER_CLUSTER, GAP_SIZE));
        let tree = Arc::new(setup_masstree15_inline(&keys));

        bencher
            .counter(divan::counter::ItemsCount::new(threads * OPS_PER_THREAD))
            .bench_local(|| {
                let warmup_done = Arc::new(Barrier::new(threads));
                let start = Arc::new(Barrier::new(threads));
                let handles: Vec<_> = (0..threads)
                    .map(|_| {
                        let tree = Arc::clone(&tree);
                        let warmup_done = Arc::clone(&warmup_done);
                        let start = Arc::clone(&start);
                        thread::spawn(move || {
                            let guard = tree.guard();

                            // === WARMUP PHASE ===
                            for _ in 0..WARMUP_OPS {
                                let mut count = 0usize;
                                tree.scan_values(
                                    RangeBound::Unbounded,
                                    RangeBound::Unbounded,
                                    |v| {
                                        black_box(v);
                                        count += 1;
                                        count < SCAN_LIMIT
                                    },
                                    &guard,
                                );
                                black_box(count);
                            }
                            warmup_done.wait();

                            // === MEASUREMENT PHASE ===
                            let mut total = 0usize;
                            start.wait();
                            pre_measurement_barrier();

                            for _ in 0..OPS_PER_THREAD {
                                let mut count = 0usize;
                                tree.scan_values(
                                    RangeBound::Unbounded,
                                    RangeBound::Unbounded,
                                    |v| {
                                        black_box(v);
                                        count += 1;
                                        count < SCAN_LIMIT
                                    },
                                    &guard,
                                );
                                total += count;
                            }

                            post_measurement_barrier();
                            black_box(total);
                        })
                    })
                    .collect();

                for h in handles {
                    h.join().unwrap();
                }
            });
    }

    // NOTE: TreeIndex uses iter() from beginning instead of range() with random starts
    // because TreeIndex's range() method hangs under concurrent load.
    #[divan::bench(args = THREAD_COUNTS)]
    fn tree_index(bencher: Bencher, threads: usize) {
        let keys = Arc::new(keys_clustered::<8>(CLUSTERS, KEYS_PER_CLUSTER, GAP_SIZE));
        let tree = Arc::new(setup_tree_index(&keys));

        bencher
            .counter(divan::counter::ItemsCount::new(threads * OPS_PER_THREAD))
            .bench_local(|| {
                let warmup_done = Arc::new(Barrier::new(threads));
                let start = Arc::new(Barrier::new(threads));
                let handles: Vec<_> = (0..threads)
                    .map(|_| {
                        let tree = Arc::clone(&tree);
                        let warmup_done = Arc::clone(&warmup_done);
                        let start = Arc::clone(&start);
                        thread::spawn(move || {
                            let guard = sdd::Guard::new();

                            // === WARMUP PHASE ===
                            for _ in 0..WARMUP_OPS {
                                let count = tree
                                    .iter(&guard)
                                    .take(SCAN_LIMIT)
                                    .inspect(|(_, v)| {
                                        black_box(*v);
                                    })
                                    .count();
                                black_box(count);
                            }
                            warmup_done.wait();

                            // === MEASUREMENT PHASE ===
                            let mut total = 0usize;
                            start.wait();
                            pre_measurement_barrier();

                            for _ in 0..OPS_PER_THREAD {
                                let count = tree
                                    .iter(&guard)
                                    .take(SCAN_LIMIT)
                                    .inspect(|(_, v)| {
                                        black_box(*v);
                                    })
                                    .count();
                                total += count;
                            }

                            post_measurement_barrier();
                            black_box(total);
                        })
                    })
                    .collect();

                for h in handles {
                    h.join().unwrap();
                }
            });
    }
}

// =============================================================================
// 04: SPARSE KEYS - Cache miss stress
// =============================================================================

#[divan::bench_group(name = "04_sparse_scan", sample_count = 200)]
mod sparse_scan {
    use super::*;

    const SPACING: u64 = 1000;

    // Both implementations scan from the beginning for fair comparison
    #[divan::bench(args = THREAD_COUNTS)]
    fn masstree15_inline(bencher: Bencher, threads: usize) {
        let keys = Arc::new(keys_sparse::<8>(N, SPACING));
        let tree = Arc::new(setup_masstree15_inline(&keys));

        bencher
            .counter(divan::counter::ItemsCount::new(threads * OPS_PER_THREAD))
            .bench_local(|| {
                let warmup_done = Arc::new(Barrier::new(threads));
                let start = Arc::new(Barrier::new(threads));
                let handles: Vec<_> = (0..threads)
                    .map(|_| {
                        let tree = Arc::clone(&tree);
                        let warmup_done = Arc::clone(&warmup_done);
                        let start = Arc::clone(&start);
                        thread::spawn(move || {
                            let guard = tree.guard();

                            // === WARMUP PHASE ===
                            for _ in 0..WARMUP_OPS {
                                let mut count = 0usize;
                                tree.scan_values(
                                    RangeBound::Unbounded,
                                    RangeBound::Unbounded,
                                    |v| {
                                        black_box(v);
                                        count += 1;
                                        count < SCAN_LIMIT
                                    },
                                    &guard,
                                );
                                black_box(count);
                            }
                            warmup_done.wait();

                            // === MEASUREMENT PHASE ===
                            let mut total = 0usize;
                            start.wait();
                            pre_measurement_barrier();

                            for _ in 0..OPS_PER_THREAD {
                                let mut count = 0usize;
                                tree.scan_values(
                                    RangeBound::Unbounded,
                                    RangeBound::Unbounded,
                                    |v| {
                                        black_box(v);
                                        count += 1;
                                        count < SCAN_LIMIT
                                    },
                                    &guard,
                                );
                                total += count;
                            }

                            post_measurement_barrier();
                            black_box(total);
                        })
                    })
                    .collect();

                for h in handles {
                    h.join().unwrap();
                }
            });
    }

    // NOTE: TreeIndex uses iter() from beginning instead of range() with random starts
    // because TreeIndex's range() method hangs under concurrent load.
    #[divan::bench(args = THREAD_COUNTS)]
    fn tree_index(bencher: Bencher, threads: usize) {
        let keys = Arc::new(keys_sparse::<8>(N, SPACING));
        let tree = Arc::new(setup_tree_index(&keys));

        bencher
            .counter(divan::counter::ItemsCount::new(threads * OPS_PER_THREAD))
            .bench_local(|| {
                let warmup_done = Arc::new(Barrier::new(threads));
                let start = Arc::new(Barrier::new(threads));
                let handles: Vec<_> = (0..threads)
                    .map(|_| {
                        let tree = Arc::clone(&tree);
                        let warmup_done = Arc::clone(&warmup_done);
                        let start = Arc::clone(&start);
                        thread::spawn(move || {
                            let guard = sdd::Guard::new();

                            // === WARMUP PHASE ===
                            for _ in 0..WARMUP_OPS {
                                let count = tree
                                    .iter(&guard)
                                    .take(SCAN_LIMIT)
                                    .inspect(|(_, v)| {
                                        black_box(*v);
                                    })
                                    .count();
                                black_box(count);
                            }
                            warmup_done.wait();

                            // === MEASUREMENT PHASE ===
                            let mut total = 0usize;
                            start.wait();
                            pre_measurement_barrier();

                            for _ in 0..OPS_PER_THREAD {
                                let count = tree
                                    .iter(&guard)
                                    .take(SCAN_LIMIT)
                                    .inspect(|(_, v)| {
                                        black_box(*v);
                                    })
                                    .count();
                                total += count;
                            }

                            post_measurement_barrier();
                            black_box(total);
                        })
                    })
                    .collect();

                for h in handles {
                    h.join().unwrap();
                }
            });
    }
}

// =============================================================================
// 05: SHARED PREFIX - MassTree trie advantage (16B keys)
// =============================================================================

#[divan::bench_group(name = "05_shared_prefix_scan", sample_count = 200)]
mod shared_prefix_scan {
    use super::*;

    const PREFIX_BUCKETS: u64 = 100; // 10k keys per prefix

    #[divan::bench(args = THREAD_COUNTS)]
    fn masstree15_inline(bencher: Bencher, threads: usize) {
        let keys = Arc::new(keys_shared_prefix::<16>(N, PREFIX_BUCKETS));
        let tree = Arc::new(setup_masstree15_inline(&keys));

        bencher
            .counter(divan::counter::ItemsCount::new(threads * OPS_PER_THREAD))
            .bench_local(|| {
                let warmup_done = Arc::new(Barrier::new(threads));
                let start = Arc::new(Barrier::new(threads));
                let handles: Vec<_> = (0..threads)
                    .map(|_| {
                        let tree = Arc::clone(&tree);
                        let warmup_done = Arc::clone(&warmup_done);
                        let start = Arc::clone(&start);
                        thread::spawn(move || {
                            let guard = tree.guard();

                            // === WARMUP PHASE ===
                            for _ in 0..WARMUP_OPS {
                                let mut count = 0usize;
                                tree.scan_values(
                                    RangeBound::Unbounded,
                                    RangeBound::Unbounded,
                                    |v| {
                                        black_box(v);
                                        count += 1;
                                        count < SCAN_LIMIT
                                    },
                                    &guard,
                                );
                                black_box(count);
                            }
                            warmup_done.wait();

                            // === MEASUREMENT PHASE ===
                            let mut total = 0usize;
                            start.wait();
                            pre_measurement_barrier();

                            for _ in 0..OPS_PER_THREAD {
                                let mut count = 0usize;
                                tree.scan_values(
                                    RangeBound::Unbounded,
                                    RangeBound::Unbounded,
                                    |v| {
                                        black_box(v);
                                        count += 1;
                                        count < SCAN_LIMIT
                                    },
                                    &guard,
                                );
                                total += count;
                            }

                            post_measurement_barrier();
                            black_box(total);
                        })
                    })
                    .collect();

                for h in handles {
                    h.join().unwrap();
                }
            });
    }

    #[divan::bench(args = THREAD_COUNTS)]
    fn tree_index(bencher: Bencher, threads: usize) {
        let keys = Arc::new(keys_shared_prefix::<16>(N, PREFIX_BUCKETS));
        let tree = Arc::new(setup_tree_index(&keys));

        bencher
            .counter(divan::counter::ItemsCount::new(threads * OPS_PER_THREAD))
            .bench_local(|| {
                let warmup_done = Arc::new(Barrier::new(threads));
                let start = Arc::new(Barrier::new(threads));
                let handles: Vec<_> = (0..threads)
                    .map(|_| {
                        let tree = Arc::clone(&tree);
                        let warmup_done = Arc::clone(&warmup_done);
                        let start = Arc::clone(&start);
                        thread::spawn(move || {
                            let guard = sdd::Guard::new();

                            // === WARMUP PHASE ===
                            for _ in 0..WARMUP_OPS {
                                let count = tree.iter(&guard).take(SCAN_LIMIT).count();
                                black_box(count);
                            }
                            warmup_done.wait();

                            // === MEASUREMENT PHASE ===
                            let mut total = 0usize;
                            start.wait();
                            pre_measurement_barrier();

                            for _ in 0..OPS_PER_THREAD {
                                let count = tree.iter(&guard).take(SCAN_LIMIT).count();
                                total += count;
                            }

                            post_measurement_barrier();
                            black_box(total);
                        })
                    })
                    .collect();

                for h in handles {
                    h.join().unwrap();
                }
            });
    }
}

// =============================================================================
// 06: SUFFIX ONLY DIFFER - MassTree suffix mechanism (32B keys)
// =============================================================================

#[divan::bench_group(name = "06_suffix_differ_scan", sample_count = 200)]
mod suffix_differ_scan {
    use super::*;

    #[divan::bench(args = THREAD_COUNTS)]
    fn masstree15_inline(bencher: Bencher, threads: usize) {
        let keys = Arc::new(keys_suffix_only_differ::<32>(N));
        let tree = Arc::new(setup_masstree15_inline(&keys));

        bencher
            .counter(divan::counter::ItemsCount::new(threads * OPS_PER_THREAD))
            .bench_local(|| {
                let warmup_done = Arc::new(Barrier::new(threads));
                let start = Arc::new(Barrier::new(threads));
                let handles: Vec<_> = (0..threads)
                    .map(|_| {
                        let tree = Arc::clone(&tree);
                        let warmup_done = Arc::clone(&warmup_done);
                        let start = Arc::clone(&start);
                        thread::spawn(move || {
                            let guard = tree.guard();

                            // === WARMUP PHASE ===
                            for _ in 0..WARMUP_OPS {
                                let mut count = 0usize;
                                tree.scan_values(
                                    RangeBound::Unbounded,
                                    RangeBound::Unbounded,
                                    |v| {
                                        black_box(v);
                                        count += 1;
                                        count < SCAN_LIMIT
                                    },
                                    &guard,
                                );
                                black_box(count);
                            }
                            warmup_done.wait();

                            // === MEASUREMENT PHASE ===
                            let mut total = 0usize;
                            start.wait();
                            pre_measurement_barrier();

                            for _ in 0..OPS_PER_THREAD {
                                let mut count = 0usize;
                                tree.scan_values(
                                    RangeBound::Unbounded,
                                    RangeBound::Unbounded,
                                    |v| {
                                        black_box(v);
                                        count += 1;
                                        count < SCAN_LIMIT
                                    },
                                    &guard,
                                );
                                total += count;
                            }

                            post_measurement_barrier();
                            black_box(total);
                        })
                    })
                    .collect();

                for h in handles {
                    h.join().unwrap();
                }
            });
    }

    #[divan::bench(args = THREAD_COUNTS)]
    fn tree_index(bencher: Bencher, threads: usize) {
        let keys = Arc::new(keys_suffix_only_differ::<32>(N));
        let tree = Arc::new(setup_tree_index(&keys));

        bencher
            .counter(divan::counter::ItemsCount::new(threads * OPS_PER_THREAD))
            .bench_local(|| {
                let warmup_done = Arc::new(Barrier::new(threads));
                let start = Arc::new(Barrier::new(threads));
                let handles: Vec<_> = (0..threads)
                    .map(|_| {
                        let tree = Arc::clone(&tree);
                        let warmup_done = Arc::clone(&warmup_done);
                        let start = Arc::clone(&start);
                        thread::spawn(move || {
                            let guard = sdd::Guard::new();

                            // === WARMUP PHASE ===
                            for _ in 0..WARMUP_OPS {
                                let count = tree.iter(&guard).take(SCAN_LIMIT).count();
                                black_box(count);
                            }
                            warmup_done.wait();

                            // === MEASUREMENT PHASE ===
                            let mut total = 0usize;
                            start.wait();
                            pre_measurement_barrier();

                            for _ in 0..OPS_PER_THREAD {
                                let count = tree.iter(&guard).take(SCAN_LIMIT).count();
                                total += count;
                            }

                            post_measurement_barrier();
                            black_box(total);
                        })
                    })
                    .collect();

                for h in handles {
                    h.join().unwrap();
                }
            });
    }
}

// =============================================================================
// 07: HIERARCHICAL KEYS - Namespace:category:id pattern (32B keys)
// =============================================================================

#[divan::bench_group(name = "07_hierarchical_scan", sample_count = 200)]
mod hierarchical_scan {
    use super::*;

    // 100 namespaces * 100 categories * 100 items = 1M keys
    const NAMESPACES: usize = 100;
    const CATEGORIES: usize = 100;
    const ITEMS: usize = 50;

    #[divan::bench(args = THREAD_COUNTS)]
    fn masstree15_inline(bencher: Bencher, threads: usize) {
        let keys = Arc::new(keys_hierarchical::<32>(NAMESPACES, CATEGORIES, ITEMS));
        let tree = Arc::new(setup_masstree15_inline(&keys));

        bencher
            .counter(divan::counter::ItemsCount::new(threads * OPS_PER_THREAD))
            .bench_local(|| {
                let warmup_done = Arc::new(Barrier::new(threads));
                let start = Arc::new(Barrier::new(threads));
                let handles: Vec<_> = (0..threads)
                    .map(|_| {
                        let tree = Arc::clone(&tree);
                        let warmup_done = Arc::clone(&warmup_done);
                        let start = Arc::clone(&start);
                        thread::spawn(move || {
                            let guard = tree.guard();

                            // === WARMUP PHASE ===
                            for _ in 0..WARMUP_OPS {
                                let mut count = 0usize;
                                tree.scan_values(
                                    RangeBound::Unbounded,
                                    RangeBound::Unbounded,
                                    |v| {
                                        black_box(v);
                                        count += 1;
                                        count < SCAN_LIMIT
                                    },
                                    &guard,
                                );
                                black_box(count);
                            }
                            warmup_done.wait();

                            // === MEASUREMENT PHASE ===
                            let mut total = 0usize;
                            start.wait();
                            pre_measurement_barrier();

                            for _ in 0..OPS_PER_THREAD {
                                let mut count = 0usize;
                                tree.scan_values(
                                    RangeBound::Unbounded,
                                    RangeBound::Unbounded,
                                    |v| {
                                        black_box(v);
                                        count += 1;
                                        count < SCAN_LIMIT
                                    },
                                    &guard,
                                );
                                total += count;
                            }

                            post_measurement_barrier();
                            black_box(total);
                        })
                    })
                    .collect();

                for h in handles {
                    h.join().unwrap();
                }
            });
    }

    #[divan::bench(args = THREAD_COUNTS)]
    fn tree_index(bencher: Bencher, threads: usize) {
        let keys = Arc::new(keys_hierarchical::<32>(NAMESPACES, CATEGORIES, ITEMS));
        let tree = Arc::new(setup_tree_index(&keys));

        bencher
            .counter(divan::counter::ItemsCount::new(threads * OPS_PER_THREAD))
            .bench_local(|| {
                let warmup_done = Arc::new(Barrier::new(threads));
                let start = Arc::new(Barrier::new(threads));
                let handles: Vec<_> = (0..threads)
                    .map(|_| {
                        let tree = Arc::clone(&tree);
                        let warmup_done = Arc::clone(&warmup_done);
                        let start = Arc::clone(&start);
                        thread::spawn(move || {
                            let guard = sdd::Guard::new();

                            // === WARMUP PHASE ===
                            for _ in 0..WARMUP_OPS {
                                let count = tree.iter(&guard).take(SCAN_LIMIT).count();
                                black_box(count);
                            }
                            warmup_done.wait();

                            // === MEASUREMENT PHASE ===
                            let mut total = 0usize;
                            start.wait();
                            pre_measurement_barrier();

                            for _ in 0..OPS_PER_THREAD {
                                let count = tree.iter(&guard).take(SCAN_LIMIT).count();
                                total += count;
                            }

                            post_measurement_barrier();
                            black_box(total);
                        })
                    })
                    .collect();

                for h in handles {
                    h.join().unwrap();
                }
            });
    }
}

// =============================================================================
// 08: ADVERSARIAL SPLITS - Split propagation stress
// =============================================================================

#[divan::bench_group(name = "08_adversarial_splits_scan", sample_count = 200)]
mod adversarial_splits_scan {
    use super::*;

    #[divan::bench(args = THREAD_COUNTS)]
    fn masstree15_inline(bencher: Bencher, threads: usize) {
        let keys = Arc::new(keys_adversarial_splits::<8>(N));
        let tree = Arc::new(setup_masstree15_inline(&keys));

        bencher
            .counter(divan::counter::ItemsCount::new(threads * OPS_PER_THREAD))
            .bench_local(|| {
                let warmup_done = Arc::new(Barrier::new(threads));
                let start = Arc::new(Barrier::new(threads));
                let handles: Vec<_> = (0..threads)
                    .map(|_| {
                        let tree = Arc::clone(&tree);
                        let warmup_done = Arc::clone(&warmup_done);
                        let start = Arc::clone(&start);
                        thread::spawn(move || {
                            let guard = tree.guard();

                            // === WARMUP PHASE ===
                            for _ in 0..WARMUP_OPS {
                                let mut count = 0usize;
                                tree.scan_values(
                                    RangeBound::Unbounded,
                                    RangeBound::Unbounded,
                                    |v| {
                                        black_box(v);
                                        count += 1;
                                        count < SCAN_LIMIT
                                    },
                                    &guard,
                                );
                                black_box(count);
                            }
                            warmup_done.wait();

                            // === MEASUREMENT PHASE ===
                            let mut total = 0usize;
                            start.wait();
                            pre_measurement_barrier();

                            for _ in 0..OPS_PER_THREAD {
                                let mut count = 0usize;
                                tree.scan_values(
                                    RangeBound::Unbounded,
                                    RangeBound::Unbounded,
                                    |v| {
                                        black_box(v);
                                        count += 1;
                                        count < SCAN_LIMIT
                                    },
                                    &guard,
                                );
                                total += count;
                            }

                            post_measurement_barrier();
                            black_box(total);
                        })
                    })
                    .collect();

                for h in handles {
                    h.join().unwrap();
                }
            });
    }

    #[divan::bench(args = THREAD_COUNTS)]
    fn tree_index(bencher: Bencher, threads: usize) {
        let keys = Arc::new(keys_adversarial_splits::<8>(N));
        let tree = Arc::new(setup_tree_index(&keys));

        bencher
            .counter(divan::counter::ItemsCount::new(threads * OPS_PER_THREAD))
            .bench_local(|| {
                let warmup_done = Arc::new(Barrier::new(threads));
                let start = Arc::new(Barrier::new(threads));
                let handles: Vec<_> = (0..threads)
                    .map(|_| {
                        let tree = Arc::clone(&tree);
                        let warmup_done = Arc::clone(&warmup_done);
                        let start = Arc::clone(&start);
                        thread::spawn(move || {
                            let guard = sdd::Guard::new();

                            // === WARMUP PHASE ===
                            for _ in 0..WARMUP_OPS {
                                let count = tree.iter(&guard).take(SCAN_LIMIT).count();
                                black_box(count);
                            }
                            warmup_done.wait();

                            // === MEASUREMENT PHASE ===
                            let mut total = 0usize;
                            start.wait();
                            pre_measurement_barrier();

                            for _ in 0..OPS_PER_THREAD {
                                let count = tree.iter(&guard).take(SCAN_LIMIT).count();
                                total += count;
                            }

                            post_measurement_barrier();
                            black_box(total);
                        })
                    })
                    .collect();

                for h in handles {
                    h.join().unwrap();
                }
            });
    }
}

// =============================================================================
// 09: INTERLEAVED RANGES - Cache thrashing stress
// =============================================================================

#[divan::bench_group(name = "09_interleaved_scan", sample_count = 200)]
mod interleaved_scan {
    use super::*;

    const HOT_RANGES: usize = 50;
    const KEYS_PER_RANGE: usize = N / HOT_RANGES;
    const COLD_GAP: u64 = 100_000;

    #[divan::bench(args = THREAD_COUNTS)]
    fn masstree15_inline(bencher: Bencher, threads: usize) {
        let keys = Arc::new(keys_interleaved_ranges::<8>(
            HOT_RANGES,
            KEYS_PER_RANGE,
            COLD_GAP,
        ));
        let tree = Arc::new(setup_masstree15_inline(&keys));

        bencher
            .counter(divan::counter::ItemsCount::new(threads * OPS_PER_THREAD))
            .bench_local(|| {
                let warmup_done = Arc::new(Barrier::new(threads));
                let start = Arc::new(Barrier::new(threads));
                let handles: Vec<_> = (0..threads)
                    .map(|_| {
                        let tree = Arc::clone(&tree);
                        let warmup_done = Arc::clone(&warmup_done);
                        let start = Arc::clone(&start);
                        thread::spawn(move || {
                            let guard = tree.guard();

                            // === WARMUP PHASE ===
                            for _ in 0..WARMUP_OPS {
                                let mut count = 0usize;
                                tree.scan_values(
                                    RangeBound::Unbounded,
                                    RangeBound::Unbounded,
                                    |v| {
                                        black_box(v);
                                        count += 1;
                                        count < SCAN_LIMIT
                                    },
                                    &guard,
                                );
                                black_box(count);
                            }
                            warmup_done.wait();

                            // === MEASUREMENT PHASE ===
                            let mut total = 0usize;
                            start.wait();
                            pre_measurement_barrier();

                            for _ in 0..OPS_PER_THREAD {
                                let mut count = 0usize;
                                tree.scan_values(
                                    RangeBound::Unbounded,
                                    RangeBound::Unbounded,
                                    |v| {
                                        black_box(v);
                                        count += 1;
                                        count < SCAN_LIMIT
                                    },
                                    &guard,
                                );
                                total += count;
                            }

                            post_measurement_barrier();
                            black_box(total);
                        })
                    })
                    .collect();

                for h in handles {
                    h.join().unwrap();
                }
            });
    }

    #[divan::bench(args = THREAD_COUNTS)]
    fn tree_index(bencher: Bencher, threads: usize) {
        let keys = Arc::new(keys_interleaved_ranges::<8>(
            HOT_RANGES,
            KEYS_PER_RANGE,
            COLD_GAP,
        ));
        let tree = Arc::new(setup_tree_index(&keys));

        bencher
            .counter(divan::counter::ItemsCount::new(threads * OPS_PER_THREAD))
            .bench_local(|| {
                let warmup_done = Arc::new(Barrier::new(threads));
                let start = Arc::new(Barrier::new(threads));
                let handles: Vec<_> = (0..threads)
                    .map(|_| {
                        let tree = Arc::clone(&tree);
                        let warmup_done = Arc::clone(&warmup_done);
                        let start = Arc::clone(&start);
                        thread::spawn(move || {
                            let guard = sdd::Guard::new();

                            // === WARMUP PHASE ===
                            for _ in 0..WARMUP_OPS {
                                let count = tree.iter(&guard).take(SCAN_LIMIT).count();
                                black_box(count);
                            }
                            warmup_done.wait();

                            // === MEASUREMENT PHASE ===
                            let mut total = 0usize;
                            start.wait();
                            pre_measurement_barrier();

                            for _ in 0..OPS_PER_THREAD {
                                let count = tree.iter(&guard).take(SCAN_LIMIT).count();
                                total += count;
                            }

                            post_measurement_barrier();
                            black_box(total);
                        })
                    })
                    .collect();

                for h in handles {
                    h.join().unwrap();
                }
            });
    }
}

// =============================================================================
// 10: B-LINK STRESS - Fragmented scan stress
// =============================================================================

#[divan::bench_group(name = "10_blink_stress_scan", sample_count = 200)]
mod blink_stress_scan {
    use super::*;

    #[divan::bench(args = THREAD_COUNTS)]
    fn masstree15_inline(bencher: Bencher, threads: usize) {
        let keys = Arc::new(keys_blink_stress::<8>(N));
        let tree = Arc::new(setup_masstree15_inline(&keys));

        bencher
            .counter(divan::counter::ItemsCount::new(threads * OPS_PER_THREAD))
            .bench_local(|| {
                let warmup_done = Arc::new(Barrier::new(threads));
                let start = Arc::new(Barrier::new(threads));
                let handles: Vec<_> = (0..threads)
                    .map(|_| {
                        let tree = Arc::clone(&tree);
                        let warmup_done = Arc::clone(&warmup_done);
                        let start = Arc::clone(&start);
                        thread::spawn(move || {
                            let guard = tree.guard();

                            // === WARMUP PHASE ===
                            for _ in 0..WARMUP_OPS {
                                let mut count = 0usize;
                                tree.scan_values(
                                    RangeBound::Unbounded,
                                    RangeBound::Unbounded,
                                    |v| {
                                        black_box(v);
                                        count += 1;
                                        count < SCAN_LIMIT
                                    },
                                    &guard,
                                );
                                black_box(count);
                            }
                            warmup_done.wait();

                            // === MEASUREMENT PHASE ===
                            let mut total = 0usize;
                            start.wait();
                            pre_measurement_barrier();

                            for _ in 0..OPS_PER_THREAD {
                                let mut count = 0usize;
                                tree.scan_values(
                                    RangeBound::Unbounded,
                                    RangeBound::Unbounded,
                                    |v| {
                                        black_box(v);
                                        count += 1;
                                        count < SCAN_LIMIT
                                    },
                                    &guard,
                                );
                                total += count;
                            }

                            post_measurement_barrier();
                            black_box(total);
                        })
                    })
                    .collect();

                for h in handles {
                    h.join().unwrap();
                }
            });
    }

    #[divan::bench(args = THREAD_COUNTS)]
    fn tree_index(bencher: Bencher, threads: usize) {
        let keys = Arc::new(keys_blink_stress::<8>(N));
        let tree = Arc::new(setup_tree_index(&keys));

        bencher
            .counter(divan::counter::ItemsCount::new(threads * OPS_PER_THREAD))
            .bench_local(|| {
                let warmup_done = Arc::new(Barrier::new(threads));
                let start = Arc::new(Barrier::new(threads));
                let handles: Vec<_> = (0..threads)
                    .map(|_| {
                        let tree = Arc::clone(&tree);
                        let warmup_done = Arc::clone(&warmup_done);
                        let start = Arc::clone(&start);
                        thread::spawn(move || {
                            let guard = sdd::Guard::new();

                            // === WARMUP PHASE ===
                            for _ in 0..WARMUP_OPS {
                                let count = tree.iter(&guard).take(SCAN_LIMIT).count();
                                black_box(count);
                            }
                            warmup_done.wait();

                            // === MEASUREMENT PHASE ===
                            let mut total = 0usize;
                            start.wait();
                            pre_measurement_barrier();

                            for _ in 0..OPS_PER_THREAD {
                                let count = tree.iter(&guard).take(SCAN_LIMIT).count();
                                total += count;
                            }

                            post_measurement_barrier();
                            black_box(total);
                        })
                    })
                    .collect();

                for h in handles {
                    h.join().unwrap();
                }
            });
    }
}

// =============================================================================
// 11: RANDOM KEYS - Baseline comparison
// =============================================================================

#[divan::bench_group(name = "11_random_keys_scan", sample_count = 200)]
mod random_keys_scan {
    use super::*;

    #[divan::bench(args = THREAD_COUNTS)]
    fn masstree15_inline(bencher: Bencher, threads: usize) {
        let keys = Arc::new(keys::<8>(N));
        let tree = Arc::new(setup_masstree15_inline(&keys));

        bencher
            .counter(divan::counter::ItemsCount::new(threads * OPS_PER_THREAD))
            .bench_local(|| {
                let warmup_done = Arc::new(Barrier::new(threads));
                let start = Arc::new(Barrier::new(threads));
                let handles: Vec<_> = (0..threads)
                    .map(|_| {
                        let tree = Arc::clone(&tree);
                        let warmup_done = Arc::clone(&warmup_done);
                        let start = Arc::clone(&start);
                        thread::spawn(move || {
                            let guard = tree.guard();

                            // === WARMUP PHASE ===
                            for _ in 0..WARMUP_OPS {
                                let mut count = 0usize;
                                tree.scan_values(
                                    RangeBound::Unbounded,
                                    RangeBound::Unbounded,
                                    |v| {
                                        black_box(v);
                                        count += 1;
                                        count < SCAN_LIMIT
                                    },
                                    &guard,
                                );
                                black_box(count);
                            }
                            warmup_done.wait();

                            // === MEASUREMENT PHASE ===
                            let mut total = 0usize;
                            start.wait();
                            pre_measurement_barrier();

                            for _ in 0..OPS_PER_THREAD {
                                let mut count = 0usize;
                                tree.scan_values(
                                    RangeBound::Unbounded,
                                    RangeBound::Unbounded,
                                    |v| {
                                        black_box(v);
                                        count += 1;
                                        count < SCAN_LIMIT
                                    },
                                    &guard,
                                );
                                total += count;
                            }

                            post_measurement_barrier();
                            black_box(total);
                        })
                    })
                    .collect();

                for h in handles {
                    h.join().unwrap();
                }
            });
    }

    #[divan::bench(args = THREAD_COUNTS)]
    fn tree_index(bencher: Bencher, threads: usize) {
        let keys = Arc::new(keys::<8>(N));
        let tree = Arc::new(setup_tree_index(&keys));

        bencher
            .counter(divan::counter::ItemsCount::new(threads * OPS_PER_THREAD))
            .bench_local(|| {
                let warmup_done = Arc::new(Barrier::new(threads));
                let start = Arc::new(Barrier::new(threads));
                let handles: Vec<_> = (0..threads)
                    .map(|_| {
                        let tree = Arc::clone(&tree);
                        let warmup_done = Arc::clone(&warmup_done);
                        let start = Arc::clone(&start);
                        thread::spawn(move || {
                            let guard = sdd::Guard::new();

                            // === WARMUP PHASE ===
                            for _ in 0..WARMUP_OPS {
                                let count = tree.iter(&guard).take(SCAN_LIMIT).count();
                                black_box(count);
                            }
                            warmup_done.wait();

                            // === MEASUREMENT PHASE ===
                            let mut total = 0usize;
                            start.wait();
                            pre_measurement_barrier();

                            for _ in 0..OPS_PER_THREAD {
                                let count = tree.iter(&guard).take(SCAN_LIMIT).count();
                                total += count;
                            }

                            post_measurement_barrier();
                            black_box(total);
                        })
                    })
                    .collect();

                for h in handles {
                    h.join().unwrap();
                }
            });
    }
}

// =============================================================================
// 12: LONG KEYS (64B) - Multi-layer traversal
// =============================================================================

#[divan::bench_group(name = "12_long_keys_64b_scan", sample_count = 200)]
mod long_keys_scan {
    use super::*;

    #[divan::bench(args = THREAD_COUNTS)]
    fn masstree15_inline(bencher: Bencher, threads: usize) {
        let keys = Arc::new(keys::<64>(N));
        let tree = Arc::new(setup_masstree15_inline(&keys));

        bencher
            .counter(divan::counter::ItemsCount::new(threads * OPS_PER_THREAD))
            .bench_local(|| {
                let warmup_done = Arc::new(Barrier::new(threads));
                let start = Arc::new(Barrier::new(threads));
                let handles: Vec<_> = (0..threads)
                    .map(|_| {
                        let tree = Arc::clone(&tree);
                        let warmup_done = Arc::clone(&warmup_done);
                        let start = Arc::clone(&start);
                        thread::spawn(move || {
                            let guard = tree.guard();

                            // === WARMUP PHASE ===
                            for _ in 0..WARMUP_OPS {
                                let mut count = 0usize;
                                tree.scan_values(
                                    RangeBound::Unbounded,
                                    RangeBound::Unbounded,
                                    |v| {
                                        black_box(v);
                                        count += 1;
                                        count < SCAN_LIMIT
                                    },
                                    &guard,
                                );
                                black_box(count);
                            }
                            warmup_done.wait();

                            // === MEASUREMENT PHASE ===
                            let mut total = 0usize;
                            start.wait();
                            pre_measurement_barrier();

                            for _ in 0..OPS_PER_THREAD {
                                let mut count = 0usize;
                                tree.scan_values(
                                    RangeBound::Unbounded,
                                    RangeBound::Unbounded,
                                    |v| {
                                        black_box(v);
                                        count += 1;
                                        count < SCAN_LIMIT
                                    },
                                    &guard,
                                );
                                total += count;
                            }

                            post_measurement_barrier();
                            black_box(total);
                        })
                    })
                    .collect();

                for h in handles {
                    h.join().unwrap();
                }
            });
    }

    #[divan::bench(args = THREAD_COUNTS)]
    fn tree_index(bencher: Bencher, threads: usize) {
        let keys = Arc::new(keys::<64>(N));
        let tree = Arc::new(setup_tree_index(&keys));

        bencher
            .counter(divan::counter::ItemsCount::new(threads * OPS_PER_THREAD))
            .bench_local(|| {
                let warmup_done = Arc::new(Barrier::new(threads));
                let start = Arc::new(Barrier::new(threads));
                let handles: Vec<_> = (0..threads)
                    .map(|_| {
                        let tree = Arc::clone(&tree);
                        let warmup_done = Arc::clone(&warmup_done);
                        let start = Arc::clone(&start);
                        thread::spawn(move || {
                            let guard = sdd::Guard::new();

                            // === WARMUP PHASE ===
                            for _ in 0..WARMUP_OPS {
                                let count = tree.iter(&guard).take(SCAN_LIMIT).count();
                                black_box(count);
                            }
                            warmup_done.wait();

                            // === MEASUREMENT PHASE ===
                            let mut total = 0usize;
                            start.wait();
                            pre_measurement_barrier();

                            for _ in 0..OPS_PER_THREAD {
                                let count = tree.iter(&guard).take(SCAN_LIMIT).count();
                                total += count;
                            }

                            post_measurement_barrier();
                            black_box(total);
                        })
                    })
                    .collect();

                for h in handles {
                    h.join().unwrap();
                }
            });
    }
}

// =============================================================================
// 13: SCAN WHILE INSERT - Mixed workload
// =============================================================================

#[divan::bench_group(name = "13_scan_while_insert", sample_count = 200)]
mod scan_while_insert {
    use super::*;

    const INITIAL_N: usize = 450_000;
    const INSERT_N: usize = 25_000;
    const WRITERS: usize = 2;
    const WRITER_WARMUP_OPS: usize = 100; // Warmup inserts for writers

    // Minimum 3 threads: 2 writers + 1 reader
    #[divan::bench(args = [3, 4, 6, 8, 12])]
    fn masstree15_inline(bencher: Bencher, threads: usize) {
        let initial_keys = keys::<8>(INITIAL_N);
        let insert_keys: Vec<_> = keys::<8>(INITIAL_N + INSERT_N)[INITIAL_N..].to_vec();
        // Extra keys for writer warmup (won't be in tree initially)
        let warmup_keys: Vec<[u8; 8]> = (0..WRITER_WARMUP_OPS * WRITERS)
            .map(|i| ((INITIAL_N + INSERT_N + i) as u64).to_be_bytes())
            .collect();
        let readers = threads - WRITERS;

        bencher
            .with_inputs(|| {
                let tree = Arc::new(setup_masstree15_inline(&initial_keys));
                let new_keys = Arc::new(insert_keys.clone());
                let warmup_keys = Arc::new(warmup_keys.clone());
                (tree, new_keys, warmup_keys)
            })
            .bench_refs(|(tree, new_keys, warmup_keys)| {
                let warmup_done = Arc::new(Barrier::new(threads));
                let start = Arc::new(Barrier::new(threads));

                // Writer threads
                let writer_handles: Vec<_> = (0..WRITERS)
                    .map(|w| {
                        let tree = Arc::clone(tree);
                        let warmup_done = Arc::clone(&warmup_done);
                        let start = Arc::clone(&start);
                        let chunk_size = INSERT_N / WRITERS;
                        let chunk: Vec<_> = new_keys[w * chunk_size..(w + 1) * chunk_size].to_vec();
                        let warmup_chunk: Vec<_> = warmup_keys
                            [w * WRITER_WARMUP_OPS..(w + 1) * WRITER_WARMUP_OPS]
                            .to_vec();
                        thread::spawn(move || {
                            let guard = tree.guard();

                            // === WARMUP PHASE ===
                            // Writers warm up with actual inserts to prime code paths
                            for (i, key) in warmup_chunk.iter().enumerate() {
                                let _ = tree.insert_with_guard(key, i as u64, &guard);
                            }
                            warmup_done.wait();

                            // === MEASUREMENT PHASE ===
                            start.wait();
                            pre_measurement_barrier();

                            for (i, key) in chunk.iter().enumerate() {
                                let _ = tree.insert_with_guard(key, (INITIAL_N + i) as u64, &guard);
                            }

                            post_measurement_barrier();
                        })
                    })
                    .collect();

                // Reader threads
                let reader_handles: Vec<_> = (0..readers)
                    .map(|_| {
                        let tree = Arc::clone(tree);
                        let warmup_done = Arc::clone(&warmup_done);
                        let start = Arc::clone(&start);
                        thread::spawn(move || {
                            let guard = tree.guard();

                            // === WARMUP PHASE ===
                            for _ in 0..WARMUP_OPS {
                                let mut count = 0usize;
                                tree.scan_values(
                                    RangeBound::Unbounded,
                                    RangeBound::Unbounded,
                                    |v| {
                                        black_box(v);
                                        count += 1;
                                        count < SCAN_LIMIT
                                    },
                                    &guard,
                                );
                                black_box(count);
                            }
                            warmup_done.wait();

                            // === MEASUREMENT PHASE ===
                            let mut total = 0usize;
                            start.wait();
                            pre_measurement_barrier();

                            for _ in 0..OPS_PER_THREAD {
                                let mut count = 0usize;
                                tree.scan_values(
                                    RangeBound::Unbounded,
                                    RangeBound::Unbounded,
                                    |v| {
                                        black_box(v);
                                        count += 1;
                                        count < SCAN_LIMIT
                                    },
                                    &guard,
                                );
                                total += count;
                            }

                            post_measurement_barrier();
                            black_box(total);
                        })
                    })
                    .collect();

                for h in writer_handles {
                    h.join().unwrap();
                }
                for h in reader_handles {
                    h.join().unwrap();
                }
            });
    }

    #[divan::bench(args = [3, 4, 6, 8, 12])]
    fn tree_index(bencher: Bencher, threads: usize) {
        let initial_keys = keys::<8>(INITIAL_N);
        let insert_keys: Vec<_> = keys::<8>(INITIAL_N + INSERT_N)[INITIAL_N..].to_vec();
        let warmup_keys: Vec<[u8; 8]> = (0..WRITER_WARMUP_OPS * WRITERS)
            .map(|i| ((INITIAL_N + INSERT_N + i) as u64).to_be_bytes())
            .collect();
        let readers = threads - WRITERS;

        bencher
            .with_inputs(|| {
                let tree = Arc::new(setup_tree_index(&initial_keys));
                let new_keys = Arc::new(insert_keys.clone());
                let warmup_keys = Arc::new(warmup_keys.clone());
                (tree, new_keys, warmup_keys)
            })
            .bench_refs(|(tree, new_keys, warmup_keys)| {
                let warmup_done = Arc::new(Barrier::new(threads));
                let start = Arc::new(Barrier::new(threads));

                // Writer threads
                let writer_handles: Vec<_> = (0..WRITERS)
                    .map(|w| {
                        let tree = Arc::clone(tree);
                        let warmup_done = Arc::clone(&warmup_done);
                        let start = Arc::clone(&start);
                        let chunk_size = INSERT_N / WRITERS;
                        let chunk: Vec<_> = new_keys[w * chunk_size..(w + 1) * chunk_size].to_vec();
                        let warmup_chunk: Vec<_> = warmup_keys
                            [w * WRITER_WARMUP_OPS..(w + 1) * WRITER_WARMUP_OPS]
                            .to_vec();
                        thread::spawn(move || {
                            // === WARMUP PHASE ===
                            for (i, key) in warmup_chunk.iter().enumerate() {
                                let _ = tree.insert_sync(*key, i as u64);
                            }
                            warmup_done.wait();

                            // === MEASUREMENT PHASE ===
                            start.wait();
                            pre_measurement_barrier();

                            for (i, key) in chunk.iter().enumerate() {
                                let _ = tree.insert_sync(*key, (INITIAL_N + i) as u64);
                            }

                            post_measurement_barrier();
                        })
                    })
                    .collect();

                // Reader threads
                let reader_handles: Vec<_> = (0..readers)
                    .map(|_| {
                        let tree = Arc::clone(tree);
                        let warmup_done = Arc::clone(&warmup_done);
                        let start = Arc::clone(&start);
                        thread::spawn(move || {
                            let guard = sdd::Guard::new();

                            // === WARMUP PHASE ===
                            for _ in 0..WARMUP_OPS {
                                let count = tree
                                    .iter(&guard)
                                    .take(SCAN_LIMIT)
                                    .inspect(|(_, v)| {
                                        black_box(*v);
                                    })
                                    .count();
                                black_box(count);
                            }
                            warmup_done.wait();

                            // === MEASUREMENT PHASE ===
                            let mut total = 0usize;
                            start.wait();
                            pre_measurement_barrier();

                            for _ in 0..OPS_PER_THREAD {
                                let count = tree
                                    .iter(&guard)
                                    .take(SCAN_LIMIT)
                                    .inspect(|(_, v)| {
                                        black_box(*v);
                                    })
                                    .count();
                                total += count;
                            }

                            post_measurement_barrier();
                            black_box(total);
                        })
                    })
                    .collect();

                for h in writer_handles {
                    h.join().unwrap();
                }
                for h in reader_handles {
                    h.join().unwrap();
                }
            });
    }
}

// =============================================================================
// 14: PREFIX SCAN - MassTree-specific optimization
// =============================================================================

#[divan::bench_group(name = "14_prefix_scan", sample_count = 200)]
mod prefix_scan {
    use super::*;

    const PREFIX_BUCKETS: u64 = 100;
    // Reduced ops for prefix scan - each scan touches 10K entries (1M/100 buckets)
    const PREFIX_OPS: usize = 50;

    #[divan::bench(args = THREAD_COUNTS)]
    fn masstree15_inline(bencher: Bencher, threads: usize) {
        let keys = Arc::new(keys_shared_prefix::<16>(N, PREFIX_BUCKETS));
        let tree = Arc::new(setup_masstree15_inline(&keys));

        bencher
            .counter(divan::counter::ItemsCount::new(threads * PREFIX_OPS))
            .bench_local(|| {
                let warmup_done = Arc::new(Barrier::new(threads));
                let start = Arc::new(Barrier::new(threads));
                let handles: Vec<_> = (0..threads)
                    .map(|t| {
                        let tree = Arc::clone(&tree);
                        let warmup_done = Arc::clone(&warmup_done);
                        let start = Arc::clone(&start);
                        // Each thread scans a different prefix
                        let thread_prefix = ((t as u64) % PREFIX_BUCKETS).to_be_bytes();
                        thread::spawn(move || {
                            let guard = tree.guard();

                            // === WARMUP PHASE ===
                            for _ in 0..WARMUP_OPS {
                                let count = tree.scan_prefix(&thread_prefix, |_, _| true, &guard);
                                black_box(count);
                            }
                            warmup_done.wait();

                            // === MEASUREMENT PHASE ===
                            let mut total = 0usize;
                            start.wait();
                            pre_measurement_barrier();

                            for _ in 0..PREFIX_OPS {
                                total += tree.scan_prefix(&thread_prefix, |_, _| true, &guard);
                            }

                            post_measurement_barrier();
                            black_box(total);
                        })
                    })
                    .collect();

                for h in handles {
                    h.join().unwrap();
                }
            });
    }
}

// =============================================================================
// 15: FULL SCAN AGGREGATE - Sum all values (reports keys/sec, not scans/sec)
// =============================================================================

#[divan::bench_group(name = "15_full_scan_aggregate", sample_count = 200)]
mod full_scan_aggregate {
    use super::*;

    const SCAN_N: usize = 25_000; // Smaller dataset for full scans
    const FULL_SCAN_OPS: usize = 50; // Reduced iterations for full scans

    #[divan::bench(args = THREAD_COUNTS)]
    fn masstree15_inline(bencher: Bencher, threads: usize) {
        let keys = Arc::new(keys_sequential::<8>(SCAN_N));
        let tree = Arc::new(setup_masstree15_inline(&keys));

        // Report keys/sec (not scans/sec) for meaningful throughput comparison
        bencher
            .counter(divan::counter::ItemsCount::new(
                threads * FULL_SCAN_OPS * SCAN_N,
            ))
            .bench_local(|| {
                let warmup_done = Arc::new(Barrier::new(threads));
                let start = Arc::new(Barrier::new(threads));
                let handles: Vec<_> = (0..threads)
                    .map(|_| {
                        let tree = Arc::clone(&tree);
                        let warmup_done = Arc::clone(&warmup_done);
                        let start = Arc::clone(&start);
                        thread::spawn(move || {
                            let guard = tree.guard();

                            // === WARMUP PHASE ===
                            for _ in 0..WARMUP_OPS {
                                let mut sum = 0u64;
                                tree.scan_values(
                                    RangeBound::Unbounded,
                                    RangeBound::Unbounded,
                                    |v| {
                                        sum += v;
                                        true
                                    },
                                    &guard,
                                );
                                black_box(sum);
                            }
                            warmup_done.wait();

                            // === MEASUREMENT PHASE ===
                            let mut grand_total = 0u64;
                            start.wait();
                            pre_measurement_barrier();

                            for _ in 0..FULL_SCAN_OPS {
                                let mut sum = 0u64;
                                tree.scan_values(
                                    RangeBound::Unbounded,
                                    RangeBound::Unbounded,
                                    |v| {
                                        sum += v;
                                        true
                                    },
                                    &guard,
                                );
                                grand_total += sum;
                            }

                            post_measurement_barrier();
                            black_box(grand_total);
                        })
                    })
                    .collect();

                for h in handles {
                    h.join().unwrap();
                }
            });
    }

    #[divan::bench(args = THREAD_COUNTS)]
    fn tree_index(bencher: Bencher, threads: usize) {
        let keys = Arc::new(keys_sequential::<8>(SCAN_N));
        let tree = Arc::new(setup_tree_index(&keys));

        // Report keys/sec (not scans/sec) for meaningful throughput comparison
        bencher
            .counter(divan::counter::ItemsCount::new(
                threads * FULL_SCAN_OPS * SCAN_N,
            ))
            .bench_local(|| {
                let warmup_done = Arc::new(Barrier::new(threads));
                let start = Arc::new(Barrier::new(threads));
                let handles: Vec<_> = (0..threads)
                    .map(|_| {
                        let tree = Arc::clone(&tree);
                        let warmup_done = Arc::clone(&warmup_done);
                        let start = Arc::clone(&start);
                        thread::spawn(move || {
                            let guard = sdd::Guard::new();

                            // === WARMUP PHASE ===
                            for _ in 0..WARMUP_OPS {
                                let sum: u64 = tree.iter(&guard).map(|(_, v)| *v).sum();
                                black_box(sum);
                            }
                            warmup_done.wait();

                            // === MEASUREMENT PHASE ===
                            let mut grand_total = 0u64;
                            start.wait();
                            pre_measurement_barrier();

                            for _ in 0..FULL_SCAN_OPS {
                                let sum: u64 = tree.iter(&guard).map(|(_, v)| *v).sum();
                                grand_total += sum;
                            }

                            post_measurement_barrier();
                            black_box(grand_total);
                        })
                    })
                    .collect();

                for h in handles {
                    h.join().unwrap();
                }
            });
    }
}

// =============================================================================
// 16: INSERT-HEAVY - 90% writes, 10% reads (high write contention)
// =============================================================================

#[divan::bench_group(name = "16_insert_heavy", sample_count = 200)]
mod insert_heavy {
    use super::*;

    const INITIAL_N: usize = 50_000;
    const OPS: usize = 5_000;
    const WRITE_RATIO: usize = 90; // 90% writes

    #[divan::bench(args = THREAD_COUNTS)]
    fn masstree15_inline(bencher: Bencher, threads: usize) {
        // Pre-generate insert keys (unique per thread)
        let initial_keys: Vec<[u8; 8]> = keys_sequential::<8>(INITIAL_N);
        let insert_keys: Vec<Vec<[u8; 8]>> = (0..threads)
            .map(|t| {
                (0..OPS)
                    .map(|i| {
                        let val = (INITIAL_N + t * OPS + i) as u64;
                        val.to_be_bytes()
                    })
                    .collect()
            })
            .collect();

        bencher
            .counter(divan::counter::ItemsCount::new(threads * OPS))
            .with_inputs(|| {
                let tree = Arc::new(setup_masstree15_inline(&initial_keys));
                (tree, insert_keys.clone())
            })
            .bench_refs(|(tree, thread_keys)| {
                let warmup_done = Arc::new(Barrier::new(threads));
                let start = Arc::new(Barrier::new(threads));
                let handles: Vec<_> = (0..threads)
                    .map(|t| {
                        let tree = Arc::clone(tree);
                        let warmup_done = Arc::clone(&warmup_done);
                        let start = Arc::clone(&start);
                        let keys = thread_keys[t].clone();
                        let read_keys = initial_keys.clone();
                        thread::spawn(move || {
                            let guard = tree.guard();

                            // === WARMUP PHASE ===
                            let mut rng_state = thread_seed(BASE_SEED, t);
                            for _ in 0..WARMUP_OPS {
                                rng_state =
                                    rng_state.wrapping_mul(6364136223846793005).wrapping_add(1);
                                let idx = (rng_state as usize) % read_keys.len();
                                black_box(tree.get_with_guard(&read_keys[idx], &guard));
                            }
                            warmup_done.wait();

                            // === MEASUREMENT PHASE ===
                            start.wait();
                            pre_measurement_barrier();

                            rng_state = thread_seed(BASE_SEED + 1, t); // Different seed for measurement phase
                            for (i, key) in keys.iter().enumerate() {
                                // Simple LCG for deterministic "random"
                                rng_state =
                                    rng_state.wrapping_mul(6364136223846793005).wrapping_add(1);
                                if (rng_state % 100) < WRITE_RATIO as u64 {
                                    let _ = tree.insert_with_guard(key, i as u64, &guard);
                                } else {
                                    let idx = (rng_state as usize) % read_keys.len();
                                    black_box(tree.get_with_guard(&read_keys[idx], &guard));
                                }
                            }

                            post_measurement_barrier();
                        })
                    })
                    .collect();

                for h in handles {
                    h.join().unwrap();
                }
            });
    }

    #[divan::bench(args = THREAD_COUNTS)]
    fn tree_index(bencher: Bencher, threads: usize) {
        let initial_keys: Vec<[u8; 8]> = keys_sequential::<8>(INITIAL_N);
        let insert_keys: Vec<Vec<[u8; 8]>> = (0..threads)
            .map(|t| {
                (0..OPS)
                    .map(|i| {
                        let val = (INITIAL_N + t * OPS + i) as u64;
                        val.to_be_bytes()
                    })
                    .collect()
            })
            .collect();

        bencher
            .counter(divan::counter::ItemsCount::new(threads * OPS))
            .with_inputs(|| {
                let tree = Arc::new(setup_tree_index(&initial_keys));
                (tree, insert_keys.clone())
            })
            .bench_refs(|(tree, thread_keys)| {
                let warmup_done = Arc::new(Barrier::new(threads));
                let start = Arc::new(Barrier::new(threads));
                let handles: Vec<_> = (0..threads)
                    .map(|t| {
                        let tree = Arc::clone(tree);
                        let warmup_done = Arc::clone(&warmup_done);
                        let start = Arc::clone(&start);
                        let keys = thread_keys[t].clone();
                        let read_keys = initial_keys.clone();
                        thread::spawn(move || {
                            let guard = sdd::Guard::new();

                            // === WARMUP PHASE ===
                            let mut rng_state = thread_seed(BASE_SEED, t);
                            for _ in 0..WARMUP_OPS {
                                rng_state =
                                    rng_state.wrapping_mul(6364136223846793005).wrapping_add(1);
                                let idx = (rng_state as usize) % read_keys.len();
                                black_box(tree.peek(&read_keys[idx], &guard));
                            }
                            warmup_done.wait();

                            // === MEASUREMENT PHASE ===
                            start.wait();
                            pre_measurement_barrier();

                            rng_state = thread_seed(BASE_SEED + 1, t); // Different seed for measurement phase
                            for (i, key) in keys.iter().enumerate() {
                                rng_state =
                                    rng_state.wrapping_mul(6364136223846793005).wrapping_add(1);
                                if (rng_state % 100) < WRITE_RATIO as u64 {
                                    let _ = tree.insert_sync(*key, i as u64);
                                } else {
                                    let idx = (rng_state as usize) % read_keys.len();
                                    black_box(tree.peek(&read_keys[idx], &guard));
                                }
                            }

                            post_measurement_barrier();
                        })
                    })
                    .collect();

                for h in handles {
                    h.join().unwrap();
                }
            });
    }
}

// =============================================================================
// 17: HOT-SPOT - All threads target narrow key range (localized contention)
// =============================================================================

#[divan::bench_group(name = "17_hot_spot", sample_count = 200)]
mod hot_spot {
    use super::*;

    const TOTAL_N: usize = 500_000;
    const HOT_RANGE: usize = 32; // Spans multiple leaves (Leaf15 width=15)
    const OPS: usize = 5_000;

    #[divan::bench(args = THREAD_COUNTS)]
    fn masstree15_inline(bencher: Bencher, threads: usize) {
        let all_keys: Vec<[u8; 8]> = keys_sequential::<8>(TOTAL_N);
        let hot_keys: Vec<[u8; 8]> = all_keys[..HOT_RANGE].to_vec();

        bencher
            .counter(divan::counter::ItemsCount::new(threads * OPS))
            .with_inputs(|| Arc::new(setup_masstree15_inline(&all_keys)))
            .bench_refs(|tree| {
                let warmup_done = Arc::new(Barrier::new(threads));
                let start = Arc::new(Barrier::new(threads));
                let handles: Vec<_> = (0..threads)
                    .map(|t| {
                        let tree = Arc::clone(tree);
                        let warmup_done = Arc::clone(&warmup_done);
                        let start = Arc::clone(&start);
                        let hot = hot_keys.clone();

                        thread::spawn(move || {
                            let guard = tree.guard();

                            // === WARMUP PHASE ===
                            let mut rng_state = thread_seed(BASE_SEED, t);
                            for _ in 0..WARMUP_OPS {
                                rng_state =
                                    rng_state.wrapping_mul(6364136223846793005).wrapping_add(1);
                                let idx = (rng_state as usize) % hot.len();
                                black_box(tree.get_with_guard(&hot[idx], &guard));
                            }
                            warmup_done.wait();

                            // === MEASUREMENT PHASE ===
                            start.wait();
                            pre_measurement_barrier();

                            rng_state = thread_seed(BASE_SEED + 1, t); // Different seed for measurement phase
                            for i in 0..OPS {
                                rng_state =
                                    rng_state.wrapping_mul(6364136223846793005).wrapping_add(1);
                                let idx = (rng_state as usize) % hot.len();

                                // 50% read, 50% update (insert over existing)
                                if rng_state.is_multiple_of(2) {
                                    black_box(tree.get_with_guard(&hot[idx], &guard));
                                } else {
                                    let _ = tree.insert_with_guard(&hot[idx], i as u64, &guard);
                                }
                            }

                            post_measurement_barrier();
                        })
                    })
                    .collect();

                for h in handles {
                    h.join().unwrap();
                }
            });
    }

    #[divan::bench(args = THREAD_COUNTS)]
    fn tree_index(bencher: Bencher, threads: usize) {
        let all_keys: Vec<[u8; 8]> = keys_sequential::<8>(TOTAL_N);
        let hot_keys: Vec<[u8; 8]> = all_keys[..HOT_RANGE].to_vec();

        bencher
            .counter(divan::counter::ItemsCount::new(threads * OPS))
            .with_inputs(|| Arc::new(setup_tree_index(&all_keys)))
            .bench_refs(|tree| {
                let warmup_done = Arc::new(Barrier::new(threads));
                let start = Arc::new(Barrier::new(threads));
                let handles: Vec<_> = (0..threads)
                    .map(|t| {
                        let tree = Arc::clone(tree);
                        let warmup_done = Arc::clone(&warmup_done);
                        let start = Arc::clone(&start);
                        let hot = hot_keys.clone();

                        thread::spawn(move || {
                            let guard = sdd::Guard::new();

                            // === WARMUP PHASE ===
                            let mut rng_state = thread_seed(BASE_SEED, t);
                            for _ in 0..WARMUP_OPS {
                                rng_state =
                                    rng_state.wrapping_mul(6364136223846793005).wrapping_add(1);
                                let idx = (rng_state as usize) % hot.len();
                                black_box(tree.peek(&hot[idx], &guard));
                            }
                            warmup_done.wait();

                            // === MEASUREMENT PHASE ===
                            start.wait();
                            pre_measurement_barrier();

                            rng_state = thread_seed(BASE_SEED + 1, t); // Different seed for measurement phase
                            for i in 0..OPS {
                                rng_state =
                                    rng_state.wrapping_mul(6364136223846793005).wrapping_add(1);
                                let idx = (rng_state as usize) % hot.len();

                                if rng_state.is_multiple_of(2) {
                                    black_box(tree.peek(&hot[idx], &guard));
                                } else {
                                    // TreeIndex doesn't provide an in-place upsert/update, so emulate
                                    // MassTree's overwrite semantics with remove+insert.
                                    tree_index_upsert_sync(&tree, hot[idx], i as u64);
                                }
                            }

                            post_measurement_barrier();
                        })
                    })
                    .collect();

                for h in handles {
                    h.join().unwrap();
                }
            });
    }
}

// =============================================================================
// 18: SPLIT-INDUCING SCAN - Sequential inserts cause splits while readers scan
// =============================================================================

#[divan::bench_group(name = "18_split_inducing_scan", sample_count = 200)]
mod split_inducing_scan {
    use super::*;

    const INITIAL_N: usize = 50_000;
    const INSERT_N: usize = 25_000;
    const WRITERS: usize = 2;

    // Minimum 3 threads: 2 writers + 1 reader
    #[divan::bench(args = [3, 4, 6, 8, 12])]
    fn masstree15_inline(bencher: Bencher, threads: usize) {
        let initial_keys = keys_sequential::<8>(INITIAL_N);
        // Sequential keys after initial range - will cause splits
        let insert_keys: Vec<[u8; 8]> = (INITIAL_N..INITIAL_N + INSERT_N)
            .map(|i| (i as u64).to_be_bytes())
            .collect();
        let readers = threads - WRITERS;

        bencher
            .with_inputs(|| {
                let tree = Arc::new(setup_masstree15_inline(&initial_keys));
                (tree, insert_keys.clone())
            })
            .bench_refs(|(tree, new_keys)| {
                let warmup_done = Arc::new(Barrier::new(threads));
                let start = Arc::new(Barrier::new(threads));

                // Writer threads - sequential inserts cause leaf splits
                let writer_handles: Vec<_> = (0..WRITERS)
                    .map(|w| {
                        let tree = Arc::clone(tree);
                        let warmup_done = Arc::clone(&warmup_done);
                        let start = Arc::clone(&start);
                        let chunk_size = INSERT_N / WRITERS;
                        let chunk: Vec<_> = new_keys[w * chunk_size..(w + 1) * chunk_size].to_vec();
                        thread::spawn(move || {
                            let guard = tree.guard();

                            // === WARMUP PHASE ===
                            // Writers do warmup inserts to prime the tree
                            for (i, key) in chunk.iter().take(WARMUP_OPS).enumerate() {
                                let _ = tree.insert_with_guard(key, i as u64, &guard);
                            }
                            warmup_done.wait();

                            // === MEASUREMENT PHASE ===
                            start.wait();
                            pre_measurement_barrier();

                            // Continue with remaining keys after warmup
                            for (i, key) in chunk.iter().skip(WARMUP_OPS).enumerate() {
                                let _ =
                                    tree.insert_with_guard(key, (WARMUP_OPS + i) as u64, &guard);
                            }

                            post_measurement_barrier();
                        })
                    })
                    .collect();

                // Reader threads - scan during structural modifications
                let reader_handles: Vec<_> = (0..readers)
                    .map(|_| {
                        let tree = Arc::clone(tree);
                        let warmup_done = Arc::clone(&warmup_done);
                        let start = Arc::clone(&start);
                        thread::spawn(move || {
                            let guard = tree.guard();

                            // === WARMUP PHASE ===
                            for _ in 0..WARMUP_OPS {
                                let mut count = 0usize;
                                tree.scan_values(
                                    RangeBound::Unbounded,
                                    RangeBound::Unbounded,
                                    |v| {
                                        black_box(v);
                                        count += 1;
                                        count < SCAN_LIMIT
                                    },
                                    &guard,
                                );
                                black_box(count);
                            }
                            warmup_done.wait();

                            // === MEASUREMENT PHASE ===
                            let mut total = 0usize;
                            start.wait();
                            pre_measurement_barrier();

                            for _ in 0..OPS_PER_THREAD {
                                let mut count = 0usize;
                                tree.scan_values(
                                    RangeBound::Unbounded,
                                    RangeBound::Unbounded,
                                    |v| {
                                        black_box(v);
                                        count += 1;
                                        count < SCAN_LIMIT
                                    },
                                    &guard,
                                );
                                total += count;
                            }

                            post_measurement_barrier();
                            black_box(total);
                        })
                    })
                    .collect();

                for h in writer_handles {
                    h.join().unwrap();
                }
                for h in reader_handles {
                    h.join().unwrap();
                }
            });
    }

    #[divan::bench(args = [3, 4, 6, 8, 12])]
    fn tree_index(bencher: Bencher, threads: usize) {
        let initial_keys = keys_sequential::<8>(INITIAL_N);
        // Sequential keys after initial range - will cause splits
        let insert_keys: Vec<[u8; 8]> = (INITIAL_N..INITIAL_N + INSERT_N)
            .map(|i| (i as u64).to_be_bytes())
            .collect();
        let readers = threads - WRITERS;

        bencher
            .with_inputs(|| {
                let tree = Arc::new(setup_tree_index(&initial_keys));
                (tree, insert_keys.clone())
            })
            .bench_refs(|(tree, new_keys)| {
                let warmup_done = Arc::new(Barrier::new(threads));
                let start = Arc::new(Barrier::new(threads));

                // Writer threads - sequential inserts cause leaf splits
                let writer_handles: Vec<_> = (0..WRITERS)
                    .map(|w| {
                        let tree = Arc::clone(tree);
                        let warmup_done = Arc::clone(&warmup_done);
                        let start = Arc::clone(&start);
                        let chunk_size = INSERT_N / WRITERS;
                        let chunk: Vec<_> = new_keys[w * chunk_size..(w + 1) * chunk_size].to_vec();
                        thread::spawn(move || {
                            // === WARMUP PHASE ===
                            // Writers do warmup inserts to prime the tree
                            for (i, key) in chunk.iter().take(WARMUP_OPS).enumerate() {
                                let _ = tree.insert_sync(*key, i as u64);
                            }
                            warmup_done.wait();

                            // === MEASUREMENT PHASE ===
                            start.wait();
                            pre_measurement_barrier();

                            // Continue with remaining keys after warmup
                            for (i, key) in chunk.iter().skip(WARMUP_OPS).enumerate() {
                                let _ = tree.insert_sync(*key, (WARMUP_OPS + i) as u64);
                            }

                            post_measurement_barrier();
                        })
                    })
                    .collect();

                // Reader threads - scan during structural modifications
                let reader_handles: Vec<_> = (0..readers)
                    .map(|_| {
                        let tree = Arc::clone(tree);
                        let warmup_done = Arc::clone(&warmup_done);
                        let start = Arc::clone(&start);
                        thread::spawn(move || {
                            let guard = sdd::Guard::new();

                            // === WARMUP PHASE ===
                            for _ in 0..WARMUP_OPS {
                                let count = tree
                                    .iter(&guard)
                                    .take(SCAN_LIMIT)
                                    .inspect(|(_, v)| {
                                        black_box(*v);
                                    })
                                    .count();
                                black_box(count);
                            }
                            warmup_done.wait();

                            // === MEASUREMENT PHASE ===
                            let mut total = 0usize;
                            start.wait();
                            pre_measurement_barrier();

                            for _ in 0..OPS_PER_THREAD {
                                let count = tree
                                    .iter(&guard)
                                    .take(SCAN_LIMIT)
                                    .inspect(|(_, v)| {
                                        black_box(*v);
                                    })
                                    .count();
                                total += count;
                            }

                            post_measurement_barrier();
                            black_box(total);
                        })
                    })
                    .collect();

                for h in writer_handles {
                    h.join().unwrap();
                }
                for h in reader_handles {
                    h.join().unwrap();
                }
            });
    }
}
