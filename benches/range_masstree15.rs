//! Concurrent range scan stress benchmarks.
//!
//! Compares MassTree15, scc::TreeIndex, and crossbeam_skiplist::SkipMap across
//! various key patterns designed to stress different aspects of concurrent
//! ordered map implementations.
//!
//! ## Configuration
//!
//! - **Dataset size**: 500,000 keys
//! - **Ops per thread**: 5,000
//! - **Thread counts**: 1, 2, 4, 6, 8, 12
//!
//! ## Key Patterns Tested
//!
//! ### General Patterns
//! - Sequential keys (best-case for range scans)
//! - Reverse keys (insertion stress)
//! - Clustered keys (hot ranges with gaps)
//! - Sparse keys (cache miss stress)
//!
//! ### MassTree-Optimized Patterns
//! - Shared prefix (trie prefix sharing)
//! - Suffix-only differ (suffix mechanism)
//! - Hierarchical (namespace:category:id)
//!
//! ### Stress Patterns
//! - Adversarial splits (split propagation)
//! - Interleaved ranges (cache thrashing)
//! - B-link stress (fragmented scans)
//!
//! ## API Differences (Fairness Notes)
//!
//! - **MassTree**: Uses `scan(callback)` — function call overhead per element
//! - **TreeIndex**: Uses `.iter().take()` — lazy iterator
//! - **SkipMap**: Uses `.iter().take()` — lazy iterator, no epoch guard needed
//!
//! These are the native APIs for each implementation. The callback vs iterator
//! difference is inherent to the designs.
//!
//! ## Methodology
//!
//! Each benchmark follows a rigorous methodology:
//! 1. **Workload-matched warmup**: Warmup mirrors measurement scan pattern
//! 2. **Independent randomness**: Each thread has independent RNG streams
//! 3. **Fresh state**: All benchmarks use `.with_inputs()` for fresh tree per sample
//! 4. **Randomized writes**: Write decisions use pre-shuffled arrays, not modulo
//! 5. **Consistent threads**: All benchmarks use [1, 2, 4, 6, 8, 12] thread counts
//! 6. **100 samples**: For statistical significance
//! 7. **Proper barrier placement**: Memory barriers after thread synchronization
//!
//! ## Running
//!
//! ```bash
//! cargo bench --bench range_masstree15
//! cargo bench --bench range_masstree15 --features mimalloc
//!
//! # Specific pattern
//! cargo bench --bench range_masstree15 -- sequential
//! cargo bench --bench range_masstree15 -- hierarchical
//! ```

#![expect(clippy::unwrap_used)]
#![expect(clippy::pedantic)]
#![expect(clippy::indexing_slicing)]

mod bench_utils;

use bench_utils::{
    keys, keys_adversarial_splits, keys_blink_stress, keys_clustered, keys_hierarchical,
    keys_interleaved_ranges, keys_reverse, keys_sequential, keys_shared_prefix, keys_sparse,
    keys_suffix_only_differ, post_measurement_barrier, pre_measurement_barrier,
};
use crossbeam_skiplist::SkipMap;
use divan::{black_box, Bencher};
use masstree::{MassTree15, RangeBound};
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

/// Standard thread counts for all benchmarks
const THREAD_COUNTS: [usize; 6] = [1, 2, 4, 6, 8, 12];

// =============================================================================
// Setup Helpers
// =============================================================================

fn setup_masstree15<const K: usize>(keys: &[[u8; K]]) -> MassTree15<u64> {
    let tree = MassTree15::new();
    {
        let guard = tree.guard();
        for (i, key) in keys.iter().enumerate() {
            let _ = tree.insert_with_guard(key, i as u64, &guard);
        }
    }
    tree
}

fn setup_skipmap<const K: usize>(keys: &[[u8; K]]) -> SkipMap<[u8; K], u64> {
    let map = SkipMap::new();
    for (i, key) in keys.iter().enumerate() {
        map.insert(*key, i as u64);
    }
    map
}

fn setup_tree_index<const K: usize>(keys: &[[u8; K]]) -> TreeIndex<[u8; K], u64> {
    let tree = TreeIndex::new();
    for (i, key) in keys.iter().enumerate() {
        let _ = tree.insert_sync(*key, i as u64);
    }
    tree
}

/// Generate a shuffled array of operation types (true = write, false = read).
/// This avoids the predictable `i % 100 < ratio` pattern which causes all threads
/// to write simultaneously and creates unrealistic branch prediction behavior.
fn shuffled_write_decisions(count: usize, write_ratio_percent: usize, seed: u64) -> Vec<bool> {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};

    let write_count = (count * write_ratio_percent) / 100;
    let mut decisions = vec![false; count];

    // Mark first `write_count` as writes
    for d in decisions.iter_mut().take(write_count) {
        *d = true;
    }

    // Fisher-Yates shuffle with seeded PRNG
    let mut hasher = DefaultHasher::new();
    seed.hash(&mut hasher);
    let mut rng_state = hasher.finish();

    for i in (1..count).rev() {
        // Simple xorshift64 PRNG
        rng_state ^= rng_state << 13;
        rng_state ^= rng_state >> 7;
        rng_state ^= rng_state << 17;

        let j = (rng_state as usize) % (i + 1);
        decisions.swap(i, j);
    }

    decisions
}

/// Generate uniform random indices with independent seed.
/// Each thread should use a different seed for independence.
fn thread_uniform_indices(n: usize, count: usize, seed: u64) -> Vec<usize> {
    let mut indices = Vec::with_capacity(count);
    let mut state = seed;

    for _ in 0..count {
        state = state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1);
        indices.push(((state >> 32) as usize) % n);
    }
    indices
}

/// Generate divergent seed for thread t to avoid correlation.
/// Uses multiplicative hashing to spread seeds across the space.
fn thread_seed(base_seed: u64, thread_id: usize) -> u64 {
    let combined = base_seed.wrapping_add(thread_id as u64);
    // Mix with golden ratio hash
    combined.wrapping_mul(0x9e3779b97f4a7c15)
}

// =============================================================================
// 01: SEQUENTIAL KEYS - Best case for range scans
// =============================================================================

#[divan::bench_group(name = "01_sequential_full_scan", sample_count = 100)]
mod sequential_full_scan {
    use super::*;

    #[divan::bench(args = THREAD_COUNTS)]
    fn masstree15(bencher: Bencher, threads: usize) {
        let keys = Arc::new(keys_sequential::<8>(N));
        let tree = Arc::new(setup_masstree15(&keys));

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
                                tree.scan_intra_leaf_batch_ref(
                                    RangeBound::Unbounded,
                                    RangeBound::Unbounded,
                                    |_, _| {
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
                                tree.scan_intra_leaf_batch_ref(
                                    RangeBound::Unbounded,
                                    RangeBound::Unbounded,
                                    |_, _| {
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
    fn skipmap(bencher: Bencher, threads: usize) {
        let keys = Arc::new(keys_sequential::<8>(N));
        let map = Arc::new(setup_skipmap(&keys));

        bencher
            .counter(divan::counter::ItemsCount::new(threads * OPS_PER_THREAD))
            .bench_local(|| {
                let warmup_done = Arc::new(Barrier::new(threads));
                let start = Arc::new(Barrier::new(threads));
                let handles: Vec<_> = (0..threads)
                    .map(|_| {
                        let map = Arc::clone(&map);
                        let warmup_done = Arc::clone(&warmup_done);
                        let start = Arc::clone(&start);
                        thread::spawn(move || {
                            // === WARMUP PHASE ===
                            for _ in 0..WARMUP_OPS {
                                let count = map.iter().take(SCAN_LIMIT).count();
                                black_box(count);
                            }
                            warmup_done.wait();

                            // === MEASUREMENT PHASE ===
                            let mut total = 0usize;
                            start.wait();
                            pre_measurement_barrier();

                            for _ in 0..OPS_PER_THREAD {
                                let count = map.iter().take(SCAN_LIMIT).count();
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
// 02: REVERSE KEYS - Insertion stress pattern
// =============================================================================

#[divan::bench_group(name = "02_reverse_scan", sample_count = 100)]
mod reverse_scan {
    use super::*;

    #[divan::bench(args = THREAD_COUNTS)]
    fn masstree15(bencher: Bencher, threads: usize) {
        let keys = Arc::new(keys_reverse::<8>(N));
        let tree = Arc::new(setup_masstree15(&keys));

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
                                tree.scan_intra_leaf_batch_ref(
                                    RangeBound::Unbounded,
                                    RangeBound::Unbounded,
                                    |_, _| {
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
                                tree.scan_intra_leaf_batch_ref(
                                    RangeBound::Unbounded,
                                    RangeBound::Unbounded,
                                    |_, _| {
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
    fn skipmap(bencher: Bencher, threads: usize) {
        let keys = Arc::new(keys_reverse::<8>(N));
        let map = Arc::new(setup_skipmap(&keys));

        bencher
            .counter(divan::counter::ItemsCount::new(threads * OPS_PER_THREAD))
            .bench_local(|| {
                let warmup_done = Arc::new(Barrier::new(threads));
                let start = Arc::new(Barrier::new(threads));
                let handles: Vec<_> = (0..threads)
                    .map(|_| {
                        let map = Arc::clone(&map);
                        let warmup_done = Arc::clone(&warmup_done);
                        let start = Arc::clone(&start);
                        thread::spawn(move || {
                            // === WARMUP PHASE ===
                            for _ in 0..WARMUP_OPS {
                                let count = map.iter().take(SCAN_LIMIT).count();
                                black_box(count);
                            }
                            warmup_done.wait();

                            // === MEASUREMENT PHASE ===
                            let mut total = 0usize;
                            start.wait();
                            pre_measurement_barrier();

                            for _ in 0..OPS_PER_THREAD {
                                let count = map.iter().take(SCAN_LIMIT).count();
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
// 03: CLUSTERED KEYS - Hot ranges with gaps
// =============================================================================

#[divan::bench_group(name = "03_clustered_scan", sample_count = 100)]
mod clustered_scan {
    use super::*;

    const CLUSTERS: usize = 500;
    const KEYS_PER_CLUSTER: usize = N / CLUSTERS;
    const GAP_SIZE: u64 = 10_000;

    #[divan::bench(args = THREAD_COUNTS)]
    fn masstree15(bencher: Bencher, threads: usize) {
        let keys = Arc::new(keys_clustered::<8>(CLUSTERS, KEYS_PER_CLUSTER, GAP_SIZE));
        let tree = Arc::new(setup_masstree15(&keys));

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
                                tree.scan_intra_leaf_batch_ref(
                                    RangeBound::Unbounded,
                                    RangeBound::Unbounded,
                                    |_, _| {
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
                                tree.scan_intra_leaf_batch_ref(
                                    RangeBound::Unbounded,
                                    RangeBound::Unbounded,
                                    |_, _| {
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
    fn skipmap(bencher: Bencher, threads: usize) {
        let keys = Arc::new(keys_clustered::<8>(CLUSTERS, KEYS_PER_CLUSTER, GAP_SIZE));
        let map = Arc::new(setup_skipmap(&keys));

        bencher
            .counter(divan::counter::ItemsCount::new(threads * OPS_PER_THREAD))
            .bench_local(|| {
                let warmup_done = Arc::new(Barrier::new(threads));
                let start = Arc::new(Barrier::new(threads));
                let handles: Vec<_> = (0..threads)
                    .map(|_| {
                        let map = Arc::clone(&map);
                        let warmup_done = Arc::clone(&warmup_done);
                        let start = Arc::clone(&start);
                        thread::spawn(move || {
                            // === WARMUP PHASE ===
                            for _ in 0..WARMUP_OPS {
                                let count = map.iter().take(SCAN_LIMIT).count();
                                black_box(count);
                            }
                            warmup_done.wait();

                            // === MEASUREMENT PHASE ===
                            let mut total = 0usize;
                            start.wait();
                            pre_measurement_barrier();

                            for _ in 0..OPS_PER_THREAD {
                                let count = map.iter().take(SCAN_LIMIT).count();
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
// 04: SPARSE KEYS - Cache miss stress
// =============================================================================

#[divan::bench_group(name = "04_sparse_scan", sample_count = 100)]
mod sparse_scan {
    use super::*;

    const SPACING: u64 = 1000;

    #[divan::bench(args = THREAD_COUNTS)]
    fn masstree15(bencher: Bencher, threads: usize) {
        let keys = Arc::new(keys_sparse::<8>(N, SPACING));
        let tree = Arc::new(setup_masstree15(&keys));

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
                                tree.scan_intra_leaf_batch_ref(
                                    RangeBound::Unbounded,
                                    RangeBound::Unbounded,
                                    |_, _| {
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
                                tree.scan_intra_leaf_batch_ref(
                                    RangeBound::Unbounded,
                                    RangeBound::Unbounded,
                                    |_, _| {
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
    fn skipmap(bencher: Bencher, threads: usize) {
        let keys = Arc::new(keys_sparse::<8>(N, SPACING));
        let map = Arc::new(setup_skipmap(&keys));

        bencher
            .counter(divan::counter::ItemsCount::new(threads * OPS_PER_THREAD))
            .bench_local(|| {
                let warmup_done = Arc::new(Barrier::new(threads));
                let start = Arc::new(Barrier::new(threads));
                let handles: Vec<_> = (0..threads)
                    .map(|_| {
                        let map = Arc::clone(&map);
                        let warmup_done = Arc::clone(&warmup_done);
                        let start = Arc::clone(&start);
                        thread::spawn(move || {
                            // === WARMUP PHASE ===
                            for _ in 0..WARMUP_OPS {
                                let count = map.iter().take(SCAN_LIMIT).count();
                                black_box(count);
                            }
                            warmup_done.wait();

                            // === MEASUREMENT PHASE ===
                            let mut total = 0usize;
                            start.wait();
                            pre_measurement_barrier();

                            for _ in 0..OPS_PER_THREAD {
                                let count = map.iter().take(SCAN_LIMIT).count();
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
// 05: SHARED PREFIX - MassTree trie advantage (16B keys)
// =============================================================================

#[divan::bench_group(name = "05_shared_prefix_scan", sample_count = 100)]
mod shared_prefix_scan {
    use super::*;

    const PREFIX_BUCKETS: u64 = 100; // 10k keys per prefix

    #[divan::bench(args = THREAD_COUNTS)]
    fn masstree15(bencher: Bencher, threads: usize) {
        let keys = Arc::new(keys_shared_prefix::<16>(N, PREFIX_BUCKETS));
        let tree = Arc::new(setup_masstree15(&keys));

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
                                tree.scan_intra_leaf_batch_ref(
                                    RangeBound::Unbounded,
                                    RangeBound::Unbounded,
                                    |_, _| {
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
                                tree.scan_intra_leaf_batch_ref(
                                    RangeBound::Unbounded,
                                    RangeBound::Unbounded,
                                    |_, _| {
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
    fn skipmap(bencher: Bencher, threads: usize) {
        let keys = Arc::new(keys_shared_prefix::<16>(N, PREFIX_BUCKETS));
        let map = Arc::new(setup_skipmap(&keys));

        bencher
            .counter(divan::counter::ItemsCount::new(threads * OPS_PER_THREAD))
            .bench_local(|| {
                let warmup_done = Arc::new(Barrier::new(threads));
                let start = Arc::new(Barrier::new(threads));
                let handles: Vec<_> = (0..threads)
                    .map(|_| {
                        let map = Arc::clone(&map);
                        let warmup_done = Arc::clone(&warmup_done);
                        let start = Arc::clone(&start);
                        thread::spawn(move || {
                            // === WARMUP PHASE ===
                            for _ in 0..WARMUP_OPS {
                                let count = map.iter().take(SCAN_LIMIT).count();
                                black_box(count);
                            }
                            warmup_done.wait();

                            // === MEASUREMENT PHASE ===
                            let mut total = 0usize;
                            start.wait();
                            pre_measurement_barrier();

                            for _ in 0..OPS_PER_THREAD {
                                let count = map.iter().take(SCAN_LIMIT).count();
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

#[divan::bench_group(name = "06_suffix_differ_scan", sample_count = 100)]
mod suffix_differ_scan {
    use super::*;

    #[divan::bench(args = THREAD_COUNTS)]
    fn masstree15(bencher: Bencher, threads: usize) {
        let keys = Arc::new(keys_suffix_only_differ::<32>(N));
        let tree = Arc::new(setup_masstree15(&keys));

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
                                tree.scan_intra_leaf_batch_ref(
                                    RangeBound::Unbounded,
                                    RangeBound::Unbounded,
                                    |_, _| {
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
                                tree.scan_intra_leaf_batch_ref(
                                    RangeBound::Unbounded,
                                    RangeBound::Unbounded,
                                    |_, _| {
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
    fn skipmap(bencher: Bencher, threads: usize) {
        let keys = Arc::new(keys_suffix_only_differ::<32>(N));
        let map = Arc::new(setup_skipmap(&keys));

        bencher
            .counter(divan::counter::ItemsCount::new(threads * OPS_PER_THREAD))
            .bench_local(|| {
                let warmup_done = Arc::new(Barrier::new(threads));
                let start = Arc::new(Barrier::new(threads));
                let handles: Vec<_> = (0..threads)
                    .map(|_| {
                        let map = Arc::clone(&map);
                        let warmup_done = Arc::clone(&warmup_done);
                        let start = Arc::clone(&start);
                        thread::spawn(move || {
                            // === WARMUP PHASE ===
                            for _ in 0..WARMUP_OPS {
                                let count = map.iter().take(SCAN_LIMIT).count();
                                black_box(count);
                            }
                            warmup_done.wait();

                            // === MEASUREMENT PHASE ===
                            let mut total = 0usize;
                            start.wait();
                            pre_measurement_barrier();

                            for _ in 0..OPS_PER_THREAD {
                                let count = map.iter().take(SCAN_LIMIT).count();
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

#[divan::bench_group(name = "07_hierarchical_scan", sample_count = 100)]
mod hierarchical_scan {
    use super::*;

    // 100 namespaces * 100 categories * 100 items = 1M keys
    const NAMESPACES: usize = 100;
    const CATEGORIES: usize = 100;
    const ITEMS: usize = 50;

    #[divan::bench(args = THREAD_COUNTS)]
    fn masstree15(bencher: Bencher, threads: usize) {
        let keys = Arc::new(keys_hierarchical::<32>(NAMESPACES, CATEGORIES, ITEMS));
        let tree = Arc::new(setup_masstree15(&keys));

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
                                tree.scan_intra_leaf_batch_ref(
                                    RangeBound::Unbounded,
                                    RangeBound::Unbounded,
                                    |_, _| {
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
                                tree.scan_intra_leaf_batch_ref(
                                    RangeBound::Unbounded,
                                    RangeBound::Unbounded,
                                    |_, _| {
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
    fn skipmap(bencher: Bencher, threads: usize) {
        let keys = Arc::new(keys_hierarchical::<32>(NAMESPACES, CATEGORIES, ITEMS));
        let map = Arc::new(setup_skipmap(&keys));

        bencher
            .counter(divan::counter::ItemsCount::new(threads * OPS_PER_THREAD))
            .bench_local(|| {
                let warmup_done = Arc::new(Barrier::new(threads));
                let start = Arc::new(Barrier::new(threads));
                let handles: Vec<_> = (0..threads)
                    .map(|_| {
                        let map = Arc::clone(&map);
                        let warmup_done = Arc::clone(&warmup_done);
                        let start = Arc::clone(&start);
                        thread::spawn(move || {
                            // === WARMUP PHASE ===
                            for _ in 0..WARMUP_OPS {
                                let count = map.iter().take(SCAN_LIMIT).count();
                                black_box(count);
                            }
                            warmup_done.wait();

                            // === MEASUREMENT PHASE ===
                            let mut total = 0usize;
                            start.wait();
                            pre_measurement_barrier();

                            for _ in 0..OPS_PER_THREAD {
                                let count = map.iter().take(SCAN_LIMIT).count();
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

#[divan::bench_group(name = "08_adversarial_splits_scan", sample_count = 100)]
mod adversarial_splits_scan {
    use super::*;

    #[divan::bench(args = THREAD_COUNTS)]
    fn masstree15(bencher: Bencher, threads: usize) {
        let keys = Arc::new(keys_adversarial_splits::<8>(N));
        let tree = Arc::new(setup_masstree15(&keys));

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
                                tree.scan_intra_leaf_batch_ref(
                                    RangeBound::Unbounded,
                                    RangeBound::Unbounded,
                                    |_, _| {
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
                                tree.scan_intra_leaf_batch_ref(
                                    RangeBound::Unbounded,
                                    RangeBound::Unbounded,
                                    |_, _| {
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
    fn skipmap(bencher: Bencher, threads: usize) {
        let keys = Arc::new(keys_adversarial_splits::<8>(N));
        let map = Arc::new(setup_skipmap(&keys));

        bencher
            .counter(divan::counter::ItemsCount::new(threads * OPS_PER_THREAD))
            .bench_local(|| {
                let warmup_done = Arc::new(Barrier::new(threads));
                let start = Arc::new(Barrier::new(threads));
                let handles: Vec<_> = (0..threads)
                    .map(|_| {
                        let map = Arc::clone(&map);
                        let warmup_done = Arc::clone(&warmup_done);
                        let start = Arc::clone(&start);
                        thread::spawn(move || {
                            // === WARMUP PHASE ===
                            for _ in 0..WARMUP_OPS {
                                let count = map.iter().take(SCAN_LIMIT).count();
                                black_box(count);
                            }
                            warmup_done.wait();

                            // === MEASUREMENT PHASE ===
                            let mut total = 0usize;
                            start.wait();
                            pre_measurement_barrier();

                            for _ in 0..OPS_PER_THREAD {
                                let count = map.iter().take(SCAN_LIMIT).count();
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

#[divan::bench_group(name = "09_interleaved_scan", sample_count = 100)]
mod interleaved_scan {
    use super::*;

    const HOT_RANGES: usize = 50;
    const KEYS_PER_RANGE: usize = N / HOT_RANGES;
    const COLD_GAP: u64 = 100_000;

    #[divan::bench(args = THREAD_COUNTS)]
    fn masstree15(bencher: Bencher, threads: usize) {
        let keys = Arc::new(keys_interleaved_ranges::<8>(
            HOT_RANGES,
            KEYS_PER_RANGE,
            COLD_GAP,
        ));
        let tree = Arc::new(setup_masstree15(&keys));

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
                                tree.scan_intra_leaf_batch_ref(
                                    RangeBound::Unbounded,
                                    RangeBound::Unbounded,
                                    |_, _| {
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
                                tree.scan_intra_leaf_batch_ref(
                                    RangeBound::Unbounded,
                                    RangeBound::Unbounded,
                                    |_, _| {
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
    fn skipmap(bencher: Bencher, threads: usize) {
        let keys = Arc::new(keys_interleaved_ranges::<8>(
            HOT_RANGES,
            KEYS_PER_RANGE,
            COLD_GAP,
        ));
        let map = Arc::new(setup_skipmap(&keys));

        bencher
            .counter(divan::counter::ItemsCount::new(threads * OPS_PER_THREAD))
            .bench_local(|| {
                let warmup_done = Arc::new(Barrier::new(threads));
                let start = Arc::new(Barrier::new(threads));
                let handles: Vec<_> = (0..threads)
                    .map(|_| {
                        let map = Arc::clone(&map);
                        let warmup_done = Arc::clone(&warmup_done);
                        let start = Arc::clone(&start);
                        thread::spawn(move || {
                            // === WARMUP PHASE ===
                            for _ in 0..WARMUP_OPS {
                                let count = map.iter().take(SCAN_LIMIT).count();
                                black_box(count);
                            }
                            warmup_done.wait();

                            // === MEASUREMENT PHASE ===
                            let mut total = 0usize;
                            start.wait();
                            pre_measurement_barrier();

                            for _ in 0..OPS_PER_THREAD {
                                let count = map.iter().take(SCAN_LIMIT).count();
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

#[divan::bench_group(name = "10_blink_stress_scan", sample_count = 100)]
mod blink_stress_scan {
    use super::*;

    #[divan::bench(args = THREAD_COUNTS)]
    fn masstree15(bencher: Bencher, threads: usize) {
        let keys = Arc::new(keys_blink_stress::<8>(N));
        let tree = Arc::new(setup_masstree15(&keys));

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
                                tree.scan_intra_leaf_batch_ref(
                                    RangeBound::Unbounded,
                                    RangeBound::Unbounded,
                                    |_, _| {
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
                                tree.scan_intra_leaf_batch_ref(
                                    RangeBound::Unbounded,
                                    RangeBound::Unbounded,
                                    |_, _| {
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
    fn skipmap(bencher: Bencher, threads: usize) {
        let keys = Arc::new(keys_blink_stress::<8>(N));
        let map = Arc::new(setup_skipmap(&keys));

        bencher
            .counter(divan::counter::ItemsCount::new(threads * OPS_PER_THREAD))
            .bench_local(|| {
                let warmup_done = Arc::new(Barrier::new(threads));
                let start = Arc::new(Barrier::new(threads));
                let handles: Vec<_> = (0..threads)
                    .map(|_| {
                        let map = Arc::clone(&map);
                        let warmup_done = Arc::clone(&warmup_done);
                        let start = Arc::clone(&start);
                        thread::spawn(move || {
                            // === WARMUP PHASE ===
                            for _ in 0..WARMUP_OPS {
                                let count = map.iter().take(SCAN_LIMIT).count();
                                black_box(count);
                            }
                            warmup_done.wait();

                            // === MEASUREMENT PHASE ===
                            let mut total = 0usize;
                            start.wait();
                            pre_measurement_barrier();

                            for _ in 0..OPS_PER_THREAD {
                                let count = map.iter().take(SCAN_LIMIT).count();
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

#[divan::bench_group(name = "11_random_keys_scan", sample_count = 100)]
mod random_keys_scan {
    use super::*;

    #[divan::bench(args = THREAD_COUNTS)]
    fn masstree15(bencher: Bencher, threads: usize) {
        let keys = Arc::new(keys::<8>(N));
        let tree = Arc::new(setup_masstree15(&keys));

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
                                tree.scan_intra_leaf_batch_ref(
                                    RangeBound::Unbounded,
                                    RangeBound::Unbounded,
                                    |_, _| {
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
                                tree.scan_intra_leaf_batch_ref(
                                    RangeBound::Unbounded,
                                    RangeBound::Unbounded,
                                    |_, _| {
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
    fn skipmap(bencher: Bencher, threads: usize) {
        let keys = Arc::new(keys::<8>(N));
        let map = Arc::new(setup_skipmap(&keys));

        bencher
            .counter(divan::counter::ItemsCount::new(threads * OPS_PER_THREAD))
            .bench_local(|| {
                let warmup_done = Arc::new(Barrier::new(threads));
                let start = Arc::new(Barrier::new(threads));
                let handles: Vec<_> = (0..threads)
                    .map(|_| {
                        let map = Arc::clone(&map);
                        let warmup_done = Arc::clone(&warmup_done);
                        let start = Arc::clone(&start);
                        thread::spawn(move || {
                            // === WARMUP PHASE ===
                            for _ in 0..WARMUP_OPS {
                                let count = map.iter().take(SCAN_LIMIT).count();
                                black_box(count);
                            }
                            warmup_done.wait();

                            // === MEASUREMENT PHASE ===
                            let mut total = 0usize;
                            start.wait();
                            pre_measurement_barrier();

                            for _ in 0..OPS_PER_THREAD {
                                let count = map.iter().take(SCAN_LIMIT).count();
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

#[divan::bench_group(name = "12_long_keys_64b_scan", sample_count = 100)]
mod long_keys_scan {
    use super::*;

    #[divan::bench(args = THREAD_COUNTS)]
    fn masstree15(bencher: Bencher, threads: usize) {
        let keys = Arc::new(keys::<64>(N));
        let tree = Arc::new(setup_masstree15(&keys));

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
                                tree.scan_intra_leaf_batch_ref(
                                    RangeBound::Unbounded,
                                    RangeBound::Unbounded,
                                    |_, _| {
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
                                tree.scan_intra_leaf_batch_ref(
                                    RangeBound::Unbounded,
                                    RangeBound::Unbounded,
                                    |_, _| {
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
    fn skipmap(bencher: Bencher, threads: usize) {
        let keys = Arc::new(keys::<64>(N));
        let map = Arc::new(setup_skipmap(&keys));

        bencher
            .counter(divan::counter::ItemsCount::new(threads * OPS_PER_THREAD))
            .bench_local(|| {
                let warmup_done = Arc::new(Barrier::new(threads));
                let start = Arc::new(Barrier::new(threads));
                let handles: Vec<_> = (0..threads)
                    .map(|_| {
                        let map = Arc::clone(&map);
                        let warmup_done = Arc::clone(&warmup_done);
                        let start = Arc::clone(&start);
                        thread::spawn(move || {
                            // === WARMUP PHASE ===
                            for _ in 0..WARMUP_OPS {
                                let count = map.iter().take(SCAN_LIMIT).count();
                                black_box(count);
                            }
                            warmup_done.wait();

                            // === MEASUREMENT PHASE ===
                            let mut total = 0usize;
                            start.wait();
                            pre_measurement_barrier();

                            for _ in 0..OPS_PER_THREAD {
                                let count = map.iter().take(SCAN_LIMIT).count();
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

#[divan::bench_group(name = "13_scan_while_insert", sample_count = 100)]
mod scan_while_insert {
    use super::*;

    const INITIAL_N: usize = 450_000;
    const INSERT_N: usize = 25_000;
    const WRITERS: usize = 2;

    // Minimum 3 threads: 2 writers + 1 reader
    #[divan::bench(args = [3, 6, 9, 12])]
    fn masstree15(bencher: Bencher, threads: usize) {
        let initial_keys = keys::<8>(INITIAL_N);
        let insert_keys: Vec<_> = keys::<8>(INITIAL_N + INSERT_N)[INITIAL_N..].to_vec();
        let readers = threads - WRITERS;

        bencher
            .with_inputs(|| {
                let tree = Arc::new(setup_masstree15(&initial_keys));
                let new_keys = Arc::new(insert_keys.clone());
                (tree, new_keys)
            })
            .bench_refs(|(tree, new_keys)| {
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
                        let warmup_keys = initial_keys.clone();
                        thread::spawn(move || {
                            let guard = tree.guard();

                            // === WARMUP PHASE (read-only to avoid consuming insert keys) ===
                            for i in 0..WARMUP_OPS {
                                let idx = i % INITIAL_N;
                                black_box(tree.get_with_guard(&warmup_keys[idx], &guard));
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
                                tree.scan_ref(
                                    RangeBound::Unbounded,
                                    RangeBound::Unbounded,
                                    |_, _| {
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
                                tree.scan_ref(
                                    RangeBound::Unbounded,
                                    RangeBound::Unbounded,
                                    |_, _| {
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
}

// =============================================================================
// 14: PREFIX SCAN - MassTree-specific optimization
// =============================================================================

#[divan::bench_group(name = "14_prefix_scan", sample_count = 100)]
mod prefix_scan {
    use super::*;

    const PREFIX_BUCKETS: u64 = 100;
    // Reduced ops for prefix scan - each scan touches 10K entries (1M/100 buckets)
    const PREFIX_OPS: usize = 50;

    #[divan::bench(args = THREAD_COUNTS)]
    fn masstree15(bencher: Bencher, threads: usize) {
        let keys = Arc::new(keys_shared_prefix::<16>(N, PREFIX_BUCKETS));
        let tree = Arc::new(setup_masstree15(&keys));

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
                                black_box(tree.scan_prefix(&thread_prefix, |_, _| true, &guard));
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
// 15: FULL SCAN AGGREGATE - Sum all values
// =============================================================================

#[divan::bench_group(name = "15_full_scan_aggregate", sample_count = 100)]
mod full_scan_aggregate {
    use super::*;

    const SCAN_N: usize = 25_000; // Smaller dataset for full scans
    const FULL_SCAN_OPS: usize = 50; // Reduced iterations for full scans

    #[divan::bench(args = THREAD_COUNTS)]
    fn masstree15(bencher: Bencher, threads: usize) {
        let keys = Arc::new(keys_sequential::<8>(SCAN_N));
        let tree = Arc::new(setup_masstree15(&keys));

        bencher
            .counter(divan::counter::ItemsCount::new(threads * FULL_SCAN_OPS))
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
                                tree.scan_intra_leaf_batch_ref(
                                    RangeBound::Unbounded,
                                    RangeBound::Unbounded,
                                    |_, v| {
                                        sum += *v;
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
                                tree.scan_intra_leaf_batch_ref(
                                    RangeBound::Unbounded,
                                    RangeBound::Unbounded,
                                    |_, v| {
                                        sum += *v;
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
    fn skipmap(bencher: Bencher, threads: usize) {
        let keys = Arc::new(keys_sequential::<8>(SCAN_N));
        let map = Arc::new(setup_skipmap(&keys));

        bencher
            .counter(divan::counter::ItemsCount::new(threads * FULL_SCAN_OPS))
            .bench_local(|| {
                let warmup_done = Arc::new(Barrier::new(threads));
                let start = Arc::new(Barrier::new(threads));
                let handles: Vec<_> = (0..threads)
                    .map(|_| {
                        let map = Arc::clone(&map);
                        let warmup_done = Arc::clone(&warmup_done);
                        let start = Arc::clone(&start);
                        thread::spawn(move || {
                            // === WARMUP PHASE ===
                            for _ in 0..WARMUP_OPS {
                                let sum: u64 = map.iter().map(|e| *e.value()).sum();
                                black_box(sum);
                            }
                            warmup_done.wait();

                            // === MEASUREMENT PHASE ===
                            let mut grand_total = 0u64;
                            start.wait();
                            pre_measurement_barrier();

                            for _ in 0..FULL_SCAN_OPS {
                                let sum: u64 = map.iter().map(|e| *e.value()).sum();
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

        bencher
            .counter(divan::counter::ItemsCount::new(threads * FULL_SCAN_OPS))
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

#[divan::bench_group(name = "16_insert_heavy", sample_count = 100)]
mod insert_heavy {
    use super::*;

    const INITIAL_N: usize = 50_000;
    const OPS: usize = 5_000;
    const WRITE_RATIO: usize = 90; // 90% writes

    #[divan::bench(args = THREAD_COUNTS)]
    fn masstree15(bencher: Bencher, threads: usize) {
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
                let tree = Arc::new(setup_masstree15(&initial_keys));
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
                            let seed = thread_seed(42, t);
                            let read_indices = thread_uniform_indices(INITIAL_N, OPS, seed);
                            let is_write = shuffled_write_decisions(OPS, WRITE_RATIO, seed);
                            let warmup_is_write = shuffled_write_decisions(
                                WARMUP_OPS,
                                WRITE_RATIO,
                                seed.wrapping_add(1),
                            );

                            let guard = tree.guard();

                            // === WARMUP PHASE (read-only to avoid consuming insert keys) ===
                            for i in 0..WARMUP_OPS {
                                let idx = read_indices[i % OPS];
                                if warmup_is_write[i] {
                                    // Warmup writes are actually reads (can't consume insert keys early)
                                    black_box(tree.get_with_guard(&read_keys[idx], &guard));
                                } else {
                                    black_box(tree.get_with_guard(&read_keys[idx], &guard));
                                }
                            }
                            warmup_done.wait();

                            // === MEASUREMENT PHASE ===
                            start.wait();
                            pre_measurement_barrier();

                            for (i, key) in keys.iter().enumerate() {
                                if is_write[i] {
                                    let _ = tree.insert_with_guard(key, i as u64, &guard);
                                } else {
                                    let idx = read_indices[i];
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
    fn skipmap(bencher: Bencher, threads: usize) {
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
                let map = Arc::new(setup_skipmap(&initial_keys));
                (map, insert_keys.clone())
            })
            .bench_refs(|(map, thread_keys)| {
                let warmup_done = Arc::new(Barrier::new(threads));
                let start = Arc::new(Barrier::new(threads));
                let handles: Vec<_> = (0..threads)
                    .map(|t| {
                        let map = Arc::clone(map);
                        let warmup_done = Arc::clone(&warmup_done);
                        let start = Arc::clone(&start);
                        let keys = thread_keys[t].clone();
                        let read_keys = initial_keys.clone();
                        thread::spawn(move || {
                            let seed = thread_seed(42, t);
                            let read_indices = thread_uniform_indices(INITIAL_N, OPS, seed);
                            let is_write = shuffled_write_decisions(OPS, WRITE_RATIO, seed);

                            // === WARMUP PHASE ===
                            for i in 0..WARMUP_OPS {
                                let idx = read_indices[i % OPS];
                                black_box(map.get(&read_keys[idx]));
                            }
                            warmup_done.wait();

                            // === MEASUREMENT PHASE ===
                            start.wait();
                            pre_measurement_barrier();

                            for (i, key) in keys.iter().enumerate() {
                                if is_write[i] {
                                    map.insert(*key, i as u64);
                                } else {
                                    let idx = read_indices[i];
                                    black_box(map.get(&read_keys[idx]));
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
                            let seed = thread_seed(42, t);
                            let read_indices = thread_uniform_indices(INITIAL_N, OPS, seed);
                            let is_write = shuffled_write_decisions(OPS, WRITE_RATIO, seed);

                            let guard = sdd::Guard::new();

                            // === WARMUP PHASE ===
                            for i in 0..WARMUP_OPS {
                                let idx = read_indices[i % OPS];
                                black_box(tree.peek(&read_keys[idx], &guard));
                            }
                            warmup_done.wait();

                            // === MEASUREMENT PHASE ===
                            start.wait();
                            pre_measurement_barrier();

                            for (i, key) in keys.iter().enumerate() {
                                if is_write[i] {
                                    let _ = tree.insert_sync(*key, i as u64);
                                } else {
                                    let idx = read_indices[i];
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
//
// Note: This benchmark measures atomic CAS retry behavior under extreme
// contention, NOT tree traversal performance. All threads compete for the
// same small set of keys, testing the lock/version protocol under worst-case.

#[divan::bench_group(name = "17_hot_spot", sample_count = 100)]
mod hot_spot {
    use super::*;

    const TOTAL_N: usize = 500_000;
    const HOT_RANGE: usize = 5; // Only 5 keys are "hot" - extreme contention
    const OPS: usize = 5_000;
    const WRITE_RATIO: usize = 50; // 50% writes

    #[divan::bench(args = THREAD_COUNTS)]
    fn masstree15(bencher: Bencher, threads: usize) {
        let all_keys: Vec<[u8; 8]> = keys_sequential::<8>(TOTAL_N);
        let hot_keys: Vec<[u8; 8]> = all_keys[..HOT_RANGE].to_vec();

        bencher
            .counter(divan::counter::ItemsCount::new(threads * OPS))
            .with_inputs(|| Arc::new(setup_masstree15(&all_keys)))
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
                            let seed = thread_seed(42, t);
                            let hot_indices = thread_uniform_indices(HOT_RANGE, OPS, seed);
                            let is_write = shuffled_write_decisions(OPS, WRITE_RATIO, seed);
                            let warmup_is_write = shuffled_write_decisions(
                                WARMUP_OPS,
                                WRITE_RATIO,
                                seed.wrapping_add(1),
                            );

                            let guard = tree.guard();

                            // === WARMUP PHASE ===
                            for i in 0..WARMUP_OPS {
                                let idx = hot_indices[i % OPS];
                                if warmup_is_write[i] {
                                    let _ = tree.insert_with_guard(&hot[idx], i as u64, &guard);
                                } else {
                                    black_box(tree.get_with_guard(&hot[idx], &guard));
                                }
                            }
                            warmup_done.wait();

                            // === MEASUREMENT PHASE ===
                            let mut sum = 0u64;
                            start.wait();
                            pre_measurement_barrier();

                            for i in 0..OPS {
                                let idx = hot_indices[i];
                                if is_write[i] {
                                    let _ = tree.insert_with_guard(&hot[idx], i as u64, &guard);
                                } else if let Some(v) = tree.get_with_guard(&hot[idx], &guard) {
                                    sum = sum.wrapping_add(*v);
                                }
                            }

                            post_measurement_barrier();
                            black_box(sum);
                        })
                    })
                    .collect();

                for h in handles {
                    h.join().unwrap();
                }
            });
    }

    #[divan::bench(args = THREAD_COUNTS)]
    fn skipmap(bencher: Bencher, threads: usize) {
        let all_keys: Vec<[u8; 8]> = keys_sequential::<8>(TOTAL_N);
        let hot_keys: Vec<[u8; 8]> = all_keys[..HOT_RANGE].to_vec();

        bencher
            .counter(divan::counter::ItemsCount::new(threads * OPS))
            .with_inputs(|| Arc::new(setup_skipmap(&all_keys)))
            .bench_refs(|map| {
                let warmup_done = Arc::new(Barrier::new(threads));
                let start = Arc::new(Barrier::new(threads));
                let handles: Vec<_> = (0..threads)
                    .map(|t| {
                        let map = Arc::clone(map);
                        let warmup_done = Arc::clone(&warmup_done);
                        let start = Arc::clone(&start);
                        let hot = hot_keys.clone();

                        thread::spawn(move || {
                            let seed = thread_seed(42, t);
                            let hot_indices = thread_uniform_indices(HOT_RANGE, OPS, seed);
                            let is_write = shuffled_write_decisions(OPS, WRITE_RATIO, seed);
                            let warmup_is_write = shuffled_write_decisions(
                                WARMUP_OPS,
                                WRITE_RATIO,
                                seed.wrapping_add(1),
                            );

                            // === WARMUP PHASE ===
                            for i in 0..WARMUP_OPS {
                                let idx = hot_indices[i % OPS];
                                if warmup_is_write[i] {
                                    map.insert(hot[idx], i as u64);
                                } else {
                                    black_box(map.get(&hot[idx]));
                                }
                            }
                            warmup_done.wait();

                            // === MEASUREMENT PHASE ===
                            let mut sum = 0u64;
                            start.wait();
                            pre_measurement_barrier();

                            for i in 0..OPS {
                                let idx = hot_indices[i];
                                if is_write[i] {
                                    map.insert(hot[idx], i as u64);
                                } else if let Some(entry) = map.get(&hot[idx]) {
                                    sum = sum.wrapping_add(*entry.value());
                                }
                            }

                            post_measurement_barrier();
                            black_box(sum);
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
                            let seed = thread_seed(42, t);
                            let hot_indices = thread_uniform_indices(HOT_RANGE, OPS, seed);
                            let is_write = shuffled_write_decisions(OPS, WRITE_RATIO, seed);
                            let warmup_is_write = shuffled_write_decisions(
                                WARMUP_OPS,
                                WRITE_RATIO,
                                seed.wrapping_add(1),
                            );

                            let guard = sdd::Guard::new();

                            // === WARMUP PHASE ===
                            for i in 0..WARMUP_OPS {
                                let idx = hot_indices[i % OPS];
                                if warmup_is_write[i] {
                                    let _ = tree.insert_sync(hot[idx], i as u64);
                                } else {
                                    black_box(tree.peek(&hot[idx], &guard));
                                }
                            }
                            warmup_done.wait();

                            // === MEASUREMENT PHASE ===
                            let mut sum = 0u64;
                            start.wait();
                            pre_measurement_barrier();

                            for i in 0..OPS {
                                let idx = hot_indices[i];
                                if is_write[i] {
                                    let _ = tree.insert_sync(hot[idx], i as u64);
                                } else if let Some(v) = tree.peek(&hot[idx], &guard) {
                                    sum = sum.wrapping_add(*v);
                                }
                            }

                            post_measurement_barrier();
                            black_box(sum);
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

#[divan::bench_group(name = "18_split_inducing_scan", sample_count = 100)]
mod split_inducing_scan {
    use super::*;

    const INITIAL_N: usize = 50_000;
    const INSERT_N: usize = 25_000;
    const WRITERS: usize = 2;

    // Minimum 3 threads: 2 writers + 1 reader
    #[divan::bench(args = [3, 6, 9, 12])]
    fn masstree15(bencher: Bencher, threads: usize) {
        let initial_keys = keys_sequential::<8>(INITIAL_N);
        // Sequential keys after initial range - will cause splits
        let insert_keys: Vec<[u8; 8]> = (INITIAL_N..INITIAL_N + INSERT_N)
            .map(|i| (i as u64).to_be_bytes())
            .collect();
        let readers = threads - WRITERS;

        bencher
            .with_inputs(|| {
                let tree = Arc::new(setup_masstree15(&initial_keys));
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
                        let warmup_keys = initial_keys.clone();
                        thread::spawn(move || {
                            let guard = tree.guard();

                            // === WARMUP PHASE (read-only to avoid consuming insert keys) ===
                            for i in 0..WARMUP_OPS {
                                let idx = i % INITIAL_N;
                                black_box(tree.get_with_guard(&warmup_keys[idx], &guard));
                            }
                            warmup_done.wait();

                            // === MEASUREMENT PHASE ===
                            start.wait();
                            pre_measurement_barrier();

                            for (i, key) in chunk.iter().enumerate() {
                                let _ = tree.insert_with_guard(key, i as u64, &guard);
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
                                tree.scan_ref(
                                    RangeBound::Unbounded,
                                    RangeBound::Unbounded,
                                    |_, v| {
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
                                tree.scan_ref(
                                    RangeBound::Unbounded,
                                    RangeBound::Unbounded,
                                    |_, v| {
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
}
