//! Concurrent range scan stress benchmarks - Criterion version.
//!
//! Compares MassTree15 across various key patterns designed to stress different
//! aspects of concurrent ordered map implementations.
//!
//! ## Configuration
//!
//! - **Dataset size**: 1,000,000 keys
//! - **Ops per thread**: 10,000
//! - **Thread counts**: 1, 2, 3, 4, 5, 6
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
//! ## Running
//!
//! ```bash
//! cargo bench --bench range_masstree15_crit
//! cargo bench --bench range_masstree15_crit --features mimalloc
//! cargo bench --bench range_masstree15_crit -- --save-baseline main
//! cargo bench --bench range_masstree15_crit -- --baseline main
//! ```
//!
//! CSV output: target/criterion/<benchmark>/new/raw.csv

#![allow(clippy::unwrap_used)]
#![allow(clippy::pedantic)]
#![allow(clippy::indexing_slicing)]

mod bench_utils;

use bench_utils::{
    keys, keys_adversarial_splits, keys_blink_stress, keys_clustered, keys_hierarchical,
    keys_interleaved_ranges, keys_reverse, keys_sequential, keys_shared_prefix, keys_sparse,
    keys_suffix_only_differ,
};
use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use masstree::{MassTree15, RangeBound};
use std::hint::black_box;
use std::sync::Arc;
use std::sync::Barrier;
use std::thread;

// =============================================================================
// Constants
// =============================================================================

const N: usize = 1_000_000;
const OPS_PER_THREAD: usize = 10_000;
const SCAN_LIMIT: usize = 100; // Early termination for partial scans

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

// =============================================================================
// 01: SEQUENTIAL KEYS - Best case for range scans
// =============================================================================

fn bench_01_sequential_full_scan(c: &mut Criterion) {
    let mut group = c.benchmark_group("01_sequential_full_scan");

    let keys = Arc::new(keys_sequential::<8>(N));
    let tree = Arc::new(setup_masstree15(&keys));

    for threads in [1, 2, 3, 4, 5, 6] {
        let tree = Arc::clone(&tree);

        group.throughput(Throughput::Elements((threads * OPS_PER_THREAD) as u64));
        group.bench_function(BenchmarkId::new("masstree15", threads), |b| {
            let tree = Arc::clone(&tree);

            b.iter(|| {
                let barrier = Arc::new(Barrier::new(threads));
                let handles: Vec<_> = (0..threads)
                    .map(|_| {
                        let tree = Arc::clone(&tree);
                        let barrier = Arc::clone(&barrier);
                        thread::spawn(move || {
                            barrier.wait();
                            let mut total = 0usize;
                            for _ in 0..OPS_PER_THREAD {
                                let guard = tree.guard();
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
                            black_box(total);
                        })
                    })
                    .collect();

                for h in handles {
                    h.join().unwrap();
                }
            });
        });
    }
    group.finish();
}

// =============================================================================
// 02: REVERSE KEYS - Insertion stress pattern
// =============================================================================

fn bench_02_reverse_scan(c: &mut Criterion) {
    let mut group = c.benchmark_group("02_reverse_scan");

    let keys = Arc::new(keys_reverse::<8>(N));
    let tree = Arc::new(setup_masstree15(&keys));

    for threads in [1, 2, 3, 4, 5, 6] {
        let tree = Arc::clone(&tree);

        group.throughput(Throughput::Elements((threads * OPS_PER_THREAD) as u64));
        group.bench_function(BenchmarkId::new("masstree15", threads), |b| {
            let tree = Arc::clone(&tree);

            b.iter(|| {
                let barrier = Arc::new(Barrier::new(threads));
                let handles: Vec<_> = (0..threads)
                    .map(|_| {
                        let tree = Arc::clone(&tree);
                        let barrier = Arc::clone(&barrier);
                        thread::spawn(move || {
                            barrier.wait();
                            let mut total = 0usize;
                            for _ in 0..OPS_PER_THREAD {
                                let guard = tree.guard();
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
                            black_box(total);
                        })
                    })
                    .collect();

                for h in handles {
                    h.join().unwrap();
                }
            });
        });
    }
    group.finish();
}

// =============================================================================
// 03: CLUSTERED KEYS - Hot ranges with gaps
// =============================================================================

fn bench_03_clustered_scan(c: &mut Criterion) {
    let mut group = c.benchmark_group("03_clustered_scan");

    const CLUSTERS: usize = 1000;
    const KEYS_PER_CLUSTER: usize = N / CLUSTERS;
    const GAP_SIZE: u64 = 10_000;

    let keys = Arc::new(keys_clustered::<8>(CLUSTERS, KEYS_PER_CLUSTER, GAP_SIZE));
    let tree = Arc::new(setup_masstree15(&keys));

    for threads in [1, 2, 3, 4, 5, 6] {
        let tree = Arc::clone(&tree);

        group.throughput(Throughput::Elements((threads * OPS_PER_THREAD) as u64));
        group.bench_function(BenchmarkId::new("masstree15", threads), |b| {
            let tree = Arc::clone(&tree);

            b.iter(|| {
                let barrier = Arc::new(Barrier::new(threads));
                let handles: Vec<_> = (0..threads)
                    .map(|_| {
                        let tree = Arc::clone(&tree);
                        let barrier = Arc::clone(&barrier);
                        thread::spawn(move || {
                            barrier.wait();
                            let mut total = 0usize;
                            for _ in 0..OPS_PER_THREAD {
                                let guard = tree.guard();
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
                            black_box(total);
                        })
                    })
                    .collect();

                for h in handles {
                    h.join().unwrap();
                }
            });
        });
    }
    group.finish();
}

// =============================================================================
// 04: SPARSE KEYS - Cache miss stress
// =============================================================================

fn bench_04_sparse_scan(c: &mut Criterion) {
    let mut group = c.benchmark_group("04_sparse_scan");

    const SPACING: u64 = 1000;

    let keys = Arc::new(keys_sparse::<8>(N, SPACING));
    let tree = Arc::new(setup_masstree15(&keys));

    for threads in [1, 2, 3, 4, 5, 6] {
        let tree = Arc::clone(&tree);

        group.throughput(Throughput::Elements((threads * OPS_PER_THREAD) as u64));
        group.bench_function(BenchmarkId::new("masstree15", threads), |b| {
            let tree = Arc::clone(&tree);

            b.iter(|| {
                let barrier = Arc::new(Barrier::new(threads));
                let handles: Vec<_> = (0..threads)
                    .map(|_| {
                        let tree = Arc::clone(&tree);
                        let barrier = Arc::clone(&barrier);
                        thread::spawn(move || {
                            barrier.wait();
                            let mut total = 0usize;
                            for _ in 0..OPS_PER_THREAD {
                                let guard = tree.guard();
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
                            black_box(total);
                        })
                    })
                    .collect();

                for h in handles {
                    h.join().unwrap();
                }
            });
        });
    }
    group.finish();
}

// =============================================================================
// 05: SHARED PREFIX - MassTree trie advantage (16B keys)
// =============================================================================

fn bench_05_shared_prefix_scan(c: &mut Criterion) {
    let mut group = c.benchmark_group("05_shared_prefix_scan");

    const PREFIX_BUCKETS: u64 = 100; // 10k keys per prefix

    let keys = Arc::new(keys_shared_prefix::<16>(N, PREFIX_BUCKETS));
    let tree = Arc::new(setup_masstree15(&keys));

    for threads in [1, 2, 3, 4, 5, 6] {
        let tree = Arc::clone(&tree);

        group.throughput(Throughput::Elements((threads * OPS_PER_THREAD) as u64));
        group.bench_function(BenchmarkId::new("masstree15", threads), |b| {
            let tree = Arc::clone(&tree);

            b.iter(|| {
                let barrier = Arc::new(Barrier::new(threads));
                let handles: Vec<_> = (0..threads)
                    .map(|_| {
                        let tree = Arc::clone(&tree);
                        let barrier = Arc::clone(&barrier);
                        thread::spawn(move || {
                            barrier.wait();
                            let mut total = 0usize;
                            for _ in 0..OPS_PER_THREAD {
                                let guard = tree.guard();
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
                            black_box(total);
                        })
                    })
                    .collect();

                for h in handles {
                    h.join().unwrap();
                }
            });
        });
    }
    group.finish();
}

// =============================================================================
// 06: SUFFIX ONLY DIFFER - MassTree suffix mechanism (32B keys)
// =============================================================================

fn bench_06_suffix_differ_scan(c: &mut Criterion) {
    let mut group = c.benchmark_group("06_suffix_differ_scan");

    let keys = Arc::new(keys_suffix_only_differ::<32>(N));
    let tree = Arc::new(setup_masstree15(&keys));

    for threads in [1, 2, 3, 4, 5, 6] {
        let tree = Arc::clone(&tree);

        group.throughput(Throughput::Elements((threads * OPS_PER_THREAD) as u64));
        group.bench_function(BenchmarkId::new("masstree15", threads), |b| {
            let tree = Arc::clone(&tree);

            b.iter(|| {
                let barrier = Arc::new(Barrier::new(threads));
                let handles: Vec<_> = (0..threads)
                    .map(|_| {
                        let tree = Arc::clone(&tree);
                        let barrier = Arc::clone(&barrier);
                        thread::spawn(move || {
                            barrier.wait();
                            let mut total = 0usize;
                            for _ in 0..OPS_PER_THREAD {
                                let guard = tree.guard();
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
                            black_box(total);
                        })
                    })
                    .collect();

                for h in handles {
                    h.join().unwrap();
                }
            });
        });
    }
    group.finish();
}

// =============================================================================
// 07: HIERARCHICAL KEYS - Namespace:category:id pattern (32B keys)
// =============================================================================

fn bench_07_hierarchical_scan(c: &mut Criterion) {
    let mut group = c.benchmark_group("07_hierarchical_scan");

    // 100 namespaces * 100 categories * 100 items = 1M keys
    const NAMESPACES: usize = 100;
    const CATEGORIES: usize = 100;
    const ITEMS: usize = 100;

    let keys = Arc::new(keys_hierarchical::<32>(NAMESPACES, CATEGORIES, ITEMS));
    let tree = Arc::new(setup_masstree15(&keys));

    for threads in [1, 2, 3, 4, 5, 6] {
        let tree = Arc::clone(&tree);

        group.throughput(Throughput::Elements((threads * OPS_PER_THREAD) as u64));
        group.bench_function(BenchmarkId::new("masstree15", threads), |b| {
            let tree = Arc::clone(&tree);

            b.iter(|| {
                let barrier = Arc::new(Barrier::new(threads));
                let handles: Vec<_> = (0..threads)
                    .map(|_| {
                        let tree = Arc::clone(&tree);
                        let barrier = Arc::clone(&barrier);
                        thread::spawn(move || {
                            barrier.wait();
                            let mut total = 0usize;
                            for _ in 0..OPS_PER_THREAD {
                                let guard = tree.guard();
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
                            black_box(total);
                        })
                    })
                    .collect();

                for h in handles {
                    h.join().unwrap();
                }
            });
        });
    }
    group.finish();
}

// =============================================================================
// 08: ADVERSARIAL SPLITS - Split propagation stress
// =============================================================================

fn bench_08_adversarial_splits_scan(c: &mut Criterion) {
    let mut group = c.benchmark_group("08_adversarial_splits_scan");

    let keys = Arc::new(keys_adversarial_splits::<8>(N));
    let tree = Arc::new(setup_masstree15(&keys));

    for threads in [1, 2, 3, 4, 5, 6] {
        let tree = Arc::clone(&tree);

        group.throughput(Throughput::Elements((threads * OPS_PER_THREAD) as u64));
        group.bench_function(BenchmarkId::new("masstree15", threads), |b| {
            let tree = Arc::clone(&tree);

            b.iter(|| {
                let barrier = Arc::new(Barrier::new(threads));
                let handles: Vec<_> = (0..threads)
                    .map(|_| {
                        let tree = Arc::clone(&tree);
                        let barrier = Arc::clone(&barrier);
                        thread::spawn(move || {
                            barrier.wait();
                            let mut total = 0usize;
                            for _ in 0..OPS_PER_THREAD {
                                let guard = tree.guard();
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
                            black_box(total);
                        })
                    })
                    .collect();

                for h in handles {
                    h.join().unwrap();
                }
            });
        });
    }
    group.finish();
}

// =============================================================================
// 09: INTERLEAVED RANGES - Cache thrashing stress
// =============================================================================

fn bench_09_interleaved_scan(c: &mut Criterion) {
    let mut group = c.benchmark_group("09_interleaved_scan");

    const HOT_RANGES: usize = 100;
    const KEYS_PER_RANGE: usize = N / HOT_RANGES;
    const COLD_GAP: u64 = 100_000;

    let keys = Arc::new(keys_interleaved_ranges::<8>(
        HOT_RANGES,
        KEYS_PER_RANGE,
        COLD_GAP,
    ));
    let tree = Arc::new(setup_masstree15(&keys));

    for threads in [1, 2, 3, 4, 5, 6] {
        let tree = Arc::clone(&tree);

        group.throughput(Throughput::Elements((threads * OPS_PER_THREAD) as u64));
        group.bench_function(BenchmarkId::new("masstree15", threads), |b| {
            let tree = Arc::clone(&tree);

            b.iter(|| {
                let barrier = Arc::new(Barrier::new(threads));
                let handles: Vec<_> = (0..threads)
                    .map(|_| {
                        let tree = Arc::clone(&tree);
                        let barrier = Arc::clone(&barrier);
                        thread::spawn(move || {
                            barrier.wait();
                            let mut total = 0usize;
                            for _ in 0..OPS_PER_THREAD {
                                let guard = tree.guard();
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
                            black_box(total);
                        })
                    })
                    .collect();

                for h in handles {
                    h.join().unwrap();
                }
            });
        });
    }
    group.finish();
}

// =============================================================================
// 10: B-LINK STRESS - Fragmented scan stress
// =============================================================================

fn bench_10_blink_stress_scan(c: &mut Criterion) {
    let mut group = c.benchmark_group("10_blink_stress_scan");

    let keys = Arc::new(keys_blink_stress::<8>(N));
    let tree = Arc::new(setup_masstree15(&keys));

    for threads in [1, 2, 3, 4, 5, 6] {
        let tree = Arc::clone(&tree);

        group.throughput(Throughput::Elements((threads * OPS_PER_THREAD) as u64));
        group.bench_function(BenchmarkId::new("masstree15", threads), |b| {
            let tree = Arc::clone(&tree);

            b.iter(|| {
                let barrier = Arc::new(Barrier::new(threads));
                let handles: Vec<_> = (0..threads)
                    .map(|_| {
                        let tree = Arc::clone(&tree);
                        let barrier = Arc::clone(&barrier);
                        thread::spawn(move || {
                            barrier.wait();
                            let mut total = 0usize;
                            for _ in 0..OPS_PER_THREAD {
                                let guard = tree.guard();
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
                            black_box(total);
                        })
                    })
                    .collect();

                for h in handles {
                    h.join().unwrap();
                }
            });
        });
    }
    group.finish();
}

// =============================================================================
// 11: RANDOM KEYS - Baseline comparison
// =============================================================================

fn bench_11_random_keys_scan(c: &mut Criterion) {
    let mut group = c.benchmark_group("11_random_keys_scan");

    let keys = Arc::new(keys::<8>(N));
    let tree = Arc::new(setup_masstree15(&keys));

    for threads in [1, 2, 3, 4, 5, 6] {
        let tree = Arc::clone(&tree);

        group.throughput(Throughput::Elements((threads * OPS_PER_THREAD) as u64));
        group.bench_function(BenchmarkId::new("masstree15", threads), |b| {
            let tree = Arc::clone(&tree);

            b.iter(|| {
                let barrier = Arc::new(Barrier::new(threads));
                let handles: Vec<_> = (0..threads)
                    .map(|_| {
                        let tree = Arc::clone(&tree);
                        let barrier = Arc::clone(&barrier);
                        thread::spawn(move || {
                            barrier.wait();
                            let mut total = 0usize;
                            for _ in 0..OPS_PER_THREAD {
                                let guard = tree.guard();
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
                            black_box(total);
                        })
                    })
                    .collect();

                for h in handles {
                    h.join().unwrap();
                }
            });
        });
    }
    group.finish();
}

// =============================================================================
// 12: LONG KEYS (64B) - Multi-layer traversal
// =============================================================================

fn bench_12_long_keys_64b_scan(c: &mut Criterion) {
    let mut group = c.benchmark_group("12_long_keys_64b_scan");

    let keys = Arc::new(keys::<64>(N));
    let tree = Arc::new(setup_masstree15(&keys));

    for threads in [1, 2, 3, 4, 5, 6] {
        let tree = Arc::clone(&tree);

        group.throughput(Throughput::Elements((threads * OPS_PER_THREAD) as u64));
        group.bench_function(BenchmarkId::new("masstree15", threads), |b| {
            let tree = Arc::clone(&tree);

            b.iter(|| {
                let barrier = Arc::new(Barrier::new(threads));
                let handles: Vec<_> = (0..threads)
                    .map(|_| {
                        let tree = Arc::clone(&tree);
                        let barrier = Arc::clone(&barrier);
                        thread::spawn(move || {
                            barrier.wait();
                            let mut total = 0usize;
                            for _ in 0..OPS_PER_THREAD {
                                let guard = tree.guard();
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
                            black_box(total);
                        })
                    })
                    .collect();

                for h in handles {
                    h.join().unwrap();
                }
            });
        });
    }
    group.finish();
}

// =============================================================================
// 13: SCAN WHILE INSERT - Mixed workload
// =============================================================================

fn bench_13_scan_while_insert(c: &mut Criterion) {
    let mut group = c.benchmark_group("13_scan_while_insert");
    group.sample_size(20);

    const INITIAL_N: usize = 900_000;
    const INSERT_N: usize = 100_000;
    const WRITERS: usize = 2;

    let initial_keys = keys::<8>(INITIAL_N);
    let insert_keys: Vec<_> = keys::<8>(INITIAL_N + INSERT_N)[INITIAL_N..].to_vec();

    // Minimum 3 threads: 2 writers + 1 reader
    for threads in [3, 4, 5, 6] {
        let initial_keys = initial_keys.clone();
        let insert_keys = insert_keys.clone();
        let readers = threads - WRITERS;

        group.bench_function(BenchmarkId::new("masstree15", threads), |b| {
            let initial_keys = initial_keys.clone();
            let insert_keys = insert_keys.clone();

            b.iter_with_setup(
                || {
                    let tree = Arc::new(setup_masstree15(&initial_keys));
                    let new_keys = Arc::new(insert_keys.clone());
                    (tree, new_keys)
                },
                |(tree, new_keys)| {
                    let barrier = Arc::new(Barrier::new(threads));

                    // Writer threads
                    let writer_handles: Vec<_> = (0..WRITERS)
                        .map(|w| {
                            let tree = Arc::clone(&tree);
                            let barrier = Arc::clone(&barrier);
                            let chunk_size = INSERT_N / WRITERS;
                            let chunk: Vec<_> =
                                new_keys[w * chunk_size..(w + 1) * chunk_size].to_vec();
                            thread::spawn(move || {
                                barrier.wait();
                                let guard = tree.guard();
                                for (i, key) in chunk.iter().enumerate() {
                                    let _ =
                                        tree.insert_with_guard(key, (INITIAL_N + i) as u64, &guard);
                                }
                            })
                        })
                        .collect();

                    // Reader threads
                    let reader_handles: Vec<_> = (0..readers)
                        .map(|_| {
                            let tree = Arc::clone(&tree);
                            let barrier = Arc::clone(&barrier);
                            thread::spawn(move || {
                                barrier.wait();
                                let mut total = 0usize;
                                for _ in 0..OPS_PER_THREAD {
                                    let guard = tree.guard();
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
                },
            );
        });
    }
    group.finish();
}

// =============================================================================
// 14: PREFIX SCAN - MassTree-specific optimization
// =============================================================================

fn bench_14_prefix_scan(c: &mut Criterion) {
    let mut group = c.benchmark_group("14_prefix_scan");
    group.sample_size(10);

    const PREFIX_BUCKETS: u64 = 100;
    // Reduced ops for prefix scan - each scan touches 10K entries (1M/100 buckets)
    const PREFIX_OPS: usize = 100;

    let keys = Arc::new(keys_shared_prefix::<16>(N, PREFIX_BUCKETS));
    let tree = Arc::new(setup_masstree15(&keys));

    for threads in [1, 2, 3, 4, 5, 6] {
        let tree = Arc::clone(&tree);

        group.throughput(Throughput::Elements((threads * PREFIX_OPS) as u64));
        group.bench_function(BenchmarkId::new("masstree15", threads), |b| {
            let tree = Arc::clone(&tree);

            b.iter(|| {
                let barrier = Arc::new(Barrier::new(threads));
                let handles: Vec<_> = (0..threads)
                    .map(|t| {
                        let tree = Arc::clone(&tree);
                        let barrier = Arc::clone(&barrier);
                        // Each thread scans a different prefix
                        let thread_prefix = ((t as u64) % PREFIX_BUCKETS).to_be_bytes();
                        thread::spawn(move || {
                            barrier.wait();
                            let mut total = 0usize;
                            for _ in 0..PREFIX_OPS {
                                let guard = tree.guard();
                                total += tree.scan_prefix(&thread_prefix, |_, _| true, &guard);
                            }
                            black_box(total);
                        })
                    })
                    .collect();

                for h in handles {
                    h.join().unwrap();
                }
            });
        });
    }
    group.finish();
}

// =============================================================================
// 15: FULL SCAN AGGREGATE - Sum all values
// =============================================================================

fn bench_15_full_scan_aggregate(c: &mut Criterion) {
    let mut group = c.benchmark_group("15_full_scan_aggregate");
    group.sample_size(50);

    const SCAN_N: usize = 50_000; // Smaller dataset for full scans
    const FULL_SCAN_OPS: usize = 100; // Reduced iterations for full scans

    let keys = Arc::new(keys_sequential::<8>(SCAN_N));
    let tree = Arc::new(setup_masstree15(&keys));

    for threads in [1, 2, 3, 4, 5, 6] {
        let tree = Arc::clone(&tree);

        group.throughput(Throughput::Elements((threads * FULL_SCAN_OPS) as u64));
        group.bench_function(BenchmarkId::new("masstree15", threads), |b| {
            let tree = Arc::clone(&tree);

            b.iter(|| {
                let barrier = Arc::new(Barrier::new(threads));
                let handles: Vec<_> = (0..threads)
                    .map(|_| {
                        let tree = Arc::clone(&tree);
                        let barrier = Arc::clone(&barrier);
                        thread::spawn(move || {
                            barrier.wait();
                            let mut grand_total = 0u64;
                            for _ in 0..FULL_SCAN_OPS {
                                let guard = tree.guard();
                                let mut sum = 0u64;
                                tree.scan_ref(
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
                            black_box(grand_total);
                        })
                    })
                    .collect();

                for h in handles {
                    h.join().unwrap();
                }
            });
        });
    }
    group.finish();
}

// =============================================================================
// 16: INSERT-HEAVY - 90% writes, 10% reads (high write contention)
// =============================================================================

fn bench_16_insert_heavy(c: &mut Criterion) {
    let mut group = c.benchmark_group("16_insert_heavy");

    const INITIAL_N: usize = 100_000;
    const OPS: usize = 10_000;
    const WRITE_RATIO: usize = 90; // 90% writes

    // Pre-generate insert keys (unique per thread)
    let initial_keys: Vec<[u8; 8]> = keys_sequential::<8>(INITIAL_N);

    for threads in [1, 2, 3, 4, 5, 6] {
        let initial_keys = initial_keys.clone();
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

        group.throughput(Throughput::Elements((threads * OPS) as u64));
        group.bench_function(BenchmarkId::new("masstree15", threads), |b| {
            let initial_keys = initial_keys.clone();
            let insert_keys = insert_keys.clone();

            b.iter_with_setup(
                || {
                    let tree = Arc::new(setup_masstree15(&initial_keys));
                    (tree, insert_keys.clone())
                },
                |(tree, thread_keys)| {
                    let barrier = Arc::new(Barrier::new(threads));
                    let handles: Vec<_> = (0..threads)
                        .map(|t| {
                            let tree = Arc::clone(&tree);
                            let barrier = Arc::clone(&barrier);
                            let keys = thread_keys[t].clone();
                            let read_keys = initial_keys.clone();
                            thread::spawn(move || {
                                barrier.wait();
                                let guard = tree.guard();
                                let mut rng_state = t as u64;
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
                            })
                        })
                        .collect();

                    for h in handles {
                        h.join().unwrap();
                    }
                },
            );
        });
    }
    group.finish();
}

// =============================================================================
// 17: HOT-SPOT - All threads target narrow key range (localized contention)
// =============================================================================

fn bench_17_hot_spot(c: &mut Criterion) {
    let mut group = c.benchmark_group("17_hot_spot");
    group.sample_size(20);

    const TOTAL_N: usize = 1_000_000;
    const HOT_RANGE: usize = 10; // Only 10 keys are "hot" - extreme contention
    const OPS: usize = 10_000;

    let all_keys: Vec<[u8; 8]> = keys_sequential::<8>(TOTAL_N);
    let hot_keys: Vec<[u8; 8]> = all_keys[..HOT_RANGE].to_vec();

    for threads in [1, 2, 3, 4, 5, 6] {
        let all_keys = all_keys.clone();
        let hot_keys = hot_keys.clone();

        group.throughput(Throughput::Elements((threads * OPS) as u64));
        group.bench_function(BenchmarkId::new("masstree15", threads), |b| {
            let all_keys = all_keys.clone();
            let hot_keys = hot_keys.clone();

            b.iter_with_setup(
                || Arc::new(setup_masstree15(&all_keys)),
                |tree| {
                    let barrier = Arc::new(Barrier::new(threads));
                    let handles: Vec<_> = (0..threads)
                        .map(|t| {
                            let tree = Arc::clone(&tree);
                            let barrier = Arc::clone(&barrier);
                            let hot = hot_keys.clone();

                            thread::spawn(move || {
                                barrier.wait();
                                let guard = tree.guard();
                                let mut rng_state = t as u64;

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
                            })
                        })
                        .collect();

                    for h in handles {
                        h.join().unwrap();
                    }
                },
            );
        });
    }
    group.finish();
}

// =============================================================================
// 18: SPLIT-INDUCING SCAN - Sequential inserts cause splits while readers scan
// =============================================================================

fn bench_18_split_inducing_scan(c: &mut Criterion) {
    let mut group = c.benchmark_group("18_split_inducing_scan");

    const INITIAL_N: usize = 100_000;
    const INSERT_N: usize = 50_000;
    const WRITERS: usize = 2;

    let initial_keys = keys_sequential::<8>(INITIAL_N);
    // Sequential keys after initial range - will cause splits
    let insert_keys: Vec<[u8; 8]> = (INITIAL_N..INITIAL_N + INSERT_N)
        .map(|i| (i as u64).to_be_bytes())
        .collect();

    // Minimum 3 threads: 2 writers + 1 reader
    for threads in [3, 4, 5, 6] {
        let initial_keys = initial_keys.clone();
        let insert_keys = insert_keys.clone();
        let readers = threads - WRITERS;

        group.bench_function(BenchmarkId::new("masstree15", threads), |b| {
            let initial_keys = initial_keys.clone();
            let insert_keys = insert_keys.clone();

            b.iter_with_setup(
                || {
                    let tree = Arc::new(setup_masstree15(&initial_keys));
                    (tree, insert_keys.clone())
                },
                |(tree, new_keys)| {
                    let barrier = Arc::new(Barrier::new(threads));

                    // Writer threads - sequential inserts cause leaf splits
                    let writer_handles: Vec<_> = (0..WRITERS)
                        .map(|w| {
                            let tree = Arc::clone(&tree);
                            let barrier = Arc::clone(&barrier);
                            let chunk_size = INSERT_N / WRITERS;
                            let chunk: Vec<_> =
                                new_keys[w * chunk_size..(w + 1) * chunk_size].to_vec();
                            thread::spawn(move || {
                                barrier.wait();
                                let guard = tree.guard();
                                for (i, key) in chunk.iter().enumerate() {
                                    let _ = tree.insert_with_guard(key, i as u64, &guard);
                                }
                            })
                        })
                        .collect();

                    // Reader threads - scan during structural modifications
                    let reader_handles: Vec<_> = (0..readers)
                        .map(|_| {
                            let tree = Arc::clone(&tree);
                            let barrier = Arc::clone(&barrier);
                            thread::spawn(move || {
                                barrier.wait();
                                let mut total = 0usize;
                                for _ in 0..OPS_PER_THREAD {
                                    let guard = tree.guard();
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
                },
            );
        });
    }
    group.finish();
}

// =============================================================================
// CRITERION GROUPS AND MAIN
// =============================================================================

criterion_group!(
    benches,
    bench_01_sequential_full_scan,
    bench_02_reverse_scan,
    bench_03_clustered_scan,
    bench_04_sparse_scan,
    bench_05_shared_prefix_scan,
    bench_06_suffix_differ_scan,
    bench_07_hierarchical_scan,
    bench_08_adversarial_splits_scan,
    bench_09_interleaved_scan,
    bench_10_blink_stress_scan,
    bench_11_random_keys_scan,
    bench_12_long_keys_64b_scan,
    bench_13_scan_while_insert,
    bench_14_prefix_scan,
    bench_15_full_scan_aggregate,
    bench_16_insert_heavy,
    bench_17_hot_spot,
    bench_18_split_inducing_scan,
);

criterion_main!(benches);
