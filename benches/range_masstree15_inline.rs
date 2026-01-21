//! Concurrent range scan stress benchmarks for MassTree15Inline.
//!
//! Compares MassTree15Inline, scc::TreeIndex, and crossbeam_skiplist::SkipMap across
//! various key patterns designed to stress different aspects of concurrent
//! ordered map implementations.
//!
//! ## Configuration
//!
//! - **Dataset size**: 500,000 keys
//! - **Ops per thread**: 5,000
//! - **Thread counts**: 1-6
//!
//! ## Understanding Items/sec (IMPORTANT)
//!
//! Divan reports **items/sec** based on `ItemsCount`. The meaning varies by group:
//!
//! | Groups | Unit | What It Measures |
//! |--------|------|------------------|
//! | 01–12, 19–24 | **50-row range queries/sec** | Each "item" = one scan visiting up to SCAN_LIMIT entries |
//! | 14 | **prefix queries/sec** | Each "item" = one `scan_prefix()` call |
//! | 15 | **keys/sec** | Each "item" = one key visited during full scan |
//! | 16–17 | **point ops/sec** | Each "item" = one read or write operation |
//! | 25+ | **50-row range queries/sec** | Random-start variants (more realistic) |
//!
//! **To convert short-scan groups to keys/sec**: multiply by `SCAN_LIMIT` (50).
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
//! ### Methodology Variants (groups 25+)
//! - Random start bounds (avoid always scanning first leaves)
//! - Iterator baseline (compare batch API vs iterator)
//!
//! ## API Notes
//!
//! - **MassTree15Inline**: Uses `for_each_intra_leaf_batch` — optimized batch scan
//! - **TreeIndex**: Uses `.iter().take()` — lazy iterator
//! - **SkipMap**: Uses `.iter().take()` — lazy iterator
//!
//! ## Running
//!
//! ```bash
//! cargo bench --bench range_masstree15_inline
//! cargo bench --bench range_masstree15_inline --features mimalloc
//!
//! # Specific pattern
//! cargo bench --bench range_masstree15_inline -- sequential
//! cargo bench --bench range_masstree15_inline -- hierarchical
//!
//! # Methodology variants (random starts, iterator baseline)
//! cargo bench --bench range_masstree15_inline -- 25_random
//! cargo bench --bench range_masstree15_inline -- 26_iterator
//! ```

#![expect(clippy::unwrap_used)]
#![expect(clippy::pedantic)]
#![expect(clippy::indexing_slicing)]

mod bench_utils;

use bench_utils::{
    keys, keys_adversarial_splits, keys_blink_stress, keys_clustered, keys_hierarchical,
    keys_interleaved_ranges, keys_reverse, keys_sequential, keys_shared_prefix, keys_sparse,
    keys_suffix_only_differ, random_start_indices,
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

// =============================================================================
// 01: SEQUENTIAL KEYS - Best case for range scans
// =============================================================================

#[divan::bench_group(name = "01_sequential_full_scan", sample_count = 100)]
mod sequential_full_scan {
    use super::*;

    #[divan::bench(args = [1, 2, 3, 4, 5, 6])]
    fn masstree15_inline(bencher: Bencher, threads: usize) {
        let keys = Arc::new(keys_sequential::<8>(N));
        let tree = Arc::new(setup_masstree15_inline(&keys));

        bencher
            .counter(divan::counter::ItemsCount::new(threads * OPS_PER_THREAD))
            .bench_local(|| {
                let barrier = Arc::new(Barrier::new(threads));
                let handles: Vec<_> = (0..threads)
                    .map(|_| {
                        let tree = Arc::clone(&tree);
                        let barrier = Arc::clone(&barrier);
                        thread::spawn(move || {
                            barrier.wait();
                            let guard = tree.guard();
                            let mut total = 0usize;
                            for _ in 0..OPS_PER_THREAD {
                                let mut count = 0usize;
                                tree.scan_values(
                                    RangeBound::Unbounded,
                                    RangeBound::Unbounded,
                                    |_| {
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
    }

    #[divan::bench(args = [1, 2, 3, 4, 5, 6])]
    fn tree_index(bencher: Bencher, threads: usize) {
        let keys = Arc::new(keys_sequential::<8>(N));
        let tree = Arc::new(setup_tree_index(&keys));

        bencher
            .counter(divan::counter::ItemsCount::new(threads * OPS_PER_THREAD))
            .bench_local(|| {
                let barrier = Arc::new(Barrier::new(threads));
                let handles: Vec<_> = (0..threads)
                    .map(|_| {
                        let tree = Arc::clone(&tree);
                        let barrier = Arc::clone(&barrier);
                        thread::spawn(move || {
                            barrier.wait();
                            let guard = sdd::Guard::new();
                            let mut total = 0usize;
                            for _ in 0..OPS_PER_THREAD {
                                let count = tree.iter(&guard).take(SCAN_LIMIT).count();
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
    }
}

// =============================================================================
// 02: REVERSE KEYS - Insertion stress pattern
// =============================================================================

#[divan::bench_group(name = "02_reverse_scan", sample_count = 100)]
mod reverse_scan {
    use super::*;

    #[divan::bench(args = [1, 2, 3, 4, 5, 6])]
    fn masstree15_inline(bencher: Bencher, threads: usize) {
        let keys = Arc::new(keys_reverse::<8>(N));
        let tree = Arc::new(setup_masstree15_inline(&keys));

        bencher
            .counter(divan::counter::ItemsCount::new(threads * OPS_PER_THREAD))
            .bench_local(|| {
                let barrier = Arc::new(Barrier::new(threads));
                let handles: Vec<_> = (0..threads)
                    .map(|_| {
                        let tree = Arc::clone(&tree);
                        let barrier = Arc::clone(&barrier);
                        thread::spawn(move || {
                            barrier.wait();
                            let guard = tree.guard();
                            let mut total = 0usize;
                            for _ in 0..OPS_PER_THREAD {
                                let mut count = 0usize;
                                tree.iter(&guard).for_each_intra_leaf_batch(|_, _| {
                                    count += 1;
                                    count < SCAN_LIMIT
                                });
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
    }

    #[divan::bench(args = [1, 2, 3, 4, 5, 6])]
    fn tree_index(bencher: Bencher, threads: usize) {
        let keys = Arc::new(keys_reverse::<8>(N));
        let tree = Arc::new(setup_tree_index(&keys));

        bencher
            .counter(divan::counter::ItemsCount::new(threads * OPS_PER_THREAD))
            .bench_local(|| {
                let barrier = Arc::new(Barrier::new(threads));
                let handles: Vec<_> = (0..threads)
                    .map(|_| {
                        let tree = Arc::clone(&tree);
                        let barrier = Arc::clone(&barrier);
                        thread::spawn(move || {
                            barrier.wait();
                            let guard = sdd::Guard::new();
                            let mut total = 0usize;
                            for _ in 0..OPS_PER_THREAD {
                                let count = tree.iter(&guard).take(SCAN_LIMIT).count();
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

    #[divan::bench(args = [1, 2, 3, 4, 5, 6])]
    fn masstree15_inline(bencher: Bencher, threads: usize) {
        let keys = Arc::new(keys_clustered::<8>(CLUSTERS, KEYS_PER_CLUSTER, GAP_SIZE));
        let tree = Arc::new(setup_masstree15_inline(&keys));

        bencher
            .counter(divan::counter::ItemsCount::new(threads * OPS_PER_THREAD))
            .bench_local(|| {
                let barrier = Arc::new(Barrier::new(threads));
                let handles: Vec<_> = (0..threads)
                    .map(|_| {
                        let tree = Arc::clone(&tree);
                        let barrier = Arc::clone(&barrier);
                        thread::spawn(move || {
                            barrier.wait();
                            let guard = tree.guard();
                            let mut total = 0usize;
                            for _ in 0..OPS_PER_THREAD {
                                let mut count = 0usize;
                                tree.scan_values(
                                    RangeBound::Unbounded,
                                    RangeBound::Unbounded,
                                    |_| {
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
    }

    #[divan::bench(args = [1, 2, 3, 4, 5, 6])]
    fn tree_index(bencher: Bencher, threads: usize) {
        let keys = Arc::new(keys_clustered::<8>(CLUSTERS, KEYS_PER_CLUSTER, GAP_SIZE));
        let tree = Arc::new(setup_tree_index(&keys));

        bencher
            .counter(divan::counter::ItemsCount::new(threads * OPS_PER_THREAD))
            .bench_local(|| {
                let barrier = Arc::new(Barrier::new(threads));
                let handles: Vec<_> = (0..threads)
                    .map(|_| {
                        let tree = Arc::clone(&tree);
                        let barrier = Arc::clone(&barrier);
                        thread::spawn(move || {
                            barrier.wait();
                            let guard = sdd::Guard::new();
                            let mut total = 0usize;
                            for _ in 0..OPS_PER_THREAD {
                                let count = tree.iter(&guard).take(SCAN_LIMIT).count();
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
    }
}

// =============================================================================
// 04: SPARSE KEYS - Cache miss stress
// =============================================================================

#[divan::bench_group(name = "04_sparse_scan", sample_count = 100)]
mod sparse_scan {
    use super::*;

    const SPACING: u64 = 1000;

    #[divan::bench(args = [1, 2, 3, 4, 5, 6])]
    fn masstree15_inline(bencher: Bencher, threads: usize) {
        let keys = Arc::new(keys_sparse::<8>(N, SPACING));
        let tree = Arc::new(setup_masstree15_inline(&keys));

        bencher
            .counter(divan::counter::ItemsCount::new(threads * OPS_PER_THREAD))
            .bench_local(|| {
                let barrier = Arc::new(Barrier::new(threads));
                let handles: Vec<_> = (0..threads)
                    .map(|_| {
                        let tree = Arc::clone(&tree);
                        let barrier = Arc::clone(&barrier);
                        thread::spawn(move || {
                            barrier.wait();
                            let guard = tree.guard();
                            let mut total = 0usize;
                            for _ in 0..OPS_PER_THREAD {
                                let mut count = 0usize;
                                tree.scan_values(
                                    RangeBound::Unbounded,
                                    RangeBound::Unbounded,
                                    |_| {
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
    }

    #[divan::bench(args = [1, 2, 3, 4, 5, 6])]
    fn tree_index(bencher: Bencher, threads: usize) {
        let keys = Arc::new(keys_sparse::<8>(N, SPACING));
        let tree = Arc::new(setup_tree_index(&keys));

        bencher
            .counter(divan::counter::ItemsCount::new(threads * OPS_PER_THREAD))
            .bench_local(|| {
                let barrier = Arc::new(Barrier::new(threads));
                let handles: Vec<_> = (0..threads)
                    .map(|_| {
                        let tree = Arc::clone(&tree);
                        let barrier = Arc::clone(&barrier);
                        thread::spawn(move || {
                            barrier.wait();
                            let guard = sdd::Guard::new();
                            let mut total = 0usize;
                            for _ in 0..OPS_PER_THREAD {
                                let count = tree.iter(&guard).take(SCAN_LIMIT).count();
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
    }
}

// =============================================================================
// 05: SHARED PREFIX - MassTree trie advantage (16B keys)
// =============================================================================

#[divan::bench_group(name = "05_shared_prefix_scan", sample_count = 100)]
mod shared_prefix_scan {
    use super::*;

    const PREFIX_BUCKETS: u64 = 100; // 10k keys per prefix

    #[divan::bench(args = [1, 2, 3, 4, 5, 6])]
    fn masstree15_inline(bencher: Bencher, threads: usize) {
        let keys = Arc::new(keys_shared_prefix::<16>(N, PREFIX_BUCKETS));
        let tree = Arc::new(setup_masstree15_inline(&keys));

        bencher
            .counter(divan::counter::ItemsCount::new(threads * OPS_PER_THREAD))
            .bench_local(|| {
                let barrier = Arc::new(Barrier::new(threads));
                let handles: Vec<_> = (0..threads)
                    .map(|_| {
                        let tree = Arc::clone(&tree);
                        let barrier = Arc::clone(&barrier);
                        thread::spawn(move || {
                            barrier.wait();
                            let guard = tree.guard();
                            let mut total = 0usize;
                            for _ in 0..OPS_PER_THREAD {
                                let mut count = 0usize;
                                tree.scan_values(
                                    RangeBound::Unbounded,
                                    RangeBound::Unbounded,
                                    |_| {
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
    }

    #[divan::bench(args = [1, 2, 3, 4, 5, 6])]
    fn tree_index(bencher: Bencher, threads: usize) {
        let keys = Arc::new(keys_shared_prefix::<16>(N, PREFIX_BUCKETS));
        let tree = Arc::new(setup_tree_index(&keys));

        bencher
            .counter(divan::counter::ItemsCount::new(threads * OPS_PER_THREAD))
            .bench_local(|| {
                let barrier = Arc::new(Barrier::new(threads));
                let handles: Vec<_> = (0..threads)
                    .map(|_| {
                        let tree = Arc::clone(&tree);
                        let barrier = Arc::clone(&barrier);
                        thread::spawn(move || {
                            barrier.wait();
                            let guard = sdd::Guard::new();
                            let mut total = 0usize;
                            for _ in 0..OPS_PER_THREAD {
                                let count = tree.iter(&guard).take(SCAN_LIMIT).count();
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
    }
}

// =============================================================================
// 06: SUFFIX ONLY DIFFER - MassTree suffix mechanism (32B keys)
// =============================================================================

#[divan::bench_group(name = "06_suffix_differ_scan", sample_count = 100)]
mod suffix_differ_scan {
    use super::*;

    #[divan::bench(args = [1, 2, 3, 4, 5, 6])]
    fn masstree15_inline(bencher: Bencher, threads: usize) {
        let keys = Arc::new(keys_suffix_only_differ::<32>(N));
        let tree = Arc::new(setup_masstree15_inline(&keys));

        bencher
            .counter(divan::counter::ItemsCount::new(threads * OPS_PER_THREAD))
            .bench_local(|| {
                let barrier = Arc::new(Barrier::new(threads));
                let handles: Vec<_> = (0..threads)
                    .map(|_| {
                        let tree = Arc::clone(&tree);
                        let barrier = Arc::clone(&barrier);
                        thread::spawn(move || {
                            barrier.wait();
                            let guard = tree.guard();
                            let mut total = 0usize;
                            for _ in 0..OPS_PER_THREAD {
                                let mut count = 0usize;
                                tree.scan_values(
                                    RangeBound::Unbounded,
                                    RangeBound::Unbounded,
                                    |_| {
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
    }

    #[divan::bench(args = [1, 2, 3, 4, 5, 6])]
    fn tree_index(bencher: Bencher, threads: usize) {
        let keys = Arc::new(keys_suffix_only_differ::<32>(N));
        let tree = Arc::new(setup_tree_index(&keys));

        bencher
            .counter(divan::counter::ItemsCount::new(threads * OPS_PER_THREAD))
            .bench_local(|| {
                let barrier = Arc::new(Barrier::new(threads));
                let handles: Vec<_> = (0..threads)
                    .map(|_| {
                        let tree = Arc::clone(&tree);
                        let barrier = Arc::clone(&barrier);
                        thread::spawn(move || {
                            barrier.wait();
                            let guard = sdd::Guard::new();
                            let mut total = 0usize;
                            for _ in 0..OPS_PER_THREAD {
                                let count = tree.iter(&guard).take(SCAN_LIMIT).count();
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

    #[divan::bench(args = [1, 2, 3, 4, 5, 6])]
    fn masstree15_inline(bencher: Bencher, threads: usize) {
        let keys = Arc::new(keys_hierarchical::<32>(NAMESPACES, CATEGORIES, ITEMS));
        let tree = Arc::new(setup_masstree15_inline(&keys));

        bencher
            .counter(divan::counter::ItemsCount::new(threads * OPS_PER_THREAD))
            .bench_local(|| {
                let barrier = Arc::new(Barrier::new(threads));
                let handles: Vec<_> = (0..threads)
                    .map(|_| {
                        let tree = Arc::clone(&tree);
                        let barrier = Arc::clone(&barrier);
                        thread::spawn(move || {
                            barrier.wait();
                            let guard = tree.guard();
                            let mut total = 0usize;
                            for _ in 0..OPS_PER_THREAD {
                                let mut count = 0usize;
                                tree.scan_values(
                                    RangeBound::Unbounded,
                                    RangeBound::Unbounded,
                                    |_| {
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
    }

    #[divan::bench(args = [1, 2, 3, 4, 5, 6])]
    fn tree_index(bencher: Bencher, threads: usize) {
        let keys = Arc::new(keys_hierarchical::<32>(NAMESPACES, CATEGORIES, ITEMS));
        let tree = Arc::new(setup_tree_index(&keys));

        bencher
            .counter(divan::counter::ItemsCount::new(threads * OPS_PER_THREAD))
            .bench_local(|| {
                let barrier = Arc::new(Barrier::new(threads));
                let handles: Vec<_> = (0..threads)
                    .map(|_| {
                        let tree = Arc::clone(&tree);
                        let barrier = Arc::clone(&barrier);
                        thread::spawn(move || {
                            barrier.wait();
                            let guard = sdd::Guard::new();
                            let mut total = 0usize;
                            for _ in 0..OPS_PER_THREAD {
                                let count = tree.iter(&guard).take(SCAN_LIMIT).count();
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
    }
}

// =============================================================================
// 08: ADVERSARIAL SPLITS - Split propagation stress
// =============================================================================

#[divan::bench_group(name = "08_adversarial_splits_scan", sample_count = 100)]
mod adversarial_splits_scan {
    use super::*;

    #[divan::bench(args = [1, 2, 3, 4, 5, 6])]
    fn masstree15_inline(bencher: Bencher, threads: usize) {
        let keys = Arc::new(keys_adversarial_splits::<8>(N));
        let tree = Arc::new(setup_masstree15_inline(&keys));

        bencher
            .counter(divan::counter::ItemsCount::new(threads * OPS_PER_THREAD))
            .bench_local(|| {
                let barrier = Arc::new(Barrier::new(threads));
                let handles: Vec<_> = (0..threads)
                    .map(|_| {
                        let tree = Arc::clone(&tree);
                        let barrier = Arc::clone(&barrier);
                        thread::spawn(move || {
                            barrier.wait();
                            let guard = tree.guard();
                            let mut total = 0usize;
                            for _ in 0..OPS_PER_THREAD {
                                let mut count = 0usize;
                                tree.scan_values(
                                    RangeBound::Unbounded,
                                    RangeBound::Unbounded,
                                    |_| {
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
    }

    #[divan::bench(args = [1, 2, 3, 4, 5, 6])]
    fn tree_index(bencher: Bencher, threads: usize) {
        let keys = Arc::new(keys_adversarial_splits::<8>(N));
        let tree = Arc::new(setup_tree_index(&keys));

        bencher
            .counter(divan::counter::ItemsCount::new(threads * OPS_PER_THREAD))
            .bench_local(|| {
                let barrier = Arc::new(Barrier::new(threads));
                let handles: Vec<_> = (0..threads)
                    .map(|_| {
                        let tree = Arc::clone(&tree);
                        let barrier = Arc::clone(&barrier);
                        thread::spawn(move || {
                            barrier.wait();
                            let guard = sdd::Guard::new();
                            let mut total = 0usize;
                            for _ in 0..OPS_PER_THREAD {
                                let count = tree.iter(&guard).take(SCAN_LIMIT).count();
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

    #[divan::bench(args = [1, 2, 3, 4, 5, 6])]
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
                let barrier = Arc::new(Barrier::new(threads));
                let handles: Vec<_> = (0..threads)
                    .map(|_| {
                        let tree = Arc::clone(&tree);
                        let barrier = Arc::clone(&barrier);
                        thread::spawn(move || {
                            barrier.wait();
                            let guard = tree.guard();
                            let mut total = 0usize;
                            for _ in 0..OPS_PER_THREAD {
                                let mut count = 0usize;
                                tree.scan_values(
                                    RangeBound::Unbounded,
                                    RangeBound::Unbounded,
                                    |_| {
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
    }

    #[divan::bench(args = [1, 2, 3, 4, 5, 6])]
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
                let barrier = Arc::new(Barrier::new(threads));
                let handles: Vec<_> = (0..threads)
                    .map(|_| {
                        let tree = Arc::clone(&tree);
                        let barrier = Arc::clone(&barrier);
                        thread::spawn(move || {
                            barrier.wait();
                            let guard = sdd::Guard::new();
                            let mut total = 0usize;
                            for _ in 0..OPS_PER_THREAD {
                                let count = tree.iter(&guard).take(SCAN_LIMIT).count();
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
    }
}

// =============================================================================
// 10: B-LINK STRESS - Fragmented scan stress
// =============================================================================

#[divan::bench_group(name = "10_blink_stress_scan", sample_count = 100)]
mod blink_stress_scan {
    use super::*;

    #[divan::bench(args = [1, 2, 3, 4, 5, 6])]
    fn masstree15_inline(bencher: Bencher, threads: usize) {
        let keys = Arc::new(keys_blink_stress::<8>(N));
        let tree = Arc::new(setup_masstree15_inline(&keys));

        bencher
            .counter(divan::counter::ItemsCount::new(threads * OPS_PER_THREAD))
            .bench_local(|| {
                let barrier = Arc::new(Barrier::new(threads));
                let handles: Vec<_> = (0..threads)
                    .map(|_| {
                        let tree = Arc::clone(&tree);
                        let barrier = Arc::clone(&barrier);
                        thread::spawn(move || {
                            barrier.wait();
                            let guard = tree.guard();
                            let mut total = 0usize;
                            for _ in 0..OPS_PER_THREAD {
                                let mut count = 0usize;
                                tree.scan_values(
                                    RangeBound::Unbounded,
                                    RangeBound::Unbounded,
                                    |_| {
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
    }

    #[divan::bench(args = [1, 2, 3, 4, 5, 6])]
    fn tree_index(bencher: Bencher, threads: usize) {
        let keys = Arc::new(keys_blink_stress::<8>(N));
        let tree = Arc::new(setup_tree_index(&keys));

        bencher
            .counter(divan::counter::ItemsCount::new(threads * OPS_PER_THREAD))
            .bench_local(|| {
                let barrier = Arc::new(Barrier::new(threads));
                let handles: Vec<_> = (0..threads)
                    .map(|_| {
                        let tree = Arc::clone(&tree);
                        let barrier = Arc::clone(&barrier);
                        thread::spawn(move || {
                            barrier.wait();
                            let guard = sdd::Guard::new();
                            let mut total = 0usize;
                            for _ in 0..OPS_PER_THREAD {
                                let count = tree.iter(&guard).take(SCAN_LIMIT).count();
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
    }
}

// =============================================================================
// 11: RANDOM KEYS - Baseline comparison
// =============================================================================

#[divan::bench_group(name = "11_random_keys_scan", sample_count = 100)]
mod random_keys_scan {
    use super::*;

    #[divan::bench(args = [1, 2, 3, 4, 5, 6])]
    fn masstree15_inline(bencher: Bencher, threads: usize) {
        let keys = Arc::new(keys::<8>(N));
        let tree = Arc::new(setup_masstree15_inline(&keys));

        bencher
            .counter(divan::counter::ItemsCount::new(threads * OPS_PER_THREAD))
            .bench_local(|| {
                let barrier = Arc::new(Barrier::new(threads));
                let handles: Vec<_> = (0..threads)
                    .map(|_| {
                        let tree = Arc::clone(&tree);
                        let barrier = Arc::clone(&barrier);
                        thread::spawn(move || {
                            barrier.wait();
                            let guard = tree.guard();
                            let mut total = 0usize;
                            for _ in 0..OPS_PER_THREAD {
                                let mut count = 0usize;
                                tree.scan_values(
                                    RangeBound::Unbounded,
                                    RangeBound::Unbounded,
                                    |_| {
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
    }

    #[divan::bench(args = [1, 2, 3, 4, 5, 6])]
    fn tree_index(bencher: Bencher, threads: usize) {
        let keys = Arc::new(keys::<8>(N));
        let tree = Arc::new(setup_tree_index(&keys));

        bencher
            .counter(divan::counter::ItemsCount::new(threads * OPS_PER_THREAD))
            .bench_local(|| {
                let barrier = Arc::new(Barrier::new(threads));
                let handles: Vec<_> = (0..threads)
                    .map(|_| {
                        let tree = Arc::clone(&tree);
                        let barrier = Arc::clone(&barrier);
                        thread::spawn(move || {
                            barrier.wait();
                            let guard = sdd::Guard::new();
                            let mut total = 0usize;
                            for _ in 0..OPS_PER_THREAD {
                                let count = tree.iter(&guard).take(SCAN_LIMIT).count();
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
    }
}

// =============================================================================
// 12: LONG KEYS (64B) - Multi-layer traversal
// =============================================================================

#[divan::bench_group(name = "12_long_keys_64b_scan", sample_count = 100)]
mod long_keys_scan {
    use super::*;

    #[divan::bench(args = [1, 2, 3, 4, 5, 6])]
    fn masstree15_inline(bencher: Bencher, threads: usize) {
        let keys = Arc::new(keys::<64>(N));
        let tree = Arc::new(setup_masstree15_inline(&keys));

        bencher
            .counter(divan::counter::ItemsCount::new(threads * OPS_PER_THREAD))
            .bench_local(|| {
                let barrier = Arc::new(Barrier::new(threads));
                let handles: Vec<_> = (0..threads)
                    .map(|_| {
                        let tree = Arc::clone(&tree);
                        let barrier = Arc::clone(&barrier);
                        thread::spawn(move || {
                            barrier.wait();
                            let guard = tree.guard();
                            let mut total = 0usize;
                            for _ in 0..OPS_PER_THREAD {
                                let mut count = 0usize;
                                tree.scan_values(
                                    RangeBound::Unbounded,
                                    RangeBound::Unbounded,
                                    |_| {
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
    }

    #[divan::bench(args = [1, 2, 3, 4, 5, 6])]
    fn tree_index(bencher: Bencher, threads: usize) {
        let keys = Arc::new(keys::<64>(N));
        let tree = Arc::new(setup_tree_index(&keys));

        bencher
            .counter(divan::counter::ItemsCount::new(threads * OPS_PER_THREAD))
            .bench_local(|| {
                let barrier = Arc::new(Barrier::new(threads));
                let handles: Vec<_> = (0..threads)
                    .map(|_| {
                        let tree = Arc::clone(&tree);
                        let barrier = Arc::clone(&barrier);
                        thread::spawn(move || {
                            barrier.wait();
                            let guard = sdd::Guard::new();
                            let mut total = 0usize;
                            for _ in 0..OPS_PER_THREAD {
                                let count = tree.iter(&guard).take(SCAN_LIMIT).count();
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
    fn masstree15_inline(bencher: Bencher, threads: usize) {
        let initial_keys = keys::<8>(INITIAL_N);
        let insert_keys: Vec<_> = keys::<8>(INITIAL_N + INSERT_N)[INITIAL_N..].to_vec();
        let readers = threads - WRITERS;

        bencher
            .with_inputs(|| {
                let tree = Arc::new(setup_masstree15_inline(&initial_keys));
                let new_keys = Arc::new(insert_keys.clone());
                (tree, new_keys)
            })
            .bench_refs(|(tree, new_keys)| {
                let barrier = Arc::new(Barrier::new(threads));

                // Writer threads
                let writer_handles: Vec<_> = (0..WRITERS)
                    .map(|w| {
                        let tree = Arc::clone(tree);
                        let barrier = Arc::clone(&barrier);
                        let chunk_size = INSERT_N / WRITERS;
                        let chunk: Vec<_> = new_keys[w * chunk_size..(w + 1) * chunk_size].to_vec();
                        thread::spawn(move || {
                            barrier.wait();
                            let guard = tree.guard();
                            for (i, key) in chunk.iter().enumerate() {
                                let _ = tree.insert_with_guard(key, (INITIAL_N + i) as u64, &guard);
                            }
                        })
                    })
                    .collect();

                // Reader threads
                let reader_handles: Vec<_> = (0..readers)
                    .map(|_| {
                        let tree = Arc::clone(tree);
                        let barrier = Arc::clone(&barrier);
                        thread::spawn(move || {
                            barrier.wait();
                            let guard = tree.guard();
                            let mut total = 0usize;
                            for _ in 0..OPS_PER_THREAD {
                                let mut count = 0usize;
                                tree.scan_values(
                                    RangeBound::Unbounded,
                                    RangeBound::Unbounded,
                                    |_| {
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

    #[divan::bench(args = [1, 2, 3, 4, 5, 6])]
    fn masstree15_inline(bencher: Bencher, threads: usize) {
        let keys = Arc::new(keys_shared_prefix::<16>(N, PREFIX_BUCKETS));
        let tree = Arc::new(setup_masstree15_inline(&keys));

        bencher
            .counter(divan::counter::ItemsCount::new(threads * PREFIX_OPS))
            .bench_local(|| {
                let barrier = Arc::new(Barrier::new(threads));
                let handles: Vec<_> = (0..threads)
                    .map(|t| {
                        let tree = Arc::clone(&tree);
                        let barrier = Arc::clone(&barrier);
                        // Each thread scans a different prefix
                        let thread_prefix = ((t as u64) % PREFIX_BUCKETS).to_be_bytes();
                        thread::spawn(move || {
                            barrier.wait();
                            let guard = tree.guard();
                            let mut total = 0usize;
                            for _ in 0..PREFIX_OPS {
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
    }
}

// =============================================================================
// 15: FULL SCAN AGGREGATE - Sum all values (reports keys/sec, not scans/sec)
// =============================================================================

#[divan::bench_group(name = "15_full_scan_aggregate", sample_count = 100)]
mod full_scan_aggregate {
    use super::*;

    const SCAN_N: usize = 25_000; // Smaller dataset for full scans
    const FULL_SCAN_OPS: usize = 50; // Reduced iterations for full scans

    #[divan::bench(args = [1, 2, 3, 4, 5, 6])]
    fn masstree15_inline(bencher: Bencher, threads: usize) {
        let keys = Arc::new(keys_sequential::<8>(SCAN_N));
        let tree = Arc::new(setup_masstree15_inline(&keys));

        // Report keys/sec (not scans/sec) for meaningful throughput comparison
        bencher
            .counter(divan::counter::ItemsCount::new(
                threads * FULL_SCAN_OPS * SCAN_N,
            ))
            .bench_local(|| {
                let barrier = Arc::new(Barrier::new(threads));
                let handles: Vec<_> = (0..threads)
                    .map(|_| {
                        let tree = Arc::clone(&tree);
                        let barrier = Arc::clone(&barrier);
                        thread::spawn(move || {
                            barrier.wait();
                            let guard = tree.guard();
                            let mut grand_total = 0u64;
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
                            black_box(grand_total);
                        })
                    })
                    .collect();

                for h in handles {
                    h.join().unwrap();
                }
            });
    }

    #[divan::bench(args = [1, 2, 3, 4, 5, 6])]
    fn tree_index(bencher: Bencher, threads: usize) {
        let keys = Arc::new(keys_sequential::<8>(SCAN_N));
        let tree = Arc::new(setup_tree_index(&keys));

        // Report keys/sec (not scans/sec) for meaningful throughput comparison
        bencher
            .counter(divan::counter::ItemsCount::new(
                threads * FULL_SCAN_OPS * SCAN_N,
            ))
            .bench_local(|| {
                let barrier = Arc::new(Barrier::new(threads));
                let handles: Vec<_> = (0..threads)
                    .map(|_| {
                        let tree = Arc::clone(&tree);
                        let barrier = Arc::clone(&barrier);
                        thread::spawn(move || {
                            barrier.wait();
                            let guard = sdd::Guard::new();
                            let mut grand_total = 0u64;
                            for _ in 0..FULL_SCAN_OPS {
                                let sum: u64 = tree.iter(&guard).map(|(_, v)| *v).sum();
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

    #[divan::bench(args = [1, 2, 3, 4, 5, 6])]
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
                let barrier = Arc::new(Barrier::new(threads));
                let handles: Vec<_> = (0..threads)
                    .map(|t| {
                        let tree = Arc::clone(tree);
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
            });
    }

    #[divan::bench(args = [1, 2, 3, 4, 5, 6])]
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
                let barrier = Arc::new(Barrier::new(threads));
                let handles: Vec<_> = (0..threads)
                    .map(|t| {
                        let tree = Arc::clone(tree);
                        let barrier = Arc::clone(&barrier);
                        let keys = thread_keys[t].clone();
                        let read_keys = initial_keys.clone();
                        thread::spawn(move || {
                            barrier.wait();
                            let guard = sdd::Guard::new();
                            let mut rng_state = t as u64;
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

#[divan::bench_group(name = "17_hot_spot", sample_count = 100)]
mod hot_spot {
    use super::*;

    const TOTAL_N: usize = 500_000;
    const HOT_RANGE: usize = 5; // Only 10 keys are "hot" - extreme contention
    const OPS: usize = 5_000;

    #[divan::bench(args = [1, 2, 3, 4, 5, 6])]
    fn masstree15_inline(bencher: Bencher, threads: usize) {
        let all_keys: Vec<[u8; 8]> = keys_sequential::<8>(TOTAL_N);
        let hot_keys: Vec<[u8; 8]> = all_keys[..HOT_RANGE].to_vec();

        bencher
            .counter(divan::counter::ItemsCount::new(threads * OPS))
            .with_inputs(|| Arc::new(setup_masstree15_inline(&all_keys)))
            .bench_refs(|tree| {
                let barrier = Arc::new(Barrier::new(threads));
                let handles: Vec<_> = (0..threads)
                    .map(|t| {
                        let tree = Arc::clone(tree);
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
            });
    }

    #[divan::bench(args = [1, 2, 3, 4, 5, 6])]
    fn tree_index(bencher: Bencher, threads: usize) {
        let all_keys: Vec<[u8; 8]> = keys_sequential::<8>(TOTAL_N);
        let hot_keys: Vec<[u8; 8]> = all_keys[..HOT_RANGE].to_vec();

        bencher
            .counter(divan::counter::ItemsCount::new(threads * OPS))
            .with_inputs(|| Arc::new(setup_tree_index(&all_keys)))
            .bench_refs(|tree| {
                let barrier = Arc::new(Barrier::new(threads));
                let handles: Vec<_> = (0..threads)
                    .map(|t| {
                        let tree = Arc::clone(tree);
                        let barrier = Arc::clone(&barrier);
                        let hot = hot_keys.clone();

                        thread::spawn(move || {
                            barrier.wait();
                            let guard = sdd::Guard::new();
                            let mut rng_state = t as u64;

                            for i in 0..OPS {
                                rng_state =
                                    rng_state.wrapping_mul(6364136223846793005).wrapping_add(1);
                                let idx = (rng_state as usize) % hot.len();

                                if rng_state.is_multiple_of(2) {
                                    black_box(tree.peek(&hot[idx], &guard));
                                } else {
                                    let _ = tree.insert_sync(hot[idx], i as u64);
                                }
                            }
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
                let barrier = Arc::new(Barrier::new(threads));

                // Writer threads - sequential inserts cause leaf splits
                let writer_handles: Vec<_> = (0..WRITERS)
                    .map(|w| {
                        let tree = Arc::clone(tree);
                        let barrier = Arc::clone(&barrier);
                        let chunk_size = INSERT_N / WRITERS;
                        let chunk: Vec<_> = new_keys[w * chunk_size..(w + 1) * chunk_size].to_vec();
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
                        let tree = Arc::clone(tree);
                        let barrier = Arc::clone(&barrier);
                        thread::spawn(move || {
                            barrier.wait();
                            let guard = tree.guard();
                            let mut total = 0usize;
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
// 19: REVERSE SCAN - Batch-optimized reverse traversal (end to start)
// =============================================================================

#[divan::bench_group(name = "19_reverse_scan_sequential", sample_count = 100)]
mod reverse_scan_sequential {
    use super::*;

    #[divan::bench(args = [1, 2, 3, 4, 5, 6])]
    fn masstree15_inline(bencher: Bencher, threads: usize) {
        let keys = Arc::new(keys_sequential::<8>(N));
        let tree = Arc::new(setup_masstree15_inline(&keys));

        bencher
            .counter(divan::counter::ItemsCount::new(threads * OPS_PER_THREAD))
            .bench_local(|| {
                let barrier = Arc::new(Barrier::new(threads));
                let handles: Vec<_> = (0..threads)
                    .map(|_| {
                        let tree = Arc::clone(&tree);
                        let barrier = Arc::clone(&barrier);
                        thread::spawn(move || {
                            barrier.wait();
                            let guard = tree.guard();
                            let mut total = 0usize;
                            for _ in 0..OPS_PER_THREAD {
                                let mut count = 0usize;
                                tree.scan_values_rev(
                                    RangeBound::Unbounded,
                                    RangeBound::Unbounded,
                                    |_| {
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
    }
}

// =============================================================================
// 20: REVERSE SCAN RANDOM - Batch-optimized reverse traversal with random keys
// =============================================================================

#[divan::bench_group(name = "20_reverse_scan_random", sample_count = 100)]
mod reverse_scan_random {
    use super::*;

    #[divan::bench(args = [1, 2, 3, 4, 5, 6])]
    fn masstree15_inline(bencher: Bencher, threads: usize) {
        let keys = Arc::new(keys::<8>(N));
        let tree = Arc::new(setup_masstree15_inline(&keys));

        bencher
            .counter(divan::counter::ItemsCount::new(threads * OPS_PER_THREAD))
            .bench_local(|| {
                let barrier = Arc::new(Barrier::new(threads));
                let handles: Vec<_> = (0..threads)
                    .map(|_| {
                        let tree = Arc::clone(&tree);
                        let barrier = Arc::clone(&barrier);
                        thread::spawn(move || {
                            barrier.wait();
                            let guard = tree.guard();
                            let mut total = 0usize;
                            for _ in 0..OPS_PER_THREAD {
                                let mut count = 0usize;
                                tree.scan_values_rev(
                                    RangeBound::Unbounded,
                                    RangeBound::Unbounded,
                                    |_| {
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
    }
}

// =============================================================================
// 21: REVERSE SCAN LONG KEYS - Multi-layer batch reverse traversal (64B keys)
// =============================================================================

#[divan::bench_group(name = "21_reverse_scan_long_keys", sample_count = 100)]
mod reverse_scan_long_keys {
    use super::*;

    #[divan::bench(args = [1, 2, 3, 4, 5, 6])]
    fn masstree15_inline(bencher: Bencher, threads: usize) {
        let keys = Arc::new(keys::<64>(N));
        let tree = Arc::new(setup_masstree15_inline(&keys));

        bencher
            .counter(divan::counter::ItemsCount::new(threads * OPS_PER_THREAD))
            .bench_local(|| {
                let barrier = Arc::new(Barrier::new(threads));
                let handles: Vec<_> = (0..threads)
                    .map(|_| {
                        let tree = Arc::clone(&tree);
                        let barrier = Arc::clone(&barrier);
                        thread::spawn(move || {
                            barrier.wait();
                            let guard = tree.guard();
                            let mut total = 0usize;
                            for _ in 0..OPS_PER_THREAD {
                                let mut count = 0usize;
                                tree.scan_values_rev(
                                    RangeBound::Unbounded,
                                    RangeBound::Unbounded,
                                    |_| {
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
    }
}

// =============================================================================
// 22: BIDIRECTIONAL SCAN - Alternating forward/reverse (meeting in middle)
// =============================================================================

#[divan::bench_group(name = "22_bidirectional_scan", sample_count = 100)]
mod bidirectional_scan {
    use super::*;

    const BIDIR_LIMIT: usize = 25; // Take 50 from each end

    #[divan::bench(args = [1, 2, 3, 4, 5, 6])]
    fn masstree15_inline(bencher: Bencher, threads: usize) {
        let keys = Arc::new(keys_sequential::<8>(N));
        let tree = Arc::new(setup_masstree15_inline(&keys));

        bencher
            .counter(divan::counter::ItemsCount::new(threads * OPS_PER_THREAD))
            .bench_local(|| {
                let barrier = Arc::new(Barrier::new(threads));
                let handles: Vec<_> = (0..threads)
                    .map(|_| {
                        let tree = Arc::clone(&tree);
                        let barrier = Arc::clone(&barrier);
                        thread::spawn(move || {
                            barrier.wait();
                            let guard = tree.guard();
                            let mut total = 0usize;
                            for _ in 0..OPS_PER_THREAD {
                                let mut iter = tree.iter(&guard);
                                let mut count = 0usize;
                                // Alternate: take from front, then back
                                for _ in 0..BIDIR_LIMIT {
                                    if iter.next().is_some() {
                                        count += 1;
                                    }
                                    if iter.next_back().is_some() {
                                        count += 1;
                                    }
                                }
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
    }
}

// =============================================================================
// 23: REVERSE FULL SCAN AGGREGATE - Batch sum values in reverse order
// =============================================================================

#[divan::bench_group(name = "23_reverse_full_aggregate", sample_count = 100)]
mod reverse_full_aggregate {
    use super::*;

    const SCAN_N: usize = 25_000;
    const FULL_SCAN_OPS: usize = 50;

    #[divan::bench(args = [1, 2, 3, 4, 5, 6])]
    fn masstree15_inline(bencher: Bencher, threads: usize) {
        let keys = Arc::new(keys_sequential::<8>(SCAN_N));
        let tree = Arc::new(setup_masstree15_inline(&keys));

        bencher
            .counter(divan::counter::ItemsCount::new(threads * FULL_SCAN_OPS))
            .bench_local(|| {
                let barrier = Arc::new(Barrier::new(threads));
                let handles: Vec<_> = (0..threads)
                    .map(|_| {
                        let tree = Arc::clone(&tree);
                        let barrier = Arc::clone(&barrier);
                        thread::spawn(move || {
                            barrier.wait();
                            let guard = tree.guard();
                            let mut grand_total = 0u64;
                            for _ in 0..FULL_SCAN_OPS {
                                let mut sum = 0u64;
                                tree.scan_values_rev(
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
// 24: REVERSE SCAN HIERARCHICAL - Batch multi-layer keys with prefix structure
// =============================================================================

#[divan::bench_group(name = "24_reverse_hierarchical", sample_count = 100)]
mod reverse_hierarchical {
    use super::*;

    const NAMESPACES: usize = 100;
    const CATEGORIES: usize = 100;
    const ITEMS: usize = 50;

    #[divan::bench(args = [1, 2, 3, 4, 5, 6])]
    fn masstree15_inline(bencher: Bencher, threads: usize) {
        let keys = Arc::new(keys_hierarchical::<32>(NAMESPACES, CATEGORIES, ITEMS));
        let tree = Arc::new(setup_masstree15_inline(&keys));

        bencher
            .counter(divan::counter::ItemsCount::new(threads * OPS_PER_THREAD))
            .bench_local(|| {
                let barrier = Arc::new(Barrier::new(threads));
                let handles: Vec<_> = (0..threads)
                    .map(|_| {
                        let tree = Arc::clone(&tree);
                        let barrier = Arc::clone(&barrier);
                        thread::spawn(move || {
                            barrier.wait();
                            let guard = tree.guard();
                            let mut total = 0usize;
                            for _ in 0..OPS_PER_THREAD {
                                let mut count = 0usize;
                                tree.scan_values_rev(
                                    RangeBound::Unbounded,
                                    RangeBound::Unbounded,
                                    |_| {
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
    }
}

// =============================================================================
// 25: RANDOM START SEQUENTIAL - Scan from random positions (more realistic)
//
// Unlike groups 01-12 which always scan from the beginning, this tests scans
// starting from random keys distributed across the tree. This avoids:
// - Always measuring the same (cache-hot) first few leaves
// - Missing performance characteristics in different tree regions
// =============================================================================

#[divan::bench_group(name = "25_random_start_sequential", sample_count = 100)]
mod random_start_sequential {
    use super::*;

    #[divan::bench(args = [1, 2, 3, 4, 5, 6])]
    fn masstree15_inline(bencher: Bencher, threads: usize) {
        let keys = Arc::new(keys_sequential::<8>(N));
        let tree = Arc::new(setup_masstree15_inline(&keys));

        // Pre-generate random start indices for each thread
        let start_indices: Vec<Vec<usize>> = (0..threads)
            .map(|t| random_start_indices(N, OPS_PER_THREAD, 0xDEAD_BEEF + t as u64))
            .collect();
        let start_indices = Arc::new(start_indices);

        bencher
            .counter(divan::counter::ItemsCount::new(threads * OPS_PER_THREAD))
            .bench_local(|| {
                let barrier = Arc::new(Barrier::new(threads));
                let handles: Vec<_> = (0..threads)
                    .map(|t| {
                        let tree = Arc::clone(&tree);
                        let keys = Arc::clone(&keys);
                        let starts = Arc::clone(&start_indices);
                        let barrier = Arc::clone(&barrier);
                        thread::spawn(move || {
                            barrier.wait();
                            let guard = tree.guard();
                            let mut total = 0usize;
                            for (op_idx, &start_idx) in starts[t].iter().enumerate() {
                                let start_key = &keys[start_idx];
                                let mut count = 0usize;
                                tree.scan_values(
                                    RangeBound::Included(start_key.as_slice()),
                                    RangeBound::Unbounded,
                                    |_| {
                                        count += 1;
                                        count < SCAN_LIMIT
                                    },
                                    &guard,
                                );
                                total += count;
                                black_box(op_idx);
                            }
                            black_box(total);
                        })
                    })
                    .collect();

                for h in handles {
                    h.join().unwrap();
                }
            });
    }

    #[divan::bench(args = [1, 2, 3, 4, 5, 6])]
    fn tree_index(bencher: Bencher, threads: usize) {
        let keys = Arc::new(keys_sequential::<8>(N));
        let tree = Arc::new(setup_tree_index(&keys));

        let start_indices: Vec<Vec<usize>> = (0..threads)
            .map(|t| random_start_indices(N, OPS_PER_THREAD, 0xDEAD_BEEF + t as u64))
            .collect();
        let start_indices = Arc::new(start_indices);

        bencher
            .counter(divan::counter::ItemsCount::new(threads * OPS_PER_THREAD))
            .bench_local(|| {
                let barrier = Arc::new(Barrier::new(threads));
                let handles: Vec<_> = (0..threads)
                    .map(|t| {
                        let tree = Arc::clone(&tree);
                        let keys = Arc::clone(&keys);
                        let starts = Arc::clone(&start_indices);
                        let barrier = Arc::clone(&barrier);
                        thread::spawn(move || {
                            barrier.wait();
                            let guard = sdd::Guard::new();
                            let mut total = 0usize;
                            for &start_idx in starts[t].iter() {
                                let start_key = &keys[start_idx];
                                let count =
                                    tree.range(*start_key.., &guard).take(SCAN_LIMIT).count();
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
    }
}

// =============================================================================
// 26: RANDOM START RANDOM KEYS - Random starts on random key distribution
// =============================================================================

#[divan::bench_group(name = "26_random_start_random", sample_count = 100)]
mod random_start_random {
    use super::*;

    #[divan::bench(args = [1, 2, 3, 4, 5, 6])]
    fn masstree15_inline(bencher: Bencher, threads: usize) {
        let keys = Arc::new(keys::<8>(N));
        // Sort keys for range scans (tree stores them sorted anyway)
        let mut sorted_keys = (*keys).clone();
        sorted_keys.sort_unstable();
        let sorted_keys = Arc::new(sorted_keys);
        let tree = Arc::new(setup_masstree15_inline(&keys));

        let start_indices: Vec<Vec<usize>> = (0..threads)
            .map(|t| random_start_indices(N, OPS_PER_THREAD, 0xCAFE_BABE + t as u64))
            .collect();
        let start_indices = Arc::new(start_indices);

        bencher
            .counter(divan::counter::ItemsCount::new(threads * OPS_PER_THREAD))
            .bench_local(|| {
                let barrier = Arc::new(Barrier::new(threads));
                let handles: Vec<_> = (0..threads)
                    .map(|t| {
                        let tree = Arc::clone(&tree);
                        let keys = Arc::clone(&sorted_keys);
                        let starts = Arc::clone(&start_indices);
                        let barrier = Arc::clone(&barrier);
                        thread::spawn(move || {
                            barrier.wait();
                            let guard = tree.guard();
                            let mut total = 0usize;
                            for &start_idx in starts[t].iter() {
                                let start_key = &keys[start_idx];
                                let mut count = 0usize;
                                tree.scan_values(
                                    RangeBound::Included(start_key.as_slice()),
                                    RangeBound::Unbounded,
                                    |_| {
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
    }

    #[divan::bench(args = [1, 2, 3, 4, 5, 6])]
    fn tree_index(bencher: Bencher, threads: usize) {
        let keys = Arc::new(keys::<8>(N));
        let mut sorted_keys = (*keys).clone();
        sorted_keys.sort_unstable();
        let sorted_keys = Arc::new(sorted_keys);
        let tree = Arc::new(setup_tree_index(&keys));

        let start_indices: Vec<Vec<usize>> = (0..threads)
            .map(|t| random_start_indices(N, OPS_PER_THREAD, 0xCAFE_BABE + t as u64))
            .collect();
        let start_indices = Arc::new(start_indices);

        bencher
            .counter(divan::counter::ItemsCount::new(threads * OPS_PER_THREAD))
            .bench_local(|| {
                let barrier = Arc::new(Barrier::new(threads));
                let handles: Vec<_> = (0..threads)
                    .map(|t| {
                        let tree = Arc::clone(&tree);
                        let keys = Arc::clone(&sorted_keys);
                        let starts = Arc::clone(&start_indices);
                        let barrier = Arc::clone(&barrier);
                        thread::spawn(move || {
                            barrier.wait();
                            let guard = sdd::Guard::new();
                            let mut total = 0usize;
                            for &start_idx in starts[t].iter() {
                                let start_key = &keys[start_idx];
                                let count =
                                    tree.range(*start_key.., &guard).take(SCAN_LIMIT).count();
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
    }
}

// =============================================================================
// 27: ITERATOR BASELINE - Compare batch API vs standard iterator
//
// This isolates the "batch API advantage" by comparing:
// - masstree15_batch: scan_values (optimized value-only batch scan)
// - masstree15_iter: tree.iter().take() (standard iterator)
// - tree_index: tree.iter().take() (reference)
//
// If batch >> iter, the advantage is in the batch API, not the data structure.
// =============================================================================

#[divan::bench_group(name = "27_iterator_baseline", sample_count = 100)]
mod iterator_baseline {
    use super::*;

    #[divan::bench(args = [1, 2, 3, 4, 5, 6])]
    fn masstree15_batch(bencher: Bencher, threads: usize) {
        let keys = Arc::new(keys_sequential::<8>(N));
        let tree = Arc::new(setup_masstree15_inline(&keys));

        bencher
            .counter(divan::counter::ItemsCount::new(threads * OPS_PER_THREAD))
            .bench_local(|| {
                let barrier = Arc::new(Barrier::new(threads));
                let handles: Vec<_> = (0..threads)
                    .map(|_| {
                        let tree = Arc::clone(&tree);
                        let barrier = Arc::clone(&barrier);
                        thread::spawn(move || {
                            barrier.wait();
                            let guard = tree.guard();
                            let mut total = 0usize;
                            for _ in 0..OPS_PER_THREAD {
                                let mut count = 0usize;
                                tree.scan_values(
                                    RangeBound::Unbounded,
                                    RangeBound::Unbounded,
                                    |_| {
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
    }

    #[divan::bench(args = [1, 2, 3, 4, 5, 6])]
    fn masstree15_iter(bencher: Bencher, threads: usize) {
        let keys = Arc::new(keys_sequential::<8>(N));
        let tree = Arc::new(setup_masstree15_inline(&keys));

        bencher
            .counter(divan::counter::ItemsCount::new(threads * OPS_PER_THREAD))
            .bench_local(|| {
                let barrier = Arc::new(Barrier::new(threads));
                let handles: Vec<_> = (0..threads)
                    .map(|_| {
                        let tree = Arc::clone(&tree);
                        let barrier = Arc::clone(&barrier);
                        thread::spawn(move || {
                            barrier.wait();
                            let guard = tree.guard();
                            let mut total = 0usize;
                            for _ in 0..OPS_PER_THREAD {
                                // Use standard iterator instead of batch API
                                let count = tree.iter(&guard).take(SCAN_LIMIT).count();
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
    }

    #[divan::bench(args = [1, 2, 3, 4, 5, 6])]
    fn tree_index(bencher: Bencher, threads: usize) {
        let keys = Arc::new(keys_sequential::<8>(N));
        let tree = Arc::new(setup_tree_index(&keys));

        bencher
            .counter(divan::counter::ItemsCount::new(threads * OPS_PER_THREAD))
            .bench_local(|| {
                let barrier = Arc::new(Barrier::new(threads));
                let handles: Vec<_> = (0..threads)
                    .map(|_| {
                        let tree = Arc::clone(&tree);
                        let barrier = Arc::clone(&barrier);
                        thread::spawn(move || {
                            barrier.wait();
                            let guard = sdd::Guard::new();
                            let mut total = 0usize;
                            for _ in 0..OPS_PER_THREAD {
                                let count = tree.iter(&guard).take(SCAN_LIMIT).count();
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
    }
}

// =============================================================================
// 28: RANDOM START LONG KEYS (64B) - Random starts with multi-layer keys
// =============================================================================

#[divan::bench_group(name = "28_random_start_long_keys", sample_count = 100)]
mod random_start_long_keys {
    use super::*;

    #[divan::bench(args = [1, 2, 3, 4, 5, 6])]
    fn masstree15_inline(bencher: Bencher, threads: usize) {
        let keys = Arc::new(keys::<64>(N));
        let mut sorted_keys = (*keys).clone();
        sorted_keys.sort_unstable();
        let sorted_keys = Arc::new(sorted_keys);
        let tree = Arc::new(setup_masstree15_inline(&keys));

        let start_indices: Vec<Vec<usize>> = (0..threads)
            .map(|t| random_start_indices(N, OPS_PER_THREAD, 0xBEEF_CAFE + t as u64))
            .collect();
        let start_indices = Arc::new(start_indices);

        bencher
            .counter(divan::counter::ItemsCount::new(threads * OPS_PER_THREAD))
            .bench_local(|| {
                let barrier = Arc::new(Barrier::new(threads));
                let handles: Vec<_> = (0..threads)
                    .map(|t| {
                        let tree = Arc::clone(&tree);
                        let keys = Arc::clone(&sorted_keys);
                        let starts = Arc::clone(&start_indices);
                        let barrier = Arc::clone(&barrier);
                        thread::spawn(move || {
                            barrier.wait();
                            let guard = tree.guard();
                            let mut total = 0usize;
                            for &start_idx in starts[t].iter() {
                                let start_key = &keys[start_idx];
                                let mut count = 0usize;
                                tree.scan_values(
                                    RangeBound::Included(start_key.as_slice()),
                                    RangeBound::Unbounded,
                                    |_| {
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
    }

    #[divan::bench(args = [1, 2, 3, 4, 5, 6])]
    fn tree_index(bencher: Bencher, threads: usize) {
        let keys = Arc::new(keys::<64>(N));
        let mut sorted_keys = (*keys).clone();
        sorted_keys.sort_unstable();
        let sorted_keys = Arc::new(sorted_keys);
        let tree = Arc::new(setup_tree_index(&keys));

        let start_indices: Vec<Vec<usize>> = (0..threads)
            .map(|t| random_start_indices(N, OPS_PER_THREAD, 0xBEEF_CAFE + t as u64))
            .collect();
        let start_indices = Arc::new(start_indices);

        bencher
            .counter(divan::counter::ItemsCount::new(threads * OPS_PER_THREAD))
            .bench_local(|| {
                let barrier = Arc::new(Barrier::new(threads));
                let handles: Vec<_> = (0..threads)
                    .map(|t| {
                        let tree = Arc::clone(&tree);
                        let keys = Arc::clone(&sorted_keys);
                        let starts = Arc::clone(&start_indices);
                        let barrier = Arc::clone(&barrier);
                        thread::spawn(move || {
                            barrier.wait();
                            let guard = sdd::Guard::new();
                            let mut total = 0usize;
                            for &start_idx in starts[t].iter() {
                                let start_key = &keys[start_idx];
                                let count =
                                    tree.range(*start_key.., &guard).take(SCAN_LIMIT).count();
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
    }
}
