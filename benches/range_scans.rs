//! Range scan benchmarks comparing MassTree24 against other ordered concurrent maps.
//!
//! Compares:
//! - **MassTree24**: Trie of B+trees (this crate)
//! - **crossbeam-skiplist**: Lock-free skip list
//! - **indexset**: Concurrent B-tree
//! - **scc::TreeIndex**: Lock-free B+tree
//!
//! ## Notes on fairness
//!
//! MassTree has two scan APIs:
//! - `scan()` / `scan_cloned()`: Callback-based, no key allocation (fair comparison)
//! - `iter()`: Iterator returning `ScanEntry { key: Vec<u8>, value }` (allocates per entry)
//!
//! The main benchmarks use callback/reference-based iteration for fair comparison.
//! Iterator variants are included separately to show the allocation overhead.
//!
//! ## Running
//!
//! ```bash
//! cargo bench --bench range_scans
//! cargo bench --bench range_scans --features mimalloc
//! ```

#![expect(clippy::unwrap_used)]
#![expect(clippy::pedantic)]
#![expect(clippy::indexing_slicing)]

mod bench_utils;

use bench_utils::{keys, keys_shared_prefix};
use crossbeam_skiplist::SkipMap;
use divan::{Bencher, black_box};
use indexset::concurrent::map::BTreeMap as IndexSetBTreeMap;
use masstree::{MassTree24, RangeBound};
use scc::TreeIndex;
use std::sync::Arc;
use std::sync::Barrier;
use std::thread;

fn main() {
    divan::main();
}

// =============================================================================
// Setup Helpers
// =============================================================================

fn setup_masstree24<const K: usize>(keys: &[[u8; K]]) -> MassTree24<u64> {
    let tree = MassTree24::new();
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

fn setup_indexset<const K: usize>(keys: &[[u8; K]]) -> IndexSetBTreeMap<[u8; K], u64> {
    let map = IndexSetBTreeMap::new();
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

// =============================================================================
// 1. Full Scan 10k - Fair comparison (callback/ref-based)
// =============================================================================

#[divan::bench_group(name = "01_full_scan_10k")]
mod full_scan_10k {
    use super::*;

    const N: usize = 10_000;

    #[divan::bench]
    fn masstree24_8b(bencher: Bencher) {
        let keys = keys::<8>(N);
        let tree = setup_masstree24(&keys);

        bencher.bench(|| {
            let guard = tree.guard();
            let count = tree.scan(
                RangeBound::Unbounded,
                RangeBound::Unbounded,
                |_, _| true,
                &guard,
            );
            black_box(count)
        });
    }

    #[divan::bench]
    fn skipmap_8b(bencher: Bencher) {
        let keys = keys::<8>(N);
        let map = setup_skipmap(&keys);

        bencher.bench(|| {
            let count = map.iter().count();
            black_box(count)
        });
    }

    #[divan::bench]
    fn indexset_8b(bencher: Bencher) {
        let keys = keys::<8>(N);
        let map = setup_indexset(&keys);

        bencher.bench(|| {
            let count = map.iter().count();
            black_box(count)
        });
    }

    #[divan::bench]
    fn tree_index_8b(bencher: Bencher) {
        let keys = keys::<8>(N);
        let tree = setup_tree_index(&keys);

        bencher.bench(|| {
            let guard = sdd::Guard::new();
            let count = tree.iter(&guard).count();
            black_box(count)
        });
    }

    #[divan::bench]
    fn masstree24_32b(bencher: Bencher) {
        let keys = keys::<32>(N);
        let tree = setup_masstree24(&keys);

        bencher.bench(|| {
            let guard = tree.guard();
            let count = tree.scan(
                RangeBound::Unbounded,
                RangeBound::Unbounded,
                |_, _| true,
                &guard,
            );
            black_box(count)
        });
    }

    #[divan::bench]
    fn skipmap_32b(bencher: Bencher) {
        let keys = keys::<32>(N);
        let map = setup_skipmap(&keys);

        bencher.bench(|| {
            let count = map.iter().count();
            black_box(count)
        });
    }

    #[divan::bench]
    fn indexset_32b(bencher: Bencher) {
        let keys = keys::<32>(N);
        let map = setup_indexset(&keys);

        bencher.bench(|| {
            let count = map.iter().count();
            black_box(count)
        });
    }

    #[divan::bench]
    fn tree_index_32b(bencher: Bencher) {
        let keys = keys::<32>(N);
        let tree = setup_tree_index(&keys);

        bencher.bench(|| {
            let guard = sdd::Guard::new();
            let count = tree.iter(&guard).count();
            black_box(count)
        });
    }
}

// =============================================================================
// 2. Full Scan 100k
// =============================================================================

#[divan::bench_group(name = "02_full_scan_100k")]
mod full_scan_100k {
    use super::*;

    const N: usize = 100_000;

    #[divan::bench]
    fn masstree24_8b(bencher: Bencher) {
        let keys = keys::<8>(N);
        let tree = setup_masstree24(&keys);

        bencher.bench(|| {
            let guard = tree.guard();
            let count = tree.scan(
                RangeBound::Unbounded,
                RangeBound::Unbounded,
                |_, _| true,
                &guard,
            );
            black_box(count)
        });
    }

    #[divan::bench]
    fn skipmap_8b(bencher: Bencher) {
        let keys = keys::<8>(N);
        let map = setup_skipmap(&keys);

        bencher.bench(|| {
            let count = map.iter().count();
            black_box(count)
        });
    }

    #[divan::bench]
    fn indexset_8b(bencher: Bencher) {
        let keys = keys::<8>(N);
        let map = setup_indexset(&keys);

        bencher.bench(|| {
            let count = map.iter().count();
            black_box(count)
        });
    }

    #[divan::bench]
    fn tree_index_8b(bencher: Bencher) {
        let keys = keys::<8>(N);
        let tree = setup_tree_index(&keys);

        bencher.bench(|| {
            let guard = sdd::Guard::new();
            let count = tree.iter(&guard).count();
            black_box(count)
        });
    }

    #[divan::bench]
    fn masstree24_32b(bencher: Bencher) {
        let keys = keys::<32>(N);
        let tree = setup_masstree24(&keys);

        bencher.bench(|| {
            let guard = tree.guard();
            let count = tree.scan(
                RangeBound::Unbounded,
                RangeBound::Unbounded,
                |_, _| true,
                &guard,
            );
            black_box(count)
        });
    }

    #[divan::bench]
    fn skipmap_32b(bencher: Bencher) {
        let keys = keys::<32>(N);
        let map = setup_skipmap(&keys);

        bencher.bench(|| {
            let count = map.iter().count();
            black_box(count)
        });
    }

    #[divan::bench]
    fn indexset_32b(bencher: Bencher) {
        let keys = keys::<32>(N);
        let map = setup_indexset(&keys);

        bencher.bench(|| {
            let count = map.iter().count();
            black_box(count)
        });
    }

    #[divan::bench]
    fn tree_index_32b(bencher: Bencher) {
        let keys = keys::<32>(N);
        let tree = setup_tree_index(&keys);

        bencher.bench(|| {
            let guard = sdd::Guard::new();
            let count = tree.iter(&guard).count();
            black_box(count)
        });
    }
}

// =============================================================================
// 3. Range First 1k (bounded scan)
// =============================================================================

#[divan::bench_group(name = "03_range_first_1k")]
mod range_first_1k {
    use super::*;

    const N: usize = 100_000;
    const LIMIT: usize = 1_000;

    #[divan::bench]
    fn masstree24_8b(bencher: Bencher) {
        let keys = keys::<8>(N);
        let tree = setup_masstree24(&keys);

        bencher.bench(|| {
            let guard = tree.guard();
            let mut count = 0usize;
            tree.scan(
                RangeBound::Unbounded,
                RangeBound::Unbounded,
                |_, _| {
                    count += 1;
                    count < LIMIT
                },
                &guard,
            );
            black_box(count)
        });
    }

    #[divan::bench]
    fn skipmap_8b(bencher: Bencher) {
        let keys = keys::<8>(N);
        let map = setup_skipmap(&keys);

        bencher.bench(|| {
            let count = map.iter().take(LIMIT).count();
            black_box(count)
        });
    }

    #[divan::bench]
    fn indexset_8b(bencher: Bencher) {
        let keys = keys::<8>(N);
        let map = setup_indexset(&keys);

        bencher.bench(|| {
            let count = map.iter().take(LIMIT).count();
            black_box(count)
        });
    }

    #[divan::bench]
    fn tree_index_8b(bencher: Bencher) {
        let keys = keys::<8>(N);
        let tree = setup_tree_index(&keys);

        bencher.bench(|| {
            let guard = sdd::Guard::new();
            let count = tree.iter(&guard).take(LIMIT).count();
            black_box(count)
        });
    }

    #[divan::bench]
    fn masstree24_32b(bencher: Bencher) {
        let keys = keys::<32>(N);
        let tree = setup_masstree24(&keys);

        bencher.bench(|| {
            let guard = tree.guard();
            let mut count = 0usize;
            tree.scan(
                RangeBound::Unbounded,
                RangeBound::Unbounded,
                |_, _| {
                    count += 1;
                    count < LIMIT
                },
                &guard,
            );
            black_box(count)
        });
    }

    #[divan::bench]
    fn skipmap_32b(bencher: Bencher) {
        let keys = keys::<32>(N);
        let map = setup_skipmap(&keys);

        bencher.bench(|| {
            let count = map.iter().take(LIMIT).count();
            black_box(count)
        });
    }

    #[divan::bench]
    fn indexset_32b(bencher: Bencher) {
        let keys = keys::<32>(N);
        let map = setup_indexset(&keys);

        bencher.bench(|| {
            let count = map.iter().take(LIMIT).count();
            black_box(count)
        });
    }

    #[divan::bench]
    fn tree_index_32b(bencher: Bencher) {
        let keys = keys::<32>(N);
        let tree = setup_tree_index(&keys);

        bencher.bench(|| {
            let guard = sdd::Guard::new();
            let count = tree.iter(&guard).take(LIMIT).count();
            black_box(count)
        });
    }
}

// =============================================================================
// 4. Shared Prefix Scan (Masstree layer traversal stress)
// =============================================================================

#[divan::bench_group(name = "04_shared_prefix_scan")]
mod shared_prefix_scan {
    use super::*;

    const N: usize = 50_000;
    const PREFIX_BUCKETS: u64 = 100;

    #[divan::bench]
    fn masstree24_16b(bencher: Bencher) {
        let keys = keys_shared_prefix::<16>(N, PREFIX_BUCKETS);
        let tree = setup_masstree24(&keys);

        bencher.bench(|| {
            let guard = tree.guard();
            let count = tree.scan(
                RangeBound::Unbounded,
                RangeBound::Unbounded,
                |_, _| true,
                &guard,
            );
            black_box(count)
        });
    }

    #[divan::bench]
    fn skipmap_16b(bencher: Bencher) {
        let keys = keys_shared_prefix::<16>(N, PREFIX_BUCKETS);
        let map = setup_skipmap(&keys);

        bencher.bench(|| {
            let count = map.iter().count();
            black_box(count)
        });
    }

    #[divan::bench]
    fn indexset_16b(bencher: Bencher) {
        let keys = keys_shared_prefix::<16>(N, PREFIX_BUCKETS);
        let map = setup_indexset(&keys);

        bencher.bench(|| {
            let count = map.iter().count();
            black_box(count)
        });
    }

    #[divan::bench]
    fn tree_index_16b(bencher: Bencher) {
        let keys = keys_shared_prefix::<16>(N, PREFIX_BUCKETS);
        let tree = setup_tree_index(&keys);

        bencher.bench(|| {
            let guard = sdd::Guard::new();
            let count = tree.iter(&guard).count();
            black_box(count)
        });
    }

    #[divan::bench]
    fn masstree24_32b(bencher: Bencher) {
        let keys = keys_shared_prefix::<32>(N, PREFIX_BUCKETS);
        let tree = setup_masstree24(&keys);

        bencher.bench(|| {
            let guard = tree.guard();
            let count = tree.scan(
                RangeBound::Unbounded,
                RangeBound::Unbounded,
                |_, _| true,
                &guard,
            );
            black_box(count)
        });
    }

    #[divan::bench]
    fn skipmap_32b(bencher: Bencher) {
        let keys = keys_shared_prefix::<32>(N, PREFIX_BUCKETS);
        let map = setup_skipmap(&keys);

        bencher.bench(|| {
            let count = map.iter().count();
            black_box(count)
        });
    }

    #[divan::bench]
    fn indexset_32b(bencher: Bencher) {
        let keys = keys_shared_prefix::<32>(N, PREFIX_BUCKETS);
        let map = setup_indexset(&keys);

        bencher.bench(|| {
            let count = map.iter().count();
            black_box(count)
        });
    }

    #[divan::bench]
    fn tree_index_32b(bencher: Bencher) {
        let keys = keys_shared_prefix::<32>(N, PREFIX_BUCKETS);
        let tree = setup_tree_index(&keys);

        bencher.bench(|| {
            let guard = sdd::Guard::new();
            let count = tree.iter(&guard).count();
            black_box(count)
        });
    }
}

// =============================================================================
// 5. Scan with Aggregation (sum values)
// =============================================================================

#[divan::bench_group(name = "05_scan_aggregate")]
mod scan_aggregate {
    use super::*;

    const N: usize = 50_000;

    #[divan::bench]
    fn masstree24_sum_8b(bencher: Bencher) {
        let keys = keys::<8>(N);
        let tree = setup_masstree24(&keys);

        bencher.bench(|| {
            let guard = tree.guard();
            let mut sum = 0u64;
            tree.scan(
                RangeBound::Unbounded,
                RangeBound::Unbounded,
                |_, v| {
                    sum += *v;
                    true
                },
                &guard,
            );
            black_box(sum)
        });
    }

    #[divan::bench]
    fn skipmap_sum_8b(bencher: Bencher) {
        let keys = keys::<8>(N);
        let map = setup_skipmap(&keys);

        bencher.bench(|| {
            let sum: u64 = map.iter().map(|e| *e.value()).sum();
            black_box(sum)
        });
    }

    #[divan::bench]
    fn indexset_sum_8b(bencher: Bencher) {
        let keys = keys::<8>(N);
        let map = setup_indexset(&keys);

        bencher.bench(|| {
            let sum: u64 = map.iter().map(|(_, v)| *v).sum();
            black_box(sum)
        });
    }

    #[divan::bench]
    fn tree_index_sum_8b(bencher: Bencher) {
        let keys = keys::<8>(N);
        let tree = setup_tree_index(&keys);

        bencher.bench(|| {
            let guard = sdd::Guard::new();
            let sum: u64 = tree.iter(&guard).map(|(_, v)| *v).sum();
            black_box(sum)
        });
    }

    #[divan::bench]
    fn masstree24_sum_32b(bencher: Bencher) {
        let keys = keys::<32>(N);
        let tree = setup_masstree24(&keys);

        bencher.bench(|| {
            let guard = tree.guard();
            let mut sum = 0u64;
            tree.scan(
                RangeBound::Unbounded,
                RangeBound::Unbounded,
                |_, v| {
                    sum += *v;
                    true
                },
                &guard,
            );
            black_box(sum)
        });
    }

    #[divan::bench]
    fn skipmap_sum_32b(bencher: Bencher) {
        let keys = keys::<32>(N);
        let map = setup_skipmap(&keys);

        bencher.bench(|| {
            let sum: u64 = map.iter().map(|e| *e.value()).sum();
            black_box(sum)
        });
    }

    #[divan::bench]
    fn indexset_sum_32b(bencher: Bencher) {
        let keys = keys::<32>(N);
        let map = setup_indexset(&keys);

        bencher.bench(|| {
            let sum: u64 = map.iter().map(|(_, v)| *v).sum();
            black_box(sum)
        });
    }

    #[divan::bench]
    fn tree_index_sum_32b(bencher: Bencher) {
        let keys = keys::<32>(N);
        let tree = setup_tree_index(&keys);

        bencher.bench(|| {
            let guard = sdd::Guard::new();
            let sum: u64 = tree.iter(&guard).map(|(_, v)| *v).sum();
            black_box(sum)
        });
    }
}

// =============================================================================
// 6. Concurrent Scan (4 threads)
// =============================================================================

#[divan::bench_group(name = "06_concurrent_scan_4t")]
mod concurrent_scan_4t {
    use super::*;

    const N: usize = 100_000;
    const THREADS: usize = 4;
    const SCANS_PER_THREAD: usize = 10;

    #[divan::bench]
    fn masstree24_8b(bencher: Bencher) {
        let keys = keys::<8>(N);
        let tree = Arc::new(setup_masstree24(&keys));

        bencher.bench(|| {
            let barrier = Arc::new(Barrier::new(THREADS));
            let handles: Vec<_> = (0..THREADS)
                .map(|_| {
                    let tree = Arc::clone(&tree);
                    let barrier = Arc::clone(&barrier);
                    thread::spawn(move || {
                        barrier.wait();
                        let mut total = 0usize;
                        for _ in 0..SCANS_PER_THREAD {
                            let guard = tree.guard();
                            total += tree.scan(
                                RangeBound::Unbounded,
                                RangeBound::Unbounded,
                                |_, _| true,
                                &guard,
                            );
                        }
                        total
                    })
                })
                .collect();

            let sum: usize = handles.into_iter().map(|h| h.join().unwrap()).sum();
            black_box(sum)
        });
    }

    #[divan::bench]
    fn skipmap_8b(bencher: Bencher) {
        let keys = keys::<8>(N);
        let map = Arc::new(setup_skipmap(&keys));

        bencher.bench(|| {
            let barrier = Arc::new(Barrier::new(THREADS));
            let handles: Vec<_> = (0..THREADS)
                .map(|_| {
                    let map = Arc::clone(&map);
                    let barrier = Arc::clone(&barrier);
                    thread::spawn(move || {
                        barrier.wait();
                        let mut total = 0usize;
                        for _ in 0..SCANS_PER_THREAD {
                            total += map.iter().count();
                        }
                        total
                    })
                })
                .collect();

            let sum: usize = handles.into_iter().map(|h| h.join().unwrap()).sum();
            black_box(sum)
        });
    }

    #[divan::bench]
    fn indexset_8b(bencher: Bencher) {
        let keys = keys::<8>(N);
        let map = Arc::new(setup_indexset(&keys));

        bencher.bench(|| {
            let barrier = Arc::new(Barrier::new(THREADS));
            let handles: Vec<_> = (0..THREADS)
                .map(|_| {
                    let map = Arc::clone(&map);
                    let barrier = Arc::clone(&barrier);
                    thread::spawn(move || {
                        barrier.wait();
                        let mut total = 0usize;
                        for _ in 0..SCANS_PER_THREAD {
                            total += map.iter().count();
                        }
                        total
                    })
                })
                .collect();

            let sum: usize = handles.into_iter().map(|h| h.join().unwrap()).sum();
            black_box(sum)
        });
    }

    #[divan::bench]
    fn tree_index_8b(bencher: Bencher) {
        let keys = keys::<8>(N);
        let tree = Arc::new(setup_tree_index(&keys));

        bencher.bench(|| {
            let barrier = Arc::new(Barrier::new(THREADS));
            let handles: Vec<_> = (0..THREADS)
                .map(|_| {
                    let tree = Arc::clone(&tree);
                    let barrier = Arc::clone(&barrier);
                    thread::spawn(move || {
                        barrier.wait();
                        let mut total = 0usize;
                        for _ in 0..SCANS_PER_THREAD {
                            let guard = sdd::Guard::new();
                            total += tree.iter(&guard).count();
                        }
                        total
                    })
                })
                .collect();

            let sum: usize = handles.into_iter().map(|h| h.join().unwrap()).sum();
            black_box(sum)
        });
    }
}

// =============================================================================
// 7. Scan While Insert (mixed workload)
// =============================================================================

#[divan::bench_group(name = "07_scan_while_insert")]
mod scan_while_insert {
    use super::*;

    const INITIAL_N: usize = 50_000;
    const INSERT_N: usize = 10_000;
    const WRITERS: usize = 2;
    const READERS: usize = 2;
    const SCANS_PER_READER: usize = 5;

    #[divan::bench]
    fn masstree24_8b(bencher: Bencher) {
        let initial_keys = keys::<8>(INITIAL_N);
        let insert_keys = keys::<8>(INSERT_N + INITIAL_N);

        bencher
            .with_inputs(|| {
                let tree = Arc::new(setup_masstree24(&initial_keys));
                let new_keys: Vec<_> = insert_keys[INITIAL_N..].to_vec();
                (tree, new_keys)
            })
            .bench_refs(|(tree, new_keys)| {
                let barrier = Arc::new(Barrier::new(WRITERS + READERS));

                let writer_handles: Vec<_> = (0..WRITERS)
                    .map(|w| {
                        let tree = Arc::clone(tree);
                        let barrier = Arc::clone(&barrier);
                        let chunk: Vec<_> = new_keys
                            .iter()
                            .skip(w * (INSERT_N / WRITERS))
                            .take(INSERT_N / WRITERS)
                            .copied()
                            .collect();
                        thread::spawn(move || {
                            barrier.wait();
                            let guard = tree.guard();
                            for (i, key) in chunk.iter().enumerate() {
                                let _ = tree.insert_with_guard(key, i as u64, &guard);
                            }
                        })
                    })
                    .collect();

                let reader_handles: Vec<_> = (0..READERS)
                    .map(|_| {
                        let tree = Arc::clone(tree);
                        let barrier = Arc::clone(&barrier);
                        thread::spawn(move || {
                            barrier.wait();
                            let mut total = 0usize;
                            for _ in 0..SCANS_PER_READER {
                                let guard = tree.guard();
                                total += tree.scan(
                                    RangeBound::Unbounded,
                                    RangeBound::Unbounded,
                                    |_, _| true,
                                    &guard,
                                );
                            }
                            total
                        })
                    })
                    .collect();

                for h in writer_handles {
                    h.join().unwrap();
                }
                let sum: usize = reader_handles.into_iter().map(|h| h.join().unwrap()).sum();
                black_box(sum)
            });
    }

    #[divan::bench]
    fn skipmap_8b(bencher: Bencher) {
        let initial_keys = keys::<8>(INITIAL_N);
        let insert_keys = keys::<8>(INSERT_N + INITIAL_N);

        bencher
            .with_inputs(|| {
                let map = Arc::new(setup_skipmap(&initial_keys));
                let new_keys: Vec<_> = insert_keys[INITIAL_N..].to_vec();
                (map, new_keys)
            })
            .bench_refs(|(map, new_keys)| {
                let barrier = Arc::new(Barrier::new(WRITERS + READERS));

                let writer_handles: Vec<_> = (0..WRITERS)
                    .map(|w| {
                        let map = Arc::clone(map);
                        let barrier = Arc::clone(&barrier);
                        let chunk: Vec<_> = new_keys
                            .iter()
                            .skip(w * (INSERT_N / WRITERS))
                            .take(INSERT_N / WRITERS)
                            .copied()
                            .collect();
                        thread::spawn(move || {
                            barrier.wait();
                            for (i, key) in chunk.iter().enumerate() {
                                map.insert(*key, i as u64);
                            }
                        })
                    })
                    .collect();

                let reader_handles: Vec<_> = (0..READERS)
                    .map(|_| {
                        let map = Arc::clone(map);
                        let barrier = Arc::clone(&barrier);
                        thread::spawn(move || {
                            barrier.wait();
                            let mut total = 0usize;
                            for _ in 0..SCANS_PER_READER {
                                total += map.iter().count();
                            }
                            total
                        })
                    })
                    .collect();

                for h in writer_handles {
                    h.join().unwrap();
                }
                let sum: usize = reader_handles.into_iter().map(|h| h.join().unwrap()).sum();
                black_box(sum)
            });
    }

    #[divan::bench]
    fn tree_index_8b(bencher: Bencher) {
        let initial_keys = keys::<8>(INITIAL_N);
        let insert_keys = keys::<8>(INSERT_N + INITIAL_N);

        bencher
            .with_inputs(|| {
                let tree = Arc::new(setup_tree_index(&initial_keys));
                let new_keys: Vec<_> = insert_keys[INITIAL_N..].to_vec();
                (tree, new_keys)
            })
            .bench_refs(|(tree, new_keys)| {
                let barrier = Arc::new(Barrier::new(WRITERS + READERS));

                let writer_handles: Vec<_> = (0..WRITERS)
                    .map(|w| {
                        let tree = Arc::clone(tree);
                        let barrier = Arc::clone(&barrier);
                        let chunk: Vec<_> = new_keys
                            .iter()
                            .skip(w * (INSERT_N / WRITERS))
                            .take(INSERT_N / WRITERS)
                            .copied()
                            .collect();
                        thread::spawn(move || {
                            barrier.wait();
                            for (i, key) in chunk.iter().enumerate() {
                                let _ = tree.insert_sync(*key, i as u64);
                            }
                        })
                    })
                    .collect();

                let reader_handles: Vec<_> = (0..READERS)
                    .map(|_| {
                        let tree = Arc::clone(tree);
                        let barrier = Arc::clone(&barrier);
                        thread::spawn(move || {
                            barrier.wait();
                            let mut total = 0usize;
                            for _ in 0..SCANS_PER_READER {
                                let guard = sdd::Guard::new();
                                total += tree.iter(&guard).count();
                            }
                            total
                        })
                    })
                    .collect();

                for h in writer_handles {
                    h.join().unwrap();
                }
                let sum: usize = reader_handles.into_iter().map(|h| h.join().unwrap()).sum();
                black_box(sum)
            });
    }
}

// =============================================================================
// 8. MassTree-specific APIs
// =============================================================================

#[divan::bench_group(name = "08_masstree_prefix_scan")]
mod masstree_prefix_scan {
    use super::*;

    const N: usize = 50_000;
    const PREFIX_BUCKETS: u64 = 100;

    #[divan::bench]
    fn scan_prefix_16b(bencher: Bencher) {
        let keys = keys_shared_prefix::<16>(N, PREFIX_BUCKETS);
        let tree = setup_masstree24(&keys);
        let prefix = 0u64.to_be_bytes();

        bencher.bench(|| {
            let guard = tree.guard();
            let count = tree.scan_prefix(&prefix, |_, _| true, &guard);
            black_box(count)
        });
    }

    #[divan::bench]
    fn scan_prefix_32b(bencher: Bencher) {
        let keys = keys_shared_prefix::<32>(N, PREFIX_BUCKETS);
        let tree = setup_masstree24(&keys);
        let prefix = 0u64.to_be_bytes();

        bencher.bench(|| {
            let guard = tree.guard();
            let count = tree.scan_prefix(&prefix, |_, _| true, &guard);
            black_box(count)
        });
    }

    #[divan::bench]
    fn range_bounded_16b(bencher: Bencher) {
        let keys = keys_shared_prefix::<16>(N, PREFIX_BUCKETS);
        let tree = setup_masstree24(&keys);
        let start = 0u64.to_be_bytes();
        let end = 10u64.to_be_bytes();

        bencher.bench(|| {
            let guard = tree.guard();
            let count = tree.scan(
                RangeBound::Included(&start),
                RangeBound::Excluded(&end),
                |_, _| true,
                &guard,
            );
            black_box(count)
        });
    }

    #[divan::bench]
    fn range_bounded_32b(bencher: Bencher) {
        let keys = keys_shared_prefix::<32>(N, PREFIX_BUCKETS);
        let tree = setup_masstree24(&keys);
        let start = 0u64.to_be_bytes();
        let end = 10u64.to_be_bytes();

        bencher.bench(|| {
            let guard = tree.guard();
            let count = tree.scan(
                RangeBound::Included(&start),
                RangeBound::Excluded(&end),
                |_, _| true,
                &guard,
            );
            black_box(count)
        });
    }
}

// =============================================================================
// 9. Iterator Overhead (shows cost of Vec allocation)
// =============================================================================

#[divan::bench_group(name = "09_iterator_overhead")]
mod iterator_overhead {
    use super::*;

    const N: usize = 10_000;

    #[divan::bench]
    fn masstree24_scan_8b(bencher: Bencher) {
        let keys = keys::<8>(N);
        let tree = setup_masstree24(&keys);

        bencher.bench(|| {
            let guard = tree.guard();
            let count = tree.scan(
                RangeBound::Unbounded,
                RangeBound::Unbounded,
                |_, _| true,
                &guard,
            );
            black_box(count)
        });
    }

    #[divan::bench]
    fn masstree24_iter_8b(bencher: Bencher) {
        let keys = keys::<8>(N);
        let tree = setup_masstree24(&keys);

        bencher.bench(|| {
            let guard = tree.guard();
            let count = tree.iter(&guard).count();
            black_box(count)
        });
    }

    #[divan::bench]
    fn masstree24_scan_32b(bencher: Bencher) {
        let keys = keys::<32>(N);
        let tree = setup_masstree24(&keys);

        bencher.bench(|| {
            let guard = tree.guard();
            let count = tree.scan(
                RangeBound::Unbounded,
                RangeBound::Unbounded,
                |_, _| true,
                &guard,
            );
            black_box(count)
        });
    }

    #[divan::bench]
    fn masstree24_iter_32b(bencher: Bencher) {
        let keys = keys::<32>(N);
        let tree = setup_masstree24(&keys);

        bencher.bench(|| {
            let guard = tree.guard();
            let count = tree.iter(&guard).count();
            black_box(count)
        });
    }
}
