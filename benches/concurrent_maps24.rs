//! Concurrent benchmarks for MassTree24 (WIDTH=24)
//!
//! Tests concurrent write performance and variance at different thread counts.
//!
//! ## Key Metrics
//!
//! - Split frequency: WIDTH=24 has 60% more capacity, fewer splits
//! - Variance: Difference between fastest and slowest runs
//! - Scaling: Performance at high thread counts (16, 32)
//!
//! ## Running
//!
//! ```bash
//! cargo bench --bench concurrent_maps24
//! cargo bench --bench concurrent_maps24 --features mimalloc
//! ```

#![expect(clippy::unwrap_used)]
#![expect(clippy::pedantic)]
#![expect(clippy::indexing_slicing)]
#![expect(clippy::redundant_locals)]

mod bench_utils;

use bench_utils::{keys, keys_shared_prefix, uniform_indices};
use crossbeam_skiplist::SkipMap;
use divan::{Bencher, black_box};
use indexset::concurrent::map::BTreeMap as IndexSetBTreeMap;
use masstree::MassTree24;
use std::sync::Arc;
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

// =============================================================================
// 01: CONCURRENT WRITES - Disjoint Ranges (Main Variance Test)
// =============================================================================

#[divan::bench_group(name = "01_concurrent_writes_disjoint")]
mod concurrent_writes_disjoint {
    use super::*;

    const OPS_PER_THREAD: usize = 50_000;

    #[divan::bench(args = [1, 2, 4, 8, 16, 32])]
    fn masstree24(bencher: Bencher, threads: usize) {
        bencher
            .with_inputs(|| Arc::new(MassTree24::<u64>::new()))
            .bench_local_values(|tree| {
                let handles: Vec<_> = (0..threads)
                    .map(|t| {
                        let tree = Arc::clone(&tree);
                        thread::spawn(move || {
                            let guard = tree.guard();
                            let base = t * OPS_PER_THREAD;
                            for i in 0..OPS_PER_THREAD {
                                let key = ((base + i) as u64).to_be_bytes();
                                let _ = tree.insert_with_guard(&key, i as u64, &guard);
                            }
                        })
                    })
                    .collect();

                for h in handles {
                    h.join().unwrap();
                }
                tree
            });
    }

    #[divan::bench(args = [1, 2, 4, 8, 16, 32])]
    fn skipmap(bencher: Bencher, threads: usize) {
        bencher
            .with_inputs(SkipMap::<[u8; 8], u64>::new)
            .bench_local_values(|map| {
                let map = Arc::new(map);
                let handles: Vec<_> = (0..threads)
                    .map(|t| {
                        let map = Arc::clone(&map);
                        thread::spawn(move || {
                            let base = t * OPS_PER_THREAD;
                            for i in 0..OPS_PER_THREAD {
                                let key = ((base + i) as u64).to_be_bytes();
                                map.insert(key, i as u64);
                            }
                        })
                    })
                    .collect();

                for h in handles {
                    h.join().unwrap();
                }
                map
            });
    }

    #[divan::bench(args = [1, 2, 4, 8, 16, 32])]
    fn indexset(bencher: Bencher, threads: usize) {
        bencher
            .with_inputs(IndexSetBTreeMap::<[u8; 8], u64>::new)
            .bench_local_values(|map| {
                let map = Arc::new(map);
                let handles: Vec<_> = (0..threads)
                    .map(|t| {
                        let map = Arc::clone(&map);
                        thread::spawn(move || {
                            let base = t * OPS_PER_THREAD;
                            for i in 0..OPS_PER_THREAD {
                                let key = ((base + i) as u64).to_be_bytes();
                                map.insert(key, i as u64);
                            }
                        })
                    })
                    .collect();

                for h in handles {
                    h.join().unwrap();
                }
                map
            });
    }
}

// =============================================================================
// 02: CONCURRENT WRITES - Contention (Same Key Range)
// =============================================================================

#[divan::bench_group(name = "02_concurrent_writes_contention")]
mod concurrent_writes_contention {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};

    const OPS_PER_THREAD: usize = 10_000;
    const KEY_SPACE: usize = 1_000; // All threads write to same 1000 keys

    #[divan::bench(args = [1, 2, 4, 8, 16, 32])]
    fn masstree24(bencher: Bencher, threads: usize) {
        let keys = Arc::new(keys::<8>(KEY_SPACE));

        bencher
            .with_inputs(|| Arc::new(setup_masstree24::<8>(keys.as_ref())))
            .bench_local_values(|tree| {
                let counter = Arc::new(AtomicUsize::new(0));
                let handles: Vec<_> = (0..threads)
                    .map(|t| {
                        let tree = Arc::clone(&tree);
                        let keys = Arc::clone(&keys);
                        let counter = Arc::clone(&counter);
                        thread::spawn(move || {
                            let guard = tree.guard();
                            let mut state = (t as u64).wrapping_mul(0x517c_c1b7_2722_0a95);
                            for _ in 0..OPS_PER_THREAD {
                                state = state
                                    .wrapping_mul(6_364_136_223_846_793_005)
                                    .wrapping_add(1);
                                let idx = (state as usize) % keys.len();
                                let val = counter.fetch_add(1, Ordering::Relaxed) as u64;
                                let _ = tree.insert_with_guard(&keys[idx], val, &guard);
                            }
                        })
                    })
                    .collect();

                for h in handles {
                    h.join().unwrap();
                }
                tree
            });
    }

    #[divan::bench(args = [1, 2, 4, 8, 16, 32])]
    fn skipmap(bencher: Bencher, threads: usize) {
        let keys = Arc::new(keys::<8>(KEY_SPACE));

        bencher
            .with_inputs(|| Arc::new(setup_skipmap::<8>(keys.as_ref())))
            .bench_local_values(|map| {
                let counter = Arc::new(AtomicUsize::new(0));
                let handles: Vec<_> = (0..threads)
                    .map(|t| {
                        let map = Arc::clone(&map);
                        let keys = Arc::clone(&keys);
                        let counter = Arc::clone(&counter);
                        thread::spawn(move || {
                            let mut state = (t as u64).wrapping_mul(0x517c_c1b7_2722_0a95);
                            for _ in 0..OPS_PER_THREAD {
                                state = state
                                    .wrapping_mul(6_364_136_223_846_793_005)
                                    .wrapping_add(1);
                                let idx = (state as usize) % keys.len();
                                let val = counter.fetch_add(1, Ordering::Relaxed) as u64;
                                map.insert(keys[idx], val);
                            }
                        })
                    })
                    .collect();

                for h in handles {
                    h.join().unwrap();
                }
                map
            });
    }

    #[divan::bench(args = [1, 2, 4, 8, 16, 32])]
    fn indexset(bencher: Bencher, threads: usize) {
        let keys = Arc::new(keys::<8>(KEY_SPACE));

        bencher
            .with_inputs(|| Arc::new(setup_indexset::<8>(keys.as_ref())))
            .bench_local_values(|map| {
                let counter = Arc::new(AtomicUsize::new(0));
                let handles: Vec<_> = (0..threads)
                    .map(|t| {
                        let map = Arc::clone(&map);
                        let keys = Arc::clone(&keys);
                        let counter = Arc::clone(&counter);
                        thread::spawn(move || {
                            let mut state = (t as u64).wrapping_mul(0x517c_c1b7_2722_0a95);
                            for _ in 0..OPS_PER_THREAD {
                                state = state
                                    .wrapping_mul(6_364_136_223_846_793_005)
                                    .wrapping_add(1);
                                let idx = (state as usize) % keys.len();
                                let val = counter.fetch_add(1, Ordering::Relaxed) as u64;
                                map.insert(keys[idx], val);
                            }
                        })
                    })
                    .collect();

                for h in handles {
                    h.join().unwrap();
                }
                map
            });
    }
}

// =============================================================================
// 03: SINGLE-THREADED INSERT (Baseline)
// =============================================================================

#[divan::bench_group(name = "03_single_threaded_insert")]
mod single_threaded_insert {
    use super::*;

    const KEY_COUNT: usize = 100_000;

    #[divan::bench]
    fn masstree24(bencher: Bencher) {
        bencher.bench_local(|| {
            let tree = MassTree24::<u64>::new();
            {
                let guard = tree.guard();
                for i in 0..KEY_COUNT {
                    let key = (i as u64).to_be_bytes();
                    let _ = tree.insert_with_guard(&key, i as u64, &guard);
                }
            }
            black_box(tree)
        });
    }

    #[divan::bench]
    fn skipmap(bencher: Bencher) {
        bencher.bench_local(|| {
            let map = SkipMap::<[u8; 8], u64>::new();
            for i in 0..KEY_COUNT {
                let key = (i as u64).to_be_bytes();
                map.insert(key, i as u64);
            }
            black_box(map)
        });
    }

    #[divan::bench]
    fn indexset(bencher: Bencher) {
        bencher.bench_local(|| {
            let map = IndexSetBTreeMap::<[u8; 8], u64>::new();
            for i in 0..KEY_COUNT {
                let key = (i as u64).to_be_bytes();
                map.insert(key, i as u64);
            }
            black_box(map)
        });
    }
}

// =============================================================================
// 04: READ AFTER WRITE (Mixed Workload)
// =============================================================================

#[divan::bench_group(name = "04_read_after_write")]
mod read_after_write {
    use super::*;

    const KEY_COUNT: usize = 50_000;

    fn local_setup_masstree24() -> MassTree24<u64> {
        let tree = MassTree24::new();
        {
            let guard = tree.guard();
            for i in 0..KEY_COUNT {
                let key = (i as u64).to_be_bytes();
                let _ = tree.insert_with_guard(&key, i as u64, &guard);
            }
        }
        tree
    }

    fn local_setup_skipmap() -> SkipMap<[u8; 8], u64> {
        let map = SkipMap::new();
        for i in 0..KEY_COUNT {
            let key = (i as u64).to_be_bytes();
            map.insert(key, i as u64);
        }
        map
    }

    fn local_setup_indexset() -> IndexSetBTreeMap<[u8; 8], u64> {
        let map = IndexSetBTreeMap::new();
        for i in 0..KEY_COUNT {
            let key = (i as u64).to_be_bytes();
            map.insert(key, i as u64);
        }
        map
    }

    #[divan::bench(args = [1, 2, 4, 8, 16, 32])]
    fn masstree24(bencher: Bencher, threads: usize) {
        bencher
            .with_inputs(|| Arc::new(local_setup_masstree24()))
            .bench_local_values(|tree| {
                let handles: Vec<_> = (0..threads)
                    .map(|t| {
                        let tree = Arc::clone(&tree);
                        thread::spawn(move || {
                            let guard = tree.guard();
                            let ops = KEY_COUNT / threads;
                            let base = t * ops;
                            for i in 0..ops {
                                let key = ((base + i) as u64).to_be_bytes();
                                black_box(tree.get_with_guard(&key, &guard));
                            }
                        })
                    })
                    .collect();

                for h in handles {
                    h.join().unwrap();
                }
                tree
            });
    }

    #[divan::bench(args = [1, 2, 4, 8, 16, 32])]
    fn skipmap(bencher: Bencher, threads: usize) {
        bencher
            .with_inputs(|| Arc::new(local_setup_skipmap()))
            .bench_local_values(|map| {
                let handles: Vec<_> = (0..threads)
                    .map(|t| {
                        let map = Arc::clone(&map);
                        thread::spawn(move || {
                            let ops = KEY_COUNT / threads;
                            let base = t * ops;
                            for i in 0..ops {
                                let key = ((base + i) as u64).to_be_bytes();
                                black_box(map.get(&key));
                            }
                        })
                    })
                    .collect();

                for h in handles {
                    h.join().unwrap();
                }
                map
            });
    }

    #[divan::bench(args = [1, 2, 4, 8, 16, 32])]
    fn indexset(bencher: Bencher, threads: usize) {
        bencher
            .with_inputs(|| Arc::new(local_setup_indexset()))
            .bench_local_values(|map| {
                let handles: Vec<_> = (0..threads)
                    .map(|t| {
                        let map = Arc::clone(&map);
                        thread::spawn(move || {
                            let ops = KEY_COUNT / threads;
                            let base = t * ops;
                            for i in 0..ops {
                                let key = ((base + i) as u64).to_be_bytes();
                                black_box(map.get(&key));
                            }
                        })
                    })
                    .collect();

                for h in handles {
                    h.join().unwrap();
                }
                map
            });
    }
}

// =============================================================================
// 05: SINGLE-THREADED GET - Variable Key Sizes
// =============================================================================

#[divan::bench_group(name = "05_get_by_key_size")]
mod get_by_key_size {
    use super::*;

    const N: usize = 10_000;

    fn bench_masstree24<const K: usize>(bencher: Bencher) {
        let keys = keys::<K>(N);
        let tree = setup_masstree24::<K>(&keys);
        let lookup_keys = uniform_indices(N, 1000, 42);

        bencher.bench_local(|| {
            let guard = tree.guard();
            let mut sum = 0u64;
            for &idx in &lookup_keys {
                if let Some(v) = tree.get_with_guard(&keys[idx], &guard) {
                    sum += *v;
                }
            }
            black_box(sum)
        });
    }

    fn bench_skipmap<const K: usize>(bencher: Bencher) {
        let keys = keys::<K>(N);
        let map = setup_skipmap::<K>(&keys);
        let lookup_keys = uniform_indices(N, 1000, 42);

        bencher.bench_local(|| {
            let mut sum = 0u64;
            for &idx in &lookup_keys {
                if let Some(e) = map.get(&keys[idx]) {
                    sum += *e.value();
                }
            }
            black_box(sum)
        });
    }

    fn bench_indexset<const K: usize>(bencher: Bencher) {
        let keys = keys::<K>(N);
        let map = setup_indexset::<K>(&keys);
        let lookup_keys = uniform_indices(N, 1000, 42);

        bencher.bench_local(|| {
            let mut sum = 0u64;
            for &idx in &lookup_keys {
                if let Some(r) = map.get(&keys[idx]) {
                    sum += r.get().value;
                }
            }
            black_box(sum)
        });
    }

    #[divan::bench(name = "masstree24_8B")]
    fn masstree24_8b(bencher: Bencher) {
        bench_masstree24::<8>(bencher);
    }

    #[divan::bench(name = "masstree24_16B")]
    fn masstree24_16b(bencher: Bencher) {
        bench_masstree24::<16>(bencher);
    }

    #[divan::bench(name = "masstree24_24B")]
    fn masstree24_24b(bencher: Bencher) {
        bench_masstree24::<24>(bencher);
    }

    #[divan::bench(name = "masstree24_32B")]
    fn masstree24_32b(bencher: Bencher) {
        bench_masstree24::<32>(bencher);
    }

    #[divan::bench(name = "skipmap_8B")]
    fn skipmap_8b(bencher: Bencher) {
        bench_skipmap::<8>(bencher);
    }

    #[divan::bench(name = "skipmap_16B")]
    fn skipmap_16b(bencher: Bencher) {
        bench_skipmap::<16>(bencher);
    }

    #[divan::bench(name = "skipmap_24B")]
    fn skipmap_24b(bencher: Bencher) {
        bench_skipmap::<24>(bencher);
    }

    #[divan::bench(name = "skipmap_32B")]
    fn skipmap_32b(bencher: Bencher) {
        bench_skipmap::<32>(bencher);
    }

    #[divan::bench(name = "indexset_8B")]
    fn indexset_8b(bencher: Bencher) {
        bench_indexset::<8>(bencher);
    }

    #[divan::bench(name = "indexset_16B")]
    fn indexset_16b(bencher: Bencher) {
        bench_indexset::<16>(bencher);
    }

    #[divan::bench(name = "indexset_24B")]
    fn indexset_24b(bencher: Bencher) {
        bench_indexset::<24>(bencher);
    }

    #[divan::bench(name = "indexset_32B")]
    fn indexset_32b(bencher: Bencher) {
        bench_indexset::<32>(bencher);
    }
}

// =============================================================================
// 06: SINGLE-THREADED INSERT - Variable Key Sizes
// =============================================================================

#[divan::bench_group(name = "06_insert_by_key_size")]
mod insert_by_key_size {
    use super::*;

    const N: usize = 1000;

    fn bench_masstree24<const K: usize>(bencher: Bencher) {
        let keys = keys::<K>(N);
        bencher
            .with_inputs(|| keys.clone())
            .bench_local_values(|keys| {
                let tree = MassTree24::<u64>::new();
                {
                    let guard = tree.guard();
                    for (i, key) in keys.iter().enumerate() {
                        let _ = tree.insert_with_guard(key, i as u64, &guard);
                    }
                }
                tree
            });
    }

    fn bench_skipmap<const K: usize>(bencher: Bencher) {
        let keys = keys::<K>(N);
        bencher
            .with_inputs(|| keys.clone())
            .bench_local_values(|keys| {
                let map = SkipMap::<[u8; K], u64>::new();
                for (i, key) in keys.iter().enumerate() {
                    map.insert(*key, i as u64);
                }
                map
            });
    }

    fn bench_indexset<const K: usize>(bencher: Bencher) {
        let keys = keys::<K>(N);
        bencher
            .with_inputs(|| keys.clone())
            .bench_local_values(|keys| {
                let map = IndexSetBTreeMap::<[u8; K], u64>::new();
                for (i, key) in keys.iter().enumerate() {
                    map.insert(*key, i as u64);
                }
                map
            });
    }

    #[divan::bench(name = "masstree24_8B")]
    fn masstree24_8b(bencher: Bencher) {
        bench_masstree24::<8>(bencher);
    }

    #[divan::bench(name = "masstree24_16B")]
    fn masstree24_16b(bencher: Bencher) {
        bench_masstree24::<16>(bencher);
    }

    #[divan::bench(name = "masstree24_24B")]
    fn masstree24_24b(bencher: Bencher) {
        bench_masstree24::<24>(bencher);
    }

    #[divan::bench(name = "masstree24_32B")]
    fn masstree24_32b(bencher: Bencher) {
        bench_masstree24::<32>(bencher);
    }

    #[divan::bench(name = "skipmap_8B")]
    fn skipmap_8b(bencher: Bencher) {
        bench_skipmap::<8>(bencher);
    }

    #[divan::bench(name = "skipmap_16B")]
    fn skipmap_16b(bencher: Bencher) {
        bench_skipmap::<16>(bencher);
    }

    #[divan::bench(name = "skipmap_24B")]
    fn skipmap_24b(bencher: Bencher) {
        bench_skipmap::<24>(bencher);
    }

    #[divan::bench(name = "skipmap_32B")]
    fn skipmap_32b(bencher: Bencher) {
        bench_skipmap::<32>(bencher);
    }

    #[divan::bench(name = "indexset_8B")]
    fn indexset_8b(bencher: Bencher) {
        bench_indexset::<8>(bencher);
    }

    #[divan::bench(name = "indexset_16B")]
    fn indexset_16b(bencher: Bencher) {
        bench_indexset::<16>(bencher);
    }

    #[divan::bench(name = "indexset_24B")]
    fn indexset_24b(bencher: Bencher) {
        bench_indexset::<24>(bencher);
    }

    #[divan::bench(name = "indexset_32B")]
    fn indexset_32b(bencher: Bencher) {
        bench_indexset::<32>(bencher);
    }
}

// =============================================================================
// 07: CONCURRENT READS - Thread Scaling (8-byte keys)
// =============================================================================

#[divan::bench_group(name = "07_concurrent_reads_scaling")]
mod concurrent_reads_scaling {
    use super::*;

    const N: usize = 10_000_000;
    const OPS_PER_THREAD: usize = 50_000;

    #[divan::bench(args = [1, 2, 4, 8, 16, 32])]
    fn masstree24(bencher: Bencher, threads: usize) {
        let keys = Arc::new(keys::<8>(N));
        let tree = Arc::new(setup_masstree24::<8>(keys.as_ref()));
        let indices = Arc::new(uniform_indices(N, OPS_PER_THREAD, 42));

        bencher.bench_local(|| {
            let handles: Vec<_> = (0..threads)
                .map(|t| {
                    let tree = Arc::clone(&tree);
                    let keys = Arc::clone(&keys);
                    let indices = Arc::clone(&indices);
                    thread::spawn(move || {
                        let guard = tree.guard();
                        let mut sum = 0u64;
                        let offset = t * 7919; // Prime offset per thread
                        for i in 0..OPS_PER_THREAD {
                            let idx = indices[(i + offset) % indices.len()];
                            if let Some(v) = tree.get_with_guard(&keys[idx], &guard) {
                                sum += *v;
                            }
                        }
                        black_box(sum);
                    })
                })
                .collect();

            for h in handles {
                h.join().unwrap();
            }
        });
    }

    #[divan::bench(args = [1, 2, 4, 8, 16, 32])]
    fn skipmap(bencher: Bencher, threads: usize) {
        let keys = Arc::new(keys::<8>(N));
        let map = Arc::new(setup_skipmap::<8>(keys.as_ref()));
        let indices = Arc::new(uniform_indices(N, OPS_PER_THREAD, 42));

        bencher.bench_local(|| {
            let handles: Vec<_> = (0..threads)
                .map(|t| {
                    let map = Arc::clone(&map);
                    let keys = Arc::clone(&keys);
                    let indices = Arc::clone(&indices);
                    thread::spawn(move || {
                        let mut sum = 0u64;
                        let offset = t * 7919;
                        for i in 0..OPS_PER_THREAD {
                            let idx = indices[(i + offset) % indices.len()];
                            if let Some(e) = map.get(&keys[idx]) {
                                sum += *e.value();
                            }
                        }
                        black_box(sum);
                    })
                })
                .collect();

            for h in handles {
                h.join().unwrap();
            }
        });
    }

    #[divan::bench(args = [1, 2, 4, 8, 16, 32])]
    fn indexset(bencher: Bencher, threads: usize) {
        let keys = Arc::new(keys::<8>(N));
        let map = Arc::new(setup_indexset::<8>(keys.as_ref()));
        let indices = Arc::new(uniform_indices(N, OPS_PER_THREAD, 42));

        bencher.bench_local(|| {
            let handles: Vec<_> = (0..threads)
                .map(|t| {
                    let map = Arc::clone(&map);
                    let keys = Arc::clone(&keys);
                    let indices = Arc::clone(&indices);
                    thread::spawn(move || {
                        let mut sum = 0u64;
                        let offset = t * 7919;
                        for i in 0..OPS_PER_THREAD {
                            let idx = indices[(i + offset) % indices.len()];
                            if let Some(r) = map.get(&keys[idx]) {
                                sum += r.get().value;
                            }
                        }
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
// 08: CONCURRENT READS - Long Keys (32-byte)
// =============================================================================

#[divan::bench_group(name = "08_concurrent_reads_long_keys")]
mod concurrent_reads_long_keys {
    use super::*;

    const N: usize = 10_000_000;
    const OPS_PER_THREAD: usize = 50_000;

    #[divan::bench(args = [1, 2, 4, 8, 16, 32])]
    fn masstree24_32b(bencher: Bencher, threads: usize) {
        let keys = Arc::new(keys::<32>(N));
        let tree = Arc::new(setup_masstree24::<32>(keys.as_ref()));
        let indices = Arc::new(uniform_indices(N, OPS_PER_THREAD, 42));

        bencher.bench_local(|| {
            let handles: Vec<_> = (0..threads)
                .map(|t| {
                    let tree = Arc::clone(&tree);
                    let keys = Arc::clone(&keys);
                    let indices = Arc::clone(&indices);
                    thread::spawn(move || {
                        let guard = tree.guard();
                        let mut sum = 0u64;
                        let offset = t * 7919;
                        for i in 0..OPS_PER_THREAD {
                            let idx = indices[(i + offset) % indices.len()];
                            if let Some(v) = tree.get_with_guard(&keys[idx], &guard) {
                                sum += *v;
                            }
                        }
                        black_box(sum);
                    })
                })
                .collect();

            for h in handles {
                h.join().unwrap();
            }
        });
    }

    #[divan::bench(args = [1, 2, 4, 8, 16, 32])]
    fn skipmap_32b(bencher: Bencher, threads: usize) {
        let keys = Arc::new(keys::<32>(N));
        let map = Arc::new(setup_skipmap::<32>(keys.as_ref()));
        let indices = Arc::new(uniform_indices(N, OPS_PER_THREAD, 42));

        bencher.bench_local(|| {
            let handles: Vec<_> = (0..threads)
                .map(|t| {
                    let map = Arc::clone(&map);
                    let keys = Arc::clone(&keys);
                    let indices = Arc::clone(&indices);
                    thread::spawn(move || {
                        let mut sum = 0u64;
                        let offset = t * 7919;
                        for i in 0..OPS_PER_THREAD {
                            let idx = indices[(i + offset) % indices.len()];
                            if let Some(e) = map.get(&keys[idx]) {
                                sum += *e.value();
                            }
                        }
                        black_box(sum);
                    })
                })
                .collect();

            for h in handles {
                h.join().unwrap();
            }
        });
    }

    #[divan::bench(args = [1, 2, 4, 8, 16, 32])]
    fn indexset_32b(bencher: Bencher, threads: usize) {
        let keys = Arc::new(keys::<32>(N));
        let map = Arc::new(setup_indexset::<32>(keys.as_ref()));
        let indices = Arc::new(uniform_indices(N, OPS_PER_THREAD, 42));

        bencher.bench_local(|| {
            let handles: Vec<_> = (0..threads)
                .map(|t| {
                    let map = Arc::clone(&map);
                    let keys = Arc::clone(&keys);
                    let indices = Arc::clone(&indices);
                    thread::spawn(move || {
                        let mut sum = 0u64;
                        let offset = t * 7919;
                        for i in 0..OPS_PER_THREAD {
                            let idx = indices[(i + offset) % indices.len()];
                            if let Some(r) = map.get(&keys[idx]) {
                                sum += r.get().value;
                            }
                        }
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
// 09: MIXED WORKLOAD - Uniform Random (No Hot Keys)
// =============================================================================

#[divan::bench_group(name = "09_mixed_uniform")]
mod mixed_uniform {
    use super::*;

    const N: usize = 100_000;
    const OPS_PER_THREAD: usize = 10_000;
    const WRITE_RATIO: usize = 10; // 10% writes

    #[divan::bench(args = [1, 2, 4, 8, 16, 32])]
    fn masstree24(bencher: Bencher, threads: usize) {
        let keys = Arc::new(keys::<8>(N));
        let indices = Arc::new(uniform_indices(N, OPS_PER_THREAD, 42));

        bencher
            .with_inputs(|| Arc::new(setup_masstree24::<8>(keys.as_ref())))
            .bench_local_values(|tree| {
                let handles: Vec<_> = (0..threads)
                    .map(|t| {
                        let tree = Arc::clone(&tree);
                        let keys = Arc::clone(&keys);
                        let indices = Arc::clone(&indices);
                        thread::spawn(move || {
                            let guard = tree.guard();
                            let mut sum = 0u64;
                            let offset = t * 7919;

                            for i in 0..OPS_PER_THREAD {
                                let idx = indices[(i + offset) % indices.len()];

                                if i % WRITE_RATIO == 0 {
                                    let _ = tree.insert_with_guard(&keys[idx], i as u64, &guard);
                                } else if let Some(v) = tree.get_with_guard(&keys[idx], &guard) {
                                    sum += *v;
                                }
                            }
                            black_box(sum);
                        })
                    })
                    .collect();

                for h in handles {
                    h.join().unwrap();
                }
                tree
            });
    }
}

// Uncomment skipmap/indexset for comparison if needed
// #[divan::bench(args = [1, 2, 4, 8, 16, 32])]
// fn skipmap(bencher: Bencher, threads: usize) {
//         let keys = Arc::new(keys::<8>(N));
//         let indices = Arc::new(uniform_indices(N, OPS_PER_THREAD, 42));
//
//         bencher
//             .with_inputs(|| Arc::new(setup_skipmap::<8>(keys.as_ref())))
//             .bench_local_values(|map| {
//                 let handles: Vec<_> = (0..threads)
//                     .map(|t| {
//                         let map = Arc::clone(&map);
//                         let keys = Arc::clone(&keys);
//                         let indices = Arc::clone(&indices);
//                         thread::spawn(move || {
//                             let mut sum = 0u64;
//                             let offset = t * 7919;
//
//                             for i in 0..OPS_PER_THREAD {
//                                 let idx = indices[(i + offset) % indices.len()];
//
//                                 if i % WRITE_RATIO == 0 {
//                                     map.insert(keys[idx], i as u64);
//                                 } else if let Some(e) = map.get(&keys[idx]) {
//                                     sum += *e.value();
//                                 }
//                             }
//                             black_box(sum);
//                         })
//                     })
//                     .collect();
//
//                 for h in handles {
//                     h.join().unwrap();
//                 }
//                 map
//             });
//     }
//
//     #[divan::bench(args = [1, 2, 4, 8, 16, 32])]
//     fn indexset(bencher: Bencher, threads: usize) {
//         let keys = Arc::new(keys::<8>(N));
//         let indices = Arc::new(uniform_indices(N, OPS_PER_THREAD, 42));
//
//         bencher
//             .with_inputs(|| Arc::new(setup_indexset::<8>(keys.as_ref())))
//             .bench_local_values(|map| {
//                 let handles: Vec<_> = (0..threads)
//                     .map(|t| {
//                         let map = Arc::clone(&map);
//                         let keys = Arc::clone(&keys);
//                         let indices = Arc::clone(&indices);
//                         thread::spawn(move || {
//                             let mut sum = 0u64;
//                             let offset = t * 7919;
//
//                             for i in 0..OPS_PER_THREAD {
//                                 let idx = indices[(i + offset) % indices.len()];
//
//                                 if i % WRITE_RATIO == 0 {
//                                     map.insert(keys[idx], i as u64);
//                                 } else if let Some(r) = map.get(&keys[idx]) {
//                                     sum += r.get().value;
//                                 }
//                             }
//                             black_box(sum);
//                         })
//                     })
//                     .collect();
//
//                 for h in handles {
//                     h.join().unwrap();
//                 }
//                 map
//             });
//     }
// }

// =============================================================================
// 10a: READ SCALING - Throughput (8-byte keys)
// =============================================================================

#[divan::bench_group(name = "10a_read_scaling_8B")]
mod read_scaling_8b {
    use super::*;

    const N: usize = 10_000_000;
    const OPS_PER_THREAD: usize = 50_000;

    #[divan::bench(args = [1, 2, 4, 8, 16, 32])]
    fn masstree24(bencher: Bencher, threads: usize) {
        let keys = Arc::new(keys::<8>(N));
        let tree = Arc::new(setup_masstree24::<8>(keys.as_ref()));

        bencher
            .counter(divan::counter::ItemsCount::new(threads * OPS_PER_THREAD))
            .bench_local(|| {
                let handles: Vec<_> = (0..threads)
                    .map(|t| {
                        let tree = Arc::clone(&tree);
                        let keys = Arc::clone(&keys);
                        thread::spawn(move || {
                            let guard = tree.guard();
                            let mut sum = 0u64;
                            let start = (t * 7919) % keys.len();
                            for i in 0..OPS_PER_THREAD {
                                let idx = (start + i) % keys.len();
                                if let Some(v) = tree.get_with_guard(&keys[idx], &guard) {
                                    sum += *v;
                                }
                            }
                            black_box(sum);
                        })
                    })
                    .collect();

                for h in handles {
                    h.join().unwrap();
                }
            });
    }

    #[divan::bench(args = [1, 2, 4, 8, 16, 32])]
    fn skipmap(bencher: Bencher, threads: usize) {
        let keys = Arc::new(keys::<8>(N));
        let map = Arc::new(setup_skipmap::<8>(keys.as_ref()));

        bencher
            .counter(divan::counter::ItemsCount::new(threads * OPS_PER_THREAD))
            .bench_local(|| {
                let handles: Vec<_> = (0..threads)
                    .map(|t| {
                        let map = Arc::clone(&map);
                        let keys = Arc::clone(&keys);
                        thread::spawn(move || {
                            let mut sum = 0u64;
                            let start = (t * 7919) % keys.len();
                            for i in 0..OPS_PER_THREAD {
                                let idx = (start + i) % keys.len();
                                if let Some(e) = map.get(&keys[idx]) {
                                    sum += *e.value();
                                }
                            }
                            black_box(sum);
                        })
                    })
                    .collect();

                for h in handles {
                    h.join().unwrap();
                }
            });
    }

    #[divan::bench(args = [1, 2, 4, 8, 16, 32])]
    fn indexset(bencher: Bencher, threads: usize) {
        let keys = Arc::new(keys::<8>(N));
        let map = Arc::new(setup_indexset::<8>(keys.as_ref()));

        bencher
            .counter(divan::counter::ItemsCount::new(threads * OPS_PER_THREAD))
            .bench_local(|| {
                let handles: Vec<_> = (0..threads)
                    .map(|t| {
                        let map = Arc::clone(&map);
                        let keys = Arc::clone(&keys);
                        thread::spawn(move || {
                            let mut sum = 0u64;
                            let start = (t * 7919) % keys.len();
                            for i in 0..OPS_PER_THREAD {
                                let idx = (start + i) % keys.len();
                                if let Some(r) = map.get(&keys[idx]) {
                                    sum += r.get().value;
                                }
                            }
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
// 10b: READ SCALING - Throughput (32-byte keys, multi-layer)
// =============================================================================

#[divan::bench_group(name = "10b_read_scaling_32B")]
mod read_scaling_32b {
    use super::*;

    const N: usize = 10_000_000;
    const OPS_PER_THREAD: usize = 50_000;

    #[divan::bench(args = [1, 2, 4, 8, 16, 32])]
    fn masstree24(bencher: Bencher, threads: usize) {
        let keys = Arc::new(keys::<32>(N));
        let tree = Arc::new(setup_masstree24::<32>(keys.as_ref()));

        bencher
            .counter(divan::counter::ItemsCount::new(threads * OPS_PER_THREAD))
            .bench_local(|| {
                let handles: Vec<_> = (0..threads)
                    .map(|t| {
                        let tree = Arc::clone(&tree);
                        let keys = Arc::clone(&keys);
                        thread::spawn(move || {
                            let guard = tree.guard();
                            let mut sum = 0u64;
                            let start = (t * 7919) % keys.len();
                            for i in 0..OPS_PER_THREAD {
                                let idx = (start + i) % keys.len();
                                if let Some(v) = tree.get_with_guard(&keys[idx], &guard) {
                                    sum += *v;
                                }
                            }
                            black_box(sum);
                        })
                    })
                    .collect();

                for h in handles {
                    h.join().unwrap();
                }
            });
    }

    #[divan::bench(args = [1, 2, 4, 8, 16, 32])]
    fn skipmap(bencher: Bencher, threads: usize) {
        let keys = Arc::new(keys::<32>(N));
        let map = Arc::new(setup_skipmap::<32>(keys.as_ref()));

        bencher
            .counter(divan::counter::ItemsCount::new(threads * OPS_PER_THREAD))
            .bench_local(|| {
                let handles: Vec<_> = (0..threads)
                    .map(|t| {
                        let map = Arc::clone(&map);
                        let keys = Arc::clone(&keys);
                        thread::spawn(move || {
                            let mut sum = 0u64;
                            let start = (t * 7919) % keys.len();
                            for i in 0..OPS_PER_THREAD {
                                let idx = (start + i) % keys.len();
                                if let Some(e) = map.get(&keys[idx]) {
                                    sum += *e.value();
                                }
                            }
                            black_box(sum);
                        })
                    })
                    .collect();

                for h in handles {
                    h.join().unwrap();
                }
            });
    }

    #[divan::bench(args = [1, 2, 4, 8, 16, 32])]
    fn indexset(bencher: Bencher, threads: usize) {
        let keys = Arc::new(keys::<32>(N));
        let map = Arc::new(setup_indexset::<32>(keys.as_ref()));

        bencher
            .counter(divan::counter::ItemsCount::new(threads * OPS_PER_THREAD))
            .bench_local(|| {
                let handles: Vec<_> = (0..threads)
                    .map(|t| {
                        let map = Arc::clone(&map);
                        let keys = Arc::clone(&keys);
                        thread::spawn(move || {
                            let mut sum = 0u64;
                            let start = (t * 7919) % keys.len();
                            for i in 0..OPS_PER_THREAD {
                                let idx = (start + i) % keys.len();
                                if let Some(r) = map.get(&keys[idx]) {
                                    sum += r.get().value;
                                }
                            }
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
// 10c: WRITE SCALING - Throughput (32-byte keys)
// =============================================================================

#[divan::bench_group(name = "10c_write_scaling_32B")]
mod write_scaling_32b {
    use super::*;

    const N: usize = 100_000;
    const OPS_PER_THREAD: usize = 10_000;

    #[divan::bench(args = [1, 2, 4, 8, 16, 32])]
    fn masstree24(bencher: Bencher, threads: usize) {
        let keys = Arc::new(keys::<32>(N));

        bencher
            .counter(divan::counter::ItemsCount::new(threads * OPS_PER_THREAD))
            .with_inputs(|| {
                let tree = MassTree24::<u64>::new();
                // Pre-populate with half the keys
                {
                    let guard = tree.guard();
                    for (i, key) in keys.iter().take(N / 2).enumerate() {
                        let _ = tree.insert_with_guard(key, i as u64, &guard);
                    }
                }
                Arc::new(tree)
            })
            .bench_local_values(|tree| {
                let handles: Vec<_> = (0..threads)
                    .map(|t| {
                        let tree = Arc::clone(&tree);
                        let keys = Arc::clone(&keys);
                        thread::spawn(move || {
                            let guard = tree.guard();
                            let start = (t * 7919) % keys.len();
                            for i in 0..OPS_PER_THREAD {
                                let idx = (start + i) % keys.len();
                                let _ = tree.insert_with_guard(&keys[idx], i as u64, &guard);
                            }
                        })
                    })
                    .collect();

                for h in handles {
                    h.join().unwrap();
                }
                tree
            });
    }

    #[divan::bench(args = [1, 2, 4, 8, 16, 32])]
    fn skipmap(bencher: Bencher, threads: usize) {
        let keys = Arc::new(keys::<32>(N));

        bencher
            .counter(divan::counter::ItemsCount::new(threads * OPS_PER_THREAD))
            .with_inputs(|| {
                let map = SkipMap::<[u8; 32], u64>::new();
                for (i, key) in keys.iter().take(N / 2).enumerate() {
                    map.insert(*key, i as u64);
                }
                Arc::new(map)
            })
            .bench_local_values(|map| {
                let handles: Vec<_> = (0..threads)
                    .map(|t| {
                        let map = Arc::clone(&map);
                        let keys = Arc::clone(&keys);
                        thread::spawn(move || {
                            let start = (t * 7919) % keys.len();
                            for i in 0..OPS_PER_THREAD {
                                let idx = (start + i) % keys.len();
                                map.insert(keys[idx], i as u64);
                            }
                        })
                    })
                    .collect();

                for h in handles {
                    h.join().unwrap();
                }
                map
            });
    }

    #[divan::bench(args = [1, 2, 4, 8, 16, 32])]
    fn indexset(bencher: Bencher, threads: usize) {
        let keys = Arc::new(keys::<32>(N));

        bencher
            .counter(divan::counter::ItemsCount::new(threads * OPS_PER_THREAD))
            .with_inputs(|| {
                let map = IndexSetBTreeMap::<[u8; 32], u64>::new();
                for (i, key) in keys.iter().take(N / 2).enumerate() {
                    map.insert(*key, i as u64);
                }
                Arc::new(map)
            })
            .bench_local_values(|map| {
                let handles: Vec<_> = (0..threads)
                    .map(|t| {
                        let map = Arc::clone(&map);
                        let keys = Arc::clone(&keys);
                        thread::spawn(move || {
                            let start = (t * 7919) % keys.len();
                            for i in 0..OPS_PER_THREAD {
                                let idx = (start + i) % keys.len();
                                map.insert(keys[idx], i as u64);
                            }
                        })
                    })
                    .collect();

                for h in handles {
                    h.join().unwrap();
                }
                map
            });
    }
}

// =============================================================================
// 11: SINGLE HOT KEY - Maximum Contention
// =============================================================================

#[divan::bench_group(name = "11_single_hot_key")]
mod single_hot_key {
    use super::*;

    const N: usize = 100_000; // Reduced from 10M for faster setup
    const OPS_PER_THREAD: usize = 10_000; // Reduced from 50k

    #[divan::bench(args = [2, 4, 8, 16, 32])]
    fn masstree24(bencher: Bencher, threads: usize) {
        let keys = keys::<8>(N);
        let hot_key = keys[N / 2]; // Single hot key

        bencher
            .with_inputs(|| Arc::new(setup_masstree24::<8>(&keys)))
            .bench_local_values(|tree| {
                let handles: Vec<_> = (0..threads)
                    .map(|t| {
                        let tree = Arc::clone(&tree);
                        let hot_key = hot_key;
                        thread::spawn(move || {
                            let guard = tree.guard();
                            let mut sum = 0u64;

                            for i in 0..OPS_PER_THREAD {
                                if i % 10 == 0 {
                                    let _ = tree.insert_with_guard(
                                        &hot_key,
                                        (t * OPS_PER_THREAD + i) as u64,
                                        &guard,
                                    );
                                } else if let Some(v) = tree.get_with_guard(&hot_key, &guard) {
                                    sum += *v;
                                }
                            }
                            black_box(sum);
                        })
                    })
                    .collect();

                for h in handles {
                    h.join().unwrap();
                }
                tree
            });
    }

    #[divan::bench(args = [2, 4, 8, 16, 32])]
    fn skipmap(bencher: Bencher, threads: usize) {
        let keys = keys::<8>(N);
        let hot_key = keys[N / 2];

        bencher
            .with_inputs(|| Arc::new(setup_skipmap::<8>(&keys)))
            .bench_local_values(|map| {
                let handles: Vec<_> = (0..threads)
                    .map(|t| {
                        let map = Arc::clone(&map);
                        let hot_key = hot_key;
                        thread::spawn(move || {
                            let mut sum = 0u64;

                            for i in 0..OPS_PER_THREAD {
                                if i % 10 == 0 {
                                    map.insert(hot_key, (t * OPS_PER_THREAD + i) as u64);
                                } else if let Some(e) = map.get(&hot_key) {
                                    sum += *e.value();
                                }
                            }
                            black_box(sum);
                        })
                    })
                    .collect();

                for h in handles {
                    h.join().unwrap();
                }
                map
            });
    }

    #[divan::bench(args = [2, 4, 8, 16, 32])]
    fn indexset(bencher: Bencher, threads: usize) {
        let keys = keys::<8>(N);
        let hot_key = keys[N / 2];

        bencher
            .with_inputs(|| Arc::new(setup_indexset::<8>(&keys)))
            .bench_local_values(|map| {
                let handles: Vec<_> = (0..threads)
                    .map(|t| {
                        let map = Arc::clone(&map);
                        let hot_key = hot_key;
                        thread::spawn(move || {
                            let mut sum = 0u64;

                            for i in 0..OPS_PER_THREAD {
                                if i % 10 == 0 {
                                    map.insert(hot_key, (t * OPS_PER_THREAD + i) as u64);
                                } else if let Some(r) = map.get(&hot_key) {
                                    sum += r.get().value;
                                }
                            }
                            black_box(sum);
                        })
                    })
                    .collect();

                for h in handles {
                    h.join().unwrap();
                }
                map
            });
    }
}

// =============================================================================
// 12: SINGLE-THREADED GET - Shared Prefix Keys (Forces Layering)
// =============================================================================

#[divan::bench_group(name = "12_get_by_key_size_shared_prefix")]
mod get_by_key_size_shared_prefix {
    use super::*;

    const N: usize = 10_000;
    const PREFIX_BUCKETS: u64 = 256; // smaller => more shared prefixes

    fn bench_masstree24<const K: usize>(bencher: Bencher) {
        let keys = keys_shared_prefix::<K>(N, PREFIX_BUCKETS);
        let tree = setup_masstree24::<K>(&keys);
        let lookup_keys = uniform_indices(N, 1000, 42);

        bencher.bench_local(|| {
            let guard = tree.guard();
            let mut sum = 0u64;
            for &idx in &lookup_keys {
                if let Some(v) = tree.get_with_guard(&keys[idx], &guard) {
                    sum += *v;
                }
            }
            black_box(sum)
        });
    }

    fn bench_skipmap<const K: usize>(bencher: Bencher) {
        let keys = keys_shared_prefix::<K>(N, PREFIX_BUCKETS);
        let map = setup_skipmap::<K>(&keys);
        let lookup_keys = uniform_indices(N, 1000, 42);

        bencher.bench_local(|| {
            let mut sum = 0u64;
            for &idx in &lookup_keys {
                if let Some(e) = map.get(&keys[idx]) {
                    sum += *e.value();
                }
            }
            black_box(sum)
        });
    }

    fn bench_indexset<const K: usize>(bencher: Bencher) {
        let keys = keys_shared_prefix::<K>(N, PREFIX_BUCKETS);
        let map = setup_indexset::<K>(&keys);
        let lookup_keys = uniform_indices(N, 1000, 42);

        bencher.bench_local(|| {
            let mut sum = 0u64;
            for &idx in &lookup_keys {
                if let Some(r) = map.get(&keys[idx]) {
                    sum += r.get().value;
                }
            }
            black_box(sum)
        });
    }

    #[divan::bench(name = "masstree24_16B")]
    fn masstree24_16b(bencher: Bencher) {
        bench_masstree24::<16>(bencher);
    }

    #[divan::bench(name = "masstree24_24B")]
    fn masstree24_24b(bencher: Bencher) {
        bench_masstree24::<24>(bencher);
    }

    #[divan::bench(name = "masstree24_32B")]
    fn masstree24_32b(bencher: Bencher) {
        bench_masstree24::<32>(bencher);
    }

    #[divan::bench(name = "skipmap_16B")]
    fn skipmap_16b(bencher: Bencher) {
        bench_skipmap::<16>(bencher);
    }

    #[divan::bench(name = "skipmap_24B")]
    fn skipmap_24b(bencher: Bencher) {
        bench_skipmap::<24>(bencher);
    }

    #[divan::bench(name = "skipmap_32B")]
    fn skipmap_32b(bencher: Bencher) {
        bench_skipmap::<32>(bencher);
    }

    #[divan::bench(name = "indexset_16B")]
    fn indexset_16b(bencher: Bencher) {
        bench_indexset::<16>(bencher);
    }

    #[divan::bench(name = "indexset_24B")]
    fn indexset_24b(bencher: Bencher) {
        bench_indexset::<24>(bencher);
    }

    #[divan::bench(name = "indexset_32B")]
    fn indexset_32b(bencher: Bencher) {
        bench_indexset::<32>(bencher);
    }
}

// =============================================================================
// 13: CONCURRENT READS - Shared Prefix Long Keys (32-byte)
// =============================================================================

#[divan::bench_group(name = "13_concurrent_reads_long_keys_shared_prefix")]
mod concurrent_reads_long_keys_shared_prefix {
    use super::*;

    const N: usize = 10_000_000;
    const OPS_PER_THREAD: usize = 50_000;
    const PREFIX_BUCKETS: u64 = 256;

    #[divan::bench(args = [1, 2, 4, 8, 16, 32])]
    fn masstree24_32b(bencher: Bencher, threads: usize) {
        let keys = Arc::new(keys_shared_prefix::<32>(N, PREFIX_BUCKETS));
        let tree = Arc::new(setup_masstree24::<32>(keys.as_ref()));
        let indices = Arc::new(uniform_indices(N, OPS_PER_THREAD, 42));

        bencher.bench_local(|| {
            let handles: Vec<_> = (0..threads)
                .map(|t| {
                    let tree = Arc::clone(&tree);
                    let keys = Arc::clone(&keys);
                    let indices = Arc::clone(&indices);
                    thread::spawn(move || {
                        let guard = tree.guard();
                        let mut sum = 0u64;
                        let offset = t * 7919;
                        for i in 0..OPS_PER_THREAD {
                            let idx = indices[(i + offset) % indices.len()];
                            if let Some(v) = tree.get_with_guard(&keys[idx], &guard) {
                                sum += *v;
                            }
                        }
                        black_box(sum);
                    })
                })
                .collect();

            for h in handles {
                h.join().unwrap();
            }
        });
    }

    #[divan::bench(args = [1, 2, 4, 8, 16, 32])]
    fn skipmap_32b(bencher: Bencher, threads: usize) {
        let keys = Arc::new(keys_shared_prefix::<32>(N, PREFIX_BUCKETS));
        let map = Arc::new(setup_skipmap::<32>(keys.as_ref()));
        let indices = Arc::new(uniform_indices(N, OPS_PER_THREAD, 42));

        bencher.bench_local(|| {
            let handles: Vec<_> = (0..threads)
                .map(|t| {
                    let map = Arc::clone(&map);
                    let keys = Arc::clone(&keys);
                    let indices = Arc::clone(&indices);
                    thread::spawn(move || {
                        let mut sum = 0u64;
                        let offset = t * 7919;
                        for i in 0..OPS_PER_THREAD {
                            let idx = indices[(i + offset) % indices.len()];
                            if let Some(e) = map.get(&keys[idx]) {
                                sum += *e.value();
                            }
                        }
                        black_box(sum);
                    })
                })
                .collect();

            for h in handles {
                h.join().unwrap();
            }
        });
    }

    #[divan::bench(args = [1, 2, 4, 8, 16, 32])]
    fn indexset_32b(bencher: Bencher, threads: usize) {
        let keys = Arc::new(keys_shared_prefix::<32>(N, PREFIX_BUCKETS));
        let map = Arc::new(setup_indexset::<32>(keys.as_ref()));
        let indices = Arc::new(uniform_indices(N, OPS_PER_THREAD, 42));

        bencher.bench_local(|| {
            let handles: Vec<_> = (0..threads)
                .map(|t| {
                    let map = Arc::clone(&map);
                    let keys = Arc::clone(&keys);
                    let indices = Arc::clone(&indices);
                    thread::spawn(move || {
                        let mut sum = 0u64;
                        let offset = t * 7919;
                        for i in 0..OPS_PER_THREAD {
                            let idx = indices[(i + offset) % indices.len()];
                            if let Some(r) = map.get(&keys[idx]) {
                                sum += r.get().value;
                            }
                        }
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
