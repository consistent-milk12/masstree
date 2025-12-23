//! Comparison benchmarks: Concurrent Map Implementations
//!
//! Compares `MassTree` against other concurrent map crates:
//! - `crossbeam-skiplist::SkipMap` - Lock-free skip list (ordered, truly concurrent)
//! - `indexset::concurrent::map::BTreeMap` - Concurrent B-tree (ordered)
//! - `dashmap::DashMap` - Sharded hash map (unordered, for scaling comparison)
//!
//! Note: DashMap is unordered but included to understand concurrent scaling characteristics.
//!
//! ## Benchmark Design Philosophy
//!
//! These benchmarks aim for objectivity by testing:
//! - **Variable key sizes**: 8, 16, 24, 32 bytes (`MassTree` optimizes for ≤8)
//! - **Realistic access patterns**: Zipfian distribution (hot keys), uniform random
//! - **True contention**: Threads read/write overlapping key ranges
//! - **High thread counts**: 1, 2, 4, 8, 16, 32 threads
//!
//! ## Running with Alternative Allocators
//!
//! ```bash
//! cargo bench --bench concurrent_maps                          # default allocator
//! cargo bench --bench concurrent_maps --features mimalloc      # mimalloc
//! cargo bench --bench concurrent_maps --features jemalloc      # jemalloc
//! ```

#![expect(clippy::unwrap_used)]
#![expect(clippy::indexing_slicing)]
#![expect(clippy::pedantic)]
#![expect(clippy::redundant_locals)]

mod bench_utils;

use congee::Congee;
use crossbeam_epoch as epoch;
use crossbeam_skiplist::SkipMap;
use crossbeam_skiplist::base::SkipList;
use dashmap::DashMap;
use divan::{Bencher, black_box};
use indexset::concurrent::map::BTreeMap as IndexSetBTreeMap;
use masstree::MassTree;
use std::sync::Arc;
use std::thread;

use bench_utils::keys_shared_prefix;
use bench_utils::keys_shared_prefix_chunks;
use bench_utils::{keys, rw1_keys, shuffled_indices, uniform_indices, zipfian_indices};

fn main() {
    divan::main();
}

// =============================================================================
// Setup Helpers
// =============================================================================

fn setup_masstree<const K: usize>(keys: &[[u8; K]]) -> MassTree<u64> {
    let mut tree = MassTree::new();
    for (i, key) in keys.iter().enumerate() {
        let _ = tree.insert(key, i as u64);
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

fn setup_skiplist<const K: usize>(keys: &[[u8; K]]) -> SkipList<[u8; K], u64> {
    let list = SkipList::new(epoch::default_collector().clone());
    let guard = &epoch::pin();

    for (i, key) in keys.iter().enumerate() {
        list.insert(*key, i as u64, guard);
    }

    list
}

fn setup_indexset<const K: usize>(keys: &[[u8; K]]) -> IndexSetBTreeMap<[u8; K], u64> {
    let map = IndexSetBTreeMap::new();
    for (i, key) in keys.iter().enumerate() {
        map.insert(*key, i as u64);
    }
    map
}

fn setup_dashmap<const K: usize>(keys: &[[u8; K]]) -> DashMap<[u8; K], u64> {
    let map = DashMap::new();
    for (i, key) in keys.iter().enumerate() {
        map.insert(*key, i as u64);
    }
    map
}

/// Congee only supports usize keys and values, so we convert [u8; 8] to usize
fn setup_congee(keys: &[[u8; 8]]) -> Congee<usize, usize> {
    let tree: Congee<usize, usize> = Congee::default();
    let guard = tree.pin();
    for (i, key) in keys.iter().enumerate() {
        let k = usize::from_be_bytes(*key);
        let _ = tree.insert(k, i, &guard);
    }
    tree
}

// =============================================================================
// 01: SINGLE-THREADED GET - Variable Key Sizes
// =============================================================================

#[divan::bench_group(name = "01_get_by_key_size")]
mod get_by_key_size {
    use super::{
        Bencher, black_box, keys, setup_dashmap, setup_indexset, setup_masstree, setup_skipmap,
        uniform_indices,
    };

    const N: usize = 10_000;

    fn bench_masstree<const K: usize>(bencher: Bencher) {
        let keys = keys::<K>(N);
        let tree = setup_masstree::<K>(&keys);
        let lookup_keys = uniform_indices(N, 1000, 42);

        bencher.bench_local(|| {
            let mut sum = 0u64;
            for &idx in &lookup_keys {
                if let Some(v) = tree.get(&keys[idx]) {
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

    fn bench_dashmap<const K: usize>(bencher: Bencher) {
        let keys = keys::<K>(N);
        let map = setup_dashmap::<K>(&keys);
        let lookup_keys = uniform_indices(N, 1000, 42);

        bencher.bench_local(|| {
            let mut sum = 0u64;
            for &idx in &lookup_keys {
                if let Some(r) = map.get(&keys[idx]) {
                    sum += *r.value();
                }
            }
            black_box(sum)
        });
    }

    #[divan::bench(name = "masstree_8B")]
    fn masstree_8b(bencher: Bencher) {
        bench_masstree::<8>(bencher);
    }

    #[divan::bench(name = "masstree_16B")]
    fn masstree_16b(bencher: Bencher) {
        bench_masstree::<16>(bencher);
    }

    #[divan::bench(name = "masstree_24B")]
    fn masstree_24b(bencher: Bencher) {
        bench_masstree::<24>(bencher);
    }

    #[divan::bench(name = "masstree_32B")]
    fn masstree_32b(bencher: Bencher) {
        bench_masstree::<32>(bencher);
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

    #[divan::bench(name = "dashmap_8B")]
    fn dashmap_8b(bencher: Bencher) {
        bench_dashmap::<8>(bencher);
    }

    #[divan::bench(name = "dashmap_16B")]
    fn dashmap_16b(bencher: Bencher) {
        bench_dashmap::<16>(bencher);
    }

    #[divan::bench(name = "dashmap_24B")]
    fn dashmap_24b(bencher: Bencher) {
        bench_dashmap::<24>(bencher);
    }

    #[divan::bench(name = "dashmap_32B")]
    fn dashmap_32b(bencher: Bencher) {
        bench_dashmap::<32>(bencher);
    }
}

// =============================================================================
// 02: SINGLE-THREADED INSERT - Variable Key Sizes
// =============================================================================

#[divan::bench_group(name = "02_insert_by_key_size")]
mod insert_by_key_size {
    use super::{Bencher, DashMap, IndexSetBTreeMap, MassTree, SkipMap, keys};

    const N: usize = 1000;

    fn bench_masstree<const K: usize>(bencher: Bencher) {
        let keys = keys::<K>(N);
        bencher
            .with_inputs(|| keys.clone())
            .bench_local_values(|keys| {
                let mut tree = MassTree::<u64>::new();
                for (i, key) in keys.iter().enumerate() {
                    let _ = tree.insert(key, i as u64);
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

    fn bench_dashmap<const K: usize>(bencher: Bencher) {
        let keys = keys::<K>(N);
        bencher
            .with_inputs(|| keys.clone())
            .bench_local_values(|keys| {
                let map = DashMap::<[u8; K], u64>::new();
                for (i, key) in keys.iter().enumerate() {
                    map.insert(*key, i as u64);
                }
                map
            });
    }

    #[divan::bench(name = "masstree_8B")]
    fn masstree_8b(bencher: Bencher) {
        bench_masstree::<8>(bencher);
    }

    #[divan::bench(name = "masstree_16B")]
    fn masstree_16b(bencher: Bencher) {
        bench_masstree::<16>(bencher);
    }

    #[divan::bench(name = "masstree_24B")]
    fn masstree_24b(bencher: Bencher) {
        bench_masstree::<24>(bencher);
    }

    #[divan::bench(name = "masstree_32B")]
    fn masstree_32b(bencher: Bencher) {
        bench_masstree::<32>(bencher);
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

    #[divan::bench(name = "dashmap_8B")]
    fn dashmap_8b(bencher: Bencher) {
        bench_dashmap::<8>(bencher);
    }

    #[divan::bench(name = "dashmap_16B")]
    fn dashmap_16b(bencher: Bencher) {
        bench_dashmap::<16>(bencher);
    }

    #[divan::bench(name = "dashmap_24B")]
    fn dashmap_24b(bencher: Bencher) {
        bench_dashmap::<24>(bencher);
    }

    #[divan::bench(name = "dashmap_32B")]
    fn dashmap_32b(bencher: Bencher) {
        bench_dashmap::<32>(bencher);
    }
}

// =============================================================================
// 03: CONCURRENT READS - Thread Scaling (8-byte keys)
// =============================================================================

#[divan::bench_group(name = "03_concurrent_reads_scaling")]
mod concurrent_reads_scaling {
    use super::{
        Arc, Bencher, black_box, keys, setup_dashmap, setup_indexset, setup_masstree,
        setup_skipmap, thread, uniform_indices,
    };

    const N: usize = 100_000;
    const OPS_PER_THREAD: usize = 10_000;

    #[divan::bench(args = [1, 2, 4, 8, 16, 32])]
    fn masstree(bencher: Bencher, threads: usize) {
        let keys = Arc::new(keys::<8>(N));
        let tree = Arc::new(setup_masstree::<8>(keys.as_ref()));
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
                            if let Some(v) = tree.get_ref(&keys[idx], &guard) {
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

    #[divan::bench(args = [1, 2, 4, 8, 16, 32])]
    fn dashmap(bencher: Bencher, threads: usize) {
        let keys = Arc::new(keys::<8>(N));
        let map = Arc::new(setup_dashmap::<8>(keys.as_ref()));
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
                                sum += *r.value();
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
// 04: CONCURRENT READS - Long Keys (32-byte)
// =============================================================================

#[divan::bench_group(name = "04_concurrent_reads_long_keys")]
mod concurrent_reads_long_keys {
    use super::{
        Arc, Bencher, black_box, keys, setup_dashmap, setup_indexset, setup_masstree,
        setup_skipmap, thread, uniform_indices,
    };

    const N: usize = 50_000;
    const OPS_PER_THREAD: usize = 5000;

    #[divan::bench(args = [1, 2, 4, 8, 16])]
    fn masstree_32b(bencher: Bencher, threads: usize) {
        let keys = Arc::new(keys::<32>(N));
        let tree = Arc::new(setup_masstree::<32>(keys.as_ref()));
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
                            if let Some(v) = tree.get_ref(&keys[idx], &guard) {
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

    #[divan::bench(args = [1, 2, 4, 8, 16])]
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

    #[divan::bench(args = [1, 2, 4, 8, 16])]
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

    #[divan::bench(args = [1, 2, 4, 8, 16])]
    fn dashmap_32b(bencher: Bencher, threads: usize) {
        let keys = Arc::new(keys::<32>(N));
        let map = Arc::new(setup_dashmap::<32>(keys.as_ref()));
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
                                sum += *r.value();
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
// 05: CONCURRENT WRITES - Disjoint Ranges (Low Contention)
// =============================================================================

// #[divan::bench_group(name = "05_concurrent_writes_disjoint")]
// mod concurrent_writes_disjoint {
//     use super::{Arc, Bencher, IndexSetBTreeMap, MassTree, SkipMap, thread};
//
//     const OPS_PER_THREAD: usize = 1000;
//
//     #[divan::bench(args = [1, 2, 4, 8, 16])]
//     fn masstree(bencher: Bencher, threads: usize) {
//         bencher
//             .with_inputs(MassTree::<u64>::new)
//             .bench_local_values(|tree| {
//                 let tree = Arc::new(tree);
//                 let handles: Vec<_> = (0..threads)
//                     .map(|t| {
//                         let tree = Arc::clone(&tree);
//                         thread::spawn(move || {
//                             let guard = tree.guard();
//                             // Each thread writes to disjoint range
//                             let base = t * OPS_PER_THREAD;
//                             for i in 0..OPS_PER_THREAD {
//                                 let key = ((base + i) as u64).to_be_bytes();
//                                 let _ = tree.insert_with_guard(&key, i as u64, &guard);
//                             }
//                         })
//                     })
//                     .collect();
//
//                 for h in handles {
//                     h.join().unwrap();
//                 }
//                 tree
//             });
//     }
//
//     #[divan::bench(args = [1, 2, 4, 8, 16])]
//     fn skipmap(bencher: Bencher, threads: usize) {
//         bencher
//             .with_inputs(SkipMap::<[u8; 8], u64>::new)
//             .bench_local_values(|map| {
//                 let map = Arc::new(map);
//                 let handles: Vec<_> = (0..threads)
//                     .map(|t| {
//                         let map = Arc::clone(&map);
//                         thread::spawn(move || {
//                             let base = t * OPS_PER_THREAD;
//                             for i in 0..OPS_PER_THREAD {
//                                 let key = ((base + i) as u64).to_be_bytes();
//                                 map.insert(key, i as u64);
//                             }
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
//     #[divan::bench(args = [1, 2, 4, 8, 16])]
//     fn indexset(bencher: Bencher, threads: usize) {
//         bencher
//             .with_inputs(IndexSetBTreeMap::<[u8; 8], u64>::new)
//             .bench_local_values(|map| {
//                 let map = Arc::new(map);
//                 let handles: Vec<_> = (0..threads)
//                     .map(|t| {
//                         let map = Arc::clone(&map);
//                         thread::spawn(move || {
//                             let base = t * OPS_PER_THREAD;
//                             for i in 0..OPS_PER_THREAD {
//                                 let key = ((base + i) as u64).to_be_bytes();
//                                 map.insert(key, i as u64);
//                             }
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
//
// // =============================================================================
// // 06: CONCURRENT WRITES - Overlapping Keys (High Contention)
// // =============================================================================
//
// #[divan::bench_group(name = "06_concurrent_writes_contention")]
// mod concurrent_writes_contention {
//     use super::{
//         Arc, AtomicUsize, Bencher, Ordering, keys, setup_indexset, setup_masstree, setup_skipmap,
//         thread,
//     };
//
//     const KEY_SPACE: usize = 1000; // Small key space = high contention
//     const OPS_PER_THREAD: usize = 5000;
//
//     #[divan::bench(args = [2, 4, 8, 16])]
//     fn masstree(bencher: Bencher, threads: usize) {
//         let keys = Arc::new(keys::<8>(KEY_SPACE));
//
//         bencher
//             .with_inputs(|| Arc::new(setup_masstree::<8>(keys.as_ref())))
//             .bench_local_values(|tree| {
//                 let counter = Arc::new(AtomicUsize::new(0));
//                 let handles: Vec<_> = (0..threads)
//                     .map(|t| {
//                         let tree = Arc::clone(&tree);
//                         let keys = Arc::clone(&keys);
//                         let counter = Arc::clone(&counter);
//                         thread::spawn(move || {
//                             let guard = tree.guard();
//                             let mut state = (t as u64).wrapping_mul(0x517c_c1b7_2722_0a95);
//                             for _ in 0..OPS_PER_THREAD {
//                                 // Random key from shared pool
//                                 state = state
//                                     .wrapping_mul(6_364_136_223_846_793_005)
//                                     .wrapping_add(1);
//                                 let idx = (state as usize) % keys.len();
//                                 let val = counter.fetch_add(1, Ordering::Relaxed) as u64;
//                                 let _ = tree.insert_with_guard(&keys[idx], val, &guard);
//                             }
//                         })
//                     })
//                     .collect();
//
//                 for h in handles {
//                     h.join().unwrap();
//                 }
//                 tree
//             });
//     }
//
//     #[divan::bench(args = [2, 4, 8, 16])]
//     fn skipmap(bencher: Bencher, threads: usize) {
//         let keys = Arc::new(keys::<8>(KEY_SPACE));
//
//         bencher
//             .with_inputs(|| Arc::new(setup_skipmap::<8>(keys.as_ref())))
//             .bench_local_values(|map| {
//                 let counter = Arc::new(AtomicUsize::new(0));
//                 let handles: Vec<_> = (0..threads)
//                     .map(|t| {
//                         let map = Arc::clone(&map);
//                         let keys = Arc::clone(&keys);
//                         let counter = Arc::clone(&counter);
//                         thread::spawn(move || {
//                             let mut state = (t as u64).wrapping_mul(0x517c_c1b7_2722_0a95);
//                             for _ in 0..OPS_PER_THREAD {
//                                 state = state
//                                     .wrapping_mul(6_364_136_223_846_793_005)
//                                     .wrapping_add(1);
//                                 let idx = (state as usize) % keys.len();
//                                 let val = counter.fetch_add(1, Ordering::Relaxed) as u64;
//                                 let key = keys[idx];
//                                 map.insert(key, val);
//                             }
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
//     #[divan::bench(args = [2, 4, 8, 16])]
//     fn indexset(bencher: Bencher, threads: usize) {
//         let keys = Arc::new(keys::<8>(KEY_SPACE));
//
//         bencher
//             .with_inputs(|| Arc::new(setup_indexset::<8>(keys.as_ref())))
//             .bench_local_values(|map| {
//                 let counter = Arc::new(AtomicUsize::new(0));
//                 let handles: Vec<_> = (0..threads)
//                     .map(|t| {
//                         let map = Arc::clone(&map);
//                         let keys = Arc::clone(&keys);
//                         let counter = Arc::clone(&counter);
//                         thread::spawn(move || {
//                             let mut state = (t as u64).wrapping_mul(0x517c_c1b7_2722_0a95);
//                             for _ in 0..OPS_PER_THREAD {
//                                 state = state
//                                     .wrapping_mul(6_364_136_223_846_793_005)
//                                     .wrapping_add(1);
//                                 let idx = (state as usize) % keys.len();
//                                 let val = counter.fetch_add(1, Ordering::Relaxed) as u64;
//                                 let key = keys[idx];
//                                 map.insert(key, val);
//                             }
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
//
// // =============================================================================
// // 07: MIXED WORKLOAD - Zipfian Access (Realistic Hot Keys)
// // =============================================================================
//
// #[divan::bench_group(name = "07_mixed_zipfian")]
// mod mixed_zipfian {
//     use super::{
//         Arc, Bencher, black_box, keys, setup_indexset, setup_masstree, setup_skipmap, thread,
//         zipfian_indices,
//     };
//
//     const N: usize = 100_000;
//     const OPS_PER_THREAD: usize = 10_000;
//     const WRITE_RATIO: usize = 10; // 10% writes
//
//     #[divan::bench(args = [1, 2, 4, 8, 16])]
//     fn masstree(bencher: Bencher, threads: usize) {
//         let keys = Arc::new(keys::<8>(N));
//         let indices = Arc::new(zipfian_indices(N, OPS_PER_THREAD, 42));
//
//         bencher
//             .with_inputs(|| Arc::new(setup_masstree::<8>(keys.as_ref())))
//             .bench_local_values(|tree| {
//                 let handles: Vec<_> = (0..threads)
//                     .map(|t| {
//                         let tree = Arc::clone(&tree);
//                         let keys = Arc::clone(&keys);
//                         let indices = Arc::clone(&indices);
//                         thread::spawn(move || {
//                             let guard = tree.guard();
//                             let mut sum = 0u64;
//                             let offset = t * 7919;
//
//                             for i in 0..OPS_PER_THREAD {
//                                 let idx = indices[(i + offset) % indices.len()];
//                                 if i % WRITE_RATIO == 0 {
//                                     let _ = tree.insert_with_guard(&keys[idx], i as u64, &guard);
//                                 } else if let Some(v) = tree.get_ref(&keys[idx], &guard) {
//                                     sum += *v;
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
//                 tree
//             });
//     }
//
//     #[divan::bench(args = [1, 2, 4, 8, 16])]
//     fn skipmap(bencher: Bencher, threads: usize) {
//         let keys = Arc::new(keys::<8>(N));
//         let indices = Arc::new(zipfian_indices(N, OPS_PER_THREAD, 42));
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
//                                 if i % WRITE_RATIO == 0 {
//                                     let key = keys[idx];
//                                     map.insert(key, i as u64);
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
//     #[divan::bench(args = [1, 2, 4, 8, 16])]
//     fn indexset(bencher: Bencher, threads: usize) {
//         let keys = Arc::new(keys::<8>(N));
//         let indices = Arc::new(zipfian_indices(N, OPS_PER_THREAD, 42));
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
//                                 if i % WRITE_RATIO == 0 {
//                                     let key = keys[idx];
//                                     map.insert(key, i as u64);
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
//
// // =============================================================================
// // 08: MIXED WORKLOAD - Uniform Random (No Hot Keys)
// // =============================================================================
//
// #[divan::bench_group(name = "08_mixed_uniform")]
// mod mixed_uniform {
//     use super::{
//         Arc, Bencher, black_box, keys, setup_indexset, setup_masstree, setup_skipmap, thread,
//         uniform_indices,
//     };
//
//     const N: usize = 100_000;
//     const OPS_PER_THREAD: usize = 10_000;
//     const WRITE_RATIO: usize = 10; // 10% writes
//
//     #[divan::bench(args = [1, 2, 4, 8, 16])]
//     fn masstree(bencher: Bencher, threads: usize) {
//         let keys = Arc::new(keys::<8>(N));
//         let indices = Arc::new(uniform_indices(N, OPS_PER_THREAD, 42));
//
//         bencher
//             .with_inputs(|| Arc::new(setup_masstree::<8>(keys.as_ref())))
//             .bench_local_values(|tree| {
//                 let handles: Vec<_> = (0..threads)
//                     .map(|t| {
//                         let tree = Arc::clone(&tree);
//                         let keys = Arc::clone(&keys);
//                         let indices = Arc::clone(&indices);
//                         thread::spawn(move || {
//                             let guard = tree.guard();
//                             let mut sum = 0u64;
//                             let offset = t * 7919;
//
//                             for i in 0..OPS_PER_THREAD {
//                                 let idx = indices[(i + offset) % indices.len()];
//                                 if i % WRITE_RATIO == 0 {
//                                     let _ = tree.insert_with_guard(&keys[idx], i as u64, &guard);
//                                 } else if let Some(v) = tree.get_ref(&keys[idx], &guard) {
//                                     sum += *v;
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
//                 tree
//             });
//     }
//
//     #[divan::bench(args = [1, 2, 4, 8, 16])]
//     fn skipmap(bencher: Bencher, threads: usize) {
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
//                                 if i % WRITE_RATIO == 0 {
//                                     let key = keys[idx];
//                                     map.insert(key, i as u64);
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
//     #[divan::bench(args = [1, 2, 4, 8, 16])]
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
//                                 if i % WRITE_RATIO == 0 {
//                                     let key = keys[idx];
//                                     map.insert(key, i as u64);
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
// 08a: READ SCALING - Throughput (8-byte keys, MassTree's sweet spot)
// =============================================================================

#[divan::bench_group(name = "08a_read_scaling_8B")]
mod read_scaling_8b {
    use super::{
        Arc, Bencher, black_box, epoch, keys, setup_congee, setup_dashmap, setup_indexset,
        setup_masstree, setup_skiplist, setup_skipmap, thread,
    };

    // 10M keys to exceed L3 cache and measure realistic memory-bound performance.
    // This matches the scale used in the original Masstree paper (EuroSys 2012).
    const N: usize = 10_000_000;
    const OPS_PER_THREAD: usize = 50_000;

    #[divan::bench(args = [1, 2, 4, 8, 16, 32])]
    fn masstree(bencher: Bencher, threads: usize) {
        let keys = Arc::new(keys::<8>(N));
        let tree = Arc::new(setup_masstree::<8>(keys.as_ref()));

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
                                if let Some(v) = tree.get_ref(&keys[idx], &guard) {
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

    /// SkipList with reusable guard (like MassTree's get_ref pattern)
    #[divan::bench(args = [1, 2, 4, 8, 16, 32])]
    fn skiplist_guarded(bencher: Bencher, threads: usize) {
        let keys = Arc::new(keys::<8>(N));
        let list = Arc::new(setup_skiplist::<8>(keys.as_ref()));

        bencher
            .counter(divan::counter::ItemsCount::new(threads * OPS_PER_THREAD))
            .bench_local(|| {
                let handles: Vec<_> = (0..threads)
                    .map(|t| {
                        let list = Arc::clone(&list);
                        let keys = Arc::clone(&keys);
                        thread::spawn(move || {
                            let guard = &epoch::pin();
                            let mut sum = 0u64;
                            let start = (t * 7919) % keys.len();
                            for i in 0..OPS_PER_THREAD {
                                let idx = (start + i) % keys.len();
                                if let Some(e) = list.get(&keys[idx], guard) {
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

    #[divan::bench(args = [1, 2, 4, 8, 16, 32])]
    fn dashmap(bencher: Bencher, threads: usize) {
        let keys = Arc::new(keys::<8>(N));
        let map = Arc::new(setup_dashmap::<8>(keys.as_ref()));

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
                                    sum += *r.value();
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

    /// Congee (ART-OLC) - state-of-the-art concurrent adaptive radix tree
    /// Note: Congee only supports usize keys, so we convert [u8; 8] -> usize
    #[divan::bench(args = [1, 2, 4, 8, 16, 32])]
    fn congee(bencher: Bencher, threads: usize) {
        use super::Congee;

        let keys = Arc::new(keys::<8>(N));
        let tree: Arc<Congee<usize, usize>> = Arc::new(setup_congee(keys.as_ref()));

        bencher
            .counter(divan::counter::ItemsCount::new(threads * OPS_PER_THREAD))
            .bench_local(|| {
                let handles: Vec<_> = (0..threads)
                    .map(|t| {
                        let tree = Arc::clone(&tree);
                        let keys = Arc::clone(&keys);
                        thread::spawn(move || {
                            let guard = tree.pin();
                            let mut sum = 0usize;
                            let start = (t * 7919) % keys.len();
                            for i in 0..OPS_PER_THREAD {
                                let idx = (start + i) % keys.len();
                                let k = usize::from_be_bytes(keys[idx]);
                                if let Some(v) = tree.get(&k, &guard) {
                                    sum += v;
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
// 08b: READ SCALING - Throughput (32-byte keys, multi-layer for MassTree)
// =============================================================================

#[divan::bench_group(name = "08b_read_scaling_32B")]
mod read_scaling_32b {
    use super::{
        Arc, Bencher, black_box, epoch, keys, setup_dashmap, setup_indexset, setup_masstree,
        setup_skiplist, setup_skipmap, thread,
    };

    // 10M keys to exceed L3 cache and measure realistic memory-bound performance.
    // This matches the scale used in the original Masstree paper (EuroSys 2012).
    const N: usize = 10_000_000;
    const OPS_PER_THREAD: usize = 50_000;

    #[divan::bench(args = [1, 2, 4, 8, 16, 32])]
    fn masstree(bencher: Bencher, threads: usize) {
        let keys = Arc::new(keys::<32>(N));
        let tree = Arc::new(setup_masstree::<32>(keys.as_ref()));

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
                                if let Some(v) = tree.get_ref(&keys[idx], &guard) {
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

    /// SkipList with reusable guard (like MassTree's get_ref pattern)
    #[divan::bench(args = [1, 2, 4, 8, 16, 32])]
    fn skiplist_guarded(bencher: Bencher, threads: usize) {
        let keys = Arc::new(keys::<32>(N));
        let list = Arc::new(setup_skiplist::<32>(keys.as_ref()));

        bencher
            .counter(divan::counter::ItemsCount::new(threads * OPS_PER_THREAD))
            .bench_local(|| {
                let handles: Vec<_> = (0..threads)
                    .map(|t| {
                        let list = Arc::clone(&list);
                        let keys = Arc::clone(&keys);
                        thread::spawn(move || {
                            let guard = &epoch::pin();
                            let mut sum = 0u64;
                            let start = (t * 7919) % keys.len();
                            for i in 0..OPS_PER_THREAD {
                                let idx = (start + i) % keys.len();
                                if let Some(e) = list.get(&keys[idx], guard) {
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

    #[divan::bench(args = [1, 2, 4, 8, 16, 32])]
    fn dashmap(bencher: Bencher, threads: usize) {
        let keys = Arc::new(keys::<32>(N));
        let map = Arc::new(setup_dashmap::<32>(keys.as_ref()));

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
                                    sum += *r.value();
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
// 09: MIXED WORKLOAD - Long Keys + Zipfian (Worst Case for MassTree)
// =============================================================================

#[divan::bench_group(name = "09_mixed_long_keys_zipfian")]
mod mixed_long_keys_zipfian {
    use super::{
        Arc, Bencher, black_box, keys, setup_dashmap, setup_indexset, setup_masstree,
        setup_skipmap, thread, zipfian_indices,
    };

    const N: usize = 50_000;
    const OPS_PER_THREAD: usize = 5000;
    const WRITE_RATIO: usize = 10;

    #[divan::bench(args = [1, 2, 4, 8, 16])]
    fn masstree_32b(bencher: Bencher, threads: usize) {
        let keys = Arc::new(keys::<32>(N));
        let indices = Arc::new(zipfian_indices(N, OPS_PER_THREAD, 42));

        bencher
            .with_inputs(|| Arc::new(setup_masstree::<32>(keys.as_ref())))
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
                                } else if let Some(v) = tree.get_ref(&keys[idx], &guard) {
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

    #[divan::bench(args = [1, 2, 4, 8, 16])]
    fn skipmap_32b(bencher: Bencher, threads: usize) {
        let keys = Arc::new(keys::<32>(N));
        let indices = Arc::new(zipfian_indices(N, OPS_PER_THREAD, 42));

        bencher
            .with_inputs(|| Arc::new(setup_skipmap::<32>(keys.as_ref())))
            .bench_local_values(|map| {
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
                                if i % WRITE_RATIO == 0 {
                                    let key = keys[idx];
                                    map.insert(key, i as u64);
                                } else if let Some(e) = map.get(&keys[idx]) {
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

    #[divan::bench(args = [1, 2, 4, 8, 16])]
    fn indexset_32b(bencher: Bencher, threads: usize) {
        let keys = Arc::new(keys::<32>(N));
        let indices = Arc::new(zipfian_indices(N, OPS_PER_THREAD, 42));

        bencher
            .with_inputs(|| Arc::new(setup_indexset::<32>(keys.as_ref())))
            .bench_local_values(|map| {
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
                                if i % WRITE_RATIO == 0 {
                                    let key = keys[idx];
                                    map.insert(key, i as u64);
                                } else if let Some(r) = map.get(&keys[idx]) {
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

    #[divan::bench(args = [1, 2, 4, 8, 16])]
    fn dashmap_32b(bencher: Bencher, threads: usize) {
        let keys = Arc::new(keys::<32>(N));
        let indices = Arc::new(zipfian_indices(N, OPS_PER_THREAD, 42));

        bencher
            .with_inputs(|| Arc::new(setup_dashmap::<32>(keys.as_ref())))
            .bench_local_values(|map| {
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
                                if i % WRITE_RATIO == 0 {
                                    let key = keys[idx];
                                    map.insert(key, i as u64);
                                } else if let Some(r) = map.get(&keys[idx]) {
                                    sum += *r.value();
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
// 10: SINGLE HOT KEY - Maximum Contention
// =============================================================================

#[divan::bench_group(name = "10_single_hot_key")]
mod single_hot_key {
    use super::{
        Arc, Bencher, black_box, keys, setup_dashmap, setup_indexset, setup_masstree,
        setup_skipmap, thread,
    };

    const N: usize = 10_000;
    const OPS_PER_THREAD: usize = 10_000;

    #[divan::bench(args = [2, 4, 8, 16])]
    fn masstree(bencher: Bencher, threads: usize) {
        let keys = keys::<8>(N);
        let hot_key = keys[N / 2]; // Single hot key

        bencher
            .with_inputs(|| Arc::new(setup_masstree::<8>(&keys)))
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
                                } else if let Some(v) = tree.get_ref(&hot_key, &guard) {
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

    #[divan::bench(args = [2, 4, 8, 16])]
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

    #[divan::bench(args = [2, 4, 8, 16])]
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

    #[divan::bench(args = [2, 4, 8, 16])]
    fn dashmap(bencher: Bencher, threads: usize) {
        let keys = keys::<8>(N);
        let hot_key = keys[N / 2];

        bencher
            .with_inputs(|| Arc::new(setup_dashmap::<8>(&keys)))
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
                                    sum += *r.value();
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
// 11: SINGLE-THREADED GET - Shared Prefix Keys (Forces Layering)
// =============================================================================

#[divan::bench_group(name = "11_get_by_key_size_shared_prefix")]
mod get_by_key_size_shared_prefix {
    use super::{
        Bencher, black_box, keys_shared_prefix, setup_dashmap, setup_indexset, setup_masstree,
        setup_skipmap, uniform_indices,
    };

    const N: usize = 10_000;
    const PREFIX_BUCKETS: u64 = 256; // smaller => more shared prefixes

    fn bench_masstree<const K: usize>(bencher: Bencher) {
        let keys = keys_shared_prefix::<K>(N, PREFIX_BUCKETS);
        let tree = setup_masstree::<K>(&keys);
        let lookup_keys = uniform_indices(N, 1000, 42);

        bencher.bench_local(|| {
            let mut sum = 0u64;
            for &idx in &lookup_keys {
                if let Some(v) = tree.get(&keys[idx]) {
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

    fn bench_dashmap<const K: usize>(bencher: Bencher) {
        let keys = keys_shared_prefix::<K>(N, PREFIX_BUCKETS);
        let map = setup_dashmap::<K>(&keys);
        let lookup_keys = uniform_indices(N, 1000, 42);

        bencher.bench_local(|| {
            let mut sum = 0u64;
            for &idx in &lookup_keys {
                if let Some(r) = map.get(&keys[idx]) {
                    sum += *r.value();
                }
            }
            black_box(sum)
        });
    }

    #[divan::bench(name = "masstree_16B")]
    fn masstree_16b(bencher: Bencher) {
        bench_masstree::<16>(bencher);
    }

    #[divan::bench(name = "masstree_24B")]
    fn masstree_24b(bencher: Bencher) {
        bench_masstree::<24>(bencher);
    }

    #[divan::bench(name = "masstree_32B")]
    fn masstree_32b(bencher: Bencher) {
        bench_masstree::<32>(bencher);
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

    #[divan::bench(name = "dashmap_16B")]
    fn dashmap_16b(bencher: Bencher) {
        bench_dashmap::<16>(bencher);
    }

    #[divan::bench(name = "dashmap_24B")]
    fn dashmap_24b(bencher: Bencher) {
        bench_dashmap::<24>(bencher);
    }

    #[divan::bench(name = "dashmap_32B")]
    fn dashmap_32b(bencher: Bencher) {
        bench_dashmap::<32>(bencher);
    }
}

// =============================================================================
// 12: CONCURRENT READS - Shared Prefix Long Keys (32-byte)
// =============================================================================

#[divan::bench_group(name = "12_concurrent_reads_long_keys_shared_prefix")]
mod concurrent_reads_long_keys_shared_prefix {
    use super::{
        Arc, Bencher, black_box, keys_shared_prefix, setup_dashmap, setup_indexset, setup_masstree,
        setup_skipmap, thread, uniform_indices,
    };

    const N: usize = 50_000;
    const OPS_PER_THREAD: usize = 5000;
    const PREFIX_BUCKETS: u64 = 256;

    #[divan::bench(args = [1, 2, 4, 8, 16])]
    fn masstree_32b(bencher: Bencher, threads: usize) {
        let keys = Arc::new(keys_shared_prefix::<32>(N, PREFIX_BUCKETS));
        let tree = Arc::new(setup_masstree::<32>(keys.as_ref()));
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
                            if let Some(v) = tree.get_ref(&keys[idx], &guard) {
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

    #[divan::bench(args = [1, 2, 4, 8, 16])]
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

    #[divan::bench(args = [1, 2, 4, 8, 16])]
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

    #[divan::bench(args = [1, 2, 4, 8, 16])]
    fn dashmap_32b(bencher: Bencher, threads: usize) {
        let keys = Arc::new(keys_shared_prefix::<32>(N, PREFIX_BUCKETS));
        let map = Arc::new(setup_dashmap::<32>(keys.as_ref()));
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
                                sum += *r.value();
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
// 13: SINGLE-THREADED GET - Deep Shared Prefix Keys (Forces Multiple Layers)
// =============================================================================

#[divan::bench_group(name = "13_get_by_key_size_shared_prefix_deep")]
mod get_by_key_size_shared_prefix_deep {
    use super::{
        Bencher, black_box, keys_shared_prefix_chunks, setup_dashmap, setup_indexset,
        setup_masstree, setup_skipmap, uniform_indices,
    };

    const N: usize = 10_000;
    const PREFIX_BUCKETS: u64 = 1; // 1 => identical prefixes (worst case)

    fn bench_masstree<const K: usize>(bencher: Bencher, prefix_chunks: usize) {
        let keys = keys_shared_prefix_chunks::<K>(N, prefix_chunks, PREFIX_BUCKETS);
        let tree = setup_masstree::<K>(&keys);
        let lookup_keys = uniform_indices(N, 1000, 42);

        bencher.bench_local(|| {
            let mut sum = 0u64;
            for &idx in &lookup_keys {
                if let Some(v) = tree.get(&keys[idx]) {
                    sum += *v;
                }
            }
            black_box(sum)
        });
    }

    fn bench_skipmap<const K: usize>(bencher: Bencher, prefix_chunks: usize) {
        let keys = keys_shared_prefix_chunks::<K>(N, prefix_chunks, PREFIX_BUCKETS);
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

    fn bench_indexset<const K: usize>(bencher: Bencher, prefix_chunks: usize) {
        let keys = keys_shared_prefix_chunks::<K>(N, prefix_chunks, PREFIX_BUCKETS);
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

    fn bench_dashmap<const K: usize>(bencher: Bencher, prefix_chunks: usize) {
        let keys = keys_shared_prefix_chunks::<K>(N, prefix_chunks, PREFIX_BUCKETS);
        let map = setup_dashmap::<K>(&keys);
        let lookup_keys = uniform_indices(N, 1000, 42);

        bencher.bench_local(|| {
            let mut sum = 0u64;
            for &idx in &lookup_keys {
                if let Some(r) = map.get(&keys[idx]) {
                    sum += *r.value();
                }
            }
            black_box(sum)
        });
    }

    // 24B: collide first 2 chunks (16 bytes), vary 3rd
    #[divan::bench(name = "masstree_24B")]
    fn masstree_24b(bencher: Bencher) {
        bench_masstree::<24>(bencher, 2);
    }

    // 32B: collide first 3 chunks (24 bytes), vary 4th
    #[divan::bench(name = "masstree_32B")]
    fn masstree_32b(bencher: Bencher) {
        bench_masstree::<32>(bencher, 3);
    }

    #[divan::bench(name = "skipmap_24B")]
    fn skipmap_24b(bencher: Bencher) {
        bench_skipmap::<24>(bencher, 2);
    }

    #[divan::bench(name = "skipmap_32B")]
    fn skipmap_32b(bencher: Bencher) {
        bench_skipmap::<32>(bencher, 3);
    }

    #[divan::bench(name = "indexset_24B")]
    fn indexset_24b(bencher: Bencher) {
        bench_indexset::<24>(bencher, 2);
    }

    #[divan::bench(name = "indexset_32B")]
    fn indexset_32b(bencher: Bencher) {
        bench_indexset::<32>(bencher, 3);
    }

    #[divan::bench(name = "dashmap_24B")]
    fn dashmap_24b(bencher: Bencher) {
        bench_dashmap::<24>(bencher, 2);
    }

    #[divan::bench(name = "dashmap_32B")]
    fn dashmap_32b(bencher: Bencher) {
        bench_dashmap::<32>(bencher, 3);
    }
}

// =============================================================================
// 14: CONCURRENT READS - Deep Shared Prefix Long Keys (32-byte)
// =============================================================================

#[divan::bench_group(name = "14_concurrent_reads_long_keys_shared_prefix_deep")]
mod concurrent_reads_long_keys_shared_prefix_deep {
    use super::{
        Arc, Bencher, black_box, epoch, keys_shared_prefix_chunks, setup_dashmap, setup_indexset,
        setup_masstree, setup_skiplist, setup_skipmap, thread, uniform_indices,
    };

    const N: usize = 50_000;
    const OPS_PER_THREAD: usize = 5000;
    const PREFIX_BUCKETS: u64 = 1;
    const PREFIX_CHUNKS: usize = 3; // 24B shared prefix for 32B keys

    #[divan::bench(args = [1, 2, 4, 8, 16])]
    fn masstree_32b(bencher: Bencher, threads: usize) {
        let keys = Arc::new(keys_shared_prefix_chunks::<32>(
            N,
            PREFIX_CHUNKS,
            PREFIX_BUCKETS,
        ));
        let tree = Arc::new(setup_masstree::<32>(keys.as_ref()));
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
                            if let Some(v) = tree.get_ref(&keys[idx], &guard) {
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

    /// Like `masstree_32b`, but uses the public `get()` API, which creates a new guard per call
    /// (amortization disabled). This better matches SkipMap's `get()` shape.
    #[divan::bench(args = [1, 2, 4, 8, 16])]
    fn masstree_32b_api_get(bencher: Bencher, threads: usize) {
        let keys = Arc::new(keys_shared_prefix_chunks::<32>(
            N,
            PREFIX_CHUNKS,
            PREFIX_BUCKETS,
        ));
        let tree = Arc::new(setup_masstree::<32>(keys.as_ref()));
        let indices = Arc::new(uniform_indices(N, OPS_PER_THREAD, 42));

        bencher.bench_local(|| {
            let handles: Vec<_> = (0..threads)
                .map(|t| {
                    let tree = Arc::clone(&tree);
                    let keys = Arc::clone(&keys);
                    let indices = Arc::clone(&indices);
                    thread::spawn(move || {
                        let mut sum = 0u64;
                        let offset = t * 7919;
                        for i in 0..OPS_PER_THREAD {
                            let idx = indices[(i + offset) % indices.len()];
                            if let Some(v) = tree.get(&keys[idx]) {
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

    #[divan::bench(args = [1, 2, 4, 8, 16])]
    fn skipmap_32b(bencher: Bencher, threads: usize) {
        let keys = Arc::new(keys_shared_prefix_chunks::<32>(
            N,
            PREFIX_CHUNKS,
            PREFIX_BUCKETS,
        ));
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

    /// Crossbeam's base SkipList API allows reusing a pinned Guard per thread, similar to
    /// `MassTree::{guard,get_ref}`.
    #[divan::bench(args = [1, 2, 4, 8, 16])]
    fn skiplist_32b_guarded(bencher: Bencher, threads: usize) {
        let keys = Arc::new(keys_shared_prefix_chunks::<32>(
            N,
            PREFIX_CHUNKS,
            PREFIX_BUCKETS,
        ));
        let list = Arc::new(setup_skiplist::<32>(keys.as_ref()));
        let indices = Arc::new(uniform_indices(N, OPS_PER_THREAD, 42));

        bencher.bench_local(|| {
            let handles: Vec<_> = (0..threads)
                .map(|t| {
                    let list = Arc::clone(&list);
                    let keys = Arc::clone(&keys);
                    let indices = Arc::clone(&indices);
                    thread::spawn(move || {
                        let guard = &epoch::pin();
                        let mut sum = 0u64;
                        let offset = t * 7919;
                        for i in 0..OPS_PER_THREAD {
                            let idx = indices[(i + offset) % indices.len()];
                            if let Some(e) = list.get(&keys[idx], guard) {
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

    #[divan::bench(args = [1, 2, 4, 8, 16])]
    fn indexset_32b(bencher: Bencher, threads: usize) {
        let keys = Arc::new(keys_shared_prefix_chunks::<32>(
            N,
            PREFIX_CHUNKS,
            PREFIX_BUCKETS,
        ));
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

    #[divan::bench(args = [1, 2, 4, 8, 16])]
    fn dashmap_32b(bencher: Bencher, threads: usize) {
        let keys = Arc::new(keys_shared_prefix_chunks::<32>(
            N,
            PREFIX_CHUNKS,
            PREFIX_BUCKETS,
        ));
        let map = Arc::new(setup_dashmap::<32>(keys.as_ref()));
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
                                sum += *r.value();
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
// 15: MIXED WORKLOAD - Long Keys + Zipfian + Shared Prefix (Stresses Layering)
// =============================================================================

#[divan::bench_group(name = "15_mixed_long_keys_zipfian_shared_prefix")]
mod mixed_long_keys_zipfian_shared_prefix {
    use super::{
        Arc, Bencher, black_box, epoch, keys_shared_prefix, setup_dashmap, setup_indexset,
        setup_masstree, setup_skiplist, setup_skipmap, thread, zipfian_indices,
    };

    const N: usize = 50_000;
    const OPS_PER_THREAD: usize = 5000;
    const WRITE_RATIO: usize = 10;
    const PREFIX_BUCKETS: u64 = 16; // smaller => more shared prefixes => deeper MassTree layers

    #[divan::bench(args = [1, 2, 4, 8, 16])]
    fn masstree_32b(bencher: Bencher, threads: usize) {
        let keys = Arc::new(keys_shared_prefix::<32>(N, PREFIX_BUCKETS));
        let indices = Arc::new(zipfian_indices(N, OPS_PER_THREAD, 42));

        bencher
            .with_inputs(|| Arc::new(setup_masstree::<32>(keys.as_ref())))
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
                                } else if let Some(v) = tree.get_ref(&keys[idx], &guard) {
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

    /// Like `masstree_32b`, but forces per-op guard creation by using the public `get()` API.
    #[divan::bench(args = [1, 2, 4, 8, 16])]
    fn masstree_32b_api_get(bencher: Bencher, threads: usize) {
        let keys = Arc::new(keys_shared_prefix::<32>(N, PREFIX_BUCKETS));
        let indices = Arc::new(zipfian_indices(N, OPS_PER_THREAD, 42));

        bencher
            .with_inputs(|| Arc::new(setup_masstree::<32>(keys.as_ref())))
            .bench_local_values(|tree| {
                let handles: Vec<_> = (0..threads)
                    .map(|t| {
                        let tree = Arc::clone(&tree);
                        let keys = Arc::clone(&keys);
                        let indices = Arc::clone(&indices);
                        thread::spawn(move || {
                            let mut sum = 0u64;
                            let offset = t * 7919;

                            for i in 0..OPS_PER_THREAD {
                                let idx = indices[(i + offset) % indices.len()];
                                if i % WRITE_RATIO == 0 {
                                    let guard = tree.guard();
                                    let _ = tree.insert_with_guard(&keys[idx], i as u64, &guard);
                                } else if let Some(v) = tree.get(&keys[idx]) {
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

    #[divan::bench(args = [1, 2, 4, 8, 16])]
    fn skipmap_32b(bencher: Bencher, threads: usize) {
        let keys = Arc::new(keys_shared_prefix::<32>(N, PREFIX_BUCKETS));
        let indices = Arc::new(zipfian_indices(N, OPS_PER_THREAD, 42));

        bencher
            .with_inputs(|| Arc::new(setup_skipmap::<32>(keys.as_ref())))
            .bench_local_values(|map| {
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
                                if i % WRITE_RATIO == 0 {
                                    let key = keys[idx];
                                    map.insert(key, i as u64);
                                } else if let Some(e) = map.get(&keys[idx]) {
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

    /// SkipList variant that reuses a pinned guard per thread for both reads and writes.
    #[divan::bench(args = [1, 2, 4, 8, 16])]
    fn skiplist_32b_guarded(bencher: Bencher, threads: usize) {
        let keys = Arc::new(keys_shared_prefix::<32>(N, PREFIX_BUCKETS));
        let indices = Arc::new(zipfian_indices(N, OPS_PER_THREAD, 42));

        bencher
            .with_inputs(|| Arc::new(setup_skiplist::<32>(keys.as_ref())))
            .bench_local_values(|list| {
                let handles: Vec<_> = (0..threads)
                    .map(|t| {
                        let list = Arc::clone(&list);
                        let keys = Arc::clone(&keys);
                        let indices = Arc::clone(&indices);
                        thread::spawn(move || {
                            let guard = &epoch::pin();
                            let mut sum = 0u64;
                            let offset = t * 7919;

                            for i in 0..OPS_PER_THREAD {
                                let idx = indices[(i + offset) % indices.len()];
                                if i % WRITE_RATIO == 0 {
                                    let key = keys[idx];
                                    list.insert(key, i as u64, guard);
                                } else if let Some(e) = list.get(&keys[idx], guard) {
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
                list
            });
    }

    #[divan::bench(args = [1, 2, 4, 8, 16])]
    fn indexset_32b(bencher: Bencher, threads: usize) {
        let keys = Arc::new(keys_shared_prefix::<32>(N, PREFIX_BUCKETS));
        let indices = Arc::new(zipfian_indices(N, OPS_PER_THREAD, 42));

        bencher
            .with_inputs(|| Arc::new(setup_indexset::<32>(keys.as_ref())))
            .bench_local_values(|map| {
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
                                if i % WRITE_RATIO == 0 {
                                    let key = keys[idx];
                                    map.insert(key, i as u64);
                                } else if let Some(r) = map.get(&keys[idx]) {
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

    #[divan::bench(args = [1, 2, 4, 8, 16])]
    fn dashmap_32b(bencher: Bencher, threads: usize) {
        let keys = Arc::new(keys_shared_prefix::<32>(N, PREFIX_BUCKETS));
        let indices = Arc::new(zipfian_indices(N, OPS_PER_THREAD, 42));

        bencher
            .with_inputs(|| Arc::new(setup_dashmap::<32>(keys.as_ref())))
            .bench_local_values(|map| {
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
                                if i % WRITE_RATIO == 0 {
                                    let key = keys[idx];
                                    map.insert(key, i as u64);
                                } else if let Some(r) = map.get(&keys[idx]) {
                                    sum += *r.value();
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
// 10: RW1 - C++ Masstree-Compatible Benchmark
// =============================================================================
//
// This benchmark replicates the C++ Masstree `rw1` test for direct comparison:
// 1. Each thread generates N random i32 keys and inserts them with value = key + 1
// 2. Shuffles the keys array
// 3. Gets all N keys in shuffled order and VERIFIES values match
//
// Key differences from our other benchmarks:
// - Uses i32 keys (4 bytes) like C++ version, stored as big-endian [u8; 8]
// - Verifies returned values (not just summing/black_box)
// - Shuffled access pattern (truly random, not sequential with offset)
// - Each thread works on its own key set (reduces contention, measures scaling)

#[divan::bench_group(name = "10_rw1_cpp_compatible")]
mod rw1_cpp_compatible {
    use super::{Arc, Bencher, black_box, rw1_keys, shuffled_indices, thread};
    use masstree::MassTree;

    // 10M keys in tree (matches 08a benchmark scale)
    const N: usize = 10_000_000;
    // Operations per thread (same as 08a benchmark)
    const OPS_PER_THREAD: usize = 50_000;
    // Seed base - each thread gets seed + thread_id
    const SEED_BASE: u64 = 31337;

    /// Convert i32 key to [u8; 8] for MassTree (big-endian, sign-extended)
    #[inline]
    const fn i32_to_key(k: i32) -> [u8; 8] {
        // Sign-extend to i64, then to big-endian bytes
        (k as i64).to_be_bytes()
    }

    /// MassTree: Insert N keys, then get all in shuffled order with verification
    #[divan::bench(args = [1, 2, 4, 8, 16, 32])]
    fn masstree_rw1(bencher: Bencher, threads: usize) {
        bencher
            .counter(divan::counter::ItemsCount::new(threads * N * 2)) // N puts + N gets per thread
            .bench_local(|| {
                // Fresh tree for each iteration
                let tree = Arc::new(MassTree::<i32>::new());

                let handles: Vec<_> = (0..threads)
                    .map(|t| {
                        let tree = Arc::clone(&tree);
                        thread::spawn(move || {
                            let seed = SEED_BASE + t as u64;
                            let (keys, expected_values) = rw1_keys(N, seed);
                            let guard = tree.guard();

                            // Phase 1: Insert all keys
                            for (i, &key) in keys.iter().enumerate() {
                                let key_bytes = i32_to_key(key);
                                let _ =
                                    tree.insert_with_guard(&key_bytes, expected_values[i], &guard);
                            }

                            // Phase 2: Get all keys in shuffled order, verify values
                            let shuffle_order = shuffled_indices(N, seed.wrapping_add(1));

                            let mut verified = 0usize;
                            let mut misses = 0usize;

                            for &idx in &shuffle_order {
                                let key = keys[idx];
                                let key_bytes = i32_to_key(key);
                                let expected = expected_values[idx];

                                if let Some(&value) = tree.get_ref(&key_bytes, &guard) {
                                    // Note: Due to concurrent inserts from other threads,
                                    // we might get a different value if keys collide.
                                    // We just verify we got *something*.
                                    if value == expected {
                                        verified += 1;
                                    }
                                } else {
                                    misses += 1;
                                }
                            }

                            black_box((verified, misses));
                        })
                    })
                    .collect();

                for h in handles {
                    h.join().unwrap();
                }
            });
    }

    /// MassTree: Read-only phase after pre-population (for fairer read comparison)
    #[divan::bench(args = [1, 2, 4, 8, 16, 32])]
    fn masstree_rw1_reads_only(bencher: Bencher, threads: usize) {
        // Pre-populate with single-threaded insert
        let mut tree = MassTree::<i32>::new();
        let (keys_i32, expected_values) = rw1_keys(N, SEED_BASE);

        // Pre-convert keys to byte arrays (avoid conversion in hot path)
        let keys_bytes: Vec<[u8; 8]> = keys_i32.iter().map(|&k| i32_to_key(k)).collect();

        for (i, key_bytes) in keys_bytes.iter().enumerate() {
            let _ = tree.insert(key_bytes, expected_values[i]);
        }

        let tree = Arc::new(tree);
        let keys_bytes = Arc::new(keys_bytes);
        let expected_values = Arc::new(expected_values);

        // Pre-generate shuffled indices for each thread
        let shuffle_orders: Vec<_> = (0..32)
            .map(|t| Arc::new(shuffled_indices(N, SEED_BASE + t as u64)))
            .collect();

        bencher
            .counter(divan::counter::ItemsCount::new(threads * OPS_PER_THREAD))
            .bench_local(|| {
                let handles: Vec<_> = (0..threads)
                    .map(|t| {
                        let tree = Arc::clone(&tree);
                        let keys_bytes = Arc::clone(&keys_bytes);
                        let expected_values = Arc::clone(&expected_values);
                        let shuffle_order = Arc::clone(&shuffle_orders[t]);
                        thread::spawn(move || {
                            let guard = tree.guard();
                            let mut verified = 0usize;

                            // Only do OPS_PER_THREAD operations, not all N
                            for i in 0..OPS_PER_THREAD {
                                let idx = shuffle_order[i];
                                let key_bytes = &keys_bytes[idx];
                                let expected = expected_values[idx];

                                if let Some(&value) = tree.get_ref(key_bytes, &guard)
                                    && value == expected
                                {
                                    verified += 1;
                                }
                            }

                            black_box(verified);
                        })
                    })
                    .collect();

                for h in handles {
                    h.join().unwrap();
                }
            });
    }
}
