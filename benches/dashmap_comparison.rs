//! Comparison benchmarks: `MassTree` vs `DashMap`
//!
//! **Key Differences:**
//! - `DashMap`: Hash-based, unordered, sharded locks, Send+Sync
//! - `MassTree`: Trie of B+trees, ordered, per-node locks, Send+Sync
//!
//! ## Benchmark Design Philosophy
//!
//! These benchmarks aim for objectivity by testing:
//! - **Variable key sizes**: 8, 16, 24, 32 bytes (`MassTree` optimizes for ≤8)
//! - **Realistic access patterns**: Zipfian distribution (hot keys), uniform random
//! - **True contention**: Threads read/write overlapping key ranges
//! - **High thread counts**: 1, 2, 4, 8, 16, 32 threads
//!
//! Run with: `cargo bench --bench dashmap_comparison`
//! With mimalloc: `cargo bench --bench dashmap_comparison --features mimalloc`

#![expect(clippy::indexing_slicing)]
#![expect(
    clippy::cast_sign_loss,
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss
)]
#![expect(clippy::unwrap_used)]

mod bench_utils;

use dashmap::DashMap;
use divan::{Bencher, black_box};
use masstree::tree::MassTree;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::thread;

use bench_utils::{keys, uniform_indices, zipfian_indices};

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

fn setup_dashmap<const K: usize>(keys: &[[u8; K]]) -> DashMap<[u8; K], u64> {
    let map = DashMap::new();
    for (i, key) in keys.iter().enumerate() {
        map.insert(*key, i as u64);
    }
    map
}

// =============================================================================
// 01: INSERT - Variable Key Sizes
// =============================================================================

#[divan::bench_group(name = "01_insert_by_key_size")]
mod insert_by_key_size {
    use super::{Bencher, DashMap, MassTree, keys};

    const N: usize = 10_000;

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
// 02: GET - Variable Key Sizes
// =============================================================================

#[divan::bench_group(name = "02_get_by_key_size")]
mod get_by_key_size {
    use super::{Bencher, black_box, keys, setup_dashmap, setup_masstree, uniform_indices};

    const N: usize = 10_000;

    fn bench_masstree<const K: usize>(bencher: Bencher) {
        let keys = keys::<K>(N);
        let tree = setup_masstree::<K>(&keys);
        let indices = uniform_indices(N, 1000, 42);

        bencher.bench_local(|| {
            let mut sum = 0u64;
            for &idx in &indices {
                if let Some(v) = tree.get(&keys[idx]) {
                    sum += *v;
                }
            }
            black_box(sum)
        });
    }

    fn bench_dashmap<const K: usize>(bencher: Bencher) {
        let keys = keys::<K>(N);
        let map = setup_dashmap::<K>(&keys);
        let indices = uniform_indices(N, 1000, 42);

        bencher.bench_local(|| {
            let mut sum = 0u64;
            for &idx in &indices {
                if let Some(v) = map.get(&keys[idx]) {
                    sum += *v;
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
// 03: CONCURRENT READS - Thread Scaling
// =============================================================================

#[divan::bench_group(name = "03_concurrent_reads_scaling")]
mod concurrent_reads_scaling {
    use super::{
        Arc, Bencher, black_box, keys, setup_dashmap, setup_masstree, thread, uniform_indices,
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
                            if let Some(v) = map.get(&keys[idx]) {
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
}

// =============================================================================
// 04: CONCURRENT READS - Long Keys (32-byte)
// =============================================================================

#[divan::bench_group(name = "04_concurrent_reads_long_keys")]
mod concurrent_reads_long_keys {
    use super::{
        Arc, Bencher, black_box, keys, setup_dashmap, setup_masstree, thread, uniform_indices,
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
                            if let Some(v) = map.get(&keys[idx]) {
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
}

// =============================================================================
// 05: CONCURRENT WRITES - Overlapping Keys (High Contention)
// =============================================================================

#[divan::bench_group(name = "05_concurrent_writes_contention")]
mod concurrent_writes_contention {
    use super::{Arc, AtomicUsize, Bencher, Ordering, keys, setup_dashmap, setup_masstree, thread};

    const KEY_SPACE: usize = 1000;
    const OPS_PER_THREAD: usize = 5000;

    #[divan::bench(args = [2, 4, 8, 16])]
    fn masstree(bencher: Bencher, threads: usize) {
        let keys = Arc::new(keys::<8>(KEY_SPACE));

        bencher
            .with_inputs(|| Arc::new(setup_masstree::<8>(keys.as_ref())))
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

    #[divan::bench(args = [2, 4, 8, 16])]
    fn dashmap(bencher: Bencher, threads: usize) {
        let keys = Arc::new(keys::<8>(KEY_SPACE));

        bencher
            .with_inputs(|| Arc::new(setup_dashmap::<8>(keys.as_ref())))
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
                                let key = keys[idx];
                                map.insert(key, val);
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
// 06: MIXED WORKLOAD - Zipfian Access (Realistic Hot Keys)
// =============================================================================

#[divan::bench_group(name = "06_mixed_zipfian")]
mod mixed_zipfian {
    use super::{
        Arc, Bencher, black_box, keys, setup_dashmap, setup_masstree, thread, zipfian_indices,
    };

    const N: usize = 100_000;
    const OPS_PER_THREAD: usize = 10_000;
    const WRITE_RATIO: usize = 10;

    #[divan::bench(args = [1, 2, 4, 8, 16])]
    fn masstree(bencher: Bencher, threads: usize) {
        let keys = Arc::new(keys::<8>(N));
        let indices = Arc::new(zipfian_indices(N, OPS_PER_THREAD, 42));

        bencher
            .with_inputs(|| Arc::new(setup_masstree::<8>(keys.as_ref())))
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
    fn dashmap(bencher: Bencher, threads: usize) {
        let keys = Arc::new(keys::<8>(N));
        let indices = Arc::new(zipfian_indices(N, OPS_PER_THREAD, 42));

        bencher
            .with_inputs(|| Arc::new(setup_dashmap::<8>(keys.as_ref())))
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
                                } else if let Some(v) = map.get(&keys[idx]) {
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
                map
            });
    }
}

// =============================================================================
// 07: MIXED WORKLOAD - Long Keys + Zipfian (Stresses MassTree's Multi-Layer)
// =============================================================================

#[divan::bench_group(name = "07_mixed_long_keys_zipfian")]
mod mixed_long_keys_zipfian {
    use super::{
        Arc, Bencher, black_box, keys, setup_dashmap, setup_masstree, thread, zipfian_indices,
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
                                } else if let Some(v) = map.get(&keys[idx]) {
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
                map
            });
    }
}

// =============================================================================
// 08: SINGLE HOT KEY - Maximum Contention
// =============================================================================

#[divan::bench_group(name = "08_single_hot_key")]
mod single_hot_key {
    use super::{Arc, Bencher, black_box, keys, setup_dashmap, setup_masstree, thread};

    const N: usize = 10_000;
    const OPS_PER_THREAD: usize = 10_000;

    #[divan::bench(args = [2, 4, 8, 16])]
    fn masstree(bencher: Bencher, threads: usize) {
        let keys = keys::<8>(N);
        let hot_key = keys[N / 2];

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
                                } else if let Some(v) = map.get(&hot_key) {
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
                map
            });
    }
}

// =============================================================================
// 09: SCALING - Read Throughput vs Thread Count
// =============================================================================

#[divan::bench_group(name = "09_read_scaling")]
mod read_scaling {
    use super::{Arc, Bencher, black_box, keys, setup_dashmap, setup_masstree, thread};

    const N: usize = 100_000;
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
                                if let Some(v) = map.get(&keys[idx]) {
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
}
