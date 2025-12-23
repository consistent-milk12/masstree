//! Comparison benchmarks: `MassTree` vs `Mutex<BTreeMap>` vs `RwLock<BTreeMap>`
//!
//! This is the fair comparison for concurrent use cases. `MassTree` is designed
//! to replace lock-wrapped `BTreeMap`, not bare `BTreeMap`.
//!
//! ## Benchmark Design Philosophy
//!
//! These benchmarks aim for objectivity by testing:
//! - **Variable key sizes**: 8, 16, 24, 32 bytes (`MassTree` optimizes for ≤8)
//! - **Realistic access patterns**: Uniform random, single hot key contention
//! - **True contention**: Threads read/write overlapping key ranges
//! - **High thread counts**: 1, 2, 4, 8, 16, 32 threads
//!
//! **Why both Mutex and `RwLock`?**
//! - `Mutex` has simpler state (locked/unlocked) and lower per-operation overhead
//! - `RwLock` allows concurrent readers but has more complex atomic operations
//! - For many workloads, `Mutex` outperforms `RwLock` due to lower overhead
//! - The crossover where `RwLock` wins requires many concurrent readers
//!
//! Run with: `cargo bench --bench lock_comparison`
//! With mimalloc: `cargo bench --bench lock_comparison --features mimalloc`

#![expect(clippy::redundant_locals)]
#![expect(clippy::indexing_slicing)]
#![expect(clippy::unwrap_used)]

mod bench_utils;

use divan::{Bencher, black_box};
use masstree::MassTree;
use std::collections::BTreeMap;
use std::sync::{Arc, Mutex, RwLock};
use std::thread;

use bench_utils::keys_shared_prefix;
use bench_utils::keys_shared_prefix_chunks;
use bench_utils::{keys, uniform_indices};

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

fn setup_mutex_btreemap<const K: usize>(keys: &[[u8; K]]) -> Mutex<BTreeMap<[u8; K], u64>> {
    let mut tree = BTreeMap::new();
    for (i, key) in keys.iter().enumerate() {
        tree.insert(*key, i as u64);
    }
    Mutex::new(tree)
}

fn setup_rwlock_btreemap<const K: usize>(keys: &[[u8; K]]) -> RwLock<BTreeMap<[u8; K], u64>> {
    let mut tree = BTreeMap::new();
    for (i, key) in keys.iter().enumerate() {
        tree.insert(*key, i as u64);
    }
    RwLock::new(tree)
}

// =============================================================================
// 01: SINGLE-THREADED GET - Variable Key Sizes
// =============================================================================

#[divan::bench_group(name = "01_get_by_key_size")]
mod get_by_key_size {
    use super::{
        Bencher, black_box, keys, setup_masstree, setup_mutex_btreemap, setup_rwlock_btreemap,
        uniform_indices,
    };

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

    fn bench_mutex<const K: usize>(bencher: Bencher) {
        let keys = keys::<K>(N);
        let tree = setup_mutex_btreemap::<K>(&keys);
        let indices = uniform_indices(N, 1000, 42);

        bencher.bench_local(|| {
            let mut sum = 0u64;
            for &idx in &indices {
                let guard = tree.lock().unwrap();
                if let Some(&v) = guard.get(&keys[idx]) {
                    sum += v;
                }
            }
            black_box(sum)
        });
    }

    fn bench_rwlock<const K: usize>(bencher: Bencher) {
        let keys = keys::<K>(N);
        let tree = setup_rwlock_btreemap::<K>(&keys);
        let indices = uniform_indices(N, 1000, 42);

        bencher.bench_local(|| {
            let mut sum = 0u64;
            for &idx in &indices {
                let guard = tree.read().unwrap();
                if let Some(&v) = guard.get(&keys[idx]) {
                    sum += v;
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

    #[divan::bench(name = "masstree_32B")]
    fn masstree_32b(bencher: Bencher) {
        bench_masstree::<32>(bencher);
    }

    #[divan::bench(name = "mutex_btreemap_8B")]
    fn mutex_8b(bencher: Bencher) {
        bench_mutex::<8>(bencher);
    }

    #[divan::bench(name = "mutex_btreemap_16B")]
    fn mutex_16b(bencher: Bencher) {
        bench_mutex::<16>(bencher);
    }

    #[divan::bench(name = "mutex_btreemap_32B")]
    fn mutex_32b(bencher: Bencher) {
        bench_mutex::<32>(bencher);
    }

    #[divan::bench(name = "rwlock_btreemap_8B")]
    fn rwlock_8b(bencher: Bencher) {
        bench_rwlock::<8>(bencher);
    }

    #[divan::bench(name = "rwlock_btreemap_16B")]
    fn rwlock_16b(bencher: Bencher) {
        bench_rwlock::<16>(bencher);
    }

    #[divan::bench(name = "rwlock_btreemap_32B")]
    fn rwlock_32b(bencher: Bencher) {
        bench_rwlock::<32>(bencher);
    }
}

// =============================================================================
// 02: CONCURRENT READS - Thread Scaling
// =============================================================================

#[divan::bench_group(name = "02_concurrent_reads_scaling")]
mod concurrent_reads_scaling {
    use super::{
        Arc, Bencher, black_box, keys, setup_masstree, setup_mutex_btreemap, setup_rwlock_btreemap,
        thread, uniform_indices,
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
    fn mutex_btreemap(bencher: Bencher, threads: usize) {
        let keys = Arc::new(keys::<8>(N));
        let tree = Arc::new(setup_mutex_btreemap::<8>(keys.as_ref()));
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
                            let guard = tree.lock().unwrap();
                            if let Some(&v) = guard.get(&keys[idx]) {
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

    #[divan::bench(args = [1, 2, 4, 8, 16, 32])]
    fn rwlock_btreemap(bencher: Bencher, threads: usize) {
        let keys = Arc::new(keys::<8>(N));
        let tree = Arc::new(setup_rwlock_btreemap::<8>(keys.as_ref()));
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
                            let guard = tree.read().unwrap();
                            if let Some(&v) = guard.get(&keys[idx]) {
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
// 03: CONCURRENT READS - Long Keys (32-byte)
// =============================================================================

#[divan::bench_group(name = "03_concurrent_reads_long_keys")]
mod concurrent_reads_long_keys {
    use super::{
        Arc, Bencher, black_box, keys, setup_masstree, setup_mutex_btreemap, setup_rwlock_btreemap,
        thread, uniform_indices,
    };

    const N: usize = 100_000;
    const OPS_PER_THREAD: usize = 10_000;

    #[divan::bench(args = [1, 2, 4, 8, 16, 32])]
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

    #[divan::bench(args = [1, 2, 4, 8, 16, 32])]
    fn mutex_btreemap_32b(bencher: Bencher, threads: usize) {
        let keys = Arc::new(keys::<32>(N));
        let tree = Arc::new(setup_mutex_btreemap::<32>(keys.as_ref()));
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
                            let guard = tree.lock().unwrap();
                            if let Some(&v) = guard.get(&keys[idx]) {
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

    #[divan::bench(args = [1, 2, 4, 8, 16, 32])]
    fn rwlock_btreemap_32b(bencher: Bencher, threads: usize) {
        let keys = Arc::new(keys::<32>(N));
        let tree = Arc::new(setup_rwlock_btreemap::<32>(keys.as_ref()));
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
                            let guard = tree.read().unwrap();
                            if let Some(&v) = guard.get(&keys[idx]) {
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
// 04: CONCURRENT WRITES - Disjoint Ranges
// =============================================================================

#[divan::bench_group(name = "04_concurrent_writes_disjoint")]
mod concurrent_writes_disjoint {
    use super::{Arc, BTreeMap, Bencher, MassTree, Mutex, RwLock, thread};

    const OPS_PER_THREAD: usize = 50_000;

    #[divan::bench(args = [1, 2, 4, 8, 16, 32])]
    fn masstree(bencher: Bencher, threads: usize) {
        bencher
            .with_inputs(|| Arc::new(MassTree::<u64>::new()))
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
    fn mutex_btreemap(bencher: Bencher, threads: usize) {
        bencher
            .with_inputs(|| Arc::new(Mutex::new(BTreeMap::<[u8; 8], u64>::new())))
            .bench_local_values(|tree| {
                let handles: Vec<_> = (0..threads)
                    .map(|t| {
                        let tree = Arc::clone(&tree);
                        thread::spawn(move || {
                            let base = t * OPS_PER_THREAD;
                            for i in 0..OPS_PER_THREAD {
                                let key = ((base + i) as u64).to_be_bytes();
                                let mut guard = tree.lock().unwrap();
                                guard.insert(key, i as u64);
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
    fn rwlock_btreemap(bencher: Bencher, threads: usize) {
        bencher
            .with_inputs(|| Arc::new(RwLock::new(BTreeMap::<[u8; 8], u64>::new())))
            .bench_local_values(|tree| {
                let handles: Vec<_> = (0..threads)
                    .map(|t| {
                        let tree = Arc::clone(&tree);
                        thread::spawn(move || {
                            let base = t * OPS_PER_THREAD;
                            for i in 0..OPS_PER_THREAD {
                                let key = ((base + i) as u64).to_be_bytes();
                                let mut guard = tree.write().unwrap();
                                guard.insert(key, i as u64);
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
}

// =============================================================================
// 05: SINGLE HOT KEY - Maximum Contention
// =============================================================================

#[divan::bench_group(name = "05_single_hot_key")]
mod single_hot_key {
    use super::{
        Arc, Bencher, black_box, keys, setup_masstree, setup_mutex_btreemap, setup_rwlock_btreemap,
        thread,
    };

    const N: usize = 100_000;
    const OPS_PER_THREAD: usize = 10_000;

    #[divan::bench(args = [2, 4, 8, 16, 32])]
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

    #[divan::bench(args = [2, 4, 8, 16, 32])]
    fn mutex_btreemap(bencher: Bencher, threads: usize) {
        let keys = keys::<8>(N);
        let hot_key = keys[N / 2];

        bencher
            .with_inputs(|| Arc::new(setup_mutex_btreemap::<8>(&keys)))
            .bench_local_values(|tree| {
                let handles: Vec<_> = (0..threads)
                    .map(|t| {
                        let tree = Arc::clone(&tree);
                        let hot_key = hot_key;
                        thread::spawn(move || {
                            let mut sum = 0u64;

                            for i in 0..OPS_PER_THREAD {
                                if i % 10 == 0 {
                                    let mut guard = tree.lock().unwrap();
                                    guard.insert(hot_key, (t * OPS_PER_THREAD + i) as u64);
                                } else {
                                    let guard = tree.lock().unwrap();
                                    if let Some(&v) = guard.get(&hot_key) {
                                        sum += v;
                                    }
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
    fn rwlock_btreemap(bencher: Bencher, threads: usize) {
        let keys = keys::<8>(N);
        let hot_key = keys[N / 2];

        bencher
            .with_inputs(|| Arc::new(setup_rwlock_btreemap::<8>(&keys)))
            .bench_local_values(|tree| {
                let handles: Vec<_> = (0..threads)
                    .map(|t| {
                        let tree = Arc::clone(&tree);
                        let hot_key = hot_key;
                        thread::spawn(move || {
                            let mut sum = 0u64;

                            for i in 0..OPS_PER_THREAD {
                                if i % 10 == 0 {
                                    let mut guard = tree.write().unwrap();
                                    guard.insert(hot_key, (t * OPS_PER_THREAD + i) as u64);
                                } else {
                                    let guard = tree.read().unwrap();
                                    if let Some(&v) = guard.get(&hot_key) {
                                        sum += v;
                                    }
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

// =============================================================================
// 08: READ SCALING - Throughput vs Thread Count
// =============================================================================

#[divan::bench_group(name = "08_read_scaling")]
mod read_scaling {
    use super::{
        Arc, Bencher, black_box, keys, setup_masstree, setup_mutex_btreemap, setup_rwlock_btreemap,
        thread,
    };

    const N: usize = 100_000;
    const OPS_PER_THREAD: usize = 10_000;

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
    fn mutex_btreemap(bencher: Bencher, threads: usize) {
        let keys = Arc::new(keys::<8>(N));
        let tree = Arc::new(setup_mutex_btreemap::<8>(keys.as_ref()));

        bencher
            .counter(divan::counter::ItemsCount::new(threads * OPS_PER_THREAD))
            .bench_local(|| {
                let handles: Vec<_> = (0..threads)
                    .map(|t| {
                        let tree = Arc::clone(&tree);
                        let keys = Arc::clone(&keys);
                        thread::spawn(move || {
                            let mut sum = 0u64;
                            let start = (t * 7919) % keys.len();
                            for i in 0..OPS_PER_THREAD {
                                let idx = (start + i) % keys.len();
                                let guard = tree.lock().unwrap();
                                if let Some(&v) = guard.get(&keys[idx]) {
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

    #[divan::bench(args = [1, 2, 4, 8, 16, 32])]
    fn rwlock_btreemap(bencher: Bencher, threads: usize) {
        let keys = Arc::new(keys::<8>(N));
        let tree = Arc::new(setup_rwlock_btreemap::<8>(keys.as_ref()));

        bencher
            .counter(divan::counter::ItemsCount::new(threads * OPS_PER_THREAD))
            .bench_local(|| {
                let handles: Vec<_> = (0..threads)
                    .map(|t| {
                        let tree = Arc::clone(&tree);
                        let keys = Arc::clone(&keys);
                        thread::spawn(move || {
                            let mut sum = 0u64;
                            let start = (t * 7919) % keys.len();
                            for i in 0..OPS_PER_THREAD {
                                let idx = (start + i) % keys.len();
                                let guard = tree.read().unwrap();
                                if let Some(&v) = guard.get(&keys[idx]) {
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
// 09: SINGLE-THREADED GET - Shared Prefix Keys (Forces Layering)
// =============================================================================

#[divan::bench_group(name = "09_get_by_key_size_shared_prefix")]
mod get_by_key_size_shared_prefix {
    use super::{
        Bencher, black_box, keys_shared_prefix, setup_masstree, setup_mutex_btreemap,
        setup_rwlock_btreemap, uniform_indices,
    };

    const N: usize = 10_000;
    const PREFIX_BUCKETS: u64 = 256;

    fn bench_masstree<const K: usize>(bencher: Bencher) {
        let keys = keys_shared_prefix::<K>(N, PREFIX_BUCKETS);
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

    fn bench_mutex<const K: usize>(bencher: Bencher) {
        let keys = keys_shared_prefix::<K>(N, PREFIX_BUCKETS);
        let tree = setup_mutex_btreemap::<K>(&keys);
        let indices = uniform_indices(N, 1000, 42);

        bencher.bench_local(|| {
            let mut sum = 0u64;
            for &idx in &indices {
                let guard = tree.lock().unwrap();
                if let Some(&v) = guard.get(&keys[idx]) {
                    sum += v;
                }
            }
            black_box(sum)
        });
    }

    fn bench_rwlock<const K: usize>(bencher: Bencher) {
        let keys = keys_shared_prefix::<K>(N, PREFIX_BUCKETS);
        let tree = setup_rwlock_btreemap::<K>(&keys);
        let indices = uniform_indices(N, 1000, 42);

        bencher.bench_local(|| {
            let mut sum = 0u64;
            for &idx in &indices {
                let guard = tree.read().unwrap();
                if let Some(&v) = guard.get(&keys[idx]) {
                    sum += v;
                }
            }
            black_box(sum)
        });
    }

    #[divan::bench(name = "masstree_16B")]
    fn masstree_16b(bencher: Bencher) {
        bench_masstree::<16>(bencher);
    }

    #[divan::bench(name = "masstree_32B")]
    fn masstree_32b(bencher: Bencher) {
        bench_masstree::<32>(bencher);
    }

    #[divan::bench(name = "mutex_btreemap_16B")]
    fn mutex_16b(bencher: Bencher) {
        bench_mutex::<16>(bencher);
    }

    #[divan::bench(name = "mutex_btreemap_32B")]
    fn mutex_32b(bencher: Bencher) {
        bench_mutex::<32>(bencher);
    }

    #[divan::bench(name = "rwlock_btreemap_16B")]
    fn rwlock_16b(bencher: Bencher) {
        bench_rwlock::<16>(bencher);
    }

    #[divan::bench(name = "rwlock_btreemap_32B")]
    fn rwlock_32b(bencher: Bencher) {
        bench_rwlock::<32>(bencher);
    }
}

// =============================================================================
// 10: CONCURRENT READS - Shared Prefix Long Keys (32-byte)
// =============================================================================

#[divan::bench_group(name = "10_concurrent_reads_long_keys_shared_prefix")]
mod concurrent_reads_long_keys_shared_prefix {
    use super::{
        Arc, Bencher, black_box, keys_shared_prefix, setup_masstree, setup_mutex_btreemap,
        setup_rwlock_btreemap, thread, uniform_indices,
    };

    const N: usize = 100_000;
    const OPS_PER_THREAD: usize = 10_000;
    const PREFIX_BUCKETS: u64 = 256;

    #[divan::bench(args = [1, 2, 4, 8, 16, 32])]
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

    #[divan::bench(args = [1, 2, 4, 8, 16, 32])]
    fn mutex_btreemap_32b(bencher: Bencher, threads: usize) {
        let keys = Arc::new(keys_shared_prefix::<32>(N, PREFIX_BUCKETS));
        let tree = Arc::new(setup_mutex_btreemap::<32>(keys.as_ref()));
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
                            let guard = tree.lock().unwrap();
                            if let Some(&v) = guard.get(&keys[idx]) {
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

    #[divan::bench(args = [1, 2, 4, 8, 16, 32])]
    fn rwlock_btreemap_32b(bencher: Bencher, threads: usize) {
        let keys = Arc::new(keys_shared_prefix::<32>(N, PREFIX_BUCKETS));
        let tree = Arc::new(setup_rwlock_btreemap::<32>(keys.as_ref()));
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
                            let guard = tree.read().unwrap();
                            if let Some(&v) = guard.get(&keys[idx]) {
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
// 11: SINGLE-THREADED GET - Deep Shared Prefix Keys (Forces Multiple Layers)
// =============================================================================

#[divan::bench_group(name = "11_get_by_key_size_shared_prefix_deep")]
mod get_by_key_size_shared_prefix_deep {
    use super::{
        Bencher, black_box, keys_shared_prefix_chunks, setup_masstree, setup_mutex_btreemap,
        setup_rwlock_btreemap, uniform_indices,
    };

    const N: usize = 10_000;
    const PREFIX_BUCKETS: u64 = 1;

    fn bench_masstree<const K: usize>(bencher: Bencher, prefix_chunks: usize) {
        let keys = keys_shared_prefix_chunks::<K>(N, prefix_chunks, PREFIX_BUCKETS);
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

    fn bench_mutex<const K: usize>(bencher: Bencher, prefix_chunks: usize) {
        let keys = keys_shared_prefix_chunks::<K>(N, prefix_chunks, PREFIX_BUCKETS);
        let tree = setup_mutex_btreemap::<K>(&keys);
        let indices = uniform_indices(N, 1000, 42);

        bencher.bench_local(|| {
            let mut sum = 0u64;
            for &idx in &indices {
                let guard = tree.lock().unwrap();
                if let Some(&v) = guard.get(&keys[idx]) {
                    sum += v;
                }
            }
            black_box(sum)
        });
    }

    fn bench_rwlock<const K: usize>(bencher: Bencher, prefix_chunks: usize) {
        let keys = keys_shared_prefix_chunks::<K>(N, prefix_chunks, PREFIX_BUCKETS);
        let tree = setup_rwlock_btreemap::<K>(&keys);
        let indices = uniform_indices(N, 1000, 42);

        bencher.bench_local(|| {
            let mut sum = 0u64;
            for &idx in &indices {
                let guard = tree.read().unwrap();
                if let Some(&v) = guard.get(&keys[idx]) {
                    sum += v;
                }
            }
            black_box(sum)
        });
    }

    #[divan::bench(name = "masstree_24B")]
    fn masstree_24b(bencher: Bencher) {
        bench_masstree::<24>(bencher, 2);
    }

    #[divan::bench(name = "masstree_32B")]
    fn masstree_32b(bencher: Bencher) {
        bench_masstree::<32>(bencher, 3);
    }

    #[divan::bench(name = "mutex_btreemap_24B")]
    fn mutex_24b(bencher: Bencher) {
        bench_mutex::<24>(bencher, 2);
    }

    #[divan::bench(name = "mutex_btreemap_32B")]
    fn mutex_32b(bencher: Bencher) {
        bench_mutex::<32>(bencher, 3);
    }

    #[divan::bench(name = "rwlock_btreemap_24B")]
    fn rwlock_24b(bencher: Bencher) {
        bench_rwlock::<24>(bencher, 2);
    }

    #[divan::bench(name = "rwlock_btreemap_32B")]
    fn rwlock_32b(bencher: Bencher) {
        bench_rwlock::<32>(bencher, 3);
    }
}

// =============================================================================
// 12: CONCURRENT READS - Deep Shared Prefix Long Keys (32-byte)
// =============================================================================

#[divan::bench_group(name = "12_concurrent_reads_long_keys_shared_prefix_deep")]
mod concurrent_reads_long_keys_shared_prefix_deep {
    use super::{
        Arc, Bencher, black_box, keys_shared_prefix_chunks, setup_masstree, setup_mutex_btreemap,
        setup_rwlock_btreemap, thread, uniform_indices,
    };

    const N: usize = 100_000;
    const OPS_PER_THREAD: usize = 10_000;
    const PREFIX_BUCKETS: u64 = 1;
    const PREFIX_CHUNKS: usize = 3;

    #[divan::bench(args = [1, 2, 4, 8, 16, 32])]
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

    #[divan::bench(args = [1, 2, 4, 8, 16, 32])]
    fn mutex_btreemap_32b(bencher: Bencher, threads: usize) {
        let keys = Arc::new(keys_shared_prefix_chunks::<32>(
            N,
            PREFIX_CHUNKS,
            PREFIX_BUCKETS,
        ));
        let tree = Arc::new(setup_mutex_btreemap::<32>(keys.as_ref()));
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
                            let guard = tree.lock().unwrap();
                            if let Some(&v) = guard.get(&keys[idx]) {
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

    #[divan::bench(args = [1, 2, 4, 8, 16, 32])]
    fn rwlock_btreemap_32b(bencher: Bencher, threads: usize) {
        let keys = Arc::new(keys_shared_prefix_chunks::<32>(
            N,
            PREFIX_CHUNKS,
            PREFIX_BUCKETS,
        ));
        let tree = Arc::new(setup_rwlock_btreemap::<32>(keys.as_ref()));
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
                            let guard = tree.read().unwrap();
                            if let Some(&v) = guard.get(&keys[idx]) {
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
