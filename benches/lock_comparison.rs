//! Comparison benchmarks: `MassTree` vs `Mutex<BTreeMap>` vs `RwLock<BTreeMap>`
//!
//! This is the fair comparison for concurrent use cases. MassTree is designed
//! to replace lock-wrapped `BTreeMap`, not bare `BTreeMap`.
//!
//! ## Benchmark Design Philosophy
//!
//! These benchmarks aim for objectivity by testing:
//! - **Variable key sizes**: 8, 16, 24, 32 bytes (MassTree optimizes for ≤8)
//! - **Realistic access patterns**: Zipfian distribution (hot keys), uniform random
//! - **True contention**: Threads read/write overlapping key ranges
//! - **High thread counts**: 1, 2, 4, 8, 16, 32 threads
//!
//! **Why both Mutex and RwLock?**
//! - `Mutex` has simpler state (locked/unlocked) and lower per-operation overhead
//! - `RwLock` allows concurrent readers but has more complex atomic operations
//! - For many workloads, `Mutex` outperforms `RwLock` due to lower overhead
//! - The crossover where `RwLock` wins requires many concurrent readers
//!
//! Run with: `cargo bench --bench lock_comparison`
//! With mimalloc: `cargo bench --bench lock_comparison --features mimalloc`

#![expect(clippy::indexing_slicing)]

// Use alternative allocator if feature is enabled
#[cfg(feature = "mimalloc")]
#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;

use divan::{Bencher, black_box};
use masstree::MassTree;
use std::collections::BTreeMap;
use std::sync::{Arc, Mutex, RwLock};
use std::thread;

fn main() {
    divan::main();
}

// =============================================================================
// Key Generation Helpers
// =============================================================================

/// Generate 8-byte keys (single MassTree layer, inline storage)
fn keys_8b(n: usize) -> Vec<Vec<u8>> {
    (0..n).map(|i| (i as u64).to_be_bytes().to_vec()).collect()
}

/// Generate 16-byte keys (2 MassTree layers)
fn keys_16b(n: usize) -> Vec<Vec<u8>> {
    (0..n)
        .map(|i| {
            let mut key = vec![0u8; 16];
            key[0..8].copy_from_slice(&(i as u64).to_be_bytes());
            key[8..16]
                .copy_from_slice(&((i as u64).wrapping_mul(0x517cc1b727220a95)).to_be_bytes());
            key
        })
        .collect()
}

/// Generate 32-byte keys (4 MassTree layers) - typical hash/UUID size
fn keys_32b(n: usize) -> Vec<Vec<u8>> {
    (0..n)
        .map(|i| {
            let mut key = vec![0u8; 32];
            key[0..8].copy_from_slice(&(i as u64).to_be_bytes());
            key[8..16]
                .copy_from_slice(&((i as u64).wrapping_mul(0x517cc1b727220a95)).to_be_bytes());
            key[16..24]
                .copy_from_slice(&((i as u64).wrapping_mul(0x9e3779b97f4a7c15)).to_be_bytes());
            key[24..32]
                .copy_from_slice(&((i as u64).wrapping_mul(0xbf58476d1ce4e5b9)).to_be_bytes());
            key
        })
        .collect()
}

/// Generate Zipfian-distributed indices (hot keys accessed more frequently)
fn zipfian_indices(n: usize, count: usize, seed: u64) -> Vec<usize> {
    let mut indices = Vec::with_capacity(count);
    let mut state = seed;

    for _ in 0..count {
        state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
        let u = (state >> 33) as f64 / (1u64 << 31) as f64;
        let idx = ((n as f64).powf(1.0 - u) - 1.0).max(0.0) as usize;
        indices.push(idx.min(n - 1));
    }
    indices
}

/// Uniform random indices
fn uniform_indices(n: usize, count: usize, seed: u64) -> Vec<usize> {
    let mut indices = Vec::with_capacity(count);
    let mut state = seed;

    for _ in 0..count {
        state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
        indices.push((state as usize) % n);
    }
    indices
}

// =============================================================================
// Setup Helpers
// =============================================================================

fn setup_masstree(keys: &[Vec<u8>]) -> MassTree<u64> {
    let mut tree = MassTree::new();
    for (i, key) in keys.iter().enumerate() {
        let _ = tree.insert(key, i as u64);
    }
    tree
}

fn setup_mutex_btreemap(keys: &[Vec<u8>]) -> Mutex<BTreeMap<Vec<u8>, u64>> {
    let mut tree = BTreeMap::new();
    for (i, key) in keys.iter().enumerate() {
        tree.insert(key.clone(), i as u64);
    }
    Mutex::new(tree)
}

fn setup_rwlock_btreemap(keys: &[Vec<u8>]) -> RwLock<BTreeMap<Vec<u8>, u64>> {
    let mut tree = BTreeMap::new();
    for (i, key) in keys.iter().enumerate() {
        tree.insert(key.clone(), i as u64);
    }
    RwLock::new(tree)
}

// =============================================================================
// 01: SINGLE-THREADED GET - Variable Key Sizes
// =============================================================================

#[divan::bench_group(name = "01_get_by_key_size")]
mod get_by_key_size {
    use super::*;

    const N: usize = 10_000;

    #[divan::bench(args = ["8B", "16B", "32B"])]
    fn masstree(bencher: Bencher, key_size: &str) {
        let keys = match key_size {
            "8B" => keys_8b(N),
            "16B" => keys_16b(N),
            "32B" => keys_32b(N),
            _ => unreachable!(),
        };
        let tree = setup_masstree(&keys);
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

    #[divan::bench(args = ["8B", "16B", "32B"])]
    fn mutex_btreemap(bencher: Bencher, key_size: &str) {
        let keys = match key_size {
            "8B" => keys_8b(N),
            "16B" => keys_16b(N),
            "32B" => keys_32b(N),
            _ => unreachable!(),
        };
        let tree = setup_mutex_btreemap(&keys);
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

    #[divan::bench(args = ["8B", "16B", "32B"])]
    fn rwlock_btreemap(bencher: Bencher, key_size: &str) {
        let keys = match key_size {
            "8B" => keys_8b(N),
            "16B" => keys_16b(N),
            "32B" => keys_32b(N),
            _ => unreachable!(),
        };
        let tree = setup_rwlock_btreemap(&keys);
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
}

// =============================================================================
// 02: CONCURRENT READS - Thread Scaling
// =============================================================================

#[divan::bench_group(name = "02_concurrent_reads_scaling")]
mod concurrent_reads_scaling {
    use super::*;

    const N: usize = 100_000;
    const OPS_PER_THREAD: usize = 10_000;

    #[divan::bench(args = [1, 2, 4, 8, 16, 32])]
    fn masstree(bencher: Bencher, threads: usize) {
        let keys = keys_8b(N);
        let tree = Arc::new(setup_masstree(&keys));
        let indices = uniform_indices(N, OPS_PER_THREAD, 42);

        bencher.bench_local(|| {
            let handles: Vec<_> = (0..threads)
                .map(|t| {
                    let tree = Arc::clone(&tree);
                    let keys = keys.clone();
                    let indices = indices.clone();
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
        let keys = keys_8b(N);
        let tree = Arc::new(setup_mutex_btreemap(&keys));
        let indices = uniform_indices(N, OPS_PER_THREAD, 42);

        bencher.bench_local(|| {
            let handles: Vec<_> = (0..threads)
                .map(|t| {
                    let tree = Arc::clone(&tree);
                    let keys = keys.clone();
                    let indices = indices.clone();
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
        let keys = keys_8b(N);
        let tree = Arc::new(setup_rwlock_btreemap(&keys));
        let indices = uniform_indices(N, OPS_PER_THREAD, 42);

        bencher.bench_local(|| {
            let handles: Vec<_> = (0..threads)
                .map(|t| {
                    let tree = Arc::clone(&tree);
                    let keys = keys.clone();
                    let indices = indices.clone();
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
    use super::*;

    const N: usize = 50_000;
    const OPS_PER_THREAD: usize = 5000;

    #[divan::bench(args = [1, 2, 4, 8, 16])]
    fn masstree_32b(bencher: Bencher, threads: usize) {
        let keys = keys_32b(N);
        let tree = Arc::new(setup_masstree(&keys));
        let indices = uniform_indices(N, OPS_PER_THREAD, 42);

        bencher.bench_local(|| {
            let handles: Vec<_> = (0..threads)
                .map(|t| {
                    let tree = Arc::clone(&tree);
                    let keys = keys.clone();
                    let indices = indices.clone();
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
    fn mutex_btreemap_32b(bencher: Bencher, threads: usize) {
        let keys = keys_32b(N);
        let tree = Arc::new(setup_mutex_btreemap(&keys));
        let indices = uniform_indices(N, OPS_PER_THREAD, 42);

        bencher.bench_local(|| {
            let handles: Vec<_> = (0..threads)
                .map(|t| {
                    let tree = Arc::clone(&tree);
                    let keys = keys.clone();
                    let indices = indices.clone();
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

    #[divan::bench(args = [1, 2, 4, 8, 16])]
    fn rwlock_btreemap_32b(bencher: Bencher, threads: usize) {
        let keys = keys_32b(N);
        let tree = Arc::new(setup_rwlock_btreemap(&keys));
        let indices = uniform_indices(N, OPS_PER_THREAD, 42);

        bencher.bench_local(|| {
            let handles: Vec<_> = (0..threads)
                .map(|t| {
                    let tree = Arc::clone(&tree);
                    let keys = keys.clone();
                    let indices = indices.clone();
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
    use super::*;

    const OPS_PER_THREAD: usize = 1000;

    #[divan::bench(args = [1, 2, 4, 8, 16])]
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

    #[divan::bench(args = [1, 2, 4, 8, 16])]
    fn mutex_btreemap(bencher: Bencher, threads: usize) {
        bencher
            .with_inputs(|| Arc::new(Mutex::new(BTreeMap::<Vec<u8>, u64>::new())))
            .bench_local_values(|tree| {
                let handles: Vec<_> = (0..threads)
                    .map(|t| {
                        let tree = Arc::clone(&tree);
                        thread::spawn(move || {
                            let base = t * OPS_PER_THREAD;
                            for i in 0..OPS_PER_THREAD {
                                let key = ((base + i) as u64).to_be_bytes().to_vec();
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

    #[divan::bench(args = [1, 2, 4, 8, 16])]
    fn rwlock_btreemap(bencher: Bencher, threads: usize) {
        bencher
            .with_inputs(|| Arc::new(RwLock::new(BTreeMap::<Vec<u8>, u64>::new())))
            .bench_local_values(|tree| {
                let handles: Vec<_> = (0..threads)
                    .map(|t| {
                        let tree = Arc::clone(&tree);
                        thread::spawn(move || {
                            let base = t * OPS_PER_THREAD;
                            for i in 0..OPS_PER_THREAD {
                                let key = ((base + i) as u64).to_be_bytes().to_vec();
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
// 05: MIXED WORKLOAD - Zipfian Access (Realistic Hot Keys)
// =============================================================================

#[divan::bench_group(name = "05_mixed_zipfian")]
mod mixed_zipfian {
    use super::*;

    const N: usize = 100_000;
    const OPS_PER_THREAD: usize = 10_000;
    const WRITE_RATIO: usize = 10;

    #[divan::bench(args = [1, 2, 4, 8, 16])]
    fn masstree(bencher: Bencher, threads: usize) {
        let keys = keys_8b(N);
        let indices = zipfian_indices(N, OPS_PER_THREAD, 42);

        bencher
            .with_inputs(|| Arc::new(setup_masstree(&keys)))
            .bench_local_values(|tree| {
                let handles: Vec<_> = (0..threads)
                    .map(|t| {
                        let tree = Arc::clone(&tree);
                        let keys = keys.clone();
                        let indices = indices.clone();
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
    fn mutex_btreemap(bencher: Bencher, threads: usize) {
        let keys = keys_8b(N);
        let indices = zipfian_indices(N, OPS_PER_THREAD, 42);

        bencher
            .with_inputs(|| Arc::new(setup_mutex_btreemap(&keys)))
            .bench_local_values(|tree| {
                let handles: Vec<_> = (0..threads)
                    .map(|t| {
                        let tree = Arc::clone(&tree);
                        let keys = keys.clone();
                        let indices = indices.clone();
                        thread::spawn(move || {
                            let mut sum = 0u64;
                            let offset = t * 7919;

                            for i in 0..OPS_PER_THREAD {
                                let idx = indices[(i + offset) % indices.len()];
                                if i % WRITE_RATIO == 0 {
                                    let mut guard = tree.lock().unwrap();
                                    guard.insert(keys[idx].clone(), i as u64);
                                } else {
                                    let guard = tree.lock().unwrap();
                                    if let Some(&v) = guard.get(&keys[idx]) {
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

    #[divan::bench(args = [1, 2, 4, 8, 16])]
    fn rwlock_btreemap(bencher: Bencher, threads: usize) {
        let keys = keys_8b(N);
        let indices = zipfian_indices(N, OPS_PER_THREAD, 42);

        bencher
            .with_inputs(|| Arc::new(setup_rwlock_btreemap(&keys)))
            .bench_local_values(|tree| {
                let handles: Vec<_> = (0..threads)
                    .map(|t| {
                        let tree = Arc::clone(&tree);
                        let keys = keys.clone();
                        let indices = indices.clone();
                        thread::spawn(move || {
                            let mut sum = 0u64;
                            let offset = t * 7919;

                            for i in 0..OPS_PER_THREAD {
                                let idx = indices[(i + offset) % indices.len()];
                                if i % WRITE_RATIO == 0 {
                                    let mut guard = tree.write().unwrap();
                                    guard.insert(keys[idx].clone(), i as u64);
                                } else {
                                    let guard = tree.read().unwrap();
                                    if let Some(&v) = guard.get(&keys[idx]) {
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
// 06: MIXED WORKLOAD - Long Keys + Zipfian
// =============================================================================

#[divan::bench_group(name = "06_mixed_long_keys_zipfian")]
mod mixed_long_keys_zipfian {
    use super::*;

    const N: usize = 50_000;
    const OPS_PER_THREAD: usize = 5000;
    const WRITE_RATIO: usize = 10;

    #[divan::bench(args = [1, 2, 4, 8, 16])]
    fn masstree_32b(bencher: Bencher, threads: usize) {
        let keys = keys_32b(N);
        let indices = zipfian_indices(N, OPS_PER_THREAD, 42);

        bencher
            .with_inputs(|| Arc::new(setup_masstree(&keys)))
            .bench_local_values(|tree| {
                let handles: Vec<_> = (0..threads)
                    .map(|t| {
                        let tree = Arc::clone(&tree);
                        let keys = keys.clone();
                        let indices = indices.clone();
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
    fn mutex_btreemap_32b(bencher: Bencher, threads: usize) {
        let keys = keys_32b(N);
        let indices = zipfian_indices(N, OPS_PER_THREAD, 42);

        bencher
            .with_inputs(|| Arc::new(setup_mutex_btreemap(&keys)))
            .bench_local_values(|tree| {
                let handles: Vec<_> = (0..threads)
                    .map(|t| {
                        let tree = Arc::clone(&tree);
                        let keys = keys.clone();
                        let indices = indices.clone();
                        thread::spawn(move || {
                            let mut sum = 0u64;
                            let offset = t * 7919;

                            for i in 0..OPS_PER_THREAD {
                                let idx = indices[(i + offset) % indices.len()];
                                if i % WRITE_RATIO == 0 {
                                    let mut guard = tree.lock().unwrap();
                                    guard.insert(keys[idx].clone(), i as u64);
                                } else {
                                    let guard = tree.lock().unwrap();
                                    if let Some(&v) = guard.get(&keys[idx]) {
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

    #[divan::bench(args = [1, 2, 4, 8, 16])]
    fn rwlock_btreemap_32b(bencher: Bencher, threads: usize) {
        let keys = keys_32b(N);
        let indices = zipfian_indices(N, OPS_PER_THREAD, 42);

        bencher
            .with_inputs(|| Arc::new(setup_rwlock_btreemap(&keys)))
            .bench_local_values(|tree| {
                let handles: Vec<_> = (0..threads)
                    .map(|t| {
                        let tree = Arc::clone(&tree);
                        let keys = keys.clone();
                        let indices = indices.clone();
                        thread::spawn(move || {
                            let mut sum = 0u64;
                            let offset = t * 7919;

                            for i in 0..OPS_PER_THREAD {
                                let idx = indices[(i + offset) % indices.len()];
                                if i % WRITE_RATIO == 0 {
                                    let mut guard = tree.write().unwrap();
                                    guard.insert(keys[idx].clone(), i as u64);
                                } else {
                                    let guard = tree.read().unwrap();
                                    if let Some(&v) = guard.get(&keys[idx]) {
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
// 07: SINGLE HOT KEY - Maximum Contention
// =============================================================================

#[divan::bench_group(name = "07_single_hot_key")]
mod single_hot_key {
    use super::*;

    const N: usize = 10_000;
    const OPS_PER_THREAD: usize = 10_000;

    #[divan::bench(args = [2, 4, 8, 16])]
    fn masstree(bencher: Bencher, threads: usize) {
        let keys = keys_8b(N);
        let hot_key = keys[N / 2].clone();

        bencher
            .with_inputs(|| Arc::new(setup_masstree(&keys)))
            .bench_local_values(|tree| {
                let handles: Vec<_> = (0..threads)
                    .map(|t| {
                        let tree = Arc::clone(&tree);
                        let hot_key = hot_key.clone();
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
    fn mutex_btreemap(bencher: Bencher, threads: usize) {
        let keys = keys_8b(N);
        let hot_key = keys[N / 2].clone();

        bencher
            .with_inputs(|| Arc::new(setup_mutex_btreemap(&keys)))
            .bench_local_values(|tree| {
                let handles: Vec<_> = (0..threads)
                    .map(|t| {
                        let tree = Arc::clone(&tree);
                        let hot_key = hot_key.clone();
                        thread::spawn(move || {
                            let mut sum = 0u64;

                            for i in 0..OPS_PER_THREAD {
                                if i % 10 == 0 {
                                    let mut guard = tree.lock().unwrap();
                                    guard.insert(hot_key.clone(), (t * OPS_PER_THREAD + i) as u64);
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

    #[divan::bench(args = [2, 4, 8, 16])]
    fn rwlock_btreemap(bencher: Bencher, threads: usize) {
        let keys = keys_8b(N);
        let hot_key = keys[N / 2].clone();

        bencher
            .with_inputs(|| Arc::new(setup_rwlock_btreemap(&keys)))
            .bench_local_values(|tree| {
                let handles: Vec<_> = (0..threads)
                    .map(|t| {
                        let tree = Arc::clone(&tree);
                        let hot_key = hot_key.clone();
                        thread::spawn(move || {
                            let mut sum = 0u64;

                            for i in 0..OPS_PER_THREAD {
                                if i % 10 == 0 {
                                    let mut guard = tree.write().unwrap();
                                    guard.insert(hot_key.clone(), (t * OPS_PER_THREAD + i) as u64);
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
    use super::*;

    const N: usize = 100_000;
    const OPS_PER_THREAD: usize = 50_000;

    #[divan::bench(args = [1, 2, 4, 8, 16, 32])]
    fn masstree(bencher: Bencher, threads: usize) {
        let keys = keys_8b(N);
        let tree = Arc::new(setup_masstree(&keys));

        bencher
            .counter(divan::counter::ItemsCount::new(threads * OPS_PER_THREAD))
            .bench_local(|| {
                let handles: Vec<_> = (0..threads)
                    .map(|t| {
                        let tree = Arc::clone(&tree);
                        let keys = keys.clone();
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
        let keys = keys_8b(N);
        let tree = Arc::new(setup_mutex_btreemap(&keys));

        bencher
            .counter(divan::counter::ItemsCount::new(threads * OPS_PER_THREAD))
            .bench_local(|| {
                let handles: Vec<_> = (0..threads)
                    .map(|t| {
                        let tree = Arc::clone(&tree);
                        let keys = keys.clone();
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
        let keys = keys_8b(N);
        let tree = Arc::new(setup_rwlock_btreemap(&keys));

        bencher
            .counter(divan::counter::ItemsCount::new(threads * OPS_PER_THREAD))
            .bench_local(|| {
                let handles: Vec<_> = (0..threads)
                    .map(|t| {
                        let tree = Arc::clone(&tree);
                        let keys = keys.clone();
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
