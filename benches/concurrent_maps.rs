//! Comparison benchmarks: Concurrent Ordered Map Implementations
//!
//! Compares MassTree against other concurrent ordered map crates:
//! - `crossbeam-skiplist::SkipMap` - Lock-free skip list (truly concurrent)
//! - `indexset::concurrent::map::BTreeMap` - Concurrent B-tree
//!
//! ## Benchmark Design Philosophy
//!
//! These benchmarks aim for objectivity by testing:
//! - **Variable key sizes**: 8, 16, 24, 32 bytes (MassTree optimizes for ≤8)
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

#![expect(clippy::indexing_slicing)]

// Use alternative allocator if feature is enabled
#[cfg(feature = "mimalloc")]
#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;

use crossbeam_skiplist::SkipMap;
use divan::{Bencher, black_box};
use indexset::concurrent::map::BTreeMap as IndexSetBTreeMap;
use masstree::MassTree;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
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

/// Generate 24-byte keys (3 MassTree layers)
fn keys_24b(n: usize) -> Vec<Vec<u8>> {
    (0..n)
        .map(|i| {
            let mut key = vec![0u8; 24];
            key[0..8].copy_from_slice(&(i as u64).to_be_bytes());
            key[8..16]
                .copy_from_slice(&((i as u64).wrapping_mul(0x517cc1b727220a95)).to_be_bytes());
            key[16..24]
                .copy_from_slice(&((i as u64).wrapping_mul(0x9e3779b97f4a7c15)).to_be_bytes());
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
/// Uses s=1.0 (standard Zipf), approximated via rejection sampling
fn zipfian_indices(n: usize, count: usize, seed: u64) -> Vec<usize> {
    let mut indices = Vec::with_capacity(count);
    let mut state = seed;

    for _ in 0..count {
        // Simple LCG for deterministic randomness
        state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
        let u = (state >> 33) as f64 / (1u64 << 31) as f64;

        // Approximate Zipfian: index = floor(n^(1-u)) - 1
        // This gives heavy bias toward low indices (hot keys)
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

fn setup_skipmap(keys: &[Vec<u8>]) -> SkipMap<Vec<u8>, u64> {
    let map = SkipMap::new();
    for (i, key) in keys.iter().enumerate() {
        map.insert(key.clone(), i as u64);
    }
    map
}

fn setup_indexset(keys: &[Vec<u8>]) -> IndexSetBTreeMap<Vec<u8>, u64> {
    let map = IndexSetBTreeMap::new();
    for (i, key) in keys.iter().enumerate() {
        map.insert(key.clone(), i as u64);
    }
    map
}

// =============================================================================
// 01: SINGLE-THREADED GET - Variable Key Sizes
// =============================================================================

#[divan::bench_group(name = "01_get_by_key_size")]
mod get_by_key_size {
    use super::*;

    const N: usize = 10_000;

    #[divan::bench(args = ["8B", "16B", "24B", "32B"])]
    fn masstree(bencher: Bencher, key_size: &str) {
        let keys = match key_size {
            "8B" => keys_8b(N),
            "16B" => keys_16b(N),
            "24B" => keys_24b(N),
            "32B" => keys_32b(N),
            _ => unreachable!(),
        };
        let tree = setup_masstree(&keys);
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

    #[divan::bench(args = ["8B", "16B", "24B", "32B"])]
    fn skipmap(bencher: Bencher, key_size: &str) {
        let keys = match key_size {
            "8B" => keys_8b(N),
            "16B" => keys_16b(N),
            "24B" => keys_24b(N),
            "32B" => keys_32b(N),
            _ => unreachable!(),
        };
        let map = setup_skipmap(&keys);
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

    #[divan::bench(args = ["8B", "16B", "24B", "32B"])]
    fn indexset(bencher: Bencher, key_size: &str) {
        let keys = match key_size {
            "8B" => keys_8b(N),
            "16B" => keys_16b(N),
            "24B" => keys_24b(N),
            "32B" => keys_32b(N),
            _ => unreachable!(),
        };
        let map = setup_indexset(&keys);
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
}

// =============================================================================
// 02: SINGLE-THREADED INSERT - Variable Key Sizes
// =============================================================================

#[divan::bench_group(name = "02_insert_by_key_size")]
mod insert_by_key_size {
    use super::*;

    const N: usize = 1000;

    #[divan::bench(args = ["8B", "16B", "24B", "32B"])]
    fn masstree(bencher: Bencher, key_size: &str) {
        let keys = match key_size {
            "8B" => keys_8b(N),
            "16B" => keys_16b(N),
            "24B" => keys_24b(N),
            "32B" => keys_32b(N),
            _ => unreachable!(),
        };

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

    #[divan::bench(args = ["8B", "16B", "24B", "32B"])]
    fn skipmap(bencher: Bencher, key_size: &str) {
        let keys = match key_size {
            "8B" => keys_8b(N),
            "16B" => keys_16b(N),
            "24B" => keys_24b(N),
            "32B" => keys_32b(N),
            _ => unreachable!(),
        };

        bencher
            .with_inputs(|| keys.clone())
            .bench_local_values(|keys| {
                let map = SkipMap::<Vec<u8>, u64>::new();
                for (i, key) in keys.iter().enumerate() {
                    map.insert(key.clone(), i as u64);
                }
                map
            });
    }

    #[divan::bench(args = ["8B", "16B", "24B", "32B"])]
    fn indexset(bencher: Bencher, key_size: &str) {
        let keys = match key_size {
            "8B" => keys_8b(N),
            "16B" => keys_16b(N),
            "24B" => keys_24b(N),
            "32B" => keys_32b(N),
            _ => unreachable!(),
        };

        bencher
            .with_inputs(|| keys.clone())
            .bench_local_values(|keys| {
                let map = IndexSetBTreeMap::<Vec<u8>, u64>::new();
                for (i, key) in keys.iter().enumerate() {
                    map.insert(key.clone(), i as u64);
                }
                map
            });
    }
}

// =============================================================================
// 03: CONCURRENT READS - Thread Scaling (8-byte keys)
// =============================================================================

#[divan::bench_group(name = "03_concurrent_reads_scaling")]
mod concurrent_reads_scaling {
    use super::*;

    const N: usize = 100_000;
    const OPS_PER_THREAD: usize = 10_000;

    #[divan::bench(args = [1, 2, 4, 8, 16, 32])]
    fn masstree(bencher: Bencher, threads: usize) {
        let keys = keys_8b(N);
        let tree = Arc::new(setup_masstree(&keys));
        let indices: Vec<usize> = uniform_indices(N, OPS_PER_THREAD, 42);

        bencher.bench_local(|| {
            let handles: Vec<_> = (0..threads)
                .map(|t| {
                    let tree = Arc::clone(&tree);
                    let keys = keys.clone();
                    let indices = indices.clone();
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
        let keys = keys_8b(N);
        let map = Arc::new(setup_skipmap(&keys));
        let indices: Vec<usize> = uniform_indices(N, OPS_PER_THREAD, 42);

        bencher.bench_local(|| {
            let handles: Vec<_> = (0..threads)
                .map(|t| {
                    let map = Arc::clone(&map);
                    let keys = keys.clone();
                    let indices = indices.clone();
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
        let keys = keys_8b(N);
        let map = Arc::new(setup_indexset(&keys));
        let indices: Vec<usize> = uniform_indices(N, OPS_PER_THREAD, 42);

        bencher.bench_local(|| {
            let handles: Vec<_> = (0..threads)
                .map(|t| {
                    let map = Arc::clone(&map);
                    let keys = keys.clone();
                    let indices = indices.clone();
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
// 04: CONCURRENT READS - Long Keys (32-byte)
// =============================================================================

#[divan::bench_group(name = "04_concurrent_reads_long_keys")]
mod concurrent_reads_long_keys {
    use super::*;

    const N: usize = 50_000;
    const OPS_PER_THREAD: usize = 5000;

    #[divan::bench(args = [1, 2, 4, 8, 16])]
    fn masstree_32b(bencher: Bencher, threads: usize) {
        let keys = keys_32b(N);
        let tree = Arc::new(setup_masstree(&keys));
        let indices: Vec<usize> = uniform_indices(N, OPS_PER_THREAD, 42);

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
    fn skipmap_32b(bencher: Bencher, threads: usize) {
        let keys = keys_32b(N);
        let map = Arc::new(setup_skipmap(&keys));
        let indices: Vec<usize> = uniform_indices(N, OPS_PER_THREAD, 42);

        bencher.bench_local(|| {
            let handles: Vec<_> = (0..threads)
                .map(|t| {
                    let map = Arc::clone(&map);
                    let keys = keys.clone();
                    let indices = indices.clone();
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
        let keys = keys_32b(N);
        let map = Arc::new(setup_indexset(&keys));
        let indices: Vec<usize> = uniform_indices(N, OPS_PER_THREAD, 42);

        bencher.bench_local(|| {
            let handles: Vec<_> = (0..threads)
                .map(|t| {
                    let map = Arc::clone(&map);
                    let keys = keys.clone();
                    let indices = indices.clone();
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
// 05: CONCURRENT WRITES - Disjoint Ranges (Low Contention)
// =============================================================================

#[divan::bench_group(name = "05_concurrent_writes_disjoint")]
mod concurrent_writes_disjoint {
    use super::*;

    const OPS_PER_THREAD: usize = 1000;

    #[divan::bench(args = [1, 2, 4, 8, 16])]
    fn masstree(bencher: Bencher, threads: usize) {
        bencher
            .with_inputs(MassTree::<u64>::new)
            .bench_local_values(|tree| {
                let tree = Arc::new(tree);
                let handles: Vec<_> = (0..threads)
                    .map(|t| {
                        let tree = Arc::clone(&tree);
                        thread::spawn(move || {
                            let guard = tree.guard();
                            // Each thread writes to disjoint range
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
    fn skipmap(bencher: Bencher, threads: usize) {
        bencher
            .with_inputs(SkipMap::<Vec<u8>, u64>::new)
            .bench_local_values(|map| {
                let map = Arc::new(map);
                let handles: Vec<_> = (0..threads)
                    .map(|t| {
                        let map = Arc::clone(&map);
                        thread::spawn(move || {
                            let base = t * OPS_PER_THREAD;
                            for i in 0..OPS_PER_THREAD {
                                let key = ((base + i) as u64).to_be_bytes().to_vec();
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

    #[divan::bench(args = [1, 2, 4, 8, 16])]
    fn indexset(bencher: Bencher, threads: usize) {
        bencher
            .with_inputs(IndexSetBTreeMap::<Vec<u8>, u64>::new)
            .bench_local_values(|map| {
                let map = Arc::new(map);
                let handles: Vec<_> = (0..threads)
                    .map(|t| {
                        let map = Arc::clone(&map);
                        thread::spawn(move || {
                            let base = t * OPS_PER_THREAD;
                            for i in 0..OPS_PER_THREAD {
                                let key = ((base + i) as u64).to_be_bytes().to_vec();
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
// 06: CONCURRENT WRITES - Overlapping Keys (High Contention)
// =============================================================================

#[divan::bench_group(name = "06_concurrent_writes_contention")]
mod concurrent_writes_contention {
    use super::*;

    const KEY_SPACE: usize = 1000; // Small key space = high contention
    const OPS_PER_THREAD: usize = 5000;

    #[divan::bench(args = [2, 4, 8, 16])]
    fn masstree(bencher: Bencher, threads: usize) {
        let keys = keys_8b(KEY_SPACE);

        bencher
            .with_inputs(|| Arc::new(setup_masstree(&keys)))
            .bench_local_values(|tree| {
                let counter = Arc::new(AtomicUsize::new(0));
                let handles: Vec<_> = (0..threads)
                    .map(|t| {
                        let tree = Arc::clone(&tree);
                        let keys = keys.clone();
                        let counter = Arc::clone(&counter);
                        thread::spawn(move || {
                            let guard = tree.guard();
                            let mut state = (t as u64).wrapping_mul(0x517cc1b727220a95);
                            for _ in 0..OPS_PER_THREAD {
                                // Random key from shared pool
                                state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
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
    fn skipmap(bencher: Bencher, threads: usize) {
        let keys = keys_8b(KEY_SPACE);

        bencher
            .with_inputs(|| Arc::new(setup_skipmap(&keys)))
            .bench_local_values(|map| {
                let counter = Arc::new(AtomicUsize::new(0));
                let handles: Vec<_> = (0..threads)
                    .map(|t| {
                        let map = Arc::clone(&map);
                        let keys = keys.clone();
                        let counter = Arc::clone(&counter);
                        thread::spawn(move || {
                            let mut state = (t as u64).wrapping_mul(0x517cc1b727220a95);
                            for _ in 0..OPS_PER_THREAD {
                                state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
                                let idx = (state as usize) % keys.len();
                                let val = counter.fetch_add(1, Ordering::Relaxed) as u64;
                                map.insert(keys[idx].clone(), val);
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

    #[divan::bench(args = [2, 4, 8, 16])]
    fn indexset(bencher: Bencher, threads: usize) {
        let keys = keys_8b(KEY_SPACE);

        bencher
            .with_inputs(|| Arc::new(setup_indexset(&keys)))
            .bench_local_values(|map| {
                let counter = Arc::new(AtomicUsize::new(0));
                let handles: Vec<_> = (0..threads)
                    .map(|t| {
                        let map = Arc::clone(&map);
                        let keys = keys.clone();
                        let counter = Arc::clone(&counter);
                        thread::spawn(move || {
                            let mut state = (t as u64).wrapping_mul(0x517cc1b727220a95);
                            for _ in 0..OPS_PER_THREAD {
                                state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
                                let idx = (state as usize) % keys.len();
                                let val = counter.fetch_add(1, Ordering::Relaxed) as u64;
                                map.insert(keys[idx].clone(), val);
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
// 07: MIXED WORKLOAD - Zipfian Access (Realistic Hot Keys)
// =============================================================================

#[divan::bench_group(name = "07_mixed_zipfian")]
mod mixed_zipfian {
    use super::*;

    const N: usize = 100_000;
    const OPS_PER_THREAD: usize = 10_000;
    const WRITE_RATIO: usize = 10; // 10% writes

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
    fn skipmap(bencher: Bencher, threads: usize) {
        let keys = keys_8b(N);
        let indices = zipfian_indices(N, OPS_PER_THREAD, 42);

        bencher
            .with_inputs(|| Arc::new(setup_skipmap(&keys)))
            .bench_local_values(|map| {
                let handles: Vec<_> = (0..threads)
                    .map(|t| {
                        let map = Arc::clone(&map);
                        let keys = keys.clone();
                        let indices = indices.clone();
                        thread::spawn(move || {
                            let mut sum = 0u64;
                            let offset = t * 7919;

                            for i in 0..OPS_PER_THREAD {
                                let idx = indices[(i + offset) % indices.len()];
                                if i % WRITE_RATIO == 0 {
                                    map.insert(keys[idx].clone(), i as u64);
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
    fn indexset(bencher: Bencher, threads: usize) {
        let keys = keys_8b(N);
        let indices = zipfian_indices(N, OPS_PER_THREAD, 42);

        bencher
            .with_inputs(|| Arc::new(setup_indexset(&keys)))
            .bench_local_values(|map| {
                let handles: Vec<_> = (0..threads)
                    .map(|t| {
                        let map = Arc::clone(&map);
                        let keys = keys.clone();
                        let indices = indices.clone();
                        thread::spawn(move || {
                            let mut sum = 0u64;
                            let offset = t * 7919;

                            for i in 0..OPS_PER_THREAD {
                                let idx = indices[(i + offset) % indices.len()];
                                if i % WRITE_RATIO == 0 {
                                    map.insert(keys[idx].clone(), i as u64);
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
}

// =============================================================================
// 08: MIXED WORKLOAD - Uniform Random (No Hot Keys)
// =============================================================================

#[divan::bench_group(name = "08_mixed_uniform")]
mod mixed_uniform {
    use super::*;

    const N: usize = 100_000;
    const OPS_PER_THREAD: usize = 10_000;
    const WRITE_RATIO: usize = 10; // 10% writes

    #[divan::bench(args = [1, 2, 4, 8, 16])]
    fn masstree(bencher: Bencher, threads: usize) {
        let keys = keys_8b(N);
        let indices = uniform_indices(N, OPS_PER_THREAD, 42);

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
    fn skipmap(bencher: Bencher, threads: usize) {
        let keys = keys_8b(N);
        let indices = uniform_indices(N, OPS_PER_THREAD, 42);

        bencher
            .with_inputs(|| Arc::new(setup_skipmap(&keys)))
            .bench_local_values(|map| {
                let handles: Vec<_> = (0..threads)
                    .map(|t| {
                        let map = Arc::clone(&map);
                        let keys = keys.clone();
                        let indices = indices.clone();
                        thread::spawn(move || {
                            let mut sum = 0u64;
                            let offset = t * 7919;

                            for i in 0..OPS_PER_THREAD {
                                let idx = indices[(i + offset) % indices.len()];
                                if i % WRITE_RATIO == 0 {
                                    map.insert(keys[idx].clone(), i as u64);
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
    fn indexset(bencher: Bencher, threads: usize) {
        let keys = keys_8b(N);
        let indices = uniform_indices(N, OPS_PER_THREAD, 42);

        bencher
            .with_inputs(|| Arc::new(setup_indexset(&keys)))
            .bench_local_values(|map| {
                let handles: Vec<_> = (0..threads)
                    .map(|t| {
                        let map = Arc::clone(&map);
                        let keys = keys.clone();
                        let indices = indices.clone();
                        thread::spawn(move || {
                            let mut sum = 0u64;
                            let offset = t * 7919;

                            for i in 0..OPS_PER_THREAD {
                                let idx = indices[(i + offset) % indices.len()];
                                if i % WRITE_RATIO == 0 {
                                    map.insert(keys[idx].clone(), i as u64);
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
}

// =============================================================================
// 09: MIXED WORKLOAD - Long Keys + Zipfian (Worst Case for MassTree)
// =============================================================================

#[divan::bench_group(name = "09_mixed_long_keys_zipfian")]
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
    fn skipmap_32b(bencher: Bencher, threads: usize) {
        let keys = keys_32b(N);
        let indices = zipfian_indices(N, OPS_PER_THREAD, 42);

        bencher
            .with_inputs(|| Arc::new(setup_skipmap(&keys)))
            .bench_local_values(|map| {
                let handles: Vec<_> = (0..threads)
                    .map(|t| {
                        let map = Arc::clone(&map);
                        let keys = keys.clone();
                        let indices = indices.clone();
                        thread::spawn(move || {
                            let mut sum = 0u64;
                            let offset = t * 7919;

                            for i in 0..OPS_PER_THREAD {
                                let idx = indices[(i + offset) % indices.len()];
                                if i % WRITE_RATIO == 0 {
                                    map.insert(keys[idx].clone(), i as u64);
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
        let keys = keys_32b(N);
        let indices = zipfian_indices(N, OPS_PER_THREAD, 42);

        bencher
            .with_inputs(|| Arc::new(setup_indexset(&keys)))
            .bench_local_values(|map| {
                let handles: Vec<_> = (0..threads)
                    .map(|t| {
                        let map = Arc::clone(&map);
                        let keys = keys.clone();
                        let indices = indices.clone();
                        thread::spawn(move || {
                            let mut sum = 0u64;
                            let offset = t * 7919;

                            for i in 0..OPS_PER_THREAD {
                                let idx = indices[(i + offset) % indices.len()];
                                if i % WRITE_RATIO == 0 {
                                    map.insert(keys[idx].clone(), i as u64);
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
}

// =============================================================================
// 10: SINGLE HOT KEY - Maximum Contention
// =============================================================================

#[divan::bench_group(name = "10_single_hot_key")]
mod single_hot_key {
    use super::*;

    const N: usize = 10_000;
    const OPS_PER_THREAD: usize = 10_000;

    #[divan::bench(args = [2, 4, 8, 16])]
    fn masstree(bencher: Bencher, threads: usize) {
        let keys = keys_8b(N);
        let hot_key = keys[N / 2].clone(); // Single hot key

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
    fn skipmap(bencher: Bencher, threads: usize) {
        let keys = keys_8b(N);
        let hot_key = keys[N / 2].clone();

        bencher
            .with_inputs(|| Arc::new(setup_skipmap(&keys)))
            .bench_local_values(|map| {
                let handles: Vec<_> = (0..threads)
                    .map(|t| {
                        let map = Arc::clone(&map);
                        let hot_key = hot_key.clone();
                        thread::spawn(move || {
                            let mut sum = 0u64;

                            for i in 0..OPS_PER_THREAD {
                                if i % 10 == 0 {
                                    map.insert(hot_key.clone(), (t * OPS_PER_THREAD + i) as u64);
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
        let keys = keys_8b(N);
        let hot_key = keys[N / 2].clone();

        bencher
            .with_inputs(|| Arc::new(setup_indexset(&keys)))
            .bench_local_values(|map| {
                let handles: Vec<_> = (0..threads)
                    .map(|t| {
                        let map = Arc::clone(&map);
                        let hot_key = hot_key.clone();
                        thread::spawn(move || {
                            let mut sum = 0u64;

                            for i in 0..OPS_PER_THREAD {
                                if i % 10 == 0 {
                                    map.insert(hot_key.clone(), (t * OPS_PER_THREAD + i) as u64);
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
