//! Comparison benchmarks: `MassTree` vs `DashMap`
//!
//! **Key Differences:**
//! - DashMap: Hash-based, unordered, sharded locks, Send+Sync
//! - MassTree: Trie of B+trees, ordered, per-node locks, Send+Sync
//!
//! **What we're measuring:**
//! - Single-threaded throughput
//! - Multi-threaded concurrent access
//! - Different key types and sizes
//! - Scaling behavior with thread count
//!
//! Run with: `cargo bench --bench dashmap_comparison`

#![expect(clippy::indexing_slicing)]

use std::sync::Arc;
use std::thread;

use dashmap::DashMap;
use divan::{black_box, Bencher};
use masstree::tree::MassTree;

fn main() {
    divan::main();
}

// =============================================================================
// Key Generation Helpers
// =============================================================================

fn sequential_keys(n: usize) -> Vec<Vec<u8>> {
    (0..n).map(|i| (i as u64).to_be_bytes().to_vec()).collect()
}

fn shuffled_indices(n: usize, seed: usize) -> Vec<usize> {
    let mut indices: Vec<usize> = (0..n).collect();
    for i in 0..indices.len() {
        let j = ((i.wrapping_mul(seed)).wrapping_add(17)) % indices.len();
        indices.swap(i, j);
    }
    indices
}

// =============================================================================
// SINGLE-THREADED: Insert
// =============================================================================

#[divan::bench_group(name = "01_insert")]
mod insert {
    use super::*;

    const SIZES: &[usize] = &[100, 1000, 10000];

    #[divan::bench(args = SIZES)]
    fn masstree(bencher: Bencher, n: usize) {
        let keys = sequential_keys(n);
        bencher
            .with_inputs(|| MassTree::<u64>::new())
            .bench_local_values(|mut tree| {
                for (i, key) in keys.iter().enumerate() {
                    tree.insert(black_box(key), black_box(i as u64));
                }
                tree
            });
    }

    #[divan::bench(args = SIZES)]
    fn dashmap(bencher: Bencher, n: usize) {
        let keys = sequential_keys(n);
        bencher
            .with_inputs(DashMap::new)
            .bench_local_values(|map| {
                for (i, key) in keys.iter().enumerate() {
                    map.insert(black_box(key.clone()), black_box(i as u64));
                }
                map
            });
    }
}

// =============================================================================
// SINGLE-THREADED: Get (Hit)
// =============================================================================

#[divan::bench_group(name = "02_get_hit")]
mod get_hit {
    use super::*;

    const SIZES: &[usize] = &[100, 1000, 10000];

    #[divan::bench(args = SIZES)]
    fn masstree(bencher: Bencher, n: usize) {
        let keys = sequential_keys(n);
        let indices = shuffled_indices(n, 42);

        bencher
            .with_inputs(|| {
                let mut tree = MassTree::<u64>::new();
                for (i, key) in keys.iter().enumerate() {
                    tree.insert(key, i as u64);
                }
                tree
            })
            .bench_local_refs(|tree| {
                for &idx in &indices {
                    let _ = black_box(tree.get(black_box(&keys[idx])));
                }
            });
    }

    #[divan::bench(args = SIZES)]
    fn dashmap(bencher: Bencher, n: usize) {
        let keys = sequential_keys(n);
        let indices = shuffled_indices(n, 42);

        bencher
            .with_inputs(|| {
                let map = DashMap::new();
                for (i, key) in keys.iter().enumerate() {
                    map.insert(key.clone(), i as u64);
                }
                map
            })
            .bench_local_refs(|map| {
                for &idx in &indices {
                    let _ = black_box(map.get(black_box(&keys[idx])));
                }
            });
    }
}

// =============================================================================
// SINGLE-THREADED: Get (Miss)
// =============================================================================

#[divan::bench_group(name = "03_get_miss")]
mod get_miss {
    use super::*;

    const SIZES: &[usize] = &[100, 1000, 10000];

    #[divan::bench(args = SIZES)]
    fn masstree(bencher: Bencher, n: usize) {
        let keys = sequential_keys(n);
        let miss_keys: Vec<Vec<u8>> = (n..n * 2)
            .map(|i| (i as u64).to_be_bytes().to_vec())
            .collect();

        bencher
            .with_inputs(|| {
                let mut tree = MassTree::<u64>::new();
                for (i, key) in keys.iter().enumerate() {
                    tree.insert(key, i as u64);
                }
                tree
            })
            .bench_local_refs(|tree| {
                for key in &miss_keys {
                    let _ = black_box(tree.get(black_box(key)));
                }
            });
    }

    #[divan::bench(args = SIZES)]
    fn dashmap(bencher: Bencher, n: usize) {
        let keys = sequential_keys(n);
        let miss_keys: Vec<Vec<u8>> = (n..n * 2)
            .map(|i| (i as u64).to_be_bytes().to_vec())
            .collect();

        bencher
            .with_inputs(|| {
                let map = DashMap::new();
                for (i, key) in keys.iter().enumerate() {
                    map.insert(key.clone(), i as u64);
                }
                map
            })
            .bench_local_refs(|map| {
                for key in &miss_keys {
                    let _ = black_box(map.get(black_box(key)));
                }
            });
    }
}

// =============================================================================
// SINGLE-THREADED: Mixed Workload (90% read, 10% write)
// =============================================================================

#[divan::bench_group(name = "04_mixed_90_10")]
mod mixed {
    use super::*;

    const SIZES: &[usize] = &[100, 1000, 10000];

    #[divan::bench(args = SIZES)]
    fn masstree(bencher: Bencher, n: usize) {
        let keys = sequential_keys(n);
        let indices = shuffled_indices(n, 42);

        bencher
            .with_inputs(|| {
                let mut tree = MassTree::<u64>::new();
                for (i, key) in keys.iter().enumerate() {
                    tree.insert(key, i as u64);
                }
                tree
            })
            .bench_local_values(|mut tree| {
                for (op, &idx) in indices.iter().enumerate() {
                    if op % 10 == 0 {
                        tree.insert(black_box(&keys[idx]), black_box(op as u64));
                    } else {
                        let _ = black_box(tree.get(black_box(&keys[idx])));
                    }
                }
                tree
            });
    }

    #[divan::bench(args = SIZES)]
    fn dashmap(bencher: Bencher, n: usize) {
        let keys = sequential_keys(n);
        let indices = shuffled_indices(n, 42);

        bencher
            .with_inputs(|| {
                let map = DashMap::new();
                for (i, key) in keys.iter().enumerate() {
                    map.insert(key.clone(), i as u64);
                }
                map
            })
            .bench_local_values(|map| {
                for (op, &idx) in indices.iter().enumerate() {
                    if op % 10 == 0 {
                        map.insert(black_box(keys[idx].clone()), black_box(op as u64));
                    } else {
                        let _ = black_box(map.get(black_box(&keys[idx])));
                    }
                }
                map
            });
    }
}

// =============================================================================
// SINGLE-THREADED: Update Existing
// =============================================================================

#[divan::bench_group(name = "05_update_existing")]
mod update {
    use super::*;

    const SIZES: &[usize] = &[100, 1000, 10000];

    #[divan::bench(args = SIZES)]
    fn masstree(bencher: Bencher, n: usize) {
        let keys = sequential_keys(n);
        let indices = shuffled_indices(n, 42);

        bencher
            .with_inputs(|| {
                let mut tree = MassTree::<u64>::new();
                for (i, key) in keys.iter().enumerate() {
                    tree.insert(key, i as u64);
                }
                tree
            })
            .bench_local_values(|mut tree| {
                for (op, &idx) in indices.iter().enumerate() {
                    tree.insert(black_box(&keys[idx]), black_box(op as u64));
                }
                tree
            });
    }

    #[divan::bench(args = SIZES)]
    fn dashmap(bencher: Bencher, n: usize) {
        let keys = sequential_keys(n);
        let indices = shuffled_indices(n, 42);

        bencher
            .with_inputs(|| {
                let map = DashMap::new();
                for (i, key) in keys.iter().enumerate() {
                    map.insert(key.clone(), i as u64);
                }
                map
            })
            .bench_local_values(|map| {
                for (op, &idx) in indices.iter().enumerate() {
                    map.insert(black_box(keys[idx].clone()), black_box(op as u64));
                }
                map
            });
    }
}

// =============================================================================
// SCALING: Throughput vs tree size (100 random gets)
// =============================================================================

#[divan::bench_group(name = "06_scaling_get")]
mod scaling {
    use super::*;

    const SIZES: &[usize] = &[100, 1000, 10000, 100000];

    #[divan::bench(args = SIZES)]
    fn masstree(bencher: Bencher, n: usize) {
        let keys = sequential_keys(n);
        let indices = shuffled_indices(n, 42);

        bencher
            .with_inputs(|| {
                let mut tree = MassTree::<u64>::new();
                for (i, key) in keys.iter().enumerate() {
                    tree.insert(key, i as u64);
                }
                tree
            })
            .bench_local_refs(|tree| {
                for &idx in indices.iter().take(100) {
                    let _ = black_box(tree.get(black_box(&keys[idx])));
                }
            });
    }

    #[divan::bench(args = SIZES)]
    fn dashmap(bencher: Bencher, n: usize) {
        let keys = sequential_keys(n);
        let indices = shuffled_indices(n, 42);

        bencher
            .with_inputs(|| {
                let map = DashMap::new();
                for (i, key) in keys.iter().enumerate() {
                    map.insert(key.clone(), i as u64);
                }
                map
            })
            .bench_local_refs(|map| {
                for &idx in indices.iter().take(100) {
                    let _ = black_box(map.get(black_box(&keys[idx])));
                }
            });
    }
}

// =============================================================================
// MEMORY: Bulk insert 10k
// =============================================================================

#[divan::bench_group(name = "07_bulk_insert_10k")]
mod bulk {
    use super::*;

    #[divan::bench]
    fn masstree(bencher: Bencher) {
        let keys = sequential_keys(10_000);
        bencher.bench_local(|| {
            let mut tree = MassTree::<u64>::new();
            for (i, key) in keys.iter().enumerate() {
                tree.insert(key, i as u64);
            }
            black_box(tree)
        });
    }

    #[divan::bench]
    fn dashmap(bencher: Bencher) {
        let keys = sequential_keys(10_000);
        bencher.bench_local(|| {
            let map = DashMap::new();
            for (i, key) in keys.iter().enumerate() {
                map.insert(key.clone(), i as u64);
            }
            black_box(map)
        });
    }
}

// =============================================================================
// PER-OP LATENCY: Single operations
// =============================================================================

#[divan::bench_group(name = "08_per_op_latency")]
mod latency {
    use super::*;

    #[divan::bench]
    fn masstree_get_1000(bencher: Bencher) {
        let keys = sequential_keys(1000);

        bencher
            .with_inputs(|| {
                let mut tree = MassTree::<u64>::new();
                for (i, key) in keys.iter().enumerate() {
                    tree.insert(key, i as u64);
                }
                (tree, keys[500].clone())
            })
            .bench_local_refs(|(tree, key)| {
                let _ = black_box(tree.get(black_box(key)));
            });
    }

    #[divan::bench]
    fn dashmap_get_1000(bencher: Bencher) {
        let keys = sequential_keys(1000);

        bencher
            .with_inputs(|| {
                let map = DashMap::new();
                for (i, key) in keys.iter().enumerate() {
                    map.insert(key.clone(), i as u64);
                }
                (map, keys[500].clone())
            })
            .bench_local_refs(|(map, key)| {
                let _ = black_box(map.get(black_box(key)));
            });
    }

    #[divan::bench]
    fn masstree_insert_1000(bencher: Bencher) {
        let keys = sequential_keys(1000);

        bencher
            .with_inputs(|| {
                let mut tree = MassTree::<u64>::new();
                for (i, key) in keys.iter().enumerate() {
                    tree.insert(key, i as u64);
                }
                tree
            })
            .bench_local_values(|mut tree| {
                tree.insert(black_box(&keys[500]), black_box(999u64));
                tree
            });
    }

    #[divan::bench]
    fn dashmap_insert_1000(bencher: Bencher) {
        let keys = sequential_keys(1000);

        bencher
            .with_inputs(|| {
                let map = DashMap::new();
                for (i, key) in keys.iter().enumerate() {
                    map.insert(key.clone(), i as u64);
                }
                map
            })
            .bench_local_values(|map| {
                map.insert(black_box(keys[500].clone()), black_box(999u64));
                map
            });
    }
}

// =============================================================================
// MULTI-THREADED: Concurrent Reads
// =============================================================================

#[divan::bench_group(name = "09_concurrent_read")]
mod concurrent_read {
    use super::*;

    const THREAD_COUNTS: &[usize] = &[2, 4, 8];
    const TREE_SIZE: usize = 10_000;
    const OPS_PER_THREAD: usize = 10_000;

    #[divan::bench(args = THREAD_COUNTS)]
    fn masstree(bencher: Bencher, num_threads: usize) {
        let keys = sequential_keys(TREE_SIZE);

        bencher
            .with_inputs(|| {
                let mut tree = MassTree::<u64>::new();
                for (i, key) in keys.iter().enumerate() {
                    tree.insert(key, i as u64);
                }
                Arc::new(tree)
            })
            .bench_values(|tree| {
                let handles: Vec<_> = (0..num_threads)
                    .map(|t| {
                        let tree = Arc::clone(&tree);
                        let keys = keys.clone();
                        thread::spawn(move || {
                            for i in 0..OPS_PER_THREAD {
                                let idx = (t * 1000 + i) % keys.len();
                                black_box(tree.get(&keys[idx]));
                            }
                        })
                    })
                    .collect();

                for h in handles {
                    h.join().unwrap();
                }
            });
    }

    #[divan::bench(args = THREAD_COUNTS)]
    fn dashmap(bencher: Bencher, num_threads: usize) {
        let keys = sequential_keys(TREE_SIZE);

        bencher
            .with_inputs(|| {
                let map = DashMap::new();
                for (i, key) in keys.iter().enumerate() {
                    map.insert(key.clone(), i as u64);
                }
                Arc::new(map)
            })
            .bench_values(|map| {
                let handles: Vec<_> = (0..num_threads)
                    .map(|t| {
                        let map = Arc::clone(&map);
                        let keys = keys.clone();
                        thread::spawn(move || {
                            for i in 0..OPS_PER_THREAD {
                                let idx = (t * 1000 + i) % keys.len();
                                black_box(map.get(&keys[idx]));
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
// MULTI-THREADED: Concurrent Mixed (90% read, 10% write)
// =============================================================================

#[divan::bench_group(name = "10_concurrent_mixed")]
mod concurrent_mixed {
    use super::*;

    const THREAD_COUNTS: &[usize] = &[2, 4, 8];
    const TREE_SIZE: usize = 10_000;
    const OPS_PER_THREAD: usize = 10_000;

    #[divan::bench(args = THREAD_COUNTS)]
    fn masstree(bencher: Bencher, num_threads: usize) {
        let keys = sequential_keys(TREE_SIZE);

        bencher
            .with_inputs(|| {
                let mut tree = MassTree::<u64>::new();
                for (i, key) in keys.iter().enumerate() {
                    tree.insert(key, i as u64);
                }
                Arc::new(tree)
            })
            .bench_values(|tree| {
                let handles: Vec<_> = (0..num_threads)
                    .map(|t| {
                        let tree = Arc::clone(&tree);
                        let keys = keys.clone();
                        thread::spawn(move || {
                            let guard = tree.guard();
                            for i in 0..OPS_PER_THREAD {
                                let idx = (t * 1000 + i) % keys.len();
                                if i % 10 == 0 {
                                    // 10% writes
                                    let _ = tree.insert_with_guard(
                                        &keys[idx],
                                        (t * OPS_PER_THREAD + i) as u64,
                                        &guard,
                                    );
                                } else {
                                    // 90% reads
                                    black_box(tree.get(&keys[idx]));
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

    #[divan::bench(args = THREAD_COUNTS)]
    fn dashmap(bencher: Bencher, num_threads: usize) {
        let keys = sequential_keys(TREE_SIZE);

        bencher
            .with_inputs(|| {
                let map = DashMap::new();
                for (i, key) in keys.iter().enumerate() {
                    map.insert(key.clone(), i as u64);
                }
                Arc::new(map)
            })
            .bench_values(|map| {
                let handles: Vec<_> = (0..num_threads)
                    .map(|t| {
                        let map = Arc::clone(&map);
                        let keys = keys.clone();
                        thread::spawn(move || {
                            for i in 0..OPS_PER_THREAD {
                                let idx = (t * 1000 + i) % keys.len();
                                if i % 10 == 0 {
                                    // 10% writes
                                    map.insert(keys[idx].clone(), (t * OPS_PER_THREAD + i) as u64);
                                } else {
                                    // 90% reads
                                    black_box(map.get(&keys[idx]));
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
// MULTI-THREADED: Read-Write Contention (same keys)
// =============================================================================

#[divan::bench_group(name = "11_contention")]
mod contention {
    use super::*;

    const THREAD_COUNTS: &[usize] = &[2, 4, 8];
    const HOTSPOT_SIZE: usize = 100; // Small set = high contention
    const OPS_PER_THREAD: usize = 10_000;

    #[divan::bench(args = THREAD_COUNTS)]
    fn masstree(bencher: Bencher, num_threads: usize) {
        let keys = sequential_keys(HOTSPOT_SIZE);

        bencher
            .with_inputs(|| {
                let mut tree = MassTree::<u64>::new();
                for (i, key) in keys.iter().enumerate() {
                    tree.insert(key, i as u64);
                }
                Arc::new(tree)
            })
            .bench_values(|tree| {
                let handles: Vec<_> = (0..num_threads)
                    .map(|t| {
                        let tree = Arc::clone(&tree);
                        let keys = keys.clone();
                        thread::spawn(move || {
                            let guard = tree.guard();
                            for i in 0..OPS_PER_THREAD {
                                let idx = i % keys.len();
                                if t % 2 == 0 {
                                    // Half threads read
                                    black_box(tree.get(&keys[idx]));
                                } else {
                                    // Half threads write
                                    let _ = tree.insert_with_guard(&keys[idx], i as u64, &guard);
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

    #[divan::bench(args = THREAD_COUNTS)]
    fn dashmap(bencher: Bencher, num_threads: usize) {
        let keys = sequential_keys(HOTSPOT_SIZE);

        bencher
            .with_inputs(|| {
                let map = DashMap::new();
                for (i, key) in keys.iter().enumerate() {
                    map.insert(key.clone(), i as u64);
                }
                Arc::new(map)
            })
            .bench_values(|map| {
                let handles: Vec<_> = (0..num_threads)
                    .map(|t| {
                        let map = Arc::clone(&map);
                        let keys = keys.clone();
                        thread::spawn(move || {
                            for i in 0..OPS_PER_THREAD {
                                let idx = i % keys.len();
                                if t % 2 == 0 {
                                    // Half threads read
                                    black_box(map.get(&keys[idx]));
                                } else {
                                    // Half threads write
                                    map.insert(keys[idx].clone(), i as u64);
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
// MULTI-THREADED: Scaling (throughput vs thread count)
// =============================================================================

#[divan::bench_group(name = "12_thread_scaling")]
mod thread_scaling {
    use super::*;

    const THREAD_COUNTS: &[usize] = &[1, 2, 4, 8, 16];
    const TREE_SIZE: usize = 100_000;
    const OPS_PER_THREAD: usize = 10_000;

    #[divan::bench(args = THREAD_COUNTS)]
    fn masstree_read_only(bencher: Bencher, num_threads: usize) {
        let keys = sequential_keys(TREE_SIZE);

        bencher
            .with_inputs(|| {
                let mut tree = MassTree::<u64>::new();
                for (i, key) in keys.iter().enumerate() {
                    tree.insert(key, i as u64);
                }
                Arc::new(tree)
            })
            .bench_values(|tree| {
                let handles: Vec<_> = (0..num_threads)
                    .map(|t| {
                        let tree = Arc::clone(&tree);
                        let keys = keys.clone();
                        thread::spawn(move || {
                            let start = (t * 7919) % keys.len(); // Prime for distribution
                            for i in 0..OPS_PER_THREAD {
                                let idx = (start + i) % keys.len();
                                black_box(tree.get(&keys[idx]));
                            }
                        })
                    })
                    .collect();

                for h in handles {
                    h.join().unwrap();
                }
            });
    }

    #[divan::bench(args = THREAD_COUNTS)]
    fn dashmap_read_only(bencher: Bencher, num_threads: usize) {
        let keys = sequential_keys(TREE_SIZE);

        bencher
            .with_inputs(|| {
                let map = DashMap::new();
                for (i, key) in keys.iter().enumerate() {
                    map.insert(key.clone(), i as u64);
                }
                Arc::new(map)
            })
            .bench_values(|map| {
                let handles: Vec<_> = (0..num_threads)
                    .map(|t| {
                        let map = Arc::clone(&map);
                        let keys = keys.clone();
                        thread::spawn(move || {
                            let start = (t * 7919) % keys.len();
                            for i in 0..OPS_PER_THREAD {
                                let idx = (start + i) % keys.len();
                                black_box(map.get(&keys[idx]));
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
