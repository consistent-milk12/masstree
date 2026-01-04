//! Concurrent 90-10 read/write benchmarks with 64-byte keys.
//!
//! Compares MassTree15 against other concurrent ordered maps under mixed
//! read-write workloads. All benchmarks use 64-byte keys to test multi-layer
//! traversal performance.
//!
//! ## Key Characteristics
//!
//! - 90% reads, 10% writes (realistic workload)
//! - 64-byte keys (8 chunks, tests suffix handling)
//! - Various access patterns: uniform, zipfian, shared prefix
//!
//! ## Running
//!
//! ```bash
//! cargo bench --bench concurrent_read_write
//! cargo bench --bench concurrent_read_write -- 01_
//! ```

#![expect(clippy::unwrap_used)]
#![expect(clippy::pedantic)]
#![expect(clippy::indexing_slicing)]
#![expect(clippy::redundant_locals)]

mod bench_utils;

use bench_utils::{keys, keys_shared_prefix_chunks, uniform_indices, zipfian_indices};
use crossbeam_skiplist::SkipMap;
use divan::{Bencher, black_box};
use indexset::concurrent::map::BTreeMap as IndexSetBTreeMap;
use masstree::MassTree15;
use scc::TreeIndex;
use sdd::Guard as SddGuard;
use std::sync::Arc;
use std::sync::Barrier;
use std::thread;

fn main() {
    divan::main();
}

// =============================================================================
// Constants
// =============================================================================

const KEY_SIZE: usize = 64;
const WRITE_RATIO: usize = 10; // 10% writes, 90% reads

// =============================================================================
// Setup Helpers
// =============================================================================

fn tree_index_upsert_sync(tree: &TreeIndex<[u8; KEY_SIZE], u64>, key: [u8; KEY_SIZE], value: u64) {
    let mut key = key;
    let mut value = value;
    for _ in 0..3 {
        match tree.insert_sync(key, value) {
            Ok(()) => return,
            Err((k, v)) => {
                tree.remove_sync(&k);
                key = k;
                value = v;
            }
        }
    }
    let _ = tree.insert_sync(key, value);
}

fn setup_masstree15(keys: &[[u8; KEY_SIZE]]) -> MassTree15<u64> {
    let tree = MassTree15::new();
    {
        let guard = tree.guard();
        for (i, key) in keys.iter().enumerate() {
            let _ = tree.insert_with_guard(key, i as u64, &guard);
        }
    }
    tree
}

fn setup_skipmap(keys: &[[u8; KEY_SIZE]]) -> SkipMap<[u8; KEY_SIZE], u64> {
    let map = SkipMap::new();
    for (i, key) in keys.iter().enumerate() {
        map.insert(*key, i as u64);
    }
    map
}

fn setup_indexset(keys: &[[u8; KEY_SIZE]]) -> IndexSetBTreeMap<[u8; KEY_SIZE], u64> {
    let map = IndexSetBTreeMap::new();
    for (i, key) in keys.iter().enumerate() {
        map.insert(*key, i as u64);
    }
    map
}

fn setup_tree_index(keys: &[[u8; KEY_SIZE]]) -> TreeIndex<[u8; KEY_SIZE], u64> {
    let tree = TreeIndex::new();
    for (i, key) in keys.iter().enumerate() {
        let _ = tree.insert_sync(*key, i as u64);
    }
    tree
}

// =============================================================================
// 01: MIXED 90-10 - Uniform Access Pattern
// =============================================================================

#[divan::bench_group(name = "01_mixed_90_10_uniform")]
mod mixed_uniform {
    use super::*;

    const N: usize = 200_000;
    const OPS_PER_THREAD: usize = 50_000;

    #[divan::bench(args = [1, 2, 3, 4, 5, 6])]
    fn masstree15(bencher: Bencher, threads: usize) {
        let keys = Arc::new(keys::<KEY_SIZE>(N));
        let indices = Arc::new(uniform_indices(N, OPS_PER_THREAD * threads, 42));

        bencher
            .counter(divan::counter::ItemsCount::new(threads * OPS_PER_THREAD))
            .with_inputs(|| Arc::new(setup_masstree15(keys.as_ref())))
            .bench_local_values(|tree| {
                let start = Arc::new(Barrier::new(threads));
                let handles: Vec<_> = (0..threads)
                    .map(|t| {
                        let tree = Arc::clone(&tree);
                        let keys = Arc::clone(&keys);
                        let indices = Arc::clone(&indices);
                        let start = Arc::clone(&start);
                        thread::spawn(move || {
                            let guard = tree.guard();
                            let mut sum = 0u64;
                            let base = t * OPS_PER_THREAD;
                            start.wait();
                            for i in 0..OPS_PER_THREAD {
                                let idx = indices[base + i];
                                if i % 100 < WRITE_RATIO {
                                    let _ = tree.insert_with_guard(&keys[idx], i as u64, &guard);
                                } else if let Some(v) = tree.get_ref(&keys[idx], &guard) {
                                    sum = sum.wrapping_add(*v);
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

    #[divan::bench(args = [1, 2, 3, 4, 5, 6])]
    fn skipmap(bencher: Bencher, threads: usize) {
        let keys = Arc::new(keys::<KEY_SIZE>(N));
        let indices = Arc::new(uniform_indices(N, OPS_PER_THREAD * threads, 42));

        bencher
            .counter(divan::counter::ItemsCount::new(threads * OPS_PER_THREAD))
            .with_inputs(|| Arc::new(setup_skipmap(keys.as_ref())))
            .bench_local_values(|map| {
                let start = Arc::new(Barrier::new(threads));
                let handles: Vec<_> = (0..threads)
                    .map(|t| {
                        let map = Arc::clone(&map);
                        let keys = Arc::clone(&keys);
                        let indices = Arc::clone(&indices);
                        let start = Arc::clone(&start);
                        thread::spawn(move || {
                            let mut sum = 0u64;
                            let base = t * OPS_PER_THREAD;
                            start.wait();
                            for i in 0..OPS_PER_THREAD {
                                let idx = indices[base + i];
                                if i % 100 < WRITE_RATIO {
                                    map.insert(keys[idx], i as u64);
                                } else if let Some(e) = map.get(&keys[idx]) {
                                    sum = sum.wrapping_add(*e.value());
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

    #[divan::bench(args = [1, 2, 3, 4, 5, 6])]
    fn indexset(bencher: Bencher, threads: usize) {
        let keys = Arc::new(keys::<KEY_SIZE>(N));
        let indices = Arc::new(uniform_indices(N, OPS_PER_THREAD * threads, 42));

        bencher
            .counter(divan::counter::ItemsCount::new(threads * OPS_PER_THREAD))
            .with_inputs(|| Arc::new(setup_indexset(keys.as_ref())))
            .bench_local_values(|map| {
                let start = Arc::new(Barrier::new(threads));
                let handles: Vec<_> = (0..threads)
                    .map(|t| {
                        let map = Arc::clone(&map);
                        let keys = Arc::clone(&keys);
                        let indices = Arc::clone(&indices);
                        let start = Arc::clone(&start);
                        thread::spawn(move || {
                            let mut sum = 0u64;
                            let base = t * OPS_PER_THREAD;
                            start.wait();
                            for i in 0..OPS_PER_THREAD {
                                let idx = indices[base + i];
                                if i % 100 < WRITE_RATIO {
                                    map.insert(keys[idx], i as u64);
                                } else if let Some(r) = map.get(&keys[idx]) {
                                    sum = sum.wrapping_add(r.get().value);
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

    #[divan::bench(args = [1, 2, 3, 4, 5, 6])]
    fn tree_index(bencher: Bencher, threads: usize) {
        let keys = Arc::new(keys::<KEY_SIZE>(N));
        let indices = Arc::new(uniform_indices(N, OPS_PER_THREAD * threads, 42));

        bencher
            .counter(divan::counter::ItemsCount::new(threads * OPS_PER_THREAD))
            .with_inputs(|| Arc::new(setup_tree_index(keys.as_ref())))
            .bench_local_values(|tree| {
                let start = Arc::new(Barrier::new(threads));
                let handles: Vec<_> = (0..threads)
                    .map(|t| {
                        let tree = Arc::clone(&tree);
                        let keys = Arc::clone(&keys);
                        let indices = Arc::clone(&indices);
                        let start = Arc::clone(&start);
                        thread::spawn(move || {
                            let guard = SddGuard::new();
                            let mut sum = 0u64;
                            let base = t * OPS_PER_THREAD;
                            start.wait();
                            for i in 0..OPS_PER_THREAD {
                                let idx = indices[base + i];
                                if i % 100 < WRITE_RATIO {
                                    tree_index_upsert_sync(&tree, keys[idx], i as u64);
                                } else if let Some(v) = tree.peek(&keys[idx], &guard) {
                                    sum = sum.wrapping_add(*v);
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
// 02: MIXED 90-10 - Zipfian Access Pattern (Hot Keys)
// =============================================================================

#[divan::bench_group(name = "02_mixed_90_10_zipfian")]
mod mixed_zipfian {
    use super::*;

    const N: usize = 200_000;
    const OPS_PER_THREAD: usize = 50_000;

    #[divan::bench(args = [1, 2, 3, 4, 5, 6])]
    fn masstree15(bencher: Bencher, threads: usize) {
        let keys = Arc::new(keys::<KEY_SIZE>(N));
        let indices = Arc::new(zipfian_indices(N, OPS_PER_THREAD * threads, 42));

        bencher
            .counter(divan::counter::ItemsCount::new(threads * OPS_PER_THREAD))
            .with_inputs(|| Arc::new(setup_masstree15(keys.as_ref())))
            .bench_local_values(|tree| {
                let start = Arc::new(Barrier::new(threads));
                let handles: Vec<_> = (0..threads)
                    .map(|t| {
                        let tree = Arc::clone(&tree);
                        let keys = Arc::clone(&keys);
                        let indices = Arc::clone(&indices);
                        let start = Arc::clone(&start);
                        thread::spawn(move || {
                            let guard = tree.guard();
                            let mut sum = 0u64;
                            let base = t * OPS_PER_THREAD;
                            start.wait();
                            for i in 0..OPS_PER_THREAD {
                                let idx = indices[base + i];
                                if i % 100 < WRITE_RATIO {
                                    let _ = tree.insert_with_guard(&keys[idx], i as u64, &guard);
                                } else if let Some(v) = tree.get_ref(&keys[idx], &guard) {
                                    sum = sum.wrapping_add(*v);
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

    #[divan::bench(args = [1, 2, 3, 4, 5, 6])]
    fn skipmap(bencher: Bencher, threads: usize) {
        let keys = Arc::new(keys::<KEY_SIZE>(N));
        let indices = Arc::new(zipfian_indices(N, OPS_PER_THREAD * threads, 42));

        bencher
            .counter(divan::counter::ItemsCount::new(threads * OPS_PER_THREAD))
            .with_inputs(|| Arc::new(setup_skipmap(keys.as_ref())))
            .bench_local_values(|map| {
                let start = Arc::new(Barrier::new(threads));
                let handles: Vec<_> = (0..threads)
                    .map(|t| {
                        let map = Arc::clone(&map);
                        let keys = Arc::clone(&keys);
                        let indices = Arc::clone(&indices);
                        let start = Arc::clone(&start);
                        thread::spawn(move || {
                            let mut sum = 0u64;
                            let base = t * OPS_PER_THREAD;
                            start.wait();
                            for i in 0..OPS_PER_THREAD {
                                let idx = indices[base + i];
                                if i % 100 < WRITE_RATIO {
                                    map.insert(keys[idx], i as u64);
                                } else if let Some(e) = map.get(&keys[idx]) {
                                    sum = sum.wrapping_add(*e.value());
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

    #[divan::bench(args = [1, 2, 3, 4, 5, 6])]
    fn indexset(bencher: Bencher, threads: usize) {
        let keys = Arc::new(keys::<KEY_SIZE>(N));
        let indices = Arc::new(zipfian_indices(N, OPS_PER_THREAD * threads, 42));

        bencher
            .counter(divan::counter::ItemsCount::new(threads * OPS_PER_THREAD))
            .with_inputs(|| Arc::new(setup_indexset(keys.as_ref())))
            .bench_local_values(|map| {
                let start = Arc::new(Barrier::new(threads));
                let handles: Vec<_> = (0..threads)
                    .map(|t| {
                        let map = Arc::clone(&map);
                        let keys = Arc::clone(&keys);
                        let indices = Arc::clone(&indices);
                        let start = Arc::clone(&start);
                        thread::spawn(move || {
                            let mut sum = 0u64;
                            let base = t * OPS_PER_THREAD;
                            start.wait();
                            for i in 0..OPS_PER_THREAD {
                                let idx = indices[base + i];
                                if i % 100 < WRITE_RATIO {
                                    map.insert(keys[idx], i as u64);
                                } else if let Some(r) = map.get(&keys[idx]) {
                                    sum = sum.wrapping_add(r.get().value);
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

    #[divan::bench(args = [1, 2, 3, 4, 5, 6])]
    fn tree_index(bencher: Bencher, threads: usize) {
        let keys = Arc::new(keys::<KEY_SIZE>(N));
        let indices = Arc::new(zipfian_indices(N, OPS_PER_THREAD * threads, 42));

        bencher
            .counter(divan::counter::ItemsCount::new(threads * OPS_PER_THREAD))
            .with_inputs(|| Arc::new(setup_tree_index(keys.as_ref())))
            .bench_local_values(|tree| {
                let start = Arc::new(Barrier::new(threads));
                let handles: Vec<_> = (0..threads)
                    .map(|t| {
                        let tree = Arc::clone(&tree);
                        let keys = Arc::clone(&keys);
                        let indices = Arc::clone(&indices);
                        let start = Arc::clone(&start);
                        thread::spawn(move || {
                            let guard = SddGuard::new();
                            let mut sum = 0u64;
                            let base = t * OPS_PER_THREAD;
                            start.wait();
                            for i in 0..OPS_PER_THREAD {
                                let idx = indices[base + i];
                                if i % 100 < WRITE_RATIO {
                                    tree_index_upsert_sync(&tree, keys[idx], i as u64);
                                } else if let Some(v) = tree.peek(&keys[idx], &guard) {
                                    sum = sum.wrapping_add(*v);
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
// 03: MIXED 90-10 - Shared Prefix (Masstree Stress Test)
// =============================================================================

#[divan::bench_group(name = "03_mixed_90_10_shared_prefix")]
mod mixed_shared_prefix {
    use super::*;

    const N: usize = 200_000;
    const OPS_PER_THREAD: usize = 50_000;
    const PREFIX_CHUNKS: usize = 3; // First 24 bytes shared
    const PREFIX_BUCKETS: u64 = 256;

    fn prefix_keys() -> Vec<[u8; KEY_SIZE]> {
        keys_shared_prefix_chunks::<KEY_SIZE>(N, PREFIX_CHUNKS, PREFIX_BUCKETS)
    }

    #[divan::bench(args = [1, 2, 3, 4, 5, 6])]
    fn masstree15(bencher: Bencher, threads: usize) {
        let keys = Arc::new(prefix_keys());
        let indices = Arc::new(uniform_indices(N, OPS_PER_THREAD * threads, 42));

        bencher
            .counter(divan::counter::ItemsCount::new(threads * OPS_PER_THREAD))
            .with_inputs(|| Arc::new(setup_masstree15(keys.as_ref())))
            .bench_local_values(|tree| {
                let start = Arc::new(Barrier::new(threads));
                let handles: Vec<_> = (0..threads)
                    .map(|t| {
                        let tree = Arc::clone(&tree);
                        let keys = Arc::clone(&keys);
                        let indices = Arc::clone(&indices);
                        let start = Arc::clone(&start);
                        thread::spawn(move || {
                            let guard = tree.guard();
                            let mut sum = 0u64;
                            let base = t * OPS_PER_THREAD;
                            start.wait();
                            for i in 0..OPS_PER_THREAD {
                                let idx = indices[base + i];
                                if i % 100 < WRITE_RATIO {
                                    let _ = tree.insert_with_guard(&keys[idx], i as u64, &guard);
                                } else if let Some(v) = tree.get_ref(&keys[idx], &guard) {
                                    sum = sum.wrapping_add(*v);
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

    #[divan::bench(args = [1, 2, 3, 4, 5, 6])]
    fn skipmap(bencher: Bencher, threads: usize) {
        let keys = Arc::new(prefix_keys());
        let indices = Arc::new(uniform_indices(N, OPS_PER_THREAD * threads, 42));

        bencher
            .counter(divan::counter::ItemsCount::new(threads * OPS_PER_THREAD))
            .with_inputs(|| Arc::new(setup_skipmap(keys.as_ref())))
            .bench_local_values(|map| {
                let start = Arc::new(Barrier::new(threads));
                let handles: Vec<_> = (0..threads)
                    .map(|t| {
                        let map = Arc::clone(&map);
                        let keys = Arc::clone(&keys);
                        let indices = Arc::clone(&indices);
                        let start = Arc::clone(&start);
                        thread::spawn(move || {
                            let mut sum = 0u64;
                            let base = t * OPS_PER_THREAD;
                            start.wait();
                            for i in 0..OPS_PER_THREAD {
                                let idx = indices[base + i];
                                if i % 100 < WRITE_RATIO {
                                    map.insert(keys[idx], i as u64);
                                } else if let Some(e) = map.get(&keys[idx]) {
                                    sum = sum.wrapping_add(*e.value());
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

    #[divan::bench(args = [1, 2, 3, 4, 5, 6])]
    fn indexset(bencher: Bencher, threads: usize) {
        let keys = Arc::new(prefix_keys());
        let indices = Arc::new(uniform_indices(N, OPS_PER_THREAD * threads, 42));

        bencher
            .counter(divan::counter::ItemsCount::new(threads * OPS_PER_THREAD))
            .with_inputs(|| Arc::new(setup_indexset(keys.as_ref())))
            .bench_local_values(|map| {
                let start = Arc::new(Barrier::new(threads));
                let handles: Vec<_> = (0..threads)
                    .map(|t| {
                        let map = Arc::clone(&map);
                        let keys = Arc::clone(&keys);
                        let indices = Arc::clone(&indices);
                        let start = Arc::clone(&start);
                        thread::spawn(move || {
                            let mut sum = 0u64;
                            let base = t * OPS_PER_THREAD;
                            start.wait();
                            for i in 0..OPS_PER_THREAD {
                                let idx = indices[base + i];
                                if i % 100 < WRITE_RATIO {
                                    map.insert(keys[idx], i as u64);
                                } else if let Some(r) = map.get(&keys[idx]) {
                                    sum = sum.wrapping_add(r.get().value);
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

    #[divan::bench(args = [1, 2, 3, 4, 5, 6])]
    fn tree_index(bencher: Bencher, threads: usize) {
        let keys = Arc::new(prefix_keys());
        let indices = Arc::new(uniform_indices(N, OPS_PER_THREAD * threads, 42));

        bencher
            .counter(divan::counter::ItemsCount::new(threads * OPS_PER_THREAD))
            .with_inputs(|| Arc::new(setup_tree_index(keys.as_ref())))
            .bench_local_values(|tree| {
                let start = Arc::new(Barrier::new(threads));
                let handles: Vec<_> = (0..threads)
                    .map(|t| {
                        let tree = Arc::clone(&tree);
                        let keys = Arc::clone(&keys);
                        let indices = Arc::clone(&indices);
                        let start = Arc::clone(&start);
                        thread::spawn(move || {
                            let guard = SddGuard::new();
                            let mut sum = 0u64;
                            let base = t * OPS_PER_THREAD;
                            start.wait();
                            for i in 0..OPS_PER_THREAD {
                                let idx = indices[base + i];
                                if i % 100 < WRITE_RATIO {
                                    tree_index_upsert_sync(&tree, keys[idx], i as u64);
                                } else if let Some(v) = tree.peek(&keys[idx], &guard) {
                                    sum = sum.wrapping_add(*v);
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
// 04: HIGH CONTENTION - Small Key Space (1000 keys)
// =============================================================================

#[divan::bench_group(name = "04_mixed_90_10_high_contention")]
mod mixed_high_contention {
    use super::*;

    const N: usize = 1_000; // Small key space = high contention
    const OPS_PER_THREAD: usize = 50_000;

    #[divan::bench(args = [1, 2, 3, 4, 5, 6])]
    fn masstree15(bencher: Bencher, threads: usize) {
        let keys = Arc::new(keys::<KEY_SIZE>(N));
        let indices = Arc::new(uniform_indices(N, OPS_PER_THREAD * threads, 42));

        bencher
            .counter(divan::counter::ItemsCount::new(threads * OPS_PER_THREAD))
            .with_inputs(|| Arc::new(setup_masstree15(keys.as_ref())))
            .bench_local_values(|tree| {
                let start = Arc::new(Barrier::new(threads));
                let handles: Vec<_> = (0..threads)
                    .map(|t| {
                        let tree = Arc::clone(&tree);
                        let keys = Arc::clone(&keys);
                        let indices = Arc::clone(&indices);
                        let start = Arc::clone(&start);
                        thread::spawn(move || {
                            let guard = tree.guard();
                            let mut sum = 0u64;
                            let base = t * OPS_PER_THREAD;
                            start.wait();
                            for i in 0..OPS_PER_THREAD {
                                let idx = indices[base + i];
                                if i % 100 < WRITE_RATIO {
                                    let _ = tree.insert_with_guard(&keys[idx], i as u64, &guard);
                                } else if let Some(v) = tree.get_ref(&keys[idx], &guard) {
                                    sum = sum.wrapping_add(*v);
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

    #[divan::bench(args = [1, 2, 3, 4, 5, 6])]
    fn skipmap(bencher: Bencher, threads: usize) {
        let keys = Arc::new(keys::<KEY_SIZE>(N));
        let indices = Arc::new(uniform_indices(N, OPS_PER_THREAD * threads, 42));

        bencher
            .counter(divan::counter::ItemsCount::new(threads * OPS_PER_THREAD))
            .with_inputs(|| Arc::new(setup_skipmap(keys.as_ref())))
            .bench_local_values(|map| {
                let start = Arc::new(Barrier::new(threads));
                let handles: Vec<_> = (0..threads)
                    .map(|t| {
                        let map = Arc::clone(&map);
                        let keys = Arc::clone(&keys);
                        let indices = Arc::clone(&indices);
                        let start = Arc::clone(&start);
                        thread::spawn(move || {
                            let mut sum = 0u64;
                            let base = t * OPS_PER_THREAD;
                            start.wait();
                            for i in 0..OPS_PER_THREAD {
                                let idx = indices[base + i];
                                if i % 100 < WRITE_RATIO {
                                    map.insert(keys[idx], i as u64);
                                } else if let Some(e) = map.get(&keys[idx]) {
                                    sum = sum.wrapping_add(*e.value());
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

    #[divan::bench(args = [1, 2, 3, 4, 5, 6])]
    fn indexset(bencher: Bencher, threads: usize) {
        let keys = Arc::new(keys::<KEY_SIZE>(N));
        let indices = Arc::new(uniform_indices(N, OPS_PER_THREAD * threads, 42));

        bencher
            .counter(divan::counter::ItemsCount::new(threads * OPS_PER_THREAD))
            .with_inputs(|| Arc::new(setup_indexset(keys.as_ref())))
            .bench_local_values(|map| {
                let start = Arc::new(Barrier::new(threads));
                let handles: Vec<_> = (0..threads)
                    .map(|t| {
                        let map = Arc::clone(&map);
                        let keys = Arc::clone(&keys);
                        let indices = Arc::clone(&indices);
                        let start = Arc::clone(&start);
                        thread::spawn(move || {
                            let mut sum = 0u64;
                            let base = t * OPS_PER_THREAD;
                            start.wait();
                            for i in 0..OPS_PER_THREAD {
                                let idx = indices[base + i];
                                if i % 100 < WRITE_RATIO {
                                    map.insert(keys[idx], i as u64);
                                } else if let Some(r) = map.get(&keys[idx]) {
                                    sum = sum.wrapping_add(r.get().value);
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

    #[divan::bench(args = [1, 2, 3, 4, 5, 6])]
    fn tree_index(bencher: Bencher, threads: usize) {
        let keys = Arc::new(keys::<KEY_SIZE>(N));
        let indices = Arc::new(uniform_indices(N, OPS_PER_THREAD * threads, 42));

        bencher
            .counter(divan::counter::ItemsCount::new(threads * OPS_PER_THREAD))
            .with_inputs(|| Arc::new(setup_tree_index(keys.as_ref())))
            .bench_local_values(|tree| {
                let start = Arc::new(Barrier::new(threads));
                let handles: Vec<_> = (0..threads)
                    .map(|t| {
                        let tree = Arc::clone(&tree);
                        let keys = Arc::clone(&keys);
                        let indices = Arc::clone(&indices);
                        let start = Arc::clone(&start);
                        thread::spawn(move || {
                            let guard = SddGuard::new();
                            let mut sum = 0u64;
                            let base = t * OPS_PER_THREAD;
                            start.wait();
                            for i in 0..OPS_PER_THREAD {
                                let idx = indices[base + i];
                                if i % 100 < WRITE_RATIO {
                                    tree_index_upsert_sync(&tree, keys[idx], i as u64);
                                } else if let Some(v) = tree.peek(&keys[idx], &guard) {
                                    sum = sum.wrapping_add(*v);
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
// 05: LARGE DATASET - 1M keys
// =============================================================================

#[divan::bench_group(name = "05_mixed_90_10_large_dataset")]
mod mixed_large_dataset {
    use super::*;

    const N: usize = 1_000_000;
    const OPS_PER_THREAD: usize = 50_000;

    #[divan::bench(args = [1, 2, 3, 4, 5, 6])]
    fn masstree15(bencher: Bencher, threads: usize) {
        let keys = Arc::new(keys::<KEY_SIZE>(N));
        let tree = Arc::new(setup_masstree15(keys.as_ref()));
        let indices = Arc::new(uniform_indices(N, OPS_PER_THREAD * threads, 42));

        bencher
            .counter(divan::counter::ItemsCount::new(threads * OPS_PER_THREAD))
            .bench_local(|| {
                let start = Arc::new(Barrier::new(threads));
                let handles: Vec<_> = (0..threads)
                    .map(|t| {
                        let tree = Arc::clone(&tree);
                        let keys = Arc::clone(&keys);
                        let indices = Arc::clone(&indices);
                        let start = Arc::clone(&start);
                        thread::spawn(move || {
                            let guard = tree.guard();
                            let mut sum = 0u64;
                            let base = t * OPS_PER_THREAD;
                            start.wait();
                            for i in 0..OPS_PER_THREAD {
                                let idx = indices[base + i];
                                if i % 100 < WRITE_RATIO {
                                    let _ = tree.insert_with_guard(&keys[idx], i as u64, &guard);
                                } else if let Some(v) = tree.get_ref(&keys[idx], &guard) {
                                    sum = sum.wrapping_add(*v);
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

    #[divan::bench(args = [1, 2, 3, 4, 5, 6])]
    fn skipmap(bencher: Bencher, threads: usize) {
        let keys = Arc::new(keys::<KEY_SIZE>(N));
        let map = Arc::new(setup_skipmap(keys.as_ref()));
        let indices = Arc::new(uniform_indices(N, OPS_PER_THREAD * threads, 42));

        bencher
            .counter(divan::counter::ItemsCount::new(threads * OPS_PER_THREAD))
            .bench_local(|| {
                let start = Arc::new(Barrier::new(threads));
                let handles: Vec<_> = (0..threads)
                    .map(|t| {
                        let map = Arc::clone(&map);
                        let keys = Arc::clone(&keys);
                        let indices = Arc::clone(&indices);
                        let start = Arc::clone(&start);
                        thread::spawn(move || {
                            let mut sum = 0u64;
                            let base = t * OPS_PER_THREAD;
                            start.wait();
                            for i in 0..OPS_PER_THREAD {
                                let idx = indices[base + i];
                                if i % 100 < WRITE_RATIO {
                                    map.insert(keys[idx], i as u64);
                                } else if let Some(e) = map.get(&keys[idx]) {
                                    sum = sum.wrapping_add(*e.value());
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

    #[divan::bench(args = [1, 2, 3, 4, 5, 6])]
    fn indexset(bencher: Bencher, threads: usize) {
        let keys = Arc::new(keys::<KEY_SIZE>(N));
        let map = Arc::new(setup_indexset(keys.as_ref()));
        let indices = Arc::new(uniform_indices(N, OPS_PER_THREAD * threads, 42));

        bencher
            .counter(divan::counter::ItemsCount::new(threads * OPS_PER_THREAD))
            .bench_local(|| {
                let start = Arc::new(Barrier::new(threads));
                let handles: Vec<_> = (0..threads)
                    .map(|t| {
                        let map = Arc::clone(&map);
                        let keys = Arc::clone(&keys);
                        let indices = Arc::clone(&indices);
                        let start = Arc::clone(&start);
                        thread::spawn(move || {
                            let mut sum = 0u64;
                            let base = t * OPS_PER_THREAD;
                            start.wait();
                            for i in 0..OPS_PER_THREAD {
                                let idx = indices[base + i];
                                if i % 100 < WRITE_RATIO {
                                    map.insert(keys[idx], i as u64);
                                } else if let Some(r) = map.get(&keys[idx]) {
                                    sum = sum.wrapping_add(r.get().value);
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

    #[divan::bench(args = [1, 2, 3, 4, 5, 6])]
    fn tree_index(bencher: Bencher, threads: usize) {
        let keys = Arc::new(keys::<KEY_SIZE>(N));
        let tree = Arc::new(setup_tree_index(keys.as_ref()));
        let indices = Arc::new(uniform_indices(N, OPS_PER_THREAD * threads, 42));

        bencher
            .counter(divan::counter::ItemsCount::new(threads * OPS_PER_THREAD))
            .bench_local(|| {
                let start = Arc::new(Barrier::new(threads));
                let handles: Vec<_> = (0..threads)
                    .map(|t| {
                        let tree = Arc::clone(&tree);
                        let keys = Arc::clone(&keys);
                        let indices = Arc::clone(&indices);
                        let start = Arc::clone(&start);
                        thread::spawn(move || {
                            let guard = SddGuard::new();
                            let mut sum = 0u64;
                            let base = t * OPS_PER_THREAD;
                            start.wait();
                            for i in 0..OPS_PER_THREAD {
                                let idx = indices[base + i];
                                if i % 100 < WRITE_RATIO {
                                    tree_index_upsert_sync(&tree, keys[idx], i as u64);
                                } else if let Some(v) = tree.peek(&keys[idx], &guard) {
                                    sum = sum.wrapping_add(*v);
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
// 06: SINGLE HOT KEY - Maximum Contention
// =============================================================================

#[divan::bench_group(name = "06_single_hot_key")]
mod single_hot_key {
    use super::*;

    const N: usize = 100_000;
    const OPS_PER_THREAD: usize = 10_000;

    #[divan::bench(args = [2, 4, 6])]
    fn masstree15(bencher: Bencher, threads: usize) {
        let keys = Arc::new(keys::<KEY_SIZE>(N));
        let hot_key = keys[N / 2];

        bencher
            .counter(divan::counter::ItemsCount::new(threads * OPS_PER_THREAD))
            .with_inputs(|| Arc::new(setup_masstree15(keys.as_ref())))
            .bench_local_values(|tree| {
                let start = Arc::new(Barrier::new(threads));
                let handles: Vec<_> = (0..threads)
                    .map(|t| {
                        let tree = Arc::clone(&tree);
                        let hot_key = hot_key;
                        let start = Arc::clone(&start);
                        thread::spawn(move || {
                            let guard = tree.guard();
                            let mut sum = 0u64;
                            start.wait();
                            for i in 0..OPS_PER_THREAD {
                                if i % 100 < WRITE_RATIO {
                                    let _ = tree.insert_with_guard(
                                        &hot_key,
                                        (t * OPS_PER_THREAD + i) as u64,
                                        &guard,
                                    );
                                } else if let Some(v) = tree.get_ref(&hot_key, &guard) {
                                    sum = sum.wrapping_add(*v);
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

    #[divan::bench(args = [2, 4, 6])]
    fn skipmap(bencher: Bencher, threads: usize) {
        let keys = Arc::new(keys::<KEY_SIZE>(N));
        let hot_key = keys[N / 2];

        bencher
            .counter(divan::counter::ItemsCount::new(threads * OPS_PER_THREAD))
            .with_inputs(|| Arc::new(setup_skipmap(keys.as_ref())))
            .bench_local_values(|map| {
                let start = Arc::new(Barrier::new(threads));
                let handles: Vec<_> = (0..threads)
                    .map(|t| {
                        let map = Arc::clone(&map);
                        let hot_key = hot_key;
                        let start = Arc::clone(&start);
                        thread::spawn(move || {
                            let mut sum = 0u64;
                            start.wait();
                            for i in 0..OPS_PER_THREAD {
                                if i % 100 < WRITE_RATIO {
                                    map.insert(hot_key, (t * OPS_PER_THREAD + i) as u64);
                                } else if let Some(e) = map.get(&hot_key) {
                                    sum = sum.wrapping_add(*e.value());
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

    #[divan::bench(args = [2, 4, 6])]
    fn indexset(bencher: Bencher, threads: usize) {
        let keys = Arc::new(keys::<KEY_SIZE>(N));
        let hot_key = keys[N / 2];

        bencher
            .counter(divan::counter::ItemsCount::new(threads * OPS_PER_THREAD))
            .with_inputs(|| Arc::new(setup_indexset(keys.as_ref())))
            .bench_local_values(|map| {
                let start = Arc::new(Barrier::new(threads));
                let handles: Vec<_> = (0..threads)
                    .map(|t| {
                        let map = Arc::clone(&map);
                        let hot_key = hot_key;
                        let start = Arc::clone(&start);
                        thread::spawn(move || {
                            let mut sum = 0u64;
                            start.wait();
                            for i in 0..OPS_PER_THREAD {
                                if i % 100 < WRITE_RATIO {
                                    map.insert(hot_key, (t * OPS_PER_THREAD + i) as u64);
                                } else if let Some(r) = map.get(&hot_key) {
                                    sum = sum.wrapping_add(r.get().value);
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

    #[divan::bench(args = [2, 4, 6])]
    fn tree_index(bencher: Bencher, threads: usize) {
        let keys = Arc::new(keys::<KEY_SIZE>(N));
        let hot_key = keys[N / 2];

        bencher
            .counter(divan::counter::ItemsCount::new(threads * OPS_PER_THREAD))
            .with_inputs(|| Arc::new(setup_tree_index(keys.as_ref())))
            .bench_local_values(|tree| {
                let start = Arc::new(Barrier::new(threads));
                let handles: Vec<_> = (0..threads)
                    .map(|t| {
                        let tree = Arc::clone(&tree);
                        let hot_key = hot_key;
                        let start = Arc::clone(&start);
                        thread::spawn(move || {
                            let guard = SddGuard::new();
                            let mut sum = 0u64;
                            start.wait();
                            for i in 0..OPS_PER_THREAD {
                                if i % 100 < WRITE_RATIO {
                                    tree_index_upsert_sync(
                                        &tree,
                                        hot_key,
                                        (t * OPS_PER_THREAD + i) as u64,
                                    );
                                } else if let Some(v) = tree.peek(&hot_key, &guard) {
                                    sum = sum.wrapping_add(*v);
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
// 07: WRITE-HEAVY - 50% reads, 50% writes
// =============================================================================

#[divan::bench_group(name = "07_mixed_50_50")]
mod mixed_50_50 {
    use super::*;

    const N: usize = 200_000;
    const OPS_PER_THREAD: usize = 50_000;
    const WRITE_RATIO_50: usize = 50; // 50% writes

    #[divan::bench(args = [1, 2, 3, 4, 5, 6])]
    fn masstree15(bencher: Bencher, threads: usize) {
        let keys = Arc::new(keys::<KEY_SIZE>(N));
        let indices = Arc::new(uniform_indices(N, OPS_PER_THREAD * threads, 42));

        bencher
            .counter(divan::counter::ItemsCount::new(threads * OPS_PER_THREAD))
            .with_inputs(|| Arc::new(setup_masstree15(keys.as_ref())))
            .bench_local_values(|tree| {
                let start = Arc::new(Barrier::new(threads));
                let handles: Vec<_> = (0..threads)
                    .map(|t| {
                        let tree = Arc::clone(&tree);
                        let keys = Arc::clone(&keys);
                        let indices = Arc::clone(&indices);
                        let start = Arc::clone(&start);
                        thread::spawn(move || {
                            let guard = tree.guard();
                            let mut sum = 0u64;
                            let base = t * OPS_PER_THREAD;
                            start.wait();
                            for i in 0..OPS_PER_THREAD {
                                let idx = indices[base + i];
                                if i % 100 < WRITE_RATIO_50 {
                                    let _ = tree.insert_with_guard(&keys[idx], i as u64, &guard);
                                } else if let Some(v) = tree.get_ref(&keys[idx], &guard) {
                                    sum = sum.wrapping_add(*v);
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

    #[divan::bench(args = [1, 2, 3, 4, 5, 6])]
    fn skipmap(bencher: Bencher, threads: usize) {
        let keys = Arc::new(keys::<KEY_SIZE>(N));
        let indices = Arc::new(uniform_indices(N, OPS_PER_THREAD * threads, 42));

        bencher
            .counter(divan::counter::ItemsCount::new(threads * OPS_PER_THREAD))
            .with_inputs(|| Arc::new(setup_skipmap(keys.as_ref())))
            .bench_local_values(|map| {
                let start = Arc::new(Barrier::new(threads));
                let handles: Vec<_> = (0..threads)
                    .map(|t| {
                        let map = Arc::clone(&map);
                        let keys = Arc::clone(&keys);
                        let indices = Arc::clone(&indices);
                        let start = Arc::clone(&start);
                        thread::spawn(move || {
                            let mut sum = 0u64;
                            let base = t * OPS_PER_THREAD;
                            start.wait();
                            for i in 0..OPS_PER_THREAD {
                                let idx = indices[base + i];
                                if i % 100 < WRITE_RATIO_50 {
                                    map.insert(keys[idx], i as u64);
                                } else if let Some(e) = map.get(&keys[idx]) {
                                    sum = sum.wrapping_add(*e.value());
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

    #[divan::bench(args = [1, 2, 3, 4, 5, 6])]
    fn indexset(bencher: Bencher, threads: usize) {
        let keys = Arc::new(keys::<KEY_SIZE>(N));
        let indices = Arc::new(uniform_indices(N, OPS_PER_THREAD * threads, 42));

        bencher
            .counter(divan::counter::ItemsCount::new(threads * OPS_PER_THREAD))
            .with_inputs(|| Arc::new(setup_indexset(keys.as_ref())))
            .bench_local_values(|map| {
                let start = Arc::new(Barrier::new(threads));
                let handles: Vec<_> = (0..threads)
                    .map(|t| {
                        let map = Arc::clone(&map);
                        let keys = Arc::clone(&keys);
                        let indices = Arc::clone(&indices);
                        let start = Arc::clone(&start);
                        thread::spawn(move || {
                            let mut sum = 0u64;
                            let base = t * OPS_PER_THREAD;
                            start.wait();
                            for i in 0..OPS_PER_THREAD {
                                let idx = indices[base + i];
                                if i % 100 < WRITE_RATIO_50 {
                                    map.insert(keys[idx], i as u64);
                                } else if let Some(r) = map.get(&keys[idx]) {
                                    sum = sum.wrapping_add(r.get().value);
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

    #[divan::bench(args = [1, 2, 3, 4, 5, 6])]
    fn tree_index(bencher: Bencher, threads: usize) {
        let keys = Arc::new(keys::<KEY_SIZE>(N));
        let indices = Arc::new(uniform_indices(N, OPS_PER_THREAD * threads, 42));

        bencher
            .counter(divan::counter::ItemsCount::new(threads * OPS_PER_THREAD))
            .with_inputs(|| Arc::new(setup_tree_index(keys.as_ref())))
            .bench_local_values(|tree| {
                let start = Arc::new(Barrier::new(threads));
                let handles: Vec<_> = (0..threads)
                    .map(|t| {
                        let tree = Arc::clone(&tree);
                        let keys = Arc::clone(&keys);
                        let indices = Arc::clone(&indices);
                        let start = Arc::clone(&start);
                        thread::spawn(move || {
                            let guard = SddGuard::new();
                            let mut sum = 0u64;
                            let base = t * OPS_PER_THREAD;
                            start.wait();
                            for i in 0..OPS_PER_THREAD {
                                let idx = indices[base + i];
                                if i % 100 < WRITE_RATIO_50 {
                                    tree_index_upsert_sync(&tree, keys[idx], i as u64);
                                } else if let Some(v) = tree.peek(&keys[idx], &guard) {
                                    sum = sum.wrapping_add(*v);
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
