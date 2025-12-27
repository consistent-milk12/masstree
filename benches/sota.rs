//! State-of-the-art concurrent map benchmarks
//!
//! - **SCC TreeIndex**: Read-optimized concurrent B+tree (ordered)
//! - **Papaya HashMap**: Lock-free concurrent hash map (unordered)
//!
//! ## Running
//!
//! ```bash
//! cargo bench --bench sota
//! cargo bench --bench sota --features mimalloc
//! ```
//!
//! ## Notes
//!
//! - TreeIndex is an ordered map (direct comparison with MassTree)
//! - Papaya is unordered but provides a performance reference point
//! - All use epoch/hyaline-based memory reclamation

#![expect(clippy::unwrap_used)]
#![expect(clippy::pedantic)]
#![expect(clippy::indexing_slicing)]
#![expect(clippy::redundant_locals)]

mod bench_utils;

use bench_utils::{keys, uniform_indices};
use divan::{Bencher, black_box};
use papaya::HashMap as PapayaMap;
use scc::TreeIndex;
use sdd::Guard as SddGuard;
use std::sync::Arc;
use std::thread;

fn main() {
    divan::main();
}

// =============================================================================
// Setup Helpers
// =============================================================================

fn setup_tree_index<const K: usize>(keys: &[[u8; K]]) -> TreeIndex<[u8; K], u64> {
    let tree = TreeIndex::new();
    for (i, key) in keys.iter().enumerate() {
        let _ = tree.insert_sync(*key, i as u64);
    }
    tree
}

fn setup_papaya<const K: usize>(keys: &[[u8; K]]) -> PapayaMap<[u8; K], u64> {
    let map = PapayaMap::new();
    {
        let guard = map.guard();
        for (i, key) in keys.iter().enumerate() {
            map.insert(*key, i as u64, &guard);
        }
    }
    map
}

// =============================================================================
// 01: CONCURRENT WRITES - Disjoint Ranges
// =============================================================================

#[divan::bench_group(name = "01_concurrent_writes_disjoint")]
mod concurrent_writes_disjoint {
    use super::*;

    const OPS_PER_THREAD: usize = 50_000;

    #[divan::bench(args = [1, 2, 4, 8, 16, 32])]
    fn tree_index(bencher: Bencher, threads: usize) {
        bencher
            .with_inputs(|| Arc::new(TreeIndex::<[u8; 8], u64>::new()))
            .bench_local_values(|tree| {
                let handles: Vec<_> = (0..threads)
                    .map(|t| {
                        let tree = Arc::clone(&tree);
                        thread::spawn(move || {
                            let base = t * OPS_PER_THREAD;
                            for i in 0..OPS_PER_THREAD {
                                let key = ((base + i) as u64).to_be_bytes();
                                let _ = tree.insert_sync(key, i as u64);
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
    fn papaya(bencher: Bencher, threads: usize) {
        bencher
            .with_inputs(|| Arc::new(PapayaMap::<[u8; 8], u64>::new()))
            .bench_local_values(|map| {
                let handles: Vec<_> = (0..threads)
                    .map(|t| {
                        let map = Arc::clone(&map);
                        thread::spawn(move || {
                            let guard = map.guard();
                            let base = t * OPS_PER_THREAD;
                            for i in 0..OPS_PER_THREAD {
                                let key = ((base + i) as u64).to_be_bytes();
                                map.insert(key, i as u64, &guard);
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
    const KEY_SPACE: usize = 1_000;

    #[divan::bench(args = [1, 2, 4, 8, 16, 32])]
    fn tree_index(bencher: Bencher, threads: usize) {
        let keys = Arc::new(keys::<8>(KEY_SPACE));

        bencher
            .with_inputs(|| Arc::new(setup_tree_index::<8>(keys.as_ref())))
            .bench_local_values(|tree| {
                let counter = Arc::new(AtomicUsize::new(0));
                let handles: Vec<_> = (0..threads)
                    .map(|t| {
                        let tree = Arc::clone(&tree);
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
                                // TreeIndex doesn't support update, use insert_sync (inserts if not exists)
                                let _ = tree.insert_sync(keys[idx], val);
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
    fn papaya(bencher: Bencher, threads: usize) {
        let keys = Arc::new(keys::<8>(KEY_SPACE));

        bencher
            .with_inputs(|| Arc::new(setup_papaya::<8>(keys.as_ref())))
            .bench_local_values(|map| {
                let counter = Arc::new(AtomicUsize::new(0));
                let handles: Vec<_> = (0..threads)
                    .map(|t| {
                        let map = Arc::clone(&map);
                        let keys = Arc::clone(&keys);
                        let counter = Arc::clone(&counter);
                        thread::spawn(move || {
                            let guard = map.guard();
                            let mut state = (t as u64).wrapping_mul(0x517c_c1b7_2722_0a95);
                            for _ in 0..OPS_PER_THREAD {
                                state = state
                                    .wrapping_mul(6_364_136_223_846_793_005)
                                    .wrapping_add(1);
                                let idx = (state as usize) % keys.len();
                                let val = counter.fetch_add(1, Ordering::Relaxed) as u64;
                                map.insert(keys[idx], val, &guard);
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
    fn tree_index(bencher: Bencher) {
        bencher.bench_local(|| {
            let tree = TreeIndex::<[u8; 8], u64>::new();
            for i in 0..KEY_COUNT {
                let key = (i as u64).to_be_bytes();
                let _ = tree.insert_sync(key, i as u64);
            }
            black_box(tree)
        });
    }

    #[divan::bench]
    fn papaya(bencher: Bencher) {
        bencher.bench_local(|| {
            let map = PapayaMap::<[u8; 8], u64>::new();
            {
                let guard = map.guard();
                for i in 0..KEY_COUNT {
                    let key = (i as u64).to_be_bytes();
                    map.insert(key, i as u64, &guard);
                }
            }
            black_box(map)
        });
    }
}

// =============================================================================
// 04: CONCURRENT READS - Thread Scaling (8-byte keys)
// =============================================================================

#[divan::bench_group(name = "04_concurrent_reads_scaling")]
mod concurrent_reads_scaling {
    use super::*;

    const N: usize = 1_000_000;
    const OPS_PER_THREAD: usize = 50_000;

    #[divan::bench(args = [1, 2, 4, 8, 16, 32])]
    fn tree_index(bencher: Bencher, threads: usize) {
        let keys = Arc::new(keys::<8>(N));
        let tree = Arc::new(setup_tree_index::<8>(keys.as_ref()));
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
                            if let Some(v) = tree.peek(&keys[idx], &SddGuard::new()) {
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
    fn papaya(bencher: Bencher, threads: usize) {
        let keys = Arc::new(keys::<8>(N));
        let map = Arc::new(setup_papaya::<8>(keys.as_ref()));
        let indices = Arc::new(uniform_indices(N, OPS_PER_THREAD, 42));

        bencher.bench_local(|| {
            let handles: Vec<_> = (0..threads)
                .map(|t| {
                    let map = Arc::clone(&map);
                    let keys = Arc::clone(&keys);
                    let indices = Arc::clone(&indices);
                    thread::spawn(move || {
                        let guard = map.guard();
                        let mut sum = 0u64;
                        let offset = t * 7919;
                        for i in 0..OPS_PER_THREAD {
                            let idx = indices[(i + offset) % indices.len()];
                            if let Some(v) = map.get(&keys[idx], &guard) {
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
// 05: READ SCALING - Throughput (8-byte keys)
// =============================================================================

#[divan::bench_group(name = "05_read_throughput_8B")]
mod read_throughput_8b {
    use super::*;

    const N: usize = 1_000_000;
    const OPS_PER_THREAD: usize = 50_000;

    #[divan::bench(args = [1, 2, 4, 8, 16, 32])]
    fn tree_index(bencher: Bencher, threads: usize) {
        let keys = Arc::new(keys::<8>(N));
        let tree = Arc::new(setup_tree_index::<8>(keys.as_ref()));

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
                                if let Some(v) = tree.peek(&keys[idx], &SddGuard::new()) {
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
    fn papaya(bencher: Bencher, threads: usize) {
        let keys = Arc::new(keys::<8>(N));
        let map = Arc::new(setup_papaya::<8>(keys.as_ref()));

        bencher
            .counter(divan::counter::ItemsCount::new(threads * OPS_PER_THREAD))
            .bench_local(|| {
                let handles: Vec<_> = (0..threads)
                    .map(|t| {
                        let map = Arc::clone(&map);
                        let keys = Arc::clone(&keys);
                        thread::spawn(move || {
                            let guard = map.guard();
                            let mut sum = 0u64;
                            let start = (t * 7919) % keys.len();
                            for i in 0..OPS_PER_THREAD {
                                let idx = (start + i) % keys.len();
                                if let Some(v) = map.get(&keys[idx], &guard) {
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
// 06: WRITE SCALING - Throughput (8-byte keys)
// =============================================================================

#[divan::bench_group(name = "06_write_throughput_8B")]
mod write_throughput_8b {
    use super::*;

    const N: usize = 100_000;
    const OPS_PER_THREAD: usize = 10_000;

    #[divan::bench(args = [1, 2, 4, 8, 16, 32])]
    fn tree_index(bencher: Bencher, threads: usize) {
        let keys = Arc::new(keys::<8>(N));

        bencher
            .counter(divan::counter::ItemsCount::new(threads * OPS_PER_THREAD))
            .with_inputs(|| {
                let tree = TreeIndex::<[u8; 8], u64>::new();
                // Pre-populate with half the keys
                for (i, key) in keys.iter().take(N / 2).enumerate() {
                    let _ = tree.insert_sync(*key, i as u64);
                }
                Arc::new(tree)
            })
            .bench_local_values(|tree| {
                let handles: Vec<_> = (0..threads)
                    .map(|t| {
                        let tree = Arc::clone(&tree);
                        let keys = Arc::clone(&keys);
                        thread::spawn(move || {
                            let start = (t * 7919) % keys.len();
                            for i in 0..OPS_PER_THREAD {
                                let idx = (start + i) % keys.len();
                                let _ = tree.insert_sync(keys[idx], i as u64);
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
    fn papaya(bencher: Bencher, threads: usize) {
        let keys = Arc::new(keys::<8>(N));

        bencher
            .counter(divan::counter::ItemsCount::new(threads * OPS_PER_THREAD))
            .with_inputs(|| {
                let map = PapayaMap::<[u8; 8], u64>::new();
                // Pre-populate with half the keys
                {
                    let guard = map.guard();
                    for (i, key) in keys.iter().take(N / 2).enumerate() {
                        map.insert(*key, i as u64, &guard);
                    }
                }
                Arc::new(map)
            })
            .bench_local_values(|map| {
                let handles: Vec<_> = (0..threads)
                    .map(|t| {
                        let map = Arc::clone(&map);
                        let keys = Arc::clone(&keys);
                        thread::spawn(move || {
                            let guard = map.guard();
                            let start = (t * 7919) % keys.len();
                            for i in 0..OPS_PER_THREAD {
                                let idx = (start + i) % keys.len();
                                map.insert(keys[idx], i as u64, &guard);
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
// 07: MIXED WORKLOAD - 90% Read / 10% Write
// =============================================================================

#[divan::bench_group(name = "07_mixed_90r_10w")]
mod mixed_90r_10w {
    use super::*;

    const N: usize = 100_000;
    const OPS_PER_THREAD: usize = 10_000;
    const WRITE_RATIO: usize = 10; // 10% writes

    #[divan::bench(args = [1, 2, 4, 8, 16, 32])]
    fn tree_index(bencher: Bencher, threads: usize) {
        let keys = Arc::new(keys::<8>(N));
        let indices = Arc::new(uniform_indices(N, OPS_PER_THREAD, 42));

        bencher
            .with_inputs(|| Arc::new(setup_tree_index::<8>(keys.as_ref())))
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
                                    let _ = tree.insert_sync(keys[idx], i as u64);
                                } else if let Some(v) = tree.peek(&keys[idx], &SddGuard::new()) {
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

    #[divan::bench(args = [1, 2, 4, 8, 16, 32])]
    fn papaya(bencher: Bencher, threads: usize) {
        let keys = Arc::new(keys::<8>(N));
        let indices = Arc::new(uniform_indices(N, OPS_PER_THREAD, 42));

        bencher
            .with_inputs(|| Arc::new(setup_papaya::<8>(keys.as_ref())))
            .bench_local_values(|map| {
                let handles: Vec<_> = (0..threads)
                    .map(|t| {
                        let map = Arc::clone(&map);
                        let keys = Arc::clone(&keys);
                        let indices = Arc::clone(&indices);
                        thread::spawn(move || {
                            let guard = map.guard();
                            let mut sum = 0u64;
                            let offset = t * 7919;

                            for i in 0..OPS_PER_THREAD {
                                let idx = indices[(i + offset) % indices.len()];

                                if i % WRITE_RATIO == 0 {
                                    map.insert(keys[idx], i as u64, &guard);
                                } else if let Some(v) = map.get(&keys[idx], &guard) {
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
    use super::*;

    const N: usize = 100_000;
    const OPS_PER_THREAD: usize = 10_000;

    #[divan::bench(args = [2, 4, 8, 16, 32])]
    fn tree_index(bencher: Bencher, threads: usize) {
        let keys = keys::<8>(N);
        let hot_key = keys[N / 2];

        bencher
            .with_inputs(|| Arc::new(setup_tree_index::<8>(&keys)))
            .bench_local_values(|tree| {
                let handles: Vec<_> = (0..threads)
                    .map(|t| {
                        let tree = Arc::clone(&tree);
                        let hot_key = hot_key;
                        thread::spawn(move || {
                            let mut sum = 0u64;

                            for i in 0..OPS_PER_THREAD {
                                if i % 10 == 0 {
                                    let _ =
                                        tree.insert_sync(hot_key, (t * OPS_PER_THREAD + i) as u64);
                                } else if let Some(v) = tree.peek(&hot_key, &SddGuard::new()) {
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
    fn papaya(bencher: Bencher, threads: usize) {
        let keys = keys::<8>(N);
        let hot_key = keys[N / 2];

        bencher
            .with_inputs(|| Arc::new(setup_papaya::<8>(&keys)))
            .bench_local_values(|map| {
                let handles: Vec<_> = (0..threads)
                    .map(|t| {
                        let map = Arc::clone(&map);
                        let hot_key = hot_key;
                        thread::spawn(move || {
                            let guard = map.guard();
                            let mut sum = 0u64;

                            for i in 0..OPS_PER_THREAD {
                                if i % 10 == 0 {
                                    map.insert(hot_key, (t * OPS_PER_THREAD + i) as u64, &guard);
                                } else if let Some(v) = map.get(&hot_key, &guard) {
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
// 09: SINGLE-THREADED GET - Variable Key Sizes (MassTree only - tests layering)
// =============================================================================

#[divan::bench_group(name = "09_get_by_key_size")]
mod get_by_key_size {
    use super::*;

    const N: usize = 10_000;

    fn bench_tree_index<const K: usize>(bencher: Bencher) {
        let keys = keys::<K>(N);
        let tree = setup_tree_index::<K>(&keys);
        let lookup_keys = uniform_indices(N, 1000, 42);

        bencher.bench_local(|| {
            let mut sum = 0u64;
            for &idx in &lookup_keys {
                if let Some(v) = tree.peek(&keys[idx], &SddGuard::new()) {
                    sum += *v;
                }
            }
            black_box(sum)
        });
    }

    fn bench_papaya<const K: usize>(bencher: Bencher) {
        let keys = keys::<K>(N);
        let map = setup_papaya::<K>(&keys);
        let lookup_keys = uniform_indices(N, 1000, 42);

        bencher.bench_local(|| {
            let guard = map.guard();
            let mut sum = 0u64;
            for &idx in &lookup_keys {
                if let Some(v) = map.get(&keys[idx], &guard) {
                    sum += *v;
                }
            }
            black_box(sum)
        });
    }

    #[divan::bench(name = "tree_index_8B")]
    fn tree_index_8b(bencher: Bencher) {
        bench_tree_index::<8>(bencher);
    }

    #[divan::bench(name = "tree_index_16B")]
    fn tree_index_16b(bencher: Bencher) {
        bench_tree_index::<16>(bencher);
    }

    #[divan::bench(name = "tree_index_32B")]
    fn tree_index_32b(bencher: Bencher) {
        bench_tree_index::<32>(bencher);
    }

    #[divan::bench(name = "papaya_8B")]
    fn papaya_8b(bencher: Bencher) {
        bench_papaya::<8>(bencher);
    }

    #[divan::bench(name = "papaya_16B")]
    fn papaya_16b(bencher: Bencher) {
        bench_papaya::<16>(bencher);
    }

    #[divan::bench(name = "papaya_32B")]
    fn papaya_32b(bencher: Bencher) {
        bench_papaya::<32>(bencher);
    }
}

// =============================================================================
// 10: SINGLE-THREADED INSERT - Variable Key Sizes
// =============================================================================

#[divan::bench_group(name = "10_insert_by_key_size")]
mod insert_by_key_size {
    use super::*;

    const N: usize = 1000;

    fn bench_tree_index<const K: usize>(bencher: Bencher) {
        let keys = keys::<K>(N);
        bencher
            .with_inputs(|| keys.clone())
            .bench_local_values(|keys| {
                let tree = TreeIndex::<[u8; K], u64>::new();
                for (i, key) in keys.iter().enumerate() {
                    let _ = tree.insert_sync(*key, i as u64);
                }
                tree
            });
    }

    fn bench_papaya<const K: usize>(bencher: Bencher) {
        let keys = keys::<K>(N);
        bencher
            .with_inputs(|| keys.clone())
            .bench_local_values(|keys| {
                let map = PapayaMap::<[u8; K], u64>::new();
                {
                    let guard = map.guard();
                    for (i, key) in keys.iter().enumerate() {
                        map.insert(*key, i as u64, &guard);
                    }
                }
                map
            });
    }

    #[divan::bench(name = "tree_index_8B")]
    fn tree_index_8b(bencher: Bencher) {
        bench_tree_index::<8>(bencher);
    }

    #[divan::bench(name = "tree_index_16B")]
    fn tree_index_16b(bencher: Bencher) {
        bench_tree_index::<16>(bencher);
    }

    #[divan::bench(name = "tree_index_32B")]
    fn tree_index_32b(bencher: Bencher) {
        bench_tree_index::<32>(bencher);
    }

    #[divan::bench(name = "papaya_8B")]
    fn papaya_8b(bencher: Bencher) {
        bench_papaya::<8>(bencher);
    }

    #[divan::bench(name = "papaya_16B")]
    fn papaya_16b(bencher: Bencher) {
        bench_papaya::<16>(bencher);
    }

    #[divan::bench(name = "papaya_32B")]
    fn papaya_32b(bencher: Bencher) {
        bench_papaya::<32>(bencher);
    }
}
