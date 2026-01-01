//! Extreme stress tests for split-heavy scenarios - Criterion version.
//!
//! These benchmarks specifically target the double-buffer optimization's intended use case:
//! reducing root restarts during concurrent splits.
//!
//! ## Scenarios Tested
//!
//! 1. **Sequential flooding**: All threads insert sequential keys into overlapping ranges
//!    - Maximizes leaf fills and splits
//!    - Creates split cascades up the tree
//!
//! 2. **Hot range contention**: All threads hammer a small key range
//!    - Forces repeated splits of the same nodes
//!    - Tests split detection and retry logic
//!
//! 3. **Layer explosion**: Long keys that create many trie layers
//!    - Tests layer creation under concurrency
//!    - Combines splits with layer pointer updates
//!
//! 4. **Mixed split storm**: Concurrent reads during heavy splits
//!    - Reader threads traverse while writers cause splits
//!    - Tests if double-buffer reduces reader restarts
//!
//! ## Running
//!
//! ```bash
//! cargo bench --bench extreme_stress_crit
//! cargo bench --bench extreme_stress_crit --features mimalloc
//! cargo bench --bench extreme_stress_crit -- --save-baseline main
//! cargo bench --bench extreme_stress_crit -- --baseline main
//! ```

#![allow(clippy::unwrap_used)]
#![allow(clippy::pedantic)]
#![allow(clippy::indexing_slicing)]

mod bench_utils;

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use masstree::MassTree15;
use scc::TreeIndex;
use sdd::Guard as SddGuard;
use std::hint::black_box;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::sync::Barrier;
use std::thread;

// =============================================================================
// 01: SEQUENTIAL FLOODING - Maximum Split Pressure
// =============================================================================
//
// All threads insert into a shared counter, causing:
// - Sequential key patterns that fill leaves completely
// - Cascading splits as leaves overflow
// - Root splits when internodes fill
//
// This is the WORST case for trees - sequential insertion with contention.

fn bench_01_sequential_flooding(c: &mut Criterion) {
    let mut group = c.benchmark_group("01_sequential_flooding");
    group.sample_size(20);

    const OPS_PER_THREAD: usize = 100_000;

    for threads in [1, 2, 3, 4, 5, 6] {
        group.bench_function(BenchmarkId::new("masstree15", threads), |b| {
            b.iter_with_setup(
                || {
                    (
                        Arc::new(MassTree15::<u64>::new()),
                        Arc::new(AtomicU64::new(0)),
                    )
                },
                |(tree, counter)| {
                    let start = Arc::new(Barrier::new(threads));
                    let handles: Vec<_> = (0..threads)
                        .map(|_| {
                            let tree = Arc::clone(&tree);
                            let counter = Arc::clone(&counter);
                            let start = Arc::clone(&start);

                            thread::spawn(move || {
                                let guard = tree.guard();
                                start.wait();

                                for _ in 0..OPS_PER_THREAD {
                                    let key_val = counter.fetch_add(1, Ordering::Relaxed);
                                    let key = key_val.to_be_bytes();
                                    let _ = tree.insert_with_guard(&key, key_val, &guard);
                                }
                            })
                        })
                        .collect();

                    for h in handles {
                        h.join().unwrap();
                    }

                    black_box(&tree);
                },
            );
        });

        group.bench_function(BenchmarkId::new("tree_index", threads), |b| {
            b.iter_with_setup(
                || {
                    (
                        Arc::new(TreeIndex::<[u8; 8], u64>::new()),
                        Arc::new(AtomicU64::new(0)),
                    )
                },
                |(tree, counter)| {
                    let start = Arc::new(Barrier::new(threads));
                    let handles: Vec<_> = (0..threads)
                        .map(|_| {
                            let tree = Arc::clone(&tree);
                            let counter = Arc::clone(&counter);
                            let start = Arc::clone(&start);

                            thread::spawn(move || {
                                start.wait();
                                for _ in 0..OPS_PER_THREAD {
                                    let key_val = counter.fetch_add(1, Ordering::Relaxed);
                                    let key = key_val.to_be_bytes();
                                    let _ = tree.insert_sync(key, key_val);
                                }
                            })
                        })
                        .collect();

                    for h in handles {
                        h.join().unwrap();
                    }

                    black_box(&tree);
                },
            );
        });
    }
    group.finish();
}

// =============================================================================
// 02: HOT RANGE CONTENTION - Read Contention on Small Key Range
// =============================================================================
//
// All threads read from a tiny pre-populated range (256 keys).
// This tests:
// - Read contention on frequently accessed nodes
// - Cache behavior under concurrent access
// - Lock-free read path efficiency
//
// Note: We test reads (not upserts) because TreeIndex lacks native upsert,
// and the remove+reinsert workaround would unfairly penalize it.

fn bench_02_hot_range_contention(c: &mut Criterion) {
    let mut group = c.benchmark_group("02_hot_range_contention");
    group.sample_size(20);

    const OPS_PER_THREAD: usize = 100_000;
    const HOT_RANGE: u64 = 256;

    // Pre-generate keys for the hot range
    let hot_keys: Vec<[u8; 8]> = (0..HOT_RANGE).map(|i| i.to_be_bytes()).collect();

    for threads in [1, 2, 3, 4, 5, 6] {
        let hot_keys = hot_keys.clone();

        group.bench_function(BenchmarkId::new("masstree15", threads), |b| {
            let hot_keys = hot_keys.clone();

            b.iter_with_setup(
                || {
                    // Pre-populate tree with hot range keys
                    let tree = Arc::new(MassTree15::<u64>::new());
                    {
                        let guard = tree.guard();
                        for (i, key) in hot_keys.iter().enumerate() {
                            let _ = tree.insert_with_guard(key, i as u64, &guard);
                        }
                    }
                    tree
                },
                |tree| {
                    let start = Arc::new(Barrier::new(threads));
                    let keys = Arc::new(hot_keys.clone());
                    let handles: Vec<_> = (0..threads)
                        .map(|t| {
                            let tree = Arc::clone(&tree);
                            let keys = Arc::clone(&keys);
                            let start = Arc::clone(&start);

                            thread::spawn(move || {
                                let guard = tree.guard();
                                let mut state = (t as u64).wrapping_mul(0x517c_c1b7_2722_0a95);
                                let mut sum = 0u64;
                                start.wait();

                                for _ in 0..OPS_PER_THREAD {
                                    state = state
                                        .wrapping_mul(6_364_136_223_846_793_005)
                                        .wrapping_add(1);
                                    let idx = (state % HOT_RANGE) as usize;
                                    if let Some(v) = tree.get_ref(&keys[idx], &guard) {
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

                    black_box(&tree);
                },
            );
        });

        let hot_keys = hot_keys.clone();

        group.bench_function(BenchmarkId::new("tree_index", threads), |b| {
            let hot_keys = hot_keys.clone();

            b.iter_with_setup(
                || {
                    // Pre-populate tree with hot range keys
                    let tree = Arc::new(TreeIndex::<[u8; 8], u64>::new());
                    for (i, key) in hot_keys.iter().enumerate() {
                        let _ = tree.insert_sync(*key, i as u64);
                    }
                    tree
                },
                |tree| {
                    let start = Arc::new(Barrier::new(threads));
                    let keys = Arc::new(hot_keys.clone());
                    let handles: Vec<_> = (0..threads)
                        .map(|t| {
                            let tree = Arc::clone(&tree);
                            let keys = Arc::clone(&keys);
                            let start = Arc::clone(&start);

                            thread::spawn(move || {
                                let guard = SddGuard::new();
                                let mut state = (t as u64).wrapping_mul(0x517c_c1b7_2722_0a95);
                                let mut sum = 0u64;
                                start.wait();

                                for _ in 0..OPS_PER_THREAD {
                                    state = state
                                        .wrapping_mul(6_364_136_223_846_793_005)
                                        .wrapping_add(1);
                                    let idx = (state % HOT_RANGE) as usize;
                                    if let Some(v) = tree.peek(&keys[idx], &guard) {
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

                    black_box(&tree);
                },
            );
        });
    }
    group.finish();
}

// =============================================================================
// 03: LAYER EXPLOSION - Long Keys Creating Many Layers
// =============================================================================
//
// Uses 64-byte keys (8 layers!) with unique data in each 8-byte chunk.
// This tests:
// - Layer creation under concurrency
// - Splits at multiple tree depths
// - Deep traversal performance (8 pointer indirections for MassTree)

fn make_long_key(val: u64) -> [u8; 64] {
    let mut key = [0u8; 64];
    // Each 8-byte chunk uses a different multiplier to ensure unique values per layer
    key[0..8].copy_from_slice(&val.to_be_bytes());
    key[8..16].copy_from_slice(&val.wrapping_mul(0x9e3779b97f4a7c15).to_be_bytes());
    key[16..24].copy_from_slice(&val.wrapping_mul(0x517cc1b727220a95).to_be_bytes());
    key[24..32].copy_from_slice(&val.wrapping_mul(0x2545f4914f6cdd1d).to_be_bytes());
    key[32..40].copy_from_slice(&val.wrapping_mul(0x1c6a5e26e2e15b3d).to_be_bytes());
    key[40..48].copy_from_slice(&val.wrapping_mul(0x369dea0f31a53f85).to_be_bytes());
    key[48..56].copy_from_slice(&val.wrapping_mul(0x27d4eb2d2d5e5db5).to_be_bytes());
    key[56..64].copy_from_slice(&val.wrapping_mul(0x6c62272e07bb0142).to_be_bytes());
    key
}

fn bench_03_layer_explosion(c: &mut Criterion) {
    let mut group = c.benchmark_group("03_layer_explosion");
    group.sample_size(20);

    const OPS_PER_THREAD: usize = 25_000;

    for threads in [1, 2, 3, 4, 5, 6] {
        group.bench_function(BenchmarkId::new("masstree15", threads), |b| {
            b.iter_with_setup(
                || {
                    (
                        Arc::new(MassTree15::<u64>::new()),
                        Arc::new(AtomicU64::new(0)),
                    )
                },
                |(tree, counter)| {
                    let start = Arc::new(Barrier::new(threads));
                    let handles: Vec<_> = (0..threads)
                        .map(|_| {
                            let tree = Arc::clone(&tree);
                            let counter = Arc::clone(&counter);
                            let start = Arc::clone(&start);

                            thread::spawn(move || {
                                let guard = tree.guard();
                                start.wait();
                                for _ in 0..OPS_PER_THREAD {
                                    let key_val = counter.fetch_add(1, Ordering::Relaxed);
                                    let key = make_long_key(key_val);
                                    let _ = tree.insert_with_guard(&key, key_val, &guard);
                                }
                            })
                        })
                        .collect();

                    for h in handles {
                        h.join().unwrap();
                    }

                    black_box(&tree);
                },
            );
        });

        group.bench_function(BenchmarkId::new("tree_index", threads), |b| {
            b.iter_with_setup(
                || {
                    (
                        Arc::new(TreeIndex::<[u8; 64], u64>::new()),
                        Arc::new(AtomicU64::new(0)),
                    )
                },
                |(tree, counter)| {
                    let start = Arc::new(Barrier::new(threads));
                    let handles: Vec<_> = (0..threads)
                        .map(|_| {
                            let tree = Arc::clone(&tree);
                            let counter = Arc::clone(&counter);
                            let start = Arc::clone(&start);

                            thread::spawn(move || {
                                start.wait();
                                for _ in 0..OPS_PER_THREAD {
                                    let key_val = counter.fetch_add(1, Ordering::Relaxed);
                                    let key = make_long_key(key_val);
                                    let _ = tree.insert_sync(key, key_val);
                                }
                            })
                        })
                        .collect();

                    for h in handles {
                        h.join().unwrap();
                    }

                    black_box(&tree);
                },
            );
        });
    }
    group.finish();
}

// =============================================================================
// 04: SPLIT STORM - Readers During Heavy Writes
// =============================================================================
//
// Half the threads do heavy inserts (causing splits).
// Half the threads do reads (must handle version changes).
//
// This directly tests whether double-buffer helps readers avoid restarts
// when writers are causing splits.

fn bench_04_split_storm(c: &mut Criterion) {
    let mut group = c.benchmark_group("04_split_storm");
    group.sample_size(20);

    const WRITE_OPS: usize = 50_000;
    const READ_OPS: usize = 100_000;

    let initial_keys: Vec<[u8; 8]> = (0u64..10_000).map(|i| i.to_be_bytes()).collect();

    for total_threads in [1, 2, 3, 4, 5, 6] {
        let write_threads = total_threads / 2;
        let read_threads = total_threads - write_threads;
        let initial_keys = initial_keys.clone();

        group.bench_function(BenchmarkId::new("masstree15", total_threads), |b| {
            let initial_keys = initial_keys.clone();

            b.iter_with_setup(
                || {
                    let tree = Arc::new(MassTree15::<u64>::new());
                    {
                        let guard = tree.guard();
                        for (i, key) in initial_keys.iter().enumerate() {
                            let _ = tree.insert_with_guard(key, i as u64, &guard);
                        }
                    }
                    (tree, Arc::new(AtomicU64::new(10_000)))
                },
                |(tree, counter)| {
                    let start = Arc::new(Barrier::new(total_threads));
                    let keys = Arc::new(initial_keys.clone());

                    let writer_handles: Vec<_> = (0..write_threads)
                        .map(|_| {
                            let tree = Arc::clone(&tree);
                            let counter = Arc::clone(&counter);
                            let start = Arc::clone(&start);

                            thread::spawn(move || {
                                let guard = tree.guard();
                                start.wait();
                                for _ in 0..WRITE_OPS {
                                    let key_val = counter.fetch_add(1, Ordering::Relaxed);
                                    let key = key_val.to_be_bytes();
                                    let _ = tree.insert_with_guard(&key, key_val, &guard);
                                }
                            })
                        })
                        .collect();

                    let reader_handles: Vec<_> = (0..read_threads)
                        .map(|t| {
                            let tree = Arc::clone(&tree);
                            let keys = Arc::clone(&keys);
                            let start = Arc::clone(&start);

                            thread::spawn(move || {
                                let guard = tree.guard();
                                let mut sum = 0u64;
                                let offset = t * 7919;
                                start.wait();
                                for i in 0..READ_OPS {
                                    let idx = (i + offset) % keys.len();
                                    if let Some(v) = tree.get_ref(&keys[idx], &guard) {
                                        sum += *v;
                                    }
                                }
                                black_box(sum);
                            })
                        })
                        .collect();

                    for h in writer_handles {
                        h.join().unwrap();
                    }
                    for h in reader_handles {
                        h.join().unwrap();
                    }

                    black_box(&tree);
                },
            );
        });

        let initial_keys = initial_keys.clone();

        group.bench_function(BenchmarkId::new("tree_index", total_threads), |b| {
            let initial_keys = initial_keys.clone();

            b.iter_with_setup(
                || {
                    let tree = Arc::new(TreeIndex::<[u8; 8], u64>::new());
                    for (i, key) in initial_keys.iter().enumerate() {
                        let _ = tree.insert_sync(*key, i as u64);
                    }
                    (tree, Arc::new(AtomicU64::new(10_000)))
                },
                |(tree, counter)| {
                    let start = Arc::new(Barrier::new(total_threads));
                    let keys = Arc::new(initial_keys.clone());

                    let writer_handles: Vec<_> = (0..write_threads)
                        .map(|_| {
                            let tree = Arc::clone(&tree);
                            let counter = Arc::clone(&counter);
                            let start = Arc::clone(&start);

                            thread::spawn(move || {
                                start.wait();
                                for _ in 0..WRITE_OPS {
                                    let key_val = counter.fetch_add(1, Ordering::Relaxed);
                                    let key = key_val.to_be_bytes();
                                    let _ = tree.insert_sync(key, key_val);
                                }
                            })
                        })
                        .collect();

                    let reader_handles: Vec<_> = (0..read_threads)
                        .map(|t| {
                            let tree = Arc::clone(&tree);
                            let keys = Arc::clone(&keys);
                            let start = Arc::clone(&start);

                            thread::spawn(move || {
                                let guard = SddGuard::new();
                                let mut sum = 0u64;
                                let offset = t * 7919;
                                start.wait();
                                for i in 0..READ_OPS {
                                    let idx = (i + offset) % keys.len();
                                    if let Some(v) = tree.peek(&keys[idx], &guard) {
                                        sum += *v;
                                    }
                                }
                                black_box(sum);
                            })
                        })
                        .collect();

                    for h in writer_handles {
                        h.join().unwrap();
                    }
                    for h in reader_handles {
                        h.join().unwrap();
                    }

                    black_box(&tree);
                },
            );
        });
    }
    group.finish();
}

// =============================================================================
// 05: CASCADING SPLITS - Force Root Splits
// =============================================================================
//
// Insert keys in a pattern that guarantees cascading splits:
// - Sequential keys fill leaves
// - When leaves split, internodes fill
// - When internodes split, root may split
//
// Uses smaller batches to ensure we see the cascade effect.

fn bench_05_cascading_splits(c: &mut Criterion) {
    let mut group = c.benchmark_group("05_cascading_splits");
    group.sample_size(20);

    const BATCHES: usize = 100;
    const KEYS_PER_BATCH: usize = 500;

    for threads in [1, 2, 3, 4, 5, 6] {
        group.bench_function(BenchmarkId::new("masstree15", threads), |b| {
            b.iter_with_setup(
                || {
                    (
                        Arc::new(MassTree15::<u64>::new()),
                        Arc::new(AtomicU64::new(0)),
                    )
                },
                |(tree, counter)| {
                    let start = Arc::new(Barrier::new(threads));
                    let handles: Vec<_> = (0..threads)
                        .map(|_| {
                            let tree = Arc::clone(&tree);
                            let counter = Arc::clone(&counter);
                            let start = Arc::clone(&start);

                            thread::spawn(move || {
                                let guard = tree.guard();
                                start.wait();
                                for _ in 0..BATCHES {
                                    for _ in 0..KEYS_PER_BATCH {
                                        let key_val = counter.fetch_add(1, Ordering::Relaxed);
                                        let key = key_val.to_be_bytes();
                                        let _ = tree.insert_with_guard(&key, key_val, &guard);
                                    }
                                    std::hint::spin_loop();
                                }
                            })
                        })
                        .collect();

                    for h in handles {
                        h.join().unwrap();
                    }

                    black_box(&tree);
                },
            );
        });

        group.bench_function(BenchmarkId::new("tree_index", threads), |b| {
            b.iter_with_setup(
                || {
                    (
                        Arc::new(TreeIndex::<[u8; 8], u64>::new()),
                        Arc::new(AtomicU64::new(0)),
                    )
                },
                |(tree, counter)| {
                    let start = Arc::new(Barrier::new(threads));
                    let handles: Vec<_> = (0..threads)
                        .map(|_| {
                            let tree = Arc::clone(&tree);
                            let counter = Arc::clone(&counter);
                            let start = Arc::clone(&start);

                            thread::spawn(move || {
                                start.wait();
                                for _ in 0..BATCHES {
                                    for _ in 0..KEYS_PER_BATCH {
                                        let key_val = counter.fetch_add(1, Ordering::Relaxed);
                                        let key = key_val.to_be_bytes();
                                        let _ = tree.insert_sync(key, key_val);
                                    }
                                    std::hint::spin_loop();
                                }
                            })
                        })
                        .collect();

                    for h in handles {
                        h.join().unwrap();
                    }

                    black_box(&tree);
                },
            );
        });
    }
    group.finish();
}

// =============================================================================
// 06: INTERLEAVED SPLITS - Alternating Key Patterns
// =============================================================================
//
// Threads insert keys that interleave with each other:
// Thread 0: 0, 2, 4, 6, ...
// Thread 1: 1, 3, 5, 7, ...
//
// This creates maximum "key escaped to sibling" scenarios that
// the double-buffer optimization is specifically designed to handle.

fn bench_06_interleaved_splits(c: &mut Criterion) {
    let mut group = c.benchmark_group("06_interleaved_splits");
    group.sample_size(20);

    const OPS_PER_THREAD: usize = 50_000;

    for threads in [1, 2, 3, 4, 5, 6] {
        group.bench_function(BenchmarkId::new("masstree15", threads), |b| {
            b.iter_with_setup(
                || Arc::new(MassTree15::<u64>::new()),
                |tree| {
                    let start = Arc::new(Barrier::new(threads));
                    let handles: Vec<_> = (0..threads)
                        .map(|t| {
                            let tree = Arc::clone(&tree);
                            let start = Arc::clone(&start);

                            thread::spawn(move || {
                                let guard = tree.guard();
                                start.wait();
                                for i in 0..OPS_PER_THREAD {
                                    let key_val = (t + i * threads) as u64;
                                    let key = key_val.to_be_bytes();
                                    let _ = tree.insert_with_guard(&key, key_val, &guard);
                                }
                            })
                        })
                        .collect();

                    for h in handles {
                        h.join().unwrap();
                    }

                    black_box(&tree);
                },
            );
        });

        group.bench_function(BenchmarkId::new("tree_index", threads), |b| {
            b.iter_with_setup(
                || Arc::new(TreeIndex::<[u8; 8], u64>::new()),
                |tree| {
                    let start = Arc::new(Barrier::new(threads));
                    let handles: Vec<_> = (0..threads)
                        .map(|t| {
                            let tree = Arc::clone(&tree);
                            let start = Arc::clone(&start);

                            thread::spawn(move || {
                                start.wait();
                                for i in 0..OPS_PER_THREAD {
                                    let key_val = (t + i * threads) as u64;
                                    let key = key_val.to_be_bytes();
                                    let _ = tree.insert_sync(key, key_val);
                                }
                            })
                        })
                        .collect();

                    for h in handles {
                        h.join().unwrap();
                    }

                    black_box(&tree);
                },
            );
        });
    }
    group.finish();
}

// =============================================================================
// 07: PREFIX COLLISION SPLITS - Shared Prefix with Suffix Variance
// =============================================================================
//
// All keys share a common prefix but differ in suffix.
// This forces:
// - Layer creation for the shared prefix
// - Splits in the deeper layers
// - Tests trie structure under split pressure

fn make_prefix_key(prefix: u64, suffix: u64) -> [u8; 24] {
    let mut key = [0u8; 24];
    key[0..8].copy_from_slice(&prefix.to_be_bytes());
    key[8..16].copy_from_slice(&suffix.to_be_bytes());
    key[16..24].copy_from_slice(&suffix.wrapping_mul(0x9e3779b97f4a7c15).to_be_bytes());
    key
}

fn bench_07_prefix_collision_splits(c: &mut Criterion) {
    let mut group = c.benchmark_group("07_prefix_collision_splits");
    group.sample_size(20);

    const OPS_PER_THREAD: usize = 25_000;

    for threads in [1, 2, 3, 4, 5, 6] {
        group.bench_function(BenchmarkId::new("masstree15", threads), |b| {
            b.iter_with_setup(
                || {
                    (
                        Arc::new(MassTree15::<u64>::new()),
                        Arc::new(AtomicU64::new(0)),
                    )
                },
                |(tree, counter)| {
                    let start = Arc::new(Barrier::new(threads));
                    let handles: Vec<_> = (0..threads)
                        .map(|t| {
                            let tree = Arc::clone(&tree);
                            let counter = Arc::clone(&counter);
                            let start = Arc::clone(&start);

                            thread::spawn(move || {
                                let guard = tree.guard();
                                let prefix = (t % 4) as u64;
                                start.wait();
                                for _ in 0..OPS_PER_THREAD {
                                    let suffix = counter.fetch_add(1, Ordering::Relaxed);
                                    let key = make_prefix_key(prefix, suffix);
                                    let _ = tree.insert_with_guard(&key, suffix, &guard);
                                }
                            })
                        })
                        .collect();

                    for h in handles {
                        h.join().unwrap();
                    }

                    black_box(&tree);
                },
            );
        });

        group.bench_function(BenchmarkId::new("tree_index", threads), |b| {
            b.iter_with_setup(
                || {
                    (
                        Arc::new(TreeIndex::<[u8; 24], u64>::new()),
                        Arc::new(AtomicU64::new(0)),
                    )
                },
                |(tree, counter)| {
                    let start = Arc::new(Barrier::new(threads));
                    let handles: Vec<_> = (0..threads)
                        .map(|t| {
                            let tree = Arc::clone(&tree);
                            let counter = Arc::clone(&counter);
                            let start = Arc::clone(&start);

                            thread::spawn(move || {
                                let prefix = (t % 4) as u64;
                                start.wait();
                                for _ in 0..OPS_PER_THREAD {
                                    let suffix = counter.fetch_add(1, Ordering::Relaxed);
                                    let key = make_prefix_key(prefix, suffix);
                                    let _ = tree.insert_sync(key, suffix);
                                }
                            })
                        })
                        .collect();

                    for h in handles {
                        h.join().unwrap();
                    }

                    black_box(&tree);
                },
            );
        });
    }
    group.finish();
}

// =============================================================================
// CRITERION GROUPS AND MAIN
// =============================================================================

criterion_group!(
    benches,
    bench_01_sequential_flooding,
    bench_02_hot_range_contention,
    bench_03_layer_explosion,
    bench_04_split_storm,
    bench_05_cascading_splits,
    bench_06_interleaved_splits,
    bench_07_prefix_collision_splits,
);

criterion_main!(benches);
