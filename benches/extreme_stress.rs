//! Extreme stress tests for split-heavy scenarios.
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
//! ## Methodology
//!
//! Each benchmark follows a rigorous methodology:
//! 1. **Explicit warmup**: Each thread warms up the data structure before measurement
//! 2. **Memory barriers**: Inserted before/after measurement to prevent reordering
//! 3. **Consistent setup**: All benchmarks use `.with_inputs()` for fresh state
//! 4. **Increased samples**: 200 samples for better statistical significance
//! 5. **Throughput counters**: Operations per second for fair comparison
//!
//! ## Running
//!
//! ```bash
//! cargo bench --bench extreme_stress --features mimalloc
//! cargo bench --bench extreme_stress --features mimalloc -- split_storm
//! ```

#![expect(clippy::unwrap_used)]
#![expect(clippy::pedantic)]
#![expect(clippy::indexing_slicing)]

mod bench_utils;

use bench_utils::{post_measurement_barrier, pre_measurement_barrier};
use divan::{Bencher, black_box};
use masstree::MassTree15Inline;
use scc::TreeIndex;
use sdd::Guard as SddGuard;
use std::sync::Arc;
use std::sync::Barrier;
use std::sync::atomic::{AtomicU64, Ordering};
use std::thread;

/// Warmup iterations per thread before measurement
const WARMUP_OPS: usize = 500;

fn main() {
    divan::main();
}

fn tree_index_upsert_sync<const K: usize>(
    tree: &TreeIndex<[u8; K], u64>,
    key: [u8; K],
    value: u64,
) {
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

#[divan::bench_group(name = "01_sequential_flooding", sample_count = 200)]
mod sequential_flooding {
    use super::*;

    const OPS_PER_THREAD: usize = 100_000;

    #[divan::bench(args = [1, 2, 3, 4, 5, 6])]
    fn masstree15(bencher: Bencher, threads: usize) {
        bencher
            .counter(divan::counter::ItemsCount::new(threads * OPS_PER_THREAD))
            .with_inputs(|| {
                (
                    Arc::new(MassTree15Inline::<u64>::new()),
                    Arc::new(AtomicU64::new(0)),
                )
            })
            .bench_local_values(|(tree, counter)| {
                let warmup_done = Arc::new(Barrier::new(threads));
                let start = Arc::new(Barrier::new(threads));
                let handles: Vec<_> = (0..threads)
                    .map(|_| {
                        let tree = Arc::clone(&tree);
                        let counter = Arc::clone(&counter);
                        let warmup_done = Arc::clone(&warmup_done);
                        let start = Arc::clone(&start);

                        thread::spawn(move || {
                            let guard = tree.guard();

                            // === WARMUP PHASE ===
                            for _ in 0..WARMUP_OPS {
                                let key_val = counter.fetch_add(1, Ordering::Relaxed);
                                let key = key_val.to_be_bytes();
                                let _ = tree.insert_with_guard(&key, key_val, &guard);
                            }
                            warmup_done.wait();

                            // === MEASUREMENT PHASE ===
                            pre_measurement_barrier();
                            start.wait();

                            for _ in 0..OPS_PER_THREAD {
                                let key_val = counter.fetch_add(1, Ordering::Relaxed);
                                let key = key_val.to_be_bytes();
                                let _ = tree.insert_with_guard(&key, key_val, &guard);
                            }

                            post_measurement_barrier();
                        })
                    })
                    .collect();

                for h in handles {
                    h.join().unwrap();
                }

                black_box(&tree);
            });
    }

    #[divan::bench(args = [1, 2, 3, 4, 5, 6])]
    fn tree_index(bencher: Bencher, threads: usize) {
        bencher
            .counter(divan::counter::ItemsCount::new(threads * OPS_PER_THREAD))
            .with_inputs(|| {
                (
                    Arc::new(TreeIndex::<[u8; 8], u64>::new()),
                    Arc::new(AtomicU64::new(0)),
                )
            })
            .bench_local_values(|(tree, counter)| {
                let warmup_done = Arc::new(Barrier::new(threads));
                let start = Arc::new(Barrier::new(threads));
                let handles: Vec<_> = (0..threads)
                    .map(|_| {
                        let tree = Arc::clone(&tree);
                        let counter = Arc::clone(&counter);
                        let warmup_done = Arc::clone(&warmup_done);
                        let start = Arc::clone(&start);

                        thread::spawn(move || {
                            // === WARMUP PHASE ===
                            for _ in 0..WARMUP_OPS {
                                let key_val = counter.fetch_add(1, Ordering::Relaxed);
                                let key = key_val.to_be_bytes();
                                let _ = tree.insert_sync(key, key_val);
                            }
                            warmup_done.wait();

                            // === MEASUREMENT PHASE ===
                            pre_measurement_barrier();
                            start.wait();

                            for _ in 0..OPS_PER_THREAD {
                                let key_val = counter.fetch_add(1, Ordering::Relaxed);
                                let key = key_val.to_be_bytes();
                                let _ = tree.insert_sync(key, key_val);
                            }

                            post_measurement_barrier();
                        })
                    })
                    .collect();

                for h in handles {
                    h.join().unwrap();
                }

                black_box(&tree);
            });
    }
}

// =============================================================================
// 02: HOT RANGE CONTENTION - Repeated Splits on Same Nodes
// =============================================================================
//
// All threads insert/update keys in a tiny range (256 keys).
// This forces:
// - The same leaves to split repeatedly
// - High lock contention on split operations
// - Frequent version changes triggering traversal retries

#[divan::bench_group(name = "02_hot_range_splits", sample_count = 200)]
mod hot_range_splits {
    use super::*;

    const OPS_PER_THREAD: usize = 50_000;
    const HOT_RANGE: u64 = 256; // Tiny range = maximum contention

    #[divan::bench(args = [1, 2, 3, 4, 5, 6])]
    fn masstree15(bencher: Bencher, threads: usize) {
        bencher
            .counter(divan::counter::ItemsCount::new(threads * OPS_PER_THREAD))
            .with_inputs(|| Arc::new(MassTree15Inline::<u64>::new()))
            .bench_local_values(|tree| {
                let warmup_done = Arc::new(Barrier::new(threads));
                let start = Arc::new(Barrier::new(threads));
                let counter = Arc::new(AtomicU64::new(0));
                let handles: Vec<_> = (0..threads)
                    .map(|t| {
                        let tree = Arc::clone(&tree);
                        let counter = Arc::clone(&counter);
                        let warmup_done = Arc::clone(&warmup_done);
                        let start = Arc::clone(&start);

                        thread::spawn(move || {
                            let guard = tree.guard();
                            let mut state = (t as u64).wrapping_mul(0x517c_c1b7_2722_0a95);

                            // === WARMUP PHASE ===
                            for _ in 0..WARMUP_OPS {
                                state = state
                                    .wrapping_mul(6_364_136_223_846_793_005)
                                    .wrapping_add(1);
                                let key_val = state % HOT_RANGE;
                                let key = key_val.to_be_bytes();
                                let val = counter.fetch_add(1, Ordering::Relaxed);
                                let _ = tree.insert_with_guard(&key, val, &guard);
                            }
                            warmup_done.wait();

                            // === MEASUREMENT PHASE ===
                            pre_measurement_barrier();
                            start.wait();

                            for _ in 0..OPS_PER_THREAD {
                                state = state
                                    .wrapping_mul(6_364_136_223_846_793_005)
                                    .wrapping_add(1);
                                let key_val = state % HOT_RANGE;
                                let key = key_val.to_be_bytes();
                                let val = counter.fetch_add(1, Ordering::Relaxed);
                                let _ = tree.insert_with_guard(&key, val, &guard);
                            }

                            post_measurement_barrier();
                        })
                    })
                    .collect();

                for h in handles {
                    h.join().unwrap();
                }

                black_box(&tree);
            });
    }

    #[divan::bench(args = [1, 2, 3, 4, 5, 6])]
    fn tree_index(bencher: Bencher, threads: usize) {
        bencher
            .counter(divan::counter::ItemsCount::new(threads * OPS_PER_THREAD))
            .with_inputs(|| Arc::new(TreeIndex::<[u8; 8], u64>::new()))
            .bench_local_values(|tree| {
                let warmup_done = Arc::new(Barrier::new(threads));
                let start = Arc::new(Barrier::new(threads));
                let counter = Arc::new(AtomicU64::new(0));
                let handles: Vec<_> = (0..threads)
                    .map(|t| {
                        let tree = Arc::clone(&tree);
                        let counter = Arc::clone(&counter);
                        let warmup_done = Arc::clone(&warmup_done);
                        let start = Arc::clone(&start);

                        thread::spawn(move || {
                            let mut state = (t as u64).wrapping_mul(0x517c_c1b7_2722_0a95);

                            // === WARMUP PHASE ===
                            for _ in 0..WARMUP_OPS {
                                state = state
                                    .wrapping_mul(6_364_136_223_846_793_005)
                                    .wrapping_add(1);
                                let key_val = state % HOT_RANGE;
                                let key = key_val.to_be_bytes();
                                let val = counter.fetch_add(1, Ordering::Relaxed);
                                tree_index_upsert_sync(&tree, key, val);
                            }
                            warmup_done.wait();

                            // === MEASUREMENT PHASE ===
                            pre_measurement_barrier();
                            start.wait();

                            for _ in 0..OPS_PER_THREAD {
                                state = state
                                    .wrapping_mul(6_364_136_223_846_793_005)
                                    .wrapping_add(1);
                                let key_val = state % HOT_RANGE;
                                let key = key_val.to_be_bytes();
                                let val = counter.fetch_add(1, Ordering::Relaxed);
                                tree_index_upsert_sync(&tree, key, val);
                            }

                            post_measurement_barrier();
                        })
                    })
                    .collect();

                for h in handles {
                    h.join().unwrap();
                }

                black_box(&tree);
            });
    }
}

// =============================================================================
// 03: LAYER EXPLOSION - Long Keys Creating Many Layers
// =============================================================================
//
// Uses 64-byte keys (8 layers!) with sequential patterns.
// This tests:
// - Layer creation under concurrency
// - Splits at multiple tree depths
// - Double-buffer handling of deep traversals

#[divan::bench_group(name = "03_layer_explosion", sample_count = 200)]
mod layer_explosion {
    use super::*;

    const OPS_PER_THREAD: usize = 25_000; // Fewer ops due to higher cost

    fn make_long_key(val: u64) -> [u8; 64] {
        let mut key = [0u8; 64];
        // Spread the value across multiple 8-byte chunks
        // This forces layer creation at each chunk boundary
        let bytes = val.to_be_bytes();
        key[0..8].copy_from_slice(&bytes);
        key[8..16].copy_from_slice(&bytes);
        key[16..24].copy_from_slice(&val.wrapping_mul(0x9e3779b97f4a7c15).to_be_bytes());
        key[24..32].copy_from_slice(&val.wrapping_mul(0x517cc1b727220a95).to_be_bytes());
        // Rest stays zero - creates shared prefix patterns
        key
    }

    #[divan::bench(args = [1, 2, 3, 4, 5, 6])]
    fn masstree15(bencher: Bencher, threads: usize) {
        bencher
            .counter(divan::counter::ItemsCount::new(threads * OPS_PER_THREAD))
            .with_inputs(|| {
                (
                    Arc::new(MassTree15Inline::<u64>::new()),
                    Arc::new(AtomicU64::new(0)),
                )
            })
            .bench_local_values(|(tree, counter)| {
                let warmup_done = Arc::new(Barrier::new(threads));
                let start = Arc::new(Barrier::new(threads));
                let handles: Vec<_> = (0..threads)
                    .map(|_| {
                        let tree = Arc::clone(&tree);
                        let counter = Arc::clone(&counter);
                        let warmup_done = Arc::clone(&warmup_done);
                        let start = Arc::clone(&start);

                        thread::spawn(move || {
                            let guard = tree.guard();

                            // === WARMUP PHASE ===
                            for _ in 0..WARMUP_OPS {
                                let key_val = counter.fetch_add(1, Ordering::Relaxed);
                                let key = make_long_key(key_val);
                                let _ = tree.insert_with_guard(&key, key_val, &guard);
                            }
                            warmup_done.wait();

                            // === MEASUREMENT PHASE ===
                            pre_measurement_barrier();
                            start.wait();

                            for _ in 0..OPS_PER_THREAD {
                                let key_val = counter.fetch_add(1, Ordering::Relaxed);
                                let key = make_long_key(key_val);
                                let _ = tree.insert_with_guard(&key, key_val, &guard);
                            }

                            post_measurement_barrier();
                        })
                    })
                    .collect();

                for h in handles {
                    h.join().unwrap();
                }

                black_box(&tree);
            });
    }

    #[divan::bench(args = [1, 2, 3, 4, 5, 6])]
    fn tree_index(bencher: Bencher, threads: usize) {
        bencher
            .counter(divan::counter::ItemsCount::new(threads * OPS_PER_THREAD))
            .with_inputs(|| {
                (
                    Arc::new(TreeIndex::<[u8; 64], u64>::new()),
                    Arc::new(AtomicU64::new(0)),
                )
            })
            .bench_local_values(|(tree, counter)| {
                let warmup_done = Arc::new(Barrier::new(threads));
                let start = Arc::new(Barrier::new(threads));
                let handles: Vec<_> = (0..threads)
                    .map(|_| {
                        let tree = Arc::clone(&tree);
                        let counter = Arc::clone(&counter);
                        let warmup_done = Arc::clone(&warmup_done);
                        let start = Arc::clone(&start);

                        thread::spawn(move || {
                            // === WARMUP PHASE ===
                            for _ in 0..WARMUP_OPS {
                                let key_val = counter.fetch_add(1, Ordering::Relaxed);
                                let key = make_long_key(key_val);
                                let _ = tree.insert_sync(key, key_val);
                            }
                            warmup_done.wait();

                            // === MEASUREMENT PHASE ===
                            pre_measurement_barrier();
                            start.wait();

                            for _ in 0..OPS_PER_THREAD {
                                let key_val = counter.fetch_add(1, Ordering::Relaxed);
                                let key = make_long_key(key_val);
                                let _ = tree.insert_sync(key, key_val);
                            }

                            post_measurement_barrier();
                        })
                    })
                    .collect();

                for h in handles {
                    h.join().unwrap();
                }

                black_box(&tree);
            });
    }
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

#[divan::bench_group(name = "04_split_storm", sample_count = 200)]
mod split_storm {
    use super::*;

    const WRITE_OPS: usize = 50_000;
    const READ_OPS: usize = 100_000;
    const INITIAL_KEYS: usize = 10_000;

    #[divan::bench(args = [1, 2, 3, 4, 5, 6])]
    fn masstree15(bencher: Bencher, total_threads: usize) {
        let write_threads = total_threads / 2;
        let read_threads = total_threads - write_threads;
        let total_ops = write_threads * WRITE_OPS + read_threads * READ_OPS;

        // Pre-populate with some keys for readers
        let initial_keys: Vec<[u8; 8]> = (0u64..INITIAL_KEYS as u64)
            .map(|i| i.to_be_bytes())
            .collect();

        bencher
            .counter(divan::counter::ItemsCount::new(total_ops))
            .with_inputs(|| {
                let tree = Arc::new(MassTree15Inline::<u64>::new());
                {
                    let guard = tree.guard();
                    for (i, key) in initial_keys.iter().enumerate() {
                        let _ = tree.insert_with_guard(key, i as u64, &guard);
                    }
                }
                (tree, Arc::new(AtomicU64::new(INITIAL_KEYS as u64)))
            })
            .bench_local_values(|(tree, counter)| {
                let warmup_done = Arc::new(Barrier::new(total_threads));
                let start = Arc::new(Barrier::new(total_threads));
                let keys = Arc::new(initial_keys.clone());

                // Writer threads - sequential inserts causing splits
                let writer_handles: Vec<_> = (0..write_threads)
                    .map(|_| {
                        let tree = Arc::clone(&tree);
                        let counter = Arc::clone(&counter);
                        let warmup_done = Arc::clone(&warmup_done);
                        let start = Arc::clone(&start);

                        thread::spawn(move || {
                            let guard = tree.guard();

                            // === WARMUP PHASE ===
                            for _ in 0..WARMUP_OPS {
                                let key_val = counter.fetch_add(1, Ordering::Relaxed);
                                let key = key_val.to_be_bytes();
                                let _ = tree.insert_with_guard(&key, key_val, &guard);
                            }
                            warmup_done.wait();

                            // === MEASUREMENT PHASE ===
                            pre_measurement_barrier();
                            start.wait();

                            for _ in 0..WRITE_OPS {
                                let key_val = counter.fetch_add(1, Ordering::Relaxed);
                                let key = key_val.to_be_bytes();
                                let _ = tree.insert_with_guard(&key, key_val, &guard);
                            }

                            post_measurement_barrier();
                        })
                    })
                    .collect();

                // Reader threads - traverse during splits
                let reader_handles: Vec<_> = (0..read_threads)
                    .map(|t| {
                        let tree = Arc::clone(&tree);
                        let keys = Arc::clone(&keys);
                        let warmup_done = Arc::clone(&warmup_done);
                        let start = Arc::clone(&start);

                        thread::spawn(move || {
                            let guard = tree.guard();
                            let offset = t * 7919;

                            // === WARMUP PHASE ===
                            for i in 0..WARMUP_OPS {
                                let idx = (i + offset) % keys.len();
                                black_box(tree.get_with_guard(&keys[idx], &guard));
                            }
                            warmup_done.wait();

                            // === MEASUREMENT PHASE ===
                            pre_measurement_barrier();
                            let mut sum = 0u64;
                            start.wait();

                            for i in 0..READ_OPS {
                                let idx = (i + offset) % keys.len();
                                if let Some(v) = tree.get_with_guard(&keys[idx], &guard) {
                                    sum += v;
                                }
                            }

                            post_measurement_barrier();
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
            });
    }

    #[divan::bench(args = [1, 2, 3, 4, 5, 6])]
    fn tree_index(bencher: Bencher, total_threads: usize) {
        let write_threads = total_threads / 2;
        let read_threads = total_threads - write_threads;
        let total_ops = write_threads * WRITE_OPS + read_threads * READ_OPS;

        // Pre-populate with some keys for readers
        let initial_keys: Vec<[u8; 8]> = (0u64..INITIAL_KEYS as u64)
            .map(|i| i.to_be_bytes())
            .collect();

        bencher
            .counter(divan::counter::ItemsCount::new(total_ops))
            .with_inputs(|| {
                let tree = Arc::new(TreeIndex::<[u8; 8], u64>::new());
                for (i, key) in initial_keys.iter().enumerate() {
                    let _ = tree.insert_sync(*key, i as u64);
                }
                (tree, Arc::new(AtomicU64::new(INITIAL_KEYS as u64)))
            })
            .bench_local_values(|(tree, counter)| {
                let warmup_done = Arc::new(Barrier::new(total_threads));
                let start = Arc::new(Barrier::new(total_threads));
                let keys = Arc::new(initial_keys.clone());

                // Writer threads - sequential inserts causing splits
                let writer_handles: Vec<_> = (0..write_threads)
                    .map(|_| {
                        let tree = Arc::clone(&tree);
                        let counter = Arc::clone(&counter);
                        let warmup_done = Arc::clone(&warmup_done);
                        let start = Arc::clone(&start);

                        thread::spawn(move || {
                            // === WARMUP PHASE ===
                            for _ in 0..WARMUP_OPS {
                                let key_val = counter.fetch_add(1, Ordering::Relaxed);
                                let key = key_val.to_be_bytes();
                                let _ = tree.insert_sync(key, key_val);
                            }
                            warmup_done.wait();

                            // === MEASUREMENT PHASE ===
                            pre_measurement_barrier();
                            start.wait();

                            for _ in 0..WRITE_OPS {
                                let key_val = counter.fetch_add(1, Ordering::Relaxed);
                                let key = key_val.to_be_bytes();
                                let _ = tree.insert_sync(key, key_val);
                            }

                            post_measurement_barrier();
                        })
                    })
                    .collect();

                // Reader threads - traverse during splits
                let reader_handles: Vec<_> = (0..read_threads)
                    .map(|t| {
                        let tree = Arc::clone(&tree);
                        let keys = Arc::clone(&keys);
                        let warmup_done = Arc::clone(&warmup_done);
                        let start = Arc::clone(&start);

                        thread::spawn(move || {
                            let guard = SddGuard::new();
                            let offset = t * 7919;

                            // === WARMUP PHASE ===
                            for i in 0..WARMUP_OPS {
                                let idx = (i + offset) % keys.len();
                                black_box(tree.peek(&keys[idx], &guard));
                            }
                            warmup_done.wait();

                            // === MEASUREMENT PHASE ===
                            pre_measurement_barrier();
                            let mut sum = 0u64;
                            start.wait();

                            for i in 0..READ_OPS {
                                let idx = (i + offset) % keys.len();
                                if let Some(v) = tree.peek(&keys[idx], &guard) {
                                    sum += *v;
                                }
                            }

                            post_measurement_barrier();
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
            });
    }
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

#[divan::bench_group(name = "05_cascading_splits", sample_count = 200)]
mod cascading_splits {
    use super::*;

    // Each thread does multiple small batches
    // This allows splits to cascade between batches
    const BATCHES: usize = 100;
    const KEYS_PER_BATCH: usize = 500;
    const WARMUP_BATCHES: usize = 5;

    #[divan::bench(args = [1, 2, 3, 4, 5, 6])]
    fn masstree15(bencher: Bencher, threads: usize) {
        let total_ops = threads * BATCHES * KEYS_PER_BATCH;

        bencher
            .counter(divan::counter::ItemsCount::new(total_ops))
            .with_inputs(|| {
                (
                    Arc::new(MassTree15Inline::<u64>::new()),
                    Arc::new(AtomicU64::new(0)),
                )
            })
            .bench_local_values(|(tree, counter)| {
                let warmup_done = Arc::new(Barrier::new(threads));
                let start = Arc::new(Barrier::new(threads));
                let handles: Vec<_> = (0..threads)
                    .map(|_| {
                        let tree = Arc::clone(&tree);
                        let counter = Arc::clone(&counter);
                        let warmup_done = Arc::clone(&warmup_done);
                        let start = Arc::clone(&start);

                        thread::spawn(move || {
                            let guard = tree.guard();

                            // === WARMUP PHASE ===
                            for _ in 0..WARMUP_BATCHES {
                                for _ in 0..KEYS_PER_BATCH {
                                    let key_val = counter.fetch_add(1, Ordering::Relaxed);
                                    let key = key_val.to_be_bytes();
                                    let _ = tree.insert_with_guard(&key, key_val, &guard);
                                }
                                std::hint::spin_loop();
                            }
                            warmup_done.wait();

                            // === MEASUREMENT PHASE ===
                            pre_measurement_barrier();
                            start.wait();

                            for _ in 0..BATCHES {
                                for _ in 0..KEYS_PER_BATCH {
                                    let key_val = counter.fetch_add(1, Ordering::Relaxed);
                                    let key = key_val.to_be_bytes();
                                    let _ = tree.insert_with_guard(&key, key_val, &guard);
                                }
                                std::hint::spin_loop();
                            }

                            post_measurement_barrier();
                        })
                    })
                    .collect();

                for h in handles {
                    h.join().unwrap();
                }

                black_box(&tree);
            });
    }

    #[divan::bench(args = [1, 2, 3, 4, 5, 6])]
    fn tree_index(bencher: Bencher, threads: usize) {
        let total_ops = threads * BATCHES * KEYS_PER_BATCH;

        bencher
            .counter(divan::counter::ItemsCount::new(total_ops))
            .with_inputs(|| {
                (
                    Arc::new(TreeIndex::<[u8; 8], u64>::new()),
                    Arc::new(AtomicU64::new(0)),
                )
            })
            .bench_local_values(|(tree, counter)| {
                let warmup_done = Arc::new(Barrier::new(threads));
                let start = Arc::new(Barrier::new(threads));
                let handles: Vec<_> = (0..threads)
                    .map(|_| {
                        let tree = Arc::clone(&tree);
                        let counter = Arc::clone(&counter);
                        let warmup_done = Arc::clone(&warmup_done);
                        let start = Arc::clone(&start);

                        thread::spawn(move || {
                            // === WARMUP PHASE ===
                            for _ in 0..WARMUP_BATCHES {
                                for _ in 0..KEYS_PER_BATCH {
                                    let key_val = counter.fetch_add(1, Ordering::Relaxed);
                                    let key = key_val.to_be_bytes();
                                    let _ = tree.insert_sync(key, key_val);
                                }
                                std::hint::spin_loop();
                            }
                            warmup_done.wait();

                            // === MEASUREMENT PHASE ===
                            pre_measurement_barrier();
                            start.wait();

                            for _ in 0..BATCHES {
                                for _ in 0..KEYS_PER_BATCH {
                                    let key_val = counter.fetch_add(1, Ordering::Relaxed);
                                    let key = key_val.to_be_bytes();
                                    let _ = tree.insert_sync(key, key_val);
                                }
                                std::hint::spin_loop();
                            }

                            post_measurement_barrier();
                        })
                    })
                    .collect();

                for h in handles {
                    h.join().unwrap();
                }

                black_box(&tree);
            });
    }
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

#[divan::bench_group(name = "06_interleaved_splits", sample_count = 200)]
mod interleaved_splits {
    use super::*;

    const OPS_PER_THREAD: usize = 50_000;

    #[divan::bench(args = [1, 2, 3, 4, 5, 6])]
    fn masstree15(bencher: Bencher, threads: usize) {
        bencher
            .counter(divan::counter::ItemsCount::new(threads * OPS_PER_THREAD))
            .with_inputs(|| Arc::new(MassTree15Inline::<u64>::new()))
            .bench_local_values(|tree| {
                let warmup_done = Arc::new(Barrier::new(threads));
                let start = Arc::new(Barrier::new(threads));
                let handles: Vec<_> = (0..threads)
                    .map(|t| {
                        let tree = Arc::clone(&tree);
                        let warmup_done = Arc::clone(&warmup_done);
                        let start = Arc::clone(&start);

                        thread::spawn(move || {
                            let guard = tree.guard();

                            // === WARMUP PHASE ===
                            // Interleaved pattern for warmup too
                            for i in 0..WARMUP_OPS {
                                let key_val = (t + i * threads) as u64;
                                let key = key_val.to_be_bytes();
                                let _ = tree.insert_with_guard(&key, key_val, &guard);
                            }
                            warmup_done.wait();

                            // === MEASUREMENT PHASE ===
                            pre_measurement_barrier();
                            start.wait();

                            // Each thread inserts keys: t, t+threads, t+2*threads, ...
                            // Continue from where warmup left off
                            for i in WARMUP_OPS..(WARMUP_OPS + OPS_PER_THREAD) {
                                let key_val = (t + i * threads) as u64;
                                let key = key_val.to_be_bytes();
                                let _ = tree.insert_with_guard(&key, key_val, &guard);
                            }

                            post_measurement_barrier();
                        })
                    })
                    .collect();

                for h in handles {
                    h.join().unwrap();
                }

                black_box(&tree);
            });
    }

    #[divan::bench(args = [1, 2, 3, 4, 5, 6])]
    fn tree_index(bencher: Bencher, threads: usize) {
        bencher
            .counter(divan::counter::ItemsCount::new(threads * OPS_PER_THREAD))
            .with_inputs(|| Arc::new(TreeIndex::<[u8; 8], u64>::new()))
            .bench_local_values(|tree| {
                let warmup_done = Arc::new(Barrier::new(threads));
                let start = Arc::new(Barrier::new(threads));
                let handles: Vec<_> = (0..threads)
                    .map(|t| {
                        let tree = Arc::clone(&tree);
                        let warmup_done = Arc::clone(&warmup_done);
                        let start = Arc::clone(&start);

                        thread::spawn(move || {
                            // === WARMUP PHASE ===
                            for i in 0..WARMUP_OPS {
                                let key_val = (t + i * threads) as u64;
                                let key = key_val.to_be_bytes();
                                let _ = tree.insert_sync(key, key_val);
                            }
                            warmup_done.wait();

                            // === MEASUREMENT PHASE ===
                            pre_measurement_barrier();
                            start.wait();

                            // Each thread inserts keys: t, t+threads, t+2*threads, ...
                            for i in WARMUP_OPS..(WARMUP_OPS + OPS_PER_THREAD) {
                                let key_val = (t + i * threads) as u64;
                                let key = key_val.to_be_bytes();
                                let _ = tree.insert_sync(key, key_val);
                            }

                            post_measurement_barrier();
                        })
                    })
                    .collect();

                for h in handles {
                    h.join().unwrap();
                }

                black_box(&tree);
            });
    }
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

#[divan::bench_group(name = "07_prefix_collision_splits", sample_count = 200)]
mod prefix_collision_splits {
    use super::*;

    const OPS_PER_THREAD: usize = 25_000;

    fn make_prefix_key(prefix: u64, suffix: u64) -> [u8; 24] {
        let mut key = [0u8; 24];
        key[0..8].copy_from_slice(&prefix.to_be_bytes());
        key[8..16].copy_from_slice(&suffix.to_be_bytes());
        key[16..24].copy_from_slice(&suffix.wrapping_mul(0x9e3779b97f4a7c15).to_be_bytes());
        key
    }

    #[divan::bench(args = [1, 2, 3, 4, 5, 6])]
    fn masstree15(bencher: Bencher, threads: usize) {
        bencher
            .counter(divan::counter::ItemsCount::new(threads * OPS_PER_THREAD))
            .with_inputs(|| {
                (
                    Arc::new(MassTree15Inline::<u64>::new()),
                    Arc::new(AtomicU64::new(0)),
                )
            })
            .bench_local_values(|(tree, counter)| {
                let warmup_done = Arc::new(Barrier::new(threads));
                let start = Arc::new(Barrier::new(threads));
                let handles: Vec<_> = (0..threads)
                    .map(|t| {
                        let tree = Arc::clone(&tree);
                        let counter = Arc::clone(&counter);
                        let warmup_done = Arc::clone(&warmup_done);
                        let start = Arc::clone(&start);

                        thread::spawn(move || {
                            let guard = tree.guard();
                            // Each thread uses a different prefix (forces some layer sharing)
                            let prefix = (t % 4) as u64;

                            // === WARMUP PHASE ===
                            for _ in 0..WARMUP_OPS {
                                let suffix = counter.fetch_add(1, Ordering::Relaxed);
                                let key = make_prefix_key(prefix, suffix);
                                let _ = tree.insert_with_guard(&key, suffix, &guard);
                            }
                            warmup_done.wait();

                            // === MEASUREMENT PHASE ===
                            pre_measurement_barrier();
                            start.wait();

                            for _ in 0..OPS_PER_THREAD {
                                let suffix = counter.fetch_add(1, Ordering::Relaxed);
                                let key = make_prefix_key(prefix, suffix);
                                let _ = tree.insert_with_guard(&key, suffix, &guard);
                            }

                            post_measurement_barrier();
                        })
                    })
                    .collect();

                for h in handles {
                    h.join().unwrap();
                }

                black_box(&tree);
            });
    }

    #[divan::bench(args = [1, 2, 3, 4, 5, 6])]
    fn tree_index(bencher: Bencher, threads: usize) {
        bencher
            .counter(divan::counter::ItemsCount::new(threads * OPS_PER_THREAD))
            .with_inputs(|| {
                (
                    Arc::new(TreeIndex::<[u8; 24], u64>::new()),
                    Arc::new(AtomicU64::new(0)),
                )
            })
            .bench_local_values(|(tree, counter)| {
                let warmup_done = Arc::new(Barrier::new(threads));
                let start = Arc::new(Barrier::new(threads));
                let handles: Vec<_> = (0..threads)
                    .map(|t| {
                        let tree = Arc::clone(&tree);
                        let counter = Arc::clone(&counter);
                        let warmup_done = Arc::clone(&warmup_done);
                        let start = Arc::clone(&start);

                        thread::spawn(move || {
                            // Each thread uses a different prefix (forces some layer sharing)
                            let prefix = (t % 4) as u64;

                            // === WARMUP PHASE ===
                            for _ in 0..WARMUP_OPS {
                                let suffix = counter.fetch_add(1, Ordering::Relaxed);
                                let key = make_prefix_key(prefix, suffix);
                                let _ = tree.insert_sync(key, suffix);
                            }
                            warmup_done.wait();

                            // === MEASUREMENT PHASE ===
                            pre_measurement_barrier();
                            start.wait();

                            for _ in 0..OPS_PER_THREAD {
                                let suffix = counter.fetch_add(1, Ordering::Relaxed);
                                let key = make_prefix_key(prefix, suffix);
                                let _ = tree.insert_sync(key, suffix);
                            }

                            post_measurement_barrier();
                        })
                    })
                    .collect();

                for h in handles {
                    h.join().unwrap();
                }

                black_box(&tree);
            });
    }
}
