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

use divan::{Bencher, black_box};
use masstree::MassTree24;
use scc::TreeIndex;
use sdd::Guard as SddGuard;
use std::sync::Arc;
use std::sync::Barrier;
use std::sync::atomic::{AtomicU64, Ordering};
use std::thread;

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

#[divan::bench_group(name = "01_sequential_flooding")]
mod sequential_flooding {
    use super::*;

    const OPS_PER_THREAD: usize = 100_000;

    #[divan::bench(args = [1, 2, 3, 4, 5, 6])]
    fn masstree24(bencher: Bencher, threads: usize) {
        bencher
            .with_inputs(|| {
                (
                    Arc::new(MassTree24::<u64>::new()),
                    Arc::new(AtomicU64::new(0)),
                )
            })
            .bench_local_values(|(tree, counter)| {
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
                                // All threads grab next sequential key
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

                tree
            });
    }

    #[divan::bench(args = [1, 2, 3, 4, 5, 6])]
    fn tree_index(bencher: Bencher, threads: usize) {
        bencher
            .with_inputs(|| {
                (
                    Arc::new(TreeIndex::<[u8; 8], u64>::new()),
                    Arc::new(AtomicU64::new(0)),
                )
            })
            .bench_local_values(|(tree, counter)| {
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

                tree
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

#[divan::bench_group(name = "02_hot_range_splits")]
mod hot_range_splits {
    use super::*;

    const OPS_PER_THREAD: usize = 50_000;
    const HOT_RANGE: u64 = 256; // Tiny range = maximum contention

    #[divan::bench(args = [1, 2, 3, 4, 5, 6])]
    fn masstree24(bencher: Bencher, threads: usize) {
        bencher
            .with_inputs(|| Arc::new(MassTree24::<u64>::new()))
            .bench_local_values(|tree| {
                let start = Arc::new(Barrier::new(threads));
                let counter = Arc::new(AtomicU64::new(0));
                let handles: Vec<_> = (0..threads)
                    .map(|t| {
                        let tree = Arc::clone(&tree);
                        let counter = Arc::clone(&counter);
                        let start = Arc::clone(&start);

                        thread::spawn(move || {
                            let guard = tree.guard();
                            // Different starting point per thread for variety
                            let mut state = (t as u64).wrapping_mul(0x517c_c1b7_2722_0a95);
                            start.wait();

                            for _ in 0..OPS_PER_THREAD {
                                // LCG for pseudo-random access within hot range
                                state = state
                                    .wrapping_mul(6_364_136_223_846_793_005)
                                    .wrapping_add(1);
                                let key_val = state % HOT_RANGE;
                                let key = key_val.to_be_bytes();
                                let val = counter.fetch_add(1, Ordering::Relaxed);
                                let _ = tree.insert_with_guard(&key, val, &guard);
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
    #[divan::bench(args = [1, 2, 3, 4, 5, 6])]
    fn tree_index(bencher: Bencher, threads: usize) {
        bencher
            .with_inputs(|| Arc::new(TreeIndex::<[u8; 8], u64>::new()))
            .bench_local_values(|tree| {
                let start = Arc::new(Barrier::new(threads));
                let counter = Arc::new(AtomicU64::new(0));
                let handles: Vec<_> = (0..threads)
                    .map(|t| {
                        let tree = Arc::clone(&tree);
                        let counter = Arc::clone(&counter);
                        let start = Arc::clone(&start);
                        thread::spawn(move || {
                            let mut state = (t as u64).wrapping_mul(0x517c_c1b7_2722_0a95);
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
// 03: LAYER EXPLOSION - Long Keys Creating Many Layers
// =============================================================================
//
// Uses 64-byte keys (8 layers!) with sequential patterns.
// This tests:
// - Layer creation under concurrency
// - Splits at multiple tree depths
// - Double-buffer handling of deep traversals

#[divan::bench_group(name = "03_layer_explosion")]
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
    fn masstree24(bencher: Bencher, threads: usize) {
        bencher
            .with_inputs(|| {
                (
                    Arc::new(MassTree24::<u64>::new()),
                    Arc::new(AtomicU64::new(0)),
                )
            })
            .bench_local_values(|(tree, counter)| {
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
                tree
            });
    }

    #[divan::bench(args = [1, 2, 3, 4, 5, 6])]
    fn tree_index(bencher: Bencher, threads: usize) {
        bencher
            .with_inputs(|| {
                (
                    Arc::new(TreeIndex::<[u8; 64], u64>::new()),
                    Arc::new(AtomicU64::new(0)),
                )
            })
            .bench_local_values(|(tree, counter)| {
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
                tree
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

#[divan::bench_group(name = "04_split_storm")]
mod split_storm {
    use super::*;

    const WRITE_OPS: usize = 50_000;
    const READ_OPS: usize = 100_000;

    #[divan::bench(args = [1, 2, 3, 4, 5, 6])]
    fn masstree24(bencher: Bencher, total_threads: usize) {
        let write_threads = total_threads / 2;
        let read_threads = total_threads - write_threads;

        // Pre-populate with some keys for readers
        let initial_keys: Vec<[u8; 8]> = (0u64..10_000).map(|i| i.to_be_bytes()).collect();

        bencher
            .with_inputs(|| {
                let tree = Arc::new(MassTree24::<u64>::new());
                {
                    let guard = tree.guard();
                    for (i, key) in initial_keys.iter().enumerate() {
                        let _ = tree.insert_with_guard(key, i as u64, &guard);
                    }
                }
                (tree, Arc::new(AtomicU64::new(10_000)))
            })
            .bench_local_values(|(tree, counter)| {
                let start = Arc::new(Barrier::new(total_threads));
                let keys = Arc::new(initial_keys.clone());

                // Writer threads - sequential inserts causing splits
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

                // Reader threads - traverse during splits
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
                tree
            });
    }

    #[divan::bench(args = [1, 2, 3, 4, 5, 6])]
    fn tree_index(bencher: Bencher, total_threads: usize) {
        let write_threads = total_threads / 2;
        let read_threads = total_threads - write_threads;

        // Pre-populate with some keys for readers
        let initial_keys: Vec<[u8; 8]> = (0u64..10_000).map(|i| i.to_be_bytes()).collect();

        bencher
            .with_inputs(|| {
                let tree = Arc::new(TreeIndex::<[u8; 8], u64>::new());
                for (i, key) in initial_keys.iter().enumerate() {
                    let _ = tree.insert_sync(*key, i as u64);
                }
                (tree, Arc::new(AtomicU64::new(10_000)))
            })
            .bench_local_values(|(tree, counter)| {
                let start = Arc::new(Barrier::new(total_threads));
                let keys = Arc::new(initial_keys.clone());

                // Writer threads - sequential inserts causing splits
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

                // Reader threads - traverse during splits
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
                tree
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

#[divan::bench_group(name = "05_cascading_splits")]
mod cascading_splits {
    use super::*;

    // Each thread does multiple small batches
    // This allows splits to cascade between batches
    const BATCHES: usize = 100;
    const KEYS_PER_BATCH: usize = 500;

    #[divan::bench(args = [1, 2, 3, 4, 5, 6])]
    fn masstree24(bencher: Bencher, threads: usize) {
        bencher
            .with_inputs(|| {
                (
                    Arc::new(MassTree24::<u64>::new()),
                    Arc::new(AtomicU64::new(0)),
                )
            })
            .bench_local_values(|(tree, counter)| {
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
                                // Insert a batch of sequential keys
                                for _ in 0..KEYS_PER_BATCH {
                                    let key_val = counter.fetch_add(1, Ordering::Relaxed);
                                    let key = key_val.to_be_bytes();
                                    let _ = tree.insert_with_guard(&key, key_val, &guard);
                                }
                                // Small yield to let other threads catch up
                                // This increases chance of concurrent splits
                                std::hint::spin_loop();
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

    #[divan::bench(args = [1, 2, 3, 4, 5, 6])]
    fn tree_index(bencher: Bencher, threads: usize) {
        bencher
            .with_inputs(|| {
                (
                    Arc::new(TreeIndex::<[u8; 8], u64>::new()),
                    Arc::new(AtomicU64::new(0)),
                )
            })
            .bench_local_values(|(tree, counter)| {
                let start = Arc::new(Barrier::new(threads));
                let handles: Vec<_> = (0..threads)
                    .map(|_| {
                        let tree = Arc::clone(&tree);
                        let counter = Arc::clone(&counter);
                        let start = Arc::clone(&start);
                        thread::spawn(move || {
                            start.wait();
                            for _ in 0..BATCHES {
                                // Insert a batch of sequential keys
                                for _ in 0..KEYS_PER_BATCH {
                                    let key_val = counter.fetch_add(1, Ordering::Relaxed);
                                    let key = key_val.to_be_bytes();
                                    let _ = tree.insert_sync(key, key_val);
                                }
                                // Small yield to let other threads catch up
                                // This increases chance of concurrent splits
                                std::hint::spin_loop();
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
// 06: INTERLEAVED SPLITS - Alternating Key Patterns
// =============================================================================
//
// Threads insert keys that interleave with each other:
// Thread 0: 0, 2, 4, 6, ...
// Thread 1: 1, 3, 5, 7, ...
//
// This creates maximum "key escaped to sibling" scenarios that
// the double-buffer optimization is specifically designed to handle.

#[divan::bench_group(name = "06_interleaved_splits")]
mod interleaved_splits {
    use super::*;

    const OPS_PER_THREAD: usize = 50_000;

    #[divan::bench(args = [1, 2, 3, 4, 5, 6])]
    fn masstree24(bencher: Bencher, threads: usize) {
        bencher
            .with_inputs(|| Arc::new(MassTree24::<u64>::new()))
            .bench_local_values(|tree| {
                let start = Arc::new(Barrier::new(threads));
                let handles: Vec<_> = (0..threads)
                    .map(|t| {
                        let tree = Arc::clone(&tree);
                        let start = Arc::clone(&start);
                        thread::spawn(move || {
                            let guard = tree.guard();
                            start.wait();
                            // Each thread inserts keys: t, t+threads, t+2*threads, ...
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
                tree
            });
    }

    #[divan::bench(args = [1, 2, 3, 4, 5, 6])]
    fn tree_index(bencher: Bencher, threads: usize) {
        bencher
            .with_inputs(|| Arc::new(TreeIndex::<[u8; 8], u64>::new()))
            .bench_local_values(|tree| {
                let start = Arc::new(Barrier::new(threads));
                let handles: Vec<_> = (0..threads)
                    .map(|t| {
                        let tree = Arc::clone(&tree);
                        let start = Arc::clone(&start);
                        thread::spawn(move || {
                            start.wait();
                            // Each thread inserts keys: t, t+threads, t+2*threads, ...
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
                tree
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

#[divan::bench_group(name = "07_prefix_collision_splits")]
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
    fn masstree24(bencher: Bencher, threads: usize) {
        bencher
            .with_inputs(|| {
                (
                    Arc::new(MassTree24::<u64>::new()),
                    Arc::new(AtomicU64::new(0)),
                )
            })
            .bench_local_values(|(tree, counter)| {
                let start = Arc::new(Barrier::new(threads));
                let handles: Vec<_> = (0..threads)
                    .map(|t| {
                        let tree = Arc::clone(&tree);
                        let counter = Arc::clone(&counter);
                        let start = Arc::clone(&start);
                        thread::spawn(move || {
                            let guard = tree.guard();
                            // Each thread uses a different prefix (forces some layer sharing)
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
                tree
            });
    }

    #[divan::bench(args = [1, 2, 3, 4, 5, 6])]
    fn tree_index(bencher: Bencher, threads: usize) {
        bencher
            .with_inputs(|| {
                (
                    Arc::new(TreeIndex::<[u8; 24], u64>::new()),
                    Arc::new(AtomicU64::new(0)),
                )
            })
            .bench_local_values(|(tree, counter)| {
                let start = Arc::new(Barrier::new(threads));
                let handles: Vec<_> = (0..threads)
                    .map(|t| {
                        let tree = Arc::clone(&tree);
                        let counter = Arc::clone(&counter);
                        let start = Arc::clone(&start);
                        thread::spawn(move || {
                            // Each thread uses a different prefix (forces some layer sharing)
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
                tree
            });
    }
}
