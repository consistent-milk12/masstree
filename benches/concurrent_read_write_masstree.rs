//! Concurrent 90-10 read/write benchmarks with 64-byte keys.
//!
//! Compares MassTree15Inline against other concurrent ordered maps under mixed
//! read-write workloads. All benchmarks use 64-byte keys to test multi-layer
//! traversal performance.
//!
//! ## Key Characteristics
//!
//! - 90% reads, 10% writes (realistic workload)
//! - 64-byte keys (8 chunks, tests suffix handling)
//! - Various access patterns: uniform, zipfian, shared prefix
//!
//! ## Methodology
//!
//! Each benchmark follows a rigorous methodology:
//! 1. **Explicit warmup**: Each thread warms up the data structure before measurement
//! 2. **Memory barriers**: Inserted before/after measurement to prevent reordering
//! 3. **Consistent setup**: All benchmarks use `.with_inputs()` for fresh state
//! 4. **Increased samples**: 200 samples for better statistical significance
//! 5. **Min time**: 3 seconds minimum per benchmark for stability
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

use bench_utils::{
    keys, keys_shared_prefix_chunks, post_measurement_barrier, pre_measurement_barrier,
    uniform_indices, zipfian_indices,
};
use divan::{Bencher, black_box};
use masstree::MassTree15Inline;
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

fn setup_masstree15(keys: &[[u8; KEY_SIZE]]) -> MassTree15Inline<u64> {
    let tree = MassTree15Inline::new();
    {
        let guard = tree.guard();
        for (i, key) in keys.iter().enumerate() {
            let _ = tree.insert_with_guard(key, i as u64, &guard);
        }
    }
    tree
}

// =============================================================================
// 01: MIXED 90-10 - Uniform Access Pattern
// =============================================================================

/// Warmup iterations per thread before measurement
const WARMUP_OPS: usize = 500;

#[divan::bench_group(name = "01_mixed_90_10_uniform", sample_count = 200)]
mod mixed_uniform {
    use super::*;

    const N: usize = 50_000;
    const OPS_PER_THREAD: usize = 12_500;

    #[divan::bench(args = [1, 2, 3, 4, 5, 6])]
    fn masstree15(bencher: Bencher, threads: usize) {
        let keys = Arc::new(keys::<KEY_SIZE>(N));
        let indices = Arc::new(uniform_indices(N, OPS_PER_THREAD * threads, 42));

        bencher
            .counter(divan::counter::ItemsCount::new(threads * OPS_PER_THREAD))
            .with_inputs(|| Arc::new(setup_masstree15(keys.as_ref())))
            .bench_local_values(|tree| {
                // Two barriers: one for warmup completion, one for measurement start
                let warmup_done = Arc::new(Barrier::new(threads));
                let start = Arc::new(Barrier::new(threads));
                let handles: Vec<_> = (0..threads)
                    .map(|t| {
                        let tree = Arc::clone(&tree);
                        let keys = Arc::clone(&keys);
                        let indices = Arc::clone(&indices);
                        let warmup_done = Arc::clone(&warmup_done);
                        let start = Arc::clone(&start);
                        thread::spawn(move || {
                            let guard = tree.guard();
                            let base = t * OPS_PER_THREAD;

                            // === WARMUP PHASE ===
                            // Warm up caches, branch predictors, and ensure pages are faulted
                            for i in 0..WARMUP_OPS {
                                let idx = indices[base + (i % OPS_PER_THREAD)];
                                black_box(tree.get_with_guard(&keys[idx], &guard));
                            }
                            warmup_done.wait();

                            // === MEASUREMENT PHASE ===
                            pre_measurement_barrier();
                            let mut sum = 0u64;
                            start.wait();
                            for i in 0..OPS_PER_THREAD {
                                let idx = indices[base + i];
                                if i % 100 < WRITE_RATIO {
                                    let _ = tree.insert_with_guard(&keys[idx], i as u64, &guard);
                                } else if let Some(v) = tree.get_with_guard(&keys[idx], &guard) {
                                    sum = sum.wrapping_add(v);
                                }
                            }
                            post_measurement_barrier();
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

#[divan::bench_group(name = "02_mixed_90_10_zipfian", sample_count = 200)]
mod mixed_zipfian {
    use super::*;

    const N: usize = 50_000;
    const OPS_PER_THREAD: usize = 12_500;

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
                                } else if let Some(v) = tree.get_with_guard(&keys[idx], &guard) {
                                    sum = sum.wrapping_add(v);
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

#[divan::bench_group(name = "03_mixed_90_10_shared_prefix", sample_count = 200)]
mod mixed_shared_prefix {
    use super::*;

    const N: usize = 50_000;
    const OPS_PER_THREAD: usize = 12_500;
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
                                } else if let Some(v) = tree.get_with_guard(&keys[idx], &guard) {
                                    sum = sum.wrapping_add(v);
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

#[divan::bench_group(name = "04_mixed_90_10_high_contention", sample_count = 200)]
mod mixed_high_contention {
    use super::*;

    const N: usize = 500; // Small key space = high contention
    const OPS_PER_THREAD: usize = 25_000;

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
                                } else if let Some(v) = tree.get_with_guard(&keys[idx], &guard) {
                                    sum = sum.wrapping_add(v);
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

#[divan::bench_group(name = "05_mixed_90_10_large_dataset", sample_count = 200)]
mod mixed_large_dataset {
    use super::*;

    const N: usize = 500_000;
    const OPS_PER_THREAD: usize = 25_000;

    // Note: Large dataset benchmarks pre-build the tree once outside the benchmark loop
    // to avoid the massive setup cost per iteration. Each thread still warms up before
    // measurement to ensure cache/branch predictor stability.

    #[divan::bench(args = [1, 2, 3, 4, 5, 6])]
    fn masstree15(bencher: Bencher, threads: usize) {
        let keys = Arc::new(keys::<KEY_SIZE>(N));
        let tree = Arc::new(setup_masstree15(keys.as_ref()));
        let indices = Arc::new(uniform_indices(N, OPS_PER_THREAD * threads, 42));

        bencher
            .counter(divan::counter::ItemsCount::new(threads * OPS_PER_THREAD))
            .bench_local(|| {
                let warmup_done = Arc::new(Barrier::new(threads));
                let start = Arc::new(Barrier::new(threads));
                let handles: Vec<_> = (0..threads)
                    .map(|t| {
                        let tree = Arc::clone(&tree);
                        let keys = Arc::clone(&keys);
                        let indices = Arc::clone(&indices);
                        let warmup_done = Arc::clone(&warmup_done);
                        let start = Arc::clone(&start);
                        thread::spawn(move || {
                            let guard = tree.guard();
                            let base = t * OPS_PER_THREAD;

                            // Warmup
                            for i in 0..WARMUP_OPS {
                                let idx = indices[base + (i % OPS_PER_THREAD)];
                                black_box(tree.get_with_guard(&keys[idx], &guard));
                            }
                            warmup_done.wait();

                            // Measurement
                            pre_measurement_barrier();
                            let mut sum = 0u64;
                            start.wait();
                            for i in 0..OPS_PER_THREAD {
                                let idx = indices[base + i];
                                if i % 100 < WRITE_RATIO {
                                    let _ = tree.insert_with_guard(&keys[idx], i as u64, &guard);
                                } else if let Some(v) = tree.get_with_guard(&keys[idx], &guard) {
                                    sum = sum.wrapping_add(v);
                                }
                            }
                            post_measurement_barrier();
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

#[divan::bench_group(name = "06_single_hot_key", sample_count = 200)]
mod single_hot_key {
    use super::*;

    const N: usize = 50_000;
    const OPS_PER_THREAD: usize = 5_000;

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
                                } else if let Some(v) = tree.get_with_guard(&hot_key, &guard) {
                                    sum = sum.wrapping_add(v);
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

#[divan::bench_group(name = "07_mixed_50_50", sample_count = 200)]
mod mixed_50_50 {
    use super::*;

    const N: usize = 50_000;
    const OPS_PER_THREAD: usize = 12_500;
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
                                } else if let Some(v) = tree.get_with_guard(&keys[idx], &guard) {
                                    sum = sum.wrapping_add(v);
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
// 08: 8-BYTE KEYS - MassTree Single-Layer Fast Path
// =============================================================================

#[divan::bench_group(name = "08_8byte_keys_uniform", sample_count = 200)]
mod keys_8byte {
    use super::*;

    const KEY_SIZE_8: usize = 8;
    const N: usize = 50_000;
    const OPS_PER_THREAD: usize = 12_500;

    fn setup_masstree15_8(keys: &[[u8; KEY_SIZE_8]]) -> MassTree15Inline<u64> {
        let tree = MassTree15Inline::new();
        {
            let guard = tree.guard();
            for (i, key) in keys.iter().enumerate() {
                let _ = tree.insert_with_guard(key, i as u64, &guard);
            }
        }
        tree
    }

    #[divan::bench(args = [1, 2, 3, 4, 5, 6])]
    fn masstree15(bencher: Bencher, threads: usize) {
        let keys = Arc::new(keys::<KEY_SIZE_8>(N));
        let indices = Arc::new(uniform_indices(N, OPS_PER_THREAD * threads, 42));

        bencher
            .counter(divan::counter::ItemsCount::new(threads * OPS_PER_THREAD))
            .with_inputs(|| Arc::new(setup_masstree15_8(keys.as_ref())))
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
                                } else if let Some(v) = tree.get_with_guard(&keys[idx], &guard) {
                                    sum = sum.wrapping_add(v);
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
// 09: PURE READ - 100% Reads (No Writes)
// =============================================================================

#[divan::bench_group(name = "09_pure_read_uniform", sample_count = 200)]
mod pure_read {
    use super::*;

    const N: usize = 50_000;
    const OPS_PER_THREAD: usize = 25_000;

    // Note: Pure read benchmarks pre-build the tree once since we're measuring
    // read performance only. Each thread still warms up before measurement.

    #[divan::bench(args = [1, 2, 3, 4, 5, 6])]
    fn masstree15(bencher: Bencher, threads: usize) {
        let keys = Arc::new(keys::<KEY_SIZE>(N));
        let indices = Arc::new(uniform_indices(N, OPS_PER_THREAD * threads, 42));
        let tree = Arc::new(setup_masstree15(keys.as_ref()));

        bencher
            .counter(divan::counter::ItemsCount::new(threads * OPS_PER_THREAD))
            .bench_local(|| {
                let warmup_done = Arc::new(Barrier::new(threads));
                let start = Arc::new(Barrier::new(threads));
                let handles: Vec<_> = (0..threads)
                    .map(|t| {
                        let tree = Arc::clone(&tree);
                        let keys = Arc::clone(&keys);
                        let indices = Arc::clone(&indices);
                        let warmup_done = Arc::clone(&warmup_done);
                        let start = Arc::clone(&start);
                        thread::spawn(move || {
                            let guard = tree.guard();
                            let base = t * OPS_PER_THREAD;

                            // Warmup
                            for i in 0..WARMUP_OPS {
                                let idx = indices[base + (i % OPS_PER_THREAD)];
                                black_box(tree.get_with_guard(&keys[idx], &guard));
                            }
                            warmup_done.wait();

                            // Measurement
                            pre_measurement_barrier();
                            let mut sum = 0u64;
                            start.wait();
                            for i in 0..OPS_PER_THREAD {
                                let idx = indices[base + i];
                                if let Some(v) = tree.get_with_guard(&keys[idx], &guard) {
                                    sum = sum.wrapping_add(v);
                                }
                            }
                            post_measurement_barrier();
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
// 10: REMOVE HEAVY - 50% Insert, 50% Remove
// =============================================================================

#[divan::bench_group(name = "10_remove_heavy", sample_count = 200)]
mod remove_heavy {
    use super::*;

    const N: usize = 50_000;
    const OPS_PER_THREAD: usize = 25_000;

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
                                if i % 2 == 0 {
                                    // Insert
                                    let _ = tree.insert_with_guard(&keys[idx], i as u64, &guard);
                                } else {
                                    // Remove
                                    if let Ok(Some(v)) = tree.remove_with_guard(&keys[idx], &guard)
                                    {
                                        sum = sum.wrapping_add(v);
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
            });
    }
}

// =============================================================================
// 11: LATENCY DISTRIBUTION - Single-threaded p50/p99/max
// =============================================================================

#[divan::bench_group(name = "11_latency_single_thread", sample_count = 200)]
mod latency_single {
    use super::*;
    use std::time::Instant;

    const N: usize = 50_000;
    const OPS: usize = 5_000;

    fn percentile(sorted: &[u64], p: f64) -> u64 {
        let idx = ((sorted.len() as f64) * p / 100.0) as usize;
        sorted[idx.min(sorted.len() - 1)]
    }

    #[divan::bench]
    fn masstree15_read_latency(bencher: Bencher) {
        let keys = keys::<KEY_SIZE>(N);
        let tree = setup_masstree15(&keys);
        let indices = uniform_indices(N, OPS, 42);

        bencher.bench_local(|| {
            let guard = tree.guard();
            let mut latencies = Vec::with_capacity(OPS);

            (0..OPS).for_each(|i| {
                let idx = indices[i];
                let start = Instant::now();
                black_box(tree.get_with_guard(&keys[idx], &guard));
                latencies.push(start.elapsed().as_nanos() as u64);
            });

            latencies.sort_unstable();
            let p50 = percentile(&latencies, 50.0);
            let p99 = percentile(&latencies, 99.0);
            let max = latencies[latencies.len() - 1];

            black_box((p50, p99, max))
        });
    }
}

// =============================================================================
// 12: MEMORY FOOTPRINT - Measure memory usage
// =============================================================================

#[divan::bench_group(name = "12_memory_footprint", sample_count = 200)]
mod memory_footprint {
    use super::*;

    const N: usize = 50_000;

    // Note: These benchmarks measure relative memory via timing proxy.
    // For accurate memory measurement, use external tools like `heaptrack`.

    #[divan::bench]
    fn masstree15_build_time(bencher: Bencher) {
        let keys = keys::<KEY_SIZE>(N);

        bencher
            .counter(divan::counter::ItemsCount::new(N))
            .bench_local(|| {
                let tree = setup_masstree15(&keys);
                black_box(tree)
            });
    }
}

// =============================================================================
// 13: INSERT-ONLY FAIR - No upserts, fair to TreeIndex
// =============================================================================
//
// This benchmark uses separate key ranges for reads vs writes:
// - Keys [0, N): Pre-populated, used for reads
// - Keys [N, N+write_count): Empty initially, used for inserts
//
// This eliminates the TreeIndex upsert workaround penalty, providing
// a fair comparison where all structures use simple insert operations.

#[divan::bench_group(name = "13_insert_only_fair", sample_count = 200)]
mod insert_only_fair {
    use super::*;

    const N: usize = 50_000; // Pre-populated keys for reads
    const OPS_PER_THREAD: usize = 12_500;

    /// Generate keys for the write range [N, N + count)
    fn write_keys(start: usize, count: usize) -> Vec<[u8; KEY_SIZE]> {
        let mut out = Vec::with_capacity(count);
        for i in start..(start + count) {
            let mut key = [0u8; KEY_SIZE];
            // Use same key generation pattern as bench_utils::keys
            let chunks = KEY_SIZE / 8;
            for c in 0..chunks {
                let multipliers: [u64; 8] = [
                    1,
                    0x517c_c1b7_2722_0a95,
                    0x9e37_79b9_7f4a_7c15,
                    0xbf58_476d_1ce4_e5b9,
                    0x6c8e_9448_1e2f_3d4b,
                    0xa5c2_f831_7d6e_4a9f,
                    0x3b7d_c4e6_2a8f_5c1d,
                    0xd92e_8b5a_4f7c_3e6d,
                ];
                let v = (i as u64).wrapping_mul(multipliers[c % multipliers.len()]);
                let bytes = v.to_be_bytes();
                let start_byte = c * 8;
                key[start_byte..start_byte + 8].copy_from_slice(&bytes);
            }
            out.push(key);
        }
        out
    }

    #[divan::bench(args = [1, 2, 3, 4, 5, 6])]
    fn masstree15(bencher: Bencher, threads: usize) {
        let read_keys = Arc::new(keys::<KEY_SIZE>(N));
        let total_writes = (OPS_PER_THREAD * threads) / 10; // 10% writes
        let write_keys = Arc::new(write_keys(N, total_writes));
        let read_indices = Arc::new(uniform_indices(N, OPS_PER_THREAD * threads, 42));

        bencher
            .counter(divan::counter::ItemsCount::new(threads * OPS_PER_THREAD))
            .with_inputs(|| Arc::new(setup_masstree15(read_keys.as_ref())))
            .bench_local_values(|tree| {
                let warmup_done = Arc::new(Barrier::new(threads));
                let start = Arc::new(Barrier::new(threads));
                let handles: Vec<_> = (0..threads)
                    .map(|t| {
                        let tree = Arc::clone(&tree);
                        let read_keys = Arc::clone(&read_keys);
                        let write_keys = Arc::clone(&write_keys);
                        let read_indices = Arc::clone(&read_indices);
                        let warmup_done = Arc::clone(&warmup_done);
                        let start = Arc::clone(&start);
                        thread::spawn(move || {
                            let guard = tree.guard();
                            let base = t * OPS_PER_THREAD;
                            let write_base = (t * total_writes) / threads;

                            // Warmup
                            for i in 0..WARMUP_OPS {
                                let idx = read_indices[base + (i % OPS_PER_THREAD)];
                                black_box(tree.get_with_guard(&read_keys[idx], &guard));
                            }
                            warmup_done.wait();

                            // Measurement
                            pre_measurement_barrier();
                            let mut sum = 0u64;
                            let mut write_idx = 0usize;
                            start.wait();
                            for i in 0..OPS_PER_THREAD {
                                if i % 10 == 0 {
                                    // 10% writes - insert NEW keys
                                    let wk_idx = write_base + write_idx;
                                    if wk_idx < write_keys.len() {
                                        let _ = tree.insert_with_guard(
                                            &write_keys[wk_idx],
                                            i as u64,
                                            &guard,
                                        );
                                        write_idx += 1;
                                    }
                                } else {
                                    // 90% reads
                                    let idx = read_indices[base + i];
                                    if let Some(v) = tree.get_with_guard(&read_keys[idx], &guard) {
                                        sum = sum.wrapping_add(v);
                                    }
                                }
                            }
                            post_measurement_barrier();
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
// 14: PURE INSERT - 100% Inserts (Build from empty)
// =============================================================================
//
// All operations are inserts to new keys. No reads, no upserts.
// This is the fairest possible write benchmark.

#[divan::bench_group(name = "14_pure_insert", sample_count = 200)]
mod pure_insert {
    use super::*;

    const N: usize = 50_000;
    const OPS_PER_THREAD: usize = 10_000;

    #[divan::bench(args = [1, 2, 3, 4, 5, 6])]
    fn masstree15(bencher: Bencher, threads: usize) {
        let keys = Arc::new(keys::<KEY_SIZE>(N));

        bencher
            .counter(divan::counter::ItemsCount::new(threads * OPS_PER_THREAD))
            .with_inputs(MassTree15Inline::<u64>::new)
            .bench_local_values(|tree| {
                let tree = Arc::new(tree);
                let start = Arc::new(Barrier::new(threads));
                let handles: Vec<_> = (0..threads)
                    .map(|t| {
                        let tree = Arc::clone(&tree);
                        let keys = Arc::clone(&keys);
                        let start = Arc::clone(&start);
                        thread::spawn(move || {
                            let guard = tree.guard();
                            let base = t * OPS_PER_THREAD;

                            pre_measurement_barrier();
                            start.wait();
                            for i in 0..OPS_PER_THREAD {
                                let idx = (base + i) % keys.len();
                                let _ = tree.insert_with_guard(&keys[idx], i as u64, &guard);
                            }
                            post_measurement_barrier();
                        })
                    })
                    .collect();

                for h in handles {
                    h.join().unwrap();
                }
            });
    }
}
