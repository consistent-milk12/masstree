//! Concurrent read/write benchmarks with 64-byte keys.
//!
//! Compares MassTree15Inline against other concurrent ordered maps under mixed
//! read-write workloads. All benchmarks use 64-byte keys to test multi-layer
//! traversal performance.
//!
//! ## Key Characteristics
//!
//! - Various read/write ratios (90-10, 50-50, pure read, pure write)
//! - 64-byte keys (8 chunks, tests suffix handling)
//! - Various access patterns: uniform, zipfian, shared prefix
//!
//! ## Methodology
//!
//! Each benchmark follows a rigorous methodology:
//! 1. **Explicit warmup**: Every benchmark warms up caches before measurement
//! 2. **Memory barriers**: Inserted before/after measurement to prevent reordering
//! 3. **Fresh state**: All benchmarks use `.with_inputs()` for fresh tree per sample
//! 4. **Randomized writes**: Write decisions use pre-shuffled arrays, not modulo
//! 5. **Consistent threads**: All benchmarks use [1, 2, 4, 6, 8, 12] thread counts
//! 6. **100 samples**: For statistical significance
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
use divan::{black_box, Bencher};
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

/// Warmup iterations per thread before measurement
const WARMUP_OPS: usize = 500;

/// Standard thread counts for all benchmarks
const THREAD_COUNTS: [usize; 6] = [1, 2, 4, 6, 8, 12];

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

/// Generate a shuffled array of operation types (true = write, false = read).
/// This avoids the predictable `i % 100 < ratio` pattern which causes all threads
/// to write simultaneously and creates unrealistic branch prediction behavior.
fn shuffled_write_decisions(count: usize, write_ratio_percent: usize, seed: u64) -> Vec<bool> {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};

    let write_count = (count * write_ratio_percent) / 100;
    let mut decisions = vec![false; count];

    // Mark first `write_count` as writes
    for d in decisions.iter_mut().take(write_count) {
        *d = true;
    }

    // Fisher-Yates shuffle with seeded PRNG
    let mut hasher = DefaultHasher::new();
    seed.hash(&mut hasher);
    let mut rng_state = hasher.finish();

    for i in (1..count).rev() {
        // Simple xorshift64 PRNG
        rng_state ^= rng_state << 13;
        rng_state ^= rng_state >> 7;
        rng_state ^= rng_state << 17;

        let j = (rng_state as usize) % (i + 1);
        decisions.swap(i, j);
    }

    decisions
}

// =============================================================================
// 01: MIXED 90-10 - Uniform Access Pattern
// =============================================================================

#[divan::bench_group(name = "01_mixed_90_10_uniform", sample_count = 100)]
mod mixed_uniform {
    use super::*;

    const N: usize = 50_000;
    const OPS_PER_THREAD: usize = 12_500;
    const WRITE_RATIO: usize = 10;

    #[divan::bench(args = THREAD_COUNTS)]
    fn masstree15(bencher: Bencher, threads: usize) {
        let keys = Arc::new(keys::<KEY_SIZE>(N));
        let indices = Arc::new(uniform_indices(N, OPS_PER_THREAD * threads, 42));
        // Pre-generate randomized write decisions per thread
        let write_decisions: Arc<Vec<Vec<bool>>> = Arc::new(
            (0..threads)
                .map(|t| shuffled_write_decisions(OPS_PER_THREAD, WRITE_RATIO, 42 + t as u64))
                .collect(),
        );

        bencher
            .counter(divan::counter::ItemsCount::new(threads * OPS_PER_THREAD))
            .with_inputs(|| Arc::new(setup_masstree15(keys.as_ref())))
            .bench_local_values(|tree| {
                let warmup_done = Arc::new(Barrier::new(threads));
                let start = Arc::new(Barrier::new(threads));
                let handles: Vec<_> = (0..threads)
                    .map(|t| {
                        let tree = Arc::clone(&tree);
                        let keys = Arc::clone(&keys);
                        let indices = Arc::clone(&indices);
                        let write_decisions = Arc::clone(&write_decisions);
                        let warmup_done = Arc::clone(&warmup_done);
                        let start = Arc::clone(&start);

                        thread::spawn(move || {
                            let guard = tree.guard();
                            let base = t * OPS_PER_THREAD;
                            let is_write = &write_decisions[t];

                            // === WARMUP PHASE ===
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
                                if is_write[i] {
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

#[divan::bench_group(name = "02_mixed_90_10_zipfian", sample_count = 100)]
mod mixed_zipfian {
    use super::*;

    const N: usize = 50_000;
    const OPS_PER_THREAD: usize = 12_500;
    const WRITE_RATIO: usize = 10;

    #[divan::bench(args = THREAD_COUNTS)]
    fn masstree15(bencher: Bencher, threads: usize) {
        let keys = Arc::new(keys::<KEY_SIZE>(N));
        let indices = Arc::new(zipfian_indices(N, OPS_PER_THREAD * threads, 42));
        let write_decisions: Arc<Vec<Vec<bool>>> = Arc::new(
            (0..threads)
                .map(|t| shuffled_write_decisions(OPS_PER_THREAD, WRITE_RATIO, 42 + t as u64))
                .collect(),
        );

        bencher
            .counter(divan::counter::ItemsCount::new(threads * OPS_PER_THREAD))
            .with_inputs(|| Arc::new(setup_masstree15(keys.as_ref())))
            .bench_local_values(|tree| {
                let warmup_done = Arc::new(Barrier::new(threads));
                let start = Arc::new(Barrier::new(threads));
                let handles: Vec<_> = (0..threads)
                    .map(|t| {
                        let tree = Arc::clone(&tree);
                        let keys = Arc::clone(&keys);
                        let indices = Arc::clone(&indices);
                        let write_decisions = Arc::clone(&write_decisions);
                        let warmup_done = Arc::clone(&warmup_done);
                        let start = Arc::clone(&start);

                        thread::spawn(move || {
                            let guard = tree.guard();
                            let base = t * OPS_PER_THREAD;
                            let is_write = &write_decisions[t];

                            // === WARMUP PHASE ===
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
                                if is_write[i] {
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
// 03: MIXED 90-10 - Shared Prefix (Masstree Stress Test)
// =============================================================================

#[divan::bench_group(name = "03_mixed_90_10_shared_prefix", sample_count = 100)]
mod mixed_shared_prefix {
    use super::*;

    const N: usize = 50_000;
    const OPS_PER_THREAD: usize = 12_500;
    const PREFIX_CHUNKS: usize = 3; // First 24 bytes shared
    const PREFIX_BUCKETS: u64 = 256;
    const WRITE_RATIO: usize = 10;

    fn prefix_keys() -> Vec<[u8; KEY_SIZE]> {
        keys_shared_prefix_chunks::<KEY_SIZE>(N, PREFIX_CHUNKS, PREFIX_BUCKETS)
    }

    #[divan::bench(args = THREAD_COUNTS)]
    fn masstree15(bencher: Bencher, threads: usize) {
        let keys = Arc::new(prefix_keys());
        let indices = Arc::new(uniform_indices(N, OPS_PER_THREAD * threads, 42));
        let write_decisions: Arc<Vec<Vec<bool>>> = Arc::new(
            (0..threads)
                .map(|t| shuffled_write_decisions(OPS_PER_THREAD, WRITE_RATIO, 42 + t as u64))
                .collect(),
        );

        bencher
            .counter(divan::counter::ItemsCount::new(threads * OPS_PER_THREAD))
            .with_inputs(|| Arc::new(setup_masstree15(keys.as_ref())))
            .bench_local_values(|tree| {
                let warmup_done = Arc::new(Barrier::new(threads));
                let start = Arc::new(Barrier::new(threads));
                let handles: Vec<_> = (0..threads)
                    .map(|t| {
                        let tree = Arc::clone(&tree);
                        let keys = Arc::clone(&keys);
                        let indices = Arc::clone(&indices);
                        let write_decisions = Arc::clone(&write_decisions);
                        let warmup_done = Arc::clone(&warmup_done);
                        let start = Arc::clone(&start);

                        thread::spawn(move || {
                            let guard = tree.guard();
                            let base = t * OPS_PER_THREAD;
                            let is_write = &write_decisions[t];

                            // === WARMUP PHASE ===
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
                                if is_write[i] {
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
// 04: HIGH CONTENTION - Small Key Space (500 keys)
// =============================================================================

#[divan::bench_group(name = "04_mixed_90_10_high_contention", sample_count = 100)]
mod mixed_high_contention {
    use super::*;

    const N: usize = 500; // Small key space = high contention
    const OPS_PER_THREAD: usize = 25_000;
    const WRITE_RATIO: usize = 10;

    #[divan::bench(args = THREAD_COUNTS)]
    fn masstree15(bencher: Bencher, threads: usize) {
        let keys = Arc::new(keys::<KEY_SIZE>(N));
        let indices = Arc::new(uniform_indices(N, OPS_PER_THREAD * threads, 42));
        let write_decisions: Arc<Vec<Vec<bool>>> = Arc::new(
            (0..threads)
                .map(|t| shuffled_write_decisions(OPS_PER_THREAD, WRITE_RATIO, 42 + t as u64))
                .collect(),
        );

        bencher
            .counter(divan::counter::ItemsCount::new(threads * OPS_PER_THREAD))
            .with_inputs(|| Arc::new(setup_masstree15(keys.as_ref())))
            .bench_local_values(|tree| {
                let warmup_done = Arc::new(Barrier::new(threads));
                let start = Arc::new(Barrier::new(threads));
                let handles: Vec<_> = (0..threads)
                    .map(|t| {
                        let tree = Arc::clone(&tree);
                        let keys = Arc::clone(&keys);
                        let indices = Arc::clone(&indices);
                        let write_decisions = Arc::clone(&write_decisions);
                        let warmup_done = Arc::clone(&warmup_done);
                        let start = Arc::clone(&start);

                        thread::spawn(move || {
                            let guard = tree.guard();
                            let base = t * OPS_PER_THREAD;
                            let is_write = &write_decisions[t];

                            // === WARMUP PHASE ===
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
                                if is_write[i] {
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
// 05: LARGE DATASET - 500K keys (read-only to avoid state accumulation)
// =============================================================================
//
// Note: This benchmark uses a shared pre-built tree to avoid massive setup cost.
// To prevent state accumulation across samples, this is READ-ONLY.
// For mixed workloads on large datasets, see benchmark 01 with larger N.

#[divan::bench_group(name = "05_large_dataset_read_only", sample_count = 100)]
mod large_dataset_read_only {
    use super::*;

    const N: usize = 500_000;
    const OPS_PER_THREAD: usize = 25_000;

    #[divan::bench(args = THREAD_COUNTS)]
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

                            // === WARMUP PHASE ===
                            for i in 0..WARMUP_OPS {
                                let idx = indices[base + (i % OPS_PER_THREAD)];
                                black_box(tree.get_with_guard(&keys[idx], &guard));
                            }
                            warmup_done.wait();

                            // === MEASUREMENT PHASE (read-only) ===
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
// 06: SINGLE HOT KEY - Maximum Contention
// =============================================================================

#[divan::bench_group(name = "06_single_hot_key", sample_count = 100)]
mod single_hot_key {
    use super::*;

    const N: usize = 50_000;
    const OPS_PER_THREAD: usize = 5_000;
    const WRITE_RATIO: usize = 10;

    #[divan::bench(args = THREAD_COUNTS)]
    fn masstree15(bencher: Bencher, threads: usize) {
        let keys = Arc::new(keys::<KEY_SIZE>(N));
        let hot_key = keys[N / 2];
        let write_decisions: Arc<Vec<Vec<bool>>> = Arc::new(
            (0..threads)
                .map(|t| shuffled_write_decisions(OPS_PER_THREAD, WRITE_RATIO, 42 + t as u64))
                .collect(),
        );

        bencher
            .counter(divan::counter::ItemsCount::new(threads * OPS_PER_THREAD))
            .with_inputs(|| Arc::new(setup_masstree15(keys.as_ref())))
            .bench_local_values(|tree| {
                let warmup_done = Arc::new(Barrier::new(threads));
                let start = Arc::new(Barrier::new(threads));
                let handles: Vec<_> = (0..threads)
                    .map(|t| {
                        let tree = Arc::clone(&tree);
                        let hot_key = hot_key;
                        let write_decisions = Arc::clone(&write_decisions);
                        let warmup_done = Arc::clone(&warmup_done);
                        let start = Arc::clone(&start);

                        thread::spawn(move || {
                            let guard = tree.guard();
                            let is_write = &write_decisions[t];

                            // === WARMUP PHASE ===
                            for _ in 0..WARMUP_OPS {
                                black_box(tree.get_with_guard(&hot_key, &guard));
                            }
                            warmup_done.wait();

                            // === MEASUREMENT PHASE ===
                            pre_measurement_barrier();
                            let mut sum = 0u64;
                            start.wait();

                            for i in 0..OPS_PER_THREAD {
                                if is_write[i] {
                                    let _ = tree.insert_with_guard(
                                        &hot_key,
                                        (t * OPS_PER_THREAD + i) as u64,
                                        &guard,
                                    );
                                } else if let Some(v) = tree.get_with_guard(&hot_key, &guard) {
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
// 07: WRITE-HEAVY - 50% reads, 50% writes
// =============================================================================

#[divan::bench_group(name = "07_mixed_50_50", sample_count = 100)]
mod mixed_50_50 {
    use super::*;

    const N: usize = 50_000;
    const OPS_PER_THREAD: usize = 12_500;
    const WRITE_RATIO: usize = 50;

    #[divan::bench(args = THREAD_COUNTS)]
    fn masstree15(bencher: Bencher, threads: usize) {
        let keys = Arc::new(keys::<KEY_SIZE>(N));
        let indices = Arc::new(uniform_indices(N, OPS_PER_THREAD * threads, 42));
        let write_decisions: Arc<Vec<Vec<bool>>> = Arc::new(
            (0..threads)
                .map(|t| shuffled_write_decisions(OPS_PER_THREAD, WRITE_RATIO, 42 + t as u64))
                .collect(),
        );

        bencher
            .counter(divan::counter::ItemsCount::new(threads * OPS_PER_THREAD))
            .with_inputs(|| Arc::new(setup_masstree15(keys.as_ref())))
            .bench_local_values(|tree| {
                let warmup_done = Arc::new(Barrier::new(threads));
                let start = Arc::new(Barrier::new(threads));
                let handles: Vec<_> = (0..threads)
                    .map(|t| {
                        let tree = Arc::clone(&tree);
                        let keys = Arc::clone(&keys);
                        let indices = Arc::clone(&indices);
                        let write_decisions = Arc::clone(&write_decisions);
                        let warmup_done = Arc::clone(&warmup_done);
                        let start = Arc::clone(&start);

                        thread::spawn(move || {
                            let guard = tree.guard();
                            let base = t * OPS_PER_THREAD;
                            let is_write = &write_decisions[t];

                            // === WARMUP PHASE ===
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
                                if is_write[i] {
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
// 08: 8-BYTE KEYS - MassTree Single-Layer Fast Path
// =============================================================================

#[divan::bench_group(name = "08_8byte_keys_uniform", sample_count = 100)]
mod keys_8byte {
    use super::*;

    const KEY_SIZE_8: usize = 8;
    const N: usize = 50_000;
    const OPS_PER_THREAD: usize = 12_500;
    const WRITE_RATIO: usize = 10;

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

    #[divan::bench(args = THREAD_COUNTS)]
    fn masstree15(bencher: Bencher, threads: usize) {
        let keys = Arc::new(keys::<KEY_SIZE_8>(N));
        let indices = Arc::new(uniform_indices(N, OPS_PER_THREAD * threads, 42));
        let write_decisions: Arc<Vec<Vec<bool>>> = Arc::new(
            (0..threads)
                .map(|t| shuffled_write_decisions(OPS_PER_THREAD, WRITE_RATIO, 42 + t as u64))
                .collect(),
        );

        bencher
            .counter(divan::counter::ItemsCount::new(threads * OPS_PER_THREAD))
            .with_inputs(|| Arc::new(setup_masstree15_8(keys.as_ref())))
            .bench_local_values(|tree| {
                let warmup_done = Arc::new(Barrier::new(threads));
                let start = Arc::new(Barrier::new(threads));
                let handles: Vec<_> = (0..threads)
                    .map(|t| {
                        let tree = Arc::clone(&tree);
                        let keys = Arc::clone(&keys);
                        let indices = Arc::clone(&indices);
                        let write_decisions = Arc::clone(&write_decisions);
                        let warmup_done = Arc::clone(&warmup_done);
                        let start = Arc::clone(&start);

                        thread::spawn(move || {
                            let guard = tree.guard();
                            let base = t * OPS_PER_THREAD;
                            let is_write = &write_decisions[t];

                            // === WARMUP PHASE ===
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
                                if is_write[i] {
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
// 09: PURE READ - 100% Reads (No Writes)
// =============================================================================
//
// Note: Pure read benchmarks pre-build the tree once since we're measuring
// read performance only and no state changes occur.

#[divan::bench_group(name = "09_pure_read_uniform", sample_count = 100)]
mod pure_read {
    use super::*;

    const N: usize = 50_000;
    const OPS_PER_THREAD: usize = 25_000;

    #[divan::bench(args = THREAD_COUNTS)]
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

                            // === WARMUP PHASE ===
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
//
// Note: This benchmark measures insert/remove throughput. Due to the random
// access pattern, some removes will fail (key not present) and some inserts
// will be upserts. This measures realistic churn behavior.

#[divan::bench_group(name = "10_remove_heavy", sample_count = 100)]
mod remove_heavy {
    use super::*;

    const N: usize = 50_000;
    const OPS_PER_THREAD: usize = 25_000;

    #[divan::bench(args = THREAD_COUNTS)]
    fn masstree15(bencher: Bencher, threads: usize) {
        let keys = Arc::new(keys::<KEY_SIZE>(N));
        let indices = Arc::new(uniform_indices(N, OPS_PER_THREAD * threads, 42));
        // Pre-generate insert/remove decisions (50/50 shuffled)
        let op_decisions: Arc<Vec<Vec<bool>>> = Arc::new(
            (0..threads)
                .map(|t| shuffled_write_decisions(OPS_PER_THREAD, 50, 42 + t as u64))
                .collect(),
        );

        bencher
            .counter(divan::counter::ItemsCount::new(threads * OPS_PER_THREAD))
            .with_inputs(|| Arc::new(setup_masstree15(keys.as_ref())))
            .bench_local_values(|tree| {
                let warmup_done = Arc::new(Barrier::new(threads));
                let start = Arc::new(Barrier::new(threads));
                let handles: Vec<_> = (0..threads)
                    .map(|t| {
                        let tree = Arc::clone(&tree);
                        let keys = Arc::clone(&keys);
                        let indices = Arc::clone(&indices);
                        let op_decisions = Arc::clone(&op_decisions);
                        let warmup_done = Arc::clone(&warmup_done);
                        let start = Arc::clone(&start);

                        thread::spawn(move || {
                            let guard = tree.guard();
                            let base = t * OPS_PER_THREAD;
                            let is_insert = &op_decisions[t];

                            // === WARMUP PHASE ===
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
                                if is_insert[i] {
                                    let _ = tree.insert_with_guard(&keys[idx], i as u64, &guard);
                                } else if let Ok(Some(v)) =
                                    tree.remove_with_guard(&keys[idx], &guard)
                                {
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
// 11: LATENCY DISTRIBUTION - Single-threaded p50/p99/max
// =============================================================================
//
// Note: Measures latency distribution using batch timing to reduce Instant::now()
// overhead. Reports aggregate statistics rather than per-operation timing.

#[divan::bench_group(name = "11_latency_single_thread", sample_count = 100)]
mod latency_single {
    use super::*;
    use std::time::Instant;

    const N: usize = 50_000;
    const OPS: usize = 5_000;
    const BATCH_SIZE: usize = 100; // Measure batches to reduce timing overhead

    fn percentile(sorted: &[u64], p: f64) -> u64 {
        let idx = ((sorted.len() as f64) * p / 100.0) as usize;
        sorted[idx.min(sorted.len() - 1)]
    }

    #[divan::bench]
    fn masstree15_read_latency(bencher: Bencher) {
        let keys = keys::<KEY_SIZE>(N);
        let tree = setup_masstree15(&keys);
        let indices = uniform_indices(N, OPS, 42);

        // Pre-allocate latency vector to avoid allocation during measurement
        let num_batches = OPS / BATCH_SIZE;

        bencher.bench_local(|| {
            let guard = tree.guard();
            let mut batch_latencies: Vec<u64> = Vec::with_capacity(num_batches);

            // Warmup
            for i in 0..WARMUP_OPS {
                let idx = indices[i % OPS];
                black_box(tree.get_with_guard(&keys[idx], &guard));
            }

            pre_measurement_barrier();

            // Measure batches
            for batch in 0..num_batches {
                let batch_start = batch * BATCH_SIZE;
                let start = Instant::now();

                for i in 0..BATCH_SIZE {
                    let idx = indices[batch_start + i];
                    black_box(tree.get_with_guard(&keys[idx], &guard));
                }

                // Store average latency per operation in this batch (in nanoseconds)
                let batch_nanos = start.elapsed().as_nanos() as u64;
                batch_latencies.push(batch_nanos / BATCH_SIZE as u64);
            }

            post_measurement_barrier();

            batch_latencies.sort_unstable();
            let p50 = percentile(&batch_latencies, 50.0);
            let p99 = percentile(&batch_latencies, 99.0);
            let max = *batch_latencies.last().unwrap_or(&0);

            black_box((p50, p99, max))
        });
    }
}

// =============================================================================
// 12: BUILD TIME - Measure tree construction throughput
// =============================================================================
//
// Measures how fast the tree can be built from scratch.
// For memory footprint measurement, use external tools like `heaptrack`.

#[divan::bench_group(name = "12_build_time", sample_count = 100)]
mod build_time {
    use super::*;

    const N: usize = 50_000;

    #[divan::bench]
    fn masstree15(bencher: Bencher) {
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
// Each thread gets a non-overlapping range of write keys to ensure
// true inserts (no upserts) and even distribution.

#[divan::bench_group(name = "13_insert_only_fair", sample_count = 100)]
mod insert_only_fair {
    use super::*;

    const N: usize = 50_000; // Pre-populated keys for reads
    const OPS_PER_THREAD: usize = 12_500;
    const WRITE_RATIO: usize = 10; // 10% writes = 1,250 per thread

    /// Generate keys for the write range [start, start + count)
    fn write_keys(start: usize, count: usize) -> Vec<[u8; KEY_SIZE]> {
        let mut out = Vec::with_capacity(count);
        for i in start..(start + count) {
            let mut key = [0u8; KEY_SIZE];
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

    #[divan::bench(args = THREAD_COUNTS)]
    fn masstree15(bencher: Bencher, threads: usize) {
        let read_keys = Arc::new(keys::<KEY_SIZE>(N));
        // Each thread gets exactly (OPS_PER_THREAD * WRITE_RATIO / 100) write keys
        let writes_per_thread = (OPS_PER_THREAD * WRITE_RATIO) / 100;
        let total_writes = writes_per_thread * threads;
        let all_write_keys = Arc::new(write_keys(N, total_writes));
        let read_indices = Arc::new(uniform_indices(N, OPS_PER_THREAD * threads, 42));
        let write_decisions: Arc<Vec<Vec<bool>>> = Arc::new(
            (0..threads)
                .map(|t| shuffled_write_decisions(OPS_PER_THREAD, WRITE_RATIO, 42 + t as u64))
                .collect(),
        );

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
                        let all_write_keys = Arc::clone(&all_write_keys);
                        let read_indices = Arc::clone(&read_indices);
                        let write_decisions = Arc::clone(&write_decisions);
                        let warmup_done = Arc::clone(&warmup_done);
                        let start = Arc::clone(&start);

                        thread::spawn(move || {
                            let guard = tree.guard();
                            let base = t * OPS_PER_THREAD;
                            // Each thread's write keys are at [t * writes_per_thread, (t+1) * writes_per_thread)
                            let write_key_base = t * writes_per_thread;
                            let is_write = &write_decisions[t];

                            // === WARMUP PHASE ===
                            for i in 0..WARMUP_OPS {
                                let idx = read_indices[base + (i % OPS_PER_THREAD)];
                                black_box(tree.get_with_guard(&read_keys[idx], &guard));
                            }
                            warmup_done.wait();

                            // === MEASUREMENT PHASE ===
                            pre_measurement_barrier();
                            let mut sum = 0u64;
                            let mut write_idx = 0usize;
                            start.wait();

                            for i in 0..OPS_PER_THREAD {
                                if is_write[i] && write_idx < writes_per_thread {
                                    // Insert NEW key (no upsert)
                                    let wk_idx = write_key_base + write_idx;
                                    let _ = tree.insert_with_guard(
                                        &all_write_keys[wk_idx],
                                        i as u64,
                                        &guard,
                                    );
                                    write_idx += 1;
                                } else {
                                    // Read
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
// All operations are inserts to unique keys. No reads, no upserts.
// Each thread gets a non-overlapping key range to ensure true inserts.

#[divan::bench_group(name = "14_pure_insert", sample_count = 100)]
mod pure_insert {
    use super::*;

    // Ensure enough unique keys for all threads: 12 threads * 10,000 = 120,000
    const N: usize = 150_000;
    const OPS_PER_THREAD: usize = 10_000;

    #[divan::bench(args = THREAD_COUNTS)]
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
                            // Each thread gets non-overlapping key range
                            let base = t * OPS_PER_THREAD;

                            pre_measurement_barrier();
                            start.wait();

                            for i in 0..OPS_PER_THREAD {
                                // Use base + i directly (no modulo) for unique keys
                                let idx = base + i;
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
