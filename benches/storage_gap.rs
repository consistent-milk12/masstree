//! Short-key benchmarks isolating value storage (BoxPolicy vs InlinePolicy).
//!
//! All benchmarks use 8-byte keys, which fit in a single ikey chunk. This
//! eliminates suffix storage entirely, so the only policy difference measured
//! is value representation: heap-allocated `Box<V>` vs inline `AtomicU64`.
//!
//! Run both back-to-back to compare:
//!
//! ```bash
//! cargo bench --bench storage_gap --features mimalloc
//! ```
//!
//! The file contains paired benchmarks: `box_policy` (MassTree15) and
//! `inline_policy` (MassTree15Inline) within each group, so divan reports
//! them side-by-side.

#![expect(clippy::unwrap_used)]
#![expect(clippy::pedantic)]
#![expect(clippy::indexing_slicing)]
#![allow(dead_code)]

mod bench_utils;

use bench_utils::{keys, post_measurement_barrier, pre_measurement_barrier, uniform_indices};
use divan::{Bencher, black_box};
use masstree::{MassTree15, MassTree15Inline};
use std::sync::Arc;
use std::sync::Barrier;
use std::thread;

fn main() {
    divan::main();
}

// =============================================================================
// Constants
// =============================================================================

const SHORT_KEY_SIZE: usize = 8;
const WARMUP_OPS: usize = 500;
const THREAD_COUNTS: [usize; 6] = [1, 2, 4, 6, 8, 12];

// =============================================================================
// Setup helpers
// =============================================================================

fn setup_box(keys: &[[u8; SHORT_KEY_SIZE]]) -> MassTree15<u64> {
    let tree = MassTree15::new();
    {
        let guard = tree.guard();
        for (i, key) in keys.iter().enumerate() {
            let _ = tree.insert_with_guard(key, i as u64, &guard);
        }
    }
    tree
}

fn setup_inline(keys: &[[u8; SHORT_KEY_SIZE]]) -> MassTree15Inline<u64> {
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
fn shuffled_write_decisions(count: usize, write_ratio_percent: usize, seed: u64) -> Vec<bool> {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};

    let write_count = (count * write_ratio_percent) / 100;
    let mut decisions = vec![false; count];
    for d in decisions.iter_mut().take(write_count) {
        *d = true;
    }

    let mut hasher = DefaultHasher::new();
    seed.hash(&mut hasher);
    let mut rng_state = hasher.finish();

    for i in (1..count).rev() {
        rng_state ^= rng_state << 13;
        rng_state ^= rng_state >> 7;
        rng_state ^= rng_state << 17;
        let j = (rng_state as usize) % (i + 1);
        decisions.swap(i, j);
    }
    decisions
}

/// Generate indices where most accesses hit a small set of hot keys.
fn hot_key_indices(
    n: usize,
    count: usize,
    num_hot: usize,
    hot_prob: usize,
    seed: u64,
) -> Vec<usize> {
    let mut indices = Vec::with_capacity(count);
    let mut state = seed;
    let hot_keys: Vec<usize> = (0..num_hot)
        .map(|i| (n / num_hot) * i + n / (2 * num_hot))
        .collect();

    for _ in 0..count {
        state = state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1);
        let is_hot = ((state >> 32) as usize) % 100 < hot_prob;
        if is_hot {
            let hot_idx = ((state >> 48) as usize) % num_hot;
            indices.push(hot_keys[hot_idx]);
        } else {
            indices.push(((state >> 32) as usize) % n);
        }
    }
    indices
}

// =============================================================================
// 01: READ-ONLY (uniform access, 8-byte keys)
//
// Pure read throughput with no suffix overhead. Isolates value load cost:
// BoxPolicy dereferences a heap pointer, InlinePolicy reads an AtomicU64.
// =============================================================================

#[divan::bench_group(name = "01_read_only_short_keys", sample_count = 200)]
mod read_only {
    use super::*;

    const N: usize = 50_000;
    const OPS_PER_THREAD: usize = 25_000;

    #[divan::bench(args = THREAD_COUNTS)]
    fn box_policy(bencher: Bencher, threads: usize) {
        let keys = Arc::new(keys::<SHORT_KEY_SIZE>(N));
        let indices = Arc::new(uniform_indices(N, OPS_PER_THREAD * threads, 42));

        bencher
            .counter(divan::counter::ItemsCount::new(threads * OPS_PER_THREAD))
            .with_inputs(|| Arc::new(setup_box(&keys)))
            .bench_local_values(|tree| {
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

                            for i in 0..WARMUP_OPS {
                                black_box(tree.get_with_guard(
                                    &keys[indices[base + (i % OPS_PER_THREAD)]],
                                    &guard,
                                ));
                            }
                            warmup_done.wait();
                            start.wait();
                            pre_measurement_barrier();
                            let mut sum = 0u64;
                            for i in 0..OPS_PER_THREAD {
                                let idx = indices[base + i];
                                if let Some(v) = tree.get_with_guard(&keys[idx], &guard) {
                                    sum = sum.wrapping_add(*v);
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

    #[divan::bench(args = THREAD_COUNTS)]
    fn inline_policy(bencher: Bencher, threads: usize) {
        let keys = Arc::new(keys::<SHORT_KEY_SIZE>(N));
        let indices = Arc::new(uniform_indices(N, OPS_PER_THREAD * threads, 42));

        bencher
            .counter(divan::counter::ItemsCount::new(threads * OPS_PER_THREAD))
            .with_inputs(|| Arc::new(setup_inline(&keys)))
            .bench_local_values(|tree| {
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

                            for i in 0..WARMUP_OPS {
                                black_box(tree.get_with_guard(
                                    &keys[indices[base + (i % OPS_PER_THREAD)]],
                                    &guard,
                                ));
                            }
                            warmup_done.wait();
                            start.wait();
                            pre_measurement_barrier();
                            let mut sum = 0u64;
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
// 02: HOT KEYS READ-ONLY (skewed access, 8-byte keys)
//
// 80% of reads hit 8 hot keys. Tests whether BoxPolicy's heap values stay
// in L1/L2 under high temporal locality. If the gap persists here, it
// points to per-access overhead (pointer chase latency, branch prediction)
// rather than cache miss costs.
// =============================================================================

#[divan::bench_group(name = "02_hot_keys_short", sample_count = 200)]
mod hot_keys {
    use super::*;

    const N: usize = 50_000;
    const OPS_PER_THREAD: usize = 25_000;
    const NUM_HOT_KEYS: usize = 8;
    const HOT_KEY_PROBABILITY: usize = 80;

    #[divan::bench(args = THREAD_COUNTS)]
    fn box_policy(bencher: Bencher, threads: usize) {
        let keys = Arc::new(keys::<SHORT_KEY_SIZE>(N));
        let indices = Arc::new(hot_key_indices(
            N,
            OPS_PER_THREAD * threads,
            NUM_HOT_KEYS,
            HOT_KEY_PROBABILITY,
            42,
        ));

        bencher
            .counter(divan::counter::ItemsCount::new(threads * OPS_PER_THREAD))
            .with_inputs(|| Arc::new(setup_box(&keys)))
            .bench_local_values(|tree| {
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

                            for i in 0..WARMUP_OPS {
                                black_box(tree.get_with_guard(
                                    &keys[indices[base + (i % OPS_PER_THREAD)]],
                                    &guard,
                                ));
                            }
                            warmup_done.wait();
                            start.wait();
                            pre_measurement_barrier();
                            let mut sum = 0u64;
                            for i in 0..OPS_PER_THREAD {
                                let idx = indices[base + i];
                                if let Some(v) = tree.get_with_guard(&keys[idx], &guard) {
                                    sum = sum.wrapping_add(*v);
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

    #[divan::bench(args = THREAD_COUNTS)]
    fn inline_policy(bencher: Bencher, threads: usize) {
        let keys = Arc::new(keys::<SHORT_KEY_SIZE>(N));
        let indices = Arc::new(hot_key_indices(
            N,
            OPS_PER_THREAD * threads,
            NUM_HOT_KEYS,
            HOT_KEY_PROBABILITY,
            42,
        ));

        bencher
            .counter(divan::counter::ItemsCount::new(threads * OPS_PER_THREAD))
            .with_inputs(|| Arc::new(setup_inline(&keys)))
            .bench_local_values(|tree| {
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

                            for i in 0..WARMUP_OPS {
                                black_box(tree.get_with_guard(
                                    &keys[indices[base + (i % OPS_PER_THREAD)]],
                                    &guard,
                                ));
                            }
                            warmup_done.wait();
                            start.wait();
                            pre_measurement_barrier();
                            let mut sum = 0u64;
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
// 03: MIXED GET/INSERT (8-byte keys, 10% writes)
//
// Isolates write-path costs: BoxPolicy allocates via Box::new and retires
// old values through EBR. InlinePolicy does an atomic u64 store with no
// allocation or retirement.
// =============================================================================

#[divan::bench_group(name = "03_mixed_rw_short_keys", sample_count = 200)]
mod mixed_rw {
    use super::*;

    const N: usize = 50_000;
    const OPS_PER_THREAD: usize = 12_500;
    const WRITE_RATIO: usize = 10;

    #[divan::bench(args = THREAD_COUNTS)]
    fn box_policy(bencher: Bencher, threads: usize) {
        let keys = Arc::new(keys::<SHORT_KEY_SIZE>(N));
        let indices = Arc::new(uniform_indices(N, OPS_PER_THREAD * threads, 42));
        let write_decisions: Arc<Vec<Vec<bool>>> = Arc::new(
            (0..threads)
                .map(|t| shuffled_write_decisions(OPS_PER_THREAD, WRITE_RATIO, 42 + t as u64))
                .collect(),
        );

        bencher
            .counter(divan::counter::ItemsCount::new(threads * OPS_PER_THREAD))
            .with_inputs(|| Arc::new(setup_box(&keys)))
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

                            for i in 0..WARMUP_OPS {
                                black_box(tree.get_with_guard(
                                    &keys[indices[base + (i % OPS_PER_THREAD)]],
                                    &guard,
                                ));
                            }
                            warmup_done.wait();
                            start.wait();
                            pre_measurement_barrier();
                            let mut sum = 0u64;
                            for i in 0..OPS_PER_THREAD {
                                let idx = indices[base + i];
                                if is_write[i] {
                                    black_box(tree.insert_with_guard(&keys[idx], i as u64, &guard));
                                } else if let Some(v) = tree.get_with_guard(&keys[idx], &guard) {
                                    sum = sum.wrapping_add(*v);
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

    #[divan::bench(args = THREAD_COUNTS)]
    fn inline_policy(bencher: Bencher, threads: usize) {
        let keys = Arc::new(keys::<SHORT_KEY_SIZE>(N));
        let indices = Arc::new(uniform_indices(N, OPS_PER_THREAD * threads, 42));
        let write_decisions: Arc<Vec<Vec<bool>>> = Arc::new(
            (0..threads)
                .map(|t| shuffled_write_decisions(OPS_PER_THREAD, WRITE_RATIO, 42 + t as u64))
                .collect(),
        );

        bencher
            .counter(divan::counter::ItemsCount::new(threads * OPS_PER_THREAD))
            .with_inputs(|| Arc::new(setup_inline(&keys)))
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

                            for i in 0..WARMUP_OPS {
                                black_box(tree.get_with_guard(
                                    &keys[indices[base + (i % OPS_PER_THREAD)]],
                                    &guard,
                                ));
                            }
                            warmup_done.wait();
                            start.wait();
                            pre_measurement_barrier();
                            let mut sum = 0u64;
                            for i in 0..OPS_PER_THREAD {
                                let idx = indices[base + i];
                                if is_write[i] {
                                    black_box(tree.insert_with_guard(&keys[idx], i as u64, &guard));
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
// 04: WRITE-HEAVY (8-byte keys, 50% writes)
//
// Amplifies allocation and retirement costs. BoxPolicy does a Box::new +
// defer_retire on every write. InlinePolicy stores a u64 atomically.
// =============================================================================

#[divan::bench_group(name = "04_write_heavy_short_keys", sample_count = 200)]
mod write_heavy {
    use super::*;

    const N: usize = 50_000;
    const OPS_PER_THREAD: usize = 12_500;
    const WRITE_RATIO: usize = 50;

    #[divan::bench(args = THREAD_COUNTS)]
    fn box_policy(bencher: Bencher, threads: usize) {
        let keys = Arc::new(keys::<SHORT_KEY_SIZE>(N));
        let indices = Arc::new(uniform_indices(N, OPS_PER_THREAD * threads, 42));
        let write_decisions: Arc<Vec<Vec<bool>>> = Arc::new(
            (0..threads)
                .map(|t| shuffled_write_decisions(OPS_PER_THREAD, WRITE_RATIO, 42 + t as u64))
                .collect(),
        );

        bencher
            .counter(divan::counter::ItemsCount::new(threads * OPS_PER_THREAD))
            .with_inputs(|| Arc::new(setup_box(&keys)))
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

                            for i in 0..WARMUP_OPS {
                                black_box(tree.get_with_guard(
                                    &keys[indices[base + (i % OPS_PER_THREAD)]],
                                    &guard,
                                ));
                            }
                            warmup_done.wait();
                            start.wait();
                            pre_measurement_barrier();
                            let mut sum = 0u64;
                            for i in 0..OPS_PER_THREAD {
                                let idx = indices[base + i];
                                if is_write[i] {
                                    black_box(tree.insert_with_guard(&keys[idx], i as u64, &guard));
                                } else if let Some(v) = tree.get_with_guard(&keys[idx], &guard) {
                                    sum = sum.wrapping_add(*v);
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

    #[divan::bench(args = THREAD_COUNTS)]
    fn inline_policy(bencher: Bencher, threads: usize) {
        let keys = Arc::new(keys::<SHORT_KEY_SIZE>(N));
        let indices = Arc::new(uniform_indices(N, OPS_PER_THREAD * threads, 42));
        let write_decisions: Arc<Vec<Vec<bool>>> = Arc::new(
            (0..threads)
                .map(|t| shuffled_write_decisions(OPS_PER_THREAD, WRITE_RATIO, 42 + t as u64))
                .collect(),
        );

        bencher
            .counter(divan::counter::ItemsCount::new(threads * OPS_PER_THREAD))
            .with_inputs(|| Arc::new(setup_inline(&keys)))
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

                            for i in 0..WARMUP_OPS {
                                black_box(tree.get_with_guard(
                                    &keys[indices[base + (i % OPS_PER_THREAD)]],
                                    &guard,
                                ));
                            }
                            warmup_done.wait();
                            start.wait();
                            pre_measurement_barrier();
                            let mut sum = 0u64;
                            for i in 0..OPS_PER_THREAD {
                                let idx = indices[base + i];
                                if is_write[i] {
                                    black_box(tree.insert_with_guard(&keys[idx], i as u64, &guard));
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
// 05: INSERT-ONLY (8-byte keys, fresh tree each iteration)
//
// Measures pure insertion throughput into an empty tree. Isolates Box::new
// allocation cost without retirement (no old values to retire on first
// insert).
// =============================================================================

#[divan::bench_group(name = "05_insert_only_short_keys", sample_count = 200)]
mod insert_only {
    use super::*;

    // N must be >= max threads * OPS_PER_THREAD to avoid OOB
    const N: usize = 72_000;
    const OPS_PER_THREAD: usize = 5_000;

    #[divan::bench(args = THREAD_COUNTS)]
    fn box_policy(bencher: Bencher, threads: usize) {
        let keys = Arc::new(keys::<SHORT_KEY_SIZE>(N));

        bencher
            .counter(divan::counter::ItemsCount::new(threads * OPS_PER_THREAD))
            .bench_local(|| {
                let tree = Arc::new(MassTree15::<u64>::new());
                let start = Arc::new(Barrier::new(threads));
                let handles: Vec<_> = (0..threads)
                    .map(|t| {
                        let tree = Arc::clone(&tree);
                        let keys = Arc::clone(&keys);
                        let start = Arc::clone(&start);
                        thread::spawn(move || {
                            let guard = tree.guard();
                            start.wait();
                            pre_measurement_barrier();
                            let base = t * OPS_PER_THREAD;
                            for i in 0..OPS_PER_THREAD {
                                let idx = base + i;
                                black_box(tree.insert_with_guard(&keys[idx], i as u64, &guard));
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

    #[divan::bench(args = THREAD_COUNTS)]
    fn inline_policy(bencher: Bencher, threads: usize) {
        let keys = Arc::new(keys::<SHORT_KEY_SIZE>(N));

        bencher
            .counter(divan::counter::ItemsCount::new(threads * OPS_PER_THREAD))
            .bench_local(|| {
                let tree = Arc::new(MassTree15Inline::<u64>::new());
                let start = Arc::new(Barrier::new(threads));
                let handles: Vec<_> = (0..threads)
                    .map(|t| {
                        let tree = Arc::clone(&tree);
                        let keys = Arc::clone(&keys);
                        let start = Arc::clone(&start);
                        thread::spawn(move || {
                            let guard = tree.guard();
                            start.wait();
                            pre_measurement_barrier();
                            let base = t * OPS_PER_THREAD;
                            for i in 0..OPS_PER_THREAD {
                                let idx = base + i;
                                black_box(tree.insert_with_guard(&keys[idx], i as u64, &guard));
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
