//! Concurrent benchmarks for MassTree15 (WIDTH=15) - Criterion version
//!
//! Equivalent to concurrent_masstree15.rs but using Criterion for more stable measurements.
//!
//! ## Running
//!
//! ```bash
//! cargo bench --bench concurrent_masstree15_crit
//! cargo bench --bench concurrent_masstree15_crit --features mimalloc
//! cargo bench --bench concurrent_masstree15_crit -- --save-baseline main
//! cargo bench --bench concurrent_masstree15_crit -- --baseline main
//! ```
//!
//! CSV output: target/criterion/<benchmark>/new/raw.csv

#![allow(clippy::unwrap_used)]
#![allow(clippy::pedantic)]
#![allow(clippy::indexing_slicing)]

mod bench_utils;
use std::hint::black_box;

use bench_utils::{keys, keys_shared_prefix, keys_shared_prefix_chunks, uniform_indices};
use criterion::{BenchmarkId, Criterion, SamplingMode, Throughput, criterion_group, criterion_main};
use masstree::MassTree15;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Barrier};
use std::thread;

// =============================================================================
// Setup Helpers
// =============================================================================

fn setup_masstree15<const K: usize>(keys: &[[u8; K]]) -> MassTree15<u64> {
    let tree = MassTree15::new();
    {
        let guard = tree.guard();
        for (i, key) in keys.iter().enumerate() {
            let _ = tree.insert_with_guard(key, i as u64, &guard);
        }
    }
    tree
}

fn setup_masstree15_string<const K: usize>(keys: &[[u8; K]]) -> MassTree15<String> {
    let tree = MassTree15::new();
    {
        let guard = tree.guard();
        for (i, key) in keys.iter().enumerate() {
            let _ = tree.insert_with_guard(key, generate_string_value(i), &guard);
        }
    }
    tree
}

fn generate_string_value(i: usize) -> String {
    format!("value_{i:016x}_padding_to_make_it_longer")
}

// =============================================================================
// 01: CONCURRENT WRITES - Disjoint Ranges
// =============================================================================

fn bench_01_concurrent_writes_disjoint(c: &mut Criterion) {
    let mut group = c.benchmark_group("01_concurrent_writes_disjoint");
    group.sampling_mode(SamplingMode::Flat);
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
                                let base = t * OPS_PER_THREAD;
                                start.wait();
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
                    black_box(tree)
                },
            );
        });
    }
    group.finish();
}

// =============================================================================
// 02: CONCURRENT WRITES - Contention
// =============================================================================

fn bench_02_concurrent_writes_contention(c: &mut Criterion) {
    let mut group = c.benchmark_group("02_concurrent_writes_contention");
    group.sampling_mode(SamplingMode::Flat);
    const OPS_PER_THREAD: usize = 10_000;
    const KEY_SPACE: usize = 1_000;

    let keys = Arc::new(keys::<8>(KEY_SPACE));

    for threads in [1, 2, 3, 4, 5, 6] {
        let keys = Arc::clone(&keys);
        group.bench_function(BenchmarkId::new("masstree15", threads), |b| {
            let keys = Arc::clone(&keys);
            b.iter_with_setup(
                || Arc::new(setup_masstree15::<8>(&keys)),
                |tree| {
                    let counter = Arc::new(AtomicUsize::new(0));
                    let start = Arc::new(Barrier::new(threads));
                    let handles: Vec<_> = (0..threads)
                        .map(|t| {
                            let tree = Arc::clone(&tree);
                            let keys = Arc::clone(&keys);
                            let counter = Arc::clone(&counter);
                            let start = Arc::clone(&start);
                            thread::spawn(move || {
                                let guard = tree.guard();
                                let mut state = (t as u64).wrapping_mul(0x517c_c1b7_2722_0a95);
                                start.wait();
                                for _ in 0..OPS_PER_THREAD {
                                    state = state
                                        .wrapping_mul(6_364_136_223_846_793_005)
                                        .wrapping_add(1);
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
                    black_box(tree)
                },
            );
        });
    }
    group.finish();
}

// =============================================================================
// 03: SINGLE-THREADED INSERT
// =============================================================================

fn bench_03_single_threaded_insert(c: &mut Criterion) {
    const KEY_COUNT: usize = 100_000;

    c.bench_function("03_single_threaded_insert/masstree15", |b| {
        b.iter(|| {
            let tree = MassTree15::<u64>::new();
            {
                let guard = tree.guard();
                for i in 0..KEY_COUNT {
                    let key = (i as u64).to_be_bytes();
                    let _ = tree.insert_with_guard(&key, i as u64, &guard);
                }
            }
            black_box(tree)
        });
    });
}

// =============================================================================
// 04: READ AFTER WRITE
// =============================================================================

fn bench_04_read_after_write(c: &mut Criterion) {
    let mut group = c.benchmark_group("04_read_after_write");
    group.sampling_mode(SamplingMode::Flat);
    const KEY_COUNT: usize = 50_000;

    fn local_setup() -> MassTree15<u64> {
        let tree = MassTree15::new();
        {
            let guard = tree.guard();
            for i in 0..KEY_COUNT {
                let key = (i as u64).to_be_bytes();
                let _ = tree.insert_with_guard(&key, i as u64, &guard);
            }
        }
        tree
    }

    for threads in [1, 2, 3, 4, 5, 6] {
        group.bench_function(BenchmarkId::new("masstree15", threads), |b| {
            b.iter_with_setup(
                || Arc::new(local_setup()),
                |tree| {
                    let start = Arc::new(Barrier::new(threads));
                    let handles: Vec<_> = (0..threads)
                        .map(|t| {
                            let tree = Arc::clone(&tree);
                            let start = Arc::clone(&start);
                            thread::spawn(move || {
                                let guard = tree.guard();
                                let ops = KEY_COUNT / threads;
                                let base = t * ops;
                                start.wait();
                                for i in 0..ops {
                                    let key = ((base + i) as u64).to_be_bytes();
                                    black_box(tree.get_ref(&key, &guard));
                                }
                            })
                        })
                        .collect();

                    for h in handles {
                        h.join().unwrap();
                    }
                    black_box(tree)
                },
            );
        });
    }
    group.finish();
}

// =============================================================================
// 05: GET BY KEY SIZE
// =============================================================================

fn bench_05_get_by_key_size(c: &mut Criterion) {
    let mut group = c.benchmark_group("05_get_by_key_size");
    group.sampling_mode(SamplingMode::Flat);
    const N: usize = 10_000;

    // 8B keys
    {
        let keys = keys::<8>(N);
        let tree = setup_masstree15::<8>(&keys);
        let lookup_keys = uniform_indices(N, 1000, 42);

        group.bench_function("masstree15_8B", |b| {
            b.iter(|| {
                let guard = tree.guard();
                let mut sum = 0u64;
                for &idx in &lookup_keys {
                    if let Some(v) = tree.get_ref(&keys[idx], &guard) {
                        sum += *v;
                    }
                }
                black_box(sum)
            });
        });
    }

    // 16B keys
    {
        let keys = keys::<16>(N);
        let tree = setup_masstree15::<16>(&keys);
        let lookup_keys = uniform_indices(N, 1000, 42);

        group.bench_function("masstree15_16B", |b| {
            b.iter(|| {
                let guard = tree.guard();
                let mut sum = 0u64;
                for &idx in &lookup_keys {
                    if let Some(v) = tree.get_ref(&keys[idx], &guard) {
                        sum += *v;
                    }
                }
                black_box(sum)
            });
        });
    }

    // 24B keys
    {
        let keys = keys::<24>(N);
        let tree = setup_masstree15::<24>(&keys);
        let lookup_keys = uniform_indices(N, 1000, 42);

        group.bench_function("masstree15_24B", |b| {
            b.iter(|| {
                let guard = tree.guard();
                let mut sum = 0u64;
                for &idx in &lookup_keys {
                    if let Some(v) = tree.get_ref(&keys[idx], &guard) {
                        sum += *v;
                    }
                }
                black_box(sum)
            });
        });
    }

    // 32B keys
    {
        let keys = keys::<32>(N);
        let tree = setup_masstree15::<32>(&keys);
        let lookup_keys = uniform_indices(N, 1000, 42);

        group.bench_function("masstree15_32B", |b| {
            b.iter(|| {
                let guard = tree.guard();
                let mut sum = 0u64;
                for &idx in &lookup_keys {
                    if let Some(v) = tree.get_ref(&keys[idx], &guard) {
                        sum += *v;
                    }
                }
                black_box(sum)
            });
        });
    }

    group.finish();
}

// =============================================================================
// 06: INSERT BY KEY SIZE
// =============================================================================

fn bench_06_insert_by_key_size(c: &mut Criterion) {
    let mut group = c.benchmark_group("06_insert_by_key_size");
    group.sampling_mode(SamplingMode::Flat);
    const N: usize = 1000;

    for (name, key_size) in [("8B", 8), ("16B", 16), ("24B", 24), ("32B", 32)] {
        match key_size {
            8 => {
                let keys = keys::<8>(N);
                group.bench_function(format!("masstree15_{name}"), |b| {
                    b.iter_with_setup(
                        || keys.clone(),
                        |keys| {
                            let tree = MassTree15::<u64>::new();
                            {
                                let guard = tree.guard();
                                for (i, key) in keys.iter().enumerate() {
                                    let _ = tree.insert_with_guard(key, i as u64, &guard);
                                }
                            }
                            black_box(tree)
                        },
                    );
                });
            }
            16 => {
                let keys = keys::<16>(N);
                group.bench_function(format!("masstree15_{name}"), |b| {
                    b.iter_with_setup(
                        || keys.clone(),
                        |keys| {
                            let tree = MassTree15::<u64>::new();
                            {
                                let guard = tree.guard();
                                for (i, key) in keys.iter().enumerate() {
                                    let _ = tree.insert_with_guard(key, i as u64, &guard);
                                }
                            }
                            black_box(tree)
                        },
                    );
                });
            }
            24 => {
                let keys = keys::<24>(N);
                group.bench_function(format!("masstree15_{name}"), |b| {
                    b.iter_with_setup(
                        || keys.clone(),
                        |keys| {
                            let tree = MassTree15::<u64>::new();
                            {
                                let guard = tree.guard();
                                for (i, key) in keys.iter().enumerate() {
                                    let _ = tree.insert_with_guard(key, i as u64, &guard);
                                }
                            }
                            black_box(tree)
                        },
                    );
                });
            }
            32 => {
                let keys = keys::<32>(N);
                group.bench_function(format!("masstree15_{name}"), |b| {
                    b.iter_with_setup(
                        || keys.clone(),
                        |keys| {
                            let tree = MassTree15::<u64>::new();
                            {
                                let guard = tree.guard();
                                for (i, key) in keys.iter().enumerate() {
                                    let _ = tree.insert_with_guard(key, i as u64, &guard);
                                }
                            }
                            black_box(tree)
                        },
                    );
                });
            }
            _ => unreachable!(),
        }
    }

    group.finish();
}

// =============================================================================
// 07: CONCURRENT READS SCALING
// =============================================================================

fn bench_07_concurrent_reads_scaling(c: &mut Criterion) {
    let mut group = c.benchmark_group("07_concurrent_reads_scaling");
    group.sampling_mode(SamplingMode::Flat);
    const N: usize = 10_000_000;
    const OPS_PER_THREAD: usize = 50_000;

    let keys = Arc::new(keys::<8>(N));
    let tree = Arc::new(setup_masstree15::<8>(&keys));
    let indices = Arc::new(uniform_indices(N, OPS_PER_THREAD, 42));

    for threads in [1, 2, 3, 4, 5, 6] {
        let keys = Arc::clone(&keys);
        let tree = Arc::clone(&tree);
        let indices = Arc::clone(&indices);

        group.bench_function(BenchmarkId::new("masstree15", threads), |b| {
            let keys = Arc::clone(&keys);
            let tree = Arc::clone(&tree);
            let indices = Arc::clone(&indices);

            b.iter(|| {
                let start_barrier = Arc::new(Barrier::new(threads));
                let handles: Vec<_> = (0..threads)
                    .map(|t| {
                        let tree = Arc::clone(&tree);
                        let keys = Arc::clone(&keys);
                        let indices = Arc::clone(&indices);
                        let start_barrier = Arc::clone(&start_barrier);
                        thread::spawn(move || {
                            let guard = tree.guard();
                            let mut sum = 0u64;
                            let offset = t * 7919;
                            start_barrier.wait();
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
        });
    }
    group.finish();
}

// =============================================================================
// 08: CONCURRENT READS LONG KEYS
// =============================================================================

fn bench_08_concurrent_reads_long_keys(c: &mut Criterion) {
    let mut group = c.benchmark_group("08_concurrent_reads_long_keys");
    group.sampling_mode(SamplingMode::Flat);
    const N: usize = 10_000_000;
    const OPS_PER_THREAD: usize = 50_000;

    let keys = Arc::new(keys::<32>(N));
    let tree = Arc::new(setup_masstree15::<32>(&keys));
    let indices = Arc::new(uniform_indices(N, OPS_PER_THREAD, 42));

    for threads in [1, 2, 3, 4, 5, 6] {
        let keys = Arc::clone(&keys);
        let tree = Arc::clone(&tree);
        let indices = Arc::clone(&indices);

        group.bench_function(BenchmarkId::new("masstree15_32b", threads), |b| {
            let keys = Arc::clone(&keys);
            let tree = Arc::clone(&tree);
            let indices = Arc::clone(&indices);

            b.iter(|| {
                let start_barrier = Arc::new(Barrier::new(threads));
                let handles: Vec<_> = (0..threads)
                    .map(|t| {
                        let tree = Arc::clone(&tree);
                        let keys = Arc::clone(&keys);
                        let indices = Arc::clone(&indices);
                        let start_barrier = Arc::clone(&start_barrier);
                        thread::spawn(move || {
                            let guard = tree.guard();
                            let mut sum = 0u64;
                            let offset = t * 7919;
                            start_barrier.wait();
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
        });
    }
    group.finish();
}

// =============================================================================
// 09: MIXED UNIFORM
// =============================================================================

fn bench_09_mixed_uniform(c: &mut Criterion) {
    let mut group = c.benchmark_group("09_mixed_uniform");
    group.sampling_mode(SamplingMode::Flat);
    const N: usize = 100_000;
    const OPS_PER_THREAD: usize = 10_000;
    const WRITE_RATIO: usize = 10;

    let keys = Arc::new(keys::<8>(N));
    let indices = Arc::new(uniform_indices(N, OPS_PER_THREAD, 42));

    for threads in [1, 2, 3, 4, 5, 6] {
        let keys = Arc::clone(&keys);
        let indices = Arc::clone(&indices);

        group.bench_function(BenchmarkId::new("masstree15", threads), |b| {
            let keys = Arc::clone(&keys);
            let indices = Arc::clone(&indices);

            b.iter_with_setup(
                || Arc::new(setup_masstree15::<8>(&keys)),
                |tree| {
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
                                let offset = t * 7919;
                                start.wait();
                                for i in 0..OPS_PER_THREAD {
                                    let idx = indices[(i + offset) % indices.len()];
                                    if i % WRITE_RATIO == 0 {
                                        let _ =
                                            tree.insert_with_guard(&keys[idx], i as u64, &guard);
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
                    black_box(tree)
                },
            );
        });
    }
    group.finish();
}

// =============================================================================
// 10a: READ SCALING 8B
// =============================================================================

fn bench_10a_read_scaling_8b(c: &mut Criterion) {
    let mut group = c.benchmark_group("10a_read_scaling_8B");
    group.sampling_mode(SamplingMode::Flat);
    const N: usize = 10_000_000;
    const OPS_PER_THREAD: usize = 50_000;

    let keys = Arc::new(keys::<8>(N));
    let tree = Arc::new(setup_masstree15::<8>(&keys));

    for threads in [1, 2, 3, 4, 5, 6] {
        let keys = Arc::clone(&keys);
        let tree = Arc::clone(&tree);

        group.throughput(Throughput::Elements((threads * OPS_PER_THREAD) as u64));
        group.bench_function(BenchmarkId::new("masstree15", threads), |b| {
            let keys = Arc::clone(&keys);
            let tree = Arc::clone(&tree);

            b.iter(|| {
                let start_barrier = Arc::new(Barrier::new(threads));
                let handles: Vec<_> = (0..threads)
                    .map(|t| {
                        let tree = Arc::clone(&tree);
                        let keys = Arc::clone(&keys);
                        let start_barrier = Arc::clone(&start_barrier);
                        thread::spawn(move || {
                            let guard = tree.guard();
                            let mut sum = 0u64;
                            let start = (t * 7919) % keys.len();
                            start_barrier.wait();
                            for i in 0..OPS_PER_THREAD {
                                let idx = (start + i) % keys.len();
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
        });
    }
    group.finish();
}

// =============================================================================
// 10b: READ SCALING 32B
// =============================================================================

fn bench_10b_read_scaling_32b(c: &mut Criterion) {
    let mut group = c.benchmark_group("10b_read_scaling_32B");
    group.sampling_mode(SamplingMode::Flat);
    const N: usize = 10_000_000;
    const OPS_PER_THREAD: usize = 50_000;

    let keys = Arc::new(keys::<32>(N));
    let tree = Arc::new(setup_masstree15::<32>(&keys));

    for threads in [1, 2, 3, 4, 5, 6] {
        let keys = Arc::clone(&keys);
        let tree = Arc::clone(&tree);

        group.throughput(Throughput::Elements((threads * OPS_PER_THREAD) as u64));
        group.bench_function(BenchmarkId::new("masstree15", threads), |b| {
            let keys = Arc::clone(&keys);
            let tree = Arc::clone(&tree);

            b.iter(|| {
                let start_barrier = Arc::new(Barrier::new(threads));
                let handles: Vec<_> = (0..threads)
                    .map(|t| {
                        let tree = Arc::clone(&tree);
                        let keys = Arc::clone(&keys);
                        let start_barrier = Arc::clone(&start_barrier);
                        thread::spawn(move || {
                            let guard = tree.guard();
                            let mut sum = 0u64;
                            let start = (t * 7919) % keys.len();
                            start_barrier.wait();
                            for i in 0..OPS_PER_THREAD {
                                let idx = (start + i) % keys.len();
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
        });
    }
    group.finish();
}

// =============================================================================
// 10c: WRITE SCALING 32B
// =============================================================================

fn bench_10c_write_scaling_32b(c: &mut Criterion) {
    let mut group = c.benchmark_group("10c_write_scaling_32B");
    group.sampling_mode(SamplingMode::Flat);
    const N: usize = 100_000;
    const OPS_PER_THREAD: usize = 10_000;

    let keys = Arc::new(keys::<32>(N));

    for threads in [1, 2, 3, 4, 5, 6] {
        let keys = Arc::clone(&keys);

        group.throughput(Throughput::Elements((threads * OPS_PER_THREAD) as u64));
        group.bench_function(BenchmarkId::new("masstree15", threads), |b| {
            let keys = Arc::clone(&keys);

            b.iter_with_setup(
                || {
                    let tree = MassTree15::<u64>::new();
                    {
                        let guard = tree.guard();
                        for (i, key) in keys.iter().take(N / 2).enumerate() {
                            let _ = tree.insert_with_guard(key, i as u64, &guard);
                        }
                    }
                    Arc::new(tree)
                },
                |tree| {
                    let start_barrier = Arc::new(Barrier::new(threads));
                    let handles: Vec<_> = (0..threads)
                        .map(|t| {
                            let tree = Arc::clone(&tree);
                            let keys = Arc::clone(&keys);
                            let start_barrier = Arc::clone(&start_barrier);
                            thread::spawn(move || {
                                let guard = tree.guard();
                                let start = (t * 7919) % keys.len();
                                start_barrier.wait();
                                for i in 0..OPS_PER_THREAD {
                                    let idx = (start + i) % keys.len();
                                    let _ = tree.insert_with_guard(&keys[idx], i as u64, &guard);
                                }
                            })
                        })
                        .collect();

                    for h in handles {
                        h.join().unwrap();
                    }
                    black_box(tree)
                },
            );
        });
    }
    group.finish();
}

// =============================================================================
// 11: SINGLE HOT KEY
// =============================================================================

fn bench_11_single_hot_key(c: &mut Criterion) {
    let mut group = c.benchmark_group("11_single_hot_key");
    group.sampling_mode(SamplingMode::Flat);
    const N: usize = 100_000;
    const OPS_PER_THREAD: usize = 10_000;

    let keys = keys::<8>(N);
    let hot_key = keys[N / 2];

    for threads in [2, 4, 8, 16, 32] {
        group.bench_function(BenchmarkId::new("masstree15", threads), |b| {
            b.iter_with_setup(
                || Arc::new(setup_masstree15::<8>(&keys)),
                |tree| {
                    let start = Arc::new(Barrier::new(threads));
                    let handles: Vec<_> = (0..threads)
                        .map(|t| {
                            let tree = Arc::clone(&tree);
                            let start = Arc::clone(&start);
                            thread::spawn(move || {
                                let guard = tree.guard();
                                let mut sum = 0u64;
                                start.wait();
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
                    black_box(tree)
                },
            );
        });
    }
    group.finish();
}

// =============================================================================
// 11a: RANDOM READ 8B
// =============================================================================

fn bench_11a_random_read_8b(c: &mut Criterion) {
    let mut group = c.benchmark_group("11a_random_read_8B");
    group.sampling_mode(SamplingMode::Flat);
    const N: usize = 1_000_000;
    const OPS_PER_THREAD: usize = 100_000;

    let keys = Arc::new(keys::<8>(N));
    let tree = Arc::new(setup_masstree15::<8>(&keys));

    for threads in [1, 2, 3, 4, 5, 6] {
        let keys = Arc::clone(&keys);
        let tree = Arc::clone(&tree);
        let indices = Arc::new(uniform_indices(N, OPS_PER_THREAD * threads, 42));

        group.throughput(Throughput::Elements((threads * OPS_PER_THREAD) as u64));
        group.bench_function(BenchmarkId::new("masstree15", threads), |b| {
            let keys = Arc::clone(&keys);
            let tree = Arc::clone(&tree);
            let indices = Arc::clone(&indices);

            b.iter(|| {
                let start_barrier = Arc::new(Barrier::new(threads));
                let handles: Vec<_> = (0..threads)
                    .map(|t| {
                        let tree = Arc::clone(&tree);
                        let keys = Arc::clone(&keys);
                        let indices = Arc::clone(&indices);
                        let start_barrier = Arc::clone(&start_barrier);
                        thread::spawn(move || {
                            let guard = tree.guard();
                            let mut sum = 0u64;
                            let start = t * OPS_PER_THREAD;
                            start_barrier.wait();
                            for i in 0..OPS_PER_THREAD {
                                let idx = indices[start + i];
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
        });
    }
    group.finish();
}

// =============================================================================
// 11b: RANDOM READ 32B
// =============================================================================

fn bench_11b_random_read_32b(c: &mut Criterion) {
    let mut group = c.benchmark_group("11b_random_read_32B");
    group.sampling_mode(SamplingMode::Flat);
    const N: usize = 1_000_000;
    const OPS_PER_THREAD: usize = 100_000;

    let keys = Arc::new(keys::<32>(N));
    let tree = Arc::new(setup_masstree15::<32>(&keys));

    for threads in [1, 2, 3, 4, 5, 6] {
        let keys = Arc::clone(&keys);
        let tree = Arc::clone(&tree);
        let indices = Arc::new(uniform_indices(N, OPS_PER_THREAD * threads, 42));

        group.throughput(Throughput::Elements((threads * OPS_PER_THREAD) as u64));
        group.bench_function(BenchmarkId::new("masstree15", threads), |b| {
            let keys = Arc::clone(&keys);
            let tree = Arc::clone(&tree);
            let indices = Arc::clone(&indices);

            b.iter(|| {
                let start_barrier = Arc::new(Barrier::new(threads));
                let handles: Vec<_> = (0..threads)
                    .map(|t| {
                        let tree = Arc::clone(&tree);
                        let keys = Arc::clone(&keys);
                        let indices = Arc::clone(&indices);
                        let start_barrier = Arc::clone(&start_barrier);
                        thread::spawn(move || {
                            let guard = tree.guard();
                            let mut sum = 0u64;
                            let start = t * OPS_PER_THREAD;
                            start_barrier.wait();
                            for i in 0..OPS_PER_THREAD {
                                let idx = indices[start + i];
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
        });
    }
    group.finish();
}

// =============================================================================
// 12: GET BY KEY SIZE SHARED PREFIX
// =============================================================================

fn bench_12_get_by_key_size_shared_prefix(c: &mut Criterion) {
    let mut group = c.benchmark_group("12_get_by_key_size_shared_prefix");
    group.sampling_mode(SamplingMode::Flat);
    const N: usize = 10_000;
    const PREFIX_BUCKETS: u64 = 256;

    // 16B
    {
        let keys = keys_shared_prefix::<16>(N, PREFIX_BUCKETS);
        let tree = setup_masstree15::<16>(&keys);
        let lookup_keys = uniform_indices(N, 1000, 42);

        group.bench_function("masstree15_16B", |b| {
            b.iter(|| {
                let guard = tree.guard();
                let mut sum = 0u64;
                for &idx in &lookup_keys {
                    if let Some(v) = tree.get_ref(&keys[idx], &guard) {
                        sum += *v;
                    }
                }
                black_box(sum)
            });
        });
    }

    // 24B
    {
        let keys = keys_shared_prefix::<24>(N, PREFIX_BUCKETS);
        let tree = setup_masstree15::<24>(&keys);
        let lookup_keys = uniform_indices(N, 1000, 42);

        group.bench_function("masstree15_24B", |b| {
            b.iter(|| {
                let guard = tree.guard();
                let mut sum = 0u64;
                for &idx in &lookup_keys {
                    if let Some(v) = tree.get_ref(&keys[idx], &guard) {
                        sum += *v;
                    }
                }
                black_box(sum)
            });
        });
    }

    // 32B
    {
        let keys = keys_shared_prefix::<32>(N, PREFIX_BUCKETS);
        let tree = setup_masstree15::<32>(&keys);
        let lookup_keys = uniform_indices(N, 1000, 42);

        group.bench_function("masstree15_32B", |b| {
            b.iter(|| {
                let guard = tree.guard();
                let mut sum = 0u64;
                for &idx in &lookup_keys {
                    if let Some(v) = tree.get_ref(&keys[idx], &guard) {
                        sum += *v;
                    }
                }
                black_box(sum)
            });
        });
    }

    group.finish();
}

// =============================================================================
// 12a: STRING VALUES READ
// =============================================================================

fn bench_12a_string_values_read(c: &mut Criterion) {
    let mut group = c.benchmark_group("12a_string_values_read");
    group.sampling_mode(SamplingMode::Flat);
    const N: usize = 500_000;
    const OPS_PER_THREAD: usize = 50_000;

    let keys = Arc::new(keys::<16>(N));
    let tree = Arc::new(setup_masstree15_string::<16>(&keys));

    for threads in [1, 2, 3, 4, 5, 6] {
        let keys = Arc::clone(&keys);
        let tree = Arc::clone(&tree);
        let indices = Arc::new(uniform_indices(N, OPS_PER_THREAD * threads, 42));

        group.throughput(Throughput::Elements((threads * OPS_PER_THREAD) as u64));
        group.bench_function(BenchmarkId::new("masstree15_string", threads), |b| {
            let keys = Arc::clone(&keys);
            let tree = Arc::clone(&tree);
            let indices = Arc::clone(&indices);

            b.iter(|| {
                let start_barrier = Arc::new(Barrier::new(threads));
                let handles: Vec<_> = (0..threads)
                    .map(|t| {
                        let tree = Arc::clone(&tree);
                        let keys = Arc::clone(&keys);
                        let indices = Arc::clone(&indices);
                        let start_barrier = Arc::clone(&start_barrier);
                        thread::spawn(move || {
                            let guard = tree.guard();
                            let mut len_sum = 0usize;
                            let start = t * OPS_PER_THREAD;
                            start_barrier.wait();
                            for i in 0..OPS_PER_THREAD {
                                let idx = indices[start + i];
                                if let Some(v) = tree.get_ref(&keys[idx], &guard) {
                                    len_sum += v.len();
                                }
                            }
                            black_box(len_sum);
                        })
                    })
                    .collect();

                for h in handles {
                    h.join().unwrap();
                }
            });
        });
    }
    group.finish();
}

// =============================================================================
// 12b: STRING VALUES WRITE
// =============================================================================

fn bench_12b_string_values_write(c: &mut Criterion) {
    let mut group = c.benchmark_group("12b_string_values_write");
    group.sampling_mode(SamplingMode::Flat);
    const N: usize = 100_000;
    const OPS_PER_THREAD: usize = 10_000;

    let keys = Arc::new(keys::<16>(N));

    for threads in [1, 2, 3, 4, 5, 6] {
        let keys = Arc::clone(&keys);

        group.throughput(Throughput::Elements((threads * OPS_PER_THREAD) as u64));
        group.bench_function(BenchmarkId::new("masstree15_string", threads), |b| {
            let keys = Arc::clone(&keys);

            b.iter_with_setup(
                || {
                    let tree = MassTree15::<String>::new();
                    {
                        let guard = tree.guard();
                        for (i, key) in keys.iter().take(N / 2).enumerate() {
                            let _ = tree.insert_with_guard(key, generate_string_value(i), &guard);
                        }
                    }
                    Arc::new(tree)
                },
                |tree| {
                    let start_barrier = Arc::new(Barrier::new(threads));
                    let handles: Vec<_> = (0..threads)
                        .map(|t| {
                            let tree = Arc::clone(&tree);
                            let keys = Arc::clone(&keys);
                            let start_barrier = Arc::clone(&start_barrier);
                            thread::spawn(move || {
                                let guard = tree.guard();
                                let start = (t * 7919) % keys.len();
                                start_barrier.wait();
                                for i in 0..OPS_PER_THREAD {
                                    let idx = (start + i) % keys.len();
                                    let _ = tree.insert_with_guard(
                                        &keys[idx],
                                        generate_string_value(i + t * OPS_PER_THREAD),
                                        &guard,
                                    );
                                }
                            })
                        })
                        .collect();

                    for h in handles {
                        h.join().unwrap();
                    }
                    black_box(tree)
                },
            );
        });
    }
    group.finish();
}

// =============================================================================
// 13: CONCURRENT READS LONG KEYS SHARED PREFIX
// =============================================================================

fn bench_13_concurrent_reads_long_keys_shared_prefix(c: &mut Criterion) {
    let mut group = c.benchmark_group("13_concurrent_reads_long_keys_shared_prefix");
    group.sampling_mode(SamplingMode::Flat);
    const N: usize = 10_000_000;
    const OPS_PER_THREAD: usize = 50_000;
    const PREFIX_BUCKETS: u64 = 256;

    let keys = Arc::new(keys_shared_prefix::<32>(N, PREFIX_BUCKETS));
    let tree = Arc::new(setup_masstree15::<32>(&keys));
    let indices = Arc::new(uniform_indices(N, OPS_PER_THREAD, 42));

    for threads in [1, 2, 3, 4, 5, 6] {
        let keys = Arc::clone(&keys);
        let tree = Arc::clone(&tree);
        let indices = Arc::clone(&indices);

        group.bench_function(BenchmarkId::new("masstree15_32b", threads), |b| {
            let keys = Arc::clone(&keys);
            let tree = Arc::clone(&tree);
            let indices = Arc::clone(&indices);

            b.iter(|| {
                let start_barrier = Arc::new(Barrier::new(threads));
                let handles: Vec<_> = (0..threads)
                    .map(|t| {
                        let tree = Arc::clone(&tree);
                        let keys = Arc::clone(&keys);
                        let indices = Arc::clone(&indices);
                        let start_barrier = Arc::clone(&start_barrier);
                        thread::spawn(move || {
                            let guard = tree.guard();
                            let mut sum = 0u64;
                            let offset = t * 7919;
                            start_barrier.wait();
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
        });
    }
    group.finish();
}

// =============================================================================
// 14a: AGGRESSIVE SHARED PREFIX READ
// =============================================================================

fn bench_14a_aggressive_shared_prefix_read(c: &mut Criterion) {
    let mut group = c.benchmark_group("14a_aggressive_shared_prefix_read");
    group.sampling_mode(SamplingMode::Flat);
    const N: usize = 1_000_000;
    const OPS_PER_THREAD: usize = 100_000;
    const PREFIX_CHUNKS: usize = 3;
    const PREFIX_BUCKETS: u64 = 16;

    let keys = Arc::new(keys_shared_prefix_chunks::<32>(
        N,
        PREFIX_CHUNKS,
        PREFIX_BUCKETS,
    ));
    let tree = Arc::new(setup_masstree15::<32>(&keys));

    for threads in [1, 2, 3, 4, 5, 6] {
        let keys = Arc::clone(&keys);
        let tree = Arc::clone(&tree);
        let indices = Arc::new(uniform_indices(N, OPS_PER_THREAD * threads, 42));

        group.throughput(Throughput::Elements((threads * OPS_PER_THREAD) as u64));
        group.bench_function(BenchmarkId::new("masstree15", threads), |b| {
            let keys = Arc::clone(&keys);
            let tree = Arc::clone(&tree);
            let indices = Arc::clone(&indices);

            b.iter(|| {
                let start_barrier = Arc::new(Barrier::new(threads));
                let handles: Vec<_> = (0..threads)
                    .map(|t| {
                        let tree = Arc::clone(&tree);
                        let keys = Arc::clone(&keys);
                        let indices = Arc::clone(&indices);
                        let start_barrier = Arc::clone(&start_barrier);
                        thread::spawn(move || {
                            let guard = tree.guard();
                            let mut sum = 0u64;
                            let start = t * OPS_PER_THREAD;
                            start_barrier.wait();
                            for i in 0..OPS_PER_THREAD {
                                let idx = indices[start + i];
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
        });
    }
    group.finish();
}

// =============================================================================
// 14b: AGGRESSIVE SHARED PREFIX WRITE
// =============================================================================

fn bench_14b_aggressive_shared_prefix_write(c: &mut Criterion) {
    let mut group = c.benchmark_group("14b_aggressive_shared_prefix_write");
    group.sampling_mode(SamplingMode::Flat);
    const N: usize = 100_000;
    const OPS_PER_THREAD: usize = 10_000;
    const PREFIX_CHUNKS: usize = 3;
    const PREFIX_BUCKETS: u64 = 16;

    let keys = Arc::new(keys_shared_prefix_chunks::<32>(
        N,
        PREFIX_CHUNKS,
        PREFIX_BUCKETS,
    ));

    for threads in [1, 2, 3, 4, 5, 6] {
        let keys = Arc::clone(&keys);

        group.throughput(Throughput::Elements((threads * OPS_PER_THREAD) as u64));
        group.bench_function(BenchmarkId::new("masstree15", threads), |b| {
            let keys = Arc::clone(&keys);

            b.iter_with_setup(
                || {
                    let tree = MassTree15::<u64>::new();
                    {
                        let guard = tree.guard();
                        for (i, key) in keys.iter().take(N / 2).enumerate() {
                            let _ = tree.insert_with_guard(key, i as u64, &guard);
                        }
                    }
                    Arc::new(tree)
                },
                |tree| {
                    let start_barrier = Arc::new(Barrier::new(threads));
                    let handles: Vec<_> = (0..threads)
                        .map(|t| {
                            let tree = Arc::clone(&tree);
                            let keys = Arc::clone(&keys);
                            let start_barrier = Arc::clone(&start_barrier);
                            thread::spawn(move || {
                                let guard = tree.guard();
                                let start = (t * 7919) % keys.len();
                                start_barrier.wait();
                                for i in 0..OPS_PER_THREAD {
                                    let idx = (start + i) % keys.len();
                                    let _ = tree.insert_with_guard(&keys[idx], i as u64, &guard);
                                }
                            })
                        })
                        .collect();

                    for h in handles {
                        h.join().unwrap();
                    }
                    black_box(tree)
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
    bench_01_concurrent_writes_disjoint,
    bench_02_concurrent_writes_contention,
    bench_03_single_threaded_insert,
    bench_04_read_after_write,
    bench_05_get_by_key_size,
    bench_06_insert_by_key_size,
    bench_07_concurrent_reads_scaling,
    bench_08_concurrent_reads_long_keys,
    bench_09_mixed_uniform,
    bench_10a_read_scaling_8b,
    bench_10b_read_scaling_32b,
    bench_10c_write_scaling_32b,
    bench_11_single_hot_key,
    bench_11a_random_read_8b,
    bench_11b_random_read_32b,
    bench_12_get_by_key_size_shared_prefix,
    bench_12a_string_values_read,
    bench_12b_string_values_write,
    bench_13_concurrent_reads_long_keys_shared_prefix,
    bench_14a_aggressive_shared_prefix_read,
    bench_14b_aggressive_shared_prefix_write,
);

criterion_main!(benches);
