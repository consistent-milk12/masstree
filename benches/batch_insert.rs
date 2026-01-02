//! ========================================================================
//!  Batch Insert Benchmarks
//! ========================================================================
//!
//! Benchmarks comparing batch insert performance against individual inserts.
//!
//! Run with:
//!   cargo bench --bench batch_insert --features mimalloc
//!
//! Key metrics:
//! - Throughput (items/sec)
//! - Speedup ratio (batch vs individual)
//! - Scaling with batch size

#![expect(clippy::unwrap_used)]
#![expect(clippy::pedantic)]

use std::hint::black_box;
use std::sync::Arc;
use std::thread;

use divan::{Bencher, counter::ItemsCount};
use masstree::{MassTree24, MassTree15, MassTree24Inline};

fn main() {
    masstree::init_tracing();
    divan::main();
}

// ============================================================================
//  Key Generation Helpers
// ============================================================================

/// Generate sequential 8-byte keys (optimal locality).
///
/// Pre-generated outside benchmark to avoid measuring allocation.
fn sequential_keys_8b(count: usize) -> Vec<Vec<u8>> {
    (0..count)
        .map(|i| (i as u64).to_be_bytes().to_vec())
        .collect()
}

/// Generate pseudo-random keys (poor locality).
fn random_keys(count: usize, seed: u64) -> Vec<Vec<u8>> {
    (0..count)
        .map(|i| {
            let hash = ((i as u64).wrapping_add(seed))
                .wrapping_mul(0x517cc1b727220a95)
                .wrapping_add(0x7f4a7c13);
            hash.to_be_bytes().to_vec()
        })
        .collect()
}

/// Generate keys with shared prefix (tests leaf clustering).
fn shared_prefix_keys(count: usize) -> Vec<Vec<u8>> {
    (0..count)
        .map(|i| {
            let mut key = b"common_prefix_".to_vec();
            key.extend(format!("{i:06}").as_bytes());
            key
        })
        .collect()
}

/// Pre-generate entries for batch operations.
///
/// Using this helper ensures entry generation isn't measured.
fn generate_entries(keys: &[Vec<u8>]) -> Vec<(Vec<u8>, u64)> {
    keys.iter()
        .enumerate()
        .map(|(i, k)| (k.clone(), i as u64))
        .collect()
}

// ============================================================================
//  Individual Insert Benchmarks (Baseline)
// ============================================================================

#[divan::bench_group(name = "01_individual_insert")]
mod individual_insert {
    use super::*;

    #[divan::bench(args = [100, 1000, 10000])]
    fn sequential(bencher: Bencher, n: usize) {
        // Pre-generate keys OUTSIDE the benchmark
        let keys = sequential_keys_8b(n);
        let values: Vec<u64> = (0..n as u64).collect();

        bencher
            .counter(ItemsCount::new(n))
            .with_inputs(|| (keys.clone(), values.clone()))
            .bench_local_values(|(keys, values)| {
                let tree: MassTree24<u64> = MassTree24::new();
                let guard = tree.guard();

                for (key, value) in keys.iter().zip(values.iter()) {
                    tree.insert_with_guard(key, *value, &guard).unwrap();
                }

                black_box(tree.len())
            });
    }

    #[divan::bench(args = [100, 1000, 10000])]
    fn random(bencher: Bencher, n: usize) {
        let keys = random_keys(n, 12345);
        let values: Vec<u64> = (0..n as u64).collect();

        bencher
            .counter(ItemsCount::new(n))
            .with_inputs(|| (keys.clone(), values.clone()))
            .bench_local_values(|(keys, values)| {
                let tree: MassTree24<u64> = MassTree24::new();
                let guard = tree.guard();

                for (key, value) in keys.iter().zip(values.iter()) {
                    tree.insert_with_guard(key, *value, &guard).unwrap();
                }

                black_box(tree.len())
            });
    }

    #[divan::bench(args = [100, 1000, 5000])]
    fn shared_prefix(bencher: Bencher, n: usize) {
        let keys = shared_prefix_keys(n);
        let values: Vec<u64> = (0..n as u64).collect();

        bencher
            .counter(ItemsCount::new(n))
            .with_inputs(|| (keys.clone(), values.clone()))
            .bench_local_values(|(keys, values)| {
                let tree: MassTree24<u64> = MassTree24::new();
                let guard = tree.guard();

                for (key, value) in keys.iter().zip(values.iter()) {
                    tree.insert_with_guard(key, *value, &guard).unwrap();
                }

                black_box(tree.len())
            });
    }
}

// ============================================================================
//  Batch Insert Benchmarks
// ============================================================================

#[divan::bench_group(name = "02_batch_insert")]
mod batch_insert {
    use super::*;

    #[divan::bench(args = [100, 1000, 10000])]
    fn sequential(bencher: Bencher, n: usize) {
        // Pre-generate entries OUTSIDE the benchmark
        let keys = sequential_keys_8b(n);
        let entries = generate_entries(&keys);

        bencher
            .counter(ItemsCount::new(n))
            .with_inputs(|| entries.clone())
            .bench_local_values(|entries| {
                let tree: MassTree24<u64> = MassTree24::new();
                let result = tree.insert_batch(entries).unwrap();
                black_box(result.inserted)
            });
    }

    #[divan::bench(args = [100, 1000, 10000])]
    fn random(bencher: Bencher, n: usize) {
        let keys = random_keys(n, 12345);
        let entries = generate_entries(&keys);

        bencher
            .counter(ItemsCount::new(n))
            .with_inputs(|| entries.clone())
            .bench_local_values(|entries| {
                let tree: MassTree24<u64> = MassTree24::new();
                let result = tree.insert_batch(entries).unwrap();
                black_box(result.inserted)
            });
    }

    #[divan::bench(args = [100, 1000, 5000])]
    fn shared_prefix(bencher: Bencher, n: usize) {
        let keys = shared_prefix_keys(n);
        let entries = generate_entries(&keys);

        bencher
            .counter(ItemsCount::new(n))
            .with_inputs(|| entries.clone())
            .bench_local_values(|entries| {
                let tree: MassTree24<u64> = MassTree24::new();
                let result = tree.insert_batch(entries).unwrap();
                black_box(result.inserted)
            });
    }
}

// ============================================================================
//  MassTree15 vs MassTree24 Batch Comparison
// ============================================================================

#[divan::bench_group(name = "03_batch_tree_comparison")]
mod batch_tree_comparison {
    use super::*;

    #[divan::bench(args = [1000, 5000, 10000])]
    fn masstree24(bencher: Bencher, n: usize) {
        let keys = sequential_keys_8b(n);
        let entries = generate_entries(&keys);

        bencher
            .counter(ItemsCount::new(n))
            .with_inputs(|| entries.clone())
            .bench_local_values(|entries| {
                let tree: MassTree24<u64> = MassTree24::new();
                let result = tree.insert_batch(entries).unwrap();
                black_box(result.inserted)
            });
    }

    #[divan::bench(args = [1000, 5000, 10000])]
    fn masstree15(bencher: Bencher, n: usize) {
        let keys = sequential_keys_8b(n);
        let entries = generate_entries(&keys);

        bencher
            .counter(ItemsCount::new(n))
            .with_inputs(|| entries.clone())
            .bench_local_values(|entries| {
                let tree: MassTree15<u64> = MassTree15::new();
                let result = tree.insert_batch(entries).unwrap();
                black_box(result.inserted)
            });
    }

    #[divan::bench(args = [1000, 5000, 10000])]
    fn masstree24_inline(bencher: Bencher, n: usize) {
        let keys = sequential_keys_8b(n);
        let entries = generate_entries(&keys);

        bencher
            .counter(ItemsCount::new(n))
            .with_inputs(|| entries.clone())
            .bench_local_values(|entries| {
                let tree: MassTree24Inline<u64> = MassTree24Inline::new();
                let result = tree.insert_batch(entries).unwrap();
                black_box(result.inserted)
            });
    }
}

// ============================================================================
//  Batch Size Scaling
// ============================================================================

#[divan::bench_group(name = "04_batch_size_scaling")]
mod batch_size_scaling {
    use super::*;

    #[divan::bench(args = [100, 500, 1000, 5000, 10000, 50000])]
    fn batch_scaling(bencher: Bencher, n: usize) {
        let keys = sequential_keys_8b(n);
        let entries = generate_entries(&keys);

        bencher
            .counter(ItemsCount::new(n))
            .with_inputs(|| entries.clone())
            .bench_local_values(|entries| {
                let tree: MassTree24<u64> = MassTree24::new();
                let result = tree.insert_batch(entries).unwrap();
                black_box(result.inserted)
            });
    }
}

// ============================================================================
//  Many Small Batches vs One Large Batch
// ============================================================================

#[divan::bench_group(name = "05_batch_amortization")]
mod batch_amortization {
    use super::*;

    const TOTAL_ENTRIES: usize = 10_000;

    #[divan::bench]
    fn many_small_batches_100x100(bencher: Bencher) {
        // Pre-generate ALL 100 batches outside benchmark
        let all_batches: Vec<Vec<(Vec<u8>, u64)>> = (0..100)
            .map(|batch_num| {
                let start = batch_num * 100;
                (0..100)
                    .map(|i| {
                        let key = ((start + i) as u64).to_be_bytes().to_vec();
                        (key, (start + i) as u64)
                    })
                    .collect()
            })
            .collect();

        bencher
            .counter(ItemsCount::new(TOTAL_ENTRIES))
            .with_inputs(|| all_batches.clone())
            .bench_local_values(|batches| {
                let tree: MassTree24<u64> = MassTree24::new();
                let guard = tree.guard();

                let mut total = 0;
                for batch in batches {
                    let result = tree.insert_batch_with_guard(batch, &guard).unwrap();
                    total += result.inserted;
                }

                black_box(total)
            });
    }

    #[divan::bench]
    fn one_large_batch_10000(bencher: Bencher) {
        let keys = sequential_keys_8b(TOTAL_ENTRIES);
        let entries = generate_entries(&keys);

        bencher
            .counter(ItemsCount::new(TOTAL_ENTRIES))
            .with_inputs(|| entries.clone())
            .bench_local_values(|entries| {
                let tree: MassTree24<u64> = MassTree24::new();
                let result = tree.insert_batch(entries).unwrap();
                black_box(result.inserted)
            });
    }

    #[divan::bench]
    fn individual_inserts_10000(bencher: Bencher) {
        let keys = sequential_keys_8b(TOTAL_ENTRIES);
        let values: Vec<u64> = (0..TOTAL_ENTRIES as u64).collect();

        bencher
            .counter(ItemsCount::new(TOTAL_ENTRIES))
            .with_inputs(|| (keys.clone(), values.clone()))
            .bench_local_values(|(keys, values)| {
                let tree: MassTree24<u64> = MassTree24::new();
                let guard = tree.guard();

                for (key, value) in keys.iter().zip(values.iter()) {
                    tree.insert_with_guard(key, *value, &guard).unwrap();
                }

                black_box(tree.len())
            });
    }
}

// ============================================================================
//  Concurrent Batch Inserts
// ============================================================================

#[divan::bench_group(name = "06_concurrent_batch")]
mod concurrent_batch {
    use super::*;

    const ENTRIES_PER_THREAD: usize = 5000;

    #[divan::bench(args = [1, 2, 4, 6])]
    fn concurrent_batch_insert(bencher: Bencher, threads: usize) {
        // Pre-generate all thread entries outside benchmark
        let all_entries: Vec<Vec<(Vec<u8>, u64)>> = (0..threads)
            .map(|t| {
                (0..ENTRIES_PER_THREAD)
                    .map(|i| {
                        let key = format!("t{t}_{i:08}").into_bytes();
                        (key, (t * ENTRIES_PER_THREAD + i) as u64)
                    })
                    .collect()
            })
            .collect();

        bencher
            .counter(ItemsCount::new(threads * ENTRIES_PER_THREAD))
            .with_inputs(|| all_entries.clone())
            .bench_local_values(|entries_per_thread| {
                let tree = Arc::new(MassTree24::<u64>::new());
                let mut handles = Vec::with_capacity(threads);

                for entries in entries_per_thread {
                    let tree = Arc::clone(&tree);
                    let handle = thread::spawn(move || {
                        tree.insert_batch(entries).unwrap()
                    });
                    handles.push(handle);
                }

                let mut total = 0;
                for handle in handles {
                    total += handle.join().unwrap().inserted;
                }

                black_box(total)
            });
    }

    #[divan::bench(args = [1, 2, 4, 6])]
    fn concurrent_individual_insert(bencher: Bencher, threads: usize) {
        let all_keys: Vec<Vec<Vec<u8>>> = (0..threads)
            .map(|t| {
                (0..ENTRIES_PER_THREAD)
                    .map(|i| format!("t{t}_{i:08}").into_bytes())
                    .collect()
            })
            .collect();

        bencher
            .counter(ItemsCount::new(threads * ENTRIES_PER_THREAD))
            .with_inputs(|| all_keys.clone())
            .bench_local_values(|keys_per_thread| {
                let tree = Arc::new(MassTree24::<u64>::new());
                let mut handles = Vec::with_capacity(threads);

                for (t, keys) in keys_per_thread.into_iter().enumerate() {
                    let tree = Arc::clone(&tree);
                    let handle = thread::spawn(move || {
                        let guard = tree.guard();
                        for (i, key) in keys.iter().enumerate() {
                            tree.insert_with_guard(key, (t * ENTRIES_PER_THREAD + i) as u64, &guard).unwrap();
                        }
                    });
                    handles.push(handle);
                }

                for handle in handles {
                    handle.join().unwrap();
                }

                black_box(tree.len())
            });
    }
}
