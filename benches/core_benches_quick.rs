//! Quick core benchmarks for regression testing.
//!
//! Run with: `cargo bench --bench core_benches_quick --features mimalloc`
//!
//! This benchmark uses small sample sizes for fast iteration during development.
//! For accurate performance measurements, use the full `concurrent_maps24` benchmark.

#![expect(clippy::indexing_slicing)]
#![expect(clippy::unwrap_used)]

use divan::{Bencher, black_box};
use masstree::{MassTree24, MassTree24Inline};

fn main() {
    divan::main();
}

// ============================================================================
//  Constants
// ============================================================================

/// Small sample size for quick regression checks
const N: usize = 100_000;

/// Operations per thread
const OPS: usize = 10_000;

// ============================================================================
//  Setup Helpers
// ============================================================================

fn setup_masstree24(keys: &[[u8; 8]]) -> MassTree24<u64> {
    let tree = MassTree24::new();
    {
        let guard = tree.guard();
        for (i, key) in keys.iter().enumerate() {
            tree.insert_with_guard(key, i as u64, &guard).unwrap();
        }
    }
    tree
}

fn setup_masstree24_inline(keys: &[[u8; 8]]) -> MassTree24Inline<u64> {
    let tree = MassTree24Inline::new();
    {
        let guard = tree.guard();
        for (i, key) in keys.iter().enumerate() {
            tree.insert_with_guard(key, i as u64, &guard).unwrap();
        }
    }
    tree
}

fn generate_keys_sequential(n: usize) -> Vec<[u8; 8]> {
    (0..n as u64).map(u64::to_be_bytes).collect()
}

fn generate_keys_random(n: usize) -> Vec<[u8; 8]> {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};

    (0..n as u64)
        .map(|i| {
            let mut hasher = DefaultHasher::new();
            i.hash(&mut hasher);
            hasher.finish().to_be_bytes()
        })
        .collect()
}

// ============================================================================
//  Single-Threaded Benchmarks
// ============================================================================

#[divan::bench]
fn st_insert_arc(bencher: Bencher<'_, '_>) {
    let keys = generate_keys_sequential(N);

    bencher
        .with_inputs(MassTree24::<u64>::new)
        .bench_values(|tree| {
            {
                let guard = tree.guard();
                for (i, key) in keys.iter().enumerate() {
                    let _ = tree.insert_with_guard(key, i as u64, &guard);
                }
            }
            tree.len()
        });
}

#[divan::bench]
fn st_insert_inline(bencher: Bencher<'_, '_>) {
    let keys = generate_keys_sequential(N);

    bencher
        .with_inputs(MassTree24Inline::<u64>::new)
        .bench_values(|tree| {
            {
                let guard = tree.guard();
                for (i, key) in keys.iter().enumerate() {
                    let _ = tree.insert_with_guard(key, i as u64, &guard);
                }
            }
            tree.len()
        });
}

#[divan::bench]
fn st_get_arc(bencher: Bencher<'_, '_>) {
    let keys = generate_keys_sequential(N);
    let tree = setup_masstree24(&keys);

    bencher.bench(|| {
        let guard = tree.guard();
        for key in &keys {
            black_box(tree.get_ref(key, &guard));
        }
    });
}

#[divan::bench]
fn st_get_inline(bencher: Bencher<'_, '_>) {
    let keys = generate_keys_sequential(N);
    let tree = setup_masstree24_inline(&keys);

    bencher.bench(|| {
        let guard = tree.guard();
        for key in &keys {
            black_box(tree.get_ref(key, &guard));
        }
    });
}

// ============================================================================
//  Multi-Threaded Read Benchmarks
// ============================================================================

#[divan::bench(threads = [1, 4, 8, 16, 32])]
fn mt_read_arc(bencher: Bencher<'_, '_>) {
    let keys = generate_keys_random(N);
    let tree = setup_masstree24(&keys);

    bencher.bench(|| {
        let guard = tree.guard();
        for i in 0..OPS {
            let key = &keys[i % keys.len()];
            black_box(tree.get_ref(key, &guard));
        }
    });
}

#[divan::bench(threads = [1, 4, 8, 16, 32])]
fn mt_read_inline(bencher: Bencher<'_, '_>) {
    let keys = generate_keys_random(N);
    let tree = setup_masstree24_inline(&keys);

    bencher.bench(|| {
        let guard = tree.guard();
        for i in 0..OPS {
            let key = &keys[i % keys.len()];
            black_box(tree.get_ref(key, &guard));
        }
    });
}

// ============================================================================
//  Multi-Threaded Write Benchmarks
// ============================================================================

#[divan::bench(threads = [1, 4, 8, 16, 32])]
fn mt_write_arc(bencher: Bencher<'_, '_>) {
    let keys = generate_keys_random(N);

    bencher
        .with_inputs(MassTree24::<u64>::new)
        .bench_values(|tree| {
            {
                let guard = tree.guard();
                for i in 0..OPS {
                    let key = &keys[i % keys.len()];
                    let _ = tree.insert_with_guard(key, i as u64, &guard);
                }
            }
            tree.len()
        });
}

#[divan::bench(threads = [1, 4, 8, 16, 32])]
fn mt_write_inline(bencher: Bencher<'_, '_>) {
    let keys = generate_keys_random(N);

    bencher
        .with_inputs(MassTree24Inline::<u64>::new)
        .bench_values(|tree| {
            {
                let guard = tree.guard();
                for i in 0..OPS {
                    let key = &keys[i % keys.len()];
                    let _ = tree.insert_with_guard(key, i as u64, &guard);
                }
            }
            tree.len()
        });
}

// ============================================================================
//  Layer Creation (Shared Prefix) Benchmarks
// ============================================================================

fn generate_keys_shared_prefix(n: usize, buckets: u64) -> Vec<[u8; 32]> {
    (0..n as u64)
        .map(|i| {
            let mut key = [0u8; 32];
            // First 8 bytes: shared prefix (limited buckets)
            let prefix = (i % buckets).to_be_bytes();
            key[0..8].copy_from_slice(&prefix);
            // Remaining bytes: unique suffix
            let suffix = i.to_be_bytes();
            key[8..16].copy_from_slice(&suffix);
            key
        })
        .collect()
}

fn setup_masstree24_32(keys: &[[u8; 32]]) -> MassTree24<u64> {
    let tree = MassTree24::new();
    {
        let guard = tree.guard();
        for (i, key) in keys.iter().enumerate() {
            tree.insert_with_guard(key, i as u64, &guard).unwrap();
        }
    }
    tree
}

fn setup_masstree24_inline_32(keys: &[[u8; 32]]) -> MassTree24Inline<u64> {
    let tree = MassTree24Inline::new();
    {
        let guard = tree.guard();
        for (i, key) in keys.iter().enumerate() {
            tree.insert_with_guard(key, i as u64, &guard).unwrap();
        }
    }
    tree
}

#[divan::bench(threads = [1, 4, 8, 16, 32])]
fn mt_layer_read_arc(bencher: Bencher<'_, '_>) {
    let keys = generate_keys_shared_prefix(N, 16);
    let tree = setup_masstree24_32(&keys);

    bencher.bench(|| {
        let guard = tree.guard();
        for i in 0..OPS {
            let key = &keys[i % keys.len()];
            black_box(tree.get_ref(key, &guard));
        }
    });
}

#[divan::bench(threads = [1, 4, 8, 16, 32])]
fn mt_layer_read_inline(bencher: Bencher<'_, '_>) {
    let keys = generate_keys_shared_prefix(N, 16);
    let tree = setup_masstree24_inline_32(&keys);

    bencher.bench(|| {
        let guard = tree.guard();
        for i in 0..OPS {
            let key = &keys[i % keys.len()];
            black_box(tree.get_ref(key, &guard));
        }
    });
}

#[divan::bench(threads = [1, 4, 8, 16, 32])]
fn mt_layer_write_arc(bencher: Bencher<'_, '_>) {
    let keys = generate_keys_shared_prefix(N, 16);

    bencher
        .with_inputs(MassTree24::<u64>::new)
        .bench_values(|tree| {
            {
                let guard = tree.guard();
                for i in 0..OPS {
                    let key = &keys[i % keys.len()];
                    let _ = tree.insert_with_guard(key, i as u64, &guard);
                }
            }
            tree.len()
        });
}

#[divan::bench(threads = [1, 4, 8, 16, 32])]
fn mt_layer_write_inline(bencher: Bencher<'_, '_>) {
    let keys = generate_keys_shared_prefix(N, 16);

    bencher
        .with_inputs(MassTree24Inline::<u64>::new)
        .bench_values(|tree| {
            {
                let guard = tree.guard();
                for i in 0..OPS {
                    let key = &keys[i % keys.len()];
                    let _ = tree.insert_with_guard(key, i as u64, &guard);
                }
            }
            tree.len()
        });
}
