//! Regression test benchmarks for MassTree15Inline.

#![expect(clippy::unwrap_used)]
#![expect(clippy::pedantic)]
#![expect(clippy::indexing_slicing)]

mod bench_utils;

use bench_utils::{
    keys, post_measurement_barrier, pre_measurement_barrier, uniform_indices, zipfian_indices,
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

/// Warmup iterations per thread (excluded from timing and counter)
const WARMUP_OPS: usize = 500;

/// Warmup scans for prefix benchmark (excluded from timing and counter)
const WARMUP_SCANS: usize = 50;

/// Standard thread counts for all benchmarks
const THREAD_COUNTS: [usize; 6] = [1, 2, 4, 6, 8, 12];

/// Long key size in bytes
const LONG_KEY_SIZE: usize = 128;

// =============================================================================
// Setup Helpers
// =============================================================================

fn setup_masstree15_long(keys: &[[u8; LONG_KEY_SIZE]]) -> MassTree15Inline<u64> {
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

// =============================================================================
// 01: LONG KEYS (128 bytes) - Suffix Handling & Long Key Comparisons
// =============================================================================
//
// Tests 128-byte keys where each key has a UNIQUE first 8-byte chunk (keys<128>
// uses index * 1 for chunk 0). This means lookups stay in the first trie layer
// and the benchmark measures:
// - Suffix storage and comparison overhead for 120-byte suffixes
// - Long key comparison costs
//
// This is NOT a multi-layer trie traversal test. For that, see benchmark 06
// (keys_shared_prefix_chunks) which forces collisions in early chunks.

#[divan::bench_group(name = "01_long_keys_128b", sample_count = 200)]
mod long_keys {
    use super::*;

    const N: usize = 50_000;
    const OPS_PER_THREAD: usize = 12_500;
    const WRITE_RATIO: usize = 10;

    #[divan::bench(args = THREAD_COUNTS)]
    fn masstree15(bencher: Bencher, threads: usize) {
        let keys = Arc::new(keys::<LONG_KEY_SIZE>(N));
        let indices = Arc::new(uniform_indices(N, OPS_PER_THREAD * threads, 42));
        let write_decisions: Arc<Vec<Vec<bool>>> = Arc::new(
            (0..threads)
                .map(|t| shuffled_write_decisions(OPS_PER_THREAD, WRITE_RATIO, 42 + t as u64))
                .collect(),
        );

        bencher
            .counter(divan::counter::ItemsCount::new(threads * OPS_PER_THREAD))
            .with_inputs(|| Arc::new(setup_masstree15_long(keys.as_ref())))
            .bench_local_values(|tree| {
                let barrier = Arc::new(Barrier::new(threads));
                let handles: Vec<_> = (0..threads)
                    .map(|t| {
                        let tree = Arc::clone(&tree);
                        let keys = Arc::clone(&keys);
                        let indices = Arc::clone(&indices);
                        let write_decisions = Arc::clone(&write_decisions);
                        let barrier = Arc::clone(&barrier);
                        thread::spawn(move || {
                            let guard = tree.guard();
                            let base = t * OPS_PER_THREAD;
                            let is_write = &write_decisions[t];

                            // Warmup: mirror measurement op mix (reads + writes)
                            for i in 0..WARMUP_OPS {
                                let idx = indices[base + i];
                                if is_write[i] {
                                    let _ = tree.insert_with_guard(&keys[idx], i as u64, &guard);
                                } else {
                                    black_box(tree.get_with_guard(&keys[idx], &guard));
                                }
                            }

                            // Single barrier after warmup
                            barrier.wait();

                            // === MEASUREMENT ===
                            pre_measurement_barrier();
                            let mut sum = 0u64;
                            for i in 0..OPS_PER_THREAD {
                                let idx = indices[base + i];
                                if is_write[i] {
                                    let _ = tree.insert_with_guard(&keys[idx], i as u64, &guard);
                                } else if let Some(v) = tree.get_with_guard(&keys[idx], &guard) {
                                    sum = black_box(sum.wrapping_add(v));
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
// 02: MULTIPLE HOT KEYS - Realistic Cache Pattern (Zipfian Distribution)
// =============================================================================
//
// Tests access patterns following a Zipfian distribution where a small number
// of keys receive the majority of accesses:
// - Top key gets ~19% of accesses, top 8 keys get ~50%
// - 10% writes mixed in
// - Uses unique-prefix keys (keys<128>), so comparisons short-circuit early

#[divan::bench_group(name = "02_multiple_hot_keys", sample_count = 200)]
mod multiple_hot_keys {
    use super::*;

    const N: usize = 50_000;
    const OPS_PER_THREAD: usize = 10_000;
    const WRITE_RATIO: usize = 10;

    #[divan::bench(args = THREAD_COUNTS)]
    fn masstree15(bencher: Bencher, threads: usize) {
        let keys = Arc::new(keys::<LONG_KEY_SIZE>(N));
        let indices = Arc::new(zipfian_indices(N, OPS_PER_THREAD * threads, 1.0, 42));
        let write_decisions: Arc<Vec<Vec<bool>>> = Arc::new(
            (0..threads)
                .map(|t| shuffled_write_decisions(OPS_PER_THREAD, WRITE_RATIO, 42 + t as u64))
                .collect(),
        );

        bencher
            .counter(divan::counter::ItemsCount::new(threads * OPS_PER_THREAD))
            .with_inputs(|| Arc::new(setup_masstree15_long(keys.as_ref())))
            .bench_local_values(|tree| {
                let barrier = Arc::new(Barrier::new(threads));
                let handles: Vec<_> = (0..threads)
                    .map(|t| {
                        let tree = Arc::clone(&tree);
                        let keys = Arc::clone(&keys);
                        let indices = Arc::clone(&indices);
                        let write_decisions = Arc::clone(&write_decisions);
                        let barrier = Arc::clone(&barrier);
                        thread::spawn(move || {
                            let guard = tree.guard();
                            let base = t * OPS_PER_THREAD;
                            let is_write = &write_decisions[t];

                            // Warmup: mirror measurement op mix
                            for i in 0..WARMUP_OPS {
                                let idx = indices[base + i];
                                if is_write[i] {
                                    let _ = tree.insert_with_guard(&keys[idx], i as u64, &guard);
                                } else {
                                    black_box(tree.get_with_guard(&keys[idx], &guard));
                                }
                            }

                            barrier.wait();

                            pre_measurement_barrier();
                            let mut sum = 0u64;
                            for i in 0..OPS_PER_THREAD {
                                let idx = indices[base + i];
                                if is_write[i] {
                                    let _ = tree.insert_with_guard(&keys[idx], i as u64, &guard);
                                } else if let Some(v) = tree.get_with_guard(&keys[idx], &guard) {
                                    sum = black_box(sum.wrapping_add(v));
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
// 03: MIXED GET/INSERT/REMOVE - Full Operation Mix (Dynamic Set)
// =============================================================================
//
// Tests all three operations: 70% get, 20% insert, 10% remove.
// Stresses memory reclamation and tests the full API surface.
//
// State drift: Over measurement, removes cause misses and inserts become updates.
// Tree is recreated each sample via `with_inputs` for consistent starting state.

#[divan::bench_group(name = "03_mixed_get_insert_remove", sample_count = 200)]
mod mixed_operations {
    use super::*;

    const N: usize = 50_000;
    const OPS_PER_THREAD: usize = 10_000;

    #[derive(Clone, Copy)]
    enum Op {
        Get,
        Insert,
        Remove,
    }

    /// Generate operation decisions: 70% get, 20% insert, 10% remove
    fn shuffled_op_decisions(count: usize, seed: u64) -> Vec<Op> {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let get_count = (count * 70) / 100;
        let insert_count = (count * 20) / 100;
        let remove_count = count - get_count - insert_count;

        let mut decisions = Vec::with_capacity(count);
        decisions.extend(std::iter::repeat_n(Op::Get, get_count));
        decisions.extend(std::iter::repeat_n(Op::Insert, insert_count));
        decisions.extend(std::iter::repeat_n(Op::Remove, remove_count));

        // Fisher-Yates shuffle
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

    #[divan::bench(args = THREAD_COUNTS)]
    fn masstree15(bencher: Bencher, threads: usize) {
        let keys = Arc::new(keys::<LONG_KEY_SIZE>(N));
        let indices = Arc::new(uniform_indices(N, OPS_PER_THREAD * threads, 42));
        let op_decisions: Arc<Vec<Vec<Op>>> = Arc::new(
            (0..threads)
                .map(|t| shuffled_op_decisions(OPS_PER_THREAD, 42 + t as u64))
                .collect(),
        );

        bencher
            .counter(divan::counter::ItemsCount::new(threads * OPS_PER_THREAD))
            .with_inputs(|| Arc::new(setup_masstree15_long(keys.as_ref())))
            .bench_local_values(|tree| {
                let barrier = Arc::new(Barrier::new(threads));
                let handles: Vec<_> = (0..threads)
                    .map(|t| {
                        let tree = Arc::clone(&tree);
                        let keys = Arc::clone(&keys);
                        let indices = Arc::clone(&indices);
                        let op_decisions = Arc::clone(&op_decisions);
                        let barrier = Arc::clone(&barrier);
                        thread::spawn(move || {
                            let guard = tree.guard();
                            let base = t * OPS_PER_THREAD;
                            let ops = &op_decisions[t];

                            // Warmup: mirror measurement op mix
                            for i in 0..WARMUP_OPS {
                                let idx = indices[base + i];
                                match ops[i] {
                                    Op::Get => {
                                        black_box(tree.get_with_guard(&keys[idx], &guard));
                                    }
                                    Op::Insert => {
                                        let _ =
                                            tree.insert_with_guard(&keys[idx], i as u64, &guard);
                                    }
                                    Op::Remove => {
                                        let _ = tree.remove_with_guard(&keys[idx], &guard);
                                    }
                                }
                            }

                            barrier.wait();

                            pre_measurement_barrier();
                            let mut sum = 0u64;
                            for i in 0..OPS_PER_THREAD {
                                let idx = indices[base + i];
                                match ops[i] {
                                    Op::Get => {
                                        if let Some(v) = tree.get_with_guard(&keys[idx], &guard) {
                                            sum = black_box(sum.wrapping_add(v));
                                        }
                                    }
                                    Op::Insert => {
                                        let _ =
                                            tree.insert_with_guard(&keys[idx], i as u64, &guard);
                                    }
                                    Op::Remove => {
                                        let _ = tree.remove_with_guard(&keys[idx], &guard);
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
// 04: VARIABLE LENGTH KEYS (64-256 bytes) - Realistic Key Sizes
// =============================================================================
//
// Tests variable-length keys (64-256 bytes) representing URLs, paths, etc.
// Masstree passes `&[u8]` slices directly with no clone overhead on writes.

#[divan::bench_group(name = "04_variable_long_keys", sample_count = 200)]
mod variable_keys {
    use super::*;

    const N: usize = 50_000;
    const OPS_PER_THREAD: usize = 12_500;
    const WRITE_RATIO: usize = 10;
    const MIN_KEY_SIZE: usize = 64;
    const MAX_KEY_SIZE: usize = 256;

    /// Generate keys with random lengths between min and max
    fn variable_length_keys(n: usize, min: usize, max: usize, seed: u64) -> Vec<Vec<u8>> {
        let mut keys = Vec::with_capacity(n);
        let mut state = seed;

        for i in 0..n {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1);
            let len = min + ((state >> 32) as usize) % (max - min + 1);

            let mut key = vec![0u8; len];
            let base = (i as u64).to_be_bytes();
            key[0..8].copy_from_slice(&base);
            for byte in key[8..].iter_mut() {
                state = state
                    .wrapping_mul(6_364_136_223_846_793_005)
                    .wrapping_add(1);
                *byte = (state >> 56) as u8;
            }
            keys.push(key);
        }
        keys
    }

    fn setup_masstree15_var(keys: &[Vec<u8>]) -> MassTree15Inline<u64> {
        let tree = MassTree15Inline::new();
        {
            let guard = tree.guard();
            for (i, key) in keys.iter().enumerate() {
                let _ = tree.insert_with_guard(key.as_slice(), i as u64, &guard);
            }
        }
        tree
    }

    #[divan::bench(args = THREAD_COUNTS)]
    fn masstree15(bencher: Bencher, threads: usize) {
        let keys = Arc::new(variable_length_keys(N, MIN_KEY_SIZE, MAX_KEY_SIZE, 42));
        let indices = Arc::new(uniform_indices(N, OPS_PER_THREAD * threads, 42));
        let write_decisions: Arc<Vec<Vec<bool>>> = Arc::new(
            (0..threads)
                .map(|t| shuffled_write_decisions(OPS_PER_THREAD, WRITE_RATIO, 42 + t as u64))
                .collect(),
        );

        bencher
            .counter(divan::counter::ItemsCount::new(threads * OPS_PER_THREAD))
            .with_inputs(|| Arc::new(setup_masstree15_var(keys.as_ref())))
            .bench_local_values(|tree| {
                let barrier = Arc::new(Barrier::new(threads));
                let handles: Vec<_> = (0..threads)
                    .map(|t| {
                        let tree = Arc::clone(&tree);
                        let keys = Arc::clone(&keys);
                        let indices = Arc::clone(&indices);
                        let write_decisions = Arc::clone(&write_decisions);
                        let barrier = Arc::clone(&barrier);
                        thread::spawn(move || {
                            let guard = tree.guard();
                            let base = t * OPS_PER_THREAD;
                            let is_write = &write_decisions[t];

                            // Warmup: mirror measurement op mix
                            for i in 0..WARMUP_OPS {
                                let idx = indices[base + i];
                                if is_write[i] {
                                    let _ = tree.insert_with_guard(
                                        keys[idx].as_slice(),
                                        i as u64,
                                        &guard,
                                    );
                                } else {
                                    black_box(tree.get_with_guard(keys[idx].as_slice(), &guard));
                                }
                            }

                            barrier.wait();

                            pre_measurement_barrier();
                            let mut sum = 0u64;
                            for i in 0..OPS_PER_THREAD {
                                let idx = indices[base + i];
                                if is_write[i] {
                                    let _ = tree.insert_with_guard(
                                        keys[idx].as_slice(),
                                        i as u64,
                                        &guard,
                                    );
                                } else if let Some(v) =
                                    tree.get_with_guard(keys[idx].as_slice(), &guard)
                                {
                                    sum = black_box(sum.wrapping_add(v));
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
// 05: PREFIX QUERIES - Prefix Scan Performance
// =============================================================================
//
// Tests scan_prefix() with 100 distinct prefixes (~500 keys each).
//
// Counter measures SCANS/SEC not records/sec. Each scan visits ~500 keys.
// To convert to records/sec, multiply by ~500.

#[divan::bench_group(name = "05_prefix_queries", sample_count = 200)]
mod prefix_queries {
    use super::*;
    use bench_utils::keys_shared_prefix;

    const N: usize = 50_000;
    const PREFIX_BUCKETS: u64 = 100; // 100 distinct prefixes, ~500 keys each
    const SCANS_PER_THREAD: usize = 500;
    const PREFIX_KEY_SIZE: usize = 64;

    fn setup_masstree15_prefix(keys: &[[u8; PREFIX_KEY_SIZE]]) -> MassTree15Inline<u64> {
        let tree = MassTree15Inline::new();
        {
            let guard = tree.guard();
            for (i, key) in keys.iter().enumerate() {
                let _ = tree.insert_with_guard(key, i as u64, &guard);
            }
        }
        tree
    }

    /// Generate random prefix indices to scan
    fn random_prefixes(count: usize, num_prefixes: u64, seed: u64) -> Vec<[u8; 8]> {
        let mut prefixes = Vec::with_capacity(count);
        let mut state = seed;

        for _ in 0..count {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1);
            let prefix_id = (state >> 32) % num_prefixes;
            prefixes.push(prefix_id.to_be_bytes());
        }
        prefixes
    }

    #[divan::bench(args = THREAD_COUNTS)]
    fn masstree15(bencher: Bencher, threads: usize) {
        let keys = Arc::new(keys_shared_prefix::<PREFIX_KEY_SIZE>(N, PREFIX_BUCKETS));
        let prefixes = Arc::new(random_prefixes(
            SCANS_PER_THREAD * threads,
            PREFIX_BUCKETS,
            42,
        ));

        bencher
            .counter(divan::counter::ItemsCount::new(threads * SCANS_PER_THREAD))
            .with_inputs(|| Arc::new(setup_masstree15_prefix(keys.as_ref())))
            .bench_local_values(|tree| {
                let barrier = Arc::new(Barrier::new(threads));
                let handles: Vec<_> = (0..threads)
                    .map(|t| {
                        let tree = Arc::clone(&tree);
                        let prefixes = Arc::clone(&prefixes);
                        let barrier = Arc::clone(&barrier);
                        thread::spawn(move || {
                            let guard = tree.guard();
                            let base = t * SCANS_PER_THREAD;

                            // Warmup: same scan pattern as measurement
                            for i in 0..WARMUP_SCANS {
                                let prefix = &prefixes[base + i];
                                let mut count = 0u64;
                                tree.scan_prefix(
                                    prefix,
                                    |_, v| {
                                        count = count.wrapping_add(v);
                                        true
                                    },
                                    &guard,
                                );
                                black_box(count);
                            }

                            barrier.wait();

                            pre_measurement_barrier();
                            let mut total = 0u64;
                            for i in 0..SCANS_PER_THREAD {
                                let prefix = &prefixes[base + i];
                                tree.scan_prefix(
                                    prefix,
                                    |_, v| {
                                        total = black_box(total.wrapping_add(v));
                                        true
                                    },
                                    &guard,
                                );
                            }
                            post_measurement_barrier();

                            black_box(total);
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
// 06: DEEP TRIE TRAVERSAL - Multi-layer Descent via Shared Prefix Chunks
// =============================================================================
//
// Uses keys_shared_prefix_chunks to force collisions in the FIRST 4 trie layers.
// With prefix_buckets=16, ~3125 keys per bucket share 4 initial chunks (32 bytes).
//
// Mixed read/write workload (10% writes). For pure traversal cost, see 07.

#[divan::bench_group(name = "06_deep_trie_traversal", sample_count = 200)]
mod deep_trie {
    use super::*;
    use bench_utils::keys_shared_prefix_chunks;

    const N: usize = 50_000;
    const OPS_PER_THREAD: usize = 12_500;
    const WRITE_RATIO: usize = 10;
    const PREFIX_CHUNKS: usize = 4;
    const PREFIX_BUCKETS: u64 = 16;

    fn setup_masstree15_deep(keys: &[[u8; LONG_KEY_SIZE]]) -> MassTree15Inline<u64> {
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
        let keys = Arc::new(keys_shared_prefix_chunks::<LONG_KEY_SIZE>(
            N,
            PREFIX_CHUNKS,
            PREFIX_BUCKETS,
        ));
        let indices = Arc::new(uniform_indices(N, OPS_PER_THREAD * threads, 42));
        let write_decisions: Arc<Vec<Vec<bool>>> = Arc::new(
            (0..threads)
                .map(|t| shuffled_write_decisions(OPS_PER_THREAD, WRITE_RATIO, 42 + t as u64))
                .collect(),
        );

        bencher
            .counter(divan::counter::ItemsCount::new(threads * OPS_PER_THREAD))
            .with_inputs(|| Arc::new(setup_masstree15_deep(keys.as_ref())))
            .bench_local_values(|tree| {
                let barrier = Arc::new(Barrier::new(threads));
                let handles: Vec<_> = (0..threads)
                    .map(|t| {
                        let tree = Arc::clone(&tree);
                        let keys = Arc::clone(&keys);
                        let indices = Arc::clone(&indices);
                        let write_decisions = Arc::clone(&write_decisions);
                        let barrier = Arc::clone(&barrier);
                        thread::spawn(move || {
                            let guard = tree.guard();
                            let base = t * OPS_PER_THREAD;
                            let is_write = &write_decisions[t];

                            // Warmup: mirror measurement op mix
                            for i in 0..WARMUP_OPS {
                                let idx = indices[base + i];
                                if is_write[i] {
                                    let _ = tree.insert_with_guard(&keys[idx], i as u64, &guard);
                                } else {
                                    black_box(tree.get_with_guard(&keys[idx], &guard));
                                }
                            }

                            barrier.wait();

                            pre_measurement_barrier();
                            let mut sum = 0u64;
                            for i in 0..OPS_PER_THREAD {
                                let idx = indices[base + i];
                                if is_write[i] {
                                    let _ = tree.insert_with_guard(&keys[idx], i as u64, &guard);
                                } else if let Some(v) = tree.get_with_guard(&keys[idx], &guard) {
                                    sum = black_box(sum.wrapping_add(v));
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
// 07: DEEP TRIE READ-ONLY - Isolate Traversal Cost from Write Overhead
// =============================================================================
//
// Pure read workload variant of benchmark 06. Isolates multi-layer trie
// traversal cost from write-path overhead (allocator behavior, etc.).

#[divan::bench_group(name = "07_deep_trie_read_only", sample_count = 200)]
mod deep_trie_read_only {
    use super::*;
    use bench_utils::keys_shared_prefix_chunks;

    const N: usize = 50_000;
    const OPS_PER_THREAD: usize = 25_000; // More ops since read-only is faster
    const PREFIX_CHUNKS: usize = 4;
    const PREFIX_BUCKETS: u64 = 16;

    fn setup_masstree15_deep(keys: &[[u8; LONG_KEY_SIZE]]) -> MassTree15Inline<u64> {
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
        let keys = Arc::new(keys_shared_prefix_chunks::<LONG_KEY_SIZE>(
            N,
            PREFIX_CHUNKS,
            PREFIX_BUCKETS,
        ));
        let indices = Arc::new(uniform_indices(N, OPS_PER_THREAD * threads, 42));

        bencher
            .counter(divan::counter::ItemsCount::new(threads * OPS_PER_THREAD))
            .with_inputs(|| Arc::new(setup_masstree15_deep(keys.as_ref())))
            .bench_local_values(|tree| {
                let barrier = Arc::new(Barrier::new(threads));
                let handles: Vec<_> = (0..threads)
                    .map(|t| {
                        let tree = Arc::clone(&tree);
                        let keys = Arc::clone(&keys);
                        let indices = Arc::clone(&indices);
                        let barrier = Arc::clone(&barrier);
                        thread::spawn(move || {
                            let guard = tree.guard();
                            let base = t * OPS_PER_THREAD;

                            // Warmup: pure reads (matches measurement)
                            for i in 0..WARMUP_OPS {
                                let idx = indices[base + i];
                                black_box(tree.get_with_guard(&keys[idx], &guard));
                            }

                            barrier.wait();

                            pre_measurement_barrier();
                            let mut sum = 0u64;
                            for i in 0..OPS_PER_THREAD {
                                let idx = indices[base + i];
                                if let Some(v) = tree.get_with_guard(&keys[idx], &guard) {
                                    sum = black_box(sum.wrapping_add(v));
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
// 08: VARIABLE LENGTH KEYS WITH ARC - Cheap Key References
// =============================================================================
//
// Uses Arc<[u8]> keys (64-256 bytes) for cheap reference cloning.
// Compare with benchmark 04 to isolate key ownership costs from structure perf.

#[divan::bench_group(name = "08_variable_keys_arc", sample_count = 200)]
mod variable_keys_arc {
    use super::*;

    const N: usize = 50_000;
    const OPS_PER_THREAD: usize = 12_500;
    const WRITE_RATIO: usize = 10;
    const MIN_KEY_SIZE: usize = 64;
    const MAX_KEY_SIZE: usize = 256;

    /// Generate keys as Arc<[u8]> for cheap cloning
    fn variable_length_keys_arc(n: usize, min: usize, max: usize, seed: u64) -> Vec<Arc<[u8]>> {
        let mut keys = Vec::with_capacity(n);
        let mut state = seed;

        for i in 0..n {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1);
            let len = min + ((state >> 32) as usize) % (max - min + 1);

            let mut key = vec![0u8; len];
            let base = (i as u64).to_be_bytes();
            key[0..8].copy_from_slice(&base);
            for byte in key[8..].iter_mut() {
                state = state
                    .wrapping_mul(6_364_136_223_846_793_005)
                    .wrapping_add(1);
                *byte = (state >> 56) as u8;
            }
            keys.push(Arc::from(key.into_boxed_slice()));
        }
        keys
    }

    fn setup_masstree15_arc(keys: &[Arc<[u8]>]) -> MassTree15Inline<u64> {
        let tree = MassTree15Inline::new();
        {
            let guard = tree.guard();
            for (i, key) in keys.iter().enumerate() {
                let _ = tree.insert_with_guard(key.as_ref(), i as u64, &guard);
            }
        }
        tree
    }

    #[divan::bench(args = THREAD_COUNTS)]
    fn masstree15(bencher: Bencher, threads: usize) {
        let keys = Arc::new(variable_length_keys_arc(N, MIN_KEY_SIZE, MAX_KEY_SIZE, 42));
        let indices = Arc::new(uniform_indices(N, OPS_PER_THREAD * threads, 42));
        let write_decisions: Arc<Vec<Vec<bool>>> = Arc::new(
            (0..threads)
                .map(|t| shuffled_write_decisions(OPS_PER_THREAD, WRITE_RATIO, 42 + t as u64))
                .collect(),
        );

        bencher
            .counter(divan::counter::ItemsCount::new(threads * OPS_PER_THREAD))
            .with_inputs(|| Arc::new(setup_masstree15_arc(keys.as_ref())))
            .bench_local_values(|tree| {
                let barrier = Arc::new(Barrier::new(threads));
                let handles: Vec<_> = (0..threads)
                    .map(|t| {
                        let tree = Arc::clone(&tree);
                        let keys = Arc::clone(&keys);
                        let indices = Arc::clone(&indices);
                        let write_decisions = Arc::clone(&write_decisions);
                        let barrier = Arc::clone(&barrier);
                        thread::spawn(move || {
                            let guard = tree.guard();
                            let base = t * OPS_PER_THREAD;
                            let is_write = &write_decisions[t];

                            // Warmup: mirror measurement op mix
                            for i in 0..WARMUP_OPS {
                                let idx = indices[base + i];
                                if is_write[i] {
                                    let _ = tree.insert_with_guard(
                                        keys[idx].as_ref(),
                                        i as u64,
                                        &guard,
                                    );
                                } else {
                                    black_box(tree.get_with_guard(keys[idx].as_ref(), &guard));
                                }
                            }

                            barrier.wait();

                            pre_measurement_barrier();
                            let mut sum = 0u64;
                            for i in 0..OPS_PER_THREAD {
                                let idx = indices[base + i];
                                if is_write[i] {
                                    let _ = tree.insert_with_guard(
                                        keys[idx].as_ref(),
                                        i as u64,
                                        &guard,
                                    );
                                } else if let Some(v) =
                                    tree.get_with_guard(keys[idx].as_ref(), &guard)
                                {
                                    sum = black_box(sum.wrapping_add(v));
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
