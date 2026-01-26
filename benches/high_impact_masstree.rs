//! High-impact benchmarks targeting Masstree's architectural advantages.
//!
//! ## Benchmark Groups
//!
//! | # | Name | Focus |
//! |---|------|-------|
//! | 01 | long_keys_128b | Suffix handling (unique prefixes) |
//! | 02 | multiple_hot_keys | Read-hot cache pattern (Zipfian) |
//! | 03 | mixed_get_insert_remove | Dynamic set with removes |
//! | 04 | variable_long_keys | API cost (Vec<u8> clones) |
//! | 05 | prefix_queries | Native scan_prefix vs range |
//! | 06 | deep_trie_traversal | Multi-layer descent (10% writes) |
//! | 07 | deep_trie_read_only | Multi-layer descent (pure reads) |
//! | 08 | variable_keys_arc | Structure-only (Arc<[u8]> keys) |
//!
//! ## Methodology Notes
//!
//! - **Warmup excluded from counters**: All benchmarks perform warmup iterations
//!   BEFORE the measurement barrier. Warmup ops are NOT included in `ItemsCount`.
//! - **Barrier synchronization**: All threads synchronize BEFORE the measurement
//!   barrier to ensure synchronization overhead is excluded from timing.
//! - **Thread spawn overhead**: Thread creation occurs inside `bench_local_values`
//!   due to divan API constraints. This adds ~10-50µs per thread per sample.
//!   For absolute numbers, consider this overhead; for relative comparisons
//!   between structures, it cancels out.
//! - **Compiler barriers**: Pre/post measurement barriers use `compiler_fence` only
//!   (no hardware fence) to minimize overhead while preventing reordering.
//! - Benchmark 04 includes `Vec<u8>` clone overhead; benchmark 08 uses `Arc<[u8]>`
//!   for structure-only comparison without allocator costs.
//! - Benchmark 05 counters measure scans/sec not records/sec (~500 records/scan).
//! - See per-benchmark comments for detailed methodology notes.
//!
//! ## Running
//!
//! ```bash
//! cargo bench --bench high_impact_masstree
//! cargo bench --bench high_impact_masstree -- 01_
//! cargo bench --bench high_impact_masstree -- 07_  # read-only deep traversal
//! ```

#![expect(clippy::unwrap_used)]
#![expect(clippy::pedantic)]
#![expect(clippy::indexing_slicing)]

mod bench_utils;

use bench_utils::{keys, post_measurement_barrier, pre_measurement_barrier, uniform_indices};
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
// - Long key comparison costs vs flat structures
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
            // Counter excludes warmup - only counts actual measured operations
            .counter(divan::counter::ItemsCount::new(threads * OPS_PER_THREAD))
            .with_inputs(|| Arc::new(setup_masstree15_long(keys.as_ref())))
            .bench_local_values(|tree| {
                let warmup_done = Arc::new(Barrier::new(threads));
                let measurement_start = Arc::new(Barrier::new(threads));
                let handles: Vec<_> = (0..threads)
                    .map(|t| {
                        let tree = Arc::clone(&tree);
                        let keys = Arc::clone(&keys);
                        let indices = Arc::clone(&indices);
                        let write_decisions = Arc::clone(&write_decisions);
                        let warmup_done = Arc::clone(&warmup_done);
                        let measurement_start = Arc::clone(&measurement_start);
                        thread::spawn(move || {
                            let guard = tree.guard();
                            let base = t * OPS_PER_THREAD;
                            let is_write = &write_decisions[t];

                            // Warmup phase (excluded from measurement)
                            for i in 0..WARMUP_OPS {
                                let idx = indices[base + i];
                                black_box(tree.get_with_guard(&keys[idx], &guard));
                            }

                            // Synchronize after warmup, BEFORE measurement barrier
                            warmup_done.wait();
                            measurement_start.wait();

                            // === MEASUREMENT REGION START ===
                            pre_measurement_barrier();
                            let mut sum = 0u64;
                            for i in 0..OPS_PER_THREAD {
                                let idx = indices[base + i];
                                if is_write[i] {
                                    let _ = tree.insert_with_guard(&keys[idx], i as u64, &guard);
                                } else if let Some(v) = tree.get_with_guard(&keys[idx], &guard) {
                                    sum = sum.wrapping_add(v);
                                }
                            }
                            post_measurement_barrier();
                            // === MEASUREMENT REGION END ===

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
// of keys receive the majority of accesses. This is more realistic than
// uniform random:
// - Caches often have a small working set
// - Database indices see power-law access patterns
// - Web servers have popular pages
//
// METHODOLOGY NOTES:
// - Uses Zipfian distribution with skew parameter s=1.0 (classic Zipf's law)
// - Top key gets ~19% of accesses, top 8 keys get ~50%
// - This is a READ-HOT benchmark with 10% writes
// - Uses unique-prefix keys (keys<128>), so comparisons short-circuit early

#[divan::bench_group(name = "02_multiple_hot_keys", sample_count = 200)]
mod multiple_hot_keys {
    use super::*;

    const N: usize = 50_000;
    const OPS_PER_THREAD: usize = 10_000;
    const WRITE_RATIO: usize = 10;

    /// Generate indices following Zipfian distribution.
    /// Rank r has probability proportional to 1/r^s where s is the skew.
    fn zipfian_indices(n: usize, count: usize, skew: f64, seed: u64) -> Vec<usize> {
        // Precompute CDF for Zipfian distribution
        let mut cdf = Vec::with_capacity(n);
        let mut sum = 0.0f64;
        for rank in 1..=n {
            sum += 1.0 / (rank as f64).powf(skew);
            cdf.push(sum);
        }
        // Normalize
        for x in cdf.iter_mut() {
            *x /= sum;
        }

        let mut indices = Vec::with_capacity(count);
        let mut state = seed;

        for _ in 0..count {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1);
            // Generate uniform random in [0, 1)
            let u = (state >> 11) as f64 / (1u64 << 53) as f64;

            // Binary search in CDF to find rank
            let rank = match cdf.binary_search_by(|x| x.partial_cmp(&u).unwrap()) {
                Ok(i) => i,
                Err(i) => i,
            };
            indices.push(rank.min(n - 1));
        }
        indices
    }

    #[divan::bench(args = THREAD_COUNTS)]
    fn masstree15(bencher: Bencher, threads: usize) {
        let keys = Arc::new(keys::<LONG_KEY_SIZE>(N));
        // Zipfian with s=1.0 (classic Zipf's law)
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
                let warmup_done = Arc::new(Barrier::new(threads));
                let measurement_start = Arc::new(Barrier::new(threads));
                let handles: Vec<_> = (0..threads)
                    .map(|t| {
                        let tree = Arc::clone(&tree);
                        let keys = Arc::clone(&keys);
                        let indices = Arc::clone(&indices);
                        let write_decisions = Arc::clone(&write_decisions);
                        let warmup_done = Arc::clone(&warmup_done);
                        let measurement_start = Arc::clone(&measurement_start);
                        thread::spawn(move || {
                            let guard = tree.guard();
                            let base = t * OPS_PER_THREAD;
                            let is_write = &write_decisions[t];

                            // Warmup phase
                            for i in 0..WARMUP_OPS {
                                let idx = indices[base + i];
                                black_box(tree.get_with_guard(&keys[idx], &guard));
                            }

                            warmup_done.wait();
                            measurement_start.wait();

                            pre_measurement_barrier();
                            let mut sum = 0u64;
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
// 03: MIXED GET/INSERT/REMOVE - Full Operation Mix (Dynamic Set)
// =============================================================================
//
// Tests all three operations with realistic distribution:
// - 70% get (reads)
// - 20% insert (upserts)
// - 10% remove (deletes)
//
// This stresses memory reclamation and tests the full API surface.
//
// METHODOLOGY NOTES:
//
// 1. State drift: Removes and inserts operate on the SAME keyspace, so:
//    - Removes can become "remove-miss" once a key was already deleted
//    - Inserts can become "update-existing" for structures with upsert semantics
//    - Set size can drift and the effective op mix changes over the run
//
//    This is intentional - it represents a dynamic working set. The tree is
//    recreated each sample via `with_inputs`, ensuring consistent starting state.

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
                let warmup_done = Arc::new(Barrier::new(threads));
                let measurement_start = Arc::new(Barrier::new(threads));
                let handles: Vec<_> = (0..threads)
                    .map(|t| {
                        let tree = Arc::clone(&tree);
                        let keys = Arc::clone(&keys);
                        let indices = Arc::clone(&indices);
                        let op_decisions = Arc::clone(&op_decisions);
                        let warmup_done = Arc::clone(&warmup_done);
                        let measurement_start = Arc::clone(&measurement_start);
                        thread::spawn(move || {
                            let guard = tree.guard();
                            let base = t * OPS_PER_THREAD;
                            let ops = &op_decisions[t];

                            // Warmup phase
                            for i in 0..WARMUP_OPS {
                                let idx = indices[base + i];
                                black_box(tree.get_with_guard(&keys[idx], &guard));
                            }

                            warmup_done.wait();
                            measurement_start.wait();

                            pre_measurement_barrier();
                            let mut sum = 0u64;
                            for i in 0..OPS_PER_THREAD {
                                let idx = indices[base + i];
                                match ops[i] {
                                    Op::Get => {
                                        if let Some(v) = tree.get_with_guard(&keys[idx], &guard) {
                                            sum = sum.wrapping_add(v);
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
// 04: VARIABLE LENGTH KEYS (64-256 bytes) - Realistic Key Sizes + API Cost
// =============================================================================
//
// Real-world keys vary in length: URLs, file paths, composite keys.
// This tests how structures handle variable-length keys from 64 to 256 bytes.
//
// METHODOLOGY NOTE: This benchmark includes key ownership/cloning costs:
// - Masstree: passes `&[u8]` slices directly, NO clone on write
// - Others: must `clone()` the `Vec<u8>` on every write (~64-256 byte alloc)
//
// This is intentional - it reflects the real API cost of using these structures
// with owned variable-length keys. For structure-only comparison with minimized
// cloning overhead, see benchmark 08 (variable_keys_arc) which uses Arc<[u8]>.

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
            // Determine key length
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1);
            let len = min + ((state >> 32) as usize) % (max - min + 1);

            // Fill key with deterministic bytes
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
                let warmup_done = Arc::new(Barrier::new(threads));
                let measurement_start = Arc::new(Barrier::new(threads));
                let handles: Vec<_> = (0..threads)
                    .map(|t| {
                        let tree = Arc::clone(&tree);
                        let keys = Arc::clone(&keys);
                        let indices = Arc::clone(&indices);
                        let write_decisions = Arc::clone(&write_decisions);
                        let warmup_done = Arc::clone(&warmup_done);
                        let measurement_start = Arc::clone(&measurement_start);
                        thread::spawn(move || {
                            let guard = tree.guard();
                            let base = t * OPS_PER_THREAD;
                            let is_write = &write_decisions[t];

                            // Warmup phase
                            for i in 0..WARMUP_OPS {
                                let idx = indices[base + i];
                                black_box(tree.get_with_guard(keys[idx].as_slice(), &guard));
                            }

                            warmup_done.wait();
                            measurement_start.wait();

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
// 05: PREFIX QUERIES - Trie Natural Advantage
// =============================================================================
//
// Tests prefix scan performance where Masstree's trie structure should excel.
// Keys share common prefixes (e.g., "/users/123/...", "/users/456/...").
// Each thread scans all keys matching a randomly selected prefix.
//
// Masstree uses native `scan_prefix()` which navigates directly to the prefix
// subtrie. Other structures use range bounds (start_key..end_key) which is
// efficient but requires computing the lexicographic successor of the prefix.
//
// METHODOLOGY NOTES:
// - Counter measures SCANS/SEC, not records/sec. Each scan visits ~N/PREFIX_BUCKETS
//   keys (~500 keys with current settings). To convert to records/sec, multiply
//   by ~500.
// - Range bounds are PRECOMPUTED to avoid in-loop allocation overhead for
//   non-Masstree structures.

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
            // Counter excludes warmup scans
            .counter(divan::counter::ItemsCount::new(threads * SCANS_PER_THREAD))
            .with_inputs(|| Arc::new(setup_masstree15_prefix(keys.as_ref())))
            .bench_local_values(|tree| {
                let warmup_done = Arc::new(Barrier::new(threads));
                let measurement_start = Arc::new(Barrier::new(threads));
                let handles: Vec<_> = (0..threads)
                    .map(|t| {
                        let tree = Arc::clone(&tree);
                        let prefixes = Arc::clone(&prefixes);
                        let warmup_done = Arc::clone(&warmup_done);
                        let measurement_start = Arc::clone(&measurement_start);
                        thread::spawn(move || {
                            let guard = tree.guard();
                            let base = t * SCANS_PER_THREAD;

                            // Warmup phase
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

                            warmup_done.wait();
                            measurement_start.wait();

                            pre_measurement_barrier();
                            let mut total = 0u64;
                            for i in 0..SCANS_PER_THREAD {
                                let prefix = &prefixes[base + i];
                                tree.scan_prefix(
                                    prefix,
                                    |_, v| {
                                        total = total.wrapping_add(v);
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
// Unlike 01_long_keys_128b which has unique first chunks, this benchmark uses
// keys_shared_prefix_chunks to force collisions in the FIRST 4 trie layers.
// With prefix_buckets=16, we get ~3125 keys per bucket sharing 4 initial chunks.
//
// This actually tests Masstree's multi-layer trie traversal and demonstrates
// the architectural advantage of prefix sharing across layers.
//
// METHODOLOGY NOTES:
// - This is a mixed read/write workload (10% writes). For pure traversal cost
//   without write-path differences, see benchmark 07 (deep_trie_read_only).

#[divan::bench_group(name = "06_deep_trie_traversal", sample_count = 200)]
mod deep_trie {
    use super::*;
    use bench_utils::keys_shared_prefix_chunks;

    const N: usize = 50_000;
    const OPS_PER_THREAD: usize = 12_500;
    const WRITE_RATIO: usize = 10;
    // 4 shared prefix chunks = 32 bytes of shared prefix
    // 16 buckets = ~3125 keys per bucket, forcing deep trie descent
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
                let warmup_done = Arc::new(Barrier::new(threads));
                let measurement_start = Arc::new(Barrier::new(threads));
                let handles: Vec<_> = (0..threads)
                    .map(|t| {
                        let tree = Arc::clone(&tree);
                        let keys = Arc::clone(&keys);
                        let indices = Arc::clone(&indices);
                        let write_decisions = Arc::clone(&write_decisions);
                        let warmup_done = Arc::clone(&warmup_done);
                        let measurement_start = Arc::clone(&measurement_start);
                        thread::spawn(move || {
                            let guard = tree.guard();
                            let base = t * OPS_PER_THREAD;
                            let is_write = &write_decisions[t];

                            // Warmup phase
                            for i in 0..WARMUP_OPS {
                                let idx = indices[base + i];
                                black_box(tree.get_with_guard(&keys[idx], &guard));
                            }

                            warmup_done.wait();
                            measurement_start.wait();

                            pre_measurement_barrier();
                            let mut sum = 0u64;
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
// 07: DEEP TRIE READ-ONLY - Isolate Traversal Cost from Write Overhead
// =============================================================================
//
// Pure read workload variant of benchmark 06. This isolates the multi-layer
// trie traversal cost from write-path differences (allocator behavior, etc.).
//
// Use this to measure the pure lookup/comparison cost when keys share common
// prefixes and force deep trie descent.

#[divan::bench_group(name = "07_deep_trie_read_only", sample_count = 200)]
mod deep_trie_read_only {
    use super::*;
    use bench_utils::keys_shared_prefix_chunks;

    const N: usize = 50_000;
    const OPS_PER_THREAD: usize = 25_000; // More ops since read-only is faster
                                          // Same prefix configuration as benchmark 06
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
                let warmup_done = Arc::new(Barrier::new(threads));
                let measurement_start = Arc::new(Barrier::new(threads));
                let handles: Vec<_> = (0..threads)
                    .map(|t| {
                        let tree = Arc::clone(&tree);
                        let keys = Arc::clone(&keys);
                        let indices = Arc::clone(&indices);
                        let warmup_done = Arc::clone(&warmup_done);
                        let measurement_start = Arc::clone(&measurement_start);
                        thread::spawn(move || {
                            let guard = tree.guard();
                            let base = t * OPS_PER_THREAD;

                            // Warmup phase
                            for i in 0..WARMUP_OPS {
                                let idx = indices[base + i];
                                black_box(tree.get_with_guard(&keys[idx], &guard));
                            }

                            warmup_done.wait();
                            measurement_start.wait();

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
// 08: VARIABLE LENGTH KEYS WITH ARC - Structure-Only Comparison
// =============================================================================
//
// This benchmark uses Arc<[u8]> keys instead of Vec<u8> to minimize per-write
// allocation overhead. All structures clone the same cheap Arc reference on
// writes, making this a fairer structure-only comparison.
//
// Compare results with benchmark 04 to see how much of the difference was due
// to API ownership costs (Vec<u8> clone) vs actual structure ordering behavior.

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
            // Determine key length
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1);
            let len = min + ((state >> 32) as usize) % (max - min + 1);

            // Fill key with deterministic bytes
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
                let warmup_done = Arc::new(Barrier::new(threads));
                let measurement_start = Arc::new(Barrier::new(threads));
                let handles: Vec<_> = (0..threads)
                    .map(|t| {
                        let tree = Arc::clone(&tree);
                        let keys = Arc::clone(&keys);
                        let indices = Arc::clone(&indices);
                        let write_decisions = Arc::clone(&write_decisions);
                        let warmup_done = Arc::clone(&warmup_done);
                        let measurement_start = Arc::clone(&measurement_start);
                        thread::spawn(move || {
                            let guard = tree.guard();
                            let base = t * OPS_PER_THREAD;
                            let is_write = &write_decisions[t];

                            // Warmup phase
                            for i in 0..WARMUP_OPS {
                                let idx = indices[base + i];
                                black_box(tree.get_with_guard(keys[idx].as_ref(), &guard));
                            }

                            warmup_done.wait();
                            measurement_start.wait();

                            pre_measurement_barrier();
                            let mut sum = 0u64;
                            for i in 0..OPS_PER_THREAD {
                                let idx = indices[base + i];
                                if is_write[i] {
                                    // Masstree still takes &[u8], no clone needed
                                    let _ = tree.insert_with_guard(
                                        keys[idx].as_ref(),
                                        i as u64,
                                        &guard,
                                    );
                                } else if let Some(v) =
                                    tree.get_with_guard(keys[idx].as_ref(), &guard)
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
