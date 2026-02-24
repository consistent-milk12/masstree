//! High-impact benchmarks targeting Masstree's architectural advantages.
//!
//! ## Benchmark Groups
//!
//! | # | Name | Focus |
//! |---|------|-------|
//! | 01 | long_keys_128b | Suffix handling (unique prefixes) |
//! | 02 | multiple_hot_keys | Read-hot cache pattern |
//! | 03 | mixed_get_insert_remove | Dynamic set with removes |
//! | 04 | variable_long_keys | API cost (Vec<u8> clones) |
//! | 05 | prefix_queries | Native scan_prefix vs range |
//! | 06 | deep_trie_traversal | Multi-layer descent (10% writes) |
//! | 07 | deep_trie_read_only | Multi-layer descent (pure reads) |
//! | 08 | variable_keys_arc | Structure-only (Arc<[u8]> keys) |
//! | 09 | prefix_realistic_mixed | Hierarchical prefixes + pagination + writes |
//!
//! ## Methodology Notes
//!
//! - **Warmup excluded from counters**: Warmup iterations happen before measurement
//!   starts (before `start.wait()`) and are NOT included in the `ItemsCount` counter.
//!   This ensures accurate ops/sec reporting for measured operations only.
//! - **Compiler barriers**: Pre/post measurement barriers use `compiler_fence` only
//!   (no hardware fences) to minimize overhead.
//! - TreeIndex writes use `remove_sync()+insert_sync()` fallback since it lacks
//!   upsert semantics. TreeIndex upsert helpers avoid extra cloning by reusing the
//!   key returned on insert failure.
//! - Benchmark 04 includes `Vec<u8>` clone overhead; benchmark 08 uses `Arc<[u8]>`
//!   for structure-only comparison without allocator costs.
//! - Benchmark 05 counters measure scans/sec not records/sec (~500 records/scan).
//! - Benchmark 09 counters measure mixed operations/sec (scan/read/write mix).
//! - See per-benchmark comments for detailed methodology notes.
//!
//! ## Running
//!
//! ```bash
//! cargo bench --bench high_impact
//! cargo bench --bench high_impact -- 01_
//! cargo bench --bench high_impact -- 07_  # read-only deep traversal
//! ```

#![expect(clippy::unwrap_used)]
#![expect(clippy::pedantic)]
#![expect(clippy::indexing_slicing)]

mod bench_utils;

use bench_utils::{keys, post_measurement_barrier, pre_measurement_barrier, uniform_indices};
use crossbeam_skiplist::SkipMap;
use divan::{Bencher, black_box};
use indexset::concurrent::map::BTreeMap as IndexSetBTreeMap;
use masstree::MassTree15Inline;
use scc::TreeIndex;
use sdd::Guard as SddGuard;
use std::sync::Arc;
use std::sync::Barrier;
use std::thread;

fn main() {
    divan::main();
}

// =============================================================================
// Constants
// =============================================================================

/// Warmup iterations per thread (included in timing and counter)
const WARMUP_OPS: usize = 500;

/// Standard thread counts for all benchmarks
const THREAD_COUNTS: [usize; 6] = [1, 2, 4, 6, 8, 12];

// =============================================================================
// Setup Helpers for 128-byte keys
// =============================================================================

const LONG_KEY_SIZE: usize = 128;

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

fn setup_skipmap_long(keys: &[[u8; LONG_KEY_SIZE]]) -> SkipMap<[u8; LONG_KEY_SIZE], u64> {
    let map = SkipMap::new();
    for (i, key) in keys.iter().enumerate() {
        map.insert(*key, i as u64);
    }
    map
}

fn setup_indexset_long(keys: &[[u8; LONG_KEY_SIZE]]) -> IndexSetBTreeMap<[u8; LONG_KEY_SIZE], u64> {
    let map = IndexSetBTreeMap::new();
    for (i, key) in keys.iter().enumerate() {
        map.insert(*key, i as u64);
    }
    map
}

fn setup_tree_index_long(keys: &[[u8; LONG_KEY_SIZE]]) -> TreeIndex<[u8; LONG_KEY_SIZE], u64> {
    let tree = TreeIndex::new();
    for (i, key) in keys.iter().enumerate() {
        let _ = tree.insert_sync(*key, i as u64);
    }
    tree
}

fn tree_index_upsert_sync_long(
    tree: &TreeIndex<[u8; LONG_KEY_SIZE], u64>,
    key: [u8; LONG_KEY_SIZE],
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
            .counter(divan::counter::ItemsCount::new(threads * OPS_PER_THREAD))
            .with_inputs(|| Arc::new(setup_masstree15_long(keys.as_ref())))
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
                                let idx = indices[base + (i % OPS_PER_THREAD)];
                                black_box(tree.get_with_guard(&keys[idx], &guard));
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

    #[divan::bench(args = THREAD_COUNTS)]
    fn skipmap(bencher: Bencher, threads: usize) {
        let keys = Arc::new(keys::<LONG_KEY_SIZE>(N));
        let indices = Arc::new(uniform_indices(N, OPS_PER_THREAD * threads, 42));
        let write_decisions: Arc<Vec<Vec<bool>>> = Arc::new(
            (0..threads)
                .map(|t| shuffled_write_decisions(OPS_PER_THREAD, WRITE_RATIO, 42 + t as u64))
                .collect(),
        );

        bencher
            .counter(divan::counter::ItemsCount::new(threads * OPS_PER_THREAD))
            .with_inputs(|| Arc::new(setup_skipmap_long(keys.as_ref())))
            .bench_local_values(|map| {
                let warmup_done = Arc::new(Barrier::new(threads));
                let start = Arc::new(Barrier::new(threads));
                let handles: Vec<_> = (0..threads)
                    .map(|t| {
                        let map = Arc::clone(&map);
                        let keys = Arc::clone(&keys);
                        let indices = Arc::clone(&indices);
                        let write_decisions = Arc::clone(&write_decisions);
                        let warmup_done = Arc::clone(&warmup_done);
                        let start = Arc::clone(&start);
                        thread::spawn(move || {
                            let base = t * OPS_PER_THREAD;
                            let is_write = &write_decisions[t];

                            for i in 0..WARMUP_OPS {
                                let idx = indices[base + (i % OPS_PER_THREAD)];
                                black_box(map.get(&keys[idx]));
                            }
                            warmup_done.wait();
                            start.wait();
                            pre_measurement_barrier();
                            let mut sum = 0u64;
                            for i in 0..OPS_PER_THREAD {
                                let idx = indices[base + i];
                                if is_write[i] {
                                    black_box(map.insert(keys[idx], i as u64));
                                } else if let Some(e) = map.get(&keys[idx]) {
                                    sum = sum.wrapping_add(*e.value());
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
    fn indexset(bencher: Bencher, threads: usize) {
        let keys = Arc::new(keys::<LONG_KEY_SIZE>(N));
        let indices = Arc::new(uniform_indices(N, OPS_PER_THREAD * threads, 42));
        let write_decisions: Arc<Vec<Vec<bool>>> = Arc::new(
            (0..threads)
                .map(|t| shuffled_write_decisions(OPS_PER_THREAD, WRITE_RATIO, 42 + t as u64))
                .collect(),
        );

        bencher
            .counter(divan::counter::ItemsCount::new(threads * OPS_PER_THREAD))
            .with_inputs(|| Arc::new(setup_indexset_long(keys.as_ref())))
            .bench_local_values(|map| {
                let warmup_done = Arc::new(Barrier::new(threads));
                let start = Arc::new(Barrier::new(threads));
                let handles: Vec<_> = (0..threads)
                    .map(|t| {
                        let map = Arc::clone(&map);
                        let keys = Arc::clone(&keys);
                        let indices = Arc::clone(&indices);
                        let write_decisions = Arc::clone(&write_decisions);
                        let warmup_done = Arc::clone(&warmup_done);
                        let start = Arc::clone(&start);
                        thread::spawn(move || {
                            let base = t * OPS_PER_THREAD;
                            let is_write = &write_decisions[t];

                            for i in 0..WARMUP_OPS {
                                let idx = indices[base + (i % OPS_PER_THREAD)];
                                black_box(map.get(&keys[idx]));
                            }
                            warmup_done.wait();
                            start.wait();
                            pre_measurement_barrier();
                            let mut sum = 0u64;
                            for i in 0..OPS_PER_THREAD {
                                let idx = indices[base + i];
                                if is_write[i] {
                                    black_box(map.insert(keys[idx], i as u64));
                                } else if let Some(r) = map.get(&keys[idx]) {
                                    sum = sum.wrapping_add(r.get().value);
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
    fn tree_index(bencher: Bencher, threads: usize) {
        let keys = Arc::new(keys::<LONG_KEY_SIZE>(N));
        let indices = Arc::new(uniform_indices(N, OPS_PER_THREAD * threads, 42));
        let write_decisions: Arc<Vec<Vec<bool>>> = Arc::new(
            (0..threads)
                .map(|t| shuffled_write_decisions(OPS_PER_THREAD, WRITE_RATIO, 42 + t as u64))
                .collect(),
        );

        bencher
            .counter(divan::counter::ItemsCount::new(threads * OPS_PER_THREAD))
            .with_inputs(|| Arc::new(setup_tree_index_long(keys.as_ref())))
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
                            let guard = SddGuard::new();
                            let base = t * OPS_PER_THREAD;
                            let is_write = &write_decisions[t];

                            for i in 0..WARMUP_OPS {
                                let idx = indices[base + (i % OPS_PER_THREAD)];
                                black_box(tree.peek(&keys[idx], &guard));
                            }
                            warmup_done.wait();
                            start.wait();
                            pre_measurement_barrier();
                            let mut sum = 0u64;
                            for i in 0..OPS_PER_THREAD {
                                let idx = indices[base + i];
                                if is_write[i] {
                                    tree_index_upsert_sync_long(&tree, keys[idx], i as u64);
                                    black_box(());
                                } else if let Some(v) = tree.peek(&keys[idx], &guard) {
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
}

// =============================================================================
// 02: MULTIPLE HOT KEYS - Realistic Cache Pattern (READ-HOT)
// =============================================================================
//
// Tests 8 hot keys that receive 80% of accesses (10% each), while remaining
// keys share 20%. This is more realistic than single-hot-key or uniform:
// - Caches often have a small working set
// - Database indices see power-law access patterns
// - Web servers have popular pages
//
// METHODOLOGY NOTES:
// - This is a READ-HOT benchmark: warmup is read-only, measuring read-dominated
//   workloads where caching effects matter.
// - Hot keys are SPREAD across the keyspace (not clustered), which tests cache
//   working-set behavior but NOT contention on shared structure nodes. For
//   contention stress testing, use benchmark 06 (deep_trie_traversal) where
//   many keys share common prefixes and compete for the same internal nodes.
// - Uses unique-prefix keys (keys<128>), so comparisons short-circuit early.

#[divan::bench_group(name = "02_multiple_hot_keys", sample_count = 200)]
mod multiple_hot_keys {
    use super::*;

    const N: usize = 50_000;
    const OPS_PER_THREAD: usize = 10_000;
    const WRITE_RATIO: usize = 10;
    const NUM_HOT_KEYS: usize = 8;
    const HOT_KEY_PROBABILITY: usize = 80; // 80% of accesses go to hot keys

    /// Generate indices where 80% hit one of 8 hot keys, 20% are uniform
    fn hot_key_indices(
        n: usize,
        count: usize,
        num_hot: usize,
        hot_prob: usize,
        seed: u64,
    ) -> Vec<usize> {
        let mut indices = Vec::with_capacity(count);
        let mut state = seed;

        // Pick hot key positions spread across the keyspace
        let hot_keys: Vec<usize> = (0..num_hot)
            .map(|i| (n / num_hot) * i + n / (2 * num_hot))
            .collect();

        for _ in 0..count {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1);
            let is_hot = ((state >> 32) as usize) % 100 < hot_prob;

            if is_hot {
                // Pick one of the hot keys
                let hot_idx = ((state >> 48) as usize) % num_hot;
                indices.push(hot_keys[hot_idx]);
            } else {
                // Uniform random
                indices.push(((state >> 32) as usize) % n);
            }
        }
        indices
    }

    #[divan::bench(args = THREAD_COUNTS)]
    fn masstree15(bencher: Bencher, threads: usize) {
        let keys = Arc::new(keys::<LONG_KEY_SIZE>(N));
        let indices = Arc::new(hot_key_indices(
            N,
            OPS_PER_THREAD * threads,
            NUM_HOT_KEYS,
            HOT_KEY_PROBABILITY,
            42,
        ));
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
                                let idx = indices[base + (i % OPS_PER_THREAD)];
                                black_box(tree.get_with_guard(&keys[idx], &guard));
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

    #[divan::bench(args = THREAD_COUNTS)]
    fn skipmap(bencher: Bencher, threads: usize) {
        let keys = Arc::new(keys::<LONG_KEY_SIZE>(N));
        let indices = Arc::new(hot_key_indices(
            N,
            OPS_PER_THREAD * threads,
            NUM_HOT_KEYS,
            HOT_KEY_PROBABILITY,
            42,
        ));
        let write_decisions: Arc<Vec<Vec<bool>>> = Arc::new(
            (0..threads)
                .map(|t| shuffled_write_decisions(OPS_PER_THREAD, WRITE_RATIO, 42 + t as u64))
                .collect(),
        );

        bencher
            .counter(divan::counter::ItemsCount::new(threads * OPS_PER_THREAD))
            .with_inputs(|| Arc::new(setup_skipmap_long(keys.as_ref())))
            .bench_local_values(|map| {
                let warmup_done = Arc::new(Barrier::new(threads));
                let start = Arc::new(Barrier::new(threads));
                let handles: Vec<_> = (0..threads)
                    .map(|t| {
                        let map = Arc::clone(&map);
                        let keys = Arc::clone(&keys);
                        let indices = Arc::clone(&indices);
                        let write_decisions = Arc::clone(&write_decisions);
                        let warmup_done = Arc::clone(&warmup_done);
                        let start = Arc::clone(&start);
                        thread::spawn(move || {
                            let base = t * OPS_PER_THREAD;
                            let is_write = &write_decisions[t];

                            for i in 0..WARMUP_OPS {
                                let idx = indices[base + (i % OPS_PER_THREAD)];
                                black_box(map.get(&keys[idx]));
                            }
                            warmup_done.wait();
                            start.wait();
                            pre_measurement_barrier();
                            let mut sum = 0u64;
                            for i in 0..OPS_PER_THREAD {
                                let idx = indices[base + i];
                                if is_write[i] {
                                    black_box(map.insert(keys[idx], i as u64));
                                } else if let Some(e) = map.get(&keys[idx]) {
                                    sum = sum.wrapping_add(*e.value());
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
    fn indexset(bencher: Bencher, threads: usize) {
        let keys = Arc::new(keys::<LONG_KEY_SIZE>(N));
        let indices = Arc::new(hot_key_indices(
            N,
            OPS_PER_THREAD * threads,
            NUM_HOT_KEYS,
            HOT_KEY_PROBABILITY,
            42,
        ));
        let write_decisions: Arc<Vec<Vec<bool>>> = Arc::new(
            (0..threads)
                .map(|t| shuffled_write_decisions(OPS_PER_THREAD, WRITE_RATIO, 42 + t as u64))
                .collect(),
        );

        bencher
            .counter(divan::counter::ItemsCount::new(threads * OPS_PER_THREAD))
            .with_inputs(|| Arc::new(setup_indexset_long(keys.as_ref())))
            .bench_local_values(|map| {
                let warmup_done = Arc::new(Barrier::new(threads));
                let start = Arc::new(Barrier::new(threads));
                let handles: Vec<_> = (0..threads)
                    .map(|t| {
                        let map = Arc::clone(&map);
                        let keys = Arc::clone(&keys);
                        let indices = Arc::clone(&indices);
                        let write_decisions = Arc::clone(&write_decisions);
                        let warmup_done = Arc::clone(&warmup_done);
                        let start = Arc::clone(&start);
                        thread::spawn(move || {
                            let base = t * OPS_PER_THREAD;
                            let is_write = &write_decisions[t];

                            for i in 0..WARMUP_OPS {
                                let idx = indices[base + (i % OPS_PER_THREAD)];
                                black_box(map.get(&keys[idx]));
                            }
                            warmup_done.wait();
                            start.wait();
                            pre_measurement_barrier();
                            let mut sum = 0u64;
                            for i in 0..OPS_PER_THREAD {
                                let idx = indices[base + i];
                                if is_write[i] {
                                    black_box(map.insert(keys[idx], i as u64));
                                } else if let Some(r) = map.get(&keys[idx]) {
                                    sum = sum.wrapping_add(r.get().value);
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
    fn tree_index(bencher: Bencher, threads: usize) {
        let keys = Arc::new(keys::<LONG_KEY_SIZE>(N));
        let indices = Arc::new(hot_key_indices(
            N,
            OPS_PER_THREAD * threads,
            NUM_HOT_KEYS,
            HOT_KEY_PROBABILITY,
            42,
        ));
        let write_decisions: Arc<Vec<Vec<bool>>> = Arc::new(
            (0..threads)
                .map(|t| shuffled_write_decisions(OPS_PER_THREAD, WRITE_RATIO, 42 + t as u64))
                .collect(),
        );

        bencher
            .counter(divan::counter::ItemsCount::new(threads * OPS_PER_THREAD))
            .with_inputs(|| Arc::new(setup_tree_index_long(keys.as_ref())))
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
                            let guard = SddGuard::new();
                            let base = t * OPS_PER_THREAD;
                            let is_write = &write_decisions[t];

                            for i in 0..WARMUP_OPS {
                                let idx = indices[base + (i % OPS_PER_THREAD)];
                                black_box(tree.peek(&keys[idx], &guard));
                            }
                            warmup_done.wait();
                            start.wait();
                            pre_measurement_barrier();
                            let mut sum = 0u64;
                            for i in 0..OPS_PER_THREAD {
                                let idx = indices[base + i];
                                if is_write[i] {
                                    tree_index_upsert_sync_long(&tree, keys[idx], i as u64);
                                    black_box(());
                                } else if let Some(v) = tree.peek(&keys[idx], &guard) {
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
// 1. TreeIndex upsert overhead: TreeIndex lacks native upsert, so writes use a
//    `remove_sync()+insert_sync()` fallback. This adds overhead compared to
//    single-call upserts in other structures.
//
// 2. State drift: Removes and inserts operate on the SAME keyspace, so:
//    - Removes can become "remove-miss" once a key was already deleted
//    - Inserts can become "update-existing" for structures with upsert semantics
//      (but remain "remove+insert" for TreeIndex)
//    - Set size can drift and the effective op mix changes over the run
//
//    This is intentional - it represents a dynamic working set. For controlled
//    semantics (insert-new-only, remove-existing-only), use separate keyspaces.

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
                            let ops = &op_decisions[t];

                            for i in 0..WARMUP_OPS {
                                let idx = indices[base + (i % OPS_PER_THREAD)];
                                black_box(tree.get_with_guard(&keys[idx], &guard));
                            }
                            warmup_done.wait();
                            start.wait();
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
                                        black_box(
                                            tree.insert_with_guard(&keys[idx], i as u64, &guard),
                                        );
                                    }
                                    Op::Remove => {
                                        let _ =
                                            black_box(tree.remove_with_guard(&keys[idx], &guard));
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

    #[divan::bench(args = THREAD_COUNTS)]
    fn skipmap(bencher: Bencher, threads: usize) {
        let keys = Arc::new(keys::<LONG_KEY_SIZE>(N));
        let indices = Arc::new(uniform_indices(N, OPS_PER_THREAD * threads, 42));
        let op_decisions: Arc<Vec<Vec<Op>>> = Arc::new(
            (0..threads)
                .map(|t| shuffled_op_decisions(OPS_PER_THREAD, 42 + t as u64))
                .collect(),
        );

        bencher
            .counter(divan::counter::ItemsCount::new(threads * OPS_PER_THREAD))
            .with_inputs(|| Arc::new(setup_skipmap_long(keys.as_ref())))
            .bench_local_values(|map| {
                let warmup_done = Arc::new(Barrier::new(threads));
                let start = Arc::new(Barrier::new(threads));
                let handles: Vec<_> = (0..threads)
                    .map(|t| {
                        let map = Arc::clone(&map);
                        let keys = Arc::clone(&keys);
                        let indices = Arc::clone(&indices);
                        let op_decisions = Arc::clone(&op_decisions);
                        let warmup_done = Arc::clone(&warmup_done);
                        let start = Arc::clone(&start);
                        thread::spawn(move || {
                            let base = t * OPS_PER_THREAD;
                            let ops = &op_decisions[t];

                            for i in 0..WARMUP_OPS {
                                let idx = indices[base + (i % OPS_PER_THREAD)];
                                black_box(map.get(&keys[idx]));
                            }
                            warmup_done.wait();
                            start.wait();
                            pre_measurement_barrier();
                            let mut sum = 0u64;
                            for i in 0..OPS_PER_THREAD {
                                let idx = indices[base + i];
                                match ops[i] {
                                    Op::Get => {
                                        if let Some(e) = map.get(&keys[idx]) {
                                            sum = sum.wrapping_add(*e.value());
                                        }
                                    }
                                    Op::Insert => {
                                        black_box(map.insert(keys[idx], i as u64));
                                    }
                                    Op::Remove => {
                                        black_box(map.remove(&keys[idx]));
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

    #[divan::bench(args = THREAD_COUNTS)]
    fn indexset(bencher: Bencher, threads: usize) {
        let keys = Arc::new(keys::<LONG_KEY_SIZE>(N));
        let indices = Arc::new(uniform_indices(N, OPS_PER_THREAD * threads, 42));
        let op_decisions: Arc<Vec<Vec<Op>>> = Arc::new(
            (0..threads)
                .map(|t| shuffled_op_decisions(OPS_PER_THREAD, 42 + t as u64))
                .collect(),
        );

        bencher
            .counter(divan::counter::ItemsCount::new(threads * OPS_PER_THREAD))
            .with_inputs(|| Arc::new(setup_indexset_long(keys.as_ref())))
            .bench_local_values(|map| {
                let warmup_done = Arc::new(Barrier::new(threads));
                let start = Arc::new(Barrier::new(threads));
                let handles: Vec<_> = (0..threads)
                    .map(|t| {
                        let map = Arc::clone(&map);
                        let keys = Arc::clone(&keys);
                        let indices = Arc::clone(&indices);
                        let op_decisions = Arc::clone(&op_decisions);
                        let warmup_done = Arc::clone(&warmup_done);
                        let start = Arc::clone(&start);
                        thread::spawn(move || {
                            let base = t * OPS_PER_THREAD;
                            let ops = &op_decisions[t];

                            for i in 0..WARMUP_OPS {
                                let idx = indices[base + (i % OPS_PER_THREAD)];
                                black_box(map.get(&keys[idx]));
                            }
                            warmup_done.wait();
                            start.wait();
                            pre_measurement_barrier();
                            let mut sum = 0u64;
                            for i in 0..OPS_PER_THREAD {
                                let idx = indices[base + i];
                                match ops[i] {
                                    Op::Get => {
                                        if let Some(r) = map.get(&keys[idx]) {
                                            sum = sum.wrapping_add(r.get().value);
                                        }
                                    }
                                    Op::Insert => {
                                        black_box(map.insert(keys[idx], i as u64));
                                    }
                                    Op::Remove => {
                                        black_box(map.remove(&keys[idx]));
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

    #[divan::bench(args = THREAD_COUNTS)]
    fn tree_index(bencher: Bencher, threads: usize) {
        let keys = Arc::new(keys::<LONG_KEY_SIZE>(N));
        let indices = Arc::new(uniform_indices(N, OPS_PER_THREAD * threads, 42));
        let op_decisions: Arc<Vec<Vec<Op>>> = Arc::new(
            (0..threads)
                .map(|t| shuffled_op_decisions(OPS_PER_THREAD, 42 + t as u64))
                .collect(),
        );

        bencher
            .counter(divan::counter::ItemsCount::new(threads * OPS_PER_THREAD))
            .with_inputs(|| Arc::new(setup_tree_index_long(keys.as_ref())))
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
                            let guard = SddGuard::new();
                            let base = t * OPS_PER_THREAD;
                            let ops = &op_decisions[t];

                            for i in 0..WARMUP_OPS {
                                let idx = indices[base + (i % OPS_PER_THREAD)];
                                black_box(tree.peek(&keys[idx], &guard));
                            }
                            warmup_done.wait();
                            start.wait();
                            pre_measurement_barrier();
                            let mut sum = 0u64;
                            for i in 0..OPS_PER_THREAD {
                                let idx = indices[base + i];
                                match ops[i] {
                                    Op::Get => {
                                        if let Some(v) = tree.peek(&keys[idx], &guard) {
                                            sum = sum.wrapping_add(*v);
                                        }
                                    }
                                    Op::Insert => {
                                        tree_index_upsert_sync_long(&tree, keys[idx], i as u64);
                                        black_box(());
                                    }
                                    Op::Remove => {
                                        black_box(tree.remove_sync(&keys[idx]));
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

    fn setup_skipmap_var(keys: &[Vec<u8>]) -> SkipMap<Vec<u8>, u64> {
        let map = SkipMap::new();
        for (i, key) in keys.iter().enumerate() {
            map.insert(key.clone(), i as u64);
        }
        map
    }

    fn setup_indexset_var(keys: &[Vec<u8>]) -> IndexSetBTreeMap<Vec<u8>, u64> {
        let map = IndexSetBTreeMap::new();
        for (i, key) in keys.iter().enumerate() {
            map.insert(key.clone(), i as u64);
        }
        map
    }

    fn setup_tree_index_var(keys: &[Vec<u8>]) -> TreeIndex<Vec<u8>, u64> {
        let tree = TreeIndex::new();
        for (i, key) in keys.iter().enumerate() {
            let _ = tree.insert_sync(key.clone(), i as u64);
        }
        tree
    }

    fn tree_index_upsert_sync_var(tree: &TreeIndex<Vec<u8>, u64>, key: Vec<u8>, value: u64) {
        let mut key = key;
        let mut value = value;
        for _ in 0..3 {
            // Use owned key directly - insert_sync returns it on failure
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
                                let idx = indices[base + (i % OPS_PER_THREAD)];
                                black_box(tree.get_with_guard(keys[idx].as_slice(), &guard));
                            }
                            warmup_done.wait();
                            start.wait();
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

    #[divan::bench(args = THREAD_COUNTS)]
    fn skipmap(bencher: Bencher, threads: usize) {
        let keys = Arc::new(variable_length_keys(N, MIN_KEY_SIZE, MAX_KEY_SIZE, 42));
        let indices = Arc::new(uniform_indices(N, OPS_PER_THREAD * threads, 42));
        let write_decisions: Arc<Vec<Vec<bool>>> = Arc::new(
            (0..threads)
                .map(|t| shuffled_write_decisions(OPS_PER_THREAD, WRITE_RATIO, 42 + t as u64))
                .collect(),
        );

        bencher
            .counter(divan::counter::ItemsCount::new(threads * OPS_PER_THREAD))
            .with_inputs(|| Arc::new(setup_skipmap_var(keys.as_ref())))
            .bench_local_values(|map| {
                let warmup_done = Arc::new(Barrier::new(threads));
                let start = Arc::new(Barrier::new(threads));
                let handles: Vec<_> = (0..threads)
                    .map(|t| {
                        let map = Arc::clone(&map);
                        let keys = Arc::clone(&keys);
                        let indices = Arc::clone(&indices);
                        let write_decisions = Arc::clone(&write_decisions);
                        let warmup_done = Arc::clone(&warmup_done);
                        let start = Arc::clone(&start);
                        thread::spawn(move || {
                            let base = t * OPS_PER_THREAD;
                            let is_write = &write_decisions[t];

                            for i in 0..WARMUP_OPS {
                                let idx = indices[base + (i % OPS_PER_THREAD)];
                                black_box(map.get(&keys[idx]));
                            }
                            warmup_done.wait();
                            start.wait();
                            pre_measurement_barrier();
                            let mut sum = 0u64;
                            for i in 0..OPS_PER_THREAD {
                                let idx = indices[base + i];
                                if is_write[i] {
                                    black_box(map.insert(keys[idx].clone(), i as u64));
                                } else if let Some(e) = map.get(&keys[idx]) {
                                    sum = sum.wrapping_add(*e.value());
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
    fn indexset(bencher: Bencher, threads: usize) {
        let keys = Arc::new(variable_length_keys(N, MIN_KEY_SIZE, MAX_KEY_SIZE, 42));
        let indices = Arc::new(uniform_indices(N, OPS_PER_THREAD * threads, 42));
        let write_decisions: Arc<Vec<Vec<bool>>> = Arc::new(
            (0..threads)
                .map(|t| shuffled_write_decisions(OPS_PER_THREAD, WRITE_RATIO, 42 + t as u64))
                .collect(),
        );

        bencher
            .counter(divan::counter::ItemsCount::new(threads * OPS_PER_THREAD))
            .with_inputs(|| Arc::new(setup_indexset_var(keys.as_ref())))
            .bench_local_values(|map| {
                let warmup_done = Arc::new(Barrier::new(threads));
                let start = Arc::new(Barrier::new(threads));
                let handles: Vec<_> = (0..threads)
                    .map(|t| {
                        let map = Arc::clone(&map);
                        let keys = Arc::clone(&keys);
                        let indices = Arc::clone(&indices);
                        let write_decisions = Arc::clone(&write_decisions);
                        let warmup_done = Arc::clone(&warmup_done);
                        let start = Arc::clone(&start);
                        thread::spawn(move || {
                            let base = t * OPS_PER_THREAD;
                            let is_write = &write_decisions[t];

                            for i in 0..WARMUP_OPS {
                                let idx = indices[base + (i % OPS_PER_THREAD)];
                                black_box(map.get(&keys[idx]));
                            }
                            warmup_done.wait();
                            start.wait();
                            pre_measurement_barrier();
                            let mut sum = 0u64;
                            for i in 0..OPS_PER_THREAD {
                                let idx = indices[base + i];
                                if is_write[i] {
                                    black_box(map.insert(keys[idx].clone(), i as u64));
                                } else if let Some(r) = map.get(&keys[idx]) {
                                    sum = sum.wrapping_add(r.get().value);
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
    fn tree_index(bencher: Bencher, threads: usize) {
        let keys = Arc::new(variable_length_keys(N, MIN_KEY_SIZE, MAX_KEY_SIZE, 42));
        let indices = Arc::new(uniform_indices(N, OPS_PER_THREAD * threads, 42));
        let write_decisions: Arc<Vec<Vec<bool>>> = Arc::new(
            (0..threads)
                .map(|t| shuffled_write_decisions(OPS_PER_THREAD, WRITE_RATIO, 42 + t as u64))
                .collect(),
        );

        bencher
            .counter(divan::counter::ItemsCount::new(threads * OPS_PER_THREAD))
            .with_inputs(|| Arc::new(setup_tree_index_var(keys.as_ref())))
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
                            let guard = SddGuard::new();
                            let base = t * OPS_PER_THREAD;
                            let is_write = &write_decisions[t];

                            for i in 0..WARMUP_OPS {
                                let idx = indices[base + (i % OPS_PER_THREAD)];
                                black_box(tree.peek(&keys[idx], &guard));
                            }
                            warmup_done.wait();
                            start.wait();
                            pre_measurement_barrier();
                            let mut sum = 0u64;
                            for i in 0..OPS_PER_THREAD {
                                let idx = indices[base + i];
                                if is_write[i] {
                                    tree_index_upsert_sync_var(&tree, keys[idx].clone(), i as u64);
                                    black_box(());
                                } else if let Some(v) = tree.peek(&keys[idx], &guard) {
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

    fn setup_skipmap_prefix(keys: &[[u8; PREFIX_KEY_SIZE]]) -> SkipMap<[u8; PREFIX_KEY_SIZE], u64> {
        let map = SkipMap::new();
        for (i, key) in keys.iter().enumerate() {
            map.insert(*key, i as u64);
        }
        map
    }

    fn setup_tree_index_prefix(
        keys: &[[u8; PREFIX_KEY_SIZE]],
    ) -> TreeIndex<[u8; PREFIX_KEY_SIZE], u64> {
        let tree = TreeIndex::new();
        for (i, key) in keys.iter().enumerate() {
            let _ = tree.insert_sync(*key, i as u64);
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

    /// Precompute range bounds (start_key, end_key) for each prefix.
    /// This avoids in-loop allocation overhead for structures using range queries.
    fn precompute_range_bounds(
        prefixes: &[[u8; 8]],
    ) -> Vec<([u8; PREFIX_KEY_SIZE], [u8; PREFIX_KEY_SIZE])> {
        prefixes
            .iter()
            .map(|prefix| {
                let mut start_key = [0u8; PREFIX_KEY_SIZE];
                start_key[0..8].copy_from_slice(prefix);
                let mut end_key = start_key;
                // Increment prefix for end bound (lexicographic successor)
                for j in (0..8).rev() {
                    if end_key[j] < 255 {
                        end_key[j] += 1;
                        break;
                    }
                    end_key[j] = 0;
                }
                (start_key, end_key)
            })
            .collect()
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
                let warmup_done = Arc::new(Barrier::new(threads));
                let start = Arc::new(Barrier::new(threads));
                let handles: Vec<_> = (0..threads)
                    .map(|t| {
                        let tree = Arc::clone(&tree);
                        let prefixes = Arc::clone(&prefixes);
                        let warmup_done = Arc::clone(&warmup_done);
                        let start = Arc::clone(&start);
                        thread::spawn(move || {
                            let guard = tree.guard();
                            let base = t * SCANS_PER_THREAD;

                            // Warmup
                            for i in 0..50 {
                                let prefix = &prefixes[base + (i % SCANS_PER_THREAD)];
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
                            start.wait();
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

    /// Masstree with value-only prefix scan (no key materialization).
    /// Uses `scan_prefix_values` which skips key reconstruction entirely.
    #[divan::bench(args = THREAD_COUNTS)]
    fn masstree15_values(bencher: Bencher, threads: usize) {
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
                let warmup_done = Arc::new(Barrier::new(threads));
                let start = Arc::new(Barrier::new(threads));
                let handles: Vec<_> = (0..threads)
                    .map(|t| {
                        let tree = Arc::clone(&tree);
                        let prefixes = Arc::clone(&prefixes);
                        let warmup_done = Arc::clone(&warmup_done);
                        let start = Arc::clone(&start);
                        thread::spawn(move || {
                            let guard = tree.guard();
                            let base = t * SCANS_PER_THREAD;

                            // Warmup
                            for i in 0..50 {
                                let prefix = &prefixes[base + (i % SCANS_PER_THREAD)];
                                let mut count = 0u64;
                                tree.scan_prefix_values(
                                    prefix,
                                    |v| {
                                        count = count.wrapping_add(v);
                                        true
                                    },
                                    &guard,
                                );
                                black_box(count);
                            }
                            warmup_done.wait();
                            start.wait();
                            pre_measurement_barrier();
                            let mut total = 0u64;
                            for i in 0..SCANS_PER_THREAD {
                                let prefix = &prefixes[base + i];
                                tree.scan_prefix_values(
                                    prefix,
                                    |v| {
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

    #[divan::bench(args = THREAD_COUNTS)]
    fn skipmap(bencher: Bencher, threads: usize) {
        let keys = Arc::new(keys_shared_prefix::<PREFIX_KEY_SIZE>(N, PREFIX_BUCKETS));
        let prefixes = random_prefixes(SCANS_PER_THREAD * threads, PREFIX_BUCKETS, 42);
        // Precompute range bounds to avoid in-loop allocation
        let range_bounds = Arc::new(precompute_range_bounds(&prefixes));

        bencher
            .counter(divan::counter::ItemsCount::new(threads * SCANS_PER_THREAD))
            .with_inputs(|| Arc::new(setup_skipmap_prefix(keys.as_ref())))
            .bench_local_values(|map| {
                let warmup_done = Arc::new(Barrier::new(threads));
                let start = Arc::new(Barrier::new(threads));
                let handles: Vec<_> = (0..threads)
                    .map(|t| {
                        let map = Arc::clone(&map);
                        let range_bounds = Arc::clone(&range_bounds);
                        let warmup_done = Arc::clone(&warmup_done);
                        let start = Arc::clone(&start);
                        thread::spawn(move || {
                            let base = t * SCANS_PER_THREAD;

                            // Warmup - use precomputed range bounds
                            for i in 0..50 {
                                let (start_key, end_key) =
                                    range_bounds[base + (i % SCANS_PER_THREAD)];
                                let mut count = 0u64;
                                for entry in map.range(start_key..end_key) {
                                    count = count.wrapping_add(*entry.value());
                                }
                                black_box(count);
                            }
                            warmup_done.wait();
                            start.wait();
                            pre_measurement_barrier();
                            let mut total = 0u64;
                            for i in 0..SCANS_PER_THREAD {
                                let (start_key, end_key) = range_bounds[base + i];
                                for entry in map.range(start_key..end_key) {
                                    total = total.wrapping_add(*entry.value());
                                }
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

    #[divan::bench(args = THREAD_COUNTS)]
    fn tree_index(bencher: Bencher, threads: usize) {
        let keys = Arc::new(keys_shared_prefix::<PREFIX_KEY_SIZE>(N, PREFIX_BUCKETS));
        let prefixes = random_prefixes(SCANS_PER_THREAD * threads, PREFIX_BUCKETS, 42);
        // Precompute range bounds to avoid in-loop allocation
        let range_bounds = Arc::new(precompute_range_bounds(&prefixes));

        bencher
            .counter(divan::counter::ItemsCount::new(threads * SCANS_PER_THREAD))
            .with_inputs(|| Arc::new(setup_tree_index_prefix(keys.as_ref())))
            .bench_local_values(|tree| {
                let warmup_done = Arc::new(Barrier::new(threads));
                let start = Arc::new(Barrier::new(threads));
                let handles: Vec<_> = (0..threads)
                    .map(|t| {
                        let tree = Arc::clone(&tree);
                        let range_bounds = Arc::clone(&range_bounds);
                        let warmup_done = Arc::clone(&warmup_done);
                        let start = Arc::clone(&start);
                        thread::spawn(move || {
                            let guard = SddGuard::new();
                            let base = t * SCANS_PER_THREAD;

                            // Warmup - use precomputed range bounds
                            for i in 0..50 {
                                let (start_key, end_key) =
                                    range_bounds[base + (i % SCANS_PER_THREAD)];
                                let mut count = 0u64;
                                tree.range(start_key..end_key, &guard).for_each(|(_, v)| {
                                    count = count.wrapping_add(*v);
                                });
                                black_box(count);
                            }
                            warmup_done.wait();
                            start.wait();
                            pre_measurement_barrier();
                            let mut total = 0u64;
                            for i in 0..SCANS_PER_THREAD {
                                let (start_key, end_key) = range_bounds[base + i];
                                tree.range(start_key..end_key, &guard).for_each(|(_, v)| {
                                    total = total.wrapping_add(*v);
                                });
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
// - TreeIndex write overhead applies here (remove+insert fallback).

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

    fn setup_skipmap_deep(keys: &[[u8; LONG_KEY_SIZE]]) -> SkipMap<[u8; LONG_KEY_SIZE], u64> {
        let map = SkipMap::new();
        for (i, key) in keys.iter().enumerate() {
            map.insert(*key, i as u64);
        }
        map
    }

    fn setup_indexset_deep(
        keys: &[[u8; LONG_KEY_SIZE]],
    ) -> IndexSetBTreeMap<[u8; LONG_KEY_SIZE], u64> {
        let map = IndexSetBTreeMap::new();
        for (i, key) in keys.iter().enumerate() {
            map.insert(*key, i as u64);
        }
        map
    }

    fn setup_tree_index_deep(keys: &[[u8; LONG_KEY_SIZE]]) -> TreeIndex<[u8; LONG_KEY_SIZE], u64> {
        let tree = TreeIndex::new();
        for (i, key) in keys.iter().enumerate() {
            let _ = tree.insert_sync(*key, i as u64);
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
                                let idx = indices[base + (i % OPS_PER_THREAD)];
                                black_box(tree.get_with_guard(&keys[idx], &guard));
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

    #[divan::bench(args = THREAD_COUNTS)]
    fn skipmap(bencher: Bencher, threads: usize) {
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
            .with_inputs(|| Arc::new(setup_skipmap_deep(keys.as_ref())))
            .bench_local_values(|map| {
                let warmup_done = Arc::new(Barrier::new(threads));
                let start = Arc::new(Barrier::new(threads));
                let handles: Vec<_> = (0..threads)
                    .map(|t| {
                        let map = Arc::clone(&map);
                        let keys = Arc::clone(&keys);
                        let indices = Arc::clone(&indices);
                        let write_decisions = Arc::clone(&write_decisions);
                        let warmup_done = Arc::clone(&warmup_done);
                        let start = Arc::clone(&start);
                        thread::spawn(move || {
                            let base = t * OPS_PER_THREAD;
                            let is_write = &write_decisions[t];

                            for i in 0..WARMUP_OPS {
                                let idx = indices[base + (i % OPS_PER_THREAD)];
                                black_box(map.get(&keys[idx]));
                            }
                            warmup_done.wait();
                            start.wait();
                            pre_measurement_barrier();
                            let mut sum = 0u64;
                            for i in 0..OPS_PER_THREAD {
                                let idx = indices[base + i];
                                if is_write[i] {
                                    black_box(map.insert(keys[idx], i as u64));
                                } else if let Some(e) = map.get(&keys[idx]) {
                                    sum = sum.wrapping_add(*e.value());
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
    fn indexset(bencher: Bencher, threads: usize) {
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
            .with_inputs(|| Arc::new(setup_indexset_deep(keys.as_ref())))
            .bench_local_values(|map| {
                let warmup_done = Arc::new(Barrier::new(threads));
                let start = Arc::new(Barrier::new(threads));
                let handles: Vec<_> = (0..threads)
                    .map(|t| {
                        let map = Arc::clone(&map);
                        let keys = Arc::clone(&keys);
                        let indices = Arc::clone(&indices);
                        let write_decisions = Arc::clone(&write_decisions);
                        let warmup_done = Arc::clone(&warmup_done);
                        let start = Arc::clone(&start);
                        thread::spawn(move || {
                            let base = t * OPS_PER_THREAD;
                            let is_write = &write_decisions[t];

                            for i in 0..WARMUP_OPS {
                                let idx = indices[base + (i % OPS_PER_THREAD)];
                                black_box(map.get(&keys[idx]));
                            }
                            warmup_done.wait();
                            start.wait();
                            pre_measurement_barrier();
                            let mut sum = 0u64;
                            for i in 0..OPS_PER_THREAD {
                                let idx = indices[base + i];
                                if is_write[i] {
                                    black_box(map.insert(keys[idx], i as u64));
                                } else if let Some(r) = map.get(&keys[idx]) {
                                    sum = sum.wrapping_add(r.get().value);
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
    fn tree_index(bencher: Bencher, threads: usize) {
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
            .with_inputs(|| Arc::new(setup_tree_index_deep(keys.as_ref())))
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
                            let guard = SddGuard::new();
                            let base = t * OPS_PER_THREAD;
                            let is_write = &write_decisions[t];

                            for i in 0..WARMUP_OPS {
                                let idx = indices[base + (i % OPS_PER_THREAD)];
                                black_box(tree.peek(&keys[idx], &guard));
                            }
                            warmup_done.wait();
                            start.wait();
                            pre_measurement_barrier();
                            let mut sum = 0u64;
                            for i in 0..OPS_PER_THREAD {
                                let idx = indices[base + i];
                                if is_write[i] {
                                    tree_index_upsert_sync_long(&tree, keys[idx], i as u64);
                                    black_box(());
                                } else if let Some(v) = tree.peek(&keys[idx], &guard) {
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
}

// =============================================================================
// 07: DEEP TRIE READ-ONLY - Isolate Traversal Cost from Write Overhead
// =============================================================================
//
// Pure read workload variant of benchmark 06. This isolates the multi-layer
// trie traversal cost from write-path differences (TreeIndex remove+insert,
// allocator behavior, etc.).
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

    fn setup_skipmap_deep(keys: &[[u8; LONG_KEY_SIZE]]) -> SkipMap<[u8; LONG_KEY_SIZE], u64> {
        let map = SkipMap::new();
        for (i, key) in keys.iter().enumerate() {
            map.insert(*key, i as u64);
        }
        map
    }

    fn setup_indexset_deep(
        keys: &[[u8; LONG_KEY_SIZE]],
    ) -> IndexSetBTreeMap<[u8; LONG_KEY_SIZE], u64> {
        let map = IndexSetBTreeMap::new();
        for (i, key) in keys.iter().enumerate() {
            map.insert(*key, i as u64);
        }
        map
    }

    fn setup_tree_index_deep(keys: &[[u8; LONG_KEY_SIZE]]) -> TreeIndex<[u8; LONG_KEY_SIZE], u64> {
        let tree = TreeIndex::new();
        for (i, key) in keys.iter().enumerate() {
            let _ = tree.insert_sync(*key, i as u64);
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
                                let idx = indices[base + (i % OPS_PER_THREAD)];
                                black_box(tree.get_with_guard(&keys[idx], &guard));
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

    #[divan::bench(args = THREAD_COUNTS)]
    fn skipmap(bencher: Bencher, threads: usize) {
        let keys = Arc::new(keys_shared_prefix_chunks::<LONG_KEY_SIZE>(
            N,
            PREFIX_CHUNKS,
            PREFIX_BUCKETS,
        ));
        let indices = Arc::new(uniform_indices(N, OPS_PER_THREAD * threads, 42));

        bencher
            .counter(divan::counter::ItemsCount::new(threads * OPS_PER_THREAD))
            .with_inputs(|| Arc::new(setup_skipmap_deep(keys.as_ref())))
            .bench_local_values(|map| {
                let warmup_done = Arc::new(Barrier::new(threads));
                let start = Arc::new(Barrier::new(threads));
                let handles: Vec<_> = (0..threads)
                    .map(|t| {
                        let map = Arc::clone(&map);
                        let keys = Arc::clone(&keys);
                        let indices = Arc::clone(&indices);
                        let warmup_done = Arc::clone(&warmup_done);
                        let start = Arc::clone(&start);
                        thread::spawn(move || {
                            let base = t * OPS_PER_THREAD;

                            for i in 0..WARMUP_OPS {
                                let idx = indices[base + (i % OPS_PER_THREAD)];
                                black_box(map.get(&keys[idx]));
                            }
                            warmup_done.wait();
                            start.wait();
                            pre_measurement_barrier();
                            let mut sum = 0u64;
                            for i in 0..OPS_PER_THREAD {
                                let idx = indices[base + i];
                                if let Some(e) = map.get(&keys[idx]) {
                                    sum = sum.wrapping_add(*e.value());
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
    fn indexset(bencher: Bencher, threads: usize) {
        let keys = Arc::new(keys_shared_prefix_chunks::<LONG_KEY_SIZE>(
            N,
            PREFIX_CHUNKS,
            PREFIX_BUCKETS,
        ));
        let indices = Arc::new(uniform_indices(N, OPS_PER_THREAD * threads, 42));

        bencher
            .counter(divan::counter::ItemsCount::new(threads * OPS_PER_THREAD))
            .with_inputs(|| Arc::new(setup_indexset_deep(keys.as_ref())))
            .bench_local_values(|map| {
                let warmup_done = Arc::new(Barrier::new(threads));
                let start = Arc::new(Barrier::new(threads));
                let handles: Vec<_> = (0..threads)
                    .map(|t| {
                        let map = Arc::clone(&map);
                        let keys = Arc::clone(&keys);
                        let indices = Arc::clone(&indices);
                        let warmup_done = Arc::clone(&warmup_done);
                        let start = Arc::clone(&start);
                        thread::spawn(move || {
                            let base = t * OPS_PER_THREAD;

                            for i in 0..WARMUP_OPS {
                                let idx = indices[base + (i % OPS_PER_THREAD)];
                                black_box(map.get(&keys[idx]));
                            }
                            warmup_done.wait();
                            start.wait();
                            pre_measurement_barrier();
                            let mut sum = 0u64;
                            for i in 0..OPS_PER_THREAD {
                                let idx = indices[base + i];
                                if let Some(r) = map.get(&keys[idx]) {
                                    sum = sum.wrapping_add(r.get().value);
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
    fn tree_index(bencher: Bencher, threads: usize) {
        let keys = Arc::new(keys_shared_prefix_chunks::<LONG_KEY_SIZE>(
            N,
            PREFIX_CHUNKS,
            PREFIX_BUCKETS,
        ));
        let indices = Arc::new(uniform_indices(N, OPS_PER_THREAD * threads, 42));

        bencher
            .counter(divan::counter::ItemsCount::new(threads * OPS_PER_THREAD))
            .with_inputs(|| Arc::new(setup_tree_index_deep(keys.as_ref())))
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
                            let guard = SddGuard::new();
                            let base = t * OPS_PER_THREAD;

                            for i in 0..WARMUP_OPS {
                                let idx = indices[base + (i % OPS_PER_THREAD)];
                                black_box(tree.peek(&keys[idx], &guard));
                            }
                            warmup_done.wait();
                            start.wait();
                            pre_measurement_barrier();
                            let mut sum = 0u64;
                            for i in 0..OPS_PER_THREAD {
                                let idx = indices[base + i];
                                if let Some(v) = tree.peek(&keys[idx], &guard) {
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

    fn setup_skipmap_arc(keys: &[Arc<[u8]>]) -> SkipMap<Arc<[u8]>, u64> {
        let map = SkipMap::new();
        for (i, key) in keys.iter().enumerate() {
            map.insert(Arc::clone(key), i as u64);
        }
        map
    }

    fn setup_indexset_arc(keys: &[Arc<[u8]>]) -> IndexSetBTreeMap<Arc<[u8]>, u64> {
        let map = IndexSetBTreeMap::new();
        for (i, key) in keys.iter().enumerate() {
            map.insert(Arc::clone(key), i as u64);
        }
        map
    }

    fn setup_tree_index_arc(keys: &[Arc<[u8]>]) -> TreeIndex<Arc<[u8]>, u64> {
        let tree = TreeIndex::new();
        for (i, key) in keys.iter().enumerate() {
            let _ = tree.insert_sync(Arc::clone(key), i as u64);
        }
        tree
    }

    fn tree_index_upsert_sync_arc(tree: &TreeIndex<Arc<[u8]>, u64>, key: Arc<[u8]>, value: u64) {
        let mut key = key;
        let mut value = value;
        for _ in 0..3 {
            // Use owned key directly - insert_sync returns it on failure
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
                                let idx = indices[base + (i % OPS_PER_THREAD)];
                                black_box(tree.get_with_guard(keys[idx].as_ref(), &guard));
                            }
                            warmup_done.wait();
                            start.wait();
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

    #[divan::bench(args = THREAD_COUNTS)]
    fn skipmap(bencher: Bencher, threads: usize) {
        let keys = Arc::new(variable_length_keys_arc(N, MIN_KEY_SIZE, MAX_KEY_SIZE, 42));
        let indices = Arc::new(uniform_indices(N, OPS_PER_THREAD * threads, 42));
        let write_decisions: Arc<Vec<Vec<bool>>> = Arc::new(
            (0..threads)
                .map(|t| shuffled_write_decisions(OPS_PER_THREAD, WRITE_RATIO, 42 + t as u64))
                .collect(),
        );

        bencher
            .counter(divan::counter::ItemsCount::new(threads * OPS_PER_THREAD))
            .with_inputs(|| Arc::new(setup_skipmap_arc(keys.as_ref())))
            .bench_local_values(|map| {
                let warmup_done = Arc::new(Barrier::new(threads));
                let start = Arc::new(Barrier::new(threads));
                let handles: Vec<_> = (0..threads)
                    .map(|t| {
                        let map = Arc::clone(&map);
                        let keys = Arc::clone(&keys);
                        let indices = Arc::clone(&indices);
                        let write_decisions = Arc::clone(&write_decisions);
                        let warmup_done = Arc::clone(&warmup_done);
                        let start = Arc::clone(&start);
                        thread::spawn(move || {
                            let base = t * OPS_PER_THREAD;
                            let is_write = &write_decisions[t];

                            for i in 0..WARMUP_OPS {
                                let idx = indices[base + (i % OPS_PER_THREAD)];
                                black_box(map.get(&keys[idx]));
                            }
                            warmup_done.wait();
                            start.wait();
                            pre_measurement_barrier();
                            let mut sum = 0u64;
                            for i in 0..OPS_PER_THREAD {
                                let idx = indices[base + i];
                                if is_write[i] {
                                    // Cheap Arc clone instead of Vec clone
                                    map.insert(Arc::clone(&keys[idx]), i as u64);
                                } else if let Some(e) = map.get(&keys[idx]) {
                                    sum = sum.wrapping_add(*e.value());
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
    fn indexset(bencher: Bencher, threads: usize) {
        let keys = Arc::new(variable_length_keys_arc(N, MIN_KEY_SIZE, MAX_KEY_SIZE, 42));
        let indices = Arc::new(uniform_indices(N, OPS_PER_THREAD * threads, 42));
        let write_decisions: Arc<Vec<Vec<bool>>> = Arc::new(
            (0..threads)
                .map(|t| shuffled_write_decisions(OPS_PER_THREAD, WRITE_RATIO, 42 + t as u64))
                .collect(),
        );

        bencher
            .counter(divan::counter::ItemsCount::new(threads * OPS_PER_THREAD))
            .with_inputs(|| Arc::new(setup_indexset_arc(keys.as_ref())))
            .bench_local_values(|map| {
                let warmup_done = Arc::new(Barrier::new(threads));
                let start = Arc::new(Barrier::new(threads));
                let handles: Vec<_> = (0..threads)
                    .map(|t| {
                        let map = Arc::clone(&map);
                        let keys = Arc::clone(&keys);
                        let indices = Arc::clone(&indices);
                        let write_decisions = Arc::clone(&write_decisions);
                        let warmup_done = Arc::clone(&warmup_done);
                        let start = Arc::clone(&start);
                        thread::spawn(move || {
                            let base = t * OPS_PER_THREAD;
                            let is_write = &write_decisions[t];

                            for i in 0..WARMUP_OPS {
                                let idx = indices[base + (i % OPS_PER_THREAD)];
                                black_box(map.get(&keys[idx]));
                            }
                            warmup_done.wait();
                            start.wait();
                            pre_measurement_barrier();
                            let mut sum = 0u64;
                            for i in 0..OPS_PER_THREAD {
                                let idx = indices[base + i];
                                if is_write[i] {
                                    // Cheap Arc clone instead of Vec clone
                                    map.insert(Arc::clone(&keys[idx]), i as u64);
                                } else if let Some(r) = map.get(&keys[idx]) {
                                    sum = sum.wrapping_add(r.get().value);
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
    fn tree_index(bencher: Bencher, threads: usize) {
        let keys = Arc::new(variable_length_keys_arc(N, MIN_KEY_SIZE, MAX_KEY_SIZE, 42));
        let indices = Arc::new(uniform_indices(N, OPS_PER_THREAD * threads, 42));
        let write_decisions: Arc<Vec<Vec<bool>>> = Arc::new(
            (0..threads)
                .map(|t| shuffled_write_decisions(OPS_PER_THREAD, WRITE_RATIO, 42 + t as u64))
                .collect(),
        );

        bencher
            .counter(divan::counter::ItemsCount::new(threads * OPS_PER_THREAD))
            .with_inputs(|| Arc::new(setup_tree_index_arc(keys.as_ref())))
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
                            let guard = SddGuard::new();
                            let base = t * OPS_PER_THREAD;
                            let is_write = &write_decisions[t];

                            for i in 0..WARMUP_OPS {
                                let idx = indices[base + (i % OPS_PER_THREAD)];
                                black_box(tree.peek(&keys[idx], &guard));
                            }
                            warmup_done.wait();
                            start.wait();
                            pre_measurement_barrier();
                            let mut sum = 0u64;
                            for i in 0..OPS_PER_THREAD {
                                let idx = indices[base + i];
                                if is_write[i] {
                                    // Cheap Arc clone instead of Vec clone
                                    tree_index_upsert_sync_arc(
                                        &tree,
                                        Arc::clone(&keys[idx]),
                                        i as u64,
                                    );
                                    black_box(());
                                } else if let Some(v) = tree.peek(&keys[idx], &guard) {
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
}

// =============================================================================
// 09: REALISTIC PREFIX MIX - Hierarchical Prefixes + Pagination + Updates
// =============================================================================
//
// Simulates a production-style listing workload:
// - Hierarchical keys (tenant -> collection -> filter token -> unique suffix)
// - Hot/cold tenant skew for scan prefixes (Zipf-like)
// - Mixed prefix depths (8/16/20-byte prefixes)
// - Pagination-style early stop (visitor returns false after N hits)
// - Concurrent point reads and upserts while scans execute
//
// Counter reports MIXED OPS/SEC (not scans/sec):
// one operation is either a prefix scan, point read, or upsert.

#[divan::bench_group(name = "09_prefix_realistic_mixed", sample_count = 200)]
mod prefix_realistic_mixed {
    use super::*;
    use bench_utils::zipfian_indices;

    const KEY_SIZE: usize = 64;
    const N: usize = 80_000;
    const OPS_PER_THREAD: usize = 5_000;

    const SCAN_RATIO: usize = 75;
    const READ_RATIO: usize = 20;
    // write ratio is implied: 100 - SCAN_RATIO - READ_RATIO

    const TENANTS: u64 = 64;
    const COLLECTIONS: u64 = 8;
    const SEGMENTS: u64 = 4;
    const DAYS: u64 = 4;
    const REGIONS: u64 = 2;
    const KINDS: u64 = 2;

    #[derive(Clone, Copy)]
    enum OpKind {
        Scan,
        Read,
        Write,
    }

    #[derive(Clone, Copy)]
    struct PrefixPlan {
        prefix: [u8; KEY_SIZE],
        prefix_len: u8,
        start_key: [u8; KEY_SIZE],
        end_key: [u8; KEY_SIZE],
        limit: u16,
    }

    struct ThreadWorkload {
        ops: Vec<OpKind>,
        scan_plans: Vec<PrefixPlan>,
        point_indices: Vec<usize>,
    }

    fn setup_masstree15_prefix_realistic(keys: &[[u8; KEY_SIZE]]) -> MassTree15Inline<u64> {
        let tree = MassTree15Inline::new();
        {
            let guard = tree.guard();
            for (i, key) in keys.iter().enumerate() {
                let _ = tree.insert_with_guard(key, i as u64, &guard);
            }
        }
        tree
    }

    fn setup_skipmap_prefix_realistic(keys: &[[u8; KEY_SIZE]]) -> SkipMap<[u8; KEY_SIZE], u64> {
        let map = SkipMap::new();
        for (i, key) in keys.iter().enumerate() {
            map.insert(*key, i as u64);
        }
        map
    }

    fn setup_tree_index_prefix_realistic(
        keys: &[[u8; KEY_SIZE]],
    ) -> TreeIndex<[u8; KEY_SIZE], u64> {
        let tree = TreeIndex::new();
        for (i, key) in keys.iter().enumerate() {
            let _ = tree.insert_sync(*key, i as u64);
        }
        tree
    }

    fn tree_index_upsert_sync_prefix(
        tree: &TreeIndex<[u8; KEY_SIZE], u64>,
        key: [u8; KEY_SIZE],
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

    const fn xorshift64(state: &mut u64) -> u64 {
        *state ^= *state << 13;
        *state ^= *state >> 7;
        *state ^= *state << 17;
        *state
    }

    fn realistic_prefix_keys() -> Vec<[u8; KEY_SIZE]> {
        let mut out = keys::<KEY_SIZE>(N);

        for (i, key) in out.iter_mut().enumerate() {
            let i_u64 = i as u64;

            let tenant = i_u64 % TENANTS;
            let collection = (i_u64 / TENANTS) % COLLECTIONS;

            // 4-byte filter token (segment/day/region/kind) + 4-byte unique tail.
            let group = i_u64 / (TENANTS * COLLECTIONS);
            let segment = (group % SEGMENTS) as u8;
            let day = ((group / SEGMENTS) % DAYS) as u8;
            let region = ((group / (SEGMENTS * DAYS)) % REGIONS) as u8;
            let kind = ((group / (SEGMENTS * DAYS * REGIONS)) % KINDS) as u8;
            let tail = ((i_u64.wrapping_mul(0x9E37_79B9_7F4A_7C15)) >> 32) as u32;

            key[0..8].copy_from_slice(&tenant.to_be_bytes());
            key[8..16].copy_from_slice(&collection.to_be_bytes());
            key[16] = segment;
            key[17] = day;
            key[18] = region;
            key[19] = kind;
            key[20..24].copy_from_slice(&tail.to_be_bytes());
        }

        out
    }

    fn prefix_upper_bound(prefix: &[u8; KEY_SIZE], len: usize) -> [u8; KEY_SIZE] {
        let mut out = [0u8; KEY_SIZE];
        out[..len].copy_from_slice(&prefix[..len]);

        for i in (0..len).rev() {
            if out[i] < u8::MAX {
                out[i] += 1;
                out[i + 1..].fill(0);
                return out;
            }
            out[i] = 0;
        }

        // Not expected for this dataset, but keep a bounded range fallback.
        [u8::MAX; KEY_SIZE]
    }

    fn shuffled_op_kinds(
        count: usize,
        scan_ratio: usize,
        read_ratio: usize,
        seed: u64,
    ) -> Vec<OpKind> {
        let scan_count = (count * scan_ratio) / 100;
        let read_count = (count * read_ratio) / 100;
        let write_count = count - scan_count - read_count;

        let mut ops = Vec::with_capacity(count);
        ops.extend(std::iter::repeat_n(OpKind::Scan, scan_count));
        ops.extend(std::iter::repeat_n(OpKind::Read, read_count));
        ops.extend(std::iter::repeat_n(OpKind::Write, write_count));
        bench_utils::shuffle(&mut ops, seed);
        ops
    }

    fn build_scan_plans(count: usize, seed: u64) -> Vec<PrefixPlan> {
        let tenants = zipfian_indices(TENANTS as usize, count, 1.15, seed ^ 0x1357_2468_ABCD_EF10);
        let mut state = seed ^ 0x9E37_79B9_7F4A_7C15;
        let mut plans = Vec::with_capacity(count);

        for tenant_idx in tenants {
            let roll = xorshift64(&mut state);
            let depth_pick = (roll % 100) as usize;
            let prefix_len = if depth_pick < 45 {
                8usize
            } else if depth_pick < 80 {
                16usize
            } else {
                20usize
            };

            let limit_pick = ((roll >> 8) % 100) as usize;
            let limit = if limit_pick < 50 {
                32usize
            } else if limit_pick < 85 {
                128usize
            } else {
                512usize
            };

            let collection = (roll >> 16) % COLLECTIONS;
            let segment = ((roll >> 24) % SEGMENTS) as u8;
            let day = ((roll >> 32) % DAYS) as u8;
            let region = ((roll >> 40) % REGIONS) as u8;
            let kind = ((roll >> 48) % KINDS) as u8;

            let mut prefix = [0u8; KEY_SIZE];
            prefix[0..8].copy_from_slice(&(tenant_idx as u64).to_be_bytes());
            prefix[8..16].copy_from_slice(&collection.to_be_bytes());
            prefix[16] = segment;
            prefix[17] = day;
            prefix[18] = region;
            prefix[19] = kind;

            let mut start_key = [0u8; KEY_SIZE];
            start_key[..prefix_len].copy_from_slice(&prefix[..prefix_len]);
            let end_key = prefix_upper_bound(&prefix, prefix_len);

            plans.push(PrefixPlan {
                prefix,
                prefix_len: prefix_len as u8,
                start_key,
                end_key,
                limit: limit as u16,
            });
        }

        plans
    }

    fn build_thread_workload(thread_id: usize) -> ThreadWorkload {
        let ops = shuffled_op_kinds(
            OPS_PER_THREAD,
            SCAN_RATIO,
            READ_RATIO,
            42 + thread_id as u64,
        );
        let scan_count = ops.iter().filter(|op| matches!(op, OpKind::Scan)).count();
        let point_count = OPS_PER_THREAD - scan_count;

        let scan_plans = build_scan_plans(scan_count, 11_111 + thread_id as u64);
        let point_indices = zipfian_indices(N, point_count, 1.05, 22_222 + thread_id as u64);

        ThreadWorkload {
            ops,
            scan_plans,
            point_indices,
        }
    }

    fn build_workloads(threads: usize) -> Arc<Vec<ThreadWorkload>> {
        Arc::new((0..threads).map(build_thread_workload).collect())
    }

    #[divan::bench(args = THREAD_COUNTS)]
    fn masstree15(bencher: Bencher, threads: usize) {
        let keys = Arc::new(realistic_prefix_keys());
        let workloads = build_workloads(threads);

        bencher
            .counter(divan::counter::ItemsCount::new(threads * OPS_PER_THREAD))
            .with_inputs(|| Arc::new(setup_masstree15_prefix_realistic(keys.as_ref())))
            .bench_local_values(|tree| {
                let warmup_done = Arc::new(Barrier::new(threads));
                let start = Arc::new(Barrier::new(threads));
                let handles: Vec<_> = (0..threads)
                    .map(|t| {
                        let tree = Arc::clone(&tree);
                        let keys = Arc::clone(&keys);
                        let workloads = Arc::clone(&workloads);
                        let warmup_done = Arc::clone(&warmup_done);
                        let start = Arc::clone(&start);

                        thread::spawn(move || {
                            let guard = tree.guard();
                            let wl = &workloads[t];

                            let mut scan_i = 0usize;
                            let mut point_i = 0usize;
                            for i in 0..WARMUP_OPS {
                                match wl.ops[i % wl.ops.len()] {
                                    OpKind::Scan => {
                                        let plan = wl.scan_plans[scan_i % wl.scan_plans.len()];
                                        scan_i += 1;
                                        let mut seen = 0usize;
                                        tree.scan_prefix(
                                            &plan.prefix[..usize::from(plan.prefix_len)],
                                            |_, _| {
                                                seen += 1;
                                                seen < usize::from(plan.limit)
                                            },
                                            &guard,
                                        );
                                    }
                                    OpKind::Read => {
                                        let idx =
                                            wl.point_indices[point_i % wl.point_indices.len()];
                                        point_i += 1;
                                        black_box(tree.get_with_guard(&keys[idx], &guard));
                                    }
                                    OpKind::Write => {
                                        let idx =
                                            wl.point_indices[point_i % wl.point_indices.len()];
                                        point_i += 1;
                                        let _ = tree.insert_with_guard(
                                            &keys[idx],
                                            (i + 1) as u64,
                                            &guard,
                                        );
                                    }
                                }
                            }
                            warmup_done.wait();
                            start.wait();

                            pre_measurement_barrier();
                            let mut total = 0u64;
                            let mut scan_i = 0usize;
                            let mut point_i = 0usize;
                            for (op_idx, op) in wl.ops.iter().enumerate() {
                                match op {
                                    OpKind::Scan => {
                                        let plan = wl.scan_plans[scan_i];
                                        scan_i += 1;
                                        let mut seen = 0usize;
                                        tree.scan_prefix(
                                            &plan.prefix[..usize::from(plan.prefix_len)],
                                            |_, v| {
                                                total = total.wrapping_add(v);
                                                seen += 1;
                                                seen < usize::from(plan.limit)
                                            },
                                            &guard,
                                        );
                                    }
                                    OpKind::Read => {
                                        let idx = wl.point_indices[point_i];
                                        point_i += 1;
                                        if let Some(v) = tree.get_with_guard(&keys[idx], &guard) {
                                            total = total.wrapping_add(v);
                                        }
                                    }
                                    OpKind::Write => {
                                        let idx = wl.point_indices[point_i];
                                        let value = ((t as u64) << 32) ^ op_idx as u64;
                                        point_i += 1;
                                        let _ = tree.insert_with_guard(&keys[idx], value, &guard);
                                    }
                                }
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

    #[divan::bench(args = THREAD_COUNTS)]
    fn masstree15_values(bencher: Bencher, threads: usize) {
        let keys = Arc::new(realistic_prefix_keys());
        let workloads = build_workloads(threads);

        bencher
            .counter(divan::counter::ItemsCount::new(threads * OPS_PER_THREAD))
            .with_inputs(|| Arc::new(setup_masstree15_prefix_realistic(keys.as_ref())))
            .bench_local_values(|tree| {
                let warmup_done = Arc::new(Barrier::new(threads));
                let start = Arc::new(Barrier::new(threads));
                let handles: Vec<_> = (0..threads)
                    .map(|t| {
                        let tree = Arc::clone(&tree);
                        let keys = Arc::clone(&keys);
                        let workloads = Arc::clone(&workloads);
                        let warmup_done = Arc::clone(&warmup_done);
                        let start = Arc::clone(&start);

                        thread::spawn(move || {
                            let guard = tree.guard();
                            let wl = &workloads[t];

                            let mut scan_i = 0usize;
                            let mut point_i = 0usize;
                            for i in 0..WARMUP_OPS {
                                match wl.ops[i % wl.ops.len()] {
                                    OpKind::Scan => {
                                        let plan = wl.scan_plans[scan_i % wl.scan_plans.len()];
                                        scan_i += 1;
                                        let mut seen = 0usize;
                                        tree.scan_prefix_values(
                                            &plan.prefix[..usize::from(plan.prefix_len)],
                                            |_| {
                                                seen += 1;
                                                seen < usize::from(plan.limit)
                                            },
                                            &guard,
                                        );
                                    }
                                    OpKind::Read => {
                                        let idx =
                                            wl.point_indices[point_i % wl.point_indices.len()];
                                        point_i += 1;
                                        black_box(tree.get_with_guard(&keys[idx], &guard));
                                    }
                                    OpKind::Write => {
                                        let idx =
                                            wl.point_indices[point_i % wl.point_indices.len()];
                                        point_i += 1;
                                        let _ = tree.insert_with_guard(
                                            &keys[idx],
                                            (i + 1) as u64,
                                            &guard,
                                        );
                                    }
                                }
                            }
                            warmup_done.wait();
                            start.wait();

                            pre_measurement_barrier();
                            let mut total = 0u64;
                            let mut scan_i = 0usize;
                            let mut point_i = 0usize;
                            for (op_idx, op) in wl.ops.iter().enumerate() {
                                match op {
                                    OpKind::Scan => {
                                        let plan = wl.scan_plans[scan_i];
                                        scan_i += 1;
                                        let mut seen = 0usize;
                                        tree.scan_prefix_values(
                                            &plan.prefix[..usize::from(plan.prefix_len)],
                                            |v| {
                                                total = total.wrapping_add(v);
                                                seen += 1;
                                                seen < usize::from(plan.limit)
                                            },
                                            &guard,
                                        );
                                    }
                                    OpKind::Read => {
                                        let idx = wl.point_indices[point_i];
                                        point_i += 1;
                                        if let Some(v) = tree.get_with_guard(&keys[idx], &guard) {
                                            total = total.wrapping_add(v);
                                        }
                                    }
                                    OpKind::Write => {
                                        let idx = wl.point_indices[point_i];
                                        let value = ((t as u64) << 32) ^ op_idx as u64;
                                        point_i += 1;
                                        let _ = tree.insert_with_guard(&keys[idx], value, &guard);
                                    }
                                }
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

    #[divan::bench(args = THREAD_COUNTS)]
    fn skipmap(bencher: Bencher, threads: usize) {
        let keys = Arc::new(realistic_prefix_keys());
        let workloads = build_workloads(threads);

        bencher
            .counter(divan::counter::ItemsCount::new(threads * OPS_PER_THREAD))
            .with_inputs(|| Arc::new(setup_skipmap_prefix_realistic(keys.as_ref())))
            .bench_local_values(|map| {
                let warmup_done = Arc::new(Barrier::new(threads));
                let start = Arc::new(Barrier::new(threads));
                let handles: Vec<_> = (0..threads)
                    .map(|t| {
                        let map = Arc::clone(&map);
                        let keys = Arc::clone(&keys);
                        let workloads = Arc::clone(&workloads);
                        let warmup_done = Arc::clone(&warmup_done);
                        let start = Arc::clone(&start);

                        thread::spawn(move || {
                            let wl = &workloads[t];

                            let mut scan_i = 0usize;
                            let mut point_i = 0usize;
                            for i in 0..WARMUP_OPS {
                                match wl.ops[i % wl.ops.len()] {
                                    OpKind::Scan => {
                                        let plan = wl.scan_plans[scan_i % wl.scan_plans.len()];
                                        scan_i += 1;
                                        let mut seen = 0usize;
                                        for _ in map.range(plan.start_key..plan.end_key) {
                                            seen += 1;
                                            if seen >= usize::from(plan.limit) {
                                                break;
                                            }
                                        }
                                    }
                                    OpKind::Read => {
                                        let idx =
                                            wl.point_indices[point_i % wl.point_indices.len()];
                                        point_i += 1;
                                        black_box(map.get(&keys[idx]));
                                    }
                                    OpKind::Write => {
                                        let idx =
                                            wl.point_indices[point_i % wl.point_indices.len()];
                                        point_i += 1;
                                        map.insert(keys[idx], (i + 1) as u64);
                                    }
                                }
                            }
                            warmup_done.wait();
                            start.wait();

                            pre_measurement_barrier();
                            let mut total = 0u64;
                            let mut scan_i = 0usize;
                            let mut point_i = 0usize;
                            for (op_idx, op) in wl.ops.iter().enumerate() {
                                match op {
                                    OpKind::Scan => {
                                        let plan = wl.scan_plans[scan_i];
                                        scan_i += 1;
                                        let mut seen = 0usize;
                                        for entry in map.range(plan.start_key..plan.end_key) {
                                            total = total.wrapping_add(*entry.value());
                                            seen += 1;
                                            if seen >= usize::from(plan.limit) {
                                                break;
                                            }
                                        }
                                    }
                                    OpKind::Read => {
                                        let idx = wl.point_indices[point_i];
                                        point_i += 1;
                                        if let Some(entry) = map.get(&keys[idx]) {
                                            total = total.wrapping_add(*entry.value());
                                        }
                                    }
                                    OpKind::Write => {
                                        let idx = wl.point_indices[point_i];
                                        let value = ((t as u64) << 32) ^ op_idx as u64;
                                        point_i += 1;
                                        map.insert(keys[idx], value);
                                    }
                                }
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

    #[divan::bench(args = THREAD_COUNTS)]
    fn tree_index(bencher: Bencher, threads: usize) {
        let keys = Arc::new(realistic_prefix_keys());
        let workloads = build_workloads(threads);

        bencher
            .counter(divan::counter::ItemsCount::new(threads * OPS_PER_THREAD))
            .with_inputs(|| Arc::new(setup_tree_index_prefix_realistic(keys.as_ref())))
            .bench_local_values(|tree| {
                let warmup_done = Arc::new(Barrier::new(threads));
                let start = Arc::new(Barrier::new(threads));
                let handles: Vec<_> = (0..threads)
                    .map(|t| {
                        let tree = Arc::clone(&tree);
                        let keys = Arc::clone(&keys);
                        let workloads = Arc::clone(&workloads);
                        let warmup_done = Arc::clone(&warmup_done);
                        let start = Arc::clone(&start);

                        thread::spawn(move || {
                            let guard = SddGuard::new();
                            let wl = &workloads[t];

                            let mut scan_i = 0usize;
                            let mut point_i = 0usize;
                            for i in 0..WARMUP_OPS {
                                match wl.ops[i % wl.ops.len()] {
                                    OpKind::Scan => {
                                        let plan = wl.scan_plans[scan_i % wl.scan_plans.len()];
                                        scan_i += 1;
                                        black_box(
                                            tree.range(plan.start_key..plan.end_key, &guard)
                                                .take(usize::from(plan.limit))
                                                .count(),
                                        );
                                    }
                                    OpKind::Read => {
                                        let idx =
                                            wl.point_indices[point_i % wl.point_indices.len()];
                                        point_i += 1;
                                        black_box(tree.peek(&keys[idx], &guard));
                                    }
                                    OpKind::Write => {
                                        let idx =
                                            wl.point_indices[point_i % wl.point_indices.len()];
                                        point_i += 1;
                                        tree_index_upsert_sync_prefix(
                                            &tree,
                                            keys[idx],
                                            (i + 1) as u64,
                                        );
                                    }
                                }
                            }
                            warmup_done.wait();
                            start.wait();

                            pre_measurement_barrier();
                            let mut total = 0u64;
                            let mut scan_i = 0usize;
                            let mut point_i = 0usize;
                            for (op_idx, op) in wl.ops.iter().enumerate() {
                                match op {
                                    OpKind::Scan => {
                                        let plan = wl.scan_plans[scan_i];
                                        scan_i += 1;
                                        for (_, v) in tree
                                            .range(plan.start_key..plan.end_key, &guard)
                                            .take(usize::from(plan.limit))
                                        {
                                            total = total.wrapping_add(*v);
                                        }
                                    }
                                    OpKind::Read => {
                                        let idx = wl.point_indices[point_i];
                                        point_i += 1;
                                        if let Some(v) = tree.peek(&keys[idx], &guard) {
                                            total = total.wrapping_add(*v);
                                        }
                                    }
                                    OpKind::Write => {
                                        let idx = wl.point_indices[point_i];
                                        let value = ((t as u64) << 32) ^ op_idx as u64;
                                        point_i += 1;
                                        tree_index_upsert_sync_prefix(&tree, keys[idx], value);
                                    }
                                }
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
