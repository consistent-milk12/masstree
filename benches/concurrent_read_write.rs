//! Concurrent read/write benchmarks.

#![expect(clippy::pedantic)]
#![expect(clippy::indexing_slicing)]

mod bench_utils;

use bench_utils::{
    keys, keys_shared_prefix_chunks, run_concurrent, shuffle, uniform_indices, zipfian_indices,
};
use crossbeam_skiplist::SkipMap;
use divan::counter::ItemsCount;
use divan::{Bencher, black_box};
use indexset::concurrent::map::BTreeMap as IndexSetBTreeMap;
use masstree::{InlineBits, MassTree15Inline};
use scc::TreeIndex;
use sdd::Guard as SddGuard;
use seize::LocalGuard;

// =============================================================================
// MyRecord: Custom InlineBits type for benchmarking
// =============================================================================

/// 8-byte record packed into a u64 via `InlineBits`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(C)]
struct MyRecord {
    a: u8,
    b: u8,
    c: u8,
    d: u8,
    payload: u32,
}

impl MyRecord {
    #[inline(always)]
    const fn new(seed: u64) -> Self {
        Self {
            a: (seed & 0xFF) as u8,
            b: ((seed >> 8) & 0xFF) as u8,
            c: ((seed >> 16) & 0xFF) as u8,
            d: ((seed >> 24) & 0xFF) as u8,
            payload: (seed >> 32) as u32,
        }
    }

    #[inline(always)]
    fn checksum(&self) -> u64 {
        u64::from(self.a)
            .wrapping_add(u64::from(self.b))
            .wrapping_add(u64::from(self.c))
            .wrapping_add(u64::from(self.d))
            .wrapping_add(u64::from(self.payload))
    }
}

impl InlineBits for MyRecord {
    #[inline(always)]
    fn to_bits(self) -> u64 {
        // SAFETY: MyRecord is repr(C), 8 bytes, all-bit-patterns valid.
        unsafe { core::mem::transmute(self) }
    }

    #[inline(always)]
    fn from_bits(bits: u64) -> Self {
        // SAFETY: MyRecord is repr(C), 8 bytes, all-bit-patterns valid.
        unsafe { core::mem::transmute(bits) }
    }
}

fn main() {
    divan::main();
}

// =============================================================================
// Constants
// =============================================================================

const KEY_SIZE: usize = 64;

/// Warmup iterations per thread before measurement (capped proportionally by harness).
const WARMUP_OPS: usize = 500;

/// Standard thread counts for all benchmarks.
const THREAD_COUNTS: [usize; 6] = [1, 2, 4, 6, 8, 12];

/// Base seed for deterministic workload generation.
const BASE_SEED: u64 = 42;

// =============================================================================
// Setup Helpers
// =============================================================================

/// TreeIndex upsert emulation via remove-then-reinsert.
fn tree_index_upsert_sync(
    tree: &TreeIndex<[u8; KEY_SIZE], MyRecord>,
    key: [u8; KEY_SIZE],
    value: MyRecord,
) {
    let mut key: [u8; 64] = key;
    let mut value: MyRecord = value;

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

fn setup_masstree15(keys: &[[u8; KEY_SIZE]]) -> MassTree15Inline<MyRecord> {
    let tree = MassTree15Inline::new();
    {
        let guard = tree.guard();

        for (i, key) in keys.iter().enumerate() {
            let _ = tree.insert_with_guard(key, MyRecord::new(i as u64), &guard);
        }
    }

    tree
}

fn setup_skipmap(keys: &[[u8; KEY_SIZE]]) -> SkipMap<[u8; KEY_SIZE], MyRecord> {
    let map: SkipMap<[u8; 64], MyRecord> = SkipMap::new();

    for (i, key) in keys.iter().enumerate() {
        map.insert(*key, MyRecord::new(i as u64));
    }

    map
}

fn setup_indexset(keys: &[[u8; KEY_SIZE]]) -> IndexSetBTreeMap<[u8; KEY_SIZE], MyRecord> {
    let map: IndexSetBTreeMap<[u8; 64], MyRecord> = IndexSetBTreeMap::new();

    for (i, key) in keys.iter().enumerate() {
        map.insert(*key, MyRecord::new(i as u64));
    }

    map
}

fn setup_tree_index(keys: &[[u8; KEY_SIZE]]) -> TreeIndex<[u8; KEY_SIZE], MyRecord> {
    let tree: TreeIndex<[u8; 64], MyRecord> = TreeIndex::new();

    for (i, key) in keys.iter().enumerate() {
        let _ = tree.insert_sync(*key, MyRecord::new(i as u64));
    }

    tree
}

/// Generate a shuffled array of operation types (true = write, false = read).
fn shuffled_write_decisions(count: usize, write_ratio_percent: usize, seed: u64) -> Vec<bool> {
    let write_count: usize = (count * write_ratio_percent) / 100;
    let mut decisions: Vec<bool> = vec![false; count];

    for d in decisions.iter_mut().take(write_count) {
        *d = true;
    }

    // ChaCha8Rng Fisher-Yates (no modulo bias, consistent with uniform_indices)
    shuffle(&mut decisions, seed);
    decisions
}

// =============================================================================
// 01: MIXED 90-10 - Uniform Access Pattern
// =============================================================================

#[divan::bench_group(name = "01_mixed_90_10_uniform", sample_count = 200, sample_size = 1)]
mod mixed_uniform {
    use super::*;

    const N: usize = 50_000;
    const OPS_PER_THREAD: usize = 12_500;
    const WRITE_RATIO: usize = 10;

    #[divan::bench(args = THREAD_COUNTS)]
    fn masstree15(bencher: Bencher, threads: usize) {
        let keys: Vec<[u8; 64]> = keys::<KEY_SIZE>(N);
        let indices: Vec<usize> = uniform_indices(N, OPS_PER_THREAD * threads, BASE_SEED);
        let write_decisions: Vec<Vec<bool>> = (0..threads)
            .map(|t: usize| {
                shuffled_write_decisions(OPS_PER_THREAD, WRITE_RATIO, BASE_SEED + t as u64)
            })
            .collect();

        bencher
            .counter(ItemsCount::new(threads * OPS_PER_THREAD))
            .with_inputs(|| setup_masstree15(&keys))
            .bench_local_values(|tree| {
                run_concurrent(threads, WARMUP_OPS, OPS_PER_THREAD, |ctx| {
                    let guard: LocalGuard<'_> = tree.guard();
                    let base: usize = ctx.tid * OPS_PER_THREAD;
                    let is_write: &Vec<bool> = &write_decisions[ctx.tid];

                    for i in 0..ctx.warmup_ops {
                        let idx: usize = indices[base + (i % OPS_PER_THREAD)];

                        black_box(tree.get_with_guard(&keys[idx], &guard));
                    }

                    ctx.finish_warmup();
                    ctx.begin_measurement();

                    let mut sum: u64 = 0;

                    for i in 0..OPS_PER_THREAD {
                        let idx: usize = indices[base + i];

                        if is_write[i] {
                            let _ =
                                tree.insert_with_guard(&keys[idx], MyRecord::new(i as u64), &guard);
                        } else if let Some(v) = tree.get_with_guard(&keys[idx], &guard) {
                            sum = sum.wrapping_add(v.checksum());
                        }
                    }

                    ctx.end_measurement();
                    black_box(sum);
                });
            });
    }

    #[divan::bench(args = THREAD_COUNTS)]
    fn skipmap(bencher: Bencher, threads: usize) {
        let keys: Vec<[u8; 64]> = keys::<KEY_SIZE>(N);
        let indices: Vec<usize> = uniform_indices(N, OPS_PER_THREAD * threads, BASE_SEED);
        let write_decisions: Vec<Vec<bool>> = (0..threads)
            .map(|t: usize| {
                shuffled_write_decisions(OPS_PER_THREAD, WRITE_RATIO, BASE_SEED + t as u64)
            })
            .collect();

        bencher
            .counter(ItemsCount::new(threads * OPS_PER_THREAD))
            .with_inputs(|| setup_skipmap(&keys))
            .bench_local_values(|map: SkipMap<[u8; 64], MyRecord>| {
                run_concurrent(threads, WARMUP_OPS, OPS_PER_THREAD, |ctx| {
                    let base: usize = ctx.tid * OPS_PER_THREAD;
                    let is_write: &Vec<bool> = &write_decisions[ctx.tid];

                    for i in 0..ctx.warmup_ops {
                        let idx: usize = indices[base + (i % OPS_PER_THREAD)];
                        black_box(map.get(&keys[idx]));
                    }

                    ctx.finish_warmup();
                    ctx.begin_measurement();

                    let mut sum: u64 = 0;

                    for i in 0..OPS_PER_THREAD {
                        let idx: usize = indices[base + i];

                        if is_write[i] {
                            map.insert(keys[idx], MyRecord::new(i as u64));
                        } else if let Some(e) = map.get(&keys[idx]) {
                            sum = sum.wrapping_add(e.value().checksum());
                        }
                    }

                    ctx.end_measurement();
                    black_box(sum);
                });
            });
    }

    #[divan::bench(args = THREAD_COUNTS)]
    fn indexset(bencher: Bencher, threads: usize) {
        let keys: Vec<[u8; 64]> = keys::<KEY_SIZE>(N);
        let indices: Vec<usize> = uniform_indices(N, OPS_PER_THREAD * threads, BASE_SEED);
        let write_decisions: Vec<Vec<bool>> = (0..threads)
            .map(|t: usize| {
                shuffled_write_decisions(OPS_PER_THREAD, WRITE_RATIO, BASE_SEED + t as u64)
            })
            .collect();

        bencher
            .counter(ItemsCount::new(threads * OPS_PER_THREAD))
            .with_inputs(|| setup_indexset(&keys))
            .bench_local_values(|map| {
                run_concurrent(threads, WARMUP_OPS, OPS_PER_THREAD, |ctx| {
                    let base: usize = ctx.tid * OPS_PER_THREAD;
                    let is_write: &Vec<bool> = &write_decisions[ctx.tid];

                    for i in 0..ctx.warmup_ops {
                        let idx: usize = indices[base + (i % OPS_PER_THREAD)];
                        black_box(map.get(&keys[idx]));
                    }

                    ctx.finish_warmup();
                    ctx.begin_measurement();

                    let mut sum: u64 = 0;

                    for i in 0..OPS_PER_THREAD {
                        let idx = indices[base + i];

                        if is_write[i] {
                            map.insert(keys[idx], MyRecord::new(i as u64));
                        } else if let Some(r) = map.get(&keys[idx]) {
                            sum = sum.wrapping_add(r.get().value.checksum());
                        }
                    }

                    ctx.end_measurement();
                    black_box(sum);
                });
            });
    }

    #[divan::bench(args = THREAD_COUNTS)]
    fn tree_index(bencher: Bencher, threads: usize) {
        let keys = keys::<KEY_SIZE>(N);
        let indices = uniform_indices(N, OPS_PER_THREAD * threads, BASE_SEED);
        let write_decisions: Vec<Vec<bool>> = (0..threads)
            .map(|t| shuffled_write_decisions(OPS_PER_THREAD, WRITE_RATIO, BASE_SEED + t as u64))
            .collect();

        bencher
            .counter(ItemsCount::new(threads * OPS_PER_THREAD))
            .with_inputs(|| setup_tree_index(&keys))
            .bench_local_values(|tree| {
                run_concurrent(threads, WARMUP_OPS, OPS_PER_THREAD, |ctx| {
                    let guard = SddGuard::new();
                    let base = ctx.tid * OPS_PER_THREAD;
                    let is_write = &write_decisions[ctx.tid];
                    for i in 0..ctx.warmup_ops {
                        let idx = indices[base + (i % OPS_PER_THREAD)];
                        black_box(tree.peek(&keys[idx], &guard));
                    }
                    ctx.finish_warmup();
                    ctx.begin_measurement();
                    let mut sum = 0u64;
                    for i in 0..OPS_PER_THREAD {
                        let idx = indices[base + i];
                        if is_write[i] {
                            // Upsert emulation (see tree_index_upsert_sync doc)
                            tree_index_upsert_sync(&tree, keys[idx], MyRecord::new(i as u64));
                        } else if let Some(v) = tree.peek(&keys[idx], &guard) {
                            sum = sum.wrapping_add(v.checksum());
                        }
                    }
                    ctx.end_measurement();
                    black_box(sum);
                });
            });
    }
}

// =============================================================================
// 02: MIXED 90-10 - Zipfian Access Pattern (Hot Keys)
// =============================================================================

#[divan::bench_group(name = "02_mixed_90_10_zipfian", sample_count = 200, sample_size = 1)]
mod mixed_zipfian {
    use super::*;

    const N: usize = 50_000;
    const OPS_PER_THREAD: usize = 12_500;
    const WRITE_RATIO: usize = 10;

    #[divan::bench(args = THREAD_COUNTS)]
    fn masstree15(bencher: Bencher, threads: usize) {
        let keys = keys::<KEY_SIZE>(N);
        let indices = zipfian_indices(N, OPS_PER_THREAD * threads, 1.0, BASE_SEED);
        let write_decisions: Vec<Vec<bool>> = (0..threads)
            .map(|t| shuffled_write_decisions(OPS_PER_THREAD, WRITE_RATIO, BASE_SEED + t as u64))
            .collect();

        bencher
            .counter(ItemsCount::new(threads * OPS_PER_THREAD))
            .with_inputs(|| setup_masstree15(&keys))
            .bench_local_values(|tree| {
                run_concurrent(threads, WARMUP_OPS, OPS_PER_THREAD, |ctx| {
                    let guard = tree.guard();
                    let base = ctx.tid * OPS_PER_THREAD;
                    let is_write = &write_decisions[ctx.tid];
                    for i in 0..ctx.warmup_ops {
                        let idx = indices[base + (i % OPS_PER_THREAD)];
                        black_box(tree.get_with_guard(&keys[idx], &guard));
                    }
                    ctx.finish_warmup();
                    ctx.begin_measurement();
                    let mut sum = 0u64;
                    for i in 0..OPS_PER_THREAD {
                        let idx = indices[base + i];
                        if is_write[i] {
                            let _ =
                                tree.insert_with_guard(&keys[idx], MyRecord::new(i as u64), &guard);
                        } else if let Some(v) = tree.get_with_guard(&keys[idx], &guard) {
                            sum = sum.wrapping_add(v.checksum());
                        }
                    }
                    ctx.end_measurement();
                    black_box(sum);
                });
            });
    }

    #[divan::bench(args = THREAD_COUNTS)]
    fn skipmap(bencher: Bencher, threads: usize) {
        let keys = keys::<KEY_SIZE>(N);
        let indices = zipfian_indices(N, OPS_PER_THREAD * threads, 1.0, BASE_SEED);
        let write_decisions: Vec<Vec<bool>> = (0..threads)
            .map(|t| shuffled_write_decisions(OPS_PER_THREAD, WRITE_RATIO, BASE_SEED + t as u64))
            .collect();

        bencher
            .counter(ItemsCount::new(threads * OPS_PER_THREAD))
            .with_inputs(|| setup_skipmap(&keys))
            .bench_local_values(|map| {
                run_concurrent(threads, WARMUP_OPS, OPS_PER_THREAD, |ctx| {
                    let base = ctx.tid * OPS_PER_THREAD;
                    let is_write = &write_decisions[ctx.tid];
                    for i in 0..ctx.warmup_ops {
                        let idx = indices[base + (i % OPS_PER_THREAD)];
                        black_box(map.get(&keys[idx]));
                    }
                    ctx.finish_warmup();
                    ctx.begin_measurement();
                    let mut sum = 0u64;
                    for i in 0..OPS_PER_THREAD {
                        let idx = indices[base + i];
                        if is_write[i] {
                            map.insert(keys[idx], MyRecord::new(i as u64));
                        } else if let Some(e) = map.get(&keys[idx]) {
                            sum = sum.wrapping_add(e.value().checksum());
                        }
                    }
                    ctx.end_measurement();
                    black_box(sum);
                });
            });
    }

    #[divan::bench(args = THREAD_COUNTS)]
    fn indexset(bencher: Bencher, threads: usize) {
        let keys = keys::<KEY_SIZE>(N);
        let indices = zipfian_indices(N, OPS_PER_THREAD * threads, 1.0, BASE_SEED);
        let write_decisions: Vec<Vec<bool>> = (0..threads)
            .map(|t| shuffled_write_decisions(OPS_PER_THREAD, WRITE_RATIO, BASE_SEED + t as u64))
            .collect();

        bencher
            .counter(ItemsCount::new(threads * OPS_PER_THREAD))
            .with_inputs(|| setup_indexset(&keys))
            .bench_local_values(|map| {
                run_concurrent(threads, WARMUP_OPS, OPS_PER_THREAD, |ctx| {
                    let base = ctx.tid * OPS_PER_THREAD;
                    let is_write = &write_decisions[ctx.tid];
                    for i in 0..ctx.warmup_ops {
                        let idx = indices[base + (i % OPS_PER_THREAD)];
                        black_box(map.get(&keys[idx]));
                    }
                    ctx.finish_warmup();
                    ctx.begin_measurement();
                    let mut sum = 0u64;
                    for i in 0..OPS_PER_THREAD {
                        let idx = indices[base + i];
                        if is_write[i] {
                            map.insert(keys[idx], MyRecord::new(i as u64));
                        } else if let Some(r) = map.get(&keys[idx]) {
                            sum = sum.wrapping_add(r.get().value.checksum());
                        }
                    }
                    ctx.end_measurement();
                    black_box(sum);
                });
            });
    }

    #[divan::bench(args = THREAD_COUNTS)]
    fn tree_index(bencher: Bencher, threads: usize) {
        let keys = keys::<KEY_SIZE>(N);
        let indices = zipfian_indices(N, OPS_PER_THREAD * threads, 1.0, BASE_SEED);
        let write_decisions: Vec<Vec<bool>> = (0..threads)
            .map(|t| shuffled_write_decisions(OPS_PER_THREAD, WRITE_RATIO, BASE_SEED + t as u64))
            .collect();

        bencher
            .counter(ItemsCount::new(threads * OPS_PER_THREAD))
            .with_inputs(|| setup_tree_index(&keys))
            .bench_local_values(|tree| {
                run_concurrent(threads, WARMUP_OPS, OPS_PER_THREAD, |ctx| {
                    let guard = SddGuard::new();
                    let base = ctx.tid * OPS_PER_THREAD;
                    let is_write = &write_decisions[ctx.tid];
                    for i in 0..ctx.warmup_ops {
                        let idx = indices[base + (i % OPS_PER_THREAD)];
                        black_box(tree.peek(&keys[idx], &guard));
                    }
                    ctx.finish_warmup();
                    ctx.begin_measurement();
                    let mut sum = 0u64;
                    for i in 0..OPS_PER_THREAD {
                        let idx = indices[base + i];
                        if is_write[i] {
                            // Upsert emulation (see tree_index_upsert_sync doc)
                            tree_index_upsert_sync(&tree, keys[idx], MyRecord::new(i as u64));
                        } else if let Some(v) = tree.peek(&keys[idx], &guard) {
                            sum = sum.wrapping_add(v.checksum());
                        }
                    }
                    ctx.end_measurement();
                    black_box(sum);
                });
            });
    }
}

// =============================================================================
// 03: MIXED 90-10 - Shared Prefix (Masstree Stress Test)
// =============================================================================

#[divan::bench_group(
    name = "03_mixed_90_10_shared_prefix",
    sample_count = 200,
    sample_size = 1
)]
mod mixed_shared_prefix {
    use super::*;

    const N: usize = 50_000;
    const OPS_PER_THREAD: usize = 12_500;
    const WRITE_RATIO: usize = 10;
    const PREFIX_CHUNKS: usize = 3; // First 24 bytes shared
    const PREFIX_BUCKETS: u64 = 256;

    fn prefix_keys() -> Vec<[u8; KEY_SIZE]> {
        keys_shared_prefix_chunks::<KEY_SIZE>(N, PREFIX_CHUNKS, PREFIX_BUCKETS)
    }

    #[divan::bench(args = THREAD_COUNTS)]
    fn masstree15(bencher: Bencher, threads: usize) {
        let keys = prefix_keys();
        let indices = uniform_indices(N, OPS_PER_THREAD * threads, BASE_SEED);
        let write_decisions: Vec<Vec<bool>> = (0..threads)
            .map(|t| shuffled_write_decisions(OPS_PER_THREAD, WRITE_RATIO, BASE_SEED + t as u64))
            .collect();

        bencher
            .counter(ItemsCount::new(threads * OPS_PER_THREAD))
            .with_inputs(|| setup_masstree15(&keys))
            .bench_local_values(|tree| {
                run_concurrent(threads, WARMUP_OPS, OPS_PER_THREAD, |ctx| {
                    let guard = tree.guard();
                    let base = ctx.tid * OPS_PER_THREAD;
                    let is_write = &write_decisions[ctx.tid];
                    for i in 0..ctx.warmup_ops {
                        let idx = indices[base + (i % OPS_PER_THREAD)];
                        black_box(tree.get_with_guard(&keys[idx], &guard));
                    }
                    ctx.finish_warmup();
                    ctx.begin_measurement();
                    let mut sum = 0u64;
                    for i in 0..OPS_PER_THREAD {
                        let idx = indices[base + i];
                        if is_write[i] {
                            let _ =
                                tree.insert_with_guard(&keys[idx], MyRecord::new(i as u64), &guard);
                        } else if let Some(v) = tree.get_with_guard(&keys[idx], &guard) {
                            sum = sum.wrapping_add(v.checksum());
                        }
                    }
                    ctx.end_measurement();
                    black_box(sum);
                });
            });
    }

    #[divan::bench(args = THREAD_COUNTS)]
    fn skipmap(bencher: Bencher, threads: usize) {
        let keys = prefix_keys();
        let indices = uniform_indices(N, OPS_PER_THREAD * threads, BASE_SEED);
        let write_decisions: Vec<Vec<bool>> = (0..threads)
            .map(|t| shuffled_write_decisions(OPS_PER_THREAD, WRITE_RATIO, BASE_SEED + t as u64))
            .collect();

        bencher
            .counter(ItemsCount::new(threads * OPS_PER_THREAD))
            .with_inputs(|| setup_skipmap(&keys))
            .bench_local_values(|map| {
                run_concurrent(threads, WARMUP_OPS, OPS_PER_THREAD, |ctx| {
                    let base = ctx.tid * OPS_PER_THREAD;
                    let is_write = &write_decisions[ctx.tid];
                    for i in 0..ctx.warmup_ops {
                        let idx = indices[base + (i % OPS_PER_THREAD)];
                        black_box(map.get(&keys[idx]));
                    }
                    ctx.finish_warmup();
                    ctx.begin_measurement();
                    let mut sum = 0u64;
                    for i in 0..OPS_PER_THREAD {
                        let idx = indices[base + i];
                        if is_write[i] {
                            map.insert(keys[idx], MyRecord::new(i as u64));
                        } else if let Some(e) = map.get(&keys[idx]) {
                            sum = sum.wrapping_add(e.value().checksum());
                        }
                    }
                    ctx.end_measurement();
                    black_box(sum);
                });
            });
    }

    #[divan::bench(args = THREAD_COUNTS)]
    fn indexset(bencher: Bencher, threads: usize) {
        let keys = prefix_keys();
        let indices = uniform_indices(N, OPS_PER_THREAD * threads, BASE_SEED);
        let write_decisions: Vec<Vec<bool>> = (0..threads)
            .map(|t| shuffled_write_decisions(OPS_PER_THREAD, WRITE_RATIO, BASE_SEED + t as u64))
            .collect();

        bencher
            .counter(ItemsCount::new(threads * OPS_PER_THREAD))
            .with_inputs(|| setup_indexset(&keys))
            .bench_local_values(|map| {
                run_concurrent(threads, WARMUP_OPS, OPS_PER_THREAD, |ctx| {
                    let base = ctx.tid * OPS_PER_THREAD;
                    let is_write = &write_decisions[ctx.tid];
                    for i in 0..ctx.warmup_ops {
                        let idx = indices[base + (i % OPS_PER_THREAD)];
                        black_box(map.get(&keys[idx]));
                    }
                    ctx.finish_warmup();
                    ctx.begin_measurement();
                    let mut sum = 0u64;
                    for i in 0..OPS_PER_THREAD {
                        let idx = indices[base + i];
                        if is_write[i] {
                            map.insert(keys[idx], MyRecord::new(i as u64));
                        } else if let Some(r) = map.get(&keys[idx]) {
                            sum = sum.wrapping_add(r.get().value.checksum());
                        }
                    }
                    ctx.end_measurement();
                    black_box(sum);
                });
            });
    }

    #[divan::bench(args = THREAD_COUNTS)]
    fn tree_index(bencher: Bencher, threads: usize) {
        let keys = prefix_keys();
        let indices = uniform_indices(N, OPS_PER_THREAD * threads, BASE_SEED);
        let write_decisions: Vec<Vec<bool>> = (0..threads)
            .map(|t| shuffled_write_decisions(OPS_PER_THREAD, WRITE_RATIO, BASE_SEED + t as u64))
            .collect();

        bencher
            .counter(ItemsCount::new(threads * OPS_PER_THREAD))
            .with_inputs(|| setup_tree_index(&keys))
            .bench_local_values(|tree| {
                run_concurrent(threads, WARMUP_OPS, OPS_PER_THREAD, |ctx| {
                    let guard = SddGuard::new();
                    let base = ctx.tid * OPS_PER_THREAD;
                    let is_write = &write_decisions[ctx.tid];
                    for i in 0..ctx.warmup_ops {
                        let idx = indices[base + (i % OPS_PER_THREAD)];
                        black_box(tree.peek(&keys[idx], &guard));
                    }
                    ctx.finish_warmup();
                    ctx.begin_measurement();
                    let mut sum = 0u64;
                    for i in 0..OPS_PER_THREAD {
                        let idx = indices[base + i];
                        if is_write[i] {
                            // Upsert emulation (see tree_index_upsert_sync doc)
                            tree_index_upsert_sync(&tree, keys[idx], MyRecord::new(i as u64));
                        } else if let Some(v) = tree.peek(&keys[idx], &guard) {
                            sum = sum.wrapping_add(v.checksum());
                        }
                    }
                    ctx.end_measurement();
                    black_box(sum);
                });
            });
    }
}

// =============================================================================
// 04: HIGH CONTENTION - Small Key Space (500 keys)
// =============================================================================

#[divan::bench_group(
    name = "04_mixed_90_10_high_contention",
    sample_count = 200,
    sample_size = 1
)]
mod mixed_high_contention {
    use super::*;

    const N: usize = 500; // Small key space = high contention
    const OPS_PER_THREAD: usize = 25_000;
    const WRITE_RATIO: usize = 10;

    #[divan::bench(args = THREAD_COUNTS)]
    fn masstree15(bencher: Bencher, threads: usize) {
        let keys = keys::<KEY_SIZE>(N);
        let indices = uniform_indices(N, OPS_PER_THREAD * threads, BASE_SEED);
        let write_decisions: Vec<Vec<bool>> = (0..threads)
            .map(|t| shuffled_write_decisions(OPS_PER_THREAD, WRITE_RATIO, BASE_SEED + t as u64))
            .collect();

        bencher
            .counter(ItemsCount::new(threads * OPS_PER_THREAD))
            .with_inputs(|| setup_masstree15(&keys))
            .bench_local_values(|tree| {
                run_concurrent(threads, WARMUP_OPS, OPS_PER_THREAD, |ctx| {
                    let guard = tree.guard();
                    let base = ctx.tid * OPS_PER_THREAD;
                    let is_write = &write_decisions[ctx.tid];
                    for i in 0..ctx.warmup_ops {
                        let idx = indices[base + (i % OPS_PER_THREAD)];
                        black_box(tree.get_with_guard(&keys[idx], &guard));
                    }
                    ctx.finish_warmup();
                    ctx.begin_measurement();
                    let mut sum = 0u64;
                    for i in 0..OPS_PER_THREAD {
                        let idx = indices[base + i];
                        if is_write[i] {
                            let _ =
                                tree.insert_with_guard(&keys[idx], MyRecord::new(i as u64), &guard);
                        } else if let Some(v) = tree.get_with_guard(&keys[idx], &guard) {
                            sum = sum.wrapping_add(v.checksum());
                        }
                    }
                    ctx.end_measurement();
                    black_box(sum);
                });
            });
    }

    #[divan::bench(args = THREAD_COUNTS)]
    fn skipmap(bencher: Bencher, threads: usize) {
        let keys = keys::<KEY_SIZE>(N);
        let indices = uniform_indices(N, OPS_PER_THREAD * threads, BASE_SEED);
        let write_decisions: Vec<Vec<bool>> = (0..threads)
            .map(|t| shuffled_write_decisions(OPS_PER_THREAD, WRITE_RATIO, BASE_SEED + t as u64))
            .collect();

        bencher
            .counter(ItemsCount::new(threads * OPS_PER_THREAD))
            .with_inputs(|| setup_skipmap(&keys))
            .bench_local_values(|map| {
                run_concurrent(threads, WARMUP_OPS, OPS_PER_THREAD, |ctx| {
                    let base = ctx.tid * OPS_PER_THREAD;
                    let is_write = &write_decisions[ctx.tid];
                    for i in 0..ctx.warmup_ops {
                        let idx = indices[base + (i % OPS_PER_THREAD)];
                        black_box(map.get(&keys[idx]));
                    }
                    ctx.finish_warmup();
                    ctx.begin_measurement();
                    let mut sum = 0u64;
                    for i in 0..OPS_PER_THREAD {
                        let idx = indices[base + i];
                        if is_write[i] {
                            map.insert(keys[idx], MyRecord::new(i as u64));
                        } else if let Some(e) = map.get(&keys[idx]) {
                            sum = sum.wrapping_add(e.value().checksum());
                        }
                    }
                    ctx.end_measurement();
                    black_box(sum);
                });
            });
    }

    #[divan::bench(args = THREAD_COUNTS)]
    fn indexset(bencher: Bencher, threads: usize) {
        let keys = keys::<KEY_SIZE>(N);
        let indices = uniform_indices(N, OPS_PER_THREAD * threads, BASE_SEED);
        let write_decisions: Vec<Vec<bool>> = (0..threads)
            .map(|t| shuffled_write_decisions(OPS_PER_THREAD, WRITE_RATIO, BASE_SEED + t as u64))
            .collect();

        bencher
            .counter(ItemsCount::new(threads * OPS_PER_THREAD))
            .with_inputs(|| setup_indexset(&keys))
            .bench_local_values(|map| {
                run_concurrent(threads, WARMUP_OPS, OPS_PER_THREAD, |ctx| {
                    let base = ctx.tid * OPS_PER_THREAD;
                    let is_write = &write_decisions[ctx.tid];
                    for i in 0..ctx.warmup_ops {
                        let idx = indices[base + (i % OPS_PER_THREAD)];
                        black_box(map.get(&keys[idx]));
                    }
                    ctx.finish_warmup();
                    ctx.begin_measurement();
                    let mut sum = 0u64;
                    for i in 0..OPS_PER_THREAD {
                        let idx = indices[base + i];
                        if is_write[i] {
                            map.insert(keys[idx], MyRecord::new(i as u64));
                        } else if let Some(r) = map.get(&keys[idx]) {
                            sum = sum.wrapping_add(r.get().value.checksum());
                        }
                    }
                    ctx.end_measurement();
                    black_box(sum);
                });
            });
    }

    #[divan::bench(args = THREAD_COUNTS)]
    fn tree_index(bencher: Bencher, threads: usize) {
        let keys = keys::<KEY_SIZE>(N);
        let indices = uniform_indices(N, OPS_PER_THREAD * threads, BASE_SEED);
        let write_decisions: Vec<Vec<bool>> = (0..threads)
            .map(|t| shuffled_write_decisions(OPS_PER_THREAD, WRITE_RATIO, BASE_SEED + t as u64))
            .collect();

        bencher
            .counter(ItemsCount::new(threads * OPS_PER_THREAD))
            .with_inputs(|| setup_tree_index(&keys))
            .bench_local_values(|tree| {
                run_concurrent(threads, WARMUP_OPS, OPS_PER_THREAD, |ctx| {
                    let guard = SddGuard::new();
                    let base = ctx.tid * OPS_PER_THREAD;
                    let is_write = &write_decisions[ctx.tid];
                    for i in 0..ctx.warmup_ops {
                        let idx = indices[base + (i % OPS_PER_THREAD)];
                        black_box(tree.peek(&keys[idx], &guard));
                    }
                    ctx.finish_warmup();
                    ctx.begin_measurement();
                    let mut sum = 0u64;
                    for i in 0..OPS_PER_THREAD {
                        let idx = indices[base + i];
                        if is_write[i] {
                            // Upsert emulation (see tree_index_upsert_sync doc)
                            tree_index_upsert_sync(&tree, keys[idx], MyRecord::new(i as u64));
                        } else if let Some(v) = tree.peek(&keys[idx], &guard) {
                            sum = sum.wrapping_add(v.checksum());
                        }
                    }
                    ctx.end_measurement();
                    black_box(sum);
                });
            });
    }
}

// =============================================================================
// 05: LARGE DATASET - 500K keys
// =============================================================================

#[divan::bench_group(
    name = "05_mixed_90_10_large_dataset",
    sample_count = 10,
    sample_size = 1
)]
mod mixed_large_dataset {
    use super::*;

    const N: usize = 500_000;
    const OPS_PER_THREAD: usize = 25_000;
    const WRITE_RATIO: usize = 10;

    #[divan::bench(args = THREAD_COUNTS)]
    fn masstree15(bencher: Bencher, threads: usize) {
        let keys = keys::<KEY_SIZE>(N);
        let indices = uniform_indices(N, OPS_PER_THREAD * threads, BASE_SEED);
        let write_decisions: Vec<Vec<bool>> = (0..threads)
            .map(|t| shuffled_write_decisions(OPS_PER_THREAD, WRITE_RATIO, BASE_SEED + t as u64))
            .collect();

        bencher
            .counter(ItemsCount::new(threads * OPS_PER_THREAD))
            .with_inputs(|| setup_masstree15(&keys))
            .bench_local_values(|tree| {
                run_concurrent(threads, WARMUP_OPS, OPS_PER_THREAD, |ctx| {
                    let guard = tree.guard();
                    let base = ctx.tid * OPS_PER_THREAD;
                    let is_write = &write_decisions[ctx.tid];
                    for i in 0..ctx.warmup_ops {
                        let idx = indices[base + (i % OPS_PER_THREAD)];
                        black_box(tree.get_with_guard(&keys[idx], &guard));
                    }
                    ctx.finish_warmup();
                    ctx.begin_measurement();
                    let mut sum = 0u64;
                    for i in 0..OPS_PER_THREAD {
                        let idx = indices[base + i];
                        if is_write[i] {
                            let _ =
                                tree.insert_with_guard(&keys[idx], MyRecord::new(i as u64), &guard);
                        } else if let Some(v) = tree.get_with_guard(&keys[idx], &guard) {
                            sum = sum.wrapping_add(v.checksum());
                        }
                    }
                    ctx.end_measurement();
                    black_box(sum);
                });
            });
    }

    #[divan::bench(args = THREAD_COUNTS)]
    fn skipmap(bencher: Bencher, threads: usize) {
        let keys = keys::<KEY_SIZE>(N);
        let indices = uniform_indices(N, OPS_PER_THREAD * threads, BASE_SEED);
        let write_decisions: Vec<Vec<bool>> = (0..threads)
            .map(|t| shuffled_write_decisions(OPS_PER_THREAD, WRITE_RATIO, BASE_SEED + t as u64))
            .collect();

        bencher
            .counter(ItemsCount::new(threads * OPS_PER_THREAD))
            .with_inputs(|| setup_skipmap(&keys))
            .bench_local_values(|map| {
                run_concurrent(threads, WARMUP_OPS, OPS_PER_THREAD, |ctx| {
                    let base = ctx.tid * OPS_PER_THREAD;
                    let is_write = &write_decisions[ctx.tid];
                    for i in 0..ctx.warmup_ops {
                        let idx = indices[base + (i % OPS_PER_THREAD)];
                        black_box(map.get(&keys[idx]));
                    }
                    ctx.finish_warmup();
                    ctx.begin_measurement();
                    let mut sum = 0u64;
                    for i in 0..OPS_PER_THREAD {
                        let idx = indices[base + i];
                        if is_write[i] {
                            map.insert(keys[idx], MyRecord::new(i as u64));
                        } else if let Some(e) = map.get(&keys[idx]) {
                            sum = sum.wrapping_add(e.value().checksum());
                        }
                    }
                    ctx.end_measurement();
                    black_box(sum);
                });
            });
    }

    #[divan::bench(args = THREAD_COUNTS)]
    fn indexset(bencher: Bencher, threads: usize) {
        let keys = keys::<KEY_SIZE>(N);
        let indices = uniform_indices(N, OPS_PER_THREAD * threads, BASE_SEED);
        let write_decisions: Vec<Vec<bool>> = (0..threads)
            .map(|t| shuffled_write_decisions(OPS_PER_THREAD, WRITE_RATIO, BASE_SEED + t as u64))
            .collect();

        bencher
            .counter(ItemsCount::new(threads * OPS_PER_THREAD))
            .with_inputs(|| setup_indexset(&keys))
            .bench_local_values(|map| {
                run_concurrent(threads, WARMUP_OPS, OPS_PER_THREAD, |ctx| {
                    let base = ctx.tid * OPS_PER_THREAD;
                    let is_write = &write_decisions[ctx.tid];
                    for i in 0..ctx.warmup_ops {
                        let idx = indices[base + (i % OPS_PER_THREAD)];
                        black_box(map.get(&keys[idx]));
                    }
                    ctx.finish_warmup();
                    ctx.begin_measurement();
                    let mut sum = 0u64;
                    for i in 0..OPS_PER_THREAD {
                        let idx = indices[base + i];
                        if is_write[i] {
                            map.insert(keys[idx], MyRecord::new(i as u64));
                        } else if let Some(r) = map.get(&keys[idx]) {
                            sum = sum.wrapping_add(r.get().value.checksum());
                        }
                    }
                    ctx.end_measurement();
                    black_box(sum);
                });
            });
    }

    #[divan::bench(args = THREAD_COUNTS)]
    fn tree_index(bencher: Bencher, threads: usize) {
        let keys = keys::<KEY_SIZE>(N);
        let indices = uniform_indices(N, OPS_PER_THREAD * threads, BASE_SEED);
        let write_decisions: Vec<Vec<bool>> = (0..threads)
            .map(|t| shuffled_write_decisions(OPS_PER_THREAD, WRITE_RATIO, BASE_SEED + t as u64))
            .collect();

        bencher
            .counter(ItemsCount::new(threads * OPS_PER_THREAD))
            .with_inputs(|| setup_tree_index(&keys))
            .bench_local_values(|tree| {
                run_concurrent(threads, WARMUP_OPS, OPS_PER_THREAD, |ctx| {
                    let guard = SddGuard::new();
                    let base = ctx.tid * OPS_PER_THREAD;
                    let is_write = &write_decisions[ctx.tid];
                    for i in 0..ctx.warmup_ops {
                        let idx = indices[base + (i % OPS_PER_THREAD)];
                        black_box(tree.peek(&keys[idx], &guard));
                    }
                    ctx.finish_warmup();
                    ctx.begin_measurement();
                    let mut sum = 0u64;
                    for i in 0..OPS_PER_THREAD {
                        let idx = indices[base + i];
                        if is_write[i] {
                            // Upsert emulation (see tree_index_upsert_sync doc)
                            tree_index_upsert_sync(&tree, keys[idx], MyRecord::new(i as u64));
                        } else if let Some(v) = tree.peek(&keys[idx], &guard) {
                            sum = sum.wrapping_add(v.checksum());
                        }
                    }
                    ctx.end_measurement();
                    black_box(sum);
                });
            });
    }
}

// =============================================================================
// 06: SINGLE HOT KEY - Maximum Contention
// =============================================================================

#[divan::bench_group(name = "06_single_hot_key", sample_count = 200, sample_size = 1)]
mod single_hot_key {
    use super::*;

    const N: usize = 50_000;
    const OPS_PER_THREAD: usize = 5_000;
    const WRITE_RATIO: usize = 10;

    #[divan::bench(args = THREAD_COUNTS)]
    fn masstree15(bencher: Bencher, threads: usize) {
        let keys = keys::<KEY_SIZE>(N);
        let hot_key = keys[N / 2];
        let write_decisions: Vec<Vec<bool>> = (0..threads)
            .map(|t| shuffled_write_decisions(OPS_PER_THREAD, WRITE_RATIO, BASE_SEED + t as u64))
            .collect();

        bencher
            .counter(ItemsCount::new(threads * OPS_PER_THREAD))
            .with_inputs(|| setup_masstree15(&keys))
            .bench_local_values(|tree| {
                run_concurrent(threads, WARMUP_OPS, OPS_PER_THREAD, |ctx| {
                    let guard = tree.guard();
                    for _ in 0..ctx.warmup_ops {
                        black_box(tree.get_with_guard(&hot_key, &guard));
                    }
                    ctx.finish_warmup();
                    ctx.begin_measurement();
                    let mut sum = 0u64;
                    (0..OPS_PER_THREAD).for_each(|i| {
                        if write_decisions[ctx.tid][i] {
                            let _ = tree.insert_with_guard(
                                &hot_key,
                                MyRecord::new((ctx.tid * OPS_PER_THREAD + i) as u64),
                                &guard,
                            );
                        } else if let Some(v) = tree.get_with_guard(&hot_key, &guard) {
                            sum = sum.wrapping_add(v.checksum());
                        }
                    });
                    ctx.end_measurement();
                    black_box(sum);
                });
            });
    }

    #[divan::bench(args = THREAD_COUNTS)]
    fn skipmap(bencher: Bencher, threads: usize) {
        let keys = keys::<KEY_SIZE>(N);
        let hot_key = keys[N / 2];
        let write_decisions: Vec<Vec<bool>> = (0..threads)
            .map(|t| shuffled_write_decisions(OPS_PER_THREAD, WRITE_RATIO, BASE_SEED + t as u64))
            .collect();

        bencher
            .counter(ItemsCount::new(threads * OPS_PER_THREAD))
            .with_inputs(|| setup_skipmap(&keys))
            .bench_local_values(|map| {
                run_concurrent(threads, WARMUP_OPS, OPS_PER_THREAD, |ctx| {
                    for _ in 0..ctx.warmup_ops {
                        black_box(map.get(&hot_key));
                    }
                    ctx.finish_warmup();
                    ctx.begin_measurement();
                    let mut sum = 0u64;
                    (0..OPS_PER_THREAD).for_each(|i| {
                        if write_decisions[ctx.tid][i] {
                            map.insert(
                                hot_key,
                                MyRecord::new((ctx.tid * OPS_PER_THREAD + i) as u64),
                            );
                        } else if let Some(e) = map.get(&hot_key) {
                            sum = sum.wrapping_add(e.value().checksum());
                        }
                    });
                    ctx.end_measurement();
                    black_box(sum);
                });
            });
    }

    #[divan::bench(args = THREAD_COUNTS)]
    fn indexset(bencher: Bencher, threads: usize) {
        let keys = keys::<KEY_SIZE>(N);
        let hot_key = keys[N / 2];
        let write_decisions: Vec<Vec<bool>> = (0..threads)
            .map(|t| shuffled_write_decisions(OPS_PER_THREAD, WRITE_RATIO, BASE_SEED + t as u64))
            .collect();

        bencher
            .counter(ItemsCount::new(threads * OPS_PER_THREAD))
            .with_inputs(|| setup_indexset(&keys))
            .bench_local_values(|map| {
                run_concurrent(threads, WARMUP_OPS, OPS_PER_THREAD, |ctx| {
                    for _ in 0..ctx.warmup_ops {
                        black_box(map.get(&hot_key));
                    }
                    ctx.finish_warmup();
                    ctx.begin_measurement();
                    let mut sum = 0u64;
                    (0..OPS_PER_THREAD).for_each(|i| {
                        if write_decisions[ctx.tid][i] {
                            map.insert(
                                hot_key,
                                MyRecord::new((ctx.tid * OPS_PER_THREAD + i) as u64),
                            );
                        } else if let Some(r) = map.get(&hot_key) {
                            sum = sum.wrapping_add(r.get().value.checksum());
                        }
                    });
                    ctx.end_measurement();
                    black_box(sum);
                });
            });
    }

    #[divan::bench(args = THREAD_COUNTS)]
    fn tree_index(bencher: Bencher, threads: usize) {
        let keys = keys::<KEY_SIZE>(N);
        let hot_key = keys[N / 2];
        let write_decisions: Vec<Vec<bool>> = (0..threads)
            .map(|t| shuffled_write_decisions(OPS_PER_THREAD, WRITE_RATIO, BASE_SEED + t as u64))
            .collect();

        bencher
            .counter(ItemsCount::new(threads * OPS_PER_THREAD))
            .with_inputs(|| setup_tree_index(&keys))
            .bench_local_values(|tree| {
                run_concurrent(threads, WARMUP_OPS, OPS_PER_THREAD, |ctx| {
                    let guard = SddGuard::new();
                    for _ in 0..ctx.warmup_ops {
                        black_box(tree.peek(&hot_key, &guard));
                    }
                    ctx.finish_warmup();
                    ctx.begin_measurement();
                    let mut sum = 0u64;
                    (0..OPS_PER_THREAD).for_each(|i| {
                        if write_decisions[ctx.tid][i] {
                            // Upsert emulation (see tree_index_upsert_sync doc)
                            tree_index_upsert_sync(
                                &tree,
                                hot_key,
                                MyRecord::new((ctx.tid * OPS_PER_THREAD + i) as u64),
                            );
                        } else if let Some(v) = tree.peek(&hot_key, &guard) {
                            sum = sum.wrapping_add(v.checksum());
                        }
                    });
                    ctx.end_measurement();
                    black_box(sum);
                });
            });
    }
}

// =============================================================================
// 07: WRITE-HEAVY - 50% reads, 50% writes
// =============================================================================

#[divan::bench_group(name = "07_mixed_50_50", sample_count = 200, sample_size = 1)]
mod mixed_50_50 {
    use super::*;

    const N: usize = 50_000;
    const OPS_PER_THREAD: usize = 12_500;
    const WRITE_RATIO: usize = 50; // 50% writes

    #[divan::bench(args = THREAD_COUNTS)]
    fn masstree15(bencher: Bencher, threads: usize) {
        let keys = keys::<KEY_SIZE>(N);
        let indices = uniform_indices(N, OPS_PER_THREAD * threads, BASE_SEED);
        let write_decisions: Vec<Vec<bool>> = (0..threads)
            .map(|t| shuffled_write_decisions(OPS_PER_THREAD, WRITE_RATIO, BASE_SEED + t as u64))
            .collect();

        bencher
            .counter(ItemsCount::new(threads * OPS_PER_THREAD))
            .with_inputs(|| setup_masstree15(&keys))
            .bench_local_values(|tree| {
                run_concurrent(threads, WARMUP_OPS, OPS_PER_THREAD, |ctx| {
                    let guard = tree.guard();
                    let base = ctx.tid * OPS_PER_THREAD;
                    let is_write = &write_decisions[ctx.tid];
                    for i in 0..ctx.warmup_ops {
                        let idx = indices[base + (i % OPS_PER_THREAD)];
                        black_box(tree.get_with_guard(&keys[idx], &guard));
                    }
                    ctx.finish_warmup();
                    ctx.begin_measurement();
                    let mut sum = 0u64;
                    for i in 0..OPS_PER_THREAD {
                        let idx = indices[base + i];
                        if is_write[i] {
                            let _ =
                                tree.insert_with_guard(&keys[idx], MyRecord::new(i as u64), &guard);
                        } else if let Some(v) = tree.get_with_guard(&keys[idx], &guard) {
                            sum = sum.wrapping_add(v.checksum());
                        }
                    }
                    ctx.end_measurement();
                    black_box(sum);
                });
            });
    }

    #[divan::bench(args = THREAD_COUNTS)]
    fn skipmap(bencher: Bencher, threads: usize) {
        let keys = keys::<KEY_SIZE>(N);
        let indices = uniform_indices(N, OPS_PER_THREAD * threads, BASE_SEED);
        let write_decisions: Vec<Vec<bool>> = (0..threads)
            .map(|t| shuffled_write_decisions(OPS_PER_THREAD, WRITE_RATIO, BASE_SEED + t as u64))
            .collect();

        bencher
            .counter(ItemsCount::new(threads * OPS_PER_THREAD))
            .with_inputs(|| setup_skipmap(&keys))
            .bench_local_values(|map| {
                run_concurrent(threads, WARMUP_OPS, OPS_PER_THREAD, |ctx| {
                    let base = ctx.tid * OPS_PER_THREAD;
                    let is_write = &write_decisions[ctx.tid];
                    for i in 0..ctx.warmup_ops {
                        let idx = indices[base + (i % OPS_PER_THREAD)];
                        black_box(map.get(&keys[idx]));
                    }
                    ctx.finish_warmup();
                    ctx.begin_measurement();
                    let mut sum = 0u64;
                    for i in 0..OPS_PER_THREAD {
                        let idx = indices[base + i];
                        if is_write[i] {
                            map.insert(keys[idx], MyRecord::new(i as u64));
                        } else if let Some(e) = map.get(&keys[idx]) {
                            sum = sum.wrapping_add(e.value().checksum());
                        }
                    }
                    ctx.end_measurement();
                    black_box(sum);
                });
            });
    }

    #[divan::bench(args = THREAD_COUNTS)]
    fn indexset(bencher: Bencher, threads: usize) {
        let keys = keys::<KEY_SIZE>(N);
        let indices = uniform_indices(N, OPS_PER_THREAD * threads, BASE_SEED);
        let write_decisions: Vec<Vec<bool>> = (0..threads)
            .map(|t| shuffled_write_decisions(OPS_PER_THREAD, WRITE_RATIO, BASE_SEED + t as u64))
            .collect();

        bencher
            .counter(ItemsCount::new(threads * OPS_PER_THREAD))
            .with_inputs(|| setup_indexset(&keys))
            .bench_local_values(|map| {
                run_concurrent(threads, WARMUP_OPS, OPS_PER_THREAD, |ctx| {
                    let base = ctx.tid * OPS_PER_THREAD;
                    let is_write = &write_decisions[ctx.tid];
                    for i in 0..ctx.warmup_ops {
                        let idx = indices[base + (i % OPS_PER_THREAD)];
                        black_box(map.get(&keys[idx]));
                    }
                    ctx.finish_warmup();
                    ctx.begin_measurement();
                    let mut sum = 0u64;
                    for i in 0..OPS_PER_THREAD {
                        let idx = indices[base + i];
                        if is_write[i] {
                            map.insert(keys[idx], MyRecord::new(i as u64));
                        } else if let Some(r) = map.get(&keys[idx]) {
                            sum = sum.wrapping_add(r.get().value.checksum());
                        }
                    }
                    ctx.end_measurement();
                    black_box(sum);
                });
            });
    }

    #[divan::bench(args = THREAD_COUNTS)]
    fn tree_index(bencher: Bencher, threads: usize) {
        let keys = keys::<KEY_SIZE>(N);
        let indices = uniform_indices(N, OPS_PER_THREAD * threads, BASE_SEED);
        let write_decisions: Vec<Vec<bool>> = (0..threads)
            .map(|t| shuffled_write_decisions(OPS_PER_THREAD, WRITE_RATIO, BASE_SEED + t as u64))
            .collect();

        bencher
            .counter(ItemsCount::new(threads * OPS_PER_THREAD))
            .with_inputs(|| setup_tree_index(&keys))
            .bench_local_values(|tree| {
                run_concurrent(threads, WARMUP_OPS, OPS_PER_THREAD, |ctx| {
                    let guard = SddGuard::new();
                    let base = ctx.tid * OPS_PER_THREAD;
                    let is_write = &write_decisions[ctx.tid];
                    for i in 0..ctx.warmup_ops {
                        let idx = indices[base + (i % OPS_PER_THREAD)];
                        black_box(tree.peek(&keys[idx], &guard));
                    }
                    ctx.finish_warmup();
                    ctx.begin_measurement();
                    let mut sum = 0u64;
                    for i in 0..OPS_PER_THREAD {
                        let idx = indices[base + i];
                        if is_write[i] {
                            // Upsert emulation (see tree_index_upsert_sync doc)
                            tree_index_upsert_sync(&tree, keys[idx], MyRecord::new(i as u64));
                        } else if let Some(v) = tree.peek(&keys[idx], &guard) {
                            sum = sum.wrapping_add(v.checksum());
                        }
                    }
                    ctx.end_measurement();
                    black_box(sum);
                });
            });
    }
}

// =============================================================================
// 08: 8-BYTE KEYS - MassTree Single-Layer Fast Path
// =============================================================================

#[divan::bench_group(name = "08_8byte_keys_uniform", sample_count = 200, sample_size = 1)]
mod keys_8byte {
    use super::*;

    const KEY_SIZE_8: usize = 8;
    const N: usize = 50_000;
    const OPS_PER_THREAD: usize = 12_500;
    const WRITE_RATIO: usize = 10;

    fn setup_masstree15_8(keys: &[[u8; KEY_SIZE_8]]) -> MassTree15Inline<MyRecord> {
        let tree = MassTree15Inline::new();
        {
            let guard = tree.guard();
            for (i, key) in keys.iter().enumerate() {
                let _ = tree.insert_with_guard(key, MyRecord::new(i as u64), &guard);
            }
        }
        tree
    }

    fn setup_skipmap_8(keys: &[[u8; KEY_SIZE_8]]) -> SkipMap<[u8; KEY_SIZE_8], MyRecord> {
        let map = SkipMap::new();
        for (i, key) in keys.iter().enumerate() {
            map.insert(*key, MyRecord::new(i as u64));
        }
        map
    }

    fn setup_indexset_8(keys: &[[u8; KEY_SIZE_8]]) -> IndexSetBTreeMap<[u8; KEY_SIZE_8], MyRecord> {
        let map = IndexSetBTreeMap::new();
        for (i, key) in keys.iter().enumerate() {
            map.insert(*key, MyRecord::new(i as u64));
        }
        map
    }

    fn setup_tree_index_8(keys: &[[u8; KEY_SIZE_8]]) -> TreeIndex<[u8; KEY_SIZE_8], MyRecord> {
        let tree = TreeIndex::new();
        for (i, key) in keys.iter().enumerate() {
            let _ = tree.insert_sync(*key, MyRecord::new(i as u64));
        }
        tree
    }

    /// 8-byte key variant of upsert emulation.
    fn tree_index_upsert_sync_8(
        tree: &TreeIndex<[u8; KEY_SIZE_8], MyRecord>,
        key: [u8; KEY_SIZE_8],
        value: MyRecord,
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

    #[divan::bench(args = THREAD_COUNTS)]
    fn masstree15(bencher: Bencher, threads: usize) {
        let keys = keys::<KEY_SIZE_8>(N);
        let indices = uniform_indices(N, OPS_PER_THREAD * threads, BASE_SEED);
        let write_decisions: Vec<Vec<bool>> = (0..threads)
            .map(|t| shuffled_write_decisions(OPS_PER_THREAD, WRITE_RATIO, BASE_SEED + t as u64))
            .collect();

        bencher
            .counter(ItemsCount::new(threads * OPS_PER_THREAD))
            .with_inputs(|| setup_masstree15_8(&keys))
            .bench_local_values(|tree| {
                run_concurrent(threads, WARMUP_OPS, OPS_PER_THREAD, |ctx| {
                    let guard = tree.guard();
                    let base = ctx.tid * OPS_PER_THREAD;
                    let is_write = &write_decisions[ctx.tid];
                    for i in 0..ctx.warmup_ops {
                        let idx = indices[base + (i % OPS_PER_THREAD)];
                        black_box(tree.get_with_guard(&keys[idx], &guard));
                    }
                    ctx.finish_warmup();
                    ctx.begin_measurement();
                    let mut sum = 0u64;
                    for i in 0..OPS_PER_THREAD {
                        let idx = indices[base + i];
                        if is_write[i] {
                            let _ =
                                tree.insert_with_guard(&keys[idx], MyRecord::new(i as u64), &guard);
                        } else if let Some(v) = tree.get_with_guard(&keys[idx], &guard) {
                            sum = sum.wrapping_add(v.checksum());
                        }
                    }
                    ctx.end_measurement();
                    black_box(sum);
                });
            });
    }

    #[divan::bench(args = THREAD_COUNTS)]
    fn skipmap(bencher: Bencher, threads: usize) {
        let keys = keys::<KEY_SIZE_8>(N);
        let indices = uniform_indices(N, OPS_PER_THREAD * threads, BASE_SEED);
        let write_decisions: Vec<Vec<bool>> = (0..threads)
            .map(|t| shuffled_write_decisions(OPS_PER_THREAD, WRITE_RATIO, BASE_SEED + t as u64))
            .collect();

        bencher
            .counter(ItemsCount::new(threads * OPS_PER_THREAD))
            .with_inputs(|| setup_skipmap_8(&keys))
            .bench_local_values(|map| {
                run_concurrent(threads, WARMUP_OPS, OPS_PER_THREAD, |ctx| {
                    let base = ctx.tid * OPS_PER_THREAD;
                    let is_write = &write_decisions[ctx.tid];
                    for i in 0..ctx.warmup_ops {
                        let idx = indices[base + (i % OPS_PER_THREAD)];
                        black_box(map.get(&keys[idx]));
                    }
                    ctx.finish_warmup();
                    ctx.begin_measurement();
                    let mut sum = 0u64;
                    for i in 0..OPS_PER_THREAD {
                        let idx = indices[base + i];
                        if is_write[i] {
                            map.insert(keys[idx], MyRecord::new(i as u64));
                        } else if let Some(e) = map.get(&keys[idx]) {
                            sum = sum.wrapping_add(e.value().checksum());
                        }
                    }
                    ctx.end_measurement();
                    black_box(sum);
                });
            });
    }

    #[divan::bench(args = THREAD_COUNTS)]
    fn indexset(bencher: Bencher, threads: usize) {
        let keys = keys::<KEY_SIZE_8>(N);
        let indices = uniform_indices(N, OPS_PER_THREAD * threads, BASE_SEED);
        let write_decisions: Vec<Vec<bool>> = (0..threads)
            .map(|t| shuffled_write_decisions(OPS_PER_THREAD, WRITE_RATIO, BASE_SEED + t as u64))
            .collect();

        bencher
            .counter(ItemsCount::new(threads * OPS_PER_THREAD))
            .with_inputs(|| setup_indexset_8(&keys))
            .bench_local_values(|map| {
                run_concurrent(threads, WARMUP_OPS, OPS_PER_THREAD, |ctx| {
                    let base = ctx.tid * OPS_PER_THREAD;
                    let is_write = &write_decisions[ctx.tid];
                    for i in 0..ctx.warmup_ops {
                        let idx = indices[base + (i % OPS_PER_THREAD)];
                        black_box(map.get(&keys[idx]));
                    }
                    ctx.finish_warmup();
                    ctx.begin_measurement();
                    let mut sum = 0u64;
                    for i in 0..OPS_PER_THREAD {
                        let idx = indices[base + i];
                        if is_write[i] {
                            map.insert(keys[idx], MyRecord::new(i as u64));
                        } else if let Some(r) = map.get(&keys[idx]) {
                            sum = sum.wrapping_add(r.get().value.checksum());
                        }
                    }
                    ctx.end_measurement();
                    black_box(sum);
                });
            });
    }

    #[divan::bench(args = THREAD_COUNTS)]
    fn tree_index(bencher: Bencher, threads: usize) {
        let keys = keys::<KEY_SIZE_8>(N);
        let indices = uniform_indices(N, OPS_PER_THREAD * threads, BASE_SEED);
        let write_decisions: Vec<Vec<bool>> = (0..threads)
            .map(|t| shuffled_write_decisions(OPS_PER_THREAD, WRITE_RATIO, BASE_SEED + t as u64))
            .collect();

        bencher
            .counter(ItemsCount::new(threads * OPS_PER_THREAD))
            .with_inputs(|| setup_tree_index_8(&keys))
            .bench_local_values(|tree| {
                run_concurrent(threads, WARMUP_OPS, OPS_PER_THREAD, |ctx| {
                    let guard = SddGuard::new();
                    let base = ctx.tid * OPS_PER_THREAD;
                    let is_write = &write_decisions[ctx.tid];
                    for i in 0..ctx.warmup_ops {
                        let idx = indices[base + (i % OPS_PER_THREAD)];
                        black_box(tree.peek(&keys[idx], &guard));
                    }
                    ctx.finish_warmup();
                    ctx.begin_measurement();
                    let mut sum = 0u64;
                    for i in 0..OPS_PER_THREAD {
                        let idx = indices[base + i];
                        if is_write[i] {
                            // Upsert emulation (see tree_index_upsert_sync_8)
                            tree_index_upsert_sync_8(&tree, keys[idx], MyRecord::new(i as u64));
                        } else if let Some(v) = tree.peek(&keys[idx], &guard) {
                            sum = sum.wrapping_add(v.checksum());
                        }
                    }
                    ctx.end_measurement();
                    black_box(sum);
                });
            });
    }
}

// =============================================================================
// 09: PURE READ - 100% Reads (No Writes)
// =============================================================================

#[divan::bench_group(name = "09_pure_read_uniform", sample_count = 200, sample_size = 1)]
mod pure_read {
    use super::*;

    const N: usize = 50_000;
    const OPS_PER_THREAD: usize = 25_000;

    #[divan::bench(args = THREAD_COUNTS)]
    fn masstree15(bencher: Bencher, threads: usize) {
        let keys = keys::<KEY_SIZE>(N);
        let indices = uniform_indices(N, OPS_PER_THREAD * threads, BASE_SEED);
        let tree = setup_masstree15(&keys);

        bencher
            .counter(ItemsCount::new(threads * OPS_PER_THREAD))
            .bench_local(|| {
                run_concurrent(threads, WARMUP_OPS, OPS_PER_THREAD, |ctx| {
                    let guard = tree.guard();
                    let base = ctx.tid * OPS_PER_THREAD;
                    for i in 0..ctx.warmup_ops {
                        let idx = indices[base + (i % OPS_PER_THREAD)];
                        black_box(tree.get_with_guard(&keys[idx], &guard));
                    }
                    ctx.finish_warmup();
                    ctx.begin_measurement();
                    let mut sum = 0u64;
                    for i in 0..OPS_PER_THREAD {
                        let idx = indices[base + i];
                        if let Some(v) = tree.get_with_guard(&keys[idx], &guard) {
                            sum = sum.wrapping_add(v.checksum());
                        }
                    }
                    ctx.end_measurement();
                    black_box(sum);
                });
            });
    }

    #[divan::bench(args = THREAD_COUNTS)]
    fn skipmap(bencher: Bencher, threads: usize) {
        let keys = keys::<KEY_SIZE>(N);
        let indices = uniform_indices(N, OPS_PER_THREAD * threads, BASE_SEED);
        let map = setup_skipmap(&keys);

        bencher
            .counter(ItemsCount::new(threads * OPS_PER_THREAD))
            .bench_local(|| {
                run_concurrent(threads, WARMUP_OPS, OPS_PER_THREAD, |ctx| {
                    let base = ctx.tid * OPS_PER_THREAD;
                    for i in 0..ctx.warmup_ops {
                        let idx = indices[base + (i % OPS_PER_THREAD)];
                        black_box(map.get(&keys[idx]));
                    }
                    ctx.finish_warmup();
                    ctx.begin_measurement();
                    let mut sum = 0u64;
                    for i in 0..OPS_PER_THREAD {
                        let idx = indices[base + i];
                        if let Some(e) = map.get(&keys[idx]) {
                            sum = sum.wrapping_add(e.value().checksum());
                        }
                    }
                    ctx.end_measurement();
                    black_box(sum);
                });
            });
    }

    #[divan::bench(args = THREAD_COUNTS)]
    fn indexset(bencher: Bencher, threads: usize) {
        let keys = keys::<KEY_SIZE>(N);
        let indices = uniform_indices(N, OPS_PER_THREAD * threads, BASE_SEED);
        let map = setup_indexset(&keys);

        bencher
            .counter(ItemsCount::new(threads * OPS_PER_THREAD))
            .bench_local(|| {
                run_concurrent(threads, WARMUP_OPS, OPS_PER_THREAD, |ctx| {
                    let base = ctx.tid * OPS_PER_THREAD;
                    for i in 0..ctx.warmup_ops {
                        let idx = indices[base + (i % OPS_PER_THREAD)];
                        black_box(map.get(&keys[idx]));
                    }
                    ctx.finish_warmup();
                    ctx.begin_measurement();
                    let mut sum = 0u64;
                    for i in 0..OPS_PER_THREAD {
                        let idx = indices[base + i];
                        if let Some(r) = map.get(&keys[idx]) {
                            sum = sum.wrapping_add(r.get().value.checksum());
                        }
                    }
                    ctx.end_measurement();
                    black_box(sum);
                });
            });
    }

    #[divan::bench(args = THREAD_COUNTS)]
    fn tree_index(bencher: Bencher, threads: usize) {
        let keys = keys::<KEY_SIZE>(N);
        let indices = uniform_indices(N, OPS_PER_THREAD * threads, BASE_SEED);
        let tree = setup_tree_index(&keys);

        bencher
            .counter(ItemsCount::new(threads * OPS_PER_THREAD))
            .bench_local(|| {
                run_concurrent(threads, WARMUP_OPS, OPS_PER_THREAD, |ctx| {
                    let guard = SddGuard::new();
                    let base = ctx.tid * OPS_PER_THREAD;
                    for i in 0..ctx.warmup_ops {
                        let idx = indices[base + (i % OPS_PER_THREAD)];
                        black_box(tree.peek(&keys[idx], &guard));
                    }
                    ctx.finish_warmup();
                    ctx.begin_measurement();
                    let mut sum = 0u64;
                    for i in 0..OPS_PER_THREAD {
                        let idx = indices[base + i];
                        if let Some(v) = tree.peek(&keys[idx], &guard) {
                            sum = sum.wrapping_add(v.checksum());
                        }
                    }
                    ctx.end_measurement();
                    black_box(sum);
                });
            });
    }
}

// =============================================================================
// 10: REMOVE HEAVY - 50% Insert, 50% Remove
// Accounting standardized: all implementations count successful removes (+1)
// =============================================================================

#[divan::bench_group(name = "10_remove_heavy", sample_count = 200, sample_size = 1)]
mod remove_heavy {
    use super::*;

    const N: usize = 50_000;
    const OPS_PER_THREAD: usize = 25_000;
    const WRITE_RATIO: usize = 50;

    #[divan::bench(args = THREAD_COUNTS)]
    fn masstree15(bencher: Bencher, threads: usize) {
        let keys = keys::<KEY_SIZE>(N);
        let indices = uniform_indices(N, OPS_PER_THREAD * threads, BASE_SEED);
        let write_decisions: Vec<Vec<bool>> = (0..threads)
            .map(|t| shuffled_write_decisions(OPS_PER_THREAD, WRITE_RATIO, BASE_SEED + t as u64))
            .collect();

        bencher
            .counter(ItemsCount::new(threads * OPS_PER_THREAD))
            .with_inputs(|| setup_masstree15(&keys))
            .bench_local_values(|tree| {
                run_concurrent(threads, WARMUP_OPS, OPS_PER_THREAD, |ctx| {
                    let guard = tree.guard();
                    let base = ctx.tid * OPS_PER_THREAD;
                    let is_write = &write_decisions[ctx.tid];
                    for i in 0..ctx.warmup_ops {
                        let idx = indices[base + (i % OPS_PER_THREAD)];
                        black_box(tree.get_with_guard(&keys[idx], &guard));
                    }
                    ctx.finish_warmup();
                    ctx.begin_measurement();
                    let mut sum = 0u64;
                    for i in 0..OPS_PER_THREAD {
                        let idx = indices[base + i];
                        if is_write[i] {
                            let _ =
                                tree.insert_with_guard(&keys[idx], MyRecord::new(i as u64), &guard);
                        } else if tree
                            .remove_with_guard(&keys[idx], &guard)
                            .is_ok_and(|v| v.is_some())
                        {
                            sum = sum.wrapping_add(1);
                        }
                    }
                    ctx.end_measurement();
                    black_box(sum);
                });
            });
    }

    #[divan::bench(args = THREAD_COUNTS)]
    fn skipmap(bencher: Bencher, threads: usize) {
        let keys = keys::<KEY_SIZE>(N);
        let indices = uniform_indices(N, OPS_PER_THREAD * threads, BASE_SEED);
        let write_decisions: Vec<Vec<bool>> = (0..threads)
            .map(|t| shuffled_write_decisions(OPS_PER_THREAD, WRITE_RATIO, BASE_SEED + t as u64))
            .collect();

        bencher
            .counter(ItemsCount::new(threads * OPS_PER_THREAD))
            .with_inputs(|| setup_skipmap(&keys))
            .bench_local_values(|map| {
                run_concurrent(threads, WARMUP_OPS, OPS_PER_THREAD, |ctx| {
                    let base = ctx.tid * OPS_PER_THREAD;
                    let is_write = &write_decisions[ctx.tid];
                    for i in 0..ctx.warmup_ops {
                        let idx = indices[base + (i % OPS_PER_THREAD)];
                        black_box(map.get(&keys[idx]));
                    }
                    ctx.finish_warmup();
                    ctx.begin_measurement();
                    let mut sum = 0u64;
                    for i in 0..OPS_PER_THREAD {
                        let idx = indices[base + i];
                        if is_write[i] {
                            map.insert(keys[idx], MyRecord::new(i as u64));
                        } else if map.remove(&keys[idx]).is_some() {
                            sum = sum.wrapping_add(1);
                        }
                    }
                    ctx.end_measurement();
                    black_box(sum);
                });
            });
    }

    #[divan::bench(args = THREAD_COUNTS)]
    fn indexset(bencher: Bencher, threads: usize) {
        let keys = keys::<KEY_SIZE>(N);
        let indices = uniform_indices(N, OPS_PER_THREAD * threads, BASE_SEED);
        let write_decisions: Vec<Vec<bool>> = (0..threads)
            .map(|t| shuffled_write_decisions(OPS_PER_THREAD, WRITE_RATIO, BASE_SEED + t as u64))
            .collect();

        bencher
            .counter(ItemsCount::new(threads * OPS_PER_THREAD))
            .with_inputs(|| setup_indexset(&keys))
            .bench_local_values(|map| {
                run_concurrent(threads, WARMUP_OPS, OPS_PER_THREAD, |ctx| {
                    let base = ctx.tid * OPS_PER_THREAD;
                    let is_write = &write_decisions[ctx.tid];
                    for i in 0..ctx.warmup_ops {
                        let idx = indices[base + (i % OPS_PER_THREAD)];
                        black_box(map.get(&keys[idx]));
                    }
                    ctx.finish_warmup();
                    ctx.begin_measurement();
                    let mut sum = 0u64;
                    for i in 0..OPS_PER_THREAD {
                        let idx = indices[base + i];
                        if is_write[i] {
                            map.insert(keys[idx], MyRecord::new(i as u64));
                        } else if map.remove(&keys[idx]).is_some() {
                            sum = sum.wrapping_add(1);
                        }
                    }
                    ctx.end_measurement();
                    black_box(sum);
                });
            });
    }

    #[divan::bench(args = THREAD_COUNTS)]
    fn tree_index(bencher: Bencher, threads: usize) {
        let keys = keys::<KEY_SIZE>(N);
        let indices = uniform_indices(N, OPS_PER_THREAD * threads, BASE_SEED);
        let write_decisions: Vec<Vec<bool>> = (0..threads)
            .map(|t| shuffled_write_decisions(OPS_PER_THREAD, WRITE_RATIO, BASE_SEED + t as u64))
            .collect();

        bencher
            .counter(ItemsCount::new(threads * OPS_PER_THREAD))
            .with_inputs(|| setup_tree_index(&keys))
            .bench_local_values(|tree| {
                run_concurrent(threads, WARMUP_OPS, OPS_PER_THREAD, |ctx| {
                    let guard = SddGuard::new();
                    let base = ctx.tid * OPS_PER_THREAD;
                    let is_write = &write_decisions[ctx.tid];
                    for i in 0..ctx.warmup_ops {
                        let idx = indices[base + (i % OPS_PER_THREAD)];
                        black_box(tree.peek(&keys[idx], &guard));
                    }
                    ctx.finish_warmup();
                    ctx.begin_measurement();
                    let mut sum = 0u64;
                    for i in 0..OPS_PER_THREAD {
                        let idx = indices[base + i];
                        if is_write[i] {
                            // Upsert emulation (see tree_index_upsert_sync doc)
                            tree_index_upsert_sync(&tree, keys[idx], MyRecord::new(i as u64));
                        } else if tree.remove_sync(&keys[idx]) {
                            sum = sum.wrapping_add(1);
                        }
                    }
                    ctx.end_measurement();
                    black_box(sum);
                });
            });
    }
}

// =============================================================================
// 12: BUILD TIME - Measure data structure construction time
// =============================================================================

#[divan::bench_group(name = "12_build_time", sample_count = 200, sample_size = 1)]
mod build_time {
    use super::*;

    const N: usize = 50_000;

    #[divan::bench]
    fn masstree15_build_time(bencher: Bencher) {
        let keys = keys::<KEY_SIZE>(N);
        bencher.counter(ItemsCount::new(N)).bench_local(|| {
            let tree = setup_masstree15(&keys);
            black_box(tree)
        });
    }

    #[divan::bench]
    fn skipmap_build_time(bencher: Bencher) {
        let keys = keys::<KEY_SIZE>(N);
        bencher.counter(ItemsCount::new(N)).bench_local(|| {
            let map = setup_skipmap(&keys);
            black_box(map)
        });
    }

    #[divan::bench]
    fn indexset_build_time(bencher: Bencher) {
        let keys = keys::<KEY_SIZE>(N);
        bencher.counter(ItemsCount::new(N)).bench_local(|| {
            let map = setup_indexset(&keys);
            black_box(map)
        });
    }

    #[divan::bench]
    fn tree_index_build_time(bencher: Bencher) {
        let keys = keys::<KEY_SIZE>(N);
        bencher.counter(ItemsCount::new(N)).bench_local(|| {
            let tree = setup_tree_index(&keys);
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

#[divan::bench_group(name = "13_insert_only_fair", sample_count = 200, sample_size = 1)]
mod insert_only_fair {
    use super::*;

    const N: usize = 50_000; // Pre-populated keys for reads
    const OPS_PER_THREAD: usize = 12_500;
    const WRITE_RATIO: usize = 10;

    #[divan::bench(args = THREAD_COUNTS)]
    fn masstree15(bencher: Bencher, threads: usize) {
        let read_keys = keys::<KEY_SIZE>(N);
        let total_writes = (OPS_PER_THREAD * threads) / 10;
        let all_write_keys = keys::<KEY_SIZE>(N + total_writes);
        let write_keys: Vec<[u8; KEY_SIZE]> = all_write_keys[N..].to_vec();
        let read_indices = uniform_indices(N, OPS_PER_THREAD * threads, BASE_SEED);
        let write_decisions: Vec<Vec<bool>> = (0..threads)
            .map(|t| shuffled_write_decisions(OPS_PER_THREAD, WRITE_RATIO, BASE_SEED + t as u64))
            .collect();

        bencher
            .counter(ItemsCount::new(threads * OPS_PER_THREAD))
            .with_inputs(|| setup_masstree15(&read_keys))
            .bench_local_values(|tree| {
                run_concurrent(threads, WARMUP_OPS, OPS_PER_THREAD, |ctx| {
                    let guard = tree.guard();
                    let base = ctx.tid * OPS_PER_THREAD;
                    let write_base = (ctx.tid * total_writes) / threads;
                    let is_write = &write_decisions[ctx.tid];
                    for i in 0..ctx.warmup_ops {
                        let idx = read_indices[base + (i % OPS_PER_THREAD)];
                        black_box(tree.get_with_guard(&read_keys[idx], &guard));
                    }
                    ctx.finish_warmup();
                    ctx.begin_measurement();
                    let mut sum = 0u64;
                    let mut write_idx = 0usize;
                    for i in 0..OPS_PER_THREAD {
                        if is_write[i] {
                            let wk_idx = write_base + write_idx;
                            if wk_idx < write_keys.len() {
                                let _ = tree.insert_with_guard(
                                    &write_keys[wk_idx],
                                    MyRecord::new(i as u64),
                                    &guard,
                                );
                                write_idx += 1;
                            }
                        } else {
                            let idx = read_indices[base + i];
                            if let Some(v) = tree.get_with_guard(&read_keys[idx], &guard) {
                                sum = sum.wrapping_add(v.checksum());
                            }
                        }
                    }
                    ctx.end_measurement();
                    black_box(sum);
                });
            });
    }

    #[divan::bench(args = THREAD_COUNTS)]
    fn skipmap(bencher: Bencher, threads: usize) {
        let read_keys = keys::<KEY_SIZE>(N);
        let total_writes = (OPS_PER_THREAD * threads) / 10;
        let all_write_keys = keys::<KEY_SIZE>(N + total_writes);
        let write_keys: Vec<[u8; KEY_SIZE]> = all_write_keys[N..].to_vec();
        let read_indices = uniform_indices(N, OPS_PER_THREAD * threads, BASE_SEED);
        let write_decisions: Vec<Vec<bool>> = (0..threads)
            .map(|t| shuffled_write_decisions(OPS_PER_THREAD, WRITE_RATIO, BASE_SEED + t as u64))
            .collect();

        bencher
            .counter(ItemsCount::new(threads * OPS_PER_THREAD))
            .with_inputs(|| setup_skipmap(&read_keys))
            .bench_local_values(|map| {
                run_concurrent(threads, WARMUP_OPS, OPS_PER_THREAD, |ctx| {
                    let base = ctx.tid * OPS_PER_THREAD;
                    let write_base = (ctx.tid * total_writes) / threads;
                    let is_write = &write_decisions[ctx.tid];
                    for i in 0..ctx.warmup_ops {
                        let idx = read_indices[base + (i % OPS_PER_THREAD)];
                        black_box(map.get(&read_keys[idx]));
                    }
                    ctx.finish_warmup();
                    ctx.begin_measurement();
                    let mut sum = 0u64;
                    let mut write_idx = 0usize;
                    for i in 0..OPS_PER_THREAD {
                        if is_write[i] {
                            let wk_idx = write_base + write_idx;
                            if wk_idx < write_keys.len() {
                                map.insert(write_keys[wk_idx], MyRecord::new(i as u64));
                                write_idx += 1;
                            }
                        } else {
                            let idx = read_indices[base + i];
                            if let Some(e) = map.get(&read_keys[idx]) {
                                sum = sum.wrapping_add(e.value().checksum());
                            }
                        }
                    }
                    ctx.end_measurement();
                    black_box(sum);
                });
            });
    }

    #[divan::bench(args = THREAD_COUNTS)]
    fn indexset(bencher: Bencher, threads: usize) {
        let read_keys = keys::<KEY_SIZE>(N);
        let total_writes = (OPS_PER_THREAD * threads) / 10;
        let all_write_keys = keys::<KEY_SIZE>(N + total_writes);
        let write_keys: Vec<[u8; KEY_SIZE]> = all_write_keys[N..].to_vec();
        let read_indices = uniform_indices(N, OPS_PER_THREAD * threads, BASE_SEED);
        let write_decisions: Vec<Vec<bool>> = (0..threads)
            .map(|t| shuffled_write_decisions(OPS_PER_THREAD, WRITE_RATIO, BASE_SEED + t as u64))
            .collect();

        bencher
            .counter(ItemsCount::new(threads * OPS_PER_THREAD))
            .with_inputs(|| setup_indexset(&read_keys))
            .bench_local_values(|map| {
                run_concurrent(threads, WARMUP_OPS, OPS_PER_THREAD, |ctx| {
                    let base = ctx.tid * OPS_PER_THREAD;
                    let write_base = (ctx.tid * total_writes) / threads;
                    let is_write = &write_decisions[ctx.tid];
                    for i in 0..ctx.warmup_ops {
                        let idx = read_indices[base + (i % OPS_PER_THREAD)];
                        black_box(map.get(&read_keys[idx]));
                    }
                    ctx.finish_warmup();
                    ctx.begin_measurement();
                    let mut sum = 0u64;
                    let mut write_idx = 0usize;
                    for i in 0..OPS_PER_THREAD {
                        if is_write[i] {
                            let wk_idx = write_base + write_idx;
                            if wk_idx < write_keys.len() {
                                map.insert(write_keys[wk_idx], MyRecord::new(i as u64));
                                write_idx += 1;
                            }
                        } else {
                            let idx = read_indices[base + i];
                            if let Some(r) = map.get(&read_keys[idx]) {
                                sum = sum.wrapping_add(r.get().value.checksum());
                            }
                        }
                    }
                    ctx.end_measurement();
                    black_box(sum);
                });
            });
    }

    #[divan::bench(args = THREAD_COUNTS)]
    fn tree_index(bencher: Bencher, threads: usize) {
        let read_keys = keys::<KEY_SIZE>(N);
        let total_writes = (OPS_PER_THREAD * threads) / 10;
        let all_write_keys = keys::<KEY_SIZE>(N + total_writes);
        let write_keys: Vec<[u8; KEY_SIZE]> = all_write_keys[N..].to_vec();
        let read_indices = uniform_indices(N, OPS_PER_THREAD * threads, BASE_SEED);
        let write_decisions: Vec<Vec<bool>> = (0..threads)
            .map(|t| shuffled_write_decisions(OPS_PER_THREAD, WRITE_RATIO, BASE_SEED + t as u64))
            .collect();

        bencher
            .counter(ItemsCount::new(threads * OPS_PER_THREAD))
            .with_inputs(|| setup_tree_index(&read_keys))
            .bench_local_values(|tree| {
                run_concurrent(threads, WARMUP_OPS, OPS_PER_THREAD, |ctx| {
                    let guard = SddGuard::new();
                    let base = ctx.tid * OPS_PER_THREAD;
                    let write_base = (ctx.tid * total_writes) / threads;
                    let is_write = &write_decisions[ctx.tid];
                    for i in 0..ctx.warmup_ops {
                        let idx = read_indices[base + (i % OPS_PER_THREAD)];
                        black_box(tree.peek(&read_keys[idx], &guard));
                    }
                    ctx.finish_warmup();
                    ctx.begin_measurement();
                    let mut sum = 0u64;
                    let mut write_idx = 0usize;
                    for i in 0..OPS_PER_THREAD {
                        if is_write[i] {
                            // FAIR: Simple insert, no upsert workaround needed
                            let wk_idx = write_base + write_idx;
                            if wk_idx < write_keys.len() {
                                let _ =
                                    tree.insert_sync(write_keys[wk_idx], MyRecord::new(i as u64));
                                write_idx += 1;
                            }
                        } else {
                            let idx = read_indices[base + i];
                            if let Some(v) = tree.peek(&read_keys[idx], &guard) {
                                sum = sum.wrapping_add(v.checksum());
                            }
                        }
                    }
                    ctx.end_measurement();
                    black_box(sum);
                });
            });
    }
}

// =============================================================================
// 14: PURE INSERT - 100% Inserts (Build from empty)
// =============================================================================
//
// All operations are inserts to new keys. No reads, no upserts.
// This is the fairest possible write benchmark.

#[divan::bench_group(name = "14_pure_insert", sample_count = 200, sample_size = 1)]
mod pure_insert {
    use crate::bench_utils::ThreadContext;

    use super::*;

    const OPS_PER_THREAD: usize = 10_000;

    #[divan::bench(args = THREAD_COUNTS)]
    fn masstree15(bencher: Bencher, threads: usize) {
        let keys = keys::<KEY_SIZE>(threads * OPS_PER_THREAD);

        bencher
            .counter(ItemsCount::new(threads * OPS_PER_THREAD))
            .with_inputs(MassTree15Inline::<MyRecord>::new)
            .bench_local_values(|tree| {
                run_concurrent(threads, WARMUP_OPS, OPS_PER_THREAD, |ctx| {
                    let guard = tree.guard();
                    let base = ctx.tid * OPS_PER_THREAD;
                    // Warmup: touch key memory to prime caches
                    for i in 0..ctx.warmup_ops {
                        black_box(&keys[base + i]);
                    }
                    ctx.finish_warmup();
                    ctx.begin_measurement();
                    for i in 0..OPS_PER_THREAD {
                        let idx = base + i;
                        let _ = tree.insert_with_guard(&keys[idx], MyRecord::new(i as u64), &guard);
                    }
                    ctx.end_measurement();
                });
            });
    }

    #[divan::bench(args = THREAD_COUNTS)]
    fn skipmap(bencher: Bencher, threads: usize) {
        let keys = keys::<KEY_SIZE>(threads * OPS_PER_THREAD);

        bencher
            .counter(ItemsCount::new(threads * OPS_PER_THREAD))
            .with_inputs(SkipMap::<[u8; KEY_SIZE], MyRecord>::new)
            .bench_local_values(|map| {
                run_concurrent(threads, WARMUP_OPS, OPS_PER_THREAD, |ctx| {
                    let base = ctx.tid * OPS_PER_THREAD;
                    for i in 0..ctx.warmup_ops {
                        black_box(&keys[base + i]);
                    }
                    ctx.finish_warmup();
                    ctx.begin_measurement();
                    for i in 0..OPS_PER_THREAD {
                        let idx = base + i;
                        map.insert(keys[idx], MyRecord::new(i as u64));
                    }
                    ctx.end_measurement();
                });
            });
    }

    #[divan::bench(args = THREAD_COUNTS)]
    fn indexset(bencher: Bencher, threads: usize) {
        let keys: Vec<[u8; 64]> = keys::<KEY_SIZE>(threads * OPS_PER_THREAD);

        bencher
            .counter(ItemsCount::new(threads * OPS_PER_THREAD))
            .with_inputs(IndexSetBTreeMap::<[u8; KEY_SIZE], MyRecord>::new)
            .bench_local_values(|map: IndexSetBTreeMap<[u8; 64], MyRecord>| {
                run_concurrent(threads, WARMUP_OPS, OPS_PER_THREAD, |ctx| {
                    let base: usize = ctx.tid * OPS_PER_THREAD;
                    for i in 0..ctx.warmup_ops {
                        black_box(&keys[base + i]);
                    }
                    ctx.finish_warmup();
                    ctx.begin_measurement();
                    for i in 0..OPS_PER_THREAD {
                        let idx = base + i;
                        map.insert(keys[idx], MyRecord::new(i as u64));
                    }
                    ctx.end_measurement();
                });
            });
    }

    #[divan::bench(args = THREAD_COUNTS)]
    fn tree_index(bencher: Bencher, threads: usize) {
        let keys = keys::<KEY_SIZE>(threads * OPS_PER_THREAD);

        bencher
            .counter(ItemsCount::new(threads * OPS_PER_THREAD))
            .with_inputs(TreeIndex::<[u8; KEY_SIZE], MyRecord>::new)
            .bench_local_values(|tree| {
                run_concurrent(
                    threads,
                    WARMUP_OPS,
                    OPS_PER_THREAD,
                    |ctx: &ThreadContext| {
                        let base: usize = ctx.tid * OPS_PER_THREAD;

                        for i in 0..ctx.warmup_ops {
                            black_box(&keys[base + i]);
                        }

                        ctx.finish_warmup();
                        ctx.begin_measurement();

                        for i in 0..OPS_PER_THREAD {
                            let idx: usize = base + i;
                            // FAIR: Simple insert_sync, no workaround
                            let _ = tree.insert_sync(keys[idx], MyRecord::new(i as u64));
                        }

                        ctx.end_measurement();
                    },
                );
            });
    }
}
