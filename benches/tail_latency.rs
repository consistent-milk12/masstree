//! Tail latency benchmark: masstree vs treeindex vs indexset vs skipmap vs dashmap.
//!
//! Measures per-operation latency distributions to capture p99/p99.9 tail
//! behavior. Every benchmark uses `sample_size = 1` so each sample represents
//! exactly one operation's latency (no batching that would average out spikes).
//!
//! Run with: `cargo bench -p masstree --bench tail_latency`
//! Filter:   `cargo bench -p masstree --bench tail_latency -- --filter get_1_`

#![expect(clippy::pedantic)]
#![expect(clippy::indexing_slicing)]
#![expect(clippy::type_complexity)]

#[path = "bench_utils.rs"]
mod bench_utils;

use std::sync::atomic::{AtomicUsize, Ordering};

use crossbeam_skiplist::SkipMap;
use dashmap::DashMap;
use indexset::concurrent::map::BTreeMap as IndexSetBTreeMap;
use masstree::{MassTree15Inline, RangeBound};
use pbench::Bencher;
use scc::TreeIndex;
use sdd::Guard as SddGuard;
use seize::LocalGuard;

use bench_utils::{uniform_indices, zipfian_indices};

// =============================================================================
// Constants
// =============================================================================

const KEY_SIZE: usize = 32;

/// 50k samples: enough for reliable p99.9 (need >= 1000) while staying under
/// pbench's 100k exact-mode threshold (sorted array, no HdrHistogram lossyness).
/// Pre-generated random index pool. Larger than sample_count to avoid cycling.
const INDEX_POOL: usize = 500_000;

// =============================================================================
// Setup Helpers
// =============================================================================

fn setup_masstree(n: usize) -> (MassTree15Inline<u64>, Vec<[u8; KEY_SIZE]>) {
    let keys: Vec<[u8; KEY_SIZE]> = bench_utils::keys(n);
    let tree: MassTree15Inline<u64> = MassTree15Inline::new();

    {
        let guard: LocalGuard<'_> = tree.guard();

        for (i, key) in keys.iter().enumerate() {
            let _ = tree.insert_with_guard(key, i as u64, &guard);
        }
    }

    (tree, keys)
}

fn setup_skipmap(n: usize) -> (SkipMap<[u8; KEY_SIZE], u64>, Vec<[u8; KEY_SIZE]>) {
    let keys: Vec<[u8; KEY_SIZE]> = bench_utils::keys(n);
    let map: SkipMap<[u8; KEY_SIZE], u64> = SkipMap::new();

    for (i, key) in keys.iter().enumerate() {
        map.insert(*key, i as u64);
    }

    (map, keys)
}

fn setup_indexset(n: usize) -> (IndexSetBTreeMap<[u8; KEY_SIZE], u64>, Vec<[u8; KEY_SIZE]>) {
    let keys: Vec<[u8; KEY_SIZE]> = bench_utils::keys(n);
    let map: IndexSetBTreeMap<[u8; KEY_SIZE], u64> = IndexSetBTreeMap::new();

    for (i, key) in keys.iter().enumerate() {
        map.insert(*key, i as u64);
    }

    (map, keys)
}

fn setup_dashmap(n: usize) -> (DashMap<[u8; KEY_SIZE], u64>, Vec<[u8; KEY_SIZE]>) {
    let keys: Vec<[u8; KEY_SIZE]> = bench_utils::keys(n);
    let map: DashMap<[u8; KEY_SIZE], u64> = DashMap::new();

    for (i, key) in keys.iter().enumerate() {
        map.insert(*key, i as u64);
    }

    (map, keys)
}

fn setup_tree_index(n: usize) -> (TreeIndex<[u8; KEY_SIZE], u64>, Vec<[u8; KEY_SIZE]>) {
    let keys: Vec<[u8; KEY_SIZE]> = bench_utils::keys(n);
    let tree: TreeIndex<[u8; KEY_SIZE], u64> = TreeIndex::new();

    for (i, key) in keys.iter().enumerate() {
        let _ = tree.insert_sync(*key, i as u64);
    }

    (tree, keys)
}

/// TreeIndex upsert emulation via remove-then-reinsert.
fn tree_index_upsert_sync(tree: &TreeIndex<[u8; KEY_SIZE], u64>, key: [u8; KEY_SIZE], value: u64) {
    let mut key: [u8; KEY_SIZE] = key;
    let mut value: u64 = value;

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

/// Deterministic per-operation read/write decision that avoids synchronized
/// write bursts. Uses a mixing hash so threads with different cursor values
/// don't all write on the same modular boundary.
#[inline]
fn is_write_op(i: usize, write_pct: u32) -> bool {
    // xorshift-style mix to break alignment across threads
    let mut x: u64 = i as u64;
    x ^= x >> 17;
    x = x.wrapping_mul(0xbf58_476d_1ce4_e5b9);
    x ^= x >> 31;
    (x % 100) < write_pct as u64
}

// =============================================================================
// 01: Single-op GET latency, 100k entries, uniform random (1 thread)
//
// Baseline single-threaded read latency. No contention, pure data structure
// traversal cost. Tail dominated by cache misses on cold nodes.
// =============================================================================

#[pbench::bench(threads = [1], sample_count = 50_000, sample_size = 1, skip_ext_time)]
fn get_1_masstree(b: &Bencher<'_>) {
    let n: usize = 100_000;
    let (tree, keys) = setup_masstree(n);
    let indices: Vec<usize> = uniform_indices(n, INDEX_POOL, 42);
    let cursor: AtomicUsize = AtomicUsize::new(0);

    b.bench_refs(|| {
        let guard: LocalGuard<'_> = tree.guard();
        let i: usize = cursor.fetch_add(1, Ordering::Relaxed);
        let idx: usize = indices[i % indices.len()];
        std::hint::black_box(tree.get_with_guard(&keys[idx], &guard));
    });
}

#[pbench::bench(threads = [1], sample_count = 50_000, sample_size = 1, skip_ext_time)]
fn get_1_skipmap(b: &Bencher<'_>) {
    let n: usize = 100_000;
    let (map, keys) = setup_skipmap(n);
    let indices: Vec<usize> = uniform_indices(n, INDEX_POOL, 42);
    let cursor: AtomicUsize = AtomicUsize::new(0);

    b.bench_refs(|| {
        let i: usize = cursor.fetch_add(1, Ordering::Relaxed);
        let idx: usize = indices[i % indices.len()];
        std::hint::black_box(map.get(&keys[idx]));
    });
}

#[pbench::bench(threads = [1], sample_count = 50_000, sample_size = 1, skip_ext_time)]
fn get_1_indexset(b: &Bencher<'_>) {
    let n: usize = 100_000;
    let (map, keys) = setup_indexset(n);
    let indices: Vec<usize> = uniform_indices(n, INDEX_POOL, 42);
    let cursor: AtomicUsize = AtomicUsize::new(0);

    b.bench_refs(|| {
        let i: usize = cursor.fetch_add(1, Ordering::Relaxed);
        let idx: usize = indices[i % indices.len()];
        std::hint::black_box(map.get(&keys[idx]));
    });
}

#[pbench::bench(threads = [1], sample_count = 50_000, sample_size = 1, skip_ext_time)]
fn get_1_treeindex(b: &Bencher<'_>) {
    let n: usize = 100_000;
    let (tree, keys) = setup_tree_index(n);
    let indices: Vec<usize> = uniform_indices(n, INDEX_POOL, 42);
    let cursor: AtomicUsize = AtomicUsize::new(0);

    b.bench_refs(|| {
        let guard: SddGuard = SddGuard::new();
        let i: usize = cursor.fetch_add(1, Ordering::Relaxed);
        let idx: usize = indices[i % indices.len()];
        std::hint::black_box(tree.peek(&keys[idx], &guard));
    });
}

#[pbench::bench(threads = [1], sample_count = 50_000, sample_size = 1, skip_ext_time)]
fn get_1_dashmap(b: &Bencher<'_>) {
    let n: usize = 100_000;
    let (map, keys) = setup_dashmap(n);
    let indices: Vec<usize> = uniform_indices(n, INDEX_POOL, 42);
    let cursor: AtomicUsize = AtomicUsize::new(0);

    b.bench_refs(|| {
        let i: usize = cursor.fetch_add(1, Ordering::Relaxed);
        let idx: usize = indices[i % indices.len()];
        std::hint::black_box(map.get(&keys[idx]));
    });
}

// =============================================================================
// 02: Single-op GET latency, 100k entries, uniform random (8 threads)
//
// Concurrent reads stress cache coherence and contention on shared nodes.
// Tail latency increases when multiple threads compete for the same cache lines.
// =============================================================================

#[pbench::bench(threads = [8], sample_count = 50_000, sample_size = 1, skip_ext_time)]
fn get_8t_masstree(b: &Bencher<'_>) {
    let n: usize = 100_000;
    let (tree, keys) = setup_masstree(n);
    let indices: Vec<usize> = uniform_indices(n, INDEX_POOL, 42);
    let cursor: AtomicUsize = AtomicUsize::new(0);

    b.bench_refs(|| {
        let guard: LocalGuard<'_> = tree.guard();
        let i: usize = cursor.fetch_add(1, Ordering::Relaxed);
        let idx: usize = indices[i % indices.len()];
        std::hint::black_box(tree.get_with_guard(&keys[idx], &guard));
    });
}

#[pbench::bench(threads = [8], sample_count = 50_000, sample_size = 1, skip_ext_time)]
fn get_8t_skipmap(b: &Bencher<'_>) {
    let n: usize = 100_000;
    let (map, keys) = setup_skipmap(n);
    let indices: Vec<usize> = uniform_indices(n, INDEX_POOL, 42);
    let cursor: AtomicUsize = AtomicUsize::new(0);

    b.bench_refs(|| {
        let i: usize = cursor.fetch_add(1, Ordering::Relaxed);
        let idx: usize = indices[i % indices.len()];
        std::hint::black_box(map.get(&keys[idx]));
    });
}

#[pbench::bench(threads = [8], sample_count = 50_000, sample_size = 1, skip_ext_time)]
fn get_8t_indexset(b: &Bencher<'_>) {
    let n: usize = 100_000;
    let (map, keys) = setup_indexset(n);
    let indices: Vec<usize> = uniform_indices(n, INDEX_POOL, 42);
    let cursor: AtomicUsize = AtomicUsize::new(0);

    b.bench_refs(|| {
        let i: usize = cursor.fetch_add(1, Ordering::Relaxed);
        let idx: usize = indices[i % indices.len()];
        std::hint::black_box(map.get(&keys[idx]));
    });
}

#[pbench::bench(threads = [8], sample_count = 50_000, sample_size = 1, skip_ext_time)]
fn get_8t_treeindex(b: &Bencher<'_>) {
    let n: usize = 100_000;
    let (tree, keys) = setup_tree_index(n);
    let indices: Vec<usize> = uniform_indices(n, INDEX_POOL, 42);
    let cursor: AtomicUsize = AtomicUsize::new(0);

    b.bench_refs(|| {
        let guard: SddGuard = SddGuard::new();
        let i: usize = cursor.fetch_add(1, Ordering::Relaxed);
        let idx: usize = indices[i % indices.len()];
        std::hint::black_box(tree.peek(&keys[idx], &guard));
    });
}

#[pbench::bench(threads = [8], sample_count = 50_000, sample_size = 1, skip_ext_time)]
fn get_8t_dashmap(b: &Bencher<'_>) {
    let n: usize = 100_000;
    let (map, keys) = setup_dashmap(n);
    let indices: Vec<usize> = uniform_indices(n, INDEX_POOL, 42);
    let cursor: AtomicUsize = AtomicUsize::new(0);

    b.bench_refs(|| {
        let i: usize = cursor.fetch_add(1, Ordering::Relaxed);
        let idx: usize = indices[i % indices.len()];
        std::hint::black_box(map.get(&keys[idx]));
    });
}

// =============================================================================
// 03: Single-op INSERT latency, fresh tree, 1 thread
//
// Measures insert path including splits. Each sample inserts one key into a
// growing tree, so later samples hit deeper/wider trees.
// =============================================================================

#[pbench::bench(threads = [1], sample_count = 50_000, sample_size = 1, skip_ext_time)]
fn insert_1_masstree(b: &Bencher<'_>) {
    let n: usize = 50_000;
    let keys: Vec<[u8; KEY_SIZE]> = bench_utils::keys(n);
    let tree: MassTree15Inline<u64> = MassTree15Inline::new();
    let cursor: AtomicUsize = AtomicUsize::new(0);

    b.bench_refs(|| {
        let guard: LocalGuard<'_> = tree.guard();
        let i: usize = cursor.fetch_add(1, Ordering::Relaxed) % n;
        let _ = tree.insert_with_guard(&keys[i], i as u64, &guard);
    });
}

#[pbench::bench(threads = [1], sample_count = 50_000, sample_size = 1, skip_ext_time)]
fn insert_1_skipmap(b: &Bencher<'_>) {
    let n: usize = 50_000;
    let keys: Vec<[u8; KEY_SIZE]> = bench_utils::keys(n);
    let map: SkipMap<[u8; KEY_SIZE], u64> = SkipMap::new();
    let cursor: AtomicUsize = AtomicUsize::new(0);

    b.bench_refs(|| {
        let i: usize = cursor.fetch_add(1, Ordering::Relaxed) % n;
        map.insert(keys[i], i as u64);
    });
}

#[pbench::bench(threads = [1], sample_count = 50_000, sample_size = 1, skip_ext_time)]
fn insert_1_indexset(b: &Bencher<'_>) {
    let n: usize = 50_000;
    let keys: Vec<[u8; KEY_SIZE]> = bench_utils::keys(n);
    let map: IndexSetBTreeMap<[u8; KEY_SIZE], u64> = IndexSetBTreeMap::new();
    let cursor: AtomicUsize = AtomicUsize::new(0);

    b.bench_refs(|| {
        let i: usize = cursor.fetch_add(1, Ordering::Relaxed) % n;
        map.insert(keys[i], i as u64);
    });
}

#[pbench::bench(threads = [1], sample_count = 50_000, sample_size = 1, skip_ext_time)]
fn insert_1_treeindex(b: &Bencher<'_>) {
    let n: usize = 50_000;
    let keys: Vec<[u8; KEY_SIZE]> = bench_utils::keys(n);
    let tree: TreeIndex<[u8; KEY_SIZE], u64> = TreeIndex::new();
    let cursor: AtomicUsize = AtomicUsize::new(0);

    b.bench_refs(|| {
        let i: usize = cursor.fetch_add(1, Ordering::Relaxed) % n;
        let _ = tree.insert_sync(keys[i], i as u64);
    });
}

#[pbench::bench(threads = [1], sample_count = 50_000, sample_size = 1, skip_ext_time)]
fn insert_1_dashmap(b: &Bencher<'_>) {
    let n: usize = 50_000;
    let keys: Vec<[u8; KEY_SIZE]> = bench_utils::keys(n);
    let map: DashMap<[u8; KEY_SIZE], u64> = DashMap::new();
    let cursor: AtomicUsize = AtomicUsize::new(0);

    b.bench_refs(|| {
        let i: usize = cursor.fetch_add(1, Ordering::Relaxed) % n;
        map.insert(keys[i], i as u64);
    });
}

// =============================================================================
// 04: Single-op INSERT latency, fresh tree, 8 threads
//
// Concurrent inserts cause splits and contention on internal nodes.
// Tail latency reveals retry/backoff overhead.
// =============================================================================

#[pbench::bench(threads = [8], sample_count = 50_000, sample_size = 1, skip_ext_time)]
fn insert_8t_masstree(b: &Bencher<'_>) {
    let n: usize = 100_000;
    let keys: Vec<[u8; KEY_SIZE]> = bench_utils::keys(n);
    let tree: MassTree15Inline<u64> = MassTree15Inline::new();
    let cursor: AtomicUsize = AtomicUsize::new(0);

    b.bench_refs(|| {
        let guard: LocalGuard<'_> = tree.guard();
        let i: usize = cursor.fetch_add(1, Ordering::Relaxed) % n;
        let _ = tree.insert_with_guard(&keys[i], i as u64, &guard);
    });
}

#[pbench::bench(threads = [8], sample_count = 50_000, sample_size = 1, skip_ext_time)]
fn insert_8t_skipmap(b: &Bencher<'_>) {
    let n: usize = 100_000;
    let keys: Vec<[u8; KEY_SIZE]> = bench_utils::keys(n);
    let map: SkipMap<[u8; KEY_SIZE], u64> = SkipMap::new();
    let cursor: AtomicUsize = AtomicUsize::new(0);

    b.bench_refs(|| {
        let i: usize = cursor.fetch_add(1, Ordering::Relaxed) % n;
        map.insert(keys[i], i as u64);
    });
}

#[pbench::bench(threads = [8], sample_count = 50_000, sample_size = 1, skip_ext_time)]
fn insert_8t_indexset(b: &Bencher<'_>) {
    let n: usize = 100_000;
    let keys: Vec<[u8; KEY_SIZE]> = bench_utils::keys(n);
    let map: IndexSetBTreeMap<[u8; KEY_SIZE], u64> = IndexSetBTreeMap::new();
    let cursor: AtomicUsize = AtomicUsize::new(0);

    b.bench_refs(|| {
        let i: usize = cursor.fetch_add(1, Ordering::Relaxed) % n;
        map.insert(keys[i], i as u64);
    });
}

#[pbench::bench(threads = [8], sample_count = 50_000, sample_size = 1, skip_ext_time)]
fn insert_8t_treeindex(b: &Bencher<'_>) {
    let n: usize = 100_000;
    let keys: Vec<[u8; KEY_SIZE]> = bench_utils::keys(n);
    let tree: TreeIndex<[u8; KEY_SIZE], u64> = TreeIndex::new();
    let cursor: AtomicUsize = AtomicUsize::new(0);

    b.bench_refs(|| {
        let i: usize = cursor.fetch_add(1, Ordering::Relaxed) % n;
        let _ = tree.insert_sync(keys[i], i as u64);
    });
}

#[pbench::bench(threads = [8], sample_count = 50_000, sample_size = 1, skip_ext_time)]
fn insert_8t_dashmap(b: &Bencher<'_>) {
    let n: usize = 100_000;
    let keys: Vec<[u8; KEY_SIZE]> = bench_utils::keys(n);
    let map: DashMap<[u8; KEY_SIZE], u64> = DashMap::new();
    let cursor: AtomicUsize = AtomicUsize::new(0);

    b.bench_refs(|| {
        let i: usize = cursor.fetch_add(1, Ordering::Relaxed) % n;
        map.insert(keys[i], i as u64);
    });
}

// =============================================================================
// 05: Mixed read-write under contention, 90/10 read/write, 8 threads
//
// The most realistic tail latency scenario: reads occasionally blocked by
// concurrent writers. Uses hash-based write decision so threads don't all
// write on the same iteration (avoiding artificial synchronized bursts).
// =============================================================================

#[pbench::bench(threads = [8], sample_count = 50_000, sample_size = 1, skip_ext_time)]
fn mixed_90_10_8t_masstree(b: &Bencher<'_>) {
    let n: usize = 100_000;
    let (tree, keys) = setup_masstree(n);
    let indices: Vec<usize> = uniform_indices(n, INDEX_POOL, 42);
    let cursor: AtomicUsize = AtomicUsize::new(0);

    b.bench_refs(|| {
        let guard: LocalGuard<'_> = tree.guard();
        let i: usize = cursor.fetch_add(1, Ordering::Relaxed);
        let idx: usize = indices[i % indices.len()];

        if is_write_op(i, 10) {
            let _ = tree.insert_with_guard(&keys[idx], i as u64, &guard);
        } else {
            std::hint::black_box(tree.get_with_guard(&keys[idx], &guard));
        }
    });
}

#[pbench::bench(threads = [8], sample_count = 50_000, sample_size = 1, skip_ext_time)]
fn mixed_90_10_8t_skipmap(b: &Bencher<'_>) {
    let n: usize = 100_000;
    let (map, keys) = setup_skipmap(n);
    let indices: Vec<usize> = uniform_indices(n, INDEX_POOL, 42);
    let cursor: AtomicUsize = AtomicUsize::new(0);

    b.bench_refs(|| {
        let i: usize = cursor.fetch_add(1, Ordering::Relaxed);
        let idx: usize = indices[i % indices.len()];

        if is_write_op(i, 10) {
            map.insert(keys[idx], i as u64);
        } else {
            std::hint::black_box(map.get(&keys[idx]));
        }
    });
}

#[pbench::bench(threads = [8], sample_count = 50_000, sample_size = 1, skip_ext_time)]
fn mixed_90_10_8t_indexset(b: &Bencher<'_>) {
    let n: usize = 100_000;
    let (map, keys) = setup_indexset(n);
    let indices: Vec<usize> = uniform_indices(n, INDEX_POOL, 42);
    let cursor: AtomicUsize = AtomicUsize::new(0);

    b.bench_refs(|| {
        let i: usize = cursor.fetch_add(1, Ordering::Relaxed);
        let idx: usize = indices[i % indices.len()];

        if is_write_op(i, 10) {
            map.insert(keys[idx], i as u64);
        } else {
            std::hint::black_box(map.get(&keys[idx]));
        }
    });
}

#[pbench::bench(threads = [8], sample_count = 50_000, sample_size = 1, skip_ext_time)]
fn mixed_90_10_8t_treeindex(b: &Bencher<'_>) {
    let n: usize = 100_000;
    let (tree, keys) = setup_tree_index(n);
    let indices: Vec<usize> = uniform_indices(n, INDEX_POOL, 42);
    let cursor: AtomicUsize = AtomicUsize::new(0);

    b.bench_refs(|| {
        let guard: SddGuard = SddGuard::new();
        let i: usize = cursor.fetch_add(1, Ordering::Relaxed);
        let idx: usize = indices[i % indices.len()];

        if is_write_op(i, 10) {
            tree_index_upsert_sync(&tree, keys[idx], i as u64);
        } else {
            std::hint::black_box(tree.peek(&keys[idx], &guard));
        }
    });
}

#[pbench::bench(threads = [8], sample_count = 50_000, sample_size = 1, skip_ext_time)]
fn mixed_90_10_8t_dashmap(b: &Bencher<'_>) {
    let n: usize = 100_000;
    let (map, keys) = setup_dashmap(n);
    let indices: Vec<usize> = uniform_indices(n, INDEX_POOL, 42);
    let cursor: AtomicUsize = AtomicUsize::new(0);

    b.bench_refs(|| {
        let i: usize = cursor.fetch_add(1, Ordering::Relaxed);
        let idx: usize = indices[i % indices.len()];

        if is_write_op(i, 10) {
            map.insert(keys[idx], i as u64);
        } else {
            std::hint::black_box(map.get(&keys[idx]));
        }
    });
}

// =============================================================================
// 06: Mixed read-write under contention, 50/50 read/write, 8 threads
//
// Heavy write load. Worst-case scenario for lock-based structures.
// Reveals how implementations degrade under sustained write pressure.
// =============================================================================

#[pbench::bench(threads = [8], sample_count = 50_000, sample_size = 1, skip_ext_time)]
fn mixed_50_50_8t_masstree(b: &Bencher<'_>) {
    let n: usize = 100_000;
    let (tree, keys) = setup_masstree(n);
    let indices: Vec<usize> = uniform_indices(n, INDEX_POOL, 55);
    let cursor: AtomicUsize = AtomicUsize::new(0);

    b.bench_refs(|| {
        let guard: LocalGuard<'_> = tree.guard();
        let i: usize = cursor.fetch_add(1, Ordering::Relaxed);
        let idx: usize = indices[i % indices.len()];

        if is_write_op(i, 50) {
            let _ = tree.insert_with_guard(&keys[idx], i as u64, &guard);
        } else {
            std::hint::black_box(tree.get_with_guard(&keys[idx], &guard));
        }
    });
}

#[pbench::bench(threads = [8], sample_count = 50_000, sample_size = 1, skip_ext_time)]
fn mixed_50_50_8t_skipmap(b: &Bencher<'_>) {
    let n: usize = 100_000;
    let (map, keys) = setup_skipmap(n);
    let indices: Vec<usize> = uniform_indices(n, INDEX_POOL, 55);
    let cursor: AtomicUsize = AtomicUsize::new(0);

    b.bench_refs(|| {
        let i: usize = cursor.fetch_add(1, Ordering::Relaxed);
        let idx: usize = indices[i % indices.len()];

        if is_write_op(i, 50) {
            map.insert(keys[idx], i as u64);
        } else {
            std::hint::black_box(map.get(&keys[idx]));
        }
    });
}

#[pbench::bench(threads = [8], sample_count = 50_000, sample_size = 1, skip_ext_time)]
fn mixed_50_50_8t_indexset(b: &Bencher<'_>) {
    let n: usize = 100_000;
    let (map, keys) = setup_indexset(n);
    let indices: Vec<usize> = uniform_indices(n, INDEX_POOL, 55);
    let cursor: AtomicUsize = AtomicUsize::new(0);

    b.bench_refs(|| {
        let i: usize = cursor.fetch_add(1, Ordering::Relaxed);
        let idx: usize = indices[i % indices.len()];

        if is_write_op(i, 50) {
            map.insert(keys[idx], i as u64);
        } else {
            std::hint::black_box(map.get(&keys[idx]));
        }
    });
}

#[pbench::bench(threads = [8], sample_count = 50_000, sample_size = 1, skip_ext_time)]
fn mixed_50_50_8t_treeindex(b: &Bencher<'_>) {
    let n: usize = 100_000;
    let (tree, keys) = setup_tree_index(n);
    let indices: Vec<usize> = uniform_indices(n, INDEX_POOL, 55);
    let cursor: AtomicUsize = AtomicUsize::new(0);

    b.bench_refs(|| {
        let guard: SddGuard = SddGuard::new();
        let i: usize = cursor.fetch_add(1, Ordering::Relaxed);
        let idx: usize = indices[i % indices.len()];

        if is_write_op(i, 50) {
            tree_index_upsert_sync(&tree, keys[idx], i as u64);
        } else {
            std::hint::black_box(tree.peek(&keys[idx], &guard));
        }
    });
}

#[pbench::bench(threads = [8], sample_count = 50_000, sample_size = 1, skip_ext_time)]
fn mixed_50_50_8t_dashmap(b: &Bencher<'_>) {
    let n: usize = 100_000;
    let (map, keys) = setup_dashmap(n);
    let indices: Vec<usize> = uniform_indices(n, INDEX_POOL, 55);
    let cursor: AtomicUsize = AtomicUsize::new(0);

    b.bench_refs(|| {
        let i: usize = cursor.fetch_add(1, Ordering::Relaxed);
        let idx: usize = indices[i % indices.len()];

        if is_write_op(i, 50) {
            map.insert(keys[idx], i as u64);
        } else {
            std::hint::black_box(map.get(&keys[idx]));
        }
    });
}

// =============================================================================
// 07: Zipfian hotspot GET, 100k entries, skew=1.0, 8 threads
//
// Hot keys under Zipfian distribution cause cache-line contention.
// Tail latency here shows the cost of false sharing and coherence traffic
// on frequently-accessed nodes.
// =============================================================================

#[pbench::bench(threads = [8], sample_count = 50_000, sample_size = 1, skip_ext_time)]
fn get_zipf_8t_masstree(b: &Bencher<'_>) {
    let n: usize = 100_000;
    let (tree, keys) = setup_masstree(n);
    let indices: Vec<usize> = zipfian_indices(n, INDEX_POOL, 1.0, 42);
    let cursor: AtomicUsize = AtomicUsize::new(0);

    b.bench_refs(|| {
        let guard: LocalGuard<'_> = tree.guard();
        let i: usize = cursor.fetch_add(1, Ordering::Relaxed);
        let idx: usize = indices[i % indices.len()];
        std::hint::black_box(tree.get_with_guard(&keys[idx], &guard));
    });
}

#[pbench::bench(threads = [8], sample_count = 50_000, sample_size = 1, skip_ext_time)]
fn get_zipf_8t_skipmap(b: &Bencher<'_>) {
    let n: usize = 100_000;
    let (map, keys) = setup_skipmap(n);
    let indices: Vec<usize> = zipfian_indices(n, INDEX_POOL, 1.0, 42);
    let cursor: AtomicUsize = AtomicUsize::new(0);

    b.bench_refs(|| {
        let i: usize = cursor.fetch_add(1, Ordering::Relaxed);
        let idx: usize = indices[i % indices.len()];
        std::hint::black_box(map.get(&keys[idx]));
    });
}

#[pbench::bench(threads = [8], sample_count = 50_000, sample_size = 1, skip_ext_time)]
fn get_zipf_8t_indexset(b: &Bencher<'_>) {
    let n: usize = 100_000;
    let (map, keys) = setup_indexset(n);
    let indices: Vec<usize> = zipfian_indices(n, INDEX_POOL, 1.0, 42);
    let cursor: AtomicUsize = AtomicUsize::new(0);

    b.bench_refs(|| {
        let i: usize = cursor.fetch_add(1, Ordering::Relaxed);
        let idx: usize = indices[i % indices.len()];
        std::hint::black_box(map.get(&keys[idx]));
    });
}

#[pbench::bench(threads = [8], sample_count = 50_000, sample_size = 1, skip_ext_time)]
fn get_zipf_8t_treeindex(b: &Bencher<'_>) {
    let n: usize = 100_000;
    let (tree, keys) = setup_tree_index(n);
    let indices: Vec<usize> = zipfian_indices(n, INDEX_POOL, 1.0, 42);
    let cursor: AtomicUsize = AtomicUsize::new(0);

    b.bench_refs(|| {
        let guard: SddGuard = SddGuard::new();
        let i: usize = cursor.fetch_add(1, Ordering::Relaxed);
        let idx: usize = indices[i % indices.len()];
        std::hint::black_box(tree.peek(&keys[idx], &guard));
    });
}

#[pbench::bench(threads = [8], sample_count = 50_000, sample_size = 1, skip_ext_time)]
fn get_zipf_8t_dashmap(b: &Bencher<'_>) {
    let n: usize = 100_000;
    let (map, keys) = setup_dashmap(n);
    let indices: Vec<usize> = zipfian_indices(n, INDEX_POOL, 1.0, 42);
    let cursor: AtomicUsize = AtomicUsize::new(0);

    b.bench_refs(|| {
        let i: usize = cursor.fetch_add(1, Ordering::Relaxed);
        let idx: usize = indices[i % indices.len()];
        std::hint::black_box(map.get(&keys[idx]));
    });
}

// =============================================================================
// 08: Large tree GET, 1M entries, uniform, 1 thread
//
// Large working set exceeds L2 cache, revealing true memory-access latency.
// Tail latency dominated by TLB misses and DRAM latency.
// =============================================================================

#[pbench::bench(threads = [1], sample_count = 50_000, sample_size = 1, max_time = 60, skip_ext_time)]
fn get_1m_1t_masstree(b: &Bencher<'_>) {
    let n: usize = 1_000_000;
    let (tree, keys) = setup_masstree(n);
    let indices: Vec<usize> = uniform_indices(n, INDEX_POOL, 42);
    let cursor: AtomicUsize = AtomicUsize::new(0);

    b.bench_refs(|| {
        let guard: LocalGuard<'_> = tree.guard();
        let i: usize = cursor.fetch_add(1, Ordering::Relaxed);
        let idx: usize = indices[i % indices.len()];
        std::hint::black_box(tree.get_with_guard(&keys[idx], &guard));
    });
}

#[pbench::bench(threads = [1], sample_count = 50_000, sample_size = 1, max_time = 60, skip_ext_time)]
fn get_1m_1t_skipmap(b: &Bencher<'_>) {
    let n: usize = 1_000_000;
    let (map, keys) = setup_skipmap(n);
    let indices: Vec<usize> = uniform_indices(n, INDEX_POOL, 42);
    let cursor: AtomicUsize = AtomicUsize::new(0);

    b.bench_refs(|| {
        let i: usize = cursor.fetch_add(1, Ordering::Relaxed);
        let idx: usize = indices[i % indices.len()];
        std::hint::black_box(map.get(&keys[idx]));
    });
}

#[pbench::bench(threads = [1], sample_count = 50_000, sample_size = 1, max_time = 60, skip_ext_time)]
fn get_1m_1t_indexset(b: &Bencher<'_>) {
    let n: usize = 1_000_000;
    let (map, keys) = setup_indexset(n);
    let indices: Vec<usize> = uniform_indices(n, INDEX_POOL, 42);
    let cursor: AtomicUsize = AtomicUsize::new(0);

    b.bench_refs(|| {
        let i: usize = cursor.fetch_add(1, Ordering::Relaxed);
        let idx: usize = indices[i % indices.len()];
        std::hint::black_box(map.get(&keys[idx]));
    });
}

#[pbench::bench(threads = [1], sample_count = 50_000, sample_size = 1, max_time = 60, skip_ext_time)]
fn get_1m_1t_treeindex(b: &Bencher<'_>) {
    let n: usize = 1_000_000;
    let (tree, keys) = setup_tree_index(n);
    let indices: Vec<usize> = uniform_indices(n, INDEX_POOL, 42);
    let cursor: AtomicUsize = AtomicUsize::new(0);

    b.bench_refs(|| {
        let guard: SddGuard = SddGuard::new();
        let i: usize = cursor.fetch_add(1, Ordering::Relaxed);
        let idx: usize = indices[i % indices.len()];
        std::hint::black_box(tree.peek(&keys[idx], &guard));
    });
}

#[pbench::bench(threads = [1], sample_count = 50_000, sample_size = 1, max_time = 60, skip_ext_time)]
fn get_1m_1t_dashmap(b: &Bencher<'_>) {
    let n: usize = 1_000_000;
    let (map, keys) = setup_dashmap(n);
    let indices: Vec<usize> = uniform_indices(n, INDEX_POOL, 42);
    let cursor: AtomicUsize = AtomicUsize::new(0);

    b.bench_refs(|| {
        let i: usize = cursor.fetch_add(1, Ordering::Relaxed);
        let idx: usize = indices[i % indices.len()];
        std::hint::black_box(map.get(&keys[idx]));
    });
}

// =============================================================================
// 09: Scan latency, 50-key forward scan, 100k entries, 8 threads
//
// Range scans touch multiple cache lines sequentially. Tail latency reveals
// cost of B-link pointer chasing and OCC retries under concurrent mutation.
// =============================================================================

#[pbench::bench(threads = [8], sample_count = 50_000, sample_size = 1, skip_ext_time)]
fn scan_50_8t_masstree(b: &Bencher<'_>) {
    let n: usize = 100_000;
    let (tree, keys) = setup_masstree(n);
    let start_indices: Vec<usize> = uniform_indices(n - 50, INDEX_POOL, 77);
    let cursor: AtomicUsize = AtomicUsize::new(0);

    b.bench_refs(|| {
        let guard: LocalGuard<'_> = tree.guard();
        let i: usize = cursor.fetch_add(1, Ordering::Relaxed);
        let start_idx: usize = start_indices[i % start_indices.len()];
        let mut count: u64 = 0;

        tree.scan(
            RangeBound::Included(&keys[start_idx]),
            RangeBound::Unbounded,
            |_key: &[u8], v: u64| {
                count = count.wrapping_add(v);
                count < 50
            },
            &guard,
        );

        std::hint::black_box(count);
    });
}

#[pbench::bench(threads = [8], sample_count = 50_000, sample_size = 1, skip_ext_time)]
fn scan_50_8t_skipmap(b: &Bencher<'_>) {
    let n: usize = 100_000;
    let (map, keys) = setup_skipmap(n);

    let mut sorted_keys: Vec<[u8; KEY_SIZE]> = keys;
    sorted_keys.sort_unstable();

    let start_indices: Vec<usize> = uniform_indices(n - 50, INDEX_POOL, 77);
    let cursor: AtomicUsize = AtomicUsize::new(0);

    b.bench_refs(|| {
        let i: usize = cursor.fetch_add(1, Ordering::Relaxed);
        let start_idx: usize = start_indices[i % start_indices.len()];
        let mut count: u64 = 0;

        for entry in map
            .range::<[u8; KEY_SIZE], _>(&sorted_keys[start_idx]..)
            .take(50)
        {
            count = count.wrapping_add(*entry.value());
        }

        std::hint::black_box(count);
    });
}

#[pbench::bench(threads = [8], sample_count = 50_000, sample_size = 1, skip_ext_time)]
fn scan_50_8t_treeindex(b: &Bencher<'_>) {
    let n: usize = 100_000;
    let (tree, keys) = setup_tree_index(n);

    let mut sorted_keys: Vec<[u8; KEY_SIZE]> = keys;
    sorted_keys.sort_unstable();

    let start_indices: Vec<usize> = uniform_indices(n - 50, INDEX_POOL, 77);
    let cursor: AtomicUsize = AtomicUsize::new(0);

    b.bench_refs(|| {
        let guard: SddGuard = SddGuard::new();
        let i: usize = cursor.fetch_add(1, Ordering::Relaxed);
        let start_idx: usize = start_indices[i % start_indices.len()];
        let mut count: u64 = 0;

        for entry in tree
            .range::<[u8; KEY_SIZE], _>(&sorted_keys[start_idx].., &guard)
            .take(50)
        {
            count = count.wrapping_add(*entry.1);
        }

        std::hint::black_box(count);
    });
}

// =============================================================================
// 10: Remove latency, 100k pre-filled entries, 8 threads
//
// Delete path triggers coalescing and node recycling. Tail latency exposes
// the cost of lock acquisition on the coalesce path and memory reclamation.
// TreeIndex and DashMap support remove; SkipMap has remove; IndexSet does not.
// =============================================================================

#[pbench::bench(threads = [8], sample_count = 50_000, sample_size = 1, skip_ext_time)]
fn remove_8t_masstree(b: &Bencher<'_>) {
    let n: usize = 100_000;
    let (tree, keys) = setup_masstree(n);
    let indices: Vec<usize> = uniform_indices(n, INDEX_POOL, 99);
    let cursor: AtomicUsize = AtomicUsize::new(0);

    b.bench_refs(|| {
        let guard: LocalGuard<'_> = tree.guard();
        let i: usize = cursor.fetch_add(1, Ordering::Relaxed);
        let idx: usize = indices[i % indices.len()];

        // Alternate remove/reinsert to keep the tree non-empty
        if is_write_op(i, 50) {
            let _ = std::hint::black_box(tree.remove_with_guard(&keys[idx], &guard));
        } else {
            let _ = tree.insert_with_guard(&keys[idx], i as u64, &guard);
        }
    });
}

#[pbench::bench(threads = [8], sample_count = 50_000, sample_size = 1, skip_ext_time)]
fn remove_8t_skipmap(b: &Bencher<'_>) {
    let n: usize = 100_000;
    let (map, keys) = setup_skipmap(n);
    let indices: Vec<usize> = uniform_indices(n, INDEX_POOL, 99);
    let cursor: AtomicUsize = AtomicUsize::new(0);

    b.bench_refs(|| {
        let i: usize = cursor.fetch_add(1, Ordering::Relaxed);
        let idx: usize = indices[i % indices.len()];

        if is_write_op(i, 50) {
            std::hint::black_box(map.remove(&keys[idx]));
        } else {
            map.insert(keys[idx], i as u64);
        }
    });
}

#[pbench::bench(threads = [8], sample_count = 50_000, sample_size = 1, skip_ext_time)]
fn remove_8t_treeindex(b: &Bencher<'_>) {
    let n: usize = 100_000;
    let (tree, keys) = setup_tree_index(n);
    let indices: Vec<usize> = uniform_indices(n, INDEX_POOL, 99);
    let cursor: AtomicUsize = AtomicUsize::new(0);

    b.bench_refs(|| {
        let i: usize = cursor.fetch_add(1, Ordering::Relaxed);
        let idx: usize = indices[i % indices.len()];

        if is_write_op(i, 50) {
            std::hint::black_box(tree.remove_sync(&keys[idx]));
        } else {
            let _ = tree.insert_sync(keys[idx], i as u64);
        }
    });
}

#[pbench::bench(threads = [8], sample_count = 50_000, sample_size = 1, skip_ext_time)]
fn remove_8t_dashmap(b: &Bencher<'_>) {
    let n: usize = 100_000;
    let (map, keys) = setup_dashmap(n);
    let indices: Vec<usize> = uniform_indices(n, INDEX_POOL, 99);
    let cursor: AtomicUsize = AtomicUsize::new(0);

    b.bench_refs(|| {
        let i: usize = cursor.fetch_add(1, Ordering::Relaxed);
        let idx: usize = indices[i % indices.len()];

        if is_write_op(i, 50) {
            std::hint::black_box(map.remove(&keys[idx]));
        } else {
            map.insert(keys[idx], i as u64);
        }
    });
}

// =============================================================================
// 11: Mixed read-write-remove (realistic CRUD), 8 threads
//
// 70% read, 15% insert, 15% remove. Models a real key-value store workload
// where entries are created, queried, and eventually expired/deleted.
// =============================================================================

#[pbench::bench(threads = [8], sample_count = 50_000, sample_size = 1, skip_ext_time)]
fn crud_8t_masstree(b: &Bencher<'_>) {
    let n: usize = 100_000;
    let (tree, keys) = setup_masstree(n);
    let indices: Vec<usize> = uniform_indices(n, INDEX_POOL, 88);
    let cursor: AtomicUsize = AtomicUsize::new(0);

    b.bench_refs(|| {
        let guard: LocalGuard<'_> = tree.guard();
        let i: usize = cursor.fetch_add(1, Ordering::Relaxed);
        let idx: usize = indices[i % indices.len()];
        let op: u64 = {
            let mut x: u64 = i as u64;
            x ^= x >> 17;
            x = x.wrapping_mul(0xbf58_476d_1ce4_e5b9);
            x ^= x >> 31;
            x % 100
        };

        if op < 70 {
            std::hint::black_box(tree.get_with_guard(&keys[idx], &guard));
        } else if op < 85 {
            let _ = tree.insert_with_guard(&keys[idx], i as u64, &guard);
        } else {
            let _ = std::hint::black_box(tree.remove_with_guard(&keys[idx], &guard));
        }
    });
}

#[pbench::bench(threads = [8], sample_count = 50_000, sample_size = 1, skip_ext_time)]
fn crud_8t_skipmap(b: &Bencher<'_>) {
    let n: usize = 100_000;
    let (map, keys) = setup_skipmap(n);
    let indices: Vec<usize> = uniform_indices(n, INDEX_POOL, 88);
    let cursor: AtomicUsize = AtomicUsize::new(0);

    b.bench_refs(|| {
        let i: usize = cursor.fetch_add(1, Ordering::Relaxed);
        let idx: usize = indices[i % indices.len()];
        let op: u64 = {
            let mut x: u64 = i as u64;
            x ^= x >> 17;
            x = x.wrapping_mul(0xbf58_476d_1ce4_e5b9);
            x ^= x >> 31;
            x % 100
        };

        if op < 70 {
            std::hint::black_box(map.get(&keys[idx]));
        } else if op < 85 {
            map.insert(keys[idx], i as u64);
        } else {
            std::hint::black_box(map.remove(&keys[idx]));
        }
    });
}

#[pbench::bench(threads = [8], sample_count = 50_000, sample_size = 1, skip_ext_time)]
fn crud_8t_treeindex(b: &Bencher<'_>) {
    let n: usize = 100_000;
    let (tree, keys) = setup_tree_index(n);
    let indices: Vec<usize> = uniform_indices(n, INDEX_POOL, 88);
    let cursor: AtomicUsize = AtomicUsize::new(0);

    b.bench_refs(|| {
        let guard: SddGuard = SddGuard::new();
        let i: usize = cursor.fetch_add(1, Ordering::Relaxed);
        let idx: usize = indices[i % indices.len()];
        let op: u64 = {
            let mut x: u64 = i as u64;
            x ^= x >> 17;
            x = x.wrapping_mul(0xbf58_476d_1ce4_e5b9);
            x ^= x >> 31;
            x % 100
        };

        if op < 70 {
            std::hint::black_box(tree.peek(&keys[idx], &guard));
        } else if op < 85 {
            tree_index_upsert_sync(&tree, keys[idx], i as u64);
        } else {
            std::hint::black_box(tree.remove_sync(&keys[idx]));
        }
    });
}

#[pbench::bench(threads = [8], sample_count = 50_000, sample_size = 1, skip_ext_time)]
fn crud_8t_dashmap(b: &Bencher<'_>) {
    let n: usize = 100_000;
    let (map, keys) = setup_dashmap(n);
    let indices: Vec<usize> = uniform_indices(n, INDEX_POOL, 88);
    let cursor: AtomicUsize = AtomicUsize::new(0);

    b.bench_refs(|| {
        let i: usize = cursor.fetch_add(1, Ordering::Relaxed);
        let idx: usize = indices[i % indices.len()];
        let op: u64 = {
            let mut x: u64 = i as u64;
            x ^= x >> 17;
            x = x.wrapping_mul(0xbf58_476d_1ce4_e5b9);
            x ^= x >> 31;
            x % 100
        };

        if op < 70 {
            std::hint::black_box(map.get(&keys[idx]));
        } else if op < 85 {
            map.insert(keys[idx], i as u64);
        } else {
            std::hint::black_box(map.remove(&keys[idx]));
        }
    });
}

// =============================================================================
// 12: Zipfian mixed read-write, 100k entries, 8 threads
//
// Combines hotspot access with writes. Models real caches where popular keys
// get both read and updated frequently. The interaction between hot-key
// contention and write locks creates the most realistic tail distribution.
// =============================================================================

#[pbench::bench(threads = [8], sample_count = 50_000, sample_size = 1, skip_ext_time)]
fn mixed_zipf_90_10_8t_masstree(b: &Bencher<'_>) {
    let n: usize = 100_000;
    let (tree, keys) = setup_masstree(n);
    let indices: Vec<usize> = zipfian_indices(n, INDEX_POOL, 1.0, 42);
    let cursor: AtomicUsize = AtomicUsize::new(0);

    b.bench_refs(|| {
        let guard: LocalGuard<'_> = tree.guard();
        let i: usize = cursor.fetch_add(1, Ordering::Relaxed);
        let idx: usize = indices[i % indices.len()];

        if is_write_op(i, 10) {
            let _ = tree.insert_with_guard(&keys[idx], i as u64, &guard);
        } else {
            std::hint::black_box(tree.get_with_guard(&keys[idx], &guard));
        }
    });
}

#[pbench::bench(threads = [8], sample_count = 50_000, sample_size = 1, skip_ext_time)]
fn mixed_zipf_90_10_8t_skipmap(b: &Bencher<'_>) {
    let n: usize = 100_000;
    let (map, keys) = setup_skipmap(n);
    let indices: Vec<usize> = zipfian_indices(n, INDEX_POOL, 1.0, 42);
    let cursor: AtomicUsize = AtomicUsize::new(0);

    b.bench_refs(|| {
        let i: usize = cursor.fetch_add(1, Ordering::Relaxed);
        let idx: usize = indices[i % indices.len()];

        if is_write_op(i, 10) {
            map.insert(keys[idx], i as u64);
        } else {
            std::hint::black_box(map.get(&keys[idx]));
        }
    });
}

#[pbench::bench(threads = [8], sample_count = 50_000, sample_size = 1, skip_ext_time)]
fn mixed_zipf_90_10_8t_treeindex(b: &Bencher<'_>) {
    let n: usize = 100_000;
    let (tree, keys) = setup_tree_index(n);
    let indices: Vec<usize> = zipfian_indices(n, INDEX_POOL, 1.0, 42);
    let cursor: AtomicUsize = AtomicUsize::new(0);

    b.bench_refs(|| {
        let guard: SddGuard = SddGuard::new();
        let i: usize = cursor.fetch_add(1, Ordering::Relaxed);
        let idx: usize = indices[i % indices.len()];

        if is_write_op(i, 10) {
            tree_index_upsert_sync(&tree, keys[idx], i as u64);
        } else {
            std::hint::black_box(tree.peek(&keys[idx], &guard));
        }
    });
}

#[pbench::bench(threads = [8], sample_count = 50_000, sample_size = 1, skip_ext_time)]
fn mixed_zipf_90_10_8t_dashmap(b: &Bencher<'_>) {
    let n: usize = 100_000;
    let (map, keys) = setup_dashmap(n);
    let indices: Vec<usize> = zipfian_indices(n, INDEX_POOL, 1.0, 42);
    let cursor: AtomicUsize = AtomicUsize::new(0);

    b.bench_refs(|| {
        let i: usize = cursor.fetch_add(1, Ordering::Relaxed);
        let idx: usize = indices[i % indices.len()];

        if is_write_op(i, 10) {
            map.insert(keys[idx], i as u64);
        } else {
            std::hint::black_box(map.get(&keys[idx]));
        }
    });
}

// =============================================================================
// 13: 8-byte keys GET, 100k entries, 8 threads
//
// Single-layer fast path for masstree (no trie descent). Tests the optimal
// case where keys fit in one ikey. Tail latency should be minimal for
// masstree since there is no inter-layer traversal.
// =============================================================================

const KEY8: usize = 8;

fn setup_masstree_8b(n: usize) -> (MassTree15Inline<u64>, Vec<[u8; KEY8]>) {
    let keys: Vec<[u8; KEY8]> = bench_utils::keys(n);
    let tree: MassTree15Inline<u64> = MassTree15Inline::new();

    {
        let guard: LocalGuard<'_> = tree.guard();

        for (i, key) in keys.iter().enumerate() {
            let _ = tree.insert_with_guard(key, i as u64, &guard);
        }
    }

    (tree, keys)
}

fn setup_skipmap_8b(n: usize) -> (SkipMap<[u8; KEY8], u64>, Vec<[u8; KEY8]>) {
    let keys: Vec<[u8; KEY8]> = bench_utils::keys(n);
    let map: SkipMap<[u8; KEY8], u64> = SkipMap::new();

    for (i, key) in keys.iter().enumerate() {
        map.insert(*key, i as u64);
    }

    (map, keys)
}

fn setup_dashmap_8b(n: usize) -> (DashMap<[u8; KEY8], u64>, Vec<[u8; KEY8]>) {
    let keys: Vec<[u8; KEY8]> = bench_utils::keys(n);
    let map: DashMap<[u8; KEY8], u64> = DashMap::new();

    for (i, key) in keys.iter().enumerate() {
        map.insert(*key, i as u64);
    }

    (map, keys)
}

fn setup_tree_index_8b(n: usize) -> (TreeIndex<[u8; KEY8], u64>, Vec<[u8; KEY8]>) {
    let keys: Vec<[u8; KEY8]> = bench_utils::keys(n);
    let tree: TreeIndex<[u8; KEY8], u64> = TreeIndex::new();

    for (i, key) in keys.iter().enumerate() {
        let _ = tree.insert_sync(*key, i as u64);
    }

    (tree, keys)
}

#[pbench::bench(threads = [8], sample_count = 50_000, sample_size = 1, skip_ext_time)]
fn get_8b_8t_masstree(b: &Bencher<'_>) {
    let n: usize = 100_000;
    let (tree, keys) = setup_masstree_8b(n);
    let indices: Vec<usize> = uniform_indices(n, INDEX_POOL, 42);
    let cursor: AtomicUsize = AtomicUsize::new(0);

    b.bench_refs(|| {
        let guard: LocalGuard<'_> = tree.guard();
        let i: usize = cursor.fetch_add(1, Ordering::Relaxed);
        let idx: usize = indices[i % indices.len()];
        std::hint::black_box(tree.get_with_guard(&keys[idx], &guard));
    });
}

#[pbench::bench(threads = [8], sample_count = 50_000, sample_size = 1, skip_ext_time)]
fn get_8b_8t_skipmap(b: &Bencher<'_>) {
    let n: usize = 100_000;
    let (map, keys) = setup_skipmap_8b(n);
    let indices: Vec<usize> = uniform_indices(n, INDEX_POOL, 42);
    let cursor: AtomicUsize = AtomicUsize::new(0);

    b.bench_refs(|| {
        let i: usize = cursor.fetch_add(1, Ordering::Relaxed);
        let idx: usize = indices[i % indices.len()];
        std::hint::black_box(map.get(&keys[idx]));
    });
}

#[pbench::bench(threads = [8], sample_count = 50_000, sample_size = 1, skip_ext_time)]
fn get_8b_8t_treeindex(b: &Bencher<'_>) {
    let n: usize = 100_000;
    let (tree, keys) = setup_tree_index_8b(n);
    let indices: Vec<usize> = uniform_indices(n, INDEX_POOL, 42);
    let cursor: AtomicUsize = AtomicUsize::new(0);

    b.bench_refs(|| {
        let guard: SddGuard = SddGuard::new();
        let i: usize = cursor.fetch_add(1, Ordering::Relaxed);
        let idx: usize = indices[i % indices.len()];
        std::hint::black_box(tree.peek(&keys[idx], &guard));
    });
}

#[pbench::bench(threads = [8], sample_count = 50_000, sample_size = 1, skip_ext_time)]
fn get_8b_8t_dashmap(b: &Bencher<'_>) {
    let n: usize = 100_000;
    let (map, keys) = setup_dashmap_8b(n);
    let indices: Vec<usize> = uniform_indices(n, INDEX_POOL, 42);
    let cursor: AtomicUsize = AtomicUsize::new(0);

    b.bench_refs(|| {
        let i: usize = cursor.fetch_add(1, Ordering::Relaxed);
        let idx: usize = indices[i % indices.len()];
        std::hint::black_box(map.get(&keys[idx]));
    });
}

// =============================================================================
// 14: Scan under concurrent writes, 50-key scan + 10% writes, 8 threads
//
// Range scans competing with concurrent inserts. This is the hardest scenario
// for OCC-based trees: scans must detect version changes mid-traversal and
// restart. Tail latency directly measures OCC retry cost.
// =============================================================================

#[pbench::bench(threads = [8], sample_count = 50_000, sample_size = 1, skip_ext_time)]
fn scan_write_8t_masstree(b: &Bencher<'_>) {
    let n: usize = 100_000;
    let (tree, keys) = setup_masstree(n);
    let indices: Vec<usize> = uniform_indices(n, INDEX_POOL, 77);
    let cursor: AtomicUsize = AtomicUsize::new(0);

    b.bench_refs(|| {
        let guard: LocalGuard<'_> = tree.guard();
        let i: usize = cursor.fetch_add(1, Ordering::Relaxed);
        let idx: usize = indices[i % indices.len()];

        if is_write_op(i, 10) {
            let _ = tree.insert_with_guard(&keys[idx], i as u64, &guard);
        } else {
            let start_idx: usize = idx.min(n - 51);
            let mut count: u64 = 0;

            tree.scan(
                RangeBound::Included(&keys[start_idx]),
                RangeBound::Unbounded,
                |_key: &[u8], v: u64| {
                    count = count.wrapping_add(v);
                    count < 50
                },
                &guard,
            );

            std::hint::black_box(count);
        }
    });
}

#[pbench::bench(threads = [8], sample_count = 50_000, sample_size = 1, skip_ext_time)]
fn scan_write_8t_skipmap(b: &Bencher<'_>) {
    let n: usize = 100_000;
    let (map, keys) = setup_skipmap(n);

    let mut sorted_keys: Vec<[u8; KEY_SIZE]> = keys.clone();
    sorted_keys.sort_unstable();

    let indices: Vec<usize> = uniform_indices(n, INDEX_POOL, 77);
    let cursor: AtomicUsize = AtomicUsize::new(0);

    b.bench_refs(|| {
        let i: usize = cursor.fetch_add(1, Ordering::Relaxed);
        let idx: usize = indices[i % indices.len()];

        if is_write_op(i, 10) {
            map.insert(keys[idx], i as u64);
        } else {
            let start_idx: usize = idx.min(n - 51);
            let mut count: u64 = 0;

            for entry in map
                .range::<[u8; KEY_SIZE], _>(&sorted_keys[start_idx]..)
                .take(50)
            {
                count = count.wrapping_add(*entry.value());
            }

            std::hint::black_box(count);
        }
    });
}

#[pbench::bench(threads = [8], sample_count = 50_000, sample_size = 1, skip_ext_time)]
fn scan_write_8t_treeindex(b: &Bencher<'_>) {
    let n: usize = 100_000;
    let (tree, keys) = setup_tree_index(n);

    let mut sorted_keys: Vec<[u8; KEY_SIZE]> = keys.clone();
    sorted_keys.sort_unstable();

    let indices: Vec<usize> = uniform_indices(n, INDEX_POOL, 77);
    let cursor: AtomicUsize = AtomicUsize::new(0);

    b.bench_refs(|| {
        let guard: SddGuard = SddGuard::new();
        let i: usize = cursor.fetch_add(1, Ordering::Relaxed);
        let idx: usize = indices[i % indices.len()];

        if is_write_op(i, 10) {
            tree_index_upsert_sync(&tree, keys[idx], i as u64);
        } else {
            let start_idx: usize = idx.min(n - 51);
            let mut count: u64 = 0;

            for entry in tree
                .range::<[u8; KEY_SIZE], _>(&sorted_keys[start_idx].., &guard)
                .take(50)
            {
                count = count.wrapping_add(*entry.1);
            }

            std::hint::black_box(count);
        }
    });
}

fn main() {
    pbench::main();
}
