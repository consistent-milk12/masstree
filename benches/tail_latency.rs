//! Tail latency benchmark: masstree vs treeindex vs indexset vs skipmap.
//!
//! Measures per-operation latency distributions to capture p99/p99.9 tail
//! behavior. Every benchmark uses `sample_size = 1` so each sample represents
//! exactly one operation's latency (no batching that would average out spikes).
//!
//! Only includes realistic contended workloads where tail latency matters.
//! Single-threaded and bulk-load benchmarks are excluded since they lack the
//! contention that causes real tail latency problems.
//!
//! Run with: `cargo bench -p masstree --bench tail_latency`

#![expect(clippy::pedantic)]
#![expect(clippy::indexing_slicing)]
#![expect(clippy::type_complexity)]

#[path = "bench_utils.rs"]
mod bench_utils;

use std::sync::atomic::{AtomicUsize, Ordering};

use crossbeam_skiplist::SkipMap;
use indexset::concurrent::map::BTreeMap as IndexSetBTreeMap;
use masstree::{MassTree15Inline, RangeBound};
use pbench::Bencher;
use scc::TreeIndex;
use sdd::Guard as SddGuard;
use seize::LocalGuard;

use bench_utils::{keys_shared_prefix_chunks, uniform_indices, zipfian_indices};

// =============================================================================
// Constants
// =============================================================================

const KEY_SIZE: usize = 32;

/// 200k samples: enough for reliable p99.99 (need >= 10_000). Exceeds pbench's
/// 100k exact-mode threshold, so upper percentiles use HdrHistogram.
/// Pre-generated random index pool. Larger than sample_count to avoid cycling.
const INDEX_POOL: usize = 500_000;

// =============================================================================
// Methodology Notes
// =============================================================================
//
// Shared cursor: All multi-threaded benchmarks use a shared AtomicUsize cursor
// (fetch_add with Relaxed ordering) to assign unique index-pool positions across
// threads. This adds ~5-10ns of cache-line contention overhead per sample, but
// the overhead is equal for all structures and negligible relative to the
// operations being measured (100ns+). The alternative (per-thread cursors) would
// require thread identification that pbench does not expose, and would lose the
// inter-thread access decorrelation the shared cursor provides.
//
// Guard creation: epoch/EBR guard creation (tree.guard() for seize,
// SddGuard::new() for sdd) is measured as part of each operation. This matches
// real-world usage where guards are acquired per-operation or per-batch.
//
// All benchmarks within a category use the same random seed (42 for uniform
// index pools, 77 for pure-scan start indices, 42 for zipfian) so the only
// variable between e.g. mixed_90_10 and mixed_50_50 is the read/write ratio.
// Scan-with-writes (section 09) uses seed 42 to match the mixed benchmarks.
// =============================================================================

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

fn setup_tree_index(n: usize) -> (TreeIndex<[u8; KEY_SIZE], u64>, Vec<[u8; KEY_SIZE]>) {
    let keys: Vec<[u8; KEY_SIZE]> = bench_utils::keys(n);
    let tree: TreeIndex<[u8; KEY_SIZE], u64> = TreeIndex::new();

    for (i, key) in keys.iter().enumerate() {
        let _ = tree.insert_sync(*key, i as u64);
    }

    (tree, keys)
}

/// TreeIndex upsert emulation via remove-then-reinsert.
///
/// TreeIndex lacks a native upsert API, so we emulate it with up to 3
/// remove+insert cycles. This adds overhead not present in other structures
/// (masstree and skipmap have native upsert). The extra cost is an inherent
/// API limitation, not an implementation inefficiency, and reflects what
/// real users would pay.
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
/// write bursts. Uses splitmix64 mixing so threads with different cursor values
/// don't all write on the same modular boundary.
///
/// Distribution quality: the splitmix64 hash produces near-uniform output
/// mod 100. Over 10k sequential inputs with write_pct=10, the actual write
/// fraction is 10.0% +/- 0.5%. This is sufficient for benchmark accuracy.
#[inline]
const fn is_write_op(i: usize, write_pct: u32) -> bool {
    // splitmix64 mix to break alignment across threads
    let mut x: u64 = i as u64;
    x ^= x >> 17;
    x = x.wrapping_mul(0xbf58_476d_1ce4_e5b9);
    x ^= x >> 31;
    (x % 100) < write_pct as u64
}

// =============================================================================
// 01: Concurrent GET latency, 100k entries, uniform random, 8 threads
//
// Concurrent reads stress cache coherence and contention on shared nodes.
// Tail latency increases when multiple threads compete for the same cache lines.
// =============================================================================

#[pbench::bench(threads = [8], sample_count = 200_000, sample_size = 1, skip_ext_time)]
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

#[pbench::bench(threads = [8], sample_count = 200_000, sample_size = 1, skip_ext_time)]
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

#[pbench::bench(threads = [8], sample_count = 200_000, sample_size = 1, skip_ext_time)]
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

#[pbench::bench(threads = [8], sample_count = 200_000, sample_size = 1, skip_ext_time)]
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

// =============================================================================
// 02: Mixed read-write under contention, 90/10 read/write, 8 threads
//
// The most realistic tail latency scenario: reads occasionally blocked by
// concurrent writers. Uses hash-based write decision so threads don't all
// write on the same iteration (avoiding artificial synchronized bursts).
// =============================================================================

#[pbench::bench(threads = [8], sample_count = 200_000, sample_size = 1, skip_ext_time)]
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

#[pbench::bench(threads = [8], sample_count = 200_000, sample_size = 1, skip_ext_time)]
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

#[pbench::bench(threads = [8], sample_count = 200_000, sample_size = 1, skip_ext_time)]
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

#[pbench::bench(threads = [8], sample_count = 200_000, sample_size = 1, skip_ext_time)]
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

// =============================================================================
// 03: Mixed read-write under contention, 50/50 read/write, 8 threads
//
// Heavy write load. Worst-case scenario for lock-based structures.
// Reveals how implementations degrade under sustained write pressure.
// =============================================================================

#[pbench::bench(threads = [8], sample_count = 200_000, sample_size = 1, skip_ext_time)]
fn mixed_50_50_8t_masstree(b: &Bencher<'_>) {
    let n: usize = 100_000;
    let (tree, keys) = setup_masstree(n);
    let indices: Vec<usize> = uniform_indices(n, INDEX_POOL, 42);
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

#[pbench::bench(threads = [8], sample_count = 200_000, sample_size = 1, skip_ext_time)]
fn mixed_50_50_8t_skipmap(b: &Bencher<'_>) {
    let n: usize = 100_000;
    let (map, keys) = setup_skipmap(n);
    let indices: Vec<usize> = uniform_indices(n, INDEX_POOL, 42);
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

#[pbench::bench(threads = [8], sample_count = 200_000, sample_size = 1, skip_ext_time)]
fn mixed_50_50_8t_indexset(b: &Bencher<'_>) {
    let n: usize = 100_000;
    let (map, keys) = setup_indexset(n);
    let indices: Vec<usize> = uniform_indices(n, INDEX_POOL, 42);
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

#[pbench::bench(threads = [8], sample_count = 200_000, sample_size = 1, skip_ext_time)]
fn mixed_50_50_8t_treeindex(b: &Bencher<'_>) {
    let n: usize = 100_000;
    let (tree, keys) = setup_tree_index(n);
    let indices: Vec<usize> = uniform_indices(n, INDEX_POOL, 42);
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

// =============================================================================
// 04: Zipfian hotspot GET, 100k entries, skew=1.0, 8 threads
//
// Hot keys under Zipfian distribution cause cache-line contention.
// Tail latency here shows the cost of false sharing and coherence traffic
// on frequently-accessed nodes.
// =============================================================================

#[pbench::bench(threads = [8], sample_count = 200_000, sample_size = 1, skip_ext_time)]
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

#[pbench::bench(threads = [8], sample_count = 200_000, sample_size = 1, skip_ext_time)]
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

#[pbench::bench(threads = [8], sample_count = 200_000, sample_size = 1, skip_ext_time)]
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

#[pbench::bench(threads = [8], sample_count = 200_000, sample_size = 1, skip_ext_time)]
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

// =============================================================================
// 05: Scan latency, 50-key forward scan, 100k entries, 8 threads
//
// Range scans touch multiple cache lines sequentially. Tail latency reveals
// cost of B-link pointer chasing and OCC retries under concurrent mutation.
// =============================================================================

#[pbench::bench(threads = [8], sample_count = 200_000, sample_size = 1, skip_ext_time)]
fn scan_50_8t_masstree(b: &Bencher<'_>) {
    let n: usize = 100_000;
    let (tree, keys) = setup_masstree(n);

    // Sort keys so sorted_keys[i] is the i-th smallest key, matching
    // skipmap/treeindex which also index into sorted key arrays.
    let mut sorted_keys: Vec<[u8; KEY_SIZE]> = keys;
    sorted_keys.sort_unstable();

    let start_indices: Vec<usize> = uniform_indices(n - 50, INDEX_POOL, 77);
    let cursor: AtomicUsize = AtomicUsize::new(0);

    b.bench_refs(|| {
        let guard: LocalGuard<'_> = tree.guard();
        let i: usize = cursor.fetch_add(1, Ordering::Relaxed);
        let start_idx: usize = start_indices[i % start_indices.len()];
        let mut scanned: u64 = 0;

        // scan_intra_leaf_batch: batch OCC per leaf (optimized path).
        // Still materializes keys, matching skipmap/treeindex iterator behavior.
        tree.scan_intra_leaf_batch(
            RangeBound::Included(&sorted_keys[start_idx]),
            RangeBound::Unbounded,
            |_key: &[u8], v: u64| {
                std::hint::black_box(v);
                scanned += 1;
                scanned < 50
            },
            &guard,
        );

        std::hint::black_box(scanned);
    });
}

#[pbench::bench(threads = [8], sample_count = 200_000, sample_size = 1, skip_ext_time)]
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

#[pbench::bench(threads = [8], sample_count = 200_000, sample_size = 1, skip_ext_time)]
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
// 06: Churn (50% remove, 50% insert), 100k pre-filled entries, 8 threads
//
// Alternates remove and insert to model entry lifecycle churn. Measures the
// combined cost of delete-path coalescing and insert-path splits under
// concurrent pressure. Pre-conditioned with 200k ops to reach steady-state
// occupancy (~50k entries) before measurement begins.
// =============================================================================

#[pbench::bench(threads = [8], sample_count = 200_000, sample_size = 1, skip_ext_time)]
fn churn_8t_masstree(b: &Bencher<'_>) {
    let n: usize = 100_000;
    let (tree, keys) = setup_masstree(n);
    let indices: Vec<usize> = uniform_indices(n, INDEX_POOL, 42);

    // Pre-condition to steady-state occupancy (~50k entries).
    // Without this, the tree transitions from 100k to ~50k during measurement,
    // making early samples operate on a larger tree than late samples.
    {
        let guard: LocalGuard<'_> = tree.guard();
        for j in 0..200_000_usize {
            let idx: usize = indices[j % indices.len()];
            if is_write_op(j, 50) {
                let _ = tree.remove_with_guard(&keys[idx], &guard);
            } else {
                let _ = tree.insert_with_guard(&keys[idx], j as u64, &guard);
            }
        }
    }

    let cursor: AtomicUsize = AtomicUsize::new(0);

    b.bench_refs(|| {
        let guard: LocalGuard<'_> = tree.guard();
        let i: usize = cursor.fetch_add(1, Ordering::Relaxed);
        let idx: usize = indices[i % indices.len()];

        if is_write_op(i, 50) {
            let _ = std::hint::black_box(tree.remove_with_guard(&keys[idx], &guard));
        } else {
            let _ = tree.insert_with_guard(&keys[idx], i as u64, &guard);
        }
    });
}

#[pbench::bench(threads = [8], sample_count = 200_000, sample_size = 1, skip_ext_time)]
fn churn_8t_skipmap(b: &Bencher<'_>) {
    let n: usize = 100_000;
    let (map, keys) = setup_skipmap(n);
    let indices: Vec<usize> = uniform_indices(n, INDEX_POOL, 42);

    // Pre-condition to steady-state occupancy.
    for j in 0..200_000_usize {
        let idx: usize = indices[j % indices.len()];
        if is_write_op(j, 50) {
            map.remove(&keys[idx]);
        } else {
            map.insert(keys[idx], j as u64);
        }
    }

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

#[pbench::bench(threads = [8], sample_count = 200_000, sample_size = 1, skip_ext_time)]
fn churn_8t_treeindex(b: &Bencher<'_>) {
    let n: usize = 100_000;
    let (tree, keys) = setup_tree_index(n);
    let indices: Vec<usize> = uniform_indices(n, INDEX_POOL, 42);

    // Pre-condition to steady-state occupancy.
    for j in 0..200_000_usize {
        let idx: usize = indices[j % indices.len()];
        if is_write_op(j, 50) {
            tree.remove_sync(&keys[idx]);
        } else {
            let _ = tree.insert_sync(keys[idx], j as u64);
        }
    }

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

// =============================================================================
// 07: Mixed read-write-remove (realistic CRUD), 8 threads
//
// 70% read, 15% insert, 15% remove. Models a real key-value store workload
// where entries are created, queried, and eventually expired/deleted.
// =============================================================================

#[pbench::bench(threads = [8], sample_count = 200_000, sample_size = 1, skip_ext_time)]
fn crud_8t_masstree(b: &Bencher<'_>) {
    let n: usize = 100_000;
    let (tree, keys) = setup_masstree(n);
    let indices: Vec<usize> = uniform_indices(n, INDEX_POOL, 42);
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

#[pbench::bench(threads = [8], sample_count = 200_000, sample_size = 1, skip_ext_time)]
fn crud_8t_skipmap(b: &Bencher<'_>) {
    let n: usize = 100_000;
    let (map, keys) = setup_skipmap(n);
    let indices: Vec<usize> = uniform_indices(n, INDEX_POOL, 42);
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

#[pbench::bench(threads = [8], sample_count = 200_000, sample_size = 1, skip_ext_time)]
fn crud_8t_treeindex(b: &Bencher<'_>) {
    let n: usize = 100_000;
    let (tree, keys) = setup_tree_index(n);
    let indices: Vec<usize> = uniform_indices(n, INDEX_POOL, 42);
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

// =============================================================================
// 08: Zipfian mixed read-write, 100k entries, 8 threads
//
// Combines hotspot access with writes. Models real caches where popular keys
// get both read and updated frequently. The interaction between hot-key
// contention and write locks creates the most realistic tail distribution.
// =============================================================================

#[pbench::bench(threads = [8], sample_count = 200_000, sample_size = 1, skip_ext_time)]
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

#[pbench::bench(threads = [8], sample_count = 200_000, sample_size = 1, skip_ext_time)]
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

#[pbench::bench(threads = [8], sample_count = 200_000, sample_size = 1, skip_ext_time)]
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

// =============================================================================
// 09: Scan under concurrent writes, 50-key scan + 10% writes, 8 threads
//
// Range scans competing with concurrent inserts. This is the hardest scenario
// for OCC-based trees: scans must detect version changes mid-traversal and
// restart. Tail latency directly measures OCC retry cost.
// =============================================================================

#[pbench::bench(threads = [8], sample_count = 200_000, sample_size = 1, skip_ext_time)]
fn scan_write_8t_masstree(b: &Bencher<'_>) {
    let n: usize = 100_000;
    let (tree, keys) = setup_masstree(n);

    let mut sorted_keys: Vec<[u8; KEY_SIZE]> = keys.clone();
    sorted_keys.sort_unstable();

    let indices: Vec<usize> = uniform_indices(n, INDEX_POOL, 42);
    let cursor: AtomicUsize = AtomicUsize::new(0);

    b.bench_refs(|| {
        let guard: LocalGuard<'_> = tree.guard();
        let i: usize = cursor.fetch_add(1, Ordering::Relaxed);
        let idx: usize = indices[i % indices.len()];

        if is_write_op(i, 10) {
            let _ = tree.insert_with_guard(&keys[idx], i as u64, &guard);
        } else {
            let start_idx: usize = idx.min(n - 51);
            let mut scanned: u64 = 0;

            // scan_intra_leaf_batch: batch OCC per leaf (optimized path).
            tree.scan_intra_leaf_batch(
                RangeBound::Included(&sorted_keys[start_idx]),
                RangeBound::Unbounded,
                |_key: &[u8], v: u64| {
                    std::hint::black_box(v);
                    scanned += 1;
                    scanned < 50
                },
                &guard,
            );

            std::hint::black_box(scanned);
        }
    });
}

#[pbench::bench(threads = [8], sample_count = 200_000, sample_size = 1, skip_ext_time)]
fn scan_write_8t_skipmap(b: &Bencher<'_>) {
    let n: usize = 100_000;
    let (map, keys) = setup_skipmap(n);

    let mut sorted_keys: Vec<[u8; KEY_SIZE]> = keys.clone();
    sorted_keys.sort_unstable();

    let indices: Vec<usize> = uniform_indices(n, INDEX_POOL, 42);
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

#[pbench::bench(threads = [8], sample_count = 200_000, sample_size = 1, skip_ext_time)]
fn scan_write_8t_treeindex(b: &Bencher<'_>) {
    let n: usize = 100_000;
    let (tree, keys) = setup_tree_index(n);

    let mut sorted_keys: Vec<[u8; KEY_SIZE]> = keys.clone();
    sorted_keys.sort_unstable();

    let indices: Vec<usize> = uniform_indices(n, INDEX_POOL, 42);
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

// =============================================================================
// 10: Deep trie mixed read/write latency, 100k entries, 90/10, 8 threads
//
// Keys share 4 prefix chunks (32 bytes), forcing multi-layer trie descent.
// Writers mutate shared internal trie layers, forcing OCC retries on readers
// traversing the same path. Tail latency directly measures the cost of
// multi-layer OCC retry under contention.
// =============================================================================

const DEEP_KEY: usize = 128;
const DEEP_PREFIX_CHUNKS: usize = 4;
const DEEP_PREFIX_BUCKETS: u64 = 16;

fn setup_masstree_deep(n: usize) -> (MassTree15Inline<u64>, Vec<[u8; DEEP_KEY]>) {
    let keys: Vec<[u8; DEEP_KEY]> =
        keys_shared_prefix_chunks(n, DEEP_PREFIX_CHUNKS, DEEP_PREFIX_BUCKETS);
    let tree: MassTree15Inline<u64> = MassTree15Inline::new();

    {
        let guard: LocalGuard<'_> = tree.guard();

        for (i, key) in keys.iter().enumerate() {
            let _ = tree.insert_with_guard(key, i as u64, &guard);
        }
    }

    (tree, keys)
}

fn setup_skipmap_deep(n: usize) -> (SkipMap<[u8; DEEP_KEY], u64>, Vec<[u8; DEEP_KEY]>) {
    let keys: Vec<[u8; DEEP_KEY]> =
        keys_shared_prefix_chunks(n, DEEP_PREFIX_CHUNKS, DEEP_PREFIX_BUCKETS);
    let map: SkipMap<[u8; DEEP_KEY], u64> = SkipMap::new();

    for (i, key) in keys.iter().enumerate() {
        map.insert(*key, i as u64);
    }

    (map, keys)
}

fn setup_tree_index_deep(n: usize) -> (TreeIndex<[u8; DEEP_KEY], u64>, Vec<[u8; DEEP_KEY]>) {
    let keys: Vec<[u8; DEEP_KEY]> =
        keys_shared_prefix_chunks(n, DEEP_PREFIX_CHUNKS, DEEP_PREFIX_BUCKETS);
    let tree: TreeIndex<[u8; DEEP_KEY], u64> = TreeIndex::new();

    for (i, key) in keys.iter().enumerate() {
        let _ = tree.insert_sync(*key, i as u64);
    }

    (tree, keys)
}

fn tree_index_upsert_sync_deep(
    tree: &TreeIndex<[u8; DEEP_KEY], u64>,
    key: [u8; DEEP_KEY],
    value: u64,
) {
    let mut key: [u8; DEEP_KEY] = key;
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

#[pbench::bench(threads = [8], sample_count = 200_000, sample_size = 1, skip_ext_time)]
fn mixed_deep_90_10_8t_masstree(b: &Bencher<'_>) {
    let n: usize = 100_000;
    let (tree, keys) = setup_masstree_deep(n);
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

#[pbench::bench(threads = [8], sample_count = 200_000, sample_size = 1, skip_ext_time)]
fn mixed_deep_90_10_8t_skipmap(b: &Bencher<'_>) {
    let n: usize = 100_000;
    let (map, keys) = setup_skipmap_deep(n);
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

#[pbench::bench(threads = [8], sample_count = 200_000, sample_size = 1, skip_ext_time)]
fn mixed_deep_90_10_8t_treeindex(b: &Bencher<'_>) {
    let n: usize = 100_000;
    let (tree, keys) = setup_tree_index_deep(n);
    let indices: Vec<usize> = uniform_indices(n, INDEX_POOL, 42);
    let cursor: AtomicUsize = AtomicUsize::new(0);

    b.bench_refs(|| {
        let guard: SddGuard = SddGuard::new();
        let i: usize = cursor.fetch_add(1, Ordering::Relaxed);
        let idx: usize = indices[i % indices.len()];

        if is_write_op(i, 10) {
            tree_index_upsert_sync_deep(&tree, keys[idx], i as u64);
        } else {
            std::hint::black_box(tree.peek(&keys[idx], &guard));
        }
    });
}

fn main() {
    pbench::main();
}
