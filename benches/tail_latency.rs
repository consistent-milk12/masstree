//! Tail latency benchmark: masstree vs treeindex vs indexset vs skipmap.
//!
//! Measures per-operation latency distributions with high sample counts to
//! capture p99/p99.9 tail behavior. Uses single-operation measurements (not
//! batched loops) so each sample represents one operation's latency.
//!
//! Run with: `cargo bench -p masstree --bench tail_latency`

#![expect(clippy::pedantic)]
#![expect(clippy::indexing_slicing)]

use std::sync::atomic::{AtomicUsize, Ordering};

use crossbeam_skiplist::SkipMap;
use indexset::concurrent::map::BTreeMap as IndexSetBTreeMap;
use masstree::{MassTree15Inline, RangeBound};
use pbench::Bencher;
use scc::TreeIndex;
use sdd::Guard as SddGuard;
use seize::LocalGuard;

// =============================================================================
// Constants
// =============================================================================

const KEY_SIZE: usize = 32;

/// Multipliers for deterministic key chunk generation.
const MULTIPLIERS: [u64; 4] = [
    1,
    0x517c_c1b7_2722_0a95,
    0x9e37_79b9_7f4a_7c15,
    0xbf58_476d_1ce4_e5b9,
];

// =============================================================================
// Key Generation
// =============================================================================

fn gen_keys(n: usize) -> Vec<[u8; KEY_SIZE]> {
    let mut out: Vec<[u8; KEY_SIZE]> = Vec::with_capacity(n);

    for i in 0..n {
        let mut key: [u8; KEY_SIZE] = [0u8; KEY_SIZE];

        for c in 0..4 {
            let v: u64 = (i as u64).wrapping_mul(MULTIPLIERS[c]);
            let start: usize = c * 8;
            key[start..start + 8].copy_from_slice(&v.to_be_bytes());
        }

        out.push(key);
    }

    out
}

/// Simple deterministic pseudo-random index generator.
fn random_indices(n: usize, count: usize, seed: u64) -> Vec<usize> {
    let mut state: u64 = seed;
    (0..count)
        .map(|_| {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1);
            ((state >> 32) as usize) % n
        })
        .collect()
}

// =============================================================================
// Setup Helpers
// =============================================================================

fn setup_masstree(n: usize) -> (MassTree15Inline<u64>, Vec<[u8; KEY_SIZE]>) {
    let keys: Vec<[u8; KEY_SIZE]> = gen_keys(n);
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
    let keys: Vec<[u8; KEY_SIZE]> = gen_keys(n);
    let map: SkipMap<[u8; KEY_SIZE], u64> = SkipMap::new();

    for (i, key) in keys.iter().enumerate() {
        map.insert(*key, i as u64);
    }

    (map, keys)
}

fn setup_indexset(n: usize) -> (IndexSetBTreeMap<[u8; KEY_SIZE], u64>, Vec<[u8; KEY_SIZE]>) {
    let keys: Vec<[u8; KEY_SIZE]> = gen_keys(n);
    let map: IndexSetBTreeMap<[u8; KEY_SIZE], u64> = IndexSetBTreeMap::new();

    for (i, key) in keys.iter().enumerate() {
        map.insert(*key, i as u64);
    }

    (map, keys)
}

fn setup_tree_index(n: usize) -> (TreeIndex<[u8; KEY_SIZE], u64>, Vec<[u8; KEY_SIZE]>) {
    let keys: Vec<[u8; KEY_SIZE]> = gen_keys(n);
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

// =============================================================================
// 01: Single-op GET latency — 100k entries, uniform random (1 thread)
//
// Each sample = 1 random-access lookup. High sample count captures tail.
// =============================================================================

#[pbench::bench(threads = [1], sample_count = 100_000)]
fn get_1_masstree(b: &Bencher<'_>) {
    let n: usize = 100_000;
    let (tree, keys) = setup_masstree(n);
    let indices: Vec<usize> = random_indices(n, 200_000, 42);
    let cursor: AtomicUsize = AtomicUsize::new(0);

    b.bench_refs(|| {
        let guard: LocalGuard<'_> = tree.guard();
        let i: usize = cursor.fetch_add(1, Ordering::Relaxed);
        let idx: usize = indices[i % indices.len()];
        std::hint::black_box(tree.get_with_guard(&keys[idx], &guard));
    });
}

#[pbench::bench(threads = [1], sample_count = 100_000)]
fn get_1_skipmap(b: &Bencher<'_>) {
    let n: usize = 100_000;
    let (map, keys) = setup_skipmap(n);
    let indices: Vec<usize> = random_indices(n, 200_000, 42);
    let cursor: AtomicUsize = AtomicUsize::new(0);

    b.bench_refs(|| {
        let i: usize = cursor.fetch_add(1, Ordering::Relaxed);
        let idx: usize = indices[i % indices.len()];
        std::hint::black_box(map.get(&keys[idx]));
    });
}

#[pbench::bench(threads = [1], sample_count = 100_000)]
fn get_1_indexset(b: &Bencher<'_>) {
    let n: usize = 100_000;
    let (map, keys) = setup_indexset(n);
    let indices: Vec<usize> = random_indices(n, 200_000, 42);
    let cursor: AtomicUsize = AtomicUsize::new(0);

    b.bench_refs(|| {
        let i: usize = cursor.fetch_add(1, Ordering::Relaxed);
        let idx: usize = indices[i % indices.len()];
        std::hint::black_box(map.get(&keys[idx]));
    });
}

#[pbench::bench(threads = [1], sample_count = 100_000)]
fn get_1_treeindex(b: &Bencher<'_>) {
    let n: usize = 100_000;
    let (tree, keys) = setup_tree_index(n);
    let indices: Vec<usize> = random_indices(n, 200_000, 42);
    let cursor: AtomicUsize = AtomicUsize::new(0);

    b.bench_refs(|| {
        let guard: SddGuard = SddGuard::new();
        let i: usize = cursor.fetch_add(1, Ordering::Relaxed);
        let idx: usize = indices[i % indices.len()];
        std::hint::black_box(tree.peek(&keys[idx], &guard));
    });
}

// =============================================================================
// 02: Single-op GET latency — 100k entries, uniform random (8 threads)
//
// Concurrent reads stress cache coherence and contention on shared nodes.
// Tail latency increases when multiple threads compete for the same cache lines.
// =============================================================================

#[pbench::bench(threads = [8], sample_count = 100_000)]
fn get_8t_masstree(b: &Bencher<'_>) {
    let n: usize = 100_000;
    let (tree, keys) = setup_masstree(n);
    let indices: Vec<usize> = random_indices(n, 200_000, 42);
    let cursor: AtomicUsize = AtomicUsize::new(0);

    b.bench_refs(|| {
        let guard: LocalGuard<'_> = tree.guard();
        let i: usize = cursor.fetch_add(1, Ordering::Relaxed);
        let idx: usize = indices[i % indices.len()];
        std::hint::black_box(tree.get_with_guard(&keys[idx], &guard));
    });
}

#[pbench::bench(threads = [8], sample_count = 100_000)]
fn get_8t_skipmap(b: &Bencher<'_>) {
    let n: usize = 100_000;
    let (map, keys) = setup_skipmap(n);
    let indices: Vec<usize> = random_indices(n, 200_000, 42);
    let cursor: AtomicUsize = AtomicUsize::new(0);

    b.bench_refs(|| {
        let i: usize = cursor.fetch_add(1, Ordering::Relaxed);
        let idx: usize = indices[i % indices.len()];
        std::hint::black_box(map.get(&keys[idx]));
    });
}

#[pbench::bench(threads = [8], sample_count = 100_000)]
fn get_8t_indexset(b: &Bencher<'_>) {
    let n: usize = 100_000;
    let (map, keys) = setup_indexset(n);
    let indices: Vec<usize> = random_indices(n, 200_000, 42);
    let cursor: AtomicUsize = AtomicUsize::new(0);

    b.bench_refs(|| {
        let i: usize = cursor.fetch_add(1, Ordering::Relaxed);
        let idx: usize = indices[i % indices.len()];
        std::hint::black_box(map.get(&keys[idx]));
    });
}

#[pbench::bench(threads = [8], sample_count = 100_000)]
fn get_8t_treeindex(b: &Bencher<'_>) {
    let n: usize = 100_000;
    let (tree, keys) = setup_tree_index(n);
    let indices: Vec<usize> = random_indices(n, 200_000, 42);
    let cursor: AtomicUsize = AtomicUsize::new(0);

    b.bench_refs(|| {
        let guard: SddGuard = SddGuard::new();
        let i: usize = cursor.fetch_add(1, Ordering::Relaxed);
        let idx: usize = indices[i % indices.len()];
        std::hint::black_box(tree.peek(&keys[idx], &guard));
    });
}

// =============================================================================
// 03: Single-op INSERT latency — fresh tree, 1 thread
//
// Measures insert path including splits. Each sample inserts one key into a
// growing tree, so later samples hit deeper/wider trees.
// =============================================================================

#[pbench::bench(threads = [1], sample_count = 50_000)]
fn insert_1_masstree(b: &Bencher<'_>) {
    let n: usize = 50_000;
    let keys: Vec<[u8; KEY_SIZE]> = gen_keys(n);
    let tree: MassTree15Inline<u64> = MassTree15Inline::new();
    let cursor: AtomicUsize = AtomicUsize::new(0);

    b.bench_refs(|| {
        let guard: LocalGuard<'_> = tree.guard();
        let i: usize = cursor.fetch_add(1, Ordering::Relaxed) % n;
        let _ = tree.insert_with_guard(&keys[i], i as u64, &guard);
    });
}

#[pbench::bench(threads = [1], sample_count = 50_000)]
fn insert_1_skipmap(b: &Bencher<'_>) {
    let n: usize = 50_000;
    let keys: Vec<[u8; KEY_SIZE]> = gen_keys(n);
    let map: SkipMap<[u8; KEY_SIZE], u64> = SkipMap::new();
    let cursor: AtomicUsize = AtomicUsize::new(0);

    b.bench_refs(|| {
        let i: usize = cursor.fetch_add(1, Ordering::Relaxed) % n;
        map.insert(keys[i], i as u64);
    });
}

#[pbench::bench(threads = [1], sample_count = 50_000)]
fn insert_1_indexset(b: &Bencher<'_>) {
    let n: usize = 50_000;
    let keys: Vec<[u8; KEY_SIZE]> = gen_keys(n);
    let map: IndexSetBTreeMap<[u8; KEY_SIZE], u64> = IndexSetBTreeMap::new();
    let cursor: AtomicUsize = AtomicUsize::new(0);

    b.bench_refs(|| {
        let i: usize = cursor.fetch_add(1, Ordering::Relaxed) % n;
        map.insert(keys[i], i as u64);
    });
}

#[pbench::bench(threads = [1], sample_count = 50_000)]
fn insert_1_treeindex(b: &Bencher<'_>) {
    let n: usize = 50_000;
    let keys: Vec<[u8; KEY_SIZE]> = gen_keys(n);
    let tree: TreeIndex<[u8; KEY_SIZE], u64> = TreeIndex::new();
    let cursor: AtomicUsize = AtomicUsize::new(0);

    b.bench_refs(|| {
        let i: usize = cursor.fetch_add(1, Ordering::Relaxed) % n;
        let _ = tree.insert_sync(keys[i], i as u64);
    });
}

// =============================================================================
// 04: Single-op INSERT latency — fresh tree, 8 threads
//
// Concurrent inserts cause splits and contention on internal nodes.
// Tail latency reveals retry/backoff overhead.
// =============================================================================

#[pbench::bench(threads = [8], sample_count = 50_000)]
fn insert_8t_masstree(b: &Bencher<'_>) {
    let n: usize = 100_000;
    let keys: Vec<[u8; KEY_SIZE]> = gen_keys(n);
    let tree: MassTree15Inline<u64> = MassTree15Inline::new();
    let cursor: AtomicUsize = AtomicUsize::new(0);

    b.bench_refs(|| {
        let guard: LocalGuard<'_> = tree.guard();
        let i: usize = cursor.fetch_add(1, Ordering::Relaxed) % n;
        let _ = tree.insert_with_guard(&keys[i], i as u64, &guard);
    });
}

#[pbench::bench(threads = [8], sample_count = 50_000)]
fn insert_8t_skipmap(b: &Bencher<'_>) {
    let n: usize = 100_000;
    let keys: Vec<[u8; KEY_SIZE]> = gen_keys(n);
    let map: SkipMap<[u8; KEY_SIZE], u64> = SkipMap::new();
    let cursor: AtomicUsize = AtomicUsize::new(0);

    b.bench_refs(|| {
        let i: usize = cursor.fetch_add(1, Ordering::Relaxed) % n;
        map.insert(keys[i], i as u64);
    });
}

#[pbench::bench(threads = [8], sample_count = 50_000)]
fn insert_8t_indexset(b: &Bencher<'_>) {
    let n: usize = 100_000;
    let keys: Vec<[u8; KEY_SIZE]> = gen_keys(n);
    let map: IndexSetBTreeMap<[u8; KEY_SIZE], u64> = IndexSetBTreeMap::new();
    let cursor: AtomicUsize = AtomicUsize::new(0);

    b.bench_refs(|| {
        let i: usize = cursor.fetch_add(1, Ordering::Relaxed) % n;
        map.insert(keys[i], i as u64);
    });
}

#[pbench::bench(threads = [8], sample_count = 50_000)]
fn insert_8t_treeindex(b: &Bencher<'_>) {
    let n: usize = 100_000;
    let keys: Vec<[u8; KEY_SIZE]> = gen_keys(n);
    let tree: TreeIndex<[u8; KEY_SIZE], u64> = TreeIndex::new();
    let cursor: AtomicUsize = AtomicUsize::new(0);

    b.bench_refs(|| {
        let i: usize = cursor.fetch_add(1, Ordering::Relaxed) % n;
        let _ = tree.insert_sync(keys[i], i as u64);
    });
}

// =============================================================================
// 05: Mixed read-write under contention — 90% read / 10% write, 8 threads
//
// The most realistic tail latency scenario: reads occasionally blocked by
// concurrent writers. Tail captures worst-case read latency during splits
// and lock contention.
// =============================================================================

#[pbench::bench(threads = [8], sample_count = 100_000)]
fn mixed_90_10_8t_masstree(b: &Bencher<'_>) {
    let n: usize = 100_000;
    let (tree, keys) = setup_masstree(n);
    let indices: Vec<usize> = random_indices(n, 500_000, 42);
    let cursor: AtomicUsize = AtomicUsize::new(0);

    b.bench_refs(|| {
        let guard: LocalGuard<'_> = tree.guard();
        let i: usize = cursor.fetch_add(1, Ordering::Relaxed);
        let idx: usize = indices[i % indices.len()];

        if i % 10 == 0 {
            let _ = tree.insert_with_guard(&keys[idx], i as u64, &guard);
        } else {
            std::hint::black_box(tree.get_with_guard(&keys[idx], &guard));
        }
    });
}

#[pbench::bench(threads = [8], sample_count = 100_000)]
fn mixed_90_10_8t_skipmap(b: &Bencher<'_>) {
    let n: usize = 100_000;
    let (map, keys) = setup_skipmap(n);
    let indices: Vec<usize> = random_indices(n, 500_000, 42);
    let cursor: AtomicUsize = AtomicUsize::new(0);

    b.bench_refs(|| {
        let i: usize = cursor.fetch_add(1, Ordering::Relaxed);
        let idx: usize = indices[i % indices.len()];

        if i % 10 == 0 {
            map.insert(keys[idx], i as u64);
        } else {
            std::hint::black_box(map.get(&keys[idx]));
        }
    });
}

#[pbench::bench(threads = [8], sample_count = 100_000)]
fn mixed_90_10_8t_indexset(b: &Bencher<'_>) {
    let n: usize = 100_000;
    let (map, keys) = setup_indexset(n);
    let indices: Vec<usize> = random_indices(n, 500_000, 42);
    let cursor: AtomicUsize = AtomicUsize::new(0);

    b.bench_refs(|| {
        let i: usize = cursor.fetch_add(1, Ordering::Relaxed);
        let idx: usize = indices[i % indices.len()];

        if i % 10 == 0 {
            map.insert(keys[idx], i as u64);
        } else {
            std::hint::black_box(map.get(&keys[idx]));
        }
    });
}

#[pbench::bench(threads = [8], sample_count = 100_000)]
fn mixed_90_10_8t_treeindex(b: &Bencher<'_>) {
    let n: usize = 100_000;
    let (tree, keys) = setup_tree_index(n);
    let indices: Vec<usize> = random_indices(n, 500_000, 42);
    let cursor: AtomicUsize = AtomicUsize::new(0);

    b.bench_refs(|| {
        let guard: SddGuard = SddGuard::new();
        let i: usize = cursor.fetch_add(1, Ordering::Relaxed);
        let idx: usize = indices[i % indices.len()];

        if i % 10 == 0 {
            tree_index_upsert_sync(&tree, keys[idx], i as u64);
        } else {
            std::hint::black_box(tree.peek(&keys[idx], &guard));
        }
    });
}

// =============================================================================
// 06: Mixed read-write under contention — 50% read / 50% write, 8 threads
//
// Heavy write load. Worst-case scenario for lock-based structures.
// Reveals how implementations degrade under sustained write pressure.
// =============================================================================

#[pbench::bench(threads = [8], sample_count = 100_000)]
fn mixed_50_50_8t_masstree(b: &Bencher<'_>) {
    let n: usize = 100_000;
    let (tree, keys) = setup_masstree(n);
    let indices: Vec<usize> = random_indices(n, 500_000, 55);
    let cursor: AtomicUsize = AtomicUsize::new(0);

    b.bench_refs(|| {
        let guard: LocalGuard<'_> = tree.guard();
        let i: usize = cursor.fetch_add(1, Ordering::Relaxed);
        let idx: usize = indices[i % indices.len()];

        if i % 2 == 0 {
            let _ = tree.insert_with_guard(&keys[idx], i as u64, &guard);
        } else {
            std::hint::black_box(tree.get_with_guard(&keys[idx], &guard));
        }
    });
}

#[pbench::bench(threads = [8], sample_count = 100_000)]
fn mixed_50_50_8t_skipmap(b: &Bencher<'_>) {
    let n: usize = 100_000;
    let (map, keys) = setup_skipmap(n);
    let indices: Vec<usize> = random_indices(n, 500_000, 55);
    let cursor: AtomicUsize = AtomicUsize::new(0);

    b.bench_refs(|| {
        let i: usize = cursor.fetch_add(1, Ordering::Relaxed);
        let idx: usize = indices[i % indices.len()];

        if i % 2 == 0 {
            map.insert(keys[idx], i as u64);
        } else {
            std::hint::black_box(map.get(&keys[idx]));
        }
    });
}

#[pbench::bench(threads = [8], sample_count = 100_000)]
fn mixed_50_50_8t_indexset(b: &Bencher<'_>) {
    let n: usize = 100_000;
    let (map, keys) = setup_indexset(n);
    let indices: Vec<usize> = random_indices(n, 500_000, 55);
    let cursor: AtomicUsize = AtomicUsize::new(0);

    b.bench_refs(|| {
        let i: usize = cursor.fetch_add(1, Ordering::Relaxed);
        let idx: usize = indices[i % indices.len()];

        if i % 2 == 0 {
            map.insert(keys[idx], i as u64);
        } else {
            std::hint::black_box(map.get(&keys[idx]));
        }
    });
}

#[pbench::bench(threads = [8], sample_count = 100_000)]
fn mixed_50_50_8t_treeindex(b: &Bencher<'_>) {
    let n: usize = 100_000;
    let (tree, keys) = setup_tree_index(n);
    let indices: Vec<usize> = random_indices(n, 500_000, 55);
    let cursor: AtomicUsize = AtomicUsize::new(0);

    b.bench_refs(|| {
        let guard: SddGuard = SddGuard::new();
        let i: usize = cursor.fetch_add(1, Ordering::Relaxed);
        let idx: usize = indices[i % indices.len()];

        if i % 2 == 0 {
            tree_index_upsert_sync(&tree, keys[idx], i as u64);
        } else {
            std::hint::black_box(tree.peek(&keys[idx], &guard));
        }
    });
}

// =============================================================================
// 07: Zipfian hotspot GET — 100k entries, skew=1.0, 8 threads
//
// Hot keys under Zipfian distribution cause cache-line contention.
// Tail latency here shows the cost of false sharing and coherence traffic
// on frequently-accessed nodes.
// =============================================================================

fn zipfian_indices(n: usize, count: usize, seed: u64) -> Vec<usize> {
    use rand::SeedableRng;
    use rand_chacha::ChaCha8Rng;
    use rand_distr::{Distribution, Zipf};

    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let zipf: Zipf<f64> = Zipf::new(n as f64, 1.0).expect("invalid Zipf parameters");

    (0..count)
        .map(|_| {
            let sample: f64 = zipf.sample(&mut rng);
            (sample as usize).saturating_sub(1).min(n - 1)
        })
        .collect()
}

#[pbench::bench(threads = [8], sample_count = 100_000)]
fn get_zipf_8t_masstree(b: &Bencher<'_>) {
    let n: usize = 100_000;
    let (tree, keys) = setup_masstree(n);
    let indices: Vec<usize> = zipfian_indices(n, 500_000, 42);
    let cursor: AtomicUsize = AtomicUsize::new(0);

    b.bench_refs(|| {
        let guard: LocalGuard<'_> = tree.guard();
        let i: usize = cursor.fetch_add(1, Ordering::Relaxed);
        let idx: usize = indices[i % indices.len()];
        std::hint::black_box(tree.get_with_guard(&keys[idx], &guard));
    });
}

#[pbench::bench(threads = [8], sample_count = 100_000)]
fn get_zipf_8t_skipmap(b: &Bencher<'_>) {
    let n: usize = 100_000;
    let (map, keys) = setup_skipmap(n);
    let indices: Vec<usize> = zipfian_indices(n, 500_000, 42);
    let cursor: AtomicUsize = AtomicUsize::new(0);

    b.bench_refs(|| {
        let i: usize = cursor.fetch_add(1, Ordering::Relaxed);
        let idx: usize = indices[i % indices.len()];
        std::hint::black_box(map.get(&keys[idx]));
    });
}

#[pbench::bench(threads = [8], sample_count = 100_000)]
fn get_zipf_8t_indexset(b: &Bencher<'_>) {
    let n: usize = 100_000;
    let (map, keys) = setup_indexset(n);
    let indices: Vec<usize> = zipfian_indices(n, 500_000, 42);
    let cursor: AtomicUsize = AtomicUsize::new(0);

    b.bench_refs(|| {
        let i: usize = cursor.fetch_add(1, Ordering::Relaxed);
        let idx: usize = indices[i % indices.len()];
        std::hint::black_box(map.get(&keys[idx]));
    });
}

#[pbench::bench(threads = [8], sample_count = 100_000)]
fn get_zipf_8t_treeindex(b: &Bencher<'_>) {
    let n: usize = 100_000;
    let (tree, keys) = setup_tree_index(n);
    let indices: Vec<usize> = zipfian_indices(n, 500_000, 42);
    let cursor: AtomicUsize = AtomicUsize::new(0);

    b.bench_refs(|| {
        let guard: SddGuard = SddGuard::new();
        let i: usize = cursor.fetch_add(1, Ordering::Relaxed);
        let idx: usize = indices[i % indices.len()];
        std::hint::black_box(tree.peek(&keys[idx], &guard));
    });
}

// =============================================================================
// 08: Large tree GET — 1M entries, uniform, 1 thread
//
// Large working set exceeds L2 cache, revealing true memory-access latency.
// Tail latency dominated by TLB misses and DRAM latency.
// =============================================================================

#[pbench::bench(threads = [1], sample_count = 100_000, max_time = 60)]
fn get_1m_1t_masstree(b: &Bencher<'_>) {
    let n: usize = 1_000_000;
    let (tree, keys) = setup_masstree(n);
    let indices: Vec<usize> = random_indices(n, 500_000, 42);
    let cursor: AtomicUsize = AtomicUsize::new(0);

    b.bench_refs(|| {
        let guard: LocalGuard<'_> = tree.guard();
        let i: usize = cursor.fetch_add(1, Ordering::Relaxed);
        let idx: usize = indices[i % indices.len()];
        std::hint::black_box(tree.get_with_guard(&keys[idx], &guard));
    });
}

#[pbench::bench(threads = [1], sample_count = 100_000, max_time = 60)]
fn get_1m_1t_skipmap(b: &Bencher<'_>) {
    let n: usize = 1_000_000;
    let (map, keys) = setup_skipmap(n);
    let indices: Vec<usize> = random_indices(n, 500_000, 42);
    let cursor: AtomicUsize = AtomicUsize::new(0);

    b.bench_refs(|| {
        let i: usize = cursor.fetch_add(1, Ordering::Relaxed);
        let idx: usize = indices[i % indices.len()];
        std::hint::black_box(map.get(&keys[idx]));
    });
}

#[pbench::bench(threads = [1], sample_count = 100_000, max_time = 60)]
fn get_1m_1t_indexset(b: &Bencher<'_>) {
    let n: usize = 1_000_000;
    let (map, keys) = setup_indexset(n);
    let indices: Vec<usize> = random_indices(n, 500_000, 42);
    let cursor: AtomicUsize = AtomicUsize::new(0);

    b.bench_refs(|| {
        let i: usize = cursor.fetch_add(1, Ordering::Relaxed);
        let idx: usize = indices[i % indices.len()];
        std::hint::black_box(map.get(&keys[idx]));
    });
}

#[pbench::bench(threads = [1], sample_count = 100_000, max_time = 60)]
fn get_1m_1t_treeindex(b: &Bencher<'_>) {
    let n: usize = 1_000_000;
    let (tree, keys) = setup_tree_index(n);
    let indices: Vec<usize> = random_indices(n, 500_000, 42);
    let cursor: AtomicUsize = AtomicUsize::new(0);

    b.bench_refs(|| {
        let guard: SddGuard = SddGuard::new();
        let i: usize = cursor.fetch_add(1, Ordering::Relaxed);
        let idx: usize = indices[i % indices.len()];
        std::hint::black_box(tree.peek(&keys[idx], &guard));
    });
}

// =============================================================================
// 09: Scan latency — 50-key forward scan, 100k entries, 8 threads
//
// Range scans touch multiple cache lines sequentially. Tail latency reveals
// cost of B-link pointer chasing and OCC retries under concurrent mutation.
// =============================================================================

#[pbench::bench(threads = [8], sample_count = 50_000)]
fn scan_50_8t_masstree(b: &Bencher<'_>) {
    let n: usize = 100_000;
    let (tree, keys) = setup_masstree(n);
    let start_indices: Vec<usize> = random_indices(n - 50, 200_000, 77);
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

#[pbench::bench(threads = [8], sample_count = 50_000)]
fn scan_50_8t_skipmap(b: &Bencher<'_>) {
    let n: usize = 100_000;
    let (map, keys) = setup_skipmap(n);
    // Sort keys for ordered access pattern matching skipmap's ordering
    let mut sorted_keys: Vec<[u8; KEY_SIZE]> = keys.clone();
    sorted_keys.sort_unstable();
    let start_indices: Vec<usize> = random_indices(n - 50, 200_000, 77);
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

#[pbench::bench(threads = [8], sample_count = 50_000)]
fn scan_50_8t_treeindex(b: &Bencher<'_>) {
    let n: usize = 100_000;
    let (tree, keys) = setup_tree_index(n);
    let mut sorted_keys: Vec<[u8; KEY_SIZE]> = keys.clone();
    sorted_keys.sort_unstable();
    let start_indices: Vec<usize> = random_indices(n - 50, 200_000, 77);
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

fn main() {
    pbench::main();
}
