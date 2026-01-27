//! Benchmarks: `RwLock<BTreeMap>` vs `MassTree15`
//!
//! This is meant to answer: “Can `MassTree15` replace `RwLock<BTreeMap>` for
//! concurrent point ops?”
//!
//! Notes:
//! - This file benchmarks *single-operation* patterns (`get`/`insert`/range scan).
//! - `RwLock<BTreeMap>` can provide larger atomic critical sections (multi-op
//!   invariants) that a concurrent map does not replicate without an outer lock.
//!
//! Running:
//! ```bash
//! cargo bench --bench rwlock_btreemap_vs_masstree15
//! cargo bench --bench rwlock_btreemap_vs_masstree15 --features mimalloc
//! ```

#![expect(clippy::unwrap_used)]
#![expect(clippy::pedantic)]
#![expect(clippy::indexing_slicing)]

mod bench_utils;

use bench_utils::{
    keys, keys_shared_prefix_chunks, post_measurement_barrier, pre_measurement_barrier,
    uniform_indices, zipfian_indices,
};
use divan::{Bencher, black_box};
use masstree::{MassTree15, RangeBound};
use parking_lot::RwLock as ParkingRwLock;
use std::collections::BTreeMap;
use std::sync::Arc;
use std::sync::Barrier;
use std::sync::RwLock as StdRwLock;
use std::thread;

fn main() {
    divan::main();
}

// =============================================================================
// Setup helpers
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

fn setup_std_rwlock_btreemap<const K: usize>(
    keys: &[[u8; K]],
) -> StdRwLock<BTreeMap<[u8; K], u64>> {
    let mut map = BTreeMap::new();
    for (i, key) in keys.iter().enumerate() {
        map.insert(*key, i as u64);
    }
    StdRwLock::new(map)
}

fn setup_parking_rwlock_btreemap<const K: usize>(
    keys: &[[u8; K]],
) -> ParkingRwLock<BTreeMap<[u8; K], u64>> {
    let mut map = BTreeMap::new();
    for (i, key) in keys.iter().enumerate() {
        map.insert(*key, i as u64);
    }
    ParkingRwLock::new(map)
}

// =============================================================================
// Warmup iterations per thread before measurement
// =============================================================================

const WARMUP_OPS: usize = 500;

// =============================================================================
// 01: Uniform random point reads (64B keys)
// =============================================================================

#[divan::bench_group(name = "01_point_get_uniform_64B", sample_count = 200)]
mod point_get_uniform_64b {
    use super::*;

    const N: usize = 500_000;
    const OPS_PER_THREAD: usize = 50_000;

    #[divan::bench(args = [1, 2, 3, 4, 5, 6])]
    fn masstree15(bencher: Bencher, threads: usize) {
        let keys = Arc::new(keys::<64>(N));
        let tree = Arc::new(setup_masstree15::<64>(keys.as_ref()));
        let indices = Arc::new(uniform_indices(N, OPS_PER_THREAD * threads, 42));

        bencher
            .counter(divan::counter::ItemsCount::new(threads * OPS_PER_THREAD))
            .bench_local(|| {
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

                            // === WARMUP PHASE ===
                            for i in 0..WARMUP_OPS {
                                let idx = indices[base + (i % OPS_PER_THREAD)];
                                black_box(tree.get_ref(&keys[idx], &guard));
                            }
                            warmup_done.wait();

                            // === MEASUREMENT PHASE ===
                            pre_measurement_barrier();
                            let mut sum = 0u64;
                            start.wait();
                            for i in 0..OPS_PER_THREAD {
                                let idx = indices[base + i];
                                if let Some(v) = tree.get_ref(&keys[idx], &guard) {
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

    #[divan::bench(args = [1, 2, 3, 4, 5, 6])]
    fn std_rwlock_btreemap(bencher: Bencher, threads: usize) {
        let keys = Arc::new(keys::<64>(N));
        let map = Arc::new(setup_std_rwlock_btreemap::<64>(keys.as_ref()));
        let indices = Arc::new(uniform_indices(N, OPS_PER_THREAD * threads, 42));

        bencher
            .counter(divan::counter::ItemsCount::new(threads * OPS_PER_THREAD))
            .bench_local(|| {
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

                            // === WARMUP PHASE ===
                            for i in 0..WARMUP_OPS {
                                let idx = indices[base + (i % OPS_PER_THREAD)];
                                let guard = map.read().unwrap();
                                black_box(guard.get(&keys[idx]));
                            }
                            warmup_done.wait();

                            // === MEASUREMENT PHASE ===
                            pre_measurement_barrier();
                            let mut sum = 0u64;
                            start.wait();
                            for i in 0..OPS_PER_THREAD {
                                let idx = indices[base + i];
                                let guard = map.read().unwrap();
                                if let Some(v) = guard.get(&keys[idx]) {
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

    /// Batched read lock: each thread acquires the read lock once for its entire loop.
    ///
    /// This models application code that does multiple reads under a single lock scope
    /// (e.g. batch processing or request-local caching) rather than lock-per-op.
    #[divan::bench(args = [1, 2, 3, 4, 5, 6])]
    fn std_rwlock_btreemap_batched(bencher: Bencher, threads: usize) {
        let keys = Arc::new(keys::<64>(N));
        let map = Arc::new(setup_std_rwlock_btreemap::<64>(keys.as_ref()));
        let indices = Arc::new(uniform_indices(N, OPS_PER_THREAD * threads, 42));

        bencher
            .counter(divan::counter::ItemsCount::new(threads * OPS_PER_THREAD))
            .bench_local(|| {
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

                            // === WARMUP PHASE ===
                            {
                                let guard = map.read().unwrap();
                                for i in 0..WARMUP_OPS {
                                    let idx = indices[base + (i % OPS_PER_THREAD)];
                                    black_box(guard.get(&keys[idx]));
                                }
                            }
                            warmup_done.wait();

                            // === MEASUREMENT PHASE ===
                            pre_measurement_barrier();
                            let mut sum = 0u64;
                            start.wait();
                            let guard = map.read().unwrap();
                            for i in 0..OPS_PER_THREAD {
                                let idx = indices[base + i];
                                if let Some(v) = guard.get(&keys[idx]) {
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

    #[divan::bench(args = [1, 2, 3, 4, 5, 6])]
    fn parking_rwlock_btreemap(bencher: Bencher, threads: usize) {
        let keys = Arc::new(keys::<64>(N));
        let map = Arc::new(setup_parking_rwlock_btreemap::<64>(keys.as_ref()));
        let indices = Arc::new(uniform_indices(N, OPS_PER_THREAD * threads, 42));

        bencher
            .counter(divan::counter::ItemsCount::new(threads * OPS_PER_THREAD))
            .bench_local(|| {
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

                            // === WARMUP PHASE ===
                            for i in 0..WARMUP_OPS {
                                let idx = indices[base + (i % OPS_PER_THREAD)];
                                let guard = map.read();
                                black_box(guard.get(&keys[idx]));
                            }
                            warmup_done.wait();

                            // === MEASUREMENT PHASE ===
                            pre_measurement_barrier();
                            let mut sum = 0u64;
                            start.wait();
                            for i in 0..OPS_PER_THREAD {
                                let idx = indices[base + i];
                                let guard = map.read();
                                if let Some(v) = guard.get(&keys[idx]) {
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

    /// Batched read lock for `parking_lot::RwLock`.
    #[divan::bench(args = [1, 2, 3, 4, 5, 6])]
    fn parking_rwlock_btreemap_batched(bencher: Bencher, threads: usize) {
        let keys = Arc::new(keys::<64>(N));
        let map = Arc::new(setup_parking_rwlock_btreemap::<64>(keys.as_ref()));
        let indices = Arc::new(uniform_indices(N, OPS_PER_THREAD * threads, 42));

        bencher
            .counter(divan::counter::ItemsCount::new(threads * OPS_PER_THREAD))
            .bench_local(|| {
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

                            // === WARMUP PHASE ===
                            {
                                let guard = map.read();
                                for i in 0..WARMUP_OPS {
                                    let idx = indices[base + (i % OPS_PER_THREAD)];
                                    black_box(guard.get(&keys[idx]));
                                }
                            }
                            warmup_done.wait();

                            // === MEASUREMENT PHASE ===
                            pre_measurement_barrier();
                            let mut sum = 0u64;
                            start.wait();
                            let guard = map.read();
                            for i in 0..OPS_PER_THREAD {
                                let idx = indices[base + i];
                                if let Some(v) = guard.get(&keys[idx]) {
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
// 02: Zipfian point reads (64B keys) - hot key distribution
// =============================================================================

#[divan::bench_group(name = "02_point_get_zipf_64B", sample_count = 200)]
mod point_get_zipf_64b {
    use super::*;

    const N: usize = 500_000;
    const OPS_PER_THREAD: usize = 50_000;

    #[divan::bench(args = [1, 2, 3, 4, 5, 6])]
    fn masstree15(bencher: Bencher, threads: usize) {
        let keys = Arc::new(keys::<64>(N));
        let tree = Arc::new(setup_masstree15::<64>(keys.as_ref()));
        let indices = Arc::new(zipfian_indices(N, OPS_PER_THREAD * threads, 1.0, 42));

        bencher
            .counter(divan::counter::ItemsCount::new(threads * OPS_PER_THREAD))
            .bench_local(|| {
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

                            // === WARMUP PHASE ===
                            for i in 0..WARMUP_OPS {
                                let idx = indices[base + (i % OPS_PER_THREAD)];
                                black_box(tree.get_ref(&keys[idx], &guard));
                            }
                            warmup_done.wait();

                            // === MEASUREMENT PHASE ===
                            pre_measurement_barrier();
                            let mut sum = 0u64;
                            start.wait();
                            for i in 0..OPS_PER_THREAD {
                                let idx = indices[base + i];
                                if let Some(v) = tree.get_ref(&keys[idx], &guard) {
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

    #[divan::bench(args = [1, 2, 3, 4, 5, 6])]
    fn std_rwlock_btreemap(bencher: Bencher, threads: usize) {
        let keys = Arc::new(keys::<64>(N));
        let map = Arc::new(setup_std_rwlock_btreemap::<64>(keys.as_ref()));
        let indices = Arc::new(zipfian_indices(N, OPS_PER_THREAD * threads, 1.0, 42));

        bencher
            .counter(divan::counter::ItemsCount::new(threads * OPS_PER_THREAD))
            .bench_local(|| {
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

                            // === WARMUP PHASE ===
                            for i in 0..WARMUP_OPS {
                                let idx = indices[base + (i % OPS_PER_THREAD)];
                                let guard = map.read().unwrap();
                                black_box(guard.get(&keys[idx]));
                            }
                            warmup_done.wait();

                            // === MEASUREMENT PHASE ===
                            pre_measurement_barrier();
                            let mut sum = 0u64;
                            start.wait();
                            for i in 0..OPS_PER_THREAD {
                                let idx = indices[base + i];
                                let guard = map.read().unwrap();
                                if let Some(v) = guard.get(&keys[idx]) {
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

    #[divan::bench(args = [1, 2, 3, 4, 5, 6])]
    fn std_rwlock_btreemap_batched(bencher: Bencher, threads: usize) {
        let keys = Arc::new(keys::<64>(N));
        let map = Arc::new(setup_std_rwlock_btreemap::<64>(keys.as_ref()));
        let indices = Arc::new(zipfian_indices(N, OPS_PER_THREAD * threads, 1.0, 42));

        bencher
            .counter(divan::counter::ItemsCount::new(threads * OPS_PER_THREAD))
            .bench_local(|| {
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

                            // === WARMUP PHASE ===
                            {
                                let guard = map.read().unwrap();
                                for i in 0..WARMUP_OPS {
                                    let idx = indices[base + (i % OPS_PER_THREAD)];
                                    black_box(guard.get(&keys[idx]));
                                }
                            }
                            warmup_done.wait();

                            // === MEASUREMENT PHASE ===
                            pre_measurement_barrier();
                            let mut sum = 0u64;
                            start.wait();
                            let guard = map.read().unwrap();
                            for i in 0..OPS_PER_THREAD {
                                let idx = indices[base + i];
                                if let Some(v) = guard.get(&keys[idx]) {
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

    #[divan::bench(args = [1, 2, 3, 4, 5, 6])]
    fn parking_rwlock_btreemap(bencher: Bencher, threads: usize) {
        let keys = Arc::new(keys::<64>(N));
        let map = Arc::new(setup_parking_rwlock_btreemap::<64>(keys.as_ref()));
        let indices = Arc::new(zipfian_indices(N, OPS_PER_THREAD * threads, 1.0, 42));

        bencher
            .counter(divan::counter::ItemsCount::new(threads * OPS_PER_THREAD))
            .bench_local(|| {
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

                            // === WARMUP PHASE ===
                            for i in 0..WARMUP_OPS {
                                let idx = indices[base + (i % OPS_PER_THREAD)];
                                let guard = map.read();
                                black_box(guard.get(&keys[idx]));
                            }
                            warmup_done.wait();

                            // === MEASUREMENT PHASE ===
                            pre_measurement_barrier();
                            let mut sum = 0u64;
                            start.wait();
                            for i in 0..OPS_PER_THREAD {
                                let idx = indices[base + i];
                                let guard = map.read();
                                if let Some(v) = guard.get(&keys[idx]) {
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

    #[divan::bench(args = [1, 2, 3, 4, 5, 6])]
    fn parking_rwlock_btreemap_batched(bencher: Bencher, threads: usize) {
        let keys = Arc::new(keys::<64>(N));
        let map = Arc::new(setup_parking_rwlock_btreemap::<64>(keys.as_ref()));
        let indices = Arc::new(zipfian_indices(N, OPS_PER_THREAD * threads, 1.0, 42));

        bencher
            .counter(divan::counter::ItemsCount::new(threads * OPS_PER_THREAD))
            .bench_local(|| {
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

                            // === WARMUP PHASE ===
                            {
                                let guard = map.read();
                                for i in 0..WARMUP_OPS {
                                    let idx = indices[base + (i % OPS_PER_THREAD)];
                                    black_box(guard.get(&keys[idx]));
                                }
                            }
                            warmup_done.wait();

                            // === MEASUREMENT PHASE ===
                            pre_measurement_barrier();
                            let mut sum = 0u64;
                            start.wait();
                            let guard = map.read();
                            for i in 0..OPS_PER_THREAD {
                                let idx = indices[base + i];
                                if let Some(v) = guard.get(&keys[idx]) {
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
// 03: Mixed workload (90% reads / 10% writes) - uniform keys (64B)
// =============================================================================

#[divan::bench_group(name = "03_mixed_uniform_90_10_64B", sample_count = 200)]
mod mixed_uniform_90_10_64b {
    use super::*;

    const N: usize = 100_000;
    const OPS_PER_THREAD: usize = 20_000;
    const WRITE_RATIO: usize = 10;

    #[divan::bench(args = [1, 2, 3, 4, 5, 6])]
    fn masstree15(bencher: Bencher, threads: usize) {
        let keys = Arc::new(keys::<64>(N));
        let indices = Arc::new(uniform_indices(N, OPS_PER_THREAD, 42));

        bencher
            .counter(divan::counter::ItemsCount::new(threads * OPS_PER_THREAD))
            .with_inputs(|| Arc::new(setup_masstree15::<64>(keys.as_ref())))
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
                            let offset = t * 7919;

                            // === WARMUP PHASE ===
                            for i in 0..WARMUP_OPS {
                                let idx = indices[(i + offset) % indices.len()];
                                black_box(tree.get_ref(&keys[idx], &guard));
                            }
                            warmup_done.wait();

                            // === MEASUREMENT PHASE ===
                            pre_measurement_barrier();
                            let mut sum = 0u64;
                            start.wait();
                            for i in 0..OPS_PER_THREAD {
                                let idx = indices[(i + offset) % indices.len()];
                                if i % WRITE_RATIO == 0 {
                                    let _ = tree.insert_with_guard(&keys[idx], i as u64, &guard);
                                } else if let Some(v) = tree.get_ref(&keys[idx], &guard) {
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
                tree
            });
    }

    #[divan::bench(args = [1, 2, 3, 4, 5, 6])]
    fn std_rwlock_btreemap(bencher: Bencher, threads: usize) {
        let keys = Arc::new(keys::<64>(N));
        let indices = Arc::new(uniform_indices(N, OPS_PER_THREAD, 42));

        bencher
            .counter(divan::counter::ItemsCount::new(threads * OPS_PER_THREAD))
            .with_inputs(|| Arc::new(setup_std_rwlock_btreemap::<64>(keys.as_ref())))
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
                            let offset = t * 7919;

                            // === WARMUP PHASE ===
                            for i in 0..WARMUP_OPS {
                                let idx = indices[(i + offset) % indices.len()];
                                let guard = map.read().unwrap();
                                black_box(guard.get(&keys[idx]));
                            }
                            warmup_done.wait();

                            // === MEASUREMENT PHASE ===
                            pre_measurement_barrier();
                            let mut sum = 0u64;
                            start.wait();
                            for i in 0..OPS_PER_THREAD {
                                let idx = indices[(i + offset) % indices.len()];
                                if i % WRITE_RATIO == 0 {
                                    let mut guard = map.write().unwrap();
                                    guard.insert(keys[idx], i as u64);
                                } else {
                                    let guard = map.read().unwrap();
                                    if let Some(v) = guard.get(&keys[idx]) {
                                        sum = sum.wrapping_add(*v);
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
                map
            });
    }

    #[divan::bench(args = [1, 2, 3, 4, 5, 6])]
    fn parking_rwlock_btreemap(bencher: Bencher, threads: usize) {
        let keys = Arc::new(keys::<64>(N));
        let indices = Arc::new(uniform_indices(N, OPS_PER_THREAD, 42));

        bencher
            .counter(divan::counter::ItemsCount::new(threads * OPS_PER_THREAD))
            .with_inputs(|| Arc::new(setup_parking_rwlock_btreemap::<64>(keys.as_ref())))
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
                            let offset = t * 7919;

                            // === WARMUP PHASE ===
                            for i in 0..WARMUP_OPS {
                                let idx = indices[(i + offset) % indices.len()];
                                let guard = map.read();
                                black_box(guard.get(&keys[idx]));
                            }
                            warmup_done.wait();

                            // === MEASUREMENT PHASE ===
                            pre_measurement_barrier();
                            let mut sum = 0u64;
                            start.wait();
                            for i in 0..OPS_PER_THREAD {
                                let idx = indices[(i + offset) % indices.len()];
                                if i % WRITE_RATIO == 0 {
                                    let mut guard = map.write();
                                    guard.insert(keys[idx], i as u64);
                                } else {
                                    let guard = map.read();
                                    if let Some(v) = guard.get(&keys[idx]) {
                                        sum = sum.wrapping_add(*v);
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
                map
            });
    }
}

// =============================================================================
// 03b: Mixed workload with realistic contention (64B keys, Zipf hotset)
//
// Models a common production pattern:
// - Large map, but most traffic hits a hot subset of keys (Zipfian access).
// - Writes update hot keys (e.g. counters, session state, caches).
// - Reads are performed in short bursts; for `RwLock<BTreeMap>`, we model this by
//   batching the read lock across consecutive reads between writes.
//
// This aims to be more realistic than uniform distribution over the entire key
// space, while still being deterministic.
// =============================================================================

#[divan::bench_group(name = "03b_mixed_zipf_hotset_95_5_64B", sample_count = 200)]
mod mixed_zipf_hotset_95_5_64b {
    use super::*;

    const N_KEYS: usize = 200_000;
    const HOTSET: usize = 4_096;
    const OPS_PER_THREAD: usize = 50_000;
    const WRITE_EVERY: usize = 20; // 5% writes

    fn access_indices(threads: usize) -> Vec<usize> {
        // Indices into hotset [0..HOTSET), Zipfian-distributed.
        zipfian_indices(HOTSET, OPS_PER_THREAD * threads, 1.0, 42)
    }

    #[divan::bench(args = [1, 2, 3, 4, 5, 6])]
    fn masstree15(bencher: Bencher, threads: usize) {
        let keys = Arc::new(keys::<64>(N_KEYS));
        let indices = Arc::new(access_indices(threads));

        bencher
            .counter(divan::counter::ItemsCount::new(threads * OPS_PER_THREAD))
            .with_inputs(|| Arc::new(setup_masstree15::<64>(keys.as_ref())))
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

                            // === WARMUP PHASE ===
                            for i in 0..WARMUP_OPS {
                                let idx = indices[base + (i % OPS_PER_THREAD)];
                                black_box(tree.get_ref(&keys[idx], &guard));
                            }
                            warmup_done.wait();

                            // === MEASUREMENT PHASE ===
                            pre_measurement_barrier();
                            let mut sum = 0u64;
                            start.wait();
                            for i in 0..OPS_PER_THREAD {
                                let idx = indices[base + i];
                                if i % WRITE_EVERY == 0 {
                                    let _ = tree.insert_with_guard(&keys[idx], i as u64, &guard);
                                } else if let Some(v) = tree.get_ref(&keys[idx], &guard) {
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

    /// Lock-per-op baseline for `std::sync::RwLock<BTreeMap>`.
    #[divan::bench(args = [1, 2, 3, 4, 5, 6])]
    fn std_rwlock_btreemap(bencher: Bencher, threads: usize) {
        let keys = Arc::new(keys::<64>(N_KEYS));
        let indices = Arc::new(access_indices(threads));

        bencher
            .counter(divan::counter::ItemsCount::new(threads * OPS_PER_THREAD))
            .with_inputs(|| Arc::new(setup_std_rwlock_btreemap::<64>(keys.as_ref())))
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

                            // === WARMUP PHASE ===
                            for i in 0..WARMUP_OPS {
                                let idx = indices[base + (i % OPS_PER_THREAD)];
                                let guard = map.read().unwrap();
                                black_box(guard.get(&keys[idx]));
                            }
                            warmup_done.wait();

                            // === MEASUREMENT PHASE ===
                            pre_measurement_barrier();
                            let mut sum = 0u64;
                            start.wait();
                            for i in 0..OPS_PER_THREAD {
                                let idx = indices[base + i];
                                if i % WRITE_EVERY == 0 {
                                    let mut guard = map.write().unwrap();
                                    guard.insert(keys[idx], i as u64);
                                } else {
                                    let guard = map.read().unwrap();
                                    if let Some(v) = guard.get(&keys[idx]) {
                                        sum = sum.wrapping_add(*v);
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

    /// Batched read locks: acquire one read lock for the entire read-run between writes.
    #[divan::bench(args = [1, 2, 3, 4, 5, 6])]
    fn std_rwlock_btreemap_batched_reads(bencher: Bencher, threads: usize) {
        let keys = Arc::new(keys::<64>(N_KEYS));
        let indices = Arc::new(access_indices(threads));

        bencher
            .counter(divan::counter::ItemsCount::new(threads * OPS_PER_THREAD))
            .with_inputs(|| Arc::new(setup_std_rwlock_btreemap::<64>(keys.as_ref())))
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

                            // === WARMUP PHASE ===
                            {
                                let guard = map.read().unwrap();
                                for i in 0..WARMUP_OPS {
                                    let idx = indices[base + (i % OPS_PER_THREAD)];
                                    black_box(guard.get(&keys[idx]));
                                }
                            }
                            warmup_done.wait();

                            // === MEASUREMENT PHASE ===
                            pre_measurement_barrier();
                            let mut sum = 0u64;
                            start.wait();

                            let mut i = 0usize;
                            while i < OPS_PER_THREAD {
                                let idx = indices[base + i];

                                if i.is_multiple_of(WRITE_EVERY) {
                                    let mut guard = map.write().unwrap();
                                    guard.insert(keys[idx], i as u64);
                                    drop(guard);

                                    i += 1;
                                    continue;
                                }

                                let next_write = i + (WRITE_EVERY - (i % WRITE_EVERY));
                                let end = next_write.min(OPS_PER_THREAD);

                                let guard = map.read().unwrap();
                                for j in i..end {
                                    let idx = indices[base + j];
                                    if let Some(v) = guard.get(&keys[idx]) {
                                        sum = sum.wrapping_add(*v);
                                    }
                                }
                                i = end;
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

    /// Lock-per-op baseline for `parking_lot::RwLock<BTreeMap>`.
    #[divan::bench(args = [1, 2, 3, 4, 5, 6])]
    fn parking_rwlock_btreemap(bencher: Bencher, threads: usize) {
        let keys = Arc::new(keys::<64>(N_KEYS));
        let indices = Arc::new(access_indices(threads));

        bencher
            .counter(divan::counter::ItemsCount::new(threads * OPS_PER_THREAD))
            .with_inputs(|| Arc::new(setup_parking_rwlock_btreemap::<64>(keys.as_ref())))
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

                            // === WARMUP PHASE ===
                            for i in 0..WARMUP_OPS {
                                let idx = indices[base + (i % OPS_PER_THREAD)];
                                let guard = map.read();
                                black_box(guard.get(&keys[idx]));
                            }
                            warmup_done.wait();

                            // === MEASUREMENT PHASE ===
                            pre_measurement_barrier();
                            let mut sum = 0u64;
                            start.wait();
                            for i in 0..OPS_PER_THREAD {
                                let idx = indices[base + i];
                                if i % WRITE_EVERY == 0 {
                                    let mut guard = map.write();
                                    guard.insert(keys[idx], i as u64);
                                } else {
                                    let guard = map.read();
                                    if let Some(v) = guard.get(&keys[idx]) {
                                        sum = sum.wrapping_add(*v);
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

    /// Batched read locks for `parking_lot::RwLock`.
    #[divan::bench(args = [1, 2, 3, 4, 5, 6])]
    fn parking_rwlock_btreemap_batched_reads(bencher: Bencher, threads: usize) {
        let keys = Arc::new(keys::<64>(N_KEYS));
        let indices = Arc::new(access_indices(threads));

        bencher
            .counter(divan::counter::ItemsCount::new(threads * OPS_PER_THREAD))
            .with_inputs(|| Arc::new(setup_parking_rwlock_btreemap::<64>(keys.as_ref())))
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

                            // === WARMUP PHASE ===
                            {
                                let guard = map.read();
                                for i in 0..WARMUP_OPS {
                                    let idx = indices[base + (i % OPS_PER_THREAD)];
                                    black_box(guard.get(&keys[idx]));
                                }
                            }
                            warmup_done.wait();

                            // === MEASUREMENT PHASE ===
                            pre_measurement_barrier();
                            let mut sum = 0u64;
                            start.wait();

                            let mut i = 0usize;
                            while i < OPS_PER_THREAD {
                                let idx = indices[base + i];

                                if i.is_multiple_of(WRITE_EVERY) {
                                    let mut guard = map.write();
                                    guard.insert(keys[idx], i as u64);
                                    drop(guard);

                                    i += 1;
                                    continue;
                                }

                                let next_write = i + (WRITE_EVERY - (i % WRITE_EVERY));
                                let end = next_write.min(OPS_PER_THREAD);

                                let guard = map.read();
                                for j in i..end {
                                    let idx = indices[base + j];
                                    if let Some(v) = guard.get(&keys[idx]) {
                                        sum = sum.wrapping_add(*v);
                                    }
                                }
                                i = end;
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
// 04: Range scan (contiguous window) - 64B keys
// =============================================================================

#[divan::bench_group(name = "04_range_scan_window_64B", sample_count = 200)]
mod range_scan_window_64b {
    use super::*;

    const N: usize = 200_000;
    const SCAN_LEN: usize = 256;
    const SCANS_PER_THREAD: usize = 2_000;

    #[divan::bench(args = [1, 2, 3, 4, 5, 6])]
    fn masstree15(bencher: Bencher, threads: usize) {
        let keys = Arc::new(keys::<64>(N));
        let tree = Arc::new(setup_masstree15::<64>(keys.as_ref()));
        let starts = Arc::new(uniform_indices(N - SCAN_LEN, SCANS_PER_THREAD * threads, 7));

        bencher
            .counter(divan::counter::ItemsCount::new(
                threads * SCANS_PER_THREAD * SCAN_LEN,
            ))
            .bench_local(|| {
                let warmup_done = Arc::new(Barrier::new(threads));
                let start = Arc::new(Barrier::new(threads));
                let handles: Vec<_> = (0..threads)
                    .map(|t| {
                        let tree = Arc::clone(&tree);
                        let keys = Arc::clone(&keys);
                        let starts = Arc::clone(&starts);
                        let warmup_done = Arc::clone(&warmup_done);
                        let start = Arc::clone(&start);
                        thread::spawn(move || {
                            let guard = tree.guard();
                            let base = t * SCANS_PER_THREAD;

                            // === WARMUP PHASE ===
                            for i in 0..WARMUP_OPS.min(SCANS_PER_THREAD) {
                                let start_idx = starts[base + i];
                                black_box(tree.get_ref(&keys[start_idx], &guard));
                            }
                            warmup_done.wait();

                            // === MEASUREMENT PHASE ===
                            pre_measurement_barrier();
                            let mut sum = 0u64;
                            start.wait();
                            for i in 0..SCANS_PER_THREAD {
                                let start_idx = starts[base + i];
                                let end_idx = start_idx + SCAN_LEN - 1;
                                let mut seen = 0usize;
                                tree.scan_ref(
                                    RangeBound::Included(&keys[start_idx]),
                                    RangeBound::Included(&keys[end_idx]),
                                    |_, v| {
                                        sum = sum.wrapping_add(*v);
                                        seen += 1;
                                        seen < SCAN_LEN
                                    },
                                    &guard,
                                );
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

    #[divan::bench(args = [1, 2, 3, 4, 5, 6])]
    fn std_rwlock_btreemap(bencher: Bencher, threads: usize) {
        let keys = Arc::new(keys::<64>(N));
        let map = Arc::new(setup_std_rwlock_btreemap::<64>(keys.as_ref()));
        let starts = Arc::new(uniform_indices(N - SCAN_LEN, SCANS_PER_THREAD * threads, 7));

        bencher
            .counter(divan::counter::ItemsCount::new(
                threads * SCANS_PER_THREAD * SCAN_LEN,
            ))
            .bench_local(|| {
                let warmup_done = Arc::new(Barrier::new(threads));
                let start = Arc::new(Barrier::new(threads));
                let handles: Vec<_> = (0..threads)
                    .map(|t| {
                        let map = Arc::clone(&map);
                        let keys = Arc::clone(&keys);
                        let starts = Arc::clone(&starts);
                        let warmup_done = Arc::clone(&warmup_done);
                        let start = Arc::clone(&start);
                        thread::spawn(move || {
                            let base = t * SCANS_PER_THREAD;

                            // === WARMUP PHASE ===
                            for i in 0..WARMUP_OPS.min(SCANS_PER_THREAD) {
                                let start_idx = starts[base + i];
                                let guard = map.read().unwrap();
                                black_box(guard.get(&keys[start_idx]));
                            }
                            warmup_done.wait();

                            // === MEASUREMENT PHASE ===
                            pre_measurement_barrier();
                            let mut sum = 0u64;
                            start.wait();
                            for i in 0..SCANS_PER_THREAD {
                                let start_idx = starts[base + i];
                                let end_idx = start_idx + SCAN_LEN - 1;
                                let guard = map.read().unwrap();
                                for (_, v) in guard.range(keys[start_idx]..=keys[end_idx]) {
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

    #[divan::bench(args = [1, 2, 3, 4, 5, 6])]
    fn parking_rwlock_btreemap(bencher: Bencher, threads: usize) {
        let keys = Arc::new(keys::<64>(N));
        let map = Arc::new(setup_parking_rwlock_btreemap::<64>(keys.as_ref()));
        let starts = Arc::new(uniform_indices(N - SCAN_LEN, SCANS_PER_THREAD * threads, 7));

        bencher
            .counter(divan::counter::ItemsCount::new(
                threads * SCANS_PER_THREAD * SCAN_LEN,
            ))
            .bench_local(|| {
                let warmup_done = Arc::new(Barrier::new(threads));
                let start = Arc::new(Barrier::new(threads));
                let handles: Vec<_> = (0..threads)
                    .map(|t| {
                        let map = Arc::clone(&map);
                        let keys = Arc::clone(&keys);
                        let starts = Arc::clone(&starts);
                        let warmup_done = Arc::clone(&warmup_done);
                        let start = Arc::clone(&start);
                        thread::spawn(move || {
                            let base = t * SCANS_PER_THREAD;

                            // === WARMUP PHASE ===
                            for i in 0..WARMUP_OPS.min(SCANS_PER_THREAD) {
                                let start_idx = starts[base + i];
                                let guard = map.read();
                                black_box(guard.get(&keys[start_idx]));
                            }
                            warmup_done.wait();

                            // === MEASUREMENT PHASE ===
                            pre_measurement_barrier();
                            let mut sum = 0u64;
                            start.wait();
                            for i in 0..SCANS_PER_THREAD {
                                let start_idx = starts[base + i];
                                let end_idx = start_idx + SCAN_LEN - 1;
                                let guard = map.read();
                                for (_, v) in guard.range(keys[start_idx]..=keys[end_idx]) {
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
// 05: Range scan while writers update keys (64B keys)
//
// This models "read-mostly scans" in the presence of background writes.
// - `RwLock<BTreeMap>` scans will block writers while holding the read lock.
// - `MassTree15` scans should progress without global blocking.
// =============================================================================

#[divan::bench_group(name = "05_range_scan_window_with_writes_64B", sample_count = 200)]
mod range_scan_window_with_writes_64b {
    use super::*;

    const N: usize = 200_000;
    const SCAN_LEN: usize = 256;
    const SCANS_PER_THREAD: usize = 1_000;
    const WRITER_THREADS: usize = 1;
    const WRITES_PER_WRITER: usize = 50_000;

    fn writer_indices() -> Vec<usize> {
        uniform_indices(N, WRITES_PER_WRITER, 123)
    }

    #[divan::bench(args = [1, 2, 3, 4, 5, 6])]
    fn masstree15(bencher: Bencher, scan_threads: usize) {
        let keys = Arc::new(keys::<64>(N));
        let starts = Arc::new(uniform_indices(
            N - SCAN_LEN,
            SCANS_PER_THREAD * scan_threads,
            7,
        ));
        let writer_idxs = Arc::new(writer_indices());

        bencher
            .counter(divan::counter::ItemsCount::new(
                scan_threads * SCANS_PER_THREAD * SCAN_LEN,
            ))
            .with_inputs(|| Arc::new(setup_masstree15::<64>(keys.as_ref())))
            .bench_local_values(|tree| {
                let total_threads = scan_threads + WRITER_THREADS;
                let warmup_done = Arc::new(Barrier::new(total_threads));
                let start = Arc::new(Barrier::new(total_threads));

                let writers: Vec<_> = (0..WRITER_THREADS)
                    .map(|t| {
                        let tree = Arc::clone(&tree);
                        let keys = Arc::clone(&keys);
                        let writer_idxs = Arc::clone(&writer_idxs);
                        let warmup_done = Arc::clone(&warmup_done);
                        let start = Arc::clone(&start);
                        thread::spawn(move || {
                            let guard = tree.guard();
                            let offset = t * 7919;

                            // === WARMUP PHASE ===
                            for i in 0..WARMUP_OPS {
                                let idx = writer_idxs[(i + offset) % writer_idxs.len()];
                                black_box(tree.get_ref(&keys[idx], &guard));
                            }
                            warmup_done.wait();

                            // === MEASUREMENT PHASE ===
                            pre_measurement_barrier();
                            start.wait();
                            for i in 0..WRITES_PER_WRITER {
                                let idx = writer_idxs[(i + offset) % writer_idxs.len()];
                                let _ = tree.insert_with_guard(&keys[idx], i as u64, &guard);
                            }
                            post_measurement_barrier();
                        })
                    })
                    .collect();

                let scanners: Vec<_> = (0..scan_threads)
                    .map(|t| {
                        let tree = Arc::clone(&tree);
                        let keys = Arc::clone(&keys);
                        let starts = Arc::clone(&starts);
                        let warmup_done = Arc::clone(&warmup_done);
                        let start = Arc::clone(&start);
                        thread::spawn(move || {
                            let guard = tree.guard();
                            let base = t * SCANS_PER_THREAD;

                            // === WARMUP PHASE ===
                            for i in 0..WARMUP_OPS.min(SCANS_PER_THREAD) {
                                let start_idx = starts[base + i];
                                black_box(tree.get_ref(&keys[start_idx], &guard));
                            }
                            warmup_done.wait();

                            // === MEASUREMENT PHASE ===
                            pre_measurement_barrier();
                            let mut sum = 0u64;
                            start.wait();
                            for i in 0..SCANS_PER_THREAD {
                                let start_idx = starts[base + i];
                                let end_idx = start_idx + SCAN_LEN - 1;
                                let mut seen = 0usize;
                                tree.scan_ref(
                                    RangeBound::Included(&keys[start_idx]),
                                    RangeBound::Included(&keys[end_idx]),
                                    |_, v| {
                                        sum = sum.wrapping_add(*v);
                                        seen += 1;
                                        seen < SCAN_LEN
                                    },
                                    &guard,
                                );
                            }
                            post_measurement_barrier();
                            black_box(sum);
                        })
                    })
                    .collect();

                for h in writers {
                    h.join().unwrap();
                }
                for h in scanners {
                    h.join().unwrap();
                }
            });
    }

    #[divan::bench(args = [1, 2, 3, 4, 5, 6])]
    fn std_rwlock_btreemap(bencher: Bencher, scan_threads: usize) {
        let keys = Arc::new(keys::<64>(N));
        let starts = Arc::new(uniform_indices(
            N - SCAN_LEN,
            SCANS_PER_THREAD * scan_threads,
            7,
        ));
        let writer_idxs = Arc::new(writer_indices());

        bencher
            .counter(divan::counter::ItemsCount::new(
                scan_threads * SCANS_PER_THREAD * SCAN_LEN,
            ))
            .with_inputs(|| Arc::new(setup_std_rwlock_btreemap::<64>(keys.as_ref())))
            .bench_local_values(|map| {
                let total_threads = scan_threads + WRITER_THREADS;
                let warmup_done = Arc::new(Barrier::new(total_threads));
                let start = Arc::new(Barrier::new(total_threads));

                let writers: Vec<_> = (0..WRITER_THREADS)
                    .map(|t| {
                        let map = Arc::clone(&map);
                        let keys = Arc::clone(&keys);
                        let writer_idxs = Arc::clone(&writer_idxs);
                        let warmup_done = Arc::clone(&warmup_done);
                        let start = Arc::clone(&start);
                        thread::spawn(move || {
                            let offset = t * 7919;

                            // === WARMUP PHASE ===
                            for i in 0..WARMUP_OPS {
                                let idx = writer_idxs[(i + offset) % writer_idxs.len()];
                                let guard = map.read().unwrap();
                                black_box(guard.get(&keys[idx]));
                            }
                            warmup_done.wait();

                            // === MEASUREMENT PHASE ===
                            pre_measurement_barrier();
                            start.wait();
                            for i in 0..WRITES_PER_WRITER {
                                let idx = writer_idxs[(i + offset) % writer_idxs.len()];
                                let mut guard = map.write().unwrap();
                                guard.insert(keys[idx], i as u64);
                            }
                            post_measurement_barrier();
                        })
                    })
                    .collect();

                let scanners: Vec<_> = (0..scan_threads)
                    .map(|t| {
                        let map = Arc::clone(&map);
                        let keys = Arc::clone(&keys);
                        let starts = Arc::clone(&starts);
                        let warmup_done = Arc::clone(&warmup_done);
                        let start = Arc::clone(&start);
                        thread::spawn(move || {
                            let base = t * SCANS_PER_THREAD;

                            // === WARMUP PHASE ===
                            for i in 0..WARMUP_OPS.min(SCANS_PER_THREAD) {
                                let start_idx = starts[base + i];
                                let guard = map.read().unwrap();
                                black_box(guard.get(&keys[start_idx]));
                            }
                            warmup_done.wait();

                            // === MEASUREMENT PHASE ===
                            pre_measurement_barrier();
                            let mut sum = 0u64;
                            start.wait();
                            for i in 0..SCANS_PER_THREAD {
                                let start_idx = starts[base + i];
                                let end_idx = start_idx + SCAN_LEN - 1;
                                let guard = map.read().unwrap();
                                for (_, v) in guard.range(keys[start_idx]..=keys[end_idx]) {
                                    sum = sum.wrapping_add(*v);
                                }
                            }
                            post_measurement_barrier();
                            black_box(sum);
                        })
                    })
                    .collect();

                for h in writers {
                    h.join().unwrap();
                }
                for h in scanners {
                    h.join().unwrap();
                }
            });
    }

    #[divan::bench(args = [1, 2, 3, 4, 5, 6])]
    fn parking_rwlock_btreemap(bencher: Bencher, scan_threads: usize) {
        let keys = Arc::new(keys::<64>(N));
        let starts = Arc::new(uniform_indices(
            N - SCAN_LEN,
            SCANS_PER_THREAD * scan_threads,
            7,
        ));
        let writer_idxs = Arc::new(writer_indices());

        bencher
            .counter(divan::counter::ItemsCount::new(
                scan_threads * SCANS_PER_THREAD * SCAN_LEN,
            ))
            .with_inputs(|| Arc::new(setup_parking_rwlock_btreemap::<64>(keys.as_ref())))
            .bench_local_values(|map| {
                let total_threads = scan_threads + WRITER_THREADS;
                let warmup_done = Arc::new(Barrier::new(total_threads));
                let start = Arc::new(Barrier::new(total_threads));

                let writers: Vec<_> = (0..WRITER_THREADS)
                    .map(|t| {
                        let map = Arc::clone(&map);
                        let keys = Arc::clone(&keys);
                        let writer_idxs = Arc::clone(&writer_idxs);
                        let warmup_done = Arc::clone(&warmup_done);
                        let start = Arc::clone(&start);
                        thread::spawn(move || {
                            let offset = t * 7919;

                            // === WARMUP PHASE ===
                            for i in 0..WARMUP_OPS {
                                let idx = writer_idxs[(i + offset) % writer_idxs.len()];
                                let guard = map.read();
                                black_box(guard.get(&keys[idx]));
                            }
                            warmup_done.wait();

                            // === MEASUREMENT PHASE ===
                            pre_measurement_barrier();
                            start.wait();
                            for i in 0..WRITES_PER_WRITER {
                                let idx = writer_idxs[(i + offset) % writer_idxs.len()];
                                let mut guard = map.write();
                                guard.insert(keys[idx], i as u64);
                            }
                            post_measurement_barrier();
                        })
                    })
                    .collect();

                let scanners: Vec<_> = (0..scan_threads)
                    .map(|t| {
                        let map = Arc::clone(&map);
                        let keys = Arc::clone(&keys);
                        let starts = Arc::clone(&starts);
                        let warmup_done = Arc::clone(&warmup_done);
                        let start = Arc::clone(&start);
                        thread::spawn(move || {
                            let base = t * SCANS_PER_THREAD;

                            // === WARMUP PHASE ===
                            for i in 0..WARMUP_OPS.min(SCANS_PER_THREAD) {
                                let start_idx = starts[base + i];
                                let guard = map.read();
                                black_box(guard.get(&keys[start_idx]));
                            }
                            warmup_done.wait();

                            // === MEASUREMENT PHASE ===
                            pre_measurement_barrier();
                            let mut sum = 0u64;
                            start.wait();
                            for i in 0..SCANS_PER_THREAD {
                                let start_idx = starts[base + i];
                                let end_idx = start_idx + SCAN_LEN - 1;
                                let guard = map.read();
                                for (_, v) in guard.range(keys[start_idx]..=keys[end_idx]) {
                                    sum = sum.wrapping_add(*v);
                                }
                            }
                            post_measurement_barrier();
                            black_box(sum);
                        })
                    })
                    .collect();

                for h in writers {
                    h.join().unwrap();
                }
                for h in scanners {
                    h.join().unwrap();
                }
            });
    }
}

// =============================================================================
// 06: Uniform random point reads (64B keys), with aggressive shared prefixes
//
// This is the regime where MassTree's layered/trie design is expected to help.
// =============================================================================

#[divan::bench_group(name = "06_point_get_uniform_shared_prefix_64B", sample_count = 200)]
mod point_get_uniform_shared_prefix_64b {
    use super::*;

    const N: usize = 200_000;
    const OPS_PER_THREAD: usize = 50_000;
    const PREFIX_CHUNKS: usize = 3;
    const PREFIX_BUCKETS: u64 = 256;

    fn prefix_keys() -> Vec<[u8; 64]> {
        keys_shared_prefix_chunks::<64>(N, PREFIX_CHUNKS, PREFIX_BUCKETS)
    }

    #[divan::bench(args = [1, 2, 3, 4, 5, 6])]
    fn masstree15(bencher: Bencher, threads: usize) {
        let keys = Arc::new(prefix_keys());
        let tree = Arc::new(setup_masstree15::<64>(keys.as_ref()));
        let indices = Arc::new(uniform_indices(N, OPS_PER_THREAD * threads, 42));

        bencher
            .counter(divan::counter::ItemsCount::new(threads * OPS_PER_THREAD))
            .bench_local(|| {
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

                            // === WARMUP PHASE ===
                            for i in 0..WARMUP_OPS {
                                let idx = indices[base + (i % OPS_PER_THREAD)];
                                black_box(tree.get_ref(&keys[idx], &guard));
                            }
                            warmup_done.wait();

                            // === MEASUREMENT PHASE ===
                            pre_measurement_barrier();
                            let mut sum = 0u64;
                            start.wait();
                            for i in 0..OPS_PER_THREAD {
                                let idx = indices[base + i];
                                if let Some(v) = tree.get_ref(&keys[idx], &guard) {
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

    #[divan::bench(args = [1, 2, 3, 4, 5, 6])]
    fn std_rwlock_btreemap_batched(bencher: Bencher, threads: usize) {
        let keys = Arc::new(prefix_keys());
        let map = Arc::new(setup_std_rwlock_btreemap::<64>(keys.as_ref()));
        let indices = Arc::new(uniform_indices(N, OPS_PER_THREAD * threads, 42));

        bencher
            .counter(divan::counter::ItemsCount::new(threads * OPS_PER_THREAD))
            .bench_local(|| {
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

                            // === WARMUP PHASE ===
                            {
                                let guard = map.read().unwrap();
                                for i in 0..WARMUP_OPS {
                                    let idx = indices[base + (i % OPS_PER_THREAD)];
                                    black_box(guard.get(&keys[idx]));
                                }
                            }
                            warmup_done.wait();

                            // === MEASUREMENT PHASE ===
                            pre_measurement_barrier();
                            let mut sum = 0u64;
                            start.wait();
                            let guard = map.read().unwrap();
                            for i in 0..OPS_PER_THREAD {
                                let idx = indices[base + i];
                                if let Some(v) = guard.get(&keys[idx]) {
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

    #[divan::bench(args = [1, 2, 3, 4, 5, 6])]
    fn parking_rwlock_btreemap_batched(bencher: Bencher, threads: usize) {
        let keys = Arc::new(prefix_keys());
        let map = Arc::new(setup_parking_rwlock_btreemap::<64>(keys.as_ref()));
        let indices = Arc::new(uniform_indices(N, OPS_PER_THREAD * threads, 42));

        bencher
            .counter(divan::counter::ItemsCount::new(threads * OPS_PER_THREAD))
            .bench_local(|| {
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

                            // === WARMUP PHASE ===
                            {
                                let guard = map.read();
                                for i in 0..WARMUP_OPS {
                                    let idx = indices[base + (i % OPS_PER_THREAD)];
                                    black_box(guard.get(&keys[idx]));
                                }
                            }
                            warmup_done.wait();

                            // === MEASUREMENT PHASE ===
                            pre_measurement_barrier();
                            let mut sum = 0u64;
                            start.wait();
                            let guard = map.read();
                            for i in 0..OPS_PER_THREAD {
                                let idx = indices[base + i];
                                if let Some(v) = guard.get(&keys[idx]) {
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

#[divan::bench_group(name = "07_mixed_90_10_shared_prefix_64B", sample_count = 200)]
mod mixed_90_10_shared_prefix_64b {
    use super::*;

    const N: usize = 200_000;
    const OPS_PER_THREAD: usize = 50_000;
    const PREFIX_CHUNKS: usize = 3;
    const PREFIX_BUCKETS: u64 = 256;
    const WRITE_RATIO: usize = 10; // 10% writes

    fn prefix_keys() -> Vec<[u8; 64]> {
        keys_shared_prefix_chunks::<64>(N, PREFIX_CHUNKS, PREFIX_BUCKETS)
    }

    #[divan::bench(args = [1, 2, 3, 4, 5, 6])]
    fn masstree15(bencher: Bencher, threads: usize) {
        let keys = Arc::new(prefix_keys());
        let indices = Arc::new(uniform_indices(N, OPS_PER_THREAD * threads, 42));

        bencher
            .counter(divan::counter::ItemsCount::new(threads * OPS_PER_THREAD))
            .with_inputs(|| Arc::new(setup_masstree15::<64>(keys.as_ref())))
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

                            // === WARMUP PHASE ===
                            for i in 0..WARMUP_OPS {
                                let idx = indices[base + (i % OPS_PER_THREAD)];
                                black_box(tree.get_ref(&keys[idx], &guard));
                            }
                            warmup_done.wait();

                            // === MEASUREMENT PHASE ===
                            pre_measurement_barrier();
                            let mut sum = 0u64;
                            start.wait();
                            for i in 0..OPS_PER_THREAD {
                                let idx = indices[base + i];
                                if i % 100 < WRITE_RATIO {
                                    // 10% writes
                                    let _ = tree.insert_with_guard(&keys[idx], idx as u64, &guard);
                                } else {
                                    // 90% reads
                                    if let Some(v) = tree.get_ref(&keys[idx], &guard) {
                                        sum = sum.wrapping_add(*v);
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

    #[divan::bench(args = [1, 2, 3, 4, 5, 6])]
    fn std_rwlock_btreemap(bencher: Bencher, threads: usize) {
        let keys = Arc::new(prefix_keys());
        let indices = Arc::new(uniform_indices(N, OPS_PER_THREAD * threads, 42));

        bencher
            .counter(divan::counter::ItemsCount::new(threads * OPS_PER_THREAD))
            .with_inputs(|| Arc::new(setup_std_rwlock_btreemap::<64>(keys.as_ref())))
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

                            // === WARMUP PHASE ===
                            for i in 0..WARMUP_OPS {
                                let idx = indices[base + (i % OPS_PER_THREAD)];
                                let guard = map.read().unwrap();
                                black_box(guard.get(&keys[idx]));
                            }
                            warmup_done.wait();

                            // === MEASUREMENT PHASE ===
                            pre_measurement_barrier();
                            let mut sum = 0u64;
                            start.wait();
                            for i in 0..OPS_PER_THREAD {
                                let idx = indices[base + i];
                                if i % 100 < WRITE_RATIO {
                                    // 10% writes - need write lock
                                    let mut guard = map.write().unwrap();
                                    guard.insert(keys[idx], idx as u64);
                                } else {
                                    // 90% reads - read lock
                                    let guard = map.read().unwrap();
                                    if let Some(v) = guard.get(&keys[idx]) {
                                        sum = sum.wrapping_add(*v);
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

    #[divan::bench(args = [1, 2, 3, 4, 5, 6])]
    fn parking_rwlock_btreemap(bencher: Bencher, threads: usize) {
        let keys = Arc::new(prefix_keys());
        let indices = Arc::new(uniform_indices(N, OPS_PER_THREAD * threads, 42));

        bencher
            .counter(divan::counter::ItemsCount::new(threads * OPS_PER_THREAD))
            .with_inputs(|| Arc::new(setup_parking_rwlock_btreemap::<64>(keys.as_ref())))
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

                            // === WARMUP PHASE ===
                            for i in 0..WARMUP_OPS {
                                let idx = indices[base + (i % OPS_PER_THREAD)];
                                let guard = map.read();
                                black_box(guard.get(&keys[idx]));
                            }
                            warmup_done.wait();

                            // === MEASUREMENT PHASE ===
                            pre_measurement_barrier();
                            let mut sum = 0u64;
                            start.wait();
                            for i in 0..OPS_PER_THREAD {
                                let idx = indices[base + i];
                                if i % 100 < WRITE_RATIO {
                                    // 10% writes - need write lock
                                    let mut guard = map.write();
                                    guard.insert(keys[idx], idx as u64);
                                } else {
                                    // 90% reads - read lock
                                    let guard = map.read();
                                    if let Some(v) = guard.get(&keys[idx]) {
                                        sum = sum.wrapping_add(*v);
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
