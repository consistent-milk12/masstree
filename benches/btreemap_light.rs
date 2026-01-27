//! Light benchmarks comparing `MassTree15` vs `RwLock<BTreeMap>` variants.
//!
//! All keys are 64 bytes. Compares:
//! - `MassTree15Inline`
//! - `std::sync::RwLock<BTreeMap>`
//! - `parking_lot::RwLock<BTreeMap>`
//!
//! ## Running
//!
//! ```bash
//! cargo bench --bench btreemap_light
//! cargo bench --bench btreemap_light --features mimalloc
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
use masstree::MassTree15Inline;
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
// Constants
// =============================================================================

const KEY_SIZE: usize = 64;
const WRITE_RATIO: usize = 10; // 10% writes, 90% reads
const WARMUP_OPS: usize = 250;

// =============================================================================
// Setup Helpers
// =============================================================================

fn setup_masstree15(keys: &[[u8; KEY_SIZE]]) -> MassTree15Inline<u64> {
    let tree = MassTree15Inline::new();
    {
        let guard = tree.guard();
        for (i, key) in keys.iter().enumerate() {
            let _ = tree.insert_with_guard(key, i as u64, &guard);
        }
    }
    tree
}

fn setup_std_rwlock(keys: &[[u8; KEY_SIZE]]) -> StdRwLock<BTreeMap<[u8; KEY_SIZE], u64>> {
    let mut map = BTreeMap::new();
    for (i, key) in keys.iter().enumerate() {
        map.insert(*key, i as u64);
    }
    StdRwLock::new(map)
}

fn setup_parking_rwlock(keys: &[[u8; KEY_SIZE]]) -> ParkingRwLock<BTreeMap<[u8; KEY_SIZE], u64>> {
    let mut map = BTreeMap::new();
    for (i, key) in keys.iter().enumerate() {
        map.insert(*key, i as u64);
    }
    ParkingRwLock::new(map)
}

// =============================================================================
// 01: MIXED 90-10 - Uniform Access Pattern
// =============================================================================

#[divan::bench_group(name = "01_mixed_90_10_uniform", sample_count = 20)]
mod mixed_uniform {
    use super::*;

    const N: usize = 25_000;
    const OPS_PER_THREAD: usize = 6_250;

    #[divan::bench(args = [2, 4, 6, 8, 12])]
    fn masstree15(bencher: Bencher, threads: usize) {
        let keys = Arc::new(keys::<KEY_SIZE>(N));
        let indices = Arc::new(uniform_indices(N, OPS_PER_THREAD * threads, 42));

        bencher
            .counter(divan::counter::ItemsCount::new(threads * OPS_PER_THREAD))
            .with_inputs(|| Arc::new(setup_masstree15(keys.as_ref())))
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
                            pre_measurement_barrier();
                            let mut sum = 0u64;
                            start.wait();

                            for i in 0..OPS_PER_THREAD {
                                let idx = indices[base + i];
                                if i % 100 < WRITE_RATIO {
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

    #[divan::bench(args = [2, 4, 6, 8, 12])]
    fn std_rwlock_btreemap(bencher: Bencher, threads: usize) {
        let keys = Arc::new(keys::<KEY_SIZE>(N));
        let indices = Arc::new(uniform_indices(N, OPS_PER_THREAD * threads, 42));

        bencher
            .counter(divan::counter::ItemsCount::new(threads * OPS_PER_THREAD))
            .with_inputs(|| Arc::new(setup_std_rwlock(keys.as_ref())))
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
                                let guard = map.read().unwrap();
                                black_box(guard.get(&keys[idx]));
                            }

                            warmup_done.wait();
                            pre_measurement_barrier();
                            let mut sum = 0u64;
                            start.wait();

                            for i in 0..OPS_PER_THREAD {
                                let idx = indices[base + i];
                                if i % 100 < WRITE_RATIO {
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

    #[divan::bench(args = [2, 4, 6, 8, 12])]
    fn parking_rwlock_btreemap(bencher: Bencher, threads: usize) {
        let keys = Arc::new(keys::<KEY_SIZE>(N));
        let indices = Arc::new(uniform_indices(N, OPS_PER_THREAD * threads, 42));

        bencher
            .counter(divan::counter::ItemsCount::new(threads * OPS_PER_THREAD))
            .with_inputs(|| Arc::new(setup_parking_rwlock(keys.as_ref())))
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
                                let guard = map.read();
                                black_box(guard.get(&keys[idx]));
                            }

                            warmup_done.wait();
                            pre_measurement_barrier();
                            let mut sum = 0u64;
                            start.wait();

                            for i in 0..OPS_PER_THREAD {
                                let idx = indices[base + i];
                                if i % 100 < WRITE_RATIO {
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
}

// =============================================================================
// 02: MIXED 90-10 - Zipfian Access Pattern (Hot Keys)
// =============================================================================

#[divan::bench_group(name = "02_mixed_90_10_zipfian", sample_count = 20)]
mod mixed_zipfian {
    use super::*;

    const N: usize = 25_000;
    const OPS_PER_THREAD: usize = 6_250;

    #[divan::bench(args = [2, 4, 6, 8, 12])]
    fn masstree15(bencher: Bencher, threads: usize) {
        let keys = Arc::new(keys::<KEY_SIZE>(N));
        let indices = Arc::new(zipfian_indices(N, OPS_PER_THREAD * threads, 1.0, 42));

        bencher
            .counter(divan::counter::ItemsCount::new(threads * OPS_PER_THREAD))
            .with_inputs(|| Arc::new(setup_masstree15(keys.as_ref())))
            .bench_local_values(|tree| {
                let start = Arc::new(Barrier::new(threads));
                let handles: Vec<_> = (0..threads)
                    .map(|t| {
                        let tree = Arc::clone(&tree);
                        let keys = Arc::clone(&keys);
                        let indices = Arc::clone(&indices);
                        let start = Arc::clone(&start);

                        thread::spawn(move || {
                            let guard = tree.guard();
                            let base = t * OPS_PER_THREAD;

                            pre_measurement_barrier();
                            let mut sum = 0u64;
                            start.wait();

                            for i in 0..OPS_PER_THREAD {
                                let idx = indices[base + i];
                                if i % 100 < WRITE_RATIO {
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

    #[divan::bench(args = [2, 4, 6, 8, 12])]
    fn std_rwlock_btreemap(bencher: Bencher, threads: usize) {
        let keys = Arc::new(keys::<KEY_SIZE>(N));
        let indices = Arc::new(zipfian_indices(N, OPS_PER_THREAD * threads, 1.0, 42));

        bencher
            .counter(divan::counter::ItemsCount::new(threads * OPS_PER_THREAD))
            .with_inputs(|| Arc::new(setup_std_rwlock(keys.as_ref())))
            .bench_local_values(|map| {
                let start = Arc::new(Barrier::new(threads));
                let handles: Vec<_> = (0..threads)
                    .map(|t| {
                        let map = Arc::clone(&map);
                        let keys = Arc::clone(&keys);
                        let indices = Arc::clone(&indices);
                        let start = Arc::clone(&start);

                        thread::spawn(move || {
                            let base = t * OPS_PER_THREAD;

                            pre_measurement_barrier();
                            let mut sum = 0u64;
                            start.wait();

                            for i in 0..OPS_PER_THREAD {
                                let idx = indices[base + i];
                                if i % 100 < WRITE_RATIO {
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

    #[divan::bench(args = [2, 4, 6, 8, 12])]
    fn parking_rwlock_btreemap(bencher: Bencher, threads: usize) {
        let keys = Arc::new(keys::<KEY_SIZE>(N));
        let indices = Arc::new(zipfian_indices(N, OPS_PER_THREAD * threads, 1.0, 42));

        bencher
            .counter(divan::counter::ItemsCount::new(threads * OPS_PER_THREAD))
            .with_inputs(|| Arc::new(setup_parking_rwlock(keys.as_ref())))
            .bench_local_values(|map| {
                let start = Arc::new(Barrier::new(threads));
                let handles: Vec<_> = (0..threads)
                    .map(|t| {
                        let map = Arc::clone(&map);
                        let keys = Arc::clone(&keys);
                        let indices = Arc::clone(&indices);
                        let start = Arc::clone(&start);

                        thread::spawn(move || {
                            let base = t * OPS_PER_THREAD;

                            pre_measurement_barrier();
                            let mut sum = 0u64;
                            start.wait();

                            for i in 0..OPS_PER_THREAD {
                                let idx = indices[base + i];
                                if i % 100 < WRITE_RATIO {
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
}

// =============================================================================
// 03: MIXED 90-10 - Shared Prefix Keys
// =============================================================================

#[divan::bench_group(name = "03_mixed_90_10_shared_prefix", sample_count = 20)]
mod mixed_shared_prefix {
    use super::*;

    const N: usize = 25_000;
    const OPS_PER_THREAD: usize = 6_250;

    fn prefix_keys() -> Vec<[u8; KEY_SIZE]> {
        keys_shared_prefix_chunks::<KEY_SIZE>(N, 3, 256)
    }

    #[divan::bench(args = [2, 4, 6, 8, 12])]
    fn masstree15(bencher: Bencher, threads: usize) {
        let keys = Arc::new(prefix_keys());
        let indices = Arc::new(uniform_indices(N, OPS_PER_THREAD * threads, 42));

        bencher
            .counter(divan::counter::ItemsCount::new(threads * OPS_PER_THREAD))
            .with_inputs(|| Arc::new(setup_masstree15(keys.as_ref())))
            .bench_local_values(|tree| {
                let start = Arc::new(Barrier::new(threads));
                let handles: Vec<_> = (0..threads)
                    .map(|t| {
                        let tree = Arc::clone(&tree);
                        let keys = Arc::clone(&keys);
                        let indices = Arc::clone(&indices);
                        let start = Arc::clone(&start);

                        thread::spawn(move || {
                            let guard = tree.guard();
                            let base = t * OPS_PER_THREAD;

                            pre_measurement_barrier();
                            let mut sum = 0u64;
                            start.wait();

                            for i in 0..OPS_PER_THREAD {
                                let idx = indices[base + i];
                                if i % 100 < WRITE_RATIO {
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

    #[divan::bench(args = [2, 4, 6, 8, 12])]
    fn std_rwlock_btreemap(bencher: Bencher, threads: usize) {
        let keys = Arc::new(prefix_keys());
        let indices = Arc::new(uniform_indices(N, OPS_PER_THREAD * threads, 42));

        bencher
            .counter(divan::counter::ItemsCount::new(threads * OPS_PER_THREAD))
            .with_inputs(|| Arc::new(setup_std_rwlock(keys.as_ref())))
            .bench_local_values(|map| {
                let start = Arc::new(Barrier::new(threads));
                let handles: Vec<_> = (0..threads)
                    .map(|t| {
                        let map = Arc::clone(&map);
                        let keys = Arc::clone(&keys);
                        let indices = Arc::clone(&indices);
                        let start = Arc::clone(&start);

                        thread::spawn(move || {
                            let base = t * OPS_PER_THREAD;

                            pre_measurement_barrier();
                            let mut sum = 0u64;
                            start.wait();

                            for i in 0..OPS_PER_THREAD {
                                let idx = indices[base + i];
                                if i % 100 < WRITE_RATIO {
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

    #[divan::bench(args = [2, 4, 6, 8, 12])]
    fn parking_rwlock_btreemap(bencher: Bencher, threads: usize) {
        let keys = Arc::new(prefix_keys());
        let indices = Arc::new(uniform_indices(N, OPS_PER_THREAD * threads, 42));

        bencher
            .counter(divan::counter::ItemsCount::new(threads * OPS_PER_THREAD))
            .with_inputs(|| Arc::new(setup_parking_rwlock(keys.as_ref())))
            .bench_local_values(|map| {
                let start = Arc::new(Barrier::new(threads));
                let handles: Vec<_> = (0..threads)
                    .map(|t| {
                        let map = Arc::clone(&map);
                        let keys = Arc::clone(&keys);
                        let indices = Arc::clone(&indices);
                        let start = Arc::clone(&start);

                        thread::spawn(move || {
                            let base = t * OPS_PER_THREAD;

                            pre_measurement_barrier();
                            let mut sum = 0u64;
                            start.wait();

                            for i in 0..OPS_PER_THREAD {
                                let idx = indices[base + i];
                                if i % 100 < WRITE_RATIO {
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
}

// =============================================================================
// 04: MIXED 90-10 - High Contention (Small Key Space)
// =============================================================================

#[divan::bench_group(name = "04_mixed_90_10_high_contention", sample_count = 20)]
mod mixed_high_contention {
    use super::*;

    const N: usize = 500;
    const OPS_PER_THREAD: usize = 6_250;

    #[divan::bench(args = [2, 4, 6, 8, 12])]
    fn masstree15(bencher: Bencher, threads: usize) {
        let keys = Arc::new(keys::<KEY_SIZE>(N));
        let indices = Arc::new(uniform_indices(N, OPS_PER_THREAD * threads, 42));

        bencher
            .counter(divan::counter::ItemsCount::new(threads * OPS_PER_THREAD))
            .with_inputs(|| Arc::new(setup_masstree15(keys.as_ref())))
            .bench_local_values(|tree| {
                let start = Arc::new(Barrier::new(threads));
                let handles: Vec<_> = (0..threads)
                    .map(|t| {
                        let tree = Arc::clone(&tree);
                        let keys = Arc::clone(&keys);
                        let indices = Arc::clone(&indices);
                        let start = Arc::clone(&start);

                        thread::spawn(move || {
                            let guard = tree.guard();
                            let base = t * OPS_PER_THREAD;

                            pre_measurement_barrier();
                            let mut sum = 0u64;
                            start.wait();

                            for i in 0..OPS_PER_THREAD {
                                let idx = indices[base + i];
                                if i % 100 < WRITE_RATIO {
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

    #[divan::bench(args = [2, 4, 6, 8, 12])]
    fn std_rwlock_btreemap(bencher: Bencher, threads: usize) {
        let keys = Arc::new(keys::<KEY_SIZE>(N));
        let indices = Arc::new(uniform_indices(N, OPS_PER_THREAD * threads, 42));

        bencher
            .counter(divan::counter::ItemsCount::new(threads * OPS_PER_THREAD))
            .with_inputs(|| Arc::new(setup_std_rwlock(keys.as_ref())))
            .bench_local_values(|map| {
                let start = Arc::new(Barrier::new(threads));
                let handles: Vec<_> = (0..threads)
                    .map(|t| {
                        let map = Arc::clone(&map);
                        let keys = Arc::clone(&keys);
                        let indices = Arc::clone(&indices);
                        let start = Arc::clone(&start);

                        thread::spawn(move || {
                            let base = t * OPS_PER_THREAD;

                            pre_measurement_barrier();
                            let mut sum = 0u64;
                            start.wait();

                            for i in 0..OPS_PER_THREAD {
                                let idx = indices[base + i];
                                if i % 100 < WRITE_RATIO {
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

    #[divan::bench(args = [2, 4, 6, 8, 12])]
    fn parking_rwlock_btreemap(bencher: Bencher, threads: usize) {
        let keys = Arc::new(keys::<KEY_SIZE>(N));
        let indices = Arc::new(uniform_indices(N, OPS_PER_THREAD * threads, 42));

        bencher
            .counter(divan::counter::ItemsCount::new(threads * OPS_PER_THREAD))
            .with_inputs(|| Arc::new(setup_parking_rwlock(keys.as_ref())))
            .bench_local_values(|map| {
                let start = Arc::new(Barrier::new(threads));
                let handles: Vec<_> = (0..threads)
                    .map(|t| {
                        let map = Arc::clone(&map);
                        let keys = Arc::clone(&keys);
                        let indices = Arc::clone(&indices);
                        let start = Arc::clone(&start);

                        thread::spawn(move || {
                            let base = t * OPS_PER_THREAD;

                            pre_measurement_barrier();
                            let mut sum = 0u64;
                            start.wait();

                            for i in 0..OPS_PER_THREAD {
                                let idx = indices[base + i];
                                if i % 100 < WRITE_RATIO {
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
}

// =============================================================================
// 05: MIXED 90-10 - Large Dataset
// =============================================================================

#[divan::bench_group(name = "05_mixed_90_10_large_dataset", sample_count = 20)]
mod mixed_large_dataset {
    use super::*;

    const N: usize = 500_000;
    const OPS_PER_THREAD: usize = 6_250;

    #[divan::bench(args = [2, 4, 6, 8, 12])]
    fn masstree15(bencher: Bencher, threads: usize) {
        let keys = Arc::new(keys::<KEY_SIZE>(N));
        let indices = Arc::new(uniform_indices(N, OPS_PER_THREAD * threads, 42));

        bencher
            .counter(divan::counter::ItemsCount::new(threads * OPS_PER_THREAD))
            .with_inputs(|| Arc::new(setup_masstree15(keys.as_ref())))
            .bench_local_values(|tree| {
                let start = Arc::new(Barrier::new(threads));
                let handles: Vec<_> = (0..threads)
                    .map(|t| {
                        let tree = Arc::clone(&tree);
                        let keys = Arc::clone(&keys);
                        let indices = Arc::clone(&indices);
                        let start = Arc::clone(&start);

                        thread::spawn(move || {
                            let guard = tree.guard();
                            let base = t * OPS_PER_THREAD;

                            pre_measurement_barrier();
                            let mut sum = 0u64;
                            start.wait();

                            for i in 0..OPS_PER_THREAD {
                                let idx = indices[base + i];
                                if i % 100 < WRITE_RATIO {
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

    #[divan::bench(args = [2, 4, 6, 8, 12])]
    fn std_rwlock_btreemap(bencher: Bencher, threads: usize) {
        let keys = Arc::new(keys::<KEY_SIZE>(N));
        let indices = Arc::new(uniform_indices(N, OPS_PER_THREAD * threads, 42));

        bencher
            .counter(divan::counter::ItemsCount::new(threads * OPS_PER_THREAD))
            .with_inputs(|| Arc::new(setup_std_rwlock(keys.as_ref())))
            .bench_local_values(|map| {
                let start = Arc::new(Barrier::new(threads));
                let handles: Vec<_> = (0..threads)
                    .map(|t| {
                        let map = Arc::clone(&map);
                        let keys = Arc::clone(&keys);
                        let indices = Arc::clone(&indices);
                        let start = Arc::clone(&start);

                        thread::spawn(move || {
                            let base = t * OPS_PER_THREAD;

                            pre_measurement_barrier();
                            let mut sum = 0u64;
                            start.wait();

                            for i in 0..OPS_PER_THREAD {
                                let idx = indices[base + i];
                                if i % 100 < WRITE_RATIO {
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

    #[divan::bench(args = [2, 4, 6, 8, 12])]
    fn parking_rwlock_btreemap(bencher: Bencher, threads: usize) {
        let keys = Arc::new(keys::<KEY_SIZE>(N));
        let indices = Arc::new(uniform_indices(N, OPS_PER_THREAD * threads, 42));

        bencher
            .counter(divan::counter::ItemsCount::new(threads * OPS_PER_THREAD))
            .with_inputs(|| Arc::new(setup_parking_rwlock(keys.as_ref())))
            .bench_local_values(|map| {
                let start = Arc::new(Barrier::new(threads));
                let handles: Vec<_> = (0..threads)
                    .map(|t| {
                        let map = Arc::clone(&map);
                        let keys = Arc::clone(&keys);
                        let indices = Arc::clone(&indices);
                        let start = Arc::clone(&start);

                        thread::spawn(move || {
                            let base = t * OPS_PER_THREAD;

                            pre_measurement_barrier();
                            let mut sum = 0u64;
                            start.wait();

                            for i in 0..OPS_PER_THREAD {
                                let idx = indices[base + i];
                                if i % 100 < WRITE_RATIO {
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
}

// =============================================================================
// 06: Single Hot Key
// =============================================================================

#[divan::bench_group(name = "06_single_hot_key", sample_count = 20)]
mod single_hot_key {
    use super::*;

    const N: usize = 25_000;
    const OPS_PER_THREAD: usize = 2_500;

    #[divan::bench(args = [2, 4, 6, 8, 12])]
    fn masstree15(bencher: Bencher, threads: usize) {
        let keys = Arc::new(keys::<KEY_SIZE>(N));

        bencher
            .counter(divan::counter::ItemsCount::new(threads * OPS_PER_THREAD))
            .with_inputs(|| Arc::new(setup_masstree15(keys.as_ref())))
            .bench_local_values(|tree| {
                let start = Arc::new(Barrier::new(threads));
                let handles: Vec<_> = (0..threads)
                    .map(|_| {
                        let tree = Arc::clone(&tree);
                        let keys = Arc::clone(&keys);
                        let start = Arc::clone(&start);

                        thread::spawn(move || {
                            let guard = tree.guard();
                            let hot_key = &keys[0];

                            pre_measurement_barrier();
                            let mut sum = 0u64;
                            start.wait();

                            for i in 0..OPS_PER_THREAD {
                                if i % 100 < WRITE_RATIO {
                                    let _ = tree.insert_with_guard(hot_key, i as u64, &guard);
                                } else if let Some(v) = tree.get_with_guard(hot_key, &guard) {
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

    #[divan::bench(args = [2, 4, 6, 8, 12])]
    fn std_rwlock_btreemap(bencher: Bencher, threads: usize) {
        let keys = Arc::new(keys::<KEY_SIZE>(N));

        bencher
            .counter(divan::counter::ItemsCount::new(threads * OPS_PER_THREAD))
            .with_inputs(|| Arc::new(setup_std_rwlock(keys.as_ref())))
            .bench_local_values(|map| {
                let start = Arc::new(Barrier::new(threads));
                let handles: Vec<_> = (0..threads)
                    .map(|_| {
                        let map = Arc::clone(&map);
                        let keys = Arc::clone(&keys);
                        let start = Arc::clone(&start);

                        thread::spawn(move || {
                            let hot_key = &keys[0];

                            pre_measurement_barrier();
                            let mut sum = 0u64;
                            start.wait();

                            for i in 0..OPS_PER_THREAD {
                                if i % 100 < WRITE_RATIO {
                                    let mut guard = map.write().unwrap();
                                    guard.insert(*hot_key, i as u64);
                                } else {
                                    let guard = map.read().unwrap();
                                    if let Some(v) = guard.get(hot_key) {
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

    #[divan::bench(args = [2, 4, 6, 8, 12])]
    fn parking_rwlock_btreemap(bencher: Bencher, threads: usize) {
        let keys = Arc::new(keys::<KEY_SIZE>(N));

        bencher
            .counter(divan::counter::ItemsCount::new(threads * OPS_PER_THREAD))
            .with_inputs(|| Arc::new(setup_parking_rwlock(keys.as_ref())))
            .bench_local_values(|map| {
                let start = Arc::new(Barrier::new(threads));
                let handles: Vec<_> = (0..threads)
                    .map(|_| {
                        let map = Arc::clone(&map);
                        let keys = Arc::clone(&keys);
                        let start = Arc::clone(&start);

                        thread::spawn(move || {
                            let hot_key = &keys[0];

                            pre_measurement_barrier();
                            let mut sum = 0u64;
                            start.wait();

                            for i in 0..OPS_PER_THREAD {
                                if i % 100 < WRITE_RATIO {
                                    let mut guard = map.write();
                                    guard.insert(*hot_key, i as u64);
                                } else {
                                    let guard = map.read();
                                    if let Some(v) = guard.get(hot_key) {
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

// =============================================================================
// 07: MIXED 50-50
// =============================================================================

#[divan::bench_group(name = "07_mixed_50_50", sample_count = 20)]
mod mixed_50_50 {
    use super::*;

    const N: usize = 25_000;
    const OPS_PER_THREAD: usize = 6_250;

    #[divan::bench(args = [2, 4, 6, 8, 12])]
    fn masstree15(bencher: Bencher, threads: usize) {
        let keys = Arc::new(keys::<KEY_SIZE>(N));
        let indices = Arc::new(uniform_indices(N, OPS_PER_THREAD * threads, 42));

        bencher
            .counter(divan::counter::ItemsCount::new(threads * OPS_PER_THREAD))
            .with_inputs(|| Arc::new(setup_masstree15(keys.as_ref())))
            .bench_local_values(|tree| {
                let start = Arc::new(Barrier::new(threads));
                let handles: Vec<_> = (0..threads)
                    .map(|t| {
                        let tree = Arc::clone(&tree);
                        let keys = Arc::clone(&keys);
                        let indices = Arc::clone(&indices);
                        let start = Arc::clone(&start);

                        thread::spawn(move || {
                            let guard = tree.guard();
                            let base = t * OPS_PER_THREAD;

                            pre_measurement_barrier();
                            let mut sum = 0u64;
                            start.wait();

                            for i in 0..OPS_PER_THREAD {
                                let idx = indices[base + i];
                                if i % 2 == 0 {
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

    #[divan::bench(args = [2, 4, 6, 8, 12])]
    fn std_rwlock_btreemap(bencher: Bencher, threads: usize) {
        let keys = Arc::new(keys::<KEY_SIZE>(N));
        let indices = Arc::new(uniform_indices(N, OPS_PER_THREAD * threads, 42));

        bencher
            .counter(divan::counter::ItemsCount::new(threads * OPS_PER_THREAD))
            .with_inputs(|| Arc::new(setup_std_rwlock(keys.as_ref())))
            .bench_local_values(|map| {
                let start = Arc::new(Barrier::new(threads));
                let handles: Vec<_> = (0..threads)
                    .map(|t| {
                        let map = Arc::clone(&map);
                        let keys = Arc::clone(&keys);
                        let indices = Arc::clone(&indices);
                        let start = Arc::clone(&start);

                        thread::spawn(move || {
                            let base = t * OPS_PER_THREAD;

                            pre_measurement_barrier();
                            let mut sum = 0u64;
                            start.wait();

                            for i in 0..OPS_PER_THREAD {
                                let idx = indices[base + i];
                                if i % 2 == 0 {
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

    #[divan::bench(args = [2, 4, 6, 8, 12])]
    fn parking_rwlock_btreemap(bencher: Bencher, threads: usize) {
        let keys = Arc::new(keys::<KEY_SIZE>(N));
        let indices = Arc::new(uniform_indices(N, OPS_PER_THREAD * threads, 42));

        bencher
            .counter(divan::counter::ItemsCount::new(threads * OPS_PER_THREAD))
            .with_inputs(|| Arc::new(setup_parking_rwlock(keys.as_ref())))
            .bench_local_values(|map| {
                let start = Arc::new(Barrier::new(threads));
                let handles: Vec<_> = (0..threads)
                    .map(|t| {
                        let map = Arc::clone(&map);
                        let keys = Arc::clone(&keys);
                        let indices = Arc::clone(&indices);
                        let start = Arc::clone(&start);

                        thread::spawn(move || {
                            let base = t * OPS_PER_THREAD;

                            pre_measurement_barrier();
                            let mut sum = 0u64;
                            start.wait();

                            for i in 0..OPS_PER_THREAD {
                                let idx = indices[base + i];
                                if i % 2 == 0 {
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
}

// =============================================================================
// 08: Pure Read Uniform
// =============================================================================

#[divan::bench_group(name = "08_pure_read_uniform", sample_count = 20)]
mod pure_read_uniform {
    use super::*;

    const N: usize = 25_000;
    const OPS_PER_THREAD: usize = 12_500;

    #[divan::bench(args = [2, 4, 6, 8, 12])]
    fn masstree15(bencher: Bencher, threads: usize) {
        let keys = Arc::new(keys::<KEY_SIZE>(N));
        let tree = Arc::new(setup_masstree15(keys.as_ref()));
        let indices = Arc::new(uniform_indices(N, OPS_PER_THREAD * threads, 42));

        bencher
            .counter(divan::counter::ItemsCount::new(threads * OPS_PER_THREAD))
            .bench_local(|| {
                let start = Arc::new(Barrier::new(threads));
                let handles: Vec<_> = (0..threads)
                    .map(|t| {
                        let tree = Arc::clone(&tree);
                        let keys = Arc::clone(&keys);
                        let indices = Arc::clone(&indices);
                        let start = Arc::clone(&start);

                        thread::spawn(move || {
                            let guard = tree.guard();
                            let base = t * OPS_PER_THREAD;

                            pre_measurement_barrier();
                            let mut sum = 0u64;
                            start.wait();

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

    #[divan::bench(args = [2, 4, 6, 8, 12])]
    fn std_rwlock_btreemap(bencher: Bencher, threads: usize) {
        let keys = Arc::new(keys::<KEY_SIZE>(N));
        let map = Arc::new(setup_std_rwlock(keys.as_ref()));
        let indices = Arc::new(uniform_indices(N, OPS_PER_THREAD * threads, 42));

        bencher
            .counter(divan::counter::ItemsCount::new(threads * OPS_PER_THREAD))
            .bench_local(|| {
                let start = Arc::new(Barrier::new(threads));
                let handles: Vec<_> = (0..threads)
                    .map(|t| {
                        let map = Arc::clone(&map);
                        let keys = Arc::clone(&keys);
                        let indices = Arc::clone(&indices);
                        let start = Arc::clone(&start);

                        thread::spawn(move || {
                            let base = t * OPS_PER_THREAD;

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

    #[divan::bench(args = [2, 4, 6, 8, 12])]
    fn parking_rwlock_btreemap(bencher: Bencher, threads: usize) {
        let keys = Arc::new(keys::<KEY_SIZE>(N));
        let map = Arc::new(setup_parking_rwlock(keys.as_ref()));
        let indices = Arc::new(uniform_indices(N, OPS_PER_THREAD * threads, 42));

        bencher
            .counter(divan::counter::ItemsCount::new(threads * OPS_PER_THREAD))
            .bench_local(|| {
                let start = Arc::new(Barrier::new(threads));
                let handles: Vec<_> = (0..threads)
                    .map(|t| {
                        let map = Arc::clone(&map);
                        let keys = Arc::clone(&keys);
                        let indices = Arc::clone(&indices);
                        let start = Arc::clone(&start);

                        thread::spawn(move || {
                            let base = t * OPS_PER_THREAD;

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
}

// =============================================================================
// 09: Pure Insert
// =============================================================================

#[divan::bench_group(name = "09_pure_insert", sample_count = 20)]
mod pure_insert {
    use super::*;

    const N: usize = 25_000;
    const OPS_PER_THREAD: usize = 2_500;

    #[divan::bench(args = [2, 4, 6, 8, 12])]
    fn masstree15(bencher: Bencher, threads: usize) {
        let keys = Arc::new(keys::<KEY_SIZE>(N * threads));

        bencher
            .counter(divan::counter::ItemsCount::new(threads * OPS_PER_THREAD))
            .with_inputs(|| Arc::new(MassTree15Inline::<u64>::new()))
            .bench_local_values(|tree| {
                let start = Arc::new(Barrier::new(threads));
                let handles: Vec<_> = (0..threads)
                    .map(|t| {
                        let tree = Arc::clone(&tree);
                        let keys = Arc::clone(&keys);
                        let start = Arc::clone(&start);

                        thread::spawn(move || {
                            let guard = tree.guard();
                            let base = t * OPS_PER_THREAD;

                            pre_measurement_barrier();
                            start.wait();

                            for i in 0..OPS_PER_THREAD {
                                let _ = tree.insert_with_guard(&keys[base + i], i as u64, &guard);
                            }

                            post_measurement_barrier();
                        })
                    })
                    .collect();

                for h in handles {
                    h.join().unwrap();
                }
            });
    }

    #[divan::bench(args = [2, 4, 6, 8, 12])]
    fn std_rwlock_btreemap(bencher: Bencher, threads: usize) {
        let keys = Arc::new(keys::<KEY_SIZE>(N * threads));

        bencher
            .counter(divan::counter::ItemsCount::new(threads * OPS_PER_THREAD))
            .with_inputs(|| Arc::new(StdRwLock::new(BTreeMap::<[u8; KEY_SIZE], u64>::new())))
            .bench_local_values(|map| {
                let start = Arc::new(Barrier::new(threads));
                let handles: Vec<_> = (0..threads)
                    .map(|t| {
                        let map = Arc::clone(&map);
                        let keys = Arc::clone(&keys);
                        let start = Arc::clone(&start);

                        thread::spawn(move || {
                            let base = t * OPS_PER_THREAD;

                            pre_measurement_barrier();
                            start.wait();

                            for i in 0..OPS_PER_THREAD {
                                let mut guard = map.write().unwrap();
                                guard.insert(keys[base + i], i as u64);
                            }

                            post_measurement_barrier();
                        })
                    })
                    .collect();

                for h in handles {
                    h.join().unwrap();
                }
            });
    }

    #[divan::bench(args = [2, 4, 6, 8, 12])]
    fn parking_rwlock_btreemap(bencher: Bencher, threads: usize) {
        let keys = Arc::new(keys::<KEY_SIZE>(N * threads));

        bencher
            .counter(divan::counter::ItemsCount::new(threads * OPS_PER_THREAD))
            .with_inputs(|| Arc::new(ParkingRwLock::new(BTreeMap::<[u8; KEY_SIZE], u64>::new())))
            .bench_local_values(|map| {
                let start = Arc::new(Barrier::new(threads));
                let handles: Vec<_> = (0..threads)
                    .map(|t| {
                        let map = Arc::clone(&map);
                        let keys = Arc::clone(&keys);
                        let start = Arc::clone(&start);

                        thread::spawn(move || {
                            let base = t * OPS_PER_THREAD;

                            pre_measurement_barrier();
                            start.wait();

                            for i in 0..OPS_PER_THREAD {
                                let mut guard = map.write();
                                guard.insert(keys[base + i], i as u64);
                            }

                            post_measurement_barrier();
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
// 10: Build Time (Single-Threaded)
// =============================================================================

#[divan::bench_group(name = "10_build_time", sample_count = 20)]
mod build_time {
    use super::*;

    const N: usize = 12_500;

    #[divan::bench]
    fn masstree15(bencher: Bencher) {
        let keys = keys::<KEY_SIZE>(N);

        bencher
            .counter(divan::counter::ItemsCount::new(N))
            .bench_local(|| {
                let tree = MassTree15Inline::<u64>::new();
                let guard = tree.guard();
                for (i, key) in keys.iter().enumerate() {
                    let _ = tree.insert_with_guard(key, i as u64, &guard);
                }
                drop(guard);
                black_box(tree)
            });
    }

    #[divan::bench]
    fn std_rwlock_btreemap(bencher: Bencher) {
        let keys = keys::<KEY_SIZE>(N);

        bencher
            .counter(divan::counter::ItemsCount::new(N))
            .bench_local(|| {
                let mut map = BTreeMap::new();
                for (i, key) in keys.iter().enumerate() {
                    map.insert(*key, i as u64);
                }
                black_box(map)
            });
    }
}
