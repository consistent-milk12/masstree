//! Comparison benchmarks: Concurrent Ordered Map Implementations
//!
//! Compares MassTree against other concurrent ordered map crates:
//! - `crossbeam-skiplist::SkipMap` - Lock-free skip list (truly concurrent)
//! - `indexset::concurrent::map::BTreeMap` - Concurrent B-tree
//!
//! NOTE: `concurrent-map::ConcurrentMap` is NOT included in multi-threaded
//! benchmarks because it's not `Sync` - each thread needs its own instance.
//!
//! Run with: `cargo bench --bench concurrent_maps`

#![expect(clippy::indexing_slicing)]

use crossbeam_skiplist::SkipMap;
use divan::{Bencher, black_box};
use indexset::concurrent::map::BTreeMap as IndexSetBTreeMap;
use masstree::MassTree;
use std::sync::Arc;
use std::thread;

fn main() {
    divan::main();
}

// =============================================================================
// Helpers
// =============================================================================

fn setup_masstree(n: usize) -> MassTree<u64> {
    let mut tree = MassTree::new();
    for i in 0..n {
        let key = (i as u64).to_be_bytes();
        let _ = tree.insert(&key, i as u64);
    }
    tree
}

fn setup_skipmap(n: usize) -> SkipMap<Vec<u8>, u64> {
    let map = SkipMap::new();
    for i in 0..n {
        let key = (i as u64).to_be_bytes().to_vec();
        map.insert(key, i as u64);
    }
    map
}

fn setup_indexset(n: usize) -> IndexSetBTreeMap<Vec<u8>, u64> {
    let map = IndexSetBTreeMap::new();
    for i in 0..n {
        let key = (i as u64).to_be_bytes().to_vec();
        map.insert(key, i as u64);
    }
    map
}

// =============================================================================
// SINGLE-THREADED: Get Performance
// =============================================================================

#[divan::bench_group(name = "01_single_get")]
mod single_get {
    use super::*;

    const SIZES: &[usize] = &[100, 1000];

    #[divan::bench(args = SIZES)]
    fn masstree(bencher: Bencher, n: usize) {
        let tree = setup_masstree(n);
        let key = ((n / 2) as u64).to_be_bytes();
        bencher.bench_local(|| tree.get(black_box(&key)));
    }

    #[divan::bench(args = SIZES)]
    fn skipmap(bencher: Bencher, n: usize) {
        let map = setup_skipmap(n);
        let key = ((n / 2) as u64).to_be_bytes().to_vec();
        bencher.bench_local(|| map.get(black_box(&key)).map(|e| *e.value()));
    }

    #[divan::bench(args = SIZES)]
    fn indexset(bencher: Bencher, n: usize) {
        let map = setup_indexset(n);
        let key = ((n / 2) as u64).to_be_bytes().to_vec();
        bencher.bench_local(|| map.get(black_box(&key)).map(|r| r.get().value));
    }
}

// =============================================================================
// SINGLE-THREADED: Insert Performance
// =============================================================================

#[divan::bench_group(name = "02_single_insert")]
mod single_insert {
    use super::*;

    const SIZES: &[usize] = &[100, 1000];

    #[divan::bench(args = SIZES)]
    fn masstree(bencher: Bencher, n: usize) {
        let new_key = (n as u64 + 1).to_be_bytes();
        bencher
            .with_inputs(|| setup_masstree(n))
            .bench_local_values(|mut tree| {
                let _ = tree.insert(black_box(&new_key), black_box(9999u64));
                tree
            });
    }

    #[divan::bench(args = SIZES)]
    fn skipmap(bencher: Bencher, n: usize) {
        let new_key = (n as u64 + 1).to_be_bytes().to_vec();
        bencher
            .with_inputs(|| setup_skipmap(n))
            .bench_local_values(|map| {
                map.insert(black_box(new_key.clone()), black_box(9999u64));
                map
            });
    }

    #[divan::bench(args = SIZES)]
    fn indexset(bencher: Bencher, n: usize) {
        let new_key = (n as u64 + 1).to_be_bytes().to_vec();
        bencher
            .with_inputs(|| setup_indexset(n))
            .bench_local_values(|map| {
                map.insert(black_box(new_key.clone()), black_box(9999u64));
                map
            });
    }
}

// =============================================================================
// MULTI-THREADED: Concurrent Reads
// =============================================================================

#[divan::bench_group(name = "03_concurrent_reads")]
mod concurrent_reads {
    use super::*;

    const OPS_PER_THREAD: usize = 1000;

    #[divan::bench(args = [1, 2, 4, 8])]
    fn masstree(bencher: Bencher, threads: usize) {
        let tree = Arc::new(setup_masstree(1000));
        let keys: Vec<[u8; 8]> = (0..100).map(|i| (i as u64 * 10).to_be_bytes()).collect();

        bencher.bench_local(|| {
            let handles: Vec<_> = (0..threads)
                .map(|_| {
                    let tree = Arc::clone(&tree);
                    let keys = keys.clone();
                    thread::spawn(move || {
                        let guard = tree.guard();
                        let mut sum = 0u64;
                        for i in 0..OPS_PER_THREAD {
                            let key = &keys[i % keys.len()];
                            if let Some(v) = tree.get_with_guard(key, &guard) {
                                sum += *v;
                            }
                        }
                        black_box(sum);
                    })
                })
                .collect();

            for h in handles {
                h.join().unwrap();
            }
        });
    }

    #[divan::bench(args = [1, 2, 4, 8])]
    fn skipmap(bencher: Bencher, threads: usize) {
        let map = Arc::new(setup_skipmap(1000));
        let keys: Vec<Vec<u8>> = (0..100)
            .map(|i| (i as u64 * 10).to_be_bytes().to_vec())
            .collect();

        bencher.bench_local(|| {
            let handles: Vec<_> = (0..threads)
                .map(|_| {
                    let map = Arc::clone(&map);
                    let keys = keys.clone();
                    thread::spawn(move || {
                        let mut sum = 0u64;
                        for i in 0..OPS_PER_THREAD {
                            let key = &keys[i % keys.len()];
                            if let Some(e) = map.get(key) {
                                sum += *e.value();
                            }
                        }
                        black_box(sum);
                    })
                })
                .collect();

            for h in handles {
                h.join().unwrap();
            }
        });
    }

    #[divan::bench(args = [1, 2, 4, 8])]
    fn indexset(bencher: Bencher, threads: usize) {
        let map = Arc::new(setup_indexset(1000));
        let keys: Vec<Vec<u8>> = (0..100)
            .map(|i| (i as u64 * 10).to_be_bytes().to_vec())
            .collect();

        bencher.bench_local(|| {
            let handles: Vec<_> = (0..threads)
                .map(|_| {
                    let map = Arc::clone(&map);
                    let keys = keys.clone();
                    thread::spawn(move || {
                        let mut sum = 0u64;
                        for i in 0..OPS_PER_THREAD {
                            let key = &keys[i % keys.len()];
                            if let Some(r) = map.get(key) {
                                sum += r.get().value;
                            }
                        }
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
// MULTI-THREADED: Concurrent Writes
// =============================================================================

#[divan::bench_group(name = "04_concurrent_writes")]
mod concurrent_writes {
    use super::*;

    const OPS_PER_THREAD: usize = 100;

    #[divan::bench(args = [1, 2, 4])]
    fn masstree(bencher: Bencher, threads: usize) {
        bencher
            .with_inputs(MassTree::<u64>::new)
            .bench_local_values(|tree| {
                let tree = Arc::new(tree);
                let handles: Vec<_> = (0..threads)
                    .map(|t| {
                        let tree = Arc::clone(&tree);
                        thread::spawn(move || {
                            let guard = tree.guard();
                            for i in 0..OPS_PER_THREAD {
                                let key = ((t * OPS_PER_THREAD + i) as u64).to_be_bytes();
                                let _ = tree.insert_with_guard(&key, i as u64, &guard);
                            }
                        })
                    })
                    .collect();

                for h in handles {
                    h.join().unwrap();
                }
                tree
            });
    }

    #[divan::bench(args = [1, 2, 4])]
    fn skipmap(bencher: Bencher, threads: usize) {
        bencher
            .with_inputs(SkipMap::<Vec<u8>, u64>::new)
            .bench_local_values(|map| {
                let map = Arc::new(map);
                let handles: Vec<_> = (0..threads)
                    .map(|t| {
                        let map = Arc::clone(&map);
                        thread::spawn(move || {
                            for i in 0..OPS_PER_THREAD {
                                let key = ((t * OPS_PER_THREAD + i) as u64).to_be_bytes().to_vec();
                                map.insert(key, i as u64);
                            }
                        })
                    })
                    .collect();

                for h in handles {
                    h.join().unwrap();
                }
                map
            });
    }

    #[divan::bench(args = [1, 2, 4])]
    fn indexset(bencher: Bencher, threads: usize) {
        bencher
            .with_inputs(IndexSetBTreeMap::<Vec<u8>, u64>::new)
            .bench_local_values(|map| {
                let map = Arc::new(map);
                let handles: Vec<_> = (0..threads)
                    .map(|t| {
                        let map = Arc::clone(&map);
                        thread::spawn(move || {
                            for i in 0..OPS_PER_THREAD {
                                let key = ((t * OPS_PER_THREAD + i) as u64).to_be_bytes().to_vec();
                                map.insert(key, i as u64);
                            }
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
// MULTI-THREADED: Mixed 90/10 Read-Heavy
// =============================================================================

#[divan::bench_group(name = "05_mixed_90_10")]
mod mixed_read_heavy {
    use super::*;

    const OPS_PER_THREAD: usize = 1000;

    #[divan::bench(args = [1, 2, 4, 8])]
    fn masstree(bencher: Bencher, threads: usize) {
        bencher
            .with_inputs(|| {
                let tree = Arc::new(setup_masstree(1000));
                let keys: Vec<[u8; 8]> = (0..100).map(|i| (i as u64 * 10).to_be_bytes()).collect();
                (tree, keys)
            })
            .bench_local_values(|(tree, keys)| {
                let handles: Vec<_> = (0..threads)
                    .map(|t| {
                        let tree = Arc::clone(&tree);
                        let keys = keys.clone();
                        thread::spawn(move || {
                            let guard = tree.guard();
                            let mut sum = 0u64;
                            let mut write_counter = 1000u64 + (t * OPS_PER_THREAD) as u64;

                            for i in 0..OPS_PER_THREAD {
                                if i % 10 == 0 {
                                    let key = write_counter.to_be_bytes();
                                    let _ = tree.insert_with_guard(&key, write_counter, &guard);
                                    write_counter += 1;
                                } else {
                                    let key = &keys[i % keys.len()];
                                    if let Some(v) = tree.get_with_guard(key, &guard) {
                                        sum += *v;
                                    }
                                }
                            }
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

    #[divan::bench(args = [1, 2, 4, 8])]
    fn skipmap(bencher: Bencher, threads: usize) {
        bencher
            .with_inputs(|| {
                let map = Arc::new(setup_skipmap(1000));
                let keys: Vec<Vec<u8>> = (0..100)
                    .map(|i| (i as u64 * 10).to_be_bytes().to_vec())
                    .collect();
                (map, keys)
            })
            .bench_local_values(|(map, keys)| {
                let handles: Vec<_> = (0..threads)
                    .map(|t| {
                        let map = Arc::clone(&map);
                        let keys = keys.clone();
                        thread::spawn(move || {
                            let mut sum = 0u64;
                            let mut write_counter = 1000u64 + (t * OPS_PER_THREAD) as u64;

                            for i in 0..OPS_PER_THREAD {
                                if i % 10 == 0 {
                                    let key = write_counter.to_be_bytes().to_vec();
                                    map.insert(key, write_counter);
                                    write_counter += 1;
                                } else {
                                    let key = &keys[i % keys.len()];
                                    if let Some(e) = map.get(key) {
                                        sum += *e.value();
                                    }
                                }
                            }
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

    #[divan::bench(args = [1, 2, 4, 8])]
    fn indexset(bencher: Bencher, threads: usize) {
        bencher
            .with_inputs(|| {
                let map = Arc::new(setup_indexset(1000));
                let keys: Vec<Vec<u8>> = (0..100)
                    .map(|i| (i as u64 * 10).to_be_bytes().to_vec())
                    .collect();
                (map, keys)
            })
            .bench_local_values(|(map, keys)| {
                let handles: Vec<_> = (0..threads)
                    .map(|t| {
                        let map = Arc::clone(&map);
                        let keys = keys.clone();
                        thread::spawn(move || {
                            let mut sum = 0u64;
                            let mut write_counter = 1000u64 + (t * OPS_PER_THREAD) as u64;

                            for i in 0..OPS_PER_THREAD {
                                if i % 10 == 0 {
                                    let key = write_counter.to_be_bytes().to_vec();
                                    map.insert(key, write_counter);
                                    write_counter += 1;
                                } else {
                                    let key = &keys[i % keys.len()];
                                    if let Some(r) = map.get(key) {
                                        sum += r.get().value;
                                    }
                                }
                            }
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
// BATCH: Sequential Insert (measures allocation/split overhead)
// =============================================================================

#[divan::bench_group(name = "06_batch_insert")]
mod batch_insert {
    use super::*;

    const SIZES: &[usize] = &[100, 500, 1000];

    #[divan::bench(args = SIZES)]
    fn masstree(bencher: Bencher, n: usize) {
        bencher
            .with_inputs(MassTree::<u64>::new)
            .bench_local_values(|mut tree| {
                for i in 0..n {
                    let key = (i as u64).to_be_bytes();
                    let _ = tree.insert(&key, i as u64);
                }
                tree
            });
    }

    #[divan::bench(args = SIZES)]
    fn skipmap(bencher: Bencher, n: usize) {
        bencher
            .with_inputs(SkipMap::<Vec<u8>, u64>::new)
            .bench_local_values(|map| {
                for i in 0..n {
                    let key = (i as u64).to_be_bytes().to_vec();
                    map.insert(key, i as u64);
                }
                map
            });
    }

    #[divan::bench(args = SIZES)]
    fn indexset(bencher: Bencher, n: usize) {
        bencher
            .with_inputs(IndexSetBTreeMap::<Vec<u8>, u64>::new)
            .bench_local_values(|map| {
                for i in 0..n {
                    let key = (i as u64).to_be_bytes().to_vec();
                    map.insert(key, i as u64);
                }
                map
            });
    }
}
