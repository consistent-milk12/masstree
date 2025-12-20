//! Comparison benchmarks: `MassTree` vs `Mutex<BTreeMap>` vs `RwLock<BTreeMap>`
//!
//! This is the fair comparison for concurrent use cases. MassTree is designed
//! to replace lock-wrapped `BTreeMap`, not bare `BTreeMap`.
//!
//! **Methodology:**
//! - Single-threaded: measures lock overhead
//! - Multi-threaded: measures contention and scaling
//! - Read-heavy workloads: MassTree's sweet spot (lock-free reads)
//! - Write-heavy workloads: where Mutex may win
//!
//! **Why both Mutex and RwLock?**
//! - `Mutex` has simpler state (locked/unlocked) and lower per-operation overhead
//! - `RwLock` allows concurrent readers but has more complex atomic operations
//! - For many workloads, `Mutex` outperforms `RwLock` due to lower overhead
//! - The crossover where `RwLock` wins requires many concurrent readers
//!
//! Run with: `cargo bench --bench lock_comparison`

#![expect(clippy::indexing_slicing)]

use divan::{black_box, Bencher};
use masstree::MassTree;
use std::collections::BTreeMap;
use std::sync::{Arc, Mutex, RwLock};
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

fn setup_mutex_btreemap(n: usize) -> Mutex<BTreeMap<Vec<u8>, u64>> {
    let mut tree = BTreeMap::new();
    for i in 0..n {
        let key = (i as u64).to_be_bytes().to_vec();
        tree.insert(key, i as u64);
    }
    Mutex::new(tree)
}

fn setup_rwlock_btreemap(n: usize) -> RwLock<BTreeMap<Vec<u8>, u64>> {
    let mut tree = BTreeMap::new();
    for i in 0..n {
        let key = (i as u64).to_be_bytes().to_vec();
        tree.insert(key, i as u64);
    }
    RwLock::new(tree)
}

// =============================================================================
// SINGLE-THREADED: Lock Overhead
// =============================================================================

#[divan::bench_group(name = "01_single_thread_get")]
mod single_thread_get {
    use super::*;

    const SIZES: &[usize] = &[100, 1000];

    #[divan::bench(args = SIZES)]
    fn masstree(bencher: Bencher, n: usize) {
        let tree = setup_masstree(n);
        let key = ((n / 2) as u64).to_be_bytes();
        bencher.bench_local(|| tree.get(black_box(&key)));
    }

    #[divan::bench(args = SIZES)]
    fn mutex_btreemap(bencher: Bencher, n: usize) {
        let tree = setup_mutex_btreemap(n);
        let key = ((n / 2) as u64).to_be_bytes().to_vec();
        bencher.bench_local(|| {
            let guard = tree.lock().unwrap();
            guard.get(black_box(&key)).copied()
        });
    }

    #[divan::bench(args = SIZES)]
    fn rwlock_btreemap(bencher: Bencher, n: usize) {
        let tree = setup_rwlock_btreemap(n);
        let key = ((n / 2) as u64).to_be_bytes().to_vec();
        bencher.bench_local(|| {
            let guard = tree.read().unwrap();
            guard.get(black_box(&key)).copied()
        });
    }
}

#[divan::bench_group(name = "02_single_thread_insert")]
mod single_thread_insert {
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
    fn mutex_btreemap(bencher: Bencher, n: usize) {
        let new_key = (n as u64 + 1).to_be_bytes().to_vec();
        bencher
            .with_inputs(|| setup_mutex_btreemap(n))
            .bench_local_values(|tree| {
                let mut guard = tree.lock().unwrap();
                guard.insert(black_box(new_key.clone()), black_box(9999u64));
                drop(guard);
                tree
            });
    }

    #[divan::bench(args = SIZES)]
    fn rwlock_btreemap(bencher: Bencher, n: usize) {
        let new_key = (n as u64 + 1).to_be_bytes().to_vec();
        bencher
            .with_inputs(|| setup_rwlock_btreemap(n))
            .bench_local_values(|tree| {
                let mut guard = tree.write().unwrap();
                guard.insert(black_box(new_key.clone()), black_box(9999u64));
                drop(guard);
                tree
            });
    }
}

// =============================================================================
// MULTI-THREADED: Read-Only Workload
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
                .map(|_t| {
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
    fn mutex_btreemap(bencher: Bencher, threads: usize) {
        let tree = Arc::new(setup_mutex_btreemap(1000));
        let keys: Vec<Vec<u8>> = (0..100)
            .map(|i| (i as u64 * 10).to_be_bytes().to_vec())
            .collect();

        bencher.bench_local(|| {
            let handles: Vec<_> = (0..threads)
                .map(|_t| {
                    let tree = Arc::clone(&tree);
                    let keys = keys.clone();
                    thread::spawn(move || {
                        let mut sum = 0u64;
                        for i in 0..OPS_PER_THREAD {
                            let key = &keys[i % keys.len()];
                            let guard = tree.lock().unwrap();
                            if let Some(&v) = guard.get(key) {
                                sum += v;
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
    fn rwlock_btreemap(bencher: Bencher, threads: usize) {
        let tree = Arc::new(setup_rwlock_btreemap(1000));
        let keys: Vec<Vec<u8>> = (0..100)
            .map(|i| (i as u64 * 10).to_be_bytes().to_vec())
            .collect();

        bencher.bench_local(|| {
            let handles: Vec<_> = (0..threads)
                .map(|_t| {
                    let tree = Arc::clone(&tree);
                    let keys = keys.clone();
                    thread::spawn(move || {
                        let mut sum = 0u64;
                        for i in 0..OPS_PER_THREAD {
                            let key = &keys[i % keys.len()];
                            let guard = tree.read().unwrap();
                            if let Some(&v) = guard.get(key) {
                                sum += v;
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
// MULTI-THREADED: Write-Only Workload
// =============================================================================

#[divan::bench_group(name = "04_concurrent_writes")]
mod concurrent_writes {
    use super::*;

    const OPS_PER_THREAD: usize = 100;

    #[divan::bench(args = [1, 2, 4])]
    fn masstree(bencher: Bencher, threads: usize) {
        bencher
            .with_inputs(|| Arc::new(MassTree::<u64>::new()))
            .bench_local_values(|tree| {
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
    fn mutex_btreemap(bencher: Bencher, threads: usize) {
        bencher
            .with_inputs(|| Arc::new(Mutex::new(BTreeMap::<Vec<u8>, u64>::new())))
            .bench_local_values(|tree| {
                let handles: Vec<_> = (0..threads)
                    .map(|t| {
                        let tree = Arc::clone(&tree);
                        thread::spawn(move || {
                            for i in 0..OPS_PER_THREAD {
                                let key = ((t * OPS_PER_THREAD + i) as u64).to_be_bytes().to_vec();
                                let mut guard = tree.lock().unwrap();
                                guard.insert(key, i as u64);
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
    fn rwlock_btreemap(bencher: Bencher, threads: usize) {
        bencher
            .with_inputs(|| Arc::new(RwLock::new(BTreeMap::<Vec<u8>, u64>::new())))
            .bench_local_values(|tree| {
                let handles: Vec<_> = (0..threads)
                    .map(|t| {
                        let tree = Arc::clone(&tree);
                        thread::spawn(move || {
                            for i in 0..OPS_PER_THREAD {
                                let key = ((t * OPS_PER_THREAD + i) as u64).to_be_bytes().to_vec();
                                let mut guard = tree.write().unwrap();
                                guard.insert(key, i as u64);
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
}

// =============================================================================
// MULTI-THREADED: Mixed 90/10 Read-Heavy
// =============================================================================

#[divan::bench_group(name = "05_mixed_90_10_read_heavy")]
mod mixed_read_heavy {
    use super::*;

    const OPS_PER_THREAD: usize = 1000;

    #[divan::bench(args = [1, 2, 4, 8])]
    fn masstree(bencher: Bencher, threads: usize) {
        bencher
            .with_inputs(|| {
                let tree = Arc::new(setup_masstree(1000));
                let keys: Vec<[u8; 8]> =
                    (0..100).map(|i| (i as u64 * 10).to_be_bytes()).collect();
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
                                    // 10% writes
                                    let key = write_counter.to_be_bytes();
                                    let _ = tree.insert_with_guard(&key, write_counter, &guard);
                                    write_counter += 1;
                                } else {
                                    // 90% reads
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
    fn mutex_btreemap(bencher: Bencher, threads: usize) {
        bencher
            .with_inputs(|| {
                let tree = Arc::new(setup_mutex_btreemap(1000));
                let keys: Vec<Vec<u8>> = (0..100)
                    .map(|i| (i as u64 * 10).to_be_bytes().to_vec())
                    .collect();
                (tree, keys)
            })
            .bench_local_values(|(tree, keys)| {
                let handles: Vec<_> = (0..threads)
                    .map(|t| {
                        let tree = Arc::clone(&tree);
                        let keys = keys.clone();
                        thread::spawn(move || {
                            let mut sum = 0u64;
                            let mut write_counter = 1000u64 + (t * OPS_PER_THREAD) as u64;

                            for i in 0..OPS_PER_THREAD {
                                if i % 10 == 0 {
                                    // 10% writes
                                    let key = write_counter.to_be_bytes().to_vec();
                                    let mut guard = tree.lock().unwrap();
                                    guard.insert(key, write_counter);
                                    write_counter += 1;
                                } else {
                                    // 90% reads
                                    let key = &keys[i % keys.len()];
                                    let guard = tree.lock().unwrap();
                                    if let Some(&v) = guard.get(key) {
                                        sum += v;
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
    fn rwlock_btreemap(bencher: Bencher, threads: usize) {
        bencher
            .with_inputs(|| {
                let tree = Arc::new(setup_rwlock_btreemap(1000));
                let keys: Vec<Vec<u8>> = (0..100)
                    .map(|i| (i as u64 * 10).to_be_bytes().to_vec())
                    .collect();
                (tree, keys)
            })
            .bench_local_values(|(tree, keys)| {
                let handles: Vec<_> = (0..threads)
                    .map(|t| {
                        let tree = Arc::clone(&tree);
                        let keys = keys.clone();
                        thread::spawn(move || {
                            let mut sum = 0u64;
                            let mut write_counter = 1000u64 + (t * OPS_PER_THREAD) as u64;

                            for i in 0..OPS_PER_THREAD {
                                if i % 10 == 0 {
                                    // 10% writes
                                    let key = write_counter.to_be_bytes().to_vec();
                                    let mut guard = tree.write().unwrap();
                                    guard.insert(key, write_counter);
                                    write_counter += 1;
                                } else {
                                    // 90% reads
                                    let key = &keys[i % keys.len()];
                                    let guard = tree.read().unwrap();
                                    if let Some(&v) = guard.get(key) {
                                        sum += v;
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
}

// =============================================================================
// MULTI-THREADED: Mixed 50/50
// =============================================================================

#[divan::bench_group(name = "06_mixed_50_50")]
mod mixed_balanced {
    use super::*;

    const OPS_PER_THREAD: usize = 500;

    #[divan::bench(args = [1, 2, 4])]
    fn masstree(bencher: Bencher, threads: usize) {
        bencher
            .with_inputs(|| {
                let tree = Arc::new(setup_masstree(1000));
                let keys: Vec<[u8; 8]> =
                    (0..100).map(|i| (i as u64 * 10).to_be_bytes()).collect();
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
                                if i % 2 == 0 {
                                    // 50% writes
                                    let key = write_counter.to_be_bytes();
                                    let _ = tree.insert_with_guard(&key, write_counter, &guard);
                                    write_counter += 1;
                                } else {
                                    // 50% reads
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

    #[divan::bench(args = [1, 2, 4])]
    fn mutex_btreemap(bencher: Bencher, threads: usize) {
        bencher
            .with_inputs(|| {
                let tree = Arc::new(setup_mutex_btreemap(1000));
                let keys: Vec<Vec<u8>> = (0..100)
                    .map(|i| (i as u64 * 10).to_be_bytes().to_vec())
                    .collect();
                (tree, keys)
            })
            .bench_local_values(|(tree, keys)| {
                let handles: Vec<_> = (0..threads)
                    .map(|t| {
                        let tree = Arc::clone(&tree);
                        let keys = keys.clone();
                        thread::spawn(move || {
                            let mut sum = 0u64;
                            let mut write_counter = 1000u64 + (t * OPS_PER_THREAD) as u64;

                            for i in 0..OPS_PER_THREAD {
                                if i % 2 == 0 {
                                    // 50% writes
                                    let key = write_counter.to_be_bytes().to_vec();
                                    let mut guard = tree.lock().unwrap();
                                    guard.insert(key, write_counter);
                                    write_counter += 1;
                                } else {
                                    // 50% reads
                                    let key = &keys[i % keys.len()];
                                    let guard = tree.lock().unwrap();
                                    if let Some(&v) = guard.get(key) {
                                        sum += v;
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

    #[divan::bench(args = [1, 2, 4])]
    fn rwlock_btreemap(bencher: Bencher, threads: usize) {
        bencher
            .with_inputs(|| {
                let tree = Arc::new(setup_rwlock_btreemap(1000));
                let keys: Vec<Vec<u8>> = (0..100)
                    .map(|i| (i as u64 * 10).to_be_bytes().to_vec())
                    .collect();
                (tree, keys)
            })
            .bench_local_values(|(tree, keys)| {
                let handles: Vec<_> = (0..threads)
                    .map(|t| {
                        let tree = Arc::clone(&tree);
                        let keys = keys.clone();
                        thread::spawn(move || {
                            let mut sum = 0u64;
                            let mut write_counter = 1000u64 + (t * OPS_PER_THREAD) as u64;

                            for i in 0..OPS_PER_THREAD {
                                if i % 2 == 0 {
                                    // 50% writes
                                    let key = write_counter.to_be_bytes().to_vec();
                                    let mut guard = tree.write().unwrap();
                                    guard.insert(key, write_counter);
                                    write_counter += 1;
                                } else {
                                    // 50% reads
                                    let key = &keys[i % keys.len()];
                                    let guard = tree.read().unwrap();
                                    if let Some(&v) = guard.get(key) {
                                        sum += v;
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
}

// =============================================================================
// CONTENTION: Single Hot Key
// =============================================================================

#[divan::bench_group(name = "07_hot_key_contention")]
mod hot_key {
    use super::*;

    const OPS_PER_THREAD: usize = 500;

    /// All threads read/write the same key - maximum contention
    #[divan::bench(args = [2, 4, 8])]
    fn masstree(bencher: Bencher, threads: usize) {
        let hot_key = 500u64.to_be_bytes();

        bencher
            .with_inputs(|| Arc::new(setup_masstree(1000)))
            .bench_local_values(|tree| {
                let handles: Vec<_> = (0..threads)
                    .map(|t| {
                        let tree = Arc::clone(&tree);
                        thread::spawn(move || {
                            let guard = tree.guard();
                            let mut sum = 0u64;

                            for i in 0..OPS_PER_THREAD {
                                if i % 10 == 0 {
                                    let _ = tree.insert_with_guard(
                                        &hot_key,
                                        (t * 1000 + i) as u64,
                                        &guard,
                                    );
                                } else if let Some(v) = tree.get_with_guard(&hot_key, &guard) {
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
                tree
            });
    }

    #[divan::bench(args = [2, 4, 8])]
    fn mutex_btreemap(bencher: Bencher, threads: usize) {
        let hot_key = 500u64.to_be_bytes().to_vec();

        bencher
            .with_inputs(|| Arc::new(setup_mutex_btreemap(1000)))
            .bench_local_values(|tree| {
                let handles: Vec<_> = (0..threads)
                    .map(|t| {
                        let tree = Arc::clone(&tree);
                        let hot_key = hot_key.clone();
                        thread::spawn(move || {
                            let mut sum = 0u64;

                            for i in 0..OPS_PER_THREAD {
                                if i % 10 == 0 {
                                    let mut guard = tree.lock().unwrap();
                                    guard.insert(hot_key.clone(), (t * 1000 + i) as u64);
                                } else {
                                    let guard = tree.lock().unwrap();
                                    if let Some(&v) = guard.get(&hot_key) {
                                        sum += v;
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

    #[divan::bench(args = [2, 4, 8])]
    fn rwlock_btreemap(bencher: Bencher, threads: usize) {
        let hot_key = 500u64.to_be_bytes().to_vec();

        bencher
            .with_inputs(|| Arc::new(setup_rwlock_btreemap(1000)))
            .bench_local_values(|tree| {
                let handles: Vec<_> = (0..threads)
                    .map(|t| {
                        let tree = Arc::clone(&tree);
                        let hot_key = hot_key.clone();
                        thread::spawn(move || {
                            let mut sum = 0u64;

                            for i in 0..OPS_PER_THREAD {
                                if i % 10 == 0 {
                                    let mut guard = tree.write().unwrap();
                                    guard.insert(hot_key.clone(), (t * 1000 + i) as u64);
                                } else {
                                    let guard = tree.read().unwrap();
                                    if let Some(&v) = guard.get(&hot_key) {
                                        sum += v;
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
}
