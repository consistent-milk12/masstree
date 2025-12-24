//! Fault isolation benchmarks for debugging race conditions.
//!
//! This benchmark file isolates problematic concurrent scenarios to reproduce
//! and debug issues like the mixed workload SIGSEGV.
//!
//! # Usage
//!
//! ```bash
//! # Run all fault benchmarks
//! cargo bench --bench fault_bench --features mimalloc
//!
//! # Run with tracing (logs to logs/masstree.json)
//! RUST_LOG=masstree=debug MASSTREE_LOG_CONSOLE=0 \
//!     cargo bench --bench fault_bench --features "mimalloc tracing"
//!
//! # Run specific test
//! cargo bench --bench fault_bench -- mixed_read_write
//!
//! # Analyze crash logs
//! cat logs/masstree.json | jq 'select(.fields.op == "split")'
//! ```

use std::hint::black_box;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Barrier};
use std::thread;

use divan::Bencher;
use masstree::MassTree24;

fn main() {
    // Initialize tracing if feature is enabled
    masstree::init_tracing();

    divan::main();
}

// =============================================================================
// Configuration
// =============================================================================

/// Number of keys to pre-populate the tree with.
const PREPOPULATE_KEYS: usize = 50_000;

/// Operations per thread in benchmarks.
const OPS_PER_THREAD: usize = 10_000;

/// Write ratio for mixed workloads (1 = 100% writes, 10 = 10% writes, 100 = 1% writes).
const WRITE_RATIO: usize = 10;

// =============================================================================
// Helper Functions
// =============================================================================

/// Generate deterministic keys for benchmarking.
fn generate_keys(count: usize) -> Vec<[u8; 8]> {
    (0..count)
        .map(|i| (i as u64).to_be_bytes())
        .collect()
}

/// Pre-populate a tree with keys.
fn prepopulate_tree(tree: &MassTree24<u64>, keys: &[[u8; 8]]) {
    let guard = tree.guard();
    for (i, key) in keys.iter().enumerate() {
        let _ = tree.insert_with_guard(key, i as u64, &guard);
    }
}

/// Simple xorshift PRNG for deterministic random access patterns.
fn xorshift(state: &mut u64) -> u64 {
    let mut x = *state;
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    *state = x;
    x
}

// =============================================================================
// 01: Mixed Read/Write - The SIGSEGV trigger
// =============================================================================

/// Mixed read/write workload that triggers SIGSEGV at 4+ threads.
///
/// Pattern:
/// - Pre-populate tree with PREPOPULATE_KEYS
/// - Each thread does OPS_PER_THREAD operations
/// - 10% writes (inserts to new keys), 90% reads
/// - Uniform random key access
#[divan::bench_group(name = "01_mixed_read_write")]
mod mixed_read_write {
    use super::*;

    #[divan::bench(args = [1, 2, 4, 8, 16])]
    fn masstree24(bencher: Bencher, threads: usize) {
        let keys: Arc<Vec<[u8; 8]>> = Arc::new(generate_keys(PREPOPULATE_KEYS));

        bencher
            .with_inputs(|| {
                let tree = Arc::new(MassTree24::<u64>::new());
                prepopulate_tree(&tree, &keys);
                tree
            })
            .bench_local_values(|tree| {
                let barrier = Arc::new(Barrier::new(threads));
                let crash_detected = Arc::new(AtomicBool::new(false));

                let handles: Vec<_> = (0..threads)
                    .map(|t| {
                        let tree = Arc::clone(&tree);
                        let keys = Arc::clone(&keys);
                        let barrier = Arc::clone(&barrier);
                        let crash_detected = Arc::clone(&crash_detected);

                        thread::spawn(move || {
                            let guard = tree.guard();
                            let mut rng_state = (t as u64 + 1) * 0x517cc1b727220a95;
                            let mut sum = 0u64;

                            barrier.wait();

                            for i in 0..OPS_PER_THREAD {
                                if crash_detected.load(Ordering::Relaxed) {
                                    break;
                                }

                                let idx = (xorshift(&mut rng_state) as usize) % keys.len();

                                if i % WRITE_RATIO == 0 {
                                    // Write: insert with a new unique key
                                    let write_key = ((t * OPS_PER_THREAD + i + PREPOPULATE_KEYS) as u64).to_be_bytes();
                                    let _ = tree.insert_with_guard(&write_key, i as u64, &guard);
                                } else {
                                    // Read: get existing key
                                    if let Some(v) = tree.get_with_guard(&keys[idx], &guard) {
                                        sum = sum.wrapping_add(*v);
                                    }
                                }
                            }

                            black_box(sum)
                        })
                    })
                    .collect();

                for h in handles {
                    let _ = h.join();
                }

                tree
            });
    }
}

// =============================================================================
// 02: Mixed with Overlapping Writes - Higher contention
// =============================================================================

/// Mixed workload where writes target existing keys (updates, not inserts).
/// This should have higher lock contention but may avoid the split-related race.
#[divan::bench_group(name = "02_mixed_overlapping")]
mod mixed_overlapping {
    use super::*;

    #[divan::bench(args = [1, 2, 4, 8, 16])]
    fn masstree24(bencher: Bencher, threads: usize) {
        let keys: Arc<Vec<[u8; 8]>> = Arc::new(generate_keys(PREPOPULATE_KEYS));

        bencher
            .with_inputs(|| {
                let tree = Arc::new(MassTree24::<u64>::new());
                prepopulate_tree(&tree, &keys);
                tree
            })
            .bench_local_values(|tree| {
                let barrier = Arc::new(Barrier::new(threads));

                let handles: Vec<_> = (0..threads)
                    .map(|t| {
                        let tree = Arc::clone(&tree);
                        let keys = Arc::clone(&keys);
                        let barrier = Arc::clone(&barrier);

                        thread::spawn(move || {
                            let guard = tree.guard();
                            let mut rng_state = (t as u64 + 1) * 0x517cc1b727220a95;
                            let mut sum = 0u64;

                            barrier.wait();

                            for i in 0..OPS_PER_THREAD {
                                let idx = (xorshift(&mut rng_state) as usize) % keys.len();

                                if i % WRITE_RATIO == 0 {
                                    // Write to EXISTING key (update, not new insert)
                                    let _ = tree.insert_with_guard(&keys[idx], i as u64, &guard);
                                } else {
                                    // Read
                                    if let Some(v) = tree.get_with_guard(&keys[idx], &guard) {
                                        sum = sum.wrapping_add(*v);
                                    }
                                }
                            }

                            black_box(sum)
                        })
                    })
                    .collect();

                for h in handles {
                    let _ = h.join();
                }

                tree
            });
    }
}

// =============================================================================
// 03: Read During Split - Targeted reproduction
// =============================================================================

/// Specifically triggers splits while reads are happening.
///
/// - Writer threads continuously insert new keys (forcing splits)
/// - Reader threads continuously read existing keys
/// - This isolates the read-during-split race
#[divan::bench_group(name = "03_read_during_split")]
mod read_during_split {
    use super::*;

    const READER_THREADS: usize = 4;
    const WRITER_THREADS: usize = 2;

    #[divan::bench(args = [1, 2, 4])]
    fn masstree24(bencher: Bencher, scale: usize) {
        let readers = READER_THREADS * scale;
        let writers = WRITER_THREADS * scale;
        let total_threads = readers + writers;

        let keys: Arc<Vec<[u8; 8]>> = Arc::new(generate_keys(PREPOPULATE_KEYS));

        bencher
            .with_inputs(|| {
                let tree = Arc::new(MassTree24::<u64>::new());
                prepopulate_tree(&tree, &keys);
                tree
            })
            .bench_local_values(|tree| {
                let barrier = Arc::new(Barrier::new(total_threads));
                let write_counter = Arc::new(AtomicU64::new(PREPOPULATE_KEYS as u64));

                let mut handles = Vec::with_capacity(total_threads);

                // Spawn reader threads
                for t in 0..readers {
                    let tree = Arc::clone(&tree);
                    let keys = Arc::clone(&keys);
                    let barrier = Arc::clone(&barrier);

                    handles.push(thread::spawn(move || {
                        let guard = tree.guard();
                        let mut rng_state = (t as u64 + 1) * 0x517cc1b727220a95;
                        let mut sum = 0u64;

                        barrier.wait();

                        for _ in 0..OPS_PER_THREAD {
                            let idx = (xorshift(&mut rng_state) as usize) % keys.len();
                            if let Some(v) = tree.get_with_guard(&keys[idx], &guard) {
                                sum = sum.wrapping_add(*v);
                            }
                        }

                        black_box(sum)
                    }));
                }

                // Spawn writer threads
                for _ in 0..writers {
                    let tree = Arc::clone(&tree);
                    let barrier = Arc::clone(&barrier);
                    let write_counter = Arc::clone(&write_counter);

                    handles.push(thread::spawn(move || {
                        let guard = tree.guard();

                        barrier.wait();

                        for _ in 0..OPS_PER_THREAD {
                            // Each write is a NEW key, forcing tree growth and splits
                            let key_val = write_counter.fetch_add(1, Ordering::Relaxed);
                            let key = key_val.to_be_bytes();
                            let _ = tree.insert_with_guard(&key, key_val, &guard);
                        }

                        0u64 // Return dummy value to match reader type
                    }));
                }

                for h in handles {
                    let _ = h.join();
                }

                tree
            });
    }
}

// =============================================================================
// 04: Sequential Read During Split - Easier to debug
// =============================================================================

/// Single reader, single writer - minimal reproduction.
/// If this crashes, we have the simplest possible reproduction case.
#[divan::bench_group(name = "04_minimal_mixed")]
mod minimal_mixed {
    use super::*;

    #[divan::bench]
    fn one_reader_one_writer(bencher: Bencher) {
        let keys: Arc<Vec<[u8; 8]>> = Arc::new(generate_keys(PREPOPULATE_KEYS));

        bencher
            .with_inputs(|| {
                let tree = Arc::new(MassTree24::<u64>::new());
                prepopulate_tree(&tree, &keys);
                tree
            })
            .bench_local_values(|tree| {
                let barrier = Arc::new(Barrier::new(2));
                let write_counter = Arc::new(AtomicU64::new(PREPOPULATE_KEYS as u64));

                // Reader thread
                let tree_r = Arc::clone(&tree);
                let keys_r = Arc::clone(&keys);
                let barrier_r = Arc::clone(&barrier);
                let reader = thread::spawn(move || {
                    let guard = tree_r.guard();
                    let mut rng_state = 0x12345678u64;
                    let mut sum = 0u64;

                    barrier_r.wait();

                    for _ in 0..(OPS_PER_THREAD * 10) {
                        let idx = (xorshift(&mut rng_state) as usize) % keys_r.len();
                        if let Some(v) = tree_r.get_with_guard(&keys_r[idx], &guard) {
                            sum = sum.wrapping_add(*v);
                        }
                    }

                    black_box(sum)
                });

                // Writer thread
                let tree_w = Arc::clone(&tree);
                let barrier_w = Arc::clone(&barrier);
                let writer = thread::spawn(move || {
                    let guard = tree_w.guard();

                    barrier_w.wait();

                    for _ in 0..OPS_PER_THREAD {
                        let key_val = write_counter.fetch_add(1, Ordering::Relaxed);
                        let key = key_val.to_be_bytes();
                        let _ = tree_w.insert_with_guard(&key, key_val, &guard);
                    }
                });

                let _ = reader.join();
                let _ = writer.join();

                tree
            });
    }
}

// =============================================================================
// 05: Stress Test - Maximum pressure
// =============================================================================

/// High thread count, high operation count - find the breaking point.
#[divan::bench_group(name = "05_stress")]
mod stress {
    use super::*;

    const STRESS_OPS: usize = 50_000;

    #[divan::bench(args = [4, 8, 16, 32])]
    fn mixed_stress(bencher: Bencher, threads: usize) {
        let keys: Arc<Vec<[u8; 8]>> = Arc::new(generate_keys(PREPOPULATE_KEYS));

        bencher
            .with_inputs(|| {
                let tree = Arc::new(MassTree24::<u64>::new());
                prepopulate_tree(&tree, &keys);
                tree
            })
            .bench_local_values(|tree| {
                let barrier = Arc::new(Barrier::new(threads));
                let write_counter = Arc::new(AtomicU64::new(PREPOPULATE_KEYS as u64));

                let handles: Vec<_> = (0..threads)
                    .map(|t| {
                        let tree = Arc::clone(&tree);
                        let keys = Arc::clone(&keys);
                        let barrier = Arc::clone(&barrier);
                        let write_counter = Arc::clone(&write_counter);

                        thread::spawn(move || {
                            let guard = tree.guard();
                            let mut rng_state = (t as u64 + 1) * 0x517cc1b727220a95;
                            let mut sum = 0u64;

                            barrier.wait();

                            for i in 0..STRESS_OPS {
                                let idx = (xorshift(&mut rng_state) as usize) % keys.len();

                                if i % WRITE_RATIO == 0 {
                                    // New key insert
                                    let key_val = write_counter.fetch_add(1, Ordering::Relaxed);
                                    let key = key_val.to_be_bytes();
                                    let _ = tree.insert_with_guard(&key, key_val, &guard);
                                } else {
                                    // Read existing
                                    if let Some(v) = tree.get_with_guard(&keys[idx], &guard) {
                                        sum = sum.wrapping_add(*v);
                                    }
                                }
                            }

                            black_box(sum)
                        })
                    })
                    .collect();

                for h in handles {
                    let _ = h.join();
                }

                tree
            });
    }
}

// =============================================================================
// 06: Write-Only Baseline - Verify writes alone don't crash
// =============================================================================

/// Pure write workload - this should NOT crash (confirmed working).
/// Used as a baseline to confirm the issue is read+write interaction.
#[divan::bench_group(name = "06_write_only_baseline")]
mod write_only_baseline {
    use super::*;

    #[divan::bench(args = [1, 2, 4, 8, 16])]
    fn masstree24(bencher: Bencher, threads: usize) {
        bencher
            .with_inputs(|| Arc::new(MassTree24::<u64>::new()))
            .bench_local_values(|tree| {
                let barrier = Arc::new(Barrier::new(threads));
                let write_counter = Arc::new(AtomicU64::new(0));

                let handles: Vec<_> = (0..threads)
                    .map(|_| {
                        let tree = Arc::clone(&tree);
                        let barrier = Arc::clone(&barrier);
                        let write_counter = Arc::clone(&write_counter);

                        thread::spawn(move || {
                            let guard = tree.guard();

                            barrier.wait();

                            for _ in 0..OPS_PER_THREAD {
                                let key_val = write_counter.fetch_add(1, Ordering::Relaxed);
                                let key = key_val.to_be_bytes();
                                let _ = tree.insert_with_guard(&key, key_val, &guard);
                            }
                        })
                    })
                    .collect();

                for h in handles {
                    let _ = h.join();
                }

                tree
            });
    }
}

// =============================================================================
// 07: Read-Only Baseline - Verify reads alone don't crash
// =============================================================================

// =============================================================================
// 08: High Contention Mixed - Reproduces SIGSEGV
// =============================================================================

/// High contention mixed workload - ALL threads access the SAME key sequence.
/// This reproduces the SIGSEGV from concurrent_maps24::mixed_uniform.
#[divan::bench_group(name = "08_high_contention_mixed")]
mod high_contention_mixed {
    use super::*;

    const N: usize = 100_000;

    /// Pre-compute shared indices (same as concurrent_maps24)
    fn uniform_indices(n: usize, count: usize, seed: u64) -> Vec<usize> {
        let mut indices = Vec::with_capacity(count);
        let mut state = seed;

        for _ in 0..count {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1);
            indices.push((state as usize) % n);
        }
        indices
    }

    #[divan::bench(args = [1, 2, 4, 8, 16])]
    fn masstree24(bencher: Bencher, threads: usize) {
        let keys: Arc<Vec<[u8; 8]>> = Arc::new(generate_keys(N));
        // SHARED indices - all threads use the same sequence with offset
        let indices: Arc<Vec<usize>> = Arc::new(uniform_indices(N, OPS_PER_THREAD, 42));

        bencher
            .with_inputs(|| {
                let tree = Arc::new(MassTree24::<u64>::new());
                prepopulate_tree(&tree, &keys);
                tree
            })
            .bench_local_values(|tree| {
                let barrier = Arc::new(Barrier::new(threads));

                let handles: Vec<_> = (0..threads)
                    .map(|t| {
                        let tree = Arc::clone(&tree);
                        let keys = Arc::clone(&keys);
                        let indices = Arc::clone(&indices);
                        let barrier = Arc::clone(&barrier);

                        thread::spawn(move || {
                            let guard = tree.guard();
                            let mut sum = 0u64;
                            // Same offset pattern as concurrent_maps24
                            let offset = t * 7919;

                            barrier.wait();

                            for i in 0..OPS_PER_THREAD {
                                let idx = indices[(i + offset) % indices.len()];

                                if i % WRITE_RATIO == 0 {
                                    // Write to EXISTING key (update, not insert)
                                    let _ = tree.insert_with_guard(&keys[idx], i as u64, &guard);
                                } else {
                                    // Read
                                    if let Some(v) = tree.get_with_guard(&keys[idx], &guard) {
                                        sum = sum.wrapping_add(*v);
                                    }
                                }
                            }

                            black_box(sum)
                        })
                    })
                    .collect();

                for h in handles {
                    let _ = h.join();
                }

                tree
            });
    }
}

/// Pure read workload on pre-populated tree - should NOT crash.
#[divan::bench_group(name = "07_read_only_baseline")]
mod read_only_baseline {
    use super::*;

    #[divan::bench(args = [1, 2, 4, 8, 16])]
    fn masstree24(bencher: Bencher, threads: usize) {
        let keys: Arc<Vec<[u8; 8]>> = Arc::new(generate_keys(PREPOPULATE_KEYS));

        bencher
            .with_inputs(|| {
                let tree = Arc::new(MassTree24::<u64>::new());
                prepopulate_tree(&tree, &keys);
                tree
            })
            .bench_local_values(|tree| {
                let barrier = Arc::new(Barrier::new(threads));

                let handles: Vec<_> = (0..threads)
                    .map(|t| {
                        let tree = Arc::clone(&tree);
                        let keys = Arc::clone(&keys);
                        let barrier = Arc::clone(&barrier);

                        thread::spawn(move || {
                            let guard = tree.guard();
                            let mut rng_state = (t as u64 + 1) * 0x517cc1b727220a95;
                            let mut sum = 0u64;

                            barrier.wait();

                            for _ in 0..OPS_PER_THREAD {
                                let idx = (xorshift(&mut rng_state) as usize) % keys.len();
                                if let Some(v) = tree.get_with_guard(&keys[idx], &guard) {
                                    sum = sum.wrapping_add(*v);
                                }
                            }

                            black_box(sum)
                        })
                    })
                    .collect();

                for h in handles {
                    let _ = h.join();
                }

                tree
            });
    }
}
