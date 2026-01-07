//! C++ Masstree Compatibility Benchmarks (Divan)
//!
//! Direct port of C++ mttest benchmark suite using proper statistical benchmarking.
//!
//! ## Methodology
//!
//! Each benchmark follows a rigorous methodology:
//! 1. **Explicit warmup**: Each thread warms up the data structure before measurement
//! 2. **Memory barriers**: Inserted before/after measurement to prevent reordering
//! 3. **Consistent setup**: All benchmarks use `.with_inputs()` for fresh state
//! 4. **Increased samples**: 200 samples for better statistical significance
//!
//! ## Running
//!
//! ```bash
//! # Run all benchmarks
//! cargo bench --bench cpp_comp --features mimalloc
//!
//! # Run specific group
//! cargo bench --bench cpp_comp --features mimalloc -- rw3
//! ```

#![allow(clippy::cast_possible_truncation)]
#![allow(clippy::cast_sign_loss)]
#![allow(clippy::indexing_slicing)]
#![allow(clippy::unwrap_used)]
#![allow(clippy::pedantic)]

mod bench_utils;

use bench_utils::{post_measurement_barrier, pre_measurement_barrier};
use divan::{Bencher, black_box};
use masstree::MassTree15Inline;
use std::sync::Arc;
use std::sync::Barrier;
use std::thread;

fn main() {
    divan::main();
}

// =============================================================================
// Constants
// =============================================================================

/// Warmup iterations per thread before measurement
const WARMUP_OPS: usize = 500;

// =============================================================================
// C++ compatible LCG RNG (from kvrandom.hh)
// =============================================================================

struct KvRandom {
    seed: u32,
}

impl KvRandom {
    const A: u32 = 1_664_525;
    const C: u32 = 1_013_904_223;
    const FIRST_SEED: u64 = 31949;

    const fn new(seed: u64) -> Self {
        Self { seed: seed as u32 }
    }

    #[allow(dead_code)]
    const fn reset(&mut self, seed: u64) {
        self.seed = seed as u32;
    }

    #[inline]
    #[allow(dead_code)]
    const fn lcg_step(&mut self) -> u32 {
        self.seed = self.seed.wrapping_mul(Self::A).wrapping_add(Self::C);
        self.seed
    }

    #[allow(dead_code)]
    const fn rand(&mut self) -> u32 {
        self.lcg_step();
        let x0 = self.lcg_step();
        self.lcg_step();
        let x1 = self.lcg_step();
        (x0 >> 15) | ((x1 & 0x7FFE) << 16)
    }

    fn bernoulli(&mut self, p: f64) -> bool {
        (f64::from(self.rand()) / f64::from(0x7FFF_FFFF)) < p
    }

    const fn uniform(&mut self, max: u32) -> u32 {
        self.rand() % max
    }
}

// =============================================================================
// Key generation (ASCII decimal like C++ quick_istr)
// =============================================================================

#[inline]
fn quick_istr<const N: usize>(mut n: u64) -> [u8; N] {
    let mut buf = [b'0'; N];
    let mut i = N;
    if n == 0 {
        return buf;
    }
    while n > 0 && i > 0 {
        i -= 1;
        buf[i] = b'0' + (n % 10) as u8;
        n /= 10;
    }
    buf
}

#[inline]
fn key8(n: u64) -> [u8; 8] {
    quick_istr::<8>(n)
}

#[inline]
fn key_var(n: u64) -> Vec<u8> {
    if n == 0 {
        return vec![b'0'];
    }
    let mut buf = Vec::with_capacity(20);
    let mut m = n;
    while m > 0 {
        buf.push(b'0' + (m % 10) as u8);
        m /= 10;
    }
    buf.reverse();
    buf
}

// =============================================================================
// RW1: Random keys - insert then shuffled read
// =============================================================================

#[divan::bench_group(name = "rw1", sample_count = 200)]
mod rw1 {
    use super::*;

    const OPS_PER_THREAD: usize = 50_000;

    #[divan::bench(args = [1, 2, 4, 6])]
    fn random_insert_then_read(bencher: Bencher, threads: usize) {
        bencher
            .counter(divan::counter::ItemsCount::new(
                threads * OPS_PER_THREAD * 2,
            ))
            .with_inputs(|| Arc::new(MassTree15Inline::<u64>::new()))
            .bench_local_values(|tree| {
                let warmup_done = Arc::new(Barrier::new(threads));
                let start = Arc::new(Barrier::new(threads));
                let handles: Vec<_> = (0..threads)
                    .map(|tid| {
                        let tree = Arc::clone(&tree);
                        let warmup_done = Arc::clone(&warmup_done);
                        let start = Arc::clone(&start);
                        thread::spawn(move || {
                            let guard = tree.guard();
                            let seed = KvRandom::FIRST_SEED + (tid % 48) as u64;
                            let mut rng = KvRandom::new(seed);

                            // === WARMUP PHASE ===
                            for _ in 0..WARMUP_OPS {
                                let x = rng.rand();
                                let key = key_var(u64::from(x));
                                black_box(tree.get_with_guard(&key, &guard));
                            }
                            warmup_done.wait();

                            // Reset RNG for measurement
                            rng = KvRandom::new(seed);

                            // === MEASUREMENT PHASE ===
                            pre_measurement_barrier();
                            start.wait();

                            // Put phase
                            let mut keys_inserted = Vec::with_capacity(OPS_PER_THREAD);
                            for _ in 0..OPS_PER_THREAD {
                                let x = rng.rand();
                                let key = key_var(u64::from(x));
                                let _ = tree.insert_with_guard(&key, u64::from(x + 1), &guard);
                                keys_inserted.push(x);
                            }

                            // Shuffle
                            for i in 0..keys_inserted.len() {
                                let j = rng.uniform(keys_inserted.len() as u32) as usize;
                                keys_inserted.swap(i, j);
                            }

                            // Get phase
                            let mut sum = 0u64;
                            for x in keys_inserted {
                                let key = key_var(u64::from(x));
                                if let Some(v) = tree.get_with_guard(&key, &guard) {
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
                tree
            });
    }
}

// =============================================================================
// RW2: Interleaved insert/read with configurable get fraction
// =============================================================================

#[divan::bench_group(name = "rw2", sample_count = 200)]
mod rw2 {
    use super::*;

    const OPS_PER_THREAD: usize = 100_000;

    fn run_rw2(bencher: Bencher, threads: usize, get_frac: f64) {
        bencher
            .counter(divan::counter::ItemsCount::new(threads * OPS_PER_THREAD))
            .with_inputs(|| Arc::new(MassTree15Inline::<u64>::new()))
            .bench_local_values(|tree| {
                let warmup_done = Arc::new(Barrier::new(threads));
                let start = Arc::new(Barrier::new(threads));
                let handles: Vec<_> = (0..threads)
                    .map(|tid| {
                        let tree = Arc::clone(&tree);
                        let warmup_done = Arc::clone(&warmup_done);
                        let start = Arc::clone(&start);
                        thread::spawn(move || {
                            let guard = tree.guard();
                            let seed = KvRandom::FIRST_SEED + (tid % 48) as u64;
                            let mut rng = KvRandom::new(seed);

                            // === WARMUP PHASE ===
                            for _ in 0..WARMUP_OPS {
                                let x = rng.rand();
                                let key = key_var(u64::from(x));
                                black_box(tree.get_with_guard(&key, &guard));
                            }
                            warmup_done.wait();

                            // Reset RNG for measurement
                            rng = KvRandom::new(seed);
                            let offset = rng.rand();
                            const C: u32 = 2_654_435_761;

                            // === MEASUREMENT PHASE ===
                            pre_measurement_barrier();
                            start.wait();

                            let mut puts = 0u64;
                            let mut sum = 0u64;
                            for _ in 0..OPS_PER_THREAD {
                                if puts == 0 || !rng.bernoulli(get_frac) {
                                    let x = (offset.wrapping_add(puts as u32)).wrapping_mul(C);
                                    let key = key_var(u64::from(x));
                                    let _ = tree.insert_with_guard(&key, u64::from(x + 1), &guard);
                                    puts += 1;
                                } else {
                                    let idx = rng.uniform(puts as u32);
                                    let x = (offset.wrapping_add(idx)).wrapping_mul(C);
                                    let key = key_var(u64::from(x));
                                    if let Some(v) = tree.get_with_guard(&key, &guard) {
                                        sum = sum.wrapping_add(v);
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
                tree
            });
    }

    #[divan::bench(name = "50pct_get", args = [1, 2, 4, 6])]
    fn rw2_50(bencher: Bencher, threads: usize) {
        run_rw2(bencher, threads, 0.5);
    }

    #[divan::bench(name = "90pct_get", args = [1, 2, 4, 6])]
    fn rw2_90(bencher: Bencher, threads: usize) {
        run_rw2(bencher, threads, 0.9);
    }

    #[divan::bench(name = "98pct_get", args = [1, 2, 4, 6])]
    fn rw2_98(bencher: Bencher, threads: usize) {
        run_rw2(bencher, threads, 0.98);
    }
}

// =============================================================================
// RW3: Sequential 8-byte keys
// =============================================================================

#[divan::bench_group(name = "rw3_sequential", sample_count = 200)]
mod rw3 {
    use super::*;

    const OPS_PER_THREAD: usize = 100_000;

    #[divan::bench(args = [1, 2, 4, 6])]
    fn sequential_insert_then_read(bencher: Bencher, threads: usize) {
        bencher
            .counter(divan::counter::ItemsCount::new(
                threads * OPS_PER_THREAD * 2,
            ))
            .with_inputs(|| Arc::new(MassTree15Inline::<u64>::new()))
            .bench_local_values(|tree| {
                let warmup_done = Arc::new(Barrier::new(threads));
                let start = Arc::new(Barrier::new(threads));
                let handles: Vec<_> = (0..threads)
                    .map(|_| {
                        let tree = Arc::clone(&tree);
                        let warmup_done = Arc::clone(&warmup_done);
                        let start = Arc::clone(&start);
                        thread::spawn(move || {
                            let guard = tree.guard();

                            // === WARMUP PHASE ===
                            for n in 0..WARMUP_OPS {
                                let key = key8(n as u64);
                                black_box(tree.get_with_guard(&key, &guard));
                            }
                            warmup_done.wait();

                            // === MEASUREMENT PHASE ===
                            pre_measurement_barrier();
                            start.wait();

                            // Put phase
                            for n in 0..OPS_PER_THREAD {
                                let key = key8(n as u64);
                                let _ = tree.insert_with_guard(&key, n as u64 + 1, &guard);
                            }

                            // Get phase
                            let mut sum = 0u64;
                            for i in 0..OPS_PER_THREAD {
                                let key = key8(i as u64);
                                if let Some(v) = tree.get_with_guard(&key, &guard) {
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
                tree
            });
    }

    #[divan::bench(args = [1, 2, 4, 6])]
    fn sequential_disjoint(bencher: Bencher, threads: usize) {
        bencher
            .counter(divan::counter::ItemsCount::new(
                threads * OPS_PER_THREAD * 2,
            ))
            .with_inputs(|| Arc::new(MassTree15Inline::<u64>::new()))
            .bench_local_values(|tree| {
                let warmup_done = Arc::new(Barrier::new(threads));
                let start = Arc::new(Barrier::new(threads));
                let handles: Vec<_> = (0..threads)
                    .map(|tid| {
                        let tree = Arc::clone(&tree);
                        let warmup_done = Arc::clone(&warmup_done);
                        let start = Arc::clone(&start);
                        thread::spawn(move || {
                            let guard = tree.guard();
                            let base = (tid as u64) * 10_000_000;

                            // === WARMUP PHASE ===
                            for n in 0..WARMUP_OPS {
                                let key = key8(base + n as u64);
                                black_box(tree.get_with_guard(&key, &guard));
                            }
                            warmup_done.wait();

                            // === MEASUREMENT PHASE ===
                            pre_measurement_barrier();
                            start.wait();

                            // Put phase
                            for n in 0..OPS_PER_THREAD {
                                let key = key8(base + n as u64);
                                let _ = tree.insert_with_guard(&key, n as u64 + 1, &guard);
                            }

                            // Get phase
                            let mut sum = 0u64;
                            for i in 0..OPS_PER_THREAD {
                                let key = key8(base + i as u64);
                                if let Some(v) = tree.get_with_guard(&key, &guard) {
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
                tree
            });
    }

    #[divan::bench(args = [1, 2, 4, 6])]
    fn sequential_binary_keys(bencher: Bencher, threads: usize) {
        bencher
            .counter(divan::counter::ItemsCount::new(
                threads * OPS_PER_THREAD * 2,
            ))
            .with_inputs(|| Arc::new(MassTree15Inline::<u64>::new()))
            .bench_local_values(|tree| {
                let warmup_done = Arc::new(Barrier::new(threads));
                let start = Arc::new(Barrier::new(threads));
                let handles: Vec<_> = (0..threads)
                    .map(|_| {
                        let tree = Arc::clone(&tree);
                        let warmup_done = Arc::clone(&warmup_done);
                        let start = Arc::clone(&start);
                        thread::spawn(move || {
                            let guard = tree.guard();

                            // === WARMUP PHASE ===
                            for n in 0..WARMUP_OPS {
                                let key = (n as u64).to_be_bytes();
                                black_box(tree.get_with_guard(&key, &guard));
                            }
                            warmup_done.wait();

                            // === MEASUREMENT PHASE ===
                            pre_measurement_barrier();
                            start.wait();

                            // Put phase - BINARY keys
                            for n in 0..OPS_PER_THREAD {
                                let key = (n as u64).to_be_bytes();
                                let _ = tree.insert_with_guard(&key, n as u64 + 1, &guard);
                            }

                            // Get phase
                            let mut sum = 0u64;
                            for i in 0..OPS_PER_THREAD {
                                let key = (i as u64).to_be_bytes();
                                if let Some(v) = tree.get_with_guard(&key, &guard) {
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
                tree
            });
    }
}

// =============================================================================
// RW4: Reverse sequential
// =============================================================================

#[divan::bench_group(name = "rw4_reverse", sample_count = 200)]
mod rw4 {
    use super::*;

    const OPS_PER_THREAD: usize = 100_000;
    const TOP: u64 = 2_147_483_647;

    #[divan::bench(args = [1, 2, 4, 6])]
    fn reverse_sequential(bencher: Bencher, threads: usize) {
        bencher
            .counter(divan::counter::ItemsCount::new(
                threads * OPS_PER_THREAD * 2,
            ))
            .with_inputs(|| Arc::new(MassTree15Inline::<u64>::new()))
            .bench_local_values(|tree| {
                let warmup_done = Arc::new(Barrier::new(threads));
                let start = Arc::new(Barrier::new(threads));
                let handles: Vec<_> = (0..threads)
                    .map(|_| {
                        let tree = Arc::clone(&tree);
                        let warmup_done = Arc::clone(&warmup_done);
                        let start = Arc::clone(&start);
                        thread::spawn(move || {
                            let guard = tree.guard();

                            // === WARMUP PHASE ===
                            for n in 0..WARMUP_OPS {
                                let key = key8(TOP - n as u64);
                                black_box(tree.get_with_guard(&key, &guard));
                            }
                            warmup_done.wait();

                            // === MEASUREMENT PHASE ===
                            pre_measurement_barrier();
                            start.wait();

                            // Put phase
                            for n in 0..OPS_PER_THREAD {
                                let key = key8(TOP - n as u64);
                                let _ = tree.insert_with_guard(&key, n as u64 + 1, &guard);
                            }

                            // Get phase
                            let mut sum = 0u64;
                            for i in 0..OPS_PER_THREAD {
                                let key = key8(TOP - i as u64);
                                if let Some(v) = tree.get_with_guard(&key, &guard) {
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
                tree
            });
    }
}

// =============================================================================
// SAME: Hot spot - 10 keys (high contention)
// =============================================================================

#[divan::bench_group(name = "same_hotspot", sample_count = 200)]
mod same {
    use super::*;

    const OPS_PER_THREAD: usize = 100_000;
    const NUM_KEYS: u32 = 10;

    #[divan::bench(args = [1, 2, 4, 6])]
    fn hot_10_keys(bencher: Bencher, threads: usize) {
        bencher
            .counter(divan::counter::ItemsCount::new(threads * OPS_PER_THREAD))
            .with_inputs(|| Arc::new(MassTree15Inline::<u64>::new()))
            .bench_local_values(|tree| {
                let warmup_done = Arc::new(Barrier::new(threads));
                let start = Arc::new(Barrier::new(threads));
                let handles: Vec<_> = (0..threads)
                    .map(|tid| {
                        let tree = Arc::clone(&tree);
                        let warmup_done = Arc::clone(&warmup_done);
                        let start = Arc::clone(&start);
                        thread::spawn(move || {
                            let guard = tree.guard();
                            let seed = KvRandom::FIRST_SEED + (tid % 48) as u64;
                            let mut rng = KvRandom::new(seed);

                            // === WARMUP PHASE ===
                            for _ in 0..WARMUP_OPS {
                                let x = rng.uniform(NUM_KEYS);
                                let key = key_var(u64::from(x));
                                black_box(tree.get_with_guard(&key, &guard));
                            }
                            warmup_done.wait();

                            // Reset RNG for measurement
                            rng = KvRandom::new(seed);

                            // === MEASUREMENT PHASE ===
                            pre_measurement_barrier();
                            start.wait();

                            for _ in 0..OPS_PER_THREAD {
                                let x = rng.uniform(NUM_KEYS);
                                let key = key_var(u64::from(x));
                                let _ = tree.insert_with_guard(&key, u64::from(x + 1), &guard);
                            }

                            post_measurement_barrier();
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
// WSCALE: Pure random writes
// =============================================================================

#[divan::bench_group(name = "wscale", sample_count = 200)]
mod wscale {
    use super::*;

    const OPS_PER_THREAD: usize = 100_000;

    #[divan::bench(args = [1, 2, 4, 6])]
    fn random_writes(bencher: Bencher, threads: usize) {
        bencher
            .counter(divan::counter::ItemsCount::new(threads * OPS_PER_THREAD))
            .with_inputs(|| Arc::new(MassTree15Inline::<u64>::new()))
            .bench_local_values(|tree| {
                let warmup_done = Arc::new(Barrier::new(threads));
                let start = Arc::new(Barrier::new(threads));
                let handles: Vec<_> = (0..threads)
                    .map(|tid| {
                        let tree = Arc::clone(&tree);
                        let warmup_done = Arc::clone(&warmup_done);
                        let start = Arc::clone(&start);
                        thread::spawn(move || {
                            let guard = tree.guard();
                            let seed = KvRandom::FIRST_SEED + (tid % 48) as u64;
                            let mut rng = KvRandom::new(seed);

                            // === WARMUP PHASE ===
                            for _ in 0..WARMUP_OPS {
                                let x = u64::from(rng.rand());
                                let key = key_var(x);
                                black_box(tree.get_with_guard(&key, &guard));
                            }
                            warmup_done.wait();

                            // Reset RNG for measurement
                            rng = KvRandom::new(seed);

                            // === MEASUREMENT PHASE ===
                            pre_measurement_barrier();
                            start.wait();

                            for _ in 0..OPS_PER_THREAD {
                                let x = u64::from(rng.rand());
                                let key = key_var(x);
                                let _ = tree.insert_with_guard(&key, x + 1, &guard);
                            }

                            post_measurement_barrier();
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
// RSCALE: Pure random reads on pre-populated tree
// =============================================================================

#[divan::bench_group(name = "rscale", sample_count = 200)]
mod rscale {
    use super::*;

    const KEYS_PER_THREAD: usize = 100_000;
    const OPS_PER_THREAD: usize = 100_000;

    #[divan::bench(args = [1, 2, 4, 6])]
    fn random_reads(bencher: Bencher, threads: usize) {
        // Pre-populate tree
        let tree = Arc::new(MassTree15Inline::<u64>::new());
        let nseqkeys = (KEYS_PER_THREAD * threads) as u64;

        {
            let guard = tree.guard();
            for part in 0..threads {
                let seed = KvRandom::FIRST_SEED + (part % 48) as u64;
                let mut rng = KvRandom::new(seed);
                let first_key = KEYS_PER_THREAD * part;

                let mut keys: Vec<usize> = (0..KEYS_PER_THREAD).map(|i| i + first_key).collect();
                for i in 0..KEYS_PER_THREAD {
                    let j = rng.uniform(KEYS_PER_THREAD as u32) as usize;
                    keys.swap(i, j);
                }

                for &k in &keys {
                    let key = key_var(k as u64);
                    let _ = tree.insert_with_guard(&key, k as u64 + 1, &guard);
                }
            }
        }

        bencher
            .counter(divan::counter::ItemsCount::new(threads * OPS_PER_THREAD))
            .bench_local(|| {
                let warmup_done = Arc::new(Barrier::new(threads));
                let start = Arc::new(Barrier::new(threads));
                let handles: Vec<_> = (0..threads)
                    .map(|tid| {
                        let tree = Arc::clone(&tree);
                        let warmup_done = Arc::clone(&warmup_done);
                        let start = Arc::clone(&start);
                        thread::spawn(move || {
                            let guard = tree.guard();
                            let seed = KvRandom::FIRST_SEED + (tid % 48) as u64;
                            let mut rng = KvRandom::new(seed);

                            // === WARMUP PHASE ===
                            for _ in 0..WARMUP_OPS {
                                let x = u64::from(rng.rand()) % nseqkeys;
                                let key = key_var(x);
                                black_box(tree.get_with_guard(&key, &guard));
                            }
                            warmup_done.wait();

                            // Reset RNG for measurement
                            rng = KvRandom::new(seed);

                            // === MEASUREMENT PHASE ===
                            pre_measurement_barrier();
                            start.wait();

                            let mut sum = 0u64;
                            for _ in 0..OPS_PER_THREAD {
                                let x = u64::from(rng.rand()) % nseqkeys;
                                let key = key_var(x);
                                if let Some(v) = tree.get_with_guard(&key, &guard) {
                                    sum += v;
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
// USCALE: Pure random updates on pre-populated tree
// =============================================================================

#[divan::bench_group(name = "uscale", sample_count = 200)]
mod uscale {
    use super::*;

    const KEYS_PER_THREAD: usize = 100_000;
    const OPS_PER_THREAD: usize = 100_000;

    #[divan::bench(args = [1, 2, 4, 6])]
    fn random_updates(bencher: Bencher, threads: usize) {
        // Pre-populate tree
        let tree = Arc::new(MassTree15Inline::<u64>::new());
        let nseqkeys = (KEYS_PER_THREAD * threads) as u64;

        {
            let guard = tree.guard();
            for i in 0..nseqkeys {
                let key = key_var(i);
                let _ = tree.insert_with_guard(&key, i + 1, &guard);
            }
        }

        bencher
            .counter(divan::counter::ItemsCount::new(threads * OPS_PER_THREAD))
            .bench_local(|| {
                let warmup_done = Arc::new(Barrier::new(threads));
                let start = Arc::new(Barrier::new(threads));
                let handles: Vec<_> = (0..threads)
                    .map(|tid| {
                        let tree = Arc::clone(&tree);
                        let warmup_done = Arc::clone(&warmup_done);
                        let start = Arc::clone(&start);
                        thread::spawn(move || {
                            let guard = tree.guard();
                            let seed = KvRandom::FIRST_SEED + tid as u64;
                            let mut rng = KvRandom::new(seed);

                            // === WARMUP PHASE ===
                            for _ in 0..WARMUP_OPS {
                                let x = u64::from(rng.rand()) % nseqkeys;
                                let key = key_var(x);
                                black_box(tree.get_with_guard(&key, &guard));
                            }
                            warmup_done.wait();

                            // Reset RNG for measurement
                            rng = KvRandom::new(seed);

                            // === MEASUREMENT PHASE ===
                            pre_measurement_barrier();
                            start.wait();

                            for _ in 0..OPS_PER_THREAD {
                                let x = u64::from(rng.rand()) % nseqkeys;
                                let key = key_var(x);
                                let _ = tree.insert_with_guard(&key, x + 1, &guard);
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
// RW1LONG: String keys with prefixes
// =============================================================================

#[divan::bench_group(name = "rw1long", sample_count = 200)]
mod rw1long {
    use super::*;

    const FORMATS: [&str; 4] = ["user", "machine", "opening", "fartparade"];
    const OPS_PER_THREAD: usize = 25_000;

    #[divan::bench(args = [1, 2, 4, 6])]
    fn long_string_keys(bencher: Bencher, threads: usize) {
        bencher
            .counter(divan::counter::ItemsCount::new(
                threads * OPS_PER_THREAD * 2,
            ))
            .with_inputs(|| Arc::new(MassTree15Inline::<u64>::new()))
            .bench_local_values(|tree| {
                let warmup_done = Arc::new(Barrier::new(threads));
                let start = Arc::new(Barrier::new(threads));
                let handles: Vec<_> = (0..threads)
                    .map(|tid| {
                        let tree = Arc::clone(&tree);
                        let warmup_done = Arc::clone(&warmup_done);
                        let start = Arc::clone(&start);
                        thread::spawn(move || {
                            let guard = tree.guard();
                            let seed = KvRandom::FIRST_SEED + (tid % 48) as u64;
                            let mut rng = KvRandom::new(seed);

                            // === WARMUP PHASE ===
                            for _ in 0..WARMUP_OPS {
                                let x = rng.rand();
                                let fmt = rng.uniform(4) as usize;
                                let key = format!("{}{}", FORMATS[fmt], x).into_bytes();
                                black_box(tree.get_with_guard(&key, &guard));
                            }
                            warmup_done.wait();

                            // Reset RNG for measurement
                            rng = KvRandom::new(seed);

                            // === MEASUREMENT PHASE ===
                            pre_measurement_barrier();
                            start.wait();

                            // Put phase
                            let mut keys_inserted = Vec::with_capacity(OPS_PER_THREAD);
                            for _ in 0..OPS_PER_THREAD {
                                let x = rng.rand();
                                let fmt = rng.uniform(4) as usize;
                                let key = format!("{}{}", FORMATS[fmt], x).into_bytes();
                                let _ = tree.insert_with_guard(&key, u64::from(x + 1), &guard);
                                keys_inserted.push((fmt, x));
                            }

                            // Shuffle
                            for i in 0..keys_inserted.len() {
                                let j = rng.uniform(keys_inserted.len() as u32) as usize;
                                keys_inserted.swap(i, j);
                            }

                            // Get phase
                            let mut sum = 0u64;
                            for (fmt, x) in keys_inserted {
                                let key = format!("{}{}", FORMATS[fmt], x).into_bytes();
                                if let Some(v) = tree.get_with_guard(&key, &guard) {
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
                tree
            });
    }
}

// =============================================================================
// RWSMALL24: 24 keys, 87.5% reads
// =============================================================================

#[divan::bench_group(name = "rwsmall24", sample_count = 200)]
mod rwsmall24 {
    use super::*;

    const NKEYS: u32 = 24;
    const OPS_PER_THREAD: usize = 100_000;

    #[divan::bench(args = [1, 2, 4, 6])]
    fn small_keyset_mixed(bencher: Bencher, threads: usize) {
        bencher
            .counter(divan::counter::ItemsCount::new(threads * OPS_PER_THREAD))
            .with_inputs(|| Arc::new(MassTree15Inline::<u64>::new()))
            .bench_local_values(|tree| {
                let warmup_done = Arc::new(Barrier::new(threads));
                let start = Arc::new(Barrier::new(threads));
                let handles: Vec<_> = (0..threads)
                    .map(|tid| {
                        let tree = Arc::clone(&tree);
                        let warmup_done = Arc::clone(&warmup_done);
                        let start = Arc::clone(&start);
                        thread::spawn(move || {
                            let guard = tree.guard();
                            let seed = KvRandom::FIRST_SEED + (tid % 48) as u64;
                            let mut rng = KvRandom::new(seed);

                            // === WARMUP PHASE ===
                            for _ in 0..WARMUP_OPS {
                                let x = rng.uniform(NKEYS << 3);
                                let key_idx = x >> 3;
                                let key = key_var(u64::from(key_idx));
                                black_box(tree.get_with_guard(&key, &guard));
                            }
                            warmup_done.wait();

                            // Reset RNG for measurement
                            rng = KvRandom::new(seed);

                            // === MEASUREMENT PHASE ===
                            pre_measurement_barrier();
                            start.wait();

                            let mut sum = 0u64;
                            for n in 0..OPS_PER_THREAD {
                                let x = rng.uniform(NKEYS << 3);
                                let key_idx = x >> 3;
                                let key = key_var(u64::from(key_idx));

                                if (x & 7) != 0 {
                                    // 87.5% reads
                                    if let Some(v) = tree.get_with_guard(&key, &guard) {
                                        sum = sum.wrapping_add(v);
                                    }
                                } else {
                                    // 12.5% writes
                                    let _ = tree.insert_with_guard(&key, n as u64, &guard);
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
}
