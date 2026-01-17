//! Profile rw1 workload (random inserts + shuffled lookups)
//!
//! Usage:
//! ```bash
//! # Build with profiling symbols
//! cargo build --profile profiling --example profile_rw1 --features mimalloc
//!
//! # Run with perf (single-threaded for cleaner profile)
//! perf record -g --call-graph dwarf -F 997 ./target/profiling/examples/profile_rw1
//!
//! # Multi-threaded (matches rw1 workload)
//! perf record -g --call-graph dwarf -F 997 ./target/profiling/examples/profile_rw1 -j 12
//!
//! # Analyze
//! perf report --stdio --sort symbol -g none --percent-limit 2
//!
//! # Flamegraph
//! perf script | stackcollapse-perf.pl | flamegraph.pl > rw1.svg
//!
//! # Heap profiling with dhat
//! cargo build --profile profiling --example profile_rw1 --features dhat-heap
//! ./target/profiling/examples/profile_rw1 -j 1 -n 1000000
//! # Opens dhat-heap.json in https://nnethercote.github.io/dh_view/dh_view.html
//! ```

#![allow(
    clippy::unwrap_used,
    clippy::indexing_slicing,
    reason = "fail fast in profiling"
)]
#![expect(clippy::cast_precision_loss)]

// =============================================================================
// Conditional dhat heap profiler
// =============================================================================

#[cfg(feature = "dhat-heap")]
#[global_allocator]
static ALLOC: dhat::Alloc = dhat::Alloc;

use masstree::MassTree15Inline;
use std::hint::black_box;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Barrier};
use std::thread;
use std::time::{Duration, Instant};

// =============================================================================
// KvRandom: C++ compatible LCG (from kvrandom.hh)
// =============================================================================

struct KvRandom {
    seed: u32,
}

impl KvRandom {
    const A: u32 = 1_664_525;
    const C: u32 = 1_013_904_223;
    const FIRST_SEED: u64 = 31949;

    #[inline]
    const fn new(seed: u64) -> Self {
        Self { seed: seed as u32 }
    }

    #[inline]
    const fn lcg_step(&mut self) -> u32 {
        self.seed = self.seed.wrapping_mul(Self::A).wrapping_add(Self::C);
        self.seed
    }

    #[inline]
    const fn rand(&mut self) -> u32 {
        self.lcg_step();
        let x0 = self.lcg_step();
        self.lcg_step();
        let x1 = self.lcg_step();
        (x0 >> 15) | ((x1 & 0x7FFE) << 16)
    }

    #[inline]
    const fn uniform(&mut self, max: u32) -> u32 {
        self.rand() % max
    }
}

// =============================================================================
// QuickIstr: Stack-based key generation (matches C++ misc.hh)
// =============================================================================

#[derive(Clone, Copy)]
struct QuickIstr {
    buf: [u8; 32],
    start: usize,
}

impl QuickIstr {
    #[inline]
    const fn new(mut x: u64) -> Self {
        let mut buf = [0u8; 32];
        let mut pos = 31;
        loop {
            buf[pos] = b'0' + (x % 10) as u8;
            x /= 10;
            if x == 0 {
                break;
            }
            pos -= 1;
        }
        Self { buf, start: pos }
    }

    #[inline]
    fn as_bytes(&self) -> &[u8] {
        &self.buf[self.start..32]
    }
}

static TIMEOUT: AtomicBool = AtomicBool::new(false);

fn main() {
    // Initialize dhat heap profiler if feature enabled
    // Profiler writes dhat-heap.json on drop
    #[cfg(feature = "dhat-heap")]
    let _profiler = dhat::Profiler::new_heap();

    let args: Vec<String> = std::env::args().collect();

    // Parse arguments
    let mut threads = 1;
    let mut duration_secs = 10;
    let mut ops: Option<u64> = None;
    let mut put_only = false;
    let mut get_only = false;

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "-j" | "--threads" => {
                i += 1;
                threads = args[i].parse().unwrap_or(1);
            }
            "-d" | "--duration" => {
                i += 1;
                duration_secs = args[i].parse().unwrap_or(10);
            }
            "-n" | "--ops" => {
                i += 1;
                ops = Some(args[i].parse().unwrap_or(10_000_000));
            }
            "--put-only" => {
                put_only = true;
            }
            "--get-only" => {
                get_only = true;
            }
            _ => {}
        }
        i += 1;
    }

    println!("=== rw1 profiling ===");
    println!("Threads: {threads}");

    if let Some(n) = ops {
        println!("Mode: Fixed ops ({n} per thread)");
        if put_only {
            println!("Phase: PUT only");
        } else if get_only {
            println!("Phase: GET only");
        } else {
            println!("Phase: PUT + GET");
        }
        run_fixed_ops(threads, n, put_only, get_only);
    } else {
        println!("Mode: Duration-based ({duration_secs}s)");
        if put_only {
            println!("Phase: PUT only");
        } else if get_only {
            println!("Phase: GET only");
        } else {
            println!("Phase: PUT + GET");
        }
        run_duration_based(
            threads,
            Duration::from_secs(duration_secs),
            put_only,
            get_only,
        );
    }
}

/// Run for fixed number of operations (better for profiling)
#[expect(clippy::similar_names)]
fn run_fixed_ops(threads: usize, ops_per_thread: u64, put_only: bool, get_only: bool) {
    let tree = Arc::new(MassTree15Inline::<u64>::new());
    let barrier = Arc::new(Barrier::new(threads));
    let get_barrier = Arc::new(Barrier::new(threads));

    let start = Instant::now();

    let handles: Vec<_> = (0..threads)
        .map(|tid| {
            let tree = Arc::clone(&tree);
            let barrier = Arc::clone(&barrier);
            let get_barrier = Arc::clone(&get_barrier);
            thread::spawn(move || {
                // Pin to core if possible
                if let Some(core_ids) = core_affinity::get_core_ids()
                    && !core_ids.is_empty()
                {
                    let core = core_ids[tid % core_ids.len()];
                    core_affinity::set_for_current(core);
                }

                let seed = KvRandom::FIRST_SEED + (tid % 48) as u64;
                let mut rng = KvRandom::new(seed);

                barrier.wait();

                // PUT phase
                let mut puts = 0u64;
                let put_start = Instant::now();

                if !get_only {
                    let guard = tree.guard();
                    for _ in 0..ops_per_thread {
                        let x = rng.rand();
                        let key = QuickIstr::new(u64::from(x));
                        let _ = black_box(tree.insert_with_guard(
                            key.as_bytes(),
                            u64::from(x + 1),
                            &guard,
                        ));
                        puts += 1;
                    }
                }
                let put_time = put_start.elapsed();

                get_barrier.wait();

                // GET phase - regenerate keys from seed, shuffle, lookup
                let mut gets = 0u64;
                let get_start = Instant::now();

                if !put_only {
                    // Regenerate keys deterministically
                    let mut rng = KvRandom::new(seed);
                    let count = if get_only { ops_per_thread } else { puts };
                    let mut keys: Vec<u32> = (0..count).map(|_| rng.rand()).collect();

                    // Shuffle using C++ pattern
                    for i in 0..keys.len() {
                        let j = rng.uniform(keys.len() as u32) as usize;
                        keys.swap(i, j);
                    }

                    let guard = tree.guard();
                    let mut sum = 0u64;
                    for x in &keys {
                        let key = QuickIstr::new(u64::from(*x));
                        if let Some(v) = tree.get_with_guard(key.as_bytes(), &guard) {
                            sum = sum.wrapping_add(v);
                        }
                        gets += 1;
                    }
                    black_box(sum);
                }
                let get_time = get_start.elapsed();

                (puts, gets, put_time, get_time)
            })
        })
        .collect();

    let mut total_puts = 0u64;
    let mut total_gets = 0u64;
    let mut total_put_time = Duration::ZERO;
    let mut total_get_time = Duration::ZERO;

    for h in handles {
        let (puts, gets, put_time, get_time) = h.join().unwrap();
        total_puts += puts;
        total_gets += gets;
        total_put_time = total_put_time.max(put_time);
        total_get_time = total_get_time.max(get_time);
    }

    let elapsed = start.elapsed();

    if total_puts > 0 {
        let put_mops = total_puts as f64 / total_put_time.as_secs_f64() / 1_000_000.0;
        println!(
            "PUT: {} ops in {:.2}s = {:.2} Mops/s",
            total_puts,
            total_put_time.as_secs_f64(),
            put_mops
        );
    }

    if total_gets > 0 {
        let get_mops = total_gets as f64 / total_get_time.as_secs_f64() / 1_000_000.0;
        println!(
            "GET: {} ops in {:.2}s = {:.2} Mops/s",
            total_gets,
            total_get_time.as_secs_f64(),
            get_mops
        );
    }

    let total_ops = total_puts + total_gets;
    let total_mops = total_ops as f64 / elapsed.as_secs_f64() / 1_000_000.0;
    println!(
        "Total: {} ops in {:.2}s = {:.2} Mops/s",
        total_ops,
        elapsed.as_secs_f64(),
        total_mops
    );
}

/// Run for fixed duration (matches mttest behavior)
#[expect(clippy::similar_names)]
fn run_duration_based(threads: usize, duration: Duration, put_only: bool, get_only: bool) {
    let tree = Arc::new(MassTree15Inline::<u64>::new());
    let barrier = Arc::new(Barrier::new(threads));
    let get_barrier = Arc::new(Barrier::new(threads));

    // Timer thread for put phase
    TIMEOUT.store(false, Ordering::Relaxed);
    thread::spawn(move || {
        thread::sleep(duration);
        TIMEOUT.store(true, Ordering::Relaxed);
    });

    let start = Instant::now();

    let handles: Vec<_> = (0..threads)
        .map(|tid| {
            let tree = Arc::clone(&tree);
            let barrier = Arc::clone(&barrier);
            let get_barrier = Arc::clone(&get_barrier);
            thread::spawn(move || {
                // Pin to core if possible
                if let Some(core_ids) = core_affinity::get_core_ids()
                    && !core_ids.is_empty()
                {
                    let core = core_ids[tid % core_ids.len()];
                    core_affinity::set_for_current(core);
                }

                let seed = KvRandom::FIRST_SEED + (tid % 48) as u64;
                let mut rng = KvRandom::new(seed);

                barrier.wait();

                // PUT phase - timed
                let mut puts = 0u64;
                let put_start = Instant::now();

                if !get_only {
                    let guard = tree.guard();
                    while !TIMEOUT.load(Ordering::Relaxed) {
                        let x = rng.rand();
                        let key = QuickIstr::new(u64::from(x));
                        let _ = black_box(tree.insert_with_guard(
                            key.as_bytes(),
                            u64::from(x + 1),
                            &guard,
                        ));
                        puts += 1;
                    }
                }
                let put_time = put_start.elapsed();

                get_barrier.wait();

                // GET phase - fixed iterations (puts count)
                let mut gets = 0u64;
                let get_start = Instant::now();

                if !put_only && puts > 0 {
                    // Regenerate keys deterministically
                    let mut rng = KvRandom::new(seed);
                    let mut keys: Vec<u32> = (0..puts).map(|_| rng.rand()).collect();

                    // Shuffle using C++ pattern
                    for i in 0..keys.len() {
                        let j = rng.uniform(keys.len() as u32) as usize;
                        keys.swap(i, j);
                    }

                    let guard = tree.guard();
                    let mut sum = 0u64;
                    for x in &keys {
                        let key = QuickIstr::new(u64::from(*x));
                        if let Some(v) = tree.get_with_guard(key.as_bytes(), &guard) {
                            sum = sum.wrapping_add(v);
                        }
                        gets += 1;
                    }
                    black_box(sum);
                }
                let get_time = get_start.elapsed();

                (puts, gets, put_time, get_time)
            })
        })
        .collect();

    let mut total_puts = 0u64;
    let mut total_gets = 0u64;
    let mut total_put_time = Duration::ZERO;
    let mut total_get_time = Duration::ZERO;

    for h in handles {
        let (puts, gets, put_time, get_time) = h.join().unwrap();
        total_puts += puts;
        total_gets += gets;
        total_put_time = total_put_time.max(put_time);
        total_get_time = total_get_time.max(get_time);
    }

    let elapsed = start.elapsed();

    if total_puts > 0 {
        let put_mops = total_puts as f64 / total_put_time.as_secs_f64() / 1_000_000.0;
        println!(
            "PUT: {} ops in {:.2}s = {:.2} Mops/s",
            total_puts,
            total_put_time.as_secs_f64(),
            put_mops
        );
    }

    if total_gets > 0 {
        let get_mops = total_gets as f64 / total_get_time.as_secs_f64() / 1_000_000.0;
        println!(
            "GET: {} ops in {:.2}s = {:.2} Mops/s",
            total_gets,
            total_get_time.as_secs_f64(),
            get_mops
        );
    }

    let total_ops = total_puts + total_gets;
    let total_mops = total_ops as f64 / elapsed.as_secs_f64() / 1_000_000.0;
    println!(
        "Total: {} ops in {:.2}s = {:.2} Mops/s",
        total_ops,
        elapsed.as_secs_f64(),
        total_mops
    );
}
