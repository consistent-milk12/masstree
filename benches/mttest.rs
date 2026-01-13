//! C++ mttest-compatible benchmark
//!
//! Runs exactly like C++ mttest: single run, direct output, no statistical sampling.
//!
//! ```bash
//! cargo bench --bench mttest --features mimalloc
//! cargo bench --bench mttest --features mimalloc -- rw3 -j6
//! cargo bench --bench mttest --features mimalloc -- all -j6 -d10 --save
//! ```

#![allow(clippy::cast_possible_truncation)]
#![allow(clippy::cast_sign_loss)]
#![allow(clippy::indexing_slicing)]
#![allow(clippy::unwrap_used)]
#![allow(clippy::inline_always)]
#![allow(clippy::map_unwrap_or)]
#![allow(clippy::cast_precision_loss)]
#![allow(clippy::too_many_arguments, clippy::too_many_lines)]
#![expect(clippy::struct_excessive_bools)]

use clap::Parser;
use core_affinity::CoreId;
use masstree::MassTree15Inline;
use serde::Serialize;
use std::fs;
use std::io::Write;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Barrier};
use std::thread;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

// =============================================================================
// Global Timeout Flag (matches C++ volatile bool timeout[2])
// =============================================================================

const C: u32 = 2_654_435_761;

/// Global timeout flag, set by timer thread (matches C++ SIGALRM approach).
/// Using a single flag since most benchmarks only need put-phase timeout.
static TIMEOUT: AtomicBool = AtomicBool::new(false);

/// Generation counter to prevent stale timer threads from affecting later benchmarks.
/// Each new benchmark increments this; timer threads only set TIMEOUT if their
/// generation matches the current one.
static TIMEOUT_GENERATION: AtomicU64 = AtomicU64::new(0);

/// Reset timeout flag and increment generation before each benchmark.
/// This invalidates any timer threads from previous benchmarks.
fn reset_timeout() {
    TIMEOUT.store(false, Ordering::Relaxed);
    TIMEOUT_GENERATION.fetch_add(1, Ordering::Relaxed);
}

/// Check if benchmark should stop (near-zero overhead - single atomic load)
#[inline(always)]
fn timed_out() -> bool {
    TIMEOUT.load(Ordering::Relaxed)
}

/// Spawn a timer thread that sets the timeout flag after `duration`.
/// Uses generation-based invalidation to prevent stale timers from affecting
/// later benchmarks if the current one finishes early.
fn start_timeout_timer(duration: Duration) {
    let generation = TIMEOUT_GENERATION.load(Ordering::Relaxed);
    thread::spawn(move || {
        thread::sleep(duration);
        // Only set timeout if this timer's generation is still current
        if TIMEOUT_GENERATION.load(Ordering::Relaxed) == generation {
            TIMEOUT.store(true, Ordering::Relaxed);
        }
    });
}

// =============================================================================
// Thread Pinning
// =============================================================================

/// Pin current thread to a specific core (if pinning is enabled)
fn pin_thread(tid: usize, core_ids: &[CoreId]) {
    if !core_ids.is_empty() {
        let core_id = core_ids[tid % core_ids.len()];
        core_affinity::set_for_current(core_id);
    }
}

// =============================================================================
// CLI Arguments
// =============================================================================

#[derive(Parser, Debug)]
#[command(
    name = "mttest",
    about = "C++ mttest-compatible Masstree benchmark",
    ignore_errors = true
)]
struct Args {
    /// Number of threads (default: number of CPU cores)
    #[arg(short = 'j', long = "threads")]
    threads: Option<usize>,

    /// Duration in seconds (default: 10, matching C++ mttest)
    #[arg(short = 'd', long = "duration", default_value = "10")]
    duration: u64,

    /// Operation limit per thread (default: unlimited)
    #[arg(short = 'l', long = "limit")]
    limit: Option<u64>,

    /// Number of trials to run (default: 1)
    #[arg(short = 'T', long = "trials", default_value = "1")]
    trials: usize,

    /// Pin threads to cores
    #[arg(short = 'p', long = "pin")]
    pin: bool,

    /// Quiet mode - less verbose output
    #[arg(short = 'q', long = "quiet")]
    quiet: bool,

    /// Enable validation mode (report read mismatches, like C++ `get_check`)
    /// Note: May report errors due to known tree bugs at high scale
    #[arg(long = "check")]
    check: bool,

    /// Output JSON results (C++ notebook format) to stdout
    #[arg(long = "json")]
    json: bool,

    /// Save results to JSON file in runs/ directory
    #[arg(long = "save", short = 's')]
    save: bool,

    /// Custom output file path (overrides --save default)
    #[arg(long = "output", short = 'o')]
    output: Option<PathBuf>,

    /// Tests to run (rw1, rw2, rw2g90, rw2g98, rw3, rw4, same, uscale, wscale, all)
    #[arg(default_value = "all")]
    tests: Vec<String>,
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
    const fn with_minlen(mut x: u64, minlen: usize) -> Self {
        let mut buf = [0u8; 32];
        let mut pos = 31;
        let mut len = 0;
        loop {
            buf[pos] = b'0' + (x % 10) as u8;
            x /= 10;
            len += 1;
            if x == 0 && len >= minlen {
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

/// Returns a key with minimum 8 decimal digits (grows for n >= 100000000).
/// This matches C++ `quick_istr(n, 8)` which uses min-width, not fixed-width.
#[inline]
const fn key8(n: u64) -> QuickIstr {
    QuickIstr::with_minlen(n, 8)
}

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
    fn bernoulli(&mut self, p: f64) -> bool {
        (f64::from(self.rand()) / f64::from(0x7FFF_FFFF)) < p
    }

    #[inline]
    const fn uniform(&mut self, max: u32) -> u32 {
        self.rand() % max
    }
}

// =============================================================================
// Result reporting (matches C++ JSON output format)
// =============================================================================

#[derive(Default, Clone, Serialize)]
struct ThreadResult {
    puts: u64,
    gets: u64,
    put_time: f64,
    get_time: f64,
}

/// Results for a single benchmark test
#[derive(Clone, Serialize)]
struct BenchmarkResult {
    name: String,
    threads: usize,
    duration_secs: u64,
    thread_results: Vec<ThreadResult>,
    total_puts: u64,
    total_gets: u64,
    total_ops: u64,
    puts_per_sec: f64,
    gets_per_sec: f64,
    ops_per_sec: f64,
}

/// All results from a benchmark run
#[derive(Serialize)]
struct RunResults {
    timestamp: String,
    rust_version: String,
    threads: usize,
    duration_secs: u64,
    pinned: bool,
    check_enabled: bool,
    benchmarks: Vec<BenchmarkResult>,
}

impl RunResults {
    fn new(threads: usize, duration_secs: u64, pinned: bool, check: bool) -> Self {
        let timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| {
                // Format as ISO 8601
                let secs = d.as_secs();
                let days_since_epoch = secs / 86400;
                let secs_today = secs % 86400;
                let hours = secs_today / 3600;
                let mins = (secs_today % 3600) / 60;
                let secs = secs_today % 60;
                // Approximate date calculation (not accounting for leap years perfectly)
                let years = 1970 + days_since_epoch / 365;
                let remaining_days = days_since_epoch % 365;
                let month = remaining_days / 30 + 1;
                let day = remaining_days % 30 + 1;
                format!("{years:04}-{month:02}-{day:02}T{hours:02}:{mins:02}:{secs:02}Z")
            })
            .unwrap_or_else(|_| "unknown".to_string());

        Self {
            timestamp,
            rust_version: env!("CARGO_PKG_VERSION").to_string(),
            threads,
            duration_secs,
            pinned,
            check_enabled: check,
            benchmarks: Vec::new(),
        }
    }

    fn add_benchmark(&mut self, name: &str, thread_results: Vec<ThreadResult>) {
        let mut total_puts = 0u64;
        let mut total_gets = 0u64;
        let mut total_put_rate = 0.0;
        let mut total_get_rate = 0.0;
        let mut total_ops_rate = 0.0;

        for r in &thread_results {
            let put_rate = if r.put_time > 0.0 {
                r.puts as f64 / r.put_time
            } else {
                0.0
            };
            let get_rate = if r.get_time > 0.0 {
                r.gets as f64 / r.get_time
            } else {
                0.0
            };
            let ops = r.puts + r.gets;
            let total_time = r.put_time + r.get_time;
            let ops_rate = if total_time > 0.0 {
                ops as f64 / total_time
            } else {
                0.0
            };

            total_puts += r.puts;
            total_gets += r.gets;
            total_put_rate += put_rate;
            total_get_rate += get_rate;
            total_ops_rate += ops_rate;
        }

        self.benchmarks.push(BenchmarkResult {
            name: name.to_string(),
            threads: self.threads,
            duration_secs: self.duration_secs,
            thread_results,
            total_puts,
            total_gets,
            total_ops: total_puts + total_gets,
            puts_per_sec: total_put_rate,
            gets_per_sec: total_get_rate,
            ops_per_sec: total_ops_rate,
        });
    }
}

/// Output mode for results
static mut OUTPUT_JSON: bool = false;
static mut CURRENT_TRIAL: usize = 0;

fn print_results(test: &str, threads: usize, results: &[ThreadResult]) {
    // SAFETY: Only accessed from single-threaded main context
    let json_mode = unsafe { OUTPUT_JSON };
    let trial = unsafe { CURRENT_TRIAL };

    if json_mode {
        // C++ compatible JSON output (one line per thread)
        for (tid, r) in results.iter().enumerate() {
            let put_rate = if r.put_time > 0.0 {
                r.puts as f64 / r.put_time
            } else {
                0.0
            };
            let get_rate = if r.get_time > 0.0 {
                r.gets as f64 / r.get_time
            } else {
                0.0
            };
            let ops = r.puts + r.gets;
            let total_time = r.put_time + r.get_time;
            let ops_rate = if total_time > 0.0 {
                ops as f64 / total_time
            } else {
                0.0
            };

            println!(
                r#"{{"table":"masstree","test":"{}","trial":{},"thread":{},"puts":{},"puts_per_sec":{:.0},"gets":{},"gets_per_sec":{:.0},"ops":{},"ops_per_sec":{:.0}}}"#,
                test, trial, tid, r.puts, put_rate, r.gets, get_rate, ops, ops_rate
            );
        }
    } else {
        // Human-readable table format
        println!("\n{test} with {threads} threads:");
        println!(
            "{:>8} {:>12} {:>14} {:>12} {:>14} {:>12} {:>14}",
            "thread", "puts", "puts/sec", "gets", "gets/sec", "ops", "ops/sec"
        );
        println!("{}", "-".repeat(90));

        let mut total_puts = 0u64;
        let mut total_gets = 0u64;
        let mut total_put_rate = 0.0;
        let mut total_get_rate = 0.0;
        let mut total_ops_rate = 0.0;

        for (tid, r) in results.iter().enumerate() {
            let put_rate = if r.put_time > 0.0 {
                r.puts as f64 / r.put_time
            } else {
                0.0
            };
            let get_rate = if r.get_time > 0.0 {
                r.gets as f64 / r.get_time
            } else {
                0.0
            };
            let ops = r.puts + r.gets;
            let total_time = r.put_time + r.get_time;
            let ops_rate = if total_time > 0.0 {
                ops as f64 / total_time
            } else {
                0.0
            };

            println!(
                "{:>8} {:>12} {:>14.0} {:>12} {:>14.0} {:>12} {:>14.0}",
                tid, r.puts, put_rate, r.gets, get_rate, ops, ops_rate
            );

            total_puts += r.puts;
            total_gets += r.gets;
            total_put_rate += put_rate;
            total_get_rate += get_rate;
            total_ops_rate += ops_rate;
        }

        println!("{}", "-".repeat(90));
        println!(
            "{:>8} {:>12} {:>14.0} {:>12} {:>14.0} {:>12} {:>14.0}",
            "TOTAL",
            total_puts,
            total_put_rate,
            total_gets,
            total_get_rate,
            total_puts + total_gets,
            total_ops_rate
        );
    }
}

// =============================================================================
// Benchmarks
// =============================================================================

fn bench_rw1(
    threads: usize,
    duration: Duration,
    limit: u64,
    check: bool,
    core_ids: &Arc<Vec<CoreId>>,
) -> Vec<ThreadResult> {
    let tree = Arc::new(MassTree15Inline::<u64>::new());
    let start_barrier = Arc::new(Barrier::new(threads));
    let get_barrier = Arc::new(Barrier::new(threads)); // Barrier between put and get phases (C++ wait_all)
    let results: Arc<Vec<_>> = Arc::new(
        (0..threads)
            .map(|_| {
                (
                    AtomicU64::new(0),
                    AtomicU64::new(0),
                    AtomicU64::new(0),
                    AtomicU64::new(0),
                )
            })
            .collect(),
    );

    // Reset and start timeout timer (matches C++ SIGALRM approach)
    reset_timeout();
    start_timeout_timer(duration);

    let handles: Vec<_> = (0..threads)
        .map(|tid| {
            let tree = Arc::clone(&tree);
            let start_barrier = Arc::clone(&start_barrier);
            let get_barrier = Arc::clone(&get_barrier);
            let results = Arc::clone(&results);
            let core_ids = Arc::clone(core_ids);
            thread::spawn(move || {
                pin_thread(tid, &core_ids);
                let guard = tree.guard();
                let seed = KvRandom::FIRST_SEED + (tid % 48) as u64;
                let mut rng = KvRandom::new(seed);

                start_barrier.wait();

                // Put phase - use atomic timeout flag instead of Instant::now()
                let put_start = Instant::now();
                let mut keys = Vec::with_capacity(1_000_000);

                while !timed_out() && (keys.len() as u64) <= limit {
                    let x = rng.rand();
                    let key = QuickIstr::new(u64::from(x));
                    let _ = tree.insert_with_guard(key.as_bytes(), u64::from(x + 1), &guard);
                    keys.push(x);
                }
                let put_time = put_start.elapsed().as_secs_f64();
                let puts = keys.len() as u64;

                // C++ wait_all() - barrier between put and get phases
                get_barrier.wait();

                // Re-seed and regenerate keys, then shuffle (matching C++ exactly)
                let mut rng = KvRandom::new(seed);
                keys.clear();
                for _ in 0..puts {
                    keys.push(rng.rand());
                }
                // Shuffle: swap a[i] with a[uniform(0..n-1)]
                for i in 0..keys.len() {
                    let j = rng.uniform(keys.len() as u32) as usize;
                    keys.swap(i, j);
                }

                // Get phase - exactly `puts` iterations, no timeout (matches C++)
                let get_start = Instant::now();
                let mut sum = 0u64;
                let mut check_errors = 0u64;
                for x in &keys {
                    let key = QuickIstr::new(u64::from(*x));
                    let expected = u64::from(*x + 1);
                    if let Some(v) = tree.get_with_guard(key.as_bytes(), &guard) {
                        if check && v != expected {
                            check_errors += 1;
                        }
                        sum = sum.wrapping_add(v);
                    } else if check {
                        check_errors += 1;
                    }
                }
                if check && check_errors > 0 {
                    eprintln!("rw1 thread {tid}: {check_errors} check errors");
                }
                let get_time = get_start.elapsed().as_secs_f64();
                std::hint::black_box(sum);

                results[tid].0.store(puts, Ordering::Relaxed);
                results[tid].1.store(puts, Ordering::Relaxed); // gets = puts
                results[tid]
                    .2
                    .store((put_time * 1e9) as u64, Ordering::Relaxed);
                results[tid]
                    .3
                    .store((get_time * 1e9) as u64, Ordering::Relaxed);
            })
        })
        .collect();

    for h in handles {
        h.join().unwrap();
    }

    let thread_results: Vec<_> = results
        .iter()
        .map(|(p, g, pt, gt)| ThreadResult {
            puts: p.load(Ordering::Relaxed),
            gets: g.load(Ordering::Relaxed),
            put_time: pt.load(Ordering::Relaxed) as f64 / 1e9,
            get_time: gt.load(Ordering::Relaxed) as f64 / 1e9,
        })
        .collect();

    print_results("rw1", threads, &thread_results);
    thread_results
}

fn bench_rw2(
    threads: usize,
    duration: Duration,
    get_frac: f64,
    name: &str,
    limit: u64,
    check: bool,
    core_ids: &Arc<Vec<CoreId>>,
) -> Vec<ThreadResult> {
    let tree = Arc::new(MassTree15Inline::<u64>::new());
    let barrier = Arc::new(Barrier::new(threads));
    let results: Arc<Vec<_>> = Arc::new(
        (0..threads)
            .map(|_| (AtomicU64::new(0), AtomicU64::new(0), AtomicU64::new(0)))
            .collect(),
    );

    // Reset and start timeout timer (matches C++ SIGALRM approach)
    reset_timeout();
    start_timeout_timer(duration);

    let handles: Vec<_> = (0..threads)
        .map(|tid| {
            let tree = Arc::clone(&tree);
            let barrier = Arc::clone(&barrier);
            let results = Arc::clone(&results);
            let name = name.to_string();
            let core_ids = Arc::clone(core_ids);
            thread::spawn(move || {
                pin_thread(tid, &core_ids);
                let guard = tree.guard();
                let seed = KvRandom::FIRST_SEED + (tid % 48) as u64;
                let mut rng = KvRandom::new(seed);
                let offset = rng.rand();

                barrier.wait();

                let start = Instant::now();
                let mut puts = 0u64;
                let mut gets = 0u64;
                let mut sum = 0u64;

                let mut check_errors = 0u64;
                while !timed_out() && (puts + gets) <= limit {
                    if puts == 0 || !rng.bernoulli(get_frac) {
                        let x = (offset.wrapping_add(puts as u32)).wrapping_mul(C);
                        let key = QuickIstr::new(u64::from(x));
                        let _ = tree.insert_with_guard(key.as_bytes(), u64::from(x + 1), &guard);
                        puts += 1;
                    } else {
                        let idx = rng.uniform(puts as u32);
                        let x = (offset.wrapping_add(idx)).wrapping_mul(C);
                        let key = QuickIstr::new(u64::from(x));
                        let expected = u64::from(x + 1);
                        if let Some(v) = tree.get_with_guard(key.as_bytes(), &guard) {
                            if check && v != expected {
                                check_errors += 1;
                            }
                            sum = sum.wrapping_add(v);
                        } else if check {
                            check_errors += 1;
                        }
                        gets += 1;
                    }
                }
                if check && check_errors > 0 {
                    eprintln!("{name} thread {tid}: {check_errors} check errors");
                }
                let elapsed = start.elapsed().as_secs_f64();
                std::hint::black_box(sum);

                results[tid].0.store(puts, Ordering::Relaxed);
                results[tid].1.store(gets, Ordering::Relaxed);
                results[tid]
                    .2
                    .store((elapsed * 1e9) as u64, Ordering::Relaxed);
            })
        })
        .collect();

    for h in handles {
        h.join().unwrap();
    }

    let thread_results: Vec<_> = results
        .iter()
        .map(|(p, g, t)| {
            let total_time = t.load(Ordering::Relaxed) as f64 / 1e9;
            let puts = p.load(Ordering::Relaxed);
            let gets = g.load(Ordering::Relaxed);
            let total_ops = puts + gets;
            // Proportionally split time based on operation counts
            // This is an approximation since puts/gets have different costs,
            // but it's better than reporting 0.0 for get_time
            let (put_time, get_time) = if total_ops > 0 {
                let put_frac = puts as f64 / total_ops as f64;
                (total_time * put_frac, total_time * (1.0 - put_frac))
            } else {
                (total_time, 0.0)
            };
            ThreadResult {
                puts,
                gets,
                put_time,
                get_time,
            }
        })
        .collect();

    print_results(name, threads, &thread_results);
    thread_results
}

fn bench_rw3(
    threads: usize,
    duration: Duration,
    limit: u64,
    check: bool,
    core_ids: &Arc<Vec<CoreId>>,
) -> Vec<ThreadResult> {
    let tree = Arc::new(MassTree15Inline::<u64>::new());
    let barrier = Arc::new(Barrier::new(threads));
    let get_barrier = Arc::new(Barrier::new(threads)); // Barrier between put and get phases
    let results: Arc<Vec<_>> = Arc::new(
        (0..threads)
            .map(|_| (AtomicU64::new(0), AtomicU64::new(0), AtomicU64::new(0)))
            .collect(),
    );

    // Reset and start timeout timer (matches C++ SIGALRM approach)
    reset_timeout();
    start_timeout_timer(duration);

    let handles: Vec<_> = (0..threads)
        .map(|tid| {
            let tree = Arc::clone(&tree);
            let barrier = Arc::clone(&barrier);
            let get_barrier = Arc::clone(&get_barrier);
            let results = Arc::clone(&results);
            let core_ids = Arc::clone(core_ids);
            thread::spawn(move || {
                pin_thread(tid, &core_ids);
                let guard = tree.guard();

                barrier.wait();

                // Put phase - sequential keys, duration-bounded via timeout flag
                let put_start = Instant::now();
                let mut n = 0u64;

                while !timed_out() && n <= limit {
                    let key = key8(n);
                    let _ = tree.insert_with_guard(key.as_bytes(), n + 1, &guard);
                    n += 1;
                }
                let put_time = put_start.elapsed().as_secs_f64();

                // Sync point (matches C++ wait_all between phases)
                get_barrier.wait();

                // Get phase - POINT LOOKUPS, exactly n iterations (matches C++ exactly)
                // C++ does: for (unsigned i = 0; i < n; ++i) { client.get_check_key8(i, i + 1); }
                let get_start = Instant::now();
                let mut sum = 0u64;
                let mut check_errors = 0u64;

                for i in 0..n {
                    let key = key8(i);
                    let expected = i + 1;
                    if let Some(v) = tree.get_with_guard(key.as_bytes(), &guard) {
                        if check && v != expected {
                            check_errors += 1;
                        }
                        sum = sum.wrapping_add(v);
                    } else if check {
                        check_errors += 1;
                    }
                }
                if check && check_errors > 0 {
                    eprintln!("rw3 thread {tid}: {check_errors} check errors");
                }
                let get_time = get_start.elapsed().as_secs_f64();
                std::hint::black_box(sum);

                results[tid].0.store(n, Ordering::Relaxed);
                results[tid]
                    .1
                    .store((put_time * 1e9) as u64, Ordering::Relaxed);
                results[tid]
                    .2
                    .store((get_time * 1e9) as u64, Ordering::Relaxed);
            })
        })
        .collect();

    for h in handles {
        h.join().unwrap();
    }

    let thread_results: Vec<_> = results
        .iter()
        .map(|(n, pt, gt)| {
            let puts = n.load(Ordering::Relaxed);
            ThreadResult {
                puts,
                gets: puts,
                put_time: pt.load(Ordering::Relaxed) as f64 / 1e9,
                get_time: gt.load(Ordering::Relaxed) as f64 / 1e9,
            }
        })
        .collect();

    print_results("rw3", threads, &thread_results);
    thread_results
}

fn bench_rw4(
    threads: usize,
    duration: Duration,
    limit: u64,
    check: bool,
    core_ids: &Arc<Vec<CoreId>>,
) -> Vec<ThreadResult> {
    const TOP: u64 = 2_147_483_647;

    let tree = Arc::new(MassTree15Inline::<u64>::new());
    let barrier = Arc::new(Barrier::new(threads));
    let get_barrier = Arc::new(Barrier::new(threads)); // Barrier between put and get phases
    let results: Arc<Vec<_>> = Arc::new(
        (0..threads)
            .map(|_| (AtomicU64::new(0), AtomicU64::new(0), AtomicU64::new(0)))
            .collect(),
    );

    // Reset and start timeout timer (matches C++ SIGALRM approach)
    reset_timeout();
    start_timeout_timer(duration);

    let handles: Vec<_> = (0..threads)
        .map(|tid| {
            let tree = Arc::clone(&tree);
            let barrier = Arc::clone(&barrier);
            let get_barrier = Arc::clone(&get_barrier);
            let results = Arc::clone(&results);
            let core_ids = Arc::clone(core_ids);
            thread::spawn(move || {
                pin_thread(tid, &core_ids);
                let guard = tree.guard();

                barrier.wait();

                // Put phase - reverse sequential keys, duration-bounded via timeout flag
                let put_start = Instant::now();
                let mut n = 0u64;

                while !timed_out() && n <= limit {
                    let key = key8(TOP - n);
                    let _ = tree.insert_with_guard(key.as_bytes(), n + 1, &guard);
                    n += 1;
                }
                let put_time = put_start.elapsed().as_secs_f64();

                // Sync point (matches C++ wait_all between phases)
                get_barrier.wait();

                // Get phase - POINT LOOKUPS in reverse order, exactly n iterations
                // C++ does: for (unsigned i = 0; i < n; ++i) { client.get_check_key8(top - i, i + 1); }
                let get_start = Instant::now();
                let mut sum = 0u64;
                let mut check_errors = 0u64;

                for i in 0..n {
                    let key = key8(TOP - i);
                    let expected = i + 1;
                    if let Some(v) = tree.get_with_guard(key.as_bytes(), &guard) {
                        if check && v != expected {
                            check_errors += 1;
                        }
                        sum = sum.wrapping_add(v);
                    } else if check {
                        check_errors += 1;
                    }
                }
                if check && check_errors > 0 {
                    eprintln!("rw4 thread {tid}: {check_errors} check errors");
                }
                let get_time = get_start.elapsed().as_secs_f64();
                std::hint::black_box(sum);

                results[tid].0.store(n, Ordering::Relaxed);
                results[tid]
                    .1
                    .store((put_time * 1e9) as u64, Ordering::Relaxed);
                results[tid]
                    .2
                    .store((get_time * 1e9) as u64, Ordering::Relaxed);
            })
        })
        .collect();

    for h in handles {
        h.join().unwrap();
    }

    let thread_results: Vec<_> = results
        .iter()
        .map(|(n, pt, gt)| {
            let puts = n.load(Ordering::Relaxed);
            ThreadResult {
                puts,
                gets: puts,
                put_time: pt.load(Ordering::Relaxed) as f64 / 1e9,
                get_time: gt.load(Ordering::Relaxed) as f64 / 1e9,
            }
        })
        .collect();

    print_results("rw4", threads, &thread_results);
    thread_results
}

fn bench_same(
    threads: usize,
    duration: Duration,
    limit: u64,
    core_ids: &Arc<Vec<CoreId>>,
) -> Vec<ThreadResult> {
    const NUM_KEYS: u32 = 10;

    let tree = Arc::new(MassTree15Inline::<u64>::new());
    let barrier = Arc::new(Barrier::new(threads));
    let results: Arc<Vec<_>> = Arc::new(
        (0..threads)
            .map(|_| (AtomicU64::new(0), AtomicU64::new(0)))
            .collect(),
    );

    // Reset and start timeout timer (matches C++ SIGALRM approach)
    reset_timeout();
    start_timeout_timer(duration);

    let handles: Vec<_> = (0..threads)
        .map(|tid| {
            let tree = Arc::clone(&tree);
            let barrier = Arc::clone(&barrier);
            let results = Arc::clone(&results);
            let core_ids = Arc::clone(core_ids);
            thread::spawn(move || {
                pin_thread(tid, &core_ids);
                let guard = tree.guard();
                let seed = KvRandom::FIRST_SEED + (tid % 48) as u64;
                let mut rng = KvRandom::new(seed);

                barrier.wait();

                let start = Instant::now();
                let mut n = 0u64;

                while !timed_out() && n <= limit {
                    let x = rng.uniform(NUM_KEYS);
                    let key = QuickIstr::new(u64::from(x));
                    let _ = tree.insert_with_guard(key.as_bytes(), u64::from(x + 1), &guard);
                    n += 1;
                }
                let elapsed = start.elapsed().as_secs_f64();

                results[tid].0.store(n, Ordering::Relaxed);
                results[tid]
                    .1
                    .store((elapsed * 1e9) as u64, Ordering::Relaxed);
            })
        })
        .collect();

    for h in handles {
        h.join().unwrap();
    }

    let thread_results: Vec<_> = results
        .iter()
        .map(|(n, t)| ThreadResult {
            puts: n.load(Ordering::Relaxed),
            gets: 0,
            put_time: t.load(Ordering::Relaxed) as f64 / 1e9,
            get_time: 0.0,
        })
        .collect();

    print_results("same", threads, &thread_results);
    thread_results
}

/// C++ `kvtest_uscale` semantics:
/// - seed = `kvtest_first_seed + tid` (NOT tid % 48)
/// - `nseqkeys = 16 * ruscale_partsz = 140,000,000`
/// - NO pre-population - writes to empty tree
fn bench_uscale(
    threads: usize,
    duration: Duration,
    core_ids: &Arc<Vec<CoreId>>,
) -> Vec<ThreadResult> {
    // C++ constants: ruscale_partsz = (140 * 1000000) / 16, nseqkeys = 16 * ruscale_partsz
    const NSEQKEYS: u64 = 140_000_000;

    let tree = Arc::new(MassTree15Inline::<u64>::new());
    let barrier = Arc::new(Barrier::new(threads));
    let results: Arc<Vec<_>> = Arc::new(
        (0..threads)
            .map(|_| (AtomicU64::new(0), AtomicU64::new(0)))
            .collect(),
    );

    // Reset and start timeout timer (matches C++ SIGALRM approach)
    reset_timeout();
    start_timeout_timer(duration);

    let handles: Vec<_> = (0..threads)
        .map(|tid| {
            let tree = Arc::clone(&tree);
            let barrier = Arc::clone(&barrier);
            let results = Arc::clone(&results);
            let core_ids = Arc::clone(core_ids);
            thread::spawn(move || {
                pin_thread(tid, &core_ids);
                let guard = tree.guard();
                // C++: seed = kvtest_first_seed + client.id() (NOT % 48)
                let seed = KvRandom::FIRST_SEED + tid as u64;
                let mut rng = KvRandom::new(seed);

                barrier.wait();

                let start = Instant::now();
                let mut n = 0u64;

                while !timed_out() {
                    let x = u64::from(rng.rand()) % NSEQKEYS;
                    let key = QuickIstr::new(x);
                    let _ = tree.insert_with_guard(key.as_bytes(), x + 1, &guard);
                    n += 1;
                }
                let elapsed = start.elapsed().as_secs_f64();

                results[tid].0.store(n, Ordering::Relaxed);
                results[tid]
                    .1
                    .store((elapsed * 1e9) as u64, Ordering::Relaxed);
            })
        })
        .collect();

    for h in handles {
        h.join().unwrap();
    }

    let thread_results: Vec<_> = results
        .iter()
        .map(|(n, t)| ThreadResult {
            puts: n.load(Ordering::Relaxed),
            gets: 0,
            put_time: t.load(Ordering::Relaxed) as f64 / 1e9,
            get_time: 0.0,
        })
        .collect();

    print_results("uscale", threads, &thread_results);
    thread_results
}

fn bench_wscale(
    threads: usize,
    duration: Duration,
    core_ids: &Arc<Vec<CoreId>>,
) -> Vec<ThreadResult> {
    let tree = Arc::new(MassTree15Inline::<u64>::new());
    let barrier = Arc::new(Barrier::new(threads));
    let results: Arc<Vec<_>> = Arc::new(
        (0..threads)
            .map(|_| (AtomicU64::new(0), AtomicU64::new(0)))
            .collect(),
    );

    // Reset and start timeout timer (matches C++ SIGALRM approach)
    reset_timeout();
    start_timeout_timer(duration);

    let handles: Vec<_> = (0..threads)
        .map(|tid| {
            let tree = Arc::clone(&tree);
            let barrier = Arc::clone(&barrier);
            let results = Arc::clone(&results);
            let core_ids = Arc::clone(core_ids);
            thread::spawn(move || {
                pin_thread(tid, &core_ids);
                let guard = tree.guard();
                let seed = KvRandom::FIRST_SEED + (tid % 48) as u64;
                let mut rng = KvRandom::new(seed);

                barrier.wait();

                let start = Instant::now();
                let mut n = 0u64;

                while !timed_out() {
                    let x = u64::from(rng.rand());
                    let key = QuickIstr::new(x);
                    let _ = tree.insert_with_guard(key.as_bytes(), x + 1, &guard);
                    n += 1;
                }
                let elapsed = start.elapsed().as_secs_f64();

                results[tid].0.store(n, Ordering::Relaxed);
                results[tid]
                    .1
                    .store((elapsed * 1e9) as u64, Ordering::Relaxed);
            })
        })
        .collect();

    for h in handles {
        h.join().unwrap();
    }

    let thread_results: Vec<_> = results
        .iter()
        .map(|(n, t)| ThreadResult {
            puts: n.load(Ordering::Relaxed),
            gets: 0,
            put_time: t.load(Ordering::Relaxed) as f64 / 1e9,
            get_time: 0.0,
        })
        .collect();

    print_results("wscale", threads, &thread_results);
    thread_results
}

// =============================================================================
// Main
// =============================================================================

fn get_num_cpus() -> usize {
    std::thread::available_parallelism()
        .map(std::num::NonZero::get)
        .unwrap_or(1)
}

fn main() {
    let args = Args::parse();
    // Default threads to #cores (matching C++ mttest)
    let threads = args.threads.unwrap_or_else(get_num_cpus);
    let duration = Duration::from_secs(args.duration);
    let limit = args.limit.unwrap_or(u64::MAX);
    let check = args.check;
    let trials = args.trials;
    let should_save = args.save || args.output.is_some();

    // SAFETY: Single-threaded main context, before any worker threads
    unsafe {
        OUTPUT_JSON = args.json;
    }

    // Parse tests - use C++ naming conventions
    let tests: Vec<&str> = if args.tests.is_empty() || args.tests[0] == "all" {
        // Default test set matching C++ convention
        vec!["rw1", "rw2g98", "rw3", "rw4", "same", "uscale", "wscale"]
    } else {
        args.tests.iter().map(String::as_str).collect()
    };

    // Get core IDs for thread pinning (empty vec if not pinning)
    let core_ids: Arc<Vec<CoreId>> = if args.pin {
        let ids = core_affinity::get_core_ids().unwrap_or_default();
        if ids.is_empty() {
            eprintln!("Warning: --pin specified but no cores available for pinning");
        }
        Arc::new(ids)
    } else {
        Arc::new(Vec::new())
    };

    if !args.quiet && !args.json {
        let pin_info = if args.pin && !core_ids.is_empty() {
            format!(", pinned to {} cores", core_ids.len())
        } else {
            String::new()
        };
        println!(
            "Rust Masstree mttest - {} threads, {}s duration{}{}{}",
            threads,
            duration.as_secs(),
            if limit < u64::MAX {
                format!(", limit={limit}")
            } else {
                String::new()
            },
            if check { ", check=true" } else { "" },
            pin_info
        );
    }

    // Collect results for JSON output
    let mut run_results = RunResults::new(threads, args.duration, args.pin, check);

    for trial in 0..trials {
        // SAFETY: Single-threaded main context, between test runs
        unsafe {
            CURRENT_TRIAL = trial;
        }

        if trials > 1 && !args.quiet && !args.json {
            println!("\n=== Trial {}/{} ===", trial + 1, trials);
        }

        for test in &tests {
            match *test {
                "rw1" => {
                    let results =
                        bench_rw1(threads, duration, limit, check, &Arc::clone(&core_ids));
                    if should_save {
                        run_results.add_benchmark("rw1", results);
                    }
                }
                // C++ naming: rw2 = 50% get fraction
                "rw2" => {
                    let results = bench_rw2(
                        threads,
                        duration,
                        0.5,
                        "rw2",
                        limit,
                        check,
                        &Arc::clone(&core_ids),
                    );
                    if should_save {
                        run_results.add_benchmark("rw2", results);
                    }
                }
                "rw2g90" => {
                    let results = bench_rw2(
                        threads,
                        duration,
                        0.9,
                        "rw2g90",
                        limit,
                        check,
                        &Arc::clone(&core_ids),
                    );
                    if should_save {
                        run_results.add_benchmark("rw2g90", results);
                    }
                }
                "rw2g98" => {
                    let results = bench_rw2(
                        threads,
                        duration,
                        0.98,
                        "rw2g98",
                        limit,
                        check,
                        &Arc::clone(&core_ids),
                    );
                    if should_save {
                        run_results.add_benchmark("rw2g98", results);
                    }
                }
                "rw3" => {
                    let results =
                        bench_rw3(threads, duration, limit, check, &Arc::clone(&core_ids));
                    if should_save {
                        run_results.add_benchmark("rw3", results);
                    }
                }
                "rw4" => {
                    let results =
                        bench_rw4(threads, duration, limit, check, &Arc::clone(&core_ids));
                    if should_save {
                        run_results.add_benchmark("rw4", results);
                    }
                }
                "same" => {
                    let results = bench_same(threads, duration, limit, &Arc::clone(&core_ids));
                    if should_save {
                        run_results.add_benchmark("same", results);
                    }
                }
                "uscale" => {
                    let results = bench_uscale(threads, duration, &Arc::clone(&core_ids));
                    if should_save {
                        run_results.add_benchmark("uscale", results);
                    }
                }
                "wscale" => {
                    let results = bench_wscale(threads, duration, &Arc::clone(&core_ids));
                    if should_save {
                        run_results.add_benchmark("wscale", results);
                    }
                }
                "all" => {
                    let r1 = bench_rw1(threads, duration, limit, check, &Arc::clone(&core_ids));
                    let r2 = bench_rw2(
                        threads,
                        duration,
                        0.98,
                        "rw2g98",
                        limit,
                        check,
                        &Arc::clone(&core_ids),
                    );
                    let r3 = bench_rw3(threads, duration, limit, check, &Arc::clone(&core_ids));
                    let r4 = bench_rw4(threads, duration, limit, check, &Arc::clone(&core_ids));
                    let r5 = bench_same(threads, duration, limit, &Arc::clone(&core_ids));
                    let r6 = bench_uscale(threads, duration, &Arc::clone(&core_ids));
                    let r7 = bench_wscale(threads, duration, &Arc::clone(&core_ids));
                    if should_save {
                        run_results.add_benchmark("rw1", r1);
                        run_results.add_benchmark("rw2g98", r2);
                        run_results.add_benchmark("rw3", r3);
                        run_results.add_benchmark("rw4", r4);
                        run_results.add_benchmark("same", r5);
                        run_results.add_benchmark("uscale", r6);
                        run_results.add_benchmark("wscale", r7);
                    }
                }
                _ => eprintln!("Unknown test: {test}"),
            }
        }
    }

    // Save results to JSON file if requested
    if should_save {
        let output_path = args.output.unwrap_or_else(|| {
            // Generate default filename: runs/mttest_<threads>t_<duration>s_<timestamp>.json
            let timestamp = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .map(|d| d.as_secs())
                .unwrap_or(0);
            PathBuf::from(format!(
                "runs/mttest_{}t_{}s_{}.json",
                threads, args.duration, timestamp
            ))
        });

        // Ensure runs/ directory exists
        if let Some(parent) = output_path.parent()
            && !parent.as_os_str().is_empty()
            && let Err(e) = fs::create_dir_all(parent)
        {
            eprintln!(
                "Warning: Failed to create directory {:?}: {e}",
                parent.display()
            );
        }

        // Write JSON file
        match serde_json::to_string_pretty(&run_results) {
            Ok(json) => match fs::File::create(&output_path) {
                Ok(mut file) => {
                    if let Err(e) = file.write_all(json.as_bytes()) {
                        eprintln!("Error writing to {:?}: {}", output_path.display(), e);
                    } else if !args.quiet {
                        eprintln!("\nResults saved to: {}", output_path.display());
                    }
                }
                Err(e) => eprintln!("Error creating file {:?}: {}", output_path.display(), e),
            },
            Err(e) => eprintln!("Error serializing results: {e}"),
        }
    }
}
