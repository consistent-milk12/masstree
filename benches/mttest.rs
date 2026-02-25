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
#![allow(unused_variables)]
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

/// Knuth multiplicative hash constant (2^32 * phi)
const KNUTH_MULT: u32 = 2_654_435_761;

/// Check timeout every N operations to reduce atomic load overhead.
/// 512 chosen as power-of-2 for efficient bitmasking, small enough for responsiveness.
const TIMEOUT_CHECK_INTERVAL: u64 = 512;

/// Refresh guard every N ops to allow epoch advancement and memory reclamation.
/// Using power-of-2 for efficient countdown reset.
const GUARD_REFRESH_INTERVAL: u64 = 131_072; // 128K ops

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

/// Check timeout only every [`TIMEOUT_CHECK_INTERVAL`] operations.
/// Returns true if timed out, false otherwise.
#[inline(always)]
fn should_stop(op_count: u64) -> bool {
    // Use bitmask for efficient modulo on power-of-2 interval
    (op_count & (TIMEOUT_CHECK_INTERVAL - 1)) == 0 && timed_out()
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

/// Parse threads with validation (must be >= 1)
fn parse_threads(s: &str) -> Result<usize, String> {
    let n: usize = s
        .parse()
        .map_err(|_| format!("'{s}' is not a valid number"))?;
    if n == 0 {
        Err("threads must be at least 1".to_string())
    } else {
        Ok(n)
    }
}

#[derive(Parser, Debug)]
#[command(
    name = "mttest",
    about = "C++ mttest-compatible Masstree benchmark",
    // Required: cargo passes --bench to the binary which we must ignore
    ignore_errors = true
)]
struct Args {
    /// Number of threads (default: number of CPU cores, must be >= 1)
    #[arg(short = 'j', long = "threads", value_parser = parse_threads)]
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

    /// Only show totals, skip per-thread breakdown
    #[arg(long = "totals-only", short = 't')]
    totals_only: bool,

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
    trial: u64,
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
    crate_version: String,
    threads: usize,
    duration_secs: u64,
    pinned: bool,
    check_enabled: bool,
    benchmarks: Vec<BenchmarkResult>,
}

impl RunResults {
    fn new(threads: usize, duration_secs: u64, pinned: bool, check: bool) -> Self {
        // Use Unix timestamp for simplicity and accuracy
        let timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_secs().to_string())
            .unwrap_or_else(|_| "0".to_string());

        Self {
            timestamp,
            crate_version: env!("CARGO_PKG_VERSION").to_string(),
            threads,
            duration_secs,
            pinned,
            check_enabled: check,
            benchmarks: Vec::new(),
        }
    }

    fn add_benchmark(&mut self, name: &str, trial: u64, thread_results: Vec<ThreadResult>) {
        let mut total_puts = 0u64;
        let mut total_gets = 0u64;
        let mut total_put_rate = 0.0;
        let mut total_get_rate = 0.0;
        let mut total_ops_rate = 0.0;

        // Reuse ComputedRates for consistent calculation with print_results
        for r in &thread_results {
            let rates = ComputedRates::from_result(r);
            total_puts += r.puts;
            total_gets += r.gets;
            total_put_rate += rates.put_rate;
            total_get_rate += rates.get_rate;
            total_ops_rate += rates.ops_rate;
        }

        self.benchmarks.push(BenchmarkResult {
            name: name.to_string(),
            trial,
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

/// Output mode for results (atomic to avoid unsafe blocks)
static OUTPUT_JSON: AtomicBool = AtomicBool::new(false);
static TOTALS_ONLY: AtomicBool = AtomicBool::new(false);
static CURRENT_TRIAL: AtomicU64 = AtomicU64::new(0);

/// Pre-computed rates for a thread result to avoid redundant calculations
struct ComputedRates {
    put_rate: f64,
    get_rate: f64,
    ops: u64,
    ops_rate: f64,
}

impl ComputedRates {
    #[inline]
    fn from_result(r: &ThreadResult) -> Self {
        const EPSILON: f64 = 1e-9;
        let put_rate = if r.put_time > EPSILON {
            r.puts as f64 / r.put_time
        } else {
            0.0
        };
        let get_rate = if r.get_time > EPSILON {
            r.gets as f64 / r.get_time
        } else {
            0.0
        };
        let ops = r.puts + r.gets;
        let total_time = r.put_time + r.get_time;
        let ops_rate = if total_time > EPSILON {
            ops as f64 / total_time
        } else {
            0.0
        };
        Self {
            put_rate,
            get_rate,
            ops,
            ops_rate,
        }
    }
}

fn print_results(test: &str, threads: usize, results: &[ThreadResult]) {
    let json_mode = OUTPUT_JSON.load(Ordering::Relaxed);
    let totals_only = TOTALS_ONLY.load(Ordering::Relaxed);
    let trial = CURRENT_TRIAL.load(Ordering::Relaxed);

    // Single pass: compute per-thread rates and accumulate totals
    let mut total_puts = 0u64;
    let mut total_gets = 0u64;
    let mut total_ops_rate = 0.0;
    let mut total_put_rate = 0.0;
    let mut total_get_rate = 0.0;

    // Pre-compute all rates in one pass (avoids redundant calculation)
    let computed: Vec<ComputedRates> = results
        .iter()
        .map(|r| {
            let rates = ComputedRates::from_result(r);
            total_puts += r.puts;
            total_gets += r.gets;
            total_put_rate += rates.put_rate;
            total_get_rate += rates.get_rate;
            total_ops_rate += rates.ops_rate;
            rates
        })
        .collect();

    if json_mode {
        // C++ compatible JSON output (one line per thread)
        for (tid, (r, rates)) in results.iter().zip(computed.iter()).enumerate() {
            println!(
                r#"{{"table":"masstree","test":"{}","trial":{},"thread":{},"puts":{},"puts_per_sec":{:.0},"gets":{},"gets_per_sec":{:.0},"ops":{},"ops_per_sec":{:.0}}}"#,
                test,
                trial,
                tid,
                r.puts,
                rates.put_rate,
                r.gets,
                rates.get_rate,
                rates.ops,
                rates.ops_rate
            );
        }
    } else if totals_only {
        // Compact single-line format: test_name: X.XX Mops/s
        let mops = total_ops_rate / 1_000_000.0;
        println!("{:<8} {:>6.2} Mops/s", format!("{test}:"), mops);
    } else {
        println!("\n{test} with {threads} threads:");
        println!(
            "{:>8} {:>12} {:>14} {:>12} {:>14} {:>12} {:>14}",
            "thread", "puts", "puts/sec", "gets", "gets/sec", "ops", "ops/sec"
        );
        println!("{}", "-".repeat(90));

        for (tid, (r, rates)) in results.iter().zip(computed.iter()).enumerate() {
            println!(
                "{:>8} {:>12} {:>14.0} {:>12} {:>14.0} {:>12} {:>14.0}",
                tid, r.puts, rates.put_rate, r.gets, rates.get_rate, rates.ops, rates.ops_rate
            );
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
                let seed = KvRandom::FIRST_SEED + (tid % 48) as u64;
                let mut rng = KvRandom::new(seed);

                start_barrier.wait();

                // Put phase - just count ops, don't store keys (saves ~80MB/thread)
                // Keys are regenerated deterministically from seed after barrier
                let put_start = Instant::now();
                let mut puts = 0u64;

                {
                    // Scoped guard for put phase - allows epoch advancement between phases
                    // This enables seize to reclaim retired nodes, reducing memory from
                    // ~16GB to ~2-3GB by not blocking reclamation for entire benchmark
                    let guard = tree.guard();
                    while !should_stop(puts) && puts < limit {
                        let x = rng.rand();
                        let key = QuickIstr::new(u64::from(x));
                        let _ = tree.insert_with_guard(key.as_bytes(), u64::from(x + 1), &guard);
                        puts += 1;
                    }
                } // guard dropped here - allows reclamation of retired nodes
                let put_time = put_start.elapsed().as_secs_f64();

                // C++ wait_all() - barrier between put and get phases
                get_barrier.wait();

                // Regenerate keys from seed (deterministic RNG), then shuffle
                // Allocate exact size needed - no growth during timed phase
                let mut rng = KvRandom::new(seed);
                let mut keys = Vec::with_capacity(puts as usize);
                for _ in 0..puts {
                    keys.push(rng.rand());
                }
                // Shuffle: swap a[i] with a[uniform(0..n-1)]
                for i in 0..keys.len() {
                    let j = rng.uniform(keys.len() as u32) as usize;
                    keys.swap(i, j);
                }

                // Get phase - fresh guard, exactly `puts` iterations, no timeout (matches C++)
                let get_start = Instant::now();
                let mut sum = 0u64;
                let mut check_errors = 0u64;
                {
                    let guard = tree.guard();
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
    name: &'static str,
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
            let core_ids = Arc::clone(core_ids);
            thread::spawn(move || {
                pin_thread(tid, &core_ids);
                let seed = KvRandom::FIRST_SEED + (tid % 48) as u64;
                let mut rng = KvRandom::new(seed);
                let offset = rng.rand();

                barrier.wait();

                let start = Instant::now();
                let mut puts = 0u64;
                let mut gets = 0u64;
                let mut sum = 0u64;
                let mut guard = tree.guard();
                // Countdown-based refresh: decrement instead of compare against running total
                let mut ops_until_refresh = GUARD_REFRESH_INTERVAL;

                let mut check_errors = 0u64;
                let mut ops = 0u64;
                loop {
                    // Check termination conditions with batched timeout check
                    if should_stop(ops) || ops >= limit {
                        break;
                    }

                    // Countdown-based guard refresh (avoids addition in hot path)
                    ops_until_refresh -= 1;
                    if ops_until_refresh == 0 {
                        drop(guard);
                        guard = tree.guard();
                        ops_until_refresh = GUARD_REFRESH_INTERVAL;
                    }

                    if puts == 0 || !rng.bernoulli(get_frac) {
                        let x = (offset.wrapping_add(puts as u32)).wrapping_mul(KNUTH_MULT);
                        let key = QuickIstr::new(u64::from(x));
                        let _ = tree.insert_with_guard(key.as_bytes(), u64::from(x + 1), &guard);
                        puts += 1;
                    } else {
                        let idx = rng.uniform(puts as u32);
                        let x = (offset.wrapping_add(idx)).wrapping_mul(KNUTH_MULT);
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
                    ops += 1;
                }
                drop(guard);
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
            let elapsed = t.load(Ordering::Relaxed) as f64 / 1e9;
            let puts = p.load(Ordering::Relaxed);
            let gets = g.load(Ordering::Relaxed);
            // For mixed workloads, puts and gets happen interleaved in the SAME time period.
            // Set get_time=0 so ops_rate = (puts+gets)/elapsed is correct.
            // Individual puts_per_sec and gets_per_sec use put_time (=elapsed) for meaningful rates.
            ThreadResult {
                puts,
                gets,
                put_time: elapsed,
                get_time: 0.0, // Mixed workload: don't double-count time
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

                barrier.wait();

                // Put phase - sequential keys, duration-bounded via timeout flag
                let put_start = Instant::now();
                let mut n = 0u64;

                {
                    // Scoped guard for put phase - allows epoch advancement between phases
                    let guard = tree.guard();
                    while !should_stop(n) && n < limit {
                        let key = key8(n);
                        let _ = tree.insert_with_guard(key.as_bytes(), n + 1, &guard);
                        n += 1;
                    }
                } // guard dropped - allows reclamation
                let put_time = put_start.elapsed().as_secs_f64();

                // Sync point (matches C++ wait_all between phases)
                get_barrier.wait();

                // Get phase - POINT LOOKUPS, exactly n iterations (matches C++ exactly)
                // C++ does: for (unsigned i = 0; i < n; ++i) { client.get_check_key8(i, i + 1); }
                let get_start = Instant::now();
                let mut sum = 0u64;
                let mut check_errors = 0u64;

                {
                    let guard = tree.guard();
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

                barrier.wait();

                // Put phase - reverse sequential keys, duration-bounded via timeout flag
                let put_start = Instant::now();
                let mut n = 0u64;

                {
                    // Scoped guard for put phase - allows epoch advancement between phases
                    let guard = tree.guard();
                    while !should_stop(n) && n < limit {
                        let key = key8(TOP - n);
                        let _ = tree.insert_with_guard(key.as_bytes(), n + 1, &guard);
                        n += 1;
                    }
                } // guard dropped - allows reclamation
                let put_time = put_start.elapsed().as_secs_f64();

                // Sync point (matches C++ wait_all between phases)
                get_barrier.wait();

                // Get phase - POINT LOOKUPS in reverse order, exactly n iterations
                // C++ does: for (unsigned i = 0; i < n; ++i) { client.get_check_key8(top - i, i + 1); }
                let get_start = Instant::now();
                let mut sum = 0u64;
                let mut check_errors = 0u64;

                {
                    let guard = tree.guard();
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
                let seed = KvRandom::FIRST_SEED + (tid % 48) as u64;
                let mut rng = KvRandom::new(seed);

                barrier.wait();

                let start = Instant::now();
                let mut n = 0u64;
                let mut guard = tree.guard();
                let mut ops_until_refresh = GUARD_REFRESH_INTERVAL;

                loop {
                    if should_stop(n) || n >= limit {
                        break;
                    }
                    // Countdown-based guard refresh
                    ops_until_refresh -= 1;
                    if ops_until_refresh == 0 {
                        drop(guard);
                        guard = tree.guard();
                        ops_until_refresh = GUARD_REFRESH_INTERVAL;
                    }
                    let x = rng.uniform(NUM_KEYS);
                    let key = QuickIstr::new(u64::from(x));
                    let _ = tree.insert_with_guard(key.as_bytes(), u64::from(x + 1), &guard);
                    n += 1;
                }
                drop(guard);
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
    limit: u64,
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
                // C++: seed = kvtest_first_seed + client.id() (NOT % 48)
                let seed = KvRandom::FIRST_SEED + tid as u64;
                let mut rng = KvRandom::new(seed);

                barrier.wait();

                let start = Instant::now();
                let mut n = 0u64;
                let mut guard = tree.guard();
                let mut ops_until_refresh = GUARD_REFRESH_INTERVAL;

                loop {
                    if should_stop(n) || n >= limit {
                        break;
                    }
                    // Countdown-based guard refresh
                    ops_until_refresh -= 1;
                    if ops_until_refresh == 0 {
                        drop(guard);
                        guard = tree.guard();
                        ops_until_refresh = GUARD_REFRESH_INTERVAL;
                    }
                    let x = u64::from(rng.rand()) % NSEQKEYS;
                    let key = QuickIstr::new(x);
                    let _ = tree.insert_with_guard(key.as_bytes(), x + 1, &guard);
                    n += 1;
                }
                drop(guard);
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
    limit: u64,
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
                let seed = KvRandom::FIRST_SEED + (tid % 48) as u64;
                let mut rng = KvRandom::new(seed);

                barrier.wait();

                let start = Instant::now();
                let mut n = 0u64;
                let mut guard = tree.guard();
                let mut ops_until_refresh = GUARD_REFRESH_INTERVAL;

                loop {
                    if should_stop(n) || n >= limit {
                        break;
                    }
                    // Countdown-based guard refresh
                    ops_until_refresh -= 1;
                    if ops_until_refresh == 0 {
                        drop(guard);
                        guard = tree.guard();
                        ops_until_refresh = GUARD_REFRESH_INTERVAL;
                    }
                    let x = u64::from(rng.rand());
                    let key = QuickIstr::new(x);
                    let _ = tree.insert_with_guard(key.as_bytes(), x + 1, &guard);
                    n += 1;
                }
                drop(guard);
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

    OUTPUT_JSON.store(args.json, Ordering::Relaxed);
    TOTALS_ONLY.store(args.totals_only, Ordering::Relaxed);

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

    for trial in 0..trials as u64 {
        CURRENT_TRIAL.store(trial, Ordering::Relaxed);

        if trials > 1 && !args.quiet && !args.json {
            println!("\n=== Trial {}/{} ===", trial + 1, trials);
        }
        for test in &tests {
            match *test {
                "rw1" => {
                    let results = bench_rw1(threads, duration, limit, check, &core_ids);
                    if should_save {
                        run_results.add_benchmark("rw1", trial, results);
                    }
                }
                // C++ naming: rw2 = 50% get fraction
                "rw2" => {
                    let results = bench_rw2(threads, duration, 0.5, "rw2", limit, check, &core_ids);
                    if should_save {
                        run_results.add_benchmark("rw2", trial, results);
                    }
                }
                "rw2g90" => {
                    let results =
                        bench_rw2(threads, duration, 0.9, "rw2g90", limit, check, &core_ids);
                    if should_save {
                        run_results.add_benchmark("rw2g90", trial, results);
                    }
                }
                "rw2g98" => {
                    let results =
                        bench_rw2(threads, duration, 0.98, "rw2g98", limit, check, &core_ids);
                    if should_save {
                        run_results.add_benchmark("rw2g98", trial, results);
                    }
                }
                "rw3" => {
                    let results = bench_rw3(threads, duration, limit, check, &core_ids);
                    if should_save {
                        run_results.add_benchmark("rw3", trial, results);
                    }
                }
                "rw4" => {
                    let results = bench_rw4(threads, duration, limit, check, &core_ids);
                    if should_save {
                        run_results.add_benchmark("rw4", trial, results);
                    }
                }
                "same" => {
                    let results = bench_same(threads, duration, limit, &core_ids);
                    if should_save {
                        run_results.add_benchmark("same", trial, results);
                    }
                }
                "uscale" => {
                    let results = bench_uscale(threads, duration, limit, &core_ids);
                    if should_save {
                        run_results.add_benchmark("uscale", trial, results);
                    }
                }
                "wscale" => {
                    let results = bench_wscale(threads, duration, limit, &core_ids);
                    if should_save {
                        run_results.add_benchmark("wscale", trial, results);
                    }
                }
                "all" => {
                    let r1 = bench_rw1(threads, duration, limit, check, &core_ids);
                    let r2 = bench_rw2(threads, duration, 0.98, "rw2g98", limit, check, &core_ids);
                    let r3 = bench_rw3(threads, duration, limit, check, &core_ids);
                    let r4 = bench_rw4(threads, duration, limit, check, &core_ids);
                    let r5 = bench_same(threads, duration, limit, &core_ids);
                    let r6 = bench_uscale(threads, duration, limit, &core_ids);
                    let r7 = bench_wscale(threads, duration, limit, &core_ids);
                    if should_save {
                        run_results.add_benchmark("rw1", trial, r1);
                        run_results.add_benchmark("rw2g98", trial, r2);
                        run_results.add_benchmark("rw3", trial, r3);
                        run_results.add_benchmark("rw4", trial, r4);
                        run_results.add_benchmark("same", trial, r5);
                        run_results.add_benchmark("uscale", trial, r6);
                        run_results.add_benchmark("wscale", trial, r7);
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
