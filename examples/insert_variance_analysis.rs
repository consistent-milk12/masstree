//! Targeted analysis of insert performance variance at low thread counts.
//!
//! This example specifically investigates the 2-3 thread insert variance observed
//! in benchmarks, where outlier iterations are 10-50x slower than median.
//!
//! # Usage
//!
//! ```bash
//! # Build with profiling symbols
//! cargo build --profile profiling --example insert_variance_analysis --features mimalloc
//!
//! # Quick analysis (default: 2 threads, 50 batches)
//! ./target/profiling/examples/insert_variance_analysis
//!
//! # Test specific thread count
//! ./target/profiling/examples/insert_variance_analysis -j3
//!
//! # Full sweep (1-6 threads)
//! ./target/profiling/examples/insert_variance_analysis --sweep
//!
//! # With insert-stats diagnostics (shows retry rates and contended mode usage)
//! cargo build --profile profiling --example insert_variance_analysis --features "mimalloc insert-stats"
//! ./target/profiling/examples/insert_variance_analysis --sweep --prefix
//!
//! # Profile with perf (focus on slow batches)
//! perf record -g --call-graph dwarf -F 4999 \
//!     ./target/profiling/examples/insert_variance_analysis -j2 --batches 100
//!
//! # Generate flamegraph
//! perf script | stackcollapse-perf.pl | flamegraph.pl > insert_variance.svg
//!
//! # Record with timestamps for correlation
//! perf record -g --call-graph dwarf -F 4999 -T \
//!     ./target/profiling/examples/insert_variance_analysis -j2
//! ```
//!
//! # What to Look For
//!
//! 1. **Split propagation storms**: Check if slow batches correlate with tree depth changes
//! 2. **Version counter contention**: Look for `fetch_add` / CAS loops in profiles
//! 3. **Memory allocation stalls**: Check for allocator contention (mimalloc vs system)
//! 4. **False sharing**: Cache line bouncing between threads
//! 5. **Adaptive insert behavior**: With `insert-stats` feature, check `contended_mode_enters`
//!
//! # Key Patterns Tested
//!
//! - Sequential keys: Maximum split pressure (all inserts at frontier)
//! - Random keys: Distributed inserts across tree
//! - Shared prefix: Trie layer contention (triggers adaptive insert)

#![allow(
    dead_code,
    clippy::cast_precision_loss,
    clippy::unwrap_used,
    clippy::indexing_slicing,
    clippy::similar_names,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss
)]

use masstree::MassTree15Inline;
use std::hint::black_box;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Barrier};
use std::thread;
use std::time::Instant;

// ============================================================================
// Configuration
// ============================================================================

const DEFAULT_THREADS: usize = 2;
const DEFAULT_BATCHES: usize = 50;
const DEFAULT_OPS_PER_BATCH: usize = 10_000;

// ============================================================================
// Key Generation
// ============================================================================

/// Fast decimal string key (matches C++ mttest pattern)
struct QuickKey {
    buf: [u8; 24],
    len: usize,
}

impl QuickKey {
    const fn from_u64(n: u64) -> Self {
        let mut buf = [b'0'; 24];
        let mut val = n;
        let mut pos = 23;

        loop {
            buf[pos] = b'0' + (val % 10) as u8;
            val /= 10;
            if val == 0 {
                break;
            }
            pos -= 1;
        }

        // Pad to minimum 8 chars
        let digit_len = 24 - pos;
        let len = if digit_len < 8 { 8 } else { digit_len };
        let start = 24 - len;

        let mut result = [0u8; 24];
        let mut i = 0;
        while i < len {
            result[i] = buf[start + i];
            i += 1;
        }

        Self { buf: result, len }
    }

    fn as_bytes(&self) -> &[u8] {
        &self.buf[..self.len]
    }
}

/// Simple LCG for reproducible random keys
struct Lcg {
    state: u64,
}

impl Lcg {
    const fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    const fn next(&mut self) -> u64 {
        self.state = self
            .state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1);
        self.state >> 32 // Use high bits
    }
}

// ============================================================================
// Insert Stats (conditional on feature)
// ============================================================================

/// Print insert stats if the feature is enabled, then reset for next test.
#[cfg(feature = "insert-stats")]
fn print_and_reset_insert_stats(label: &str) {
    let stats = masstree::insert_stats::get_stats();
    println!("\n=== Insert Stats: {label} ===");
    println!("  Attempts:             {:>12}", stats.attempts);
    println!("  Successful inserts:   {:>12}", stats.successful_inserts);
    println!("  Successful updates:   {:>12}", stats.successful_updates);
    println!("  ---");
    println!("  Validation failures:  {:>12}", stats.validation_failures);
    println!("  Membership failures:  {:>12}", stats.membership_failures);
    println!("  Null ptr retries:     {:>12}", stats.null_ptr_retries);
    println!("  Split retries:        {:>12}", stats.split_retries);
    println!("  Hop limit exceeded:   {:>12}", stats.hop_limit_exceeded);
    println!(
        "  Deleted layer retries:{:>12}",
        stats.deleted_layer_retries
    );

    println!("  Total retries:        {:>12}", stats.total_retries());
    println!("  Retry rate:           {:>11.2}%", stats.retry_rate());

    masstree::insert_stats::reset_stats();
}

#[cfg(not(feature = "insert-stats"))]
fn print_and_reset_insert_stats(_label: &str) {
    // No-op when feature is disabled
}

// ============================================================================
// Timing Collection
// ============================================================================

#[derive(Clone, Debug)]
struct BatchTiming {
    batch_id: usize,
    thread_id: usize,
    duration_us: u64,
    ops: usize,
}

#[derive(Debug)]
struct TimingStats {
    timings: Vec<BatchTiming>,
}

impl TimingStats {
    const fn new() -> Self {
        Self {
            timings: Vec::new(),
        }
    }

    fn add(&mut self, timing: BatchTiming) {
        self.timings.push(timing);
    }

    fn analyze(&self) {
        if self.timings.is_empty() {
            println!("No timings collected");
            return;
        }

        let mut durations: Vec<u64> = self.timings.iter().map(|t| t.duration_us).collect();
        durations.sort_unstable();

        let n = durations.len();
        let min = durations[0];
        let max = durations[n - 1];
        let median = durations[n / 2];
        let p90 = durations[(n * 90) / 100];
        let p99 = durations[(n * 99) / 100];
        let p999 = durations[(n * 999) / 1000];

        let sum: u64 = durations.iter().sum();
        let mean = sum / n as u64;

        // Variance
        let variance: f64 = durations
            .iter()
            .map(|&d| {
                let diff = d as f64 - mean as f64;
                diff * diff
            })
            .sum::<f64>()
            / n as f64;
        let stddev = variance.sqrt();

        println!("\n=== Timing Statistics ===");
        println!("  Samples:    {n}");
        println!("  Min:        {min} µs");
        println!("  Median:     {median} µs");
        println!("  Mean:       {mean} µs");
        println!("  P90:        {p90} µs");
        println!("  P99:        {p99} µs");
        println!("  P99.9:      {p999} µs");
        println!("  Max:        {max} µs");
        println!("  Stddev:     {stddev:.1} µs");
        println!("  Max/Median: {:.1}x", max as f64 / median as f64);
        println!("  P99/Median: {:.1}x", p99 as f64 / median as f64);

        // Outlier detection (>3 stddev from mean)
        let outlier_threshold = 3.0f64.mul_add(stddev, mean as f64);
        let outliers: Vec<_> = self
            .timings
            .iter()
            .filter(|t| t.duration_us as f64 > outlier_threshold)
            .collect();

        if !outliers.is_empty() {
            println!("\n=== Outliers (>{outlier_threshold:.0} µs, >3σ) ===");
            for t in outliers.iter().take(10) {
                println!(
                    "  Batch {:3} (thread {}): {:6} µs ({:.1}x median)",
                    t.batch_id,
                    t.thread_id,
                    t.duration_us,
                    t.duration_us as f64 / median as f64
                );
            }
            if outliers.len() > 10 {
                println!("  ... and {} more", outliers.len() - 10);
            }
        }

        // Histogram
        println!("\n=== Histogram ===");
        let buckets = [
            (0, 100, "0-100µs"),
            (100, 200, "100-200µs"),
            (200, 500, "200-500µs"),
            (500, 1000, "500µs-1ms"),
            (1000, 2000, "1-2ms"),
            (2000, 5000, "2-5ms"),
            (5000, 10000, "5-10ms"),
            (10000, 50000, "10-50ms"),
            (50000, u64::MAX, ">50ms"),
        ];

        for (lo, hi, label) in buckets {
            let count = durations.iter().filter(|&&d| d >= lo && d < hi).count();
            if count > 0 {
                let pct = 100.0 * count as f64 / n as f64;
                let bar_len = (pct / 2.0) as usize;
                let bar: String = "#".repeat(bar_len);
                println!("  {label:>12}: {count:4} ({pct:5.1}%) {bar}");
            }
        }
    }
}

// ============================================================================
// Test Runners
// ============================================================================

fn run_sequential_insert_test(threads: usize, batches: usize, ops_per_batch: usize) -> TimingStats {
    println!("\n=== Sequential Insert Test ===");
    println!("Threads: {threads}, Batches: {batches}, Ops/batch: {ops_per_batch}");

    let tree = Arc::new(MassTree15Inline::<u64>::new());
    let barrier = Arc::new(Barrier::new(threads));
    let batch_counter = Arc::new(AtomicUsize::new(0));
    let all_timings = Arc::new(std::sync::Mutex::new(TimingStats::new()));

    let handles: Vec<_> = (0..threads)
        .map(|tid| {
            let tree = Arc::clone(&tree);
            let barrier = Arc::clone(&barrier);
            let batch_counter = Arc::clone(&batch_counter);
            let all_timings = Arc::clone(&all_timings);

            thread::spawn(move || {
                // Optional: pin to core
                if let Some(core_ids) = core_affinity::get_core_ids()
                    && !core_ids.is_empty()
                {
                    let core = core_ids[tid % core_ids.len()];
                    core_affinity::set_for_current(core);
                }

                let guard = tree.guard();
                barrier.wait();

                let mut local_timings = Vec::new();
                let mut key_counter = (tid as u64) * 1_000_000_000; // Offset per thread

                loop {
                    let batch_id = batch_counter.fetch_add(1, Ordering::SeqCst);
                    if batch_id >= batches {
                        break;
                    }

                    // Time this batch
                    let start = Instant::now();
                    for _ in 0..ops_per_batch {
                        let key = QuickKey::from_u64(key_counter);
                        let _ =
                            black_box(tree.insert_with_guard(key.as_bytes(), key_counter, &guard));
                        key_counter += 1;
                    }
                    let elapsed = start.elapsed();

                    local_timings.push(BatchTiming {
                        batch_id,
                        thread_id: tid,
                        duration_us: elapsed.as_micros() as u64,
                        ops: ops_per_batch,
                    });
                }

                // Merge timings
                let mut all = all_timings.lock().unwrap();
                for t in local_timings {
                    all.add(t);
                }
            })
        })
        .collect();

    for h in handles {
        h.join().unwrap();
    }

    let stats = Arc::try_unwrap(all_timings).unwrap().into_inner().unwrap();

    let total_ops = batches * ops_per_batch;
    let tree_size = tree.len();
    println!("Total ops: {total_ops}, Tree size: {tree_size}");

    stats
}

fn run_random_insert_test(threads: usize, batches: usize, ops_per_batch: usize) -> TimingStats {
    println!("\n=== Random Insert Test ===");
    println!("Threads: {threads}, Batches: {batches}, Ops/batch: {ops_per_batch}");

    let tree = Arc::new(MassTree15Inline::<u64>::new());
    let barrier = Arc::new(Barrier::new(threads));
    let batch_counter = Arc::new(AtomicUsize::new(0));
    let all_timings = Arc::new(std::sync::Mutex::new(TimingStats::new()));

    let handles: Vec<_> = (0..threads)
        .map(|tid| {
            let tree = Arc::clone(&tree);
            let barrier = Arc::clone(&barrier);
            let batch_counter = Arc::clone(&batch_counter);
            let all_timings = Arc::clone(&all_timings);

            thread::spawn(move || {
                if let Some(core_ids) = core_affinity::get_core_ids()
                    && !core_ids.is_empty()
                {
                    let core = core_ids[tid % core_ids.len()];
                    core_affinity::set_for_current(core);
                }

                let guard = tree.guard();
                let mut rng = Lcg::new(tid as u64 * 12345 + 67890);
                barrier.wait();

                let mut local_timings = Vec::new();

                loop {
                    let batch_id = batch_counter.fetch_add(1, Ordering::SeqCst);
                    if batch_id >= batches {
                        break;
                    }

                    let start = Instant::now();
                    for _ in 0..ops_per_batch {
                        let key_val = rng.next();
                        let key = QuickKey::from_u64(key_val);
                        let _ = black_box(tree.insert_with_guard(key.as_bytes(), key_val, &guard));
                    }
                    let elapsed = start.elapsed();

                    local_timings.push(BatchTiming {
                        batch_id,
                        thread_id: tid,
                        duration_us: elapsed.as_micros() as u64,
                        ops: ops_per_batch,
                    });
                }

                let mut all = all_timings.lock().unwrap();
                for t in local_timings {
                    all.add(t);
                }
            })
        })
        .collect();

    for h in handles {
        h.join().unwrap();
    }

    let stats = Arc::try_unwrap(all_timings).unwrap().into_inner().unwrap();
    println!("Tree size: {}", tree.len());
    stats
}

fn run_shared_prefix_test(threads: usize, batches: usize, ops_per_batch: usize) -> TimingStats {
    println!("\n=== Shared Prefix Insert Test ===");
    println!("Threads: {threads}, Batches: {batches}, Ops/batch: {ops_per_batch}");
    println!("Pattern: All keys share 'prefix__' (8 bytes), differ in suffix");

    let tree = Arc::new(MassTree15Inline::<u64>::new());
    let barrier = Arc::new(Barrier::new(threads));
    let batch_counter = Arc::new(AtomicUsize::new(0));
    let global_counter = Arc::new(AtomicU64::new(0));
    let all_timings = Arc::new(std::sync::Mutex::new(TimingStats::new()));

    let handles: Vec<_> = (0..threads)
        .map(|tid| {
            let tree = Arc::clone(&tree);
            let barrier = Arc::clone(&barrier);
            let batch_counter = Arc::clone(&batch_counter);
            let global_counter = Arc::clone(&global_counter);
            let all_timings = Arc::clone(&all_timings);

            thread::spawn(move || {
                if let Some(core_ids) = core_affinity::get_core_ids()
                    && !core_ids.is_empty()
                {
                    let core = core_ids[tid % core_ids.len()];
                    core_affinity::set_for_current(core);
                }

                let guard = tree.guard();
                barrier.wait();

                let mut local_timings = Vec::new();

                loop {
                    let batch_id = batch_counter.fetch_add(1, Ordering::SeqCst);
                    if batch_id >= batches {
                        break;
                    }

                    let start = Instant::now();
                    for _ in 0..ops_per_batch {
                        let suffix = global_counter.fetch_add(1, Ordering::Relaxed);
                        // Key format: "prefix__" + suffix (forces all keys through same first layer)
                        let mut key = [0u8; 16];
                        key[..8].copy_from_slice(b"prefix__");
                        key[8..].copy_from_slice(&suffix.to_be_bytes());
                        let _ = black_box(tree.insert_with_guard(&key, suffix, &guard));
                    }
                    let elapsed = start.elapsed();

                    local_timings.push(BatchTiming {
                        batch_id,
                        thread_id: tid,
                        duration_us: elapsed.as_micros() as u64,
                        ops: ops_per_batch,
                    });
                }

                let mut all = all_timings.lock().unwrap();
                for t in local_timings {
                    all.add(t);
                }
            })
        })
        .collect();

    for h in handles {
        h.join().unwrap();
    }

    let stats = Arc::try_unwrap(all_timings).unwrap().into_inner().unwrap();
    println!("Tree size: {}", tree.len());
    stats
}

// ============================================================================
// Main
// ============================================================================

fn main() {
    let args: Vec<String> = std::env::args().collect();

    let mut threads = DEFAULT_THREADS;
    let mut batches = DEFAULT_BATCHES;
    let mut ops_per_batch = DEFAULT_OPS_PER_BATCH;
    let mut sweep = false;
    let mut test_type = "all";

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "-j" | "--threads" => {
                i += 1;
                threads = args
                    .get(i)
                    .and_then(|s| s.parse().ok())
                    .unwrap_or(DEFAULT_THREADS);
            }
            "-b" | "--batches" => {
                i += 1;
                batches = args
                    .get(i)
                    .and_then(|s| s.parse().ok())
                    .unwrap_or(DEFAULT_BATCHES);
            }
            "-o" | "--ops" => {
                i += 1;
                ops_per_batch = args
                    .get(i)
                    .and_then(|s| s.parse().ok())
                    .unwrap_or(DEFAULT_OPS_PER_BATCH);
            }
            "--sweep" => {
                sweep = true;
            }
            "--sequential" => {
                test_type = "sequential";
            }
            "--random" => {
                test_type = "random";
            }
            "--prefix" => {
                test_type = "prefix";
            }
            "-h" | "--help" => {
                print_help();
                return;
            }
            _ => {}
        }
        i += 1;
    }

    println!("╔═══════════════════════════════════════════════════════════════╗");
    println!("║           INSERT VARIANCE ANALYSIS                            ║");
    println!("║                                                               ║");
    println!("║  Investigating 2-3 thread insert performance outliers         ║");
    println!("╚═══════════════════════════════════════════════════════════════╝");

    if sweep {
        run_sweep(batches, ops_per_batch);
    } else {
        run_tests(threads, batches, ops_per_batch, test_type);
    }
}

fn run_tests(threads: usize, batches: usize, ops_per_batch: usize, test_type: &str) {
    match test_type {
        "sequential" => {
            let stats = run_sequential_insert_test(threads, batches, ops_per_batch);
            stats.analyze();
            print_and_reset_insert_stats("Sequential");
        }
        "random" => {
            let stats = run_random_insert_test(threads, batches, ops_per_batch);
            stats.analyze();
            print_and_reset_insert_stats("Random");
        }
        "prefix" => {
            let stats = run_shared_prefix_test(threads, batches, ops_per_batch);
            stats.analyze();
            print_and_reset_insert_stats("Shared Prefix");
        }
        _ => {
            // Run all tests
            let stats1 = run_sequential_insert_test(threads, batches, ops_per_batch);
            stats1.analyze();
            print_and_reset_insert_stats("Sequential");

            let stats2 = run_random_insert_test(threads, batches, ops_per_batch);
            stats2.analyze();
            print_and_reset_insert_stats("Random");

            let stats3 = run_shared_prefix_test(threads, batches, ops_per_batch);
            stats3.analyze();
            print_and_reset_insert_stats("Shared Prefix");
        }
    }
}

fn run_sweep(batches: usize, ops_per_batch: usize) {
    println!("\n╔═══════════════════════════════════════════════════════════════╗");
    println!("║                    THREAD COUNT SWEEP                         ║");
    println!("╚═══════════════════════════════════════════════════════════════╝");

    println!("\n>>> Sequential Insert Pattern <<<");
    println!(
        "{:>8} {:>10} {:>10} {:>10} {:>10} {:>10}",
        "Threads", "Median", "P99", "Max", "P99/Med", "Max/Med"
    );
    println!("{}", "-".repeat(68));

    for threads in [1, 2, 3, 4, 5, 6] {
        let stats = run_sequential_insert_test(threads, batches, ops_per_batch);
        print_sweep_row(threads, &stats);
    }
    print_and_reset_insert_stats("Sequential (all threads)");

    println!("\n>>> Random Insert Pattern <<<");
    println!(
        "{:>8} {:>10} {:>10} {:>10} {:>10} {:>10}",
        "Threads", "Median", "P99", "Max", "P99/Med", "Max/Med"
    );
    println!("{}", "-".repeat(68));

    for threads in [1, 2, 3, 4, 5, 6] {
        let stats = run_random_insert_test(threads, batches, ops_per_batch);
        print_sweep_row(threads, &stats);
    }
    print_and_reset_insert_stats("Random (all threads)");

    println!("\n>>> Shared Prefix Pattern <<<");
    println!(
        "{:>8} {:>10} {:>10} {:>10} {:>10} {:>10}",
        "Threads", "Median", "P99", "Max", "P99/Med", "Max/Med"
    );
    println!("{}", "-".repeat(68));

    for threads in [1, 2, 3, 4, 5, 6] {
        let stats = run_shared_prefix_test(threads, batches, ops_per_batch);
        print_sweep_row(threads, &stats);
    }
    print_and_reset_insert_stats("Shared Prefix (all threads)");
}

fn print_sweep_row(threads: usize, stats: &TimingStats) {
    let mut durations: Vec<u64> = stats.timings.iter().map(|t| t.duration_us).collect();
    durations.sort_unstable();

    if durations.is_empty() {
        println!("{threads:>8} (no data)");
        return;
    }

    let n = durations.len();
    let median = durations[n / 2];
    let p99 = durations[(n * 99) / 100];
    let max = durations[n - 1];

    println!(
        "{:>8} {:>8}µs {:>8}µs {:>8}µs {:>9.1}x {:>9.1}x",
        threads,
        median,
        p99,
        max,
        p99 as f64 / median as f64,
        max as f64 / median as f64
    );
}

fn print_help() {
    println!("Insert Variance Analysis for MassTree");
    println!();
    println!("USAGE:");
    println!("    insert_variance_analysis [OPTIONS]");
    println!();
    println!("OPTIONS:");
    println!("    -j, --threads <N>     Number of threads (default: 2)");
    println!("    -b, --batches <N>     Number of batches to run (default: 50)");
    println!("    -o, --ops <N>         Operations per batch (default: 10000)");
    println!("    --sweep               Run 1-6 thread sweep");
    println!("    --sequential          Sequential keys only");
    println!("    --random              Random keys only");
    println!("    --prefix              Shared prefix keys only");
    println!("    -h, --help            Print this help");
    println!();
    println!("FEATURES:");
    println!("    insert-stats          Enable detailed insert statistics (retry rates,");
    println!("                          contended mode usage, failure breakdowns)");
    println!();
    println!("EXAMPLES:");
    println!("    # Quick 2-thread analysis");
    println!("    ./insert_variance_analysis -j2");
    println!();
    println!("    # Full sweep with more samples");
    println!("    ./insert_variance_analysis --sweep -b 100");
    println!();
    println!("    # Build with insert-stats diagnostics");
    println!("    cargo build --release --example insert_variance_analysis \\");
    println!("        --features \"mimalloc insert-stats\"");
    println!();
    println!("    # Profile with perf");
    println!("    perf record -g --call-graph dwarf -F 4999 \\");
    println!("        ./insert_variance_analysis -j2 --sequential -b 100");
}
