//! Lock Contention Profiling Binary
//!
//! Profiles insert latency and debug counters to identify the source of
//! multi-second outliers in concurrent write benchmarks. When tracing is
//! enabled, slow locks from `NodeVersion` and slow ops from this binary are
//! written to a JSON log.
//!
//! Run with:
//! ```bash
//! # Without tracing (fast, just stats)
//! cargo run --release --features mimalloc --bin lock_contention
//!
//! # With tracing (writes to logs/lock_contention.json)
//! RUST_LOG=masstree=warn,lock_contention=warn cargo run --release --features "mimalloc,tracing" --bin lock_contention
//!
//! # View slow operations:
//! rg "SLOW_(OP|LOCK)" logs/lock_contention.json
//! ```

#![allow(clippy::unwrap_used)]
#![allow(clippy::cast_possible_truncation)]
#![allow(clippy::cast_precision_loss)]

use masstree::MassTree24;
#[cfg(feature = "tracing")]
use masstree::{DebugCounters, ParentWaitStats};
use std::sync::Arc;
use std::thread;
use std::time::{Duration, Instant};

#[cfg(feature = "tracing")]
type TracingGuard = tracing_appender::non_blocking::WorkerGuard;

#[cfg(not(feature = "tracing"))]
type TracingGuard = ();

// =============================================================================
// Custom Tracing Initialization (JSON to file)
// =============================================================================

#[cfg(feature = "tracing")]
fn init_json_tracing() -> TracingGuard {
    use tracing_subscriber::{EnvFilter, Layer, layer::SubscriberExt, util::SubscriberInitExt};

    let log_dir = "logs";
    let filter_str = std::env::var("RUST_LOG")
        .unwrap_or_else(|_| "masstree=warn,lock_contention=warn".to_string());

    // Create log directory
    let _ = std::fs::create_dir_all(log_dir);

    // File appender - JSON format
    let file_appender = tracing_appender::rolling::never(log_dir, "lock_contention.json");
    let (non_blocking, guard) = tracing_appender::non_blocking(file_appender);

    // JSON file layer with full details
    let file_layer = tracing_subscriber::fmt::layer()
        .with_writer(non_blocking)
        .with_thread_ids(true)
        .with_target(true)
        .with_file(true)
        .with_line_number(true)
        .with_ansi(false)
        .json()
        .with_filter(EnvFilter::try_new(&filter_str).unwrap_or_else(|_| EnvFilter::new("warn")));

    let _ = tracing_subscriber::registry().with(file_layer).try_init();

    println!("Tracing enabled: logs/lock_contention.json (filter: {filter_str})");

    guard
}

#[cfg(not(feature = "tracing"))]
fn init_json_tracing() -> TracingGuard {
    println!("Tracing disabled (compile with --features tracing)");
}

// =============================================================================
// Operation Stats (Thread-Local + Aggregation)
// =============================================================================

/// Per-thread operation timing statistics
#[derive(Default)]
struct ThreadOpStats {
    /// Longest operation overall (insert)
    max_op_ns: u64,

    /// Number of slow ops (>10ms)
    slow_ops_10ms: u64,

    /// Number of very slow ops (>100ms)
    slow_ops_100ms: u64,

    /// Number of extremely slow ops (>1s)
    slow_ops_1s: u64,
}

impl ThreadOpStats {
    const fn record_op(&mut self, op_ns: u64) {
        if op_ns > self.max_op_ns {
            self.max_op_ns = op_ns;
        }

        if op_ns > 10_000_000 {
            self.slow_ops_10ms += 1;
        }

        if op_ns > 100_000_000 {
            self.slow_ops_100ms += 1;
        }

        if op_ns > 1_000_000_000 {
            self.slow_ops_1s += 1;
        }
    }

    const fn merge(&mut self, other: &Self) {
        if other.max_op_ns > self.max_op_ns {
            self.max_op_ns = other.max_op_ns;
        }

        self.slow_ops_10ms += other.slow_ops_10ms;
        self.slow_ops_100ms += other.slow_ops_100ms;
        self.slow_ops_1s += other.slow_ops_1s;
    }
}

// =============================================================================
// Benchmark Runner
// =============================================================================

struct BenchmarkConfig {
    threads: usize,
    ops_per_thread: usize,
    key_size: usize,
}

struct RunResult {
    elapsed: Duration,
    stats: ThreadOpStats,
    #[cfg(feature = "tracing")]
    parent_wait: ParentWaitStats,
    #[cfg(feature = "tracing")]
    debug: DebugCounters,
}

#[expect(clippy::panic)]
#[expect(clippy::too_many_lines)]
#[expect(clippy::indexing_slicing)]
fn run_benchmark(config: &BenchmarkConfig) -> RunResult {
    let tree = Arc::new(MassTree24::<u64>::new());

    // Reset debug counters (only with tracing)
    #[cfg(feature = "tracing")]
    masstree::reset_debug_counters();

    let start = Instant::now();

    let handles: Vec<_> = (0..config.threads)
        .map(|t| {
            let tree = Arc::clone(&tree);
            let ops = config.ops_per_thread;
            let key_size = config.key_size;

            thread::spawn(move || {
                let mut stats = ThreadOpStats::default();
                let guard = tree.guard();
                let base = t * ops;

                for i in 0..ops {
                    let key_val = (base + i) as u64;

                    // Create key based on size
                    let key: Vec<u8> = match key_size {
                        8 => key_val.to_be_bytes().to_vec(),

                        16 => {
                            let mut k = vec![0u8; 16];
                            k[0..8].copy_from_slice(&key_val.to_be_bytes());
                            k[8..16].copy_from_slice(
                                &key_val.wrapping_mul(0x517c_c1b7_2722_0a95).to_be_bytes(),
                            );

                            k
                        }

                        32 => {
                            let mut k = vec![0u8; 32];
                            k[0..8].copy_from_slice(&key_val.to_be_bytes());
                            k[8..16].copy_from_slice(
                                &key_val.wrapping_mul(0x517c_c1b7_2722_0a95).to_be_bytes(),
                            );
                            k[16..24].copy_from_slice(
                                &key_val.wrapping_mul(0x9e37_79b9_7f4a_7c15).to_be_bytes(),
                            );
                            k[24..32].copy_from_slice(
                                &key_val.wrapping_mul(0xbf58_476d_1ce4_e5b9).to_be_bytes(),
                            );
                            k
                        }

                        _ => panic!("Unsupported key size"),
                    };

                    let op_start = Instant::now();
                    let _ = tree.insert_with_guard(&key, key_val, &guard);
                    let op_elapsed = op_start.elapsed().as_nanos() as u64;

                    stats.record_op(op_elapsed);

                    // Log extremely slow operations in real-time
                    if op_elapsed > 100_000_000 {
                        // >100ms
                        #[cfg(feature = "tracing")]
                        tracing::warn!(
                            thread = t,
                            op_index = i,
                            key_hex = format_args!("{:016x}", key_val),
                            elapsed_ms = op_elapsed as f64 / 1_000_000.0,
                            "SLOW_OP"
                        );

                        #[cfg(not(feature = "tracing"))]
                        eprintln!(
                            "[T{:02}] SLOW_OP: i={} key=0x{:016x} took {:.2}ms",
                            t,
                            i,
                            key_val,
                            op_elapsed as f64 / 1_000_000.0
                        );
                    }
                }

                stats
            })
        })
        .collect();

    // Collect and merge stats
    let mut merged = ThreadOpStats::default();
    for h in handles {
        let thread_stats = h.join().unwrap();
        merged.merge(&thread_stats);
    }

    let elapsed = start.elapsed();
    #[cfg(feature = "tracing")]
    let parent_wait = masstree::get_parent_wait_stats();
    #[cfg(feature = "tracing")]
    let debug = masstree::get_all_debug_counters();

    RunResult {
        elapsed,
        stats: merged,
        #[cfg(feature = "tracing")]
        parent_wait,
        #[cfg(feature = "tracing")]
        debug,
    }
}

fn print_stats(config: &BenchmarkConfig, result: &RunResult, baseline: Duration) {
    let elapsed = result.elapsed;
    let stats = &result.stats;

    let total_ops = config.threads * config.ops_per_thread;
    let ops_per_sec = total_ops as f64 / elapsed.as_secs_f64();

    println!("\n{}", "=".repeat(80));
    println!(
        "RESULTS: {} threads x {} ops = {} total ({}-byte keys)",
        config.threads, config.ops_per_thread, total_ops, config.key_size
    );
    println!("{}", "=".repeat(80));

    println!("\n--- Timing ---");
    println!("Elapsed:     {elapsed:?}");
    println!("Throughput:  {ops_per_sec:.0} ops/sec");

    println!("\n--- Operation Latency ---");
    println!(
        "Max op:      {:.2} ms",
        stats.max_op_ns as f64 / 1_000_000.0
    );
    println!("Slow >10ms:  {}", stats.slow_ops_10ms);
    println!("Slow >100ms: {}", stats.slow_ops_100ms);
    println!("Slow >1s:    {}", stats.slow_ops_1s);

    #[cfg(feature = "tracing")]
    {
        let parent_wait = result.parent_wait;
        let debug = result.debug;
        let elapsed_ms = elapsed.as_secs_f64() * 1000.0;

        println!("\n--- Parent Wait (NULL parent spin loop) ---");
        println!("Hits:        {}", parent_wait.hits);
        println!("Total spins: {}", parent_wait.total_spins);
        println!("Max spins:   {}", parent_wait.max_spins);
        println!("Avg spins:   {:.2}", parent_wait.avg_spins);
        println!("Max wait:    {:.2} ms", parent_wait.max_us / 1000.0);
        println!("Avg wait:    {:.2} us", parent_wait.avg_us);

        let parent_wait_ms = parent_wait.total_us / 1000.0;
        let avg_thread_wait_ms = if config.threads > 0 {
            parent_wait_ms / config.threads as f64
        } else {
            0.0
        };
        println!("Total wait:  {parent_wait_ms:.2} ms (thread-time)");
        println!("Avg/thread:  {avg_thread_wait_ms:.2} ms");

        if parent_wait.hits > 0 && elapsed_ms > 0.0 {
            let avg_thread_pct = (avg_thread_wait_ms / elapsed_ms) * 100.0;
            println!("Avg/thread share: {avg_thread_pct:.1}% of elapsed (approx)");
        }

        println!("\n--- Debug Counters ---");
        println!("Splits:            {}", debug.split);
        println!("CAS success:       {}", debug.cas_insert_success);
        println!("CAS retry:         {}", debug.cas_insert_retry);
        println!("CAS fallback:      {}", debug.cas_insert_fallback);
        println!("Locked inserts:    {}", debug.locked_insert);
        println!("B-link advance:    {}", debug.advance_blink);

        println!("\n--- Anomaly Counters ---");
        println!("B-link should:     {}", debug.blink_should_follow);
        println!("Search not found:  {}", debug.search_not_found);
        println!("Wrong leaf insert: {}", debug.wrong_leaf_insert);
        println!("B-link anomaly:    {}", debug.blink_advance_anomaly);

        if debug.blink_should_follow > 0
            || debug.search_not_found > 0
            || debug.wrong_leaf_insert > 0
            || debug.blink_advance_anomaly > 0
        {
            println!("\n!!! Anomaly counters are non-zero, inspect logs");
        }

        // Check for outliers
        let baseline_ms = baseline.as_secs_f64() * 1000.0;
        if baseline_ms > 0.0 && elapsed_ms > baseline_ms * 3.0 {
            let ratio = elapsed_ms / baseline_ms;
            println!("\n!!! OUTLIER DETECTED: This run was ~{ratio:.1}x slower than median");
        }
    }
}

// =============================================================================
// Main
// =============================================================================

fn main() {
    // Initialize JSON tracing to logs/lock_contention.json
    let _guard = init_json_tracing();

    println!("Lock Contention Profiling");
    println!("=========================\n");

    // Match the benchmark configuration
    let configs = vec![BenchmarkConfig {
        threads: 32,
        ops_per_thread: 100_000,
        key_size: 8,
    }];

    for config in &configs {
        println!(
            "\nRunning: {} threads x {} ops ({}-byte keys)...",
            config.threads, config.ops_per_thread, config.key_size
        );

        // Run multiple iterations to catch outliers
        let mut results: Vec<RunResult> = Vec::new();
        for run in 1..=10 {
            print!("  Run {run}/10... ");
            std::io::Write::flush(&mut std::io::stdout()).unwrap();

            let result = run_benchmark(config);
            println!("{:?}", result.elapsed);

            results.push(result);
        }

        #[expect(clippy::indexing_slicing)]
        let baseline = if results.is_empty() {
            Duration::from_secs(0)
        } else {
            let mut sorted: Vec<Duration> = results.iter().map(|result| result.elapsed).collect();
            sorted.sort_by_key(Duration::as_nanos);
            let mid = sorted.len() / 2;
            if sorted.len() % 2 == 1 {
                sorted[mid]
            } else {
                let lo = sorted[mid - 1].as_secs_f64();
                let hi = sorted[mid].as_secs_f64();

                Duration::from_secs_f64(f64::midpoint(lo, hi))
            }
        };

        // Find the slowest run
        let (slowest_idx, slowest_result) = results
            .iter()
            .enumerate()
            .max_by_key(|(_, result)| result.elapsed.as_nanos())
            .unwrap();

        println!("\n>>> Slowest run was #{} <<<", slowest_idx + 1);
        println!("Baseline (median) run: {baseline:?}");
        print_stats(config, slowest_result, baseline);

        // Also print the fastest for comparison
        let (fastest_idx, fastest_result) = results
            .iter()
            .enumerate()
            .min_by_key(|(_, result)| result.elapsed.as_nanos())
            .unwrap();

        println!(
            "\nFastest run #{}: {:?} (ratio: {:.1}x)",
            fastest_idx + 1,
            fastest_result.elapsed,
            slowest_result.elapsed.as_secs_f64() / fastest_result.elapsed.as_secs_f64()
        );
    }
}
