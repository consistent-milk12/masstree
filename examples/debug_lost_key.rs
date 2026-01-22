//! Debug reproduction for the "insert then get loses key" flaky test.
//!
//! This reproduces the bug where:
//! - Thread inserts a key, insert returns Ok
//! - Immediate get returns None
//! - But the key IS findable via scan
//!
//! Run with: `cargo run --example debug_lost_key --release --features tracing`
//!
//! Stress mode (runs until failure):
//!   `cargo run --example debug_lost_key --release --features tracing -- --stress`
//!
//! For extra routing debug:
//!   `cargo run --example debug_lost_key --release --features "tracing,debug-routing"`

#![expect(clippy::unwrap_used)]
#![expect(clippy::indexing_slicing)]
#![expect(clippy::cast_possible_truncation, clippy::cast_precision_loss)]

use masstree::{MassTree, RangeBound};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::thread;
use std::time::{Duration, Instant};

#[cfg(feature = "tracing")]
use tracing_subscriber::{EnvFilter, fmt, prelude::*};

// Match the failing unit test parameters exactly
const NUM_THREADS: usize = 4;
const OPS_PER_THREAD: usize = 500;
const MAX_RUNS: usize = 5000;

// Key spacing strategies to trigger different split patterns
const KEY_STRATEGIES: &[&str] = &["interleaved", "sequential", "reverse", "random"];

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let stress_mode = args.iter().any(|a| a == "--stress");
    let strategy_arg = args
        .iter()
        .position(|a| a == "--strategy")
        .and_then(|i| args.get(i + 1))
        .map(String::as_str);

    println!("Debug: Reproducing lost key bug");
    println!("Threads: {NUM_THREADS}, Ops/thread: {OPS_PER_THREAD}");

    if stress_mode {
        println!("Mode: STRESS (runs until failure)\n");
        run_stress_mode(strategy_arg);
    } else {
        println!("Running up to {MAX_RUNS} iterations...\n");
        run_normal_mode(strategy_arg);
    }
}

fn run_stress_mode(strategy: Option<&str>) {
    let start = Instant::now();
    let mut run = 0u64;

    loop {
        run += 1;

        // Cycle through strategies if none specified
        let strat = strategy.unwrap_or(KEY_STRATEGIES[(run as usize) % KEY_STRATEGIES.len()]);

        if let Some(failure) = run_single_iteration(run, strat) {
            println!("\n=== FAILURE at run {run} ({strat} strategy) ===");
            print_failure(&failure);
            diagnose_around_key(&failure.tree, failure.key_val);

            let elapsed = start.elapsed();
            println!(
                "\nFound bug after {} runs in {:.2}s",
                run,
                elapsed.as_secs_f64()
            );
            return;
        }

        if run.is_multiple_of(100) {
            let elapsed = start.elapsed();
            println!(
                "Completed {} runs in {:.2}s ({:.0} runs/sec)",
                run,
                elapsed.as_secs_f64(),
                run as f64 / elapsed.as_secs_f64()
            );
        }
    }
}

fn run_normal_mode(strategy: Option<&str>) {
    let start = Instant::now();
    let mut failures = 0;

    for run in 0..MAX_RUNS as u64 {
        let strat = strategy.unwrap_or(KEY_STRATEGIES[(run as usize) % KEY_STRATEGIES.len()]);

        if let Some(failure) = run_single_iteration(run, strat) {
            failures += 1;
            println!("\n=== FAILURE #{failures} at run {run} ({strat}) ===");
            print_failure(&failure);

            // Re-run with tracing if it's a timing bug
            if failure.found_after_join {
                println!("\n--- Re-running with tracing to capture the race ---");
                run_traced_reproduction(failure.thread_id, failure.iteration, strat);
            }

            diagnose_around_key(&failure.tree, failure.key_val);
            println!();

            if failures >= 5 {
                break;
            }
        }

        if (run + 1) % 500 == 0 {
            println!("Completed {} runs, {} failures so far", run + 1, failures);
        }
    }

    let elapsed = start.elapsed();
    println!(
        "\nFinished {} runs in {:.2}s",
        MAX_RUNS,
        elapsed.as_secs_f64()
    );
    println!("Total failures: {failures}");

    if failures == 0 {
        println!("Bug did not reproduce. Try: --stress mode or increasing MAX_RUNS");
    }
}

fn print_failure(failure: &Failure) {
    println!("Thread: {}", failure.thread_id);
    println!("Iteration: {}", failure.iteration);
    println!(
        "Key value: {} (0x{:016x})",
        failure.key_val, failure.key_val
    );
    println!("Tree len: {}", failure.tree_len);
    println!("Found via scan: {}", failure.found_via_scan);
    println!(
        "Found via get after threads joined: {}",
        failure.found_after_join
    );
    println!(
        "Found via get after 1ms delay: {}",
        failure.found_after_delay
    );

    if failure.found_via_scan && !failure.found_after_join {
        println!("\n*** BUG: Key in scan but not get() - ROUTING BUG ***");
    } else if failure.found_after_join || failure.found_after_delay {
        println!("\n*** BUG: Key visible later - MEMORY ORDERING BUG ***");
    } else if failure.found_via_scan {
        println!("\n*** BUG: Key in scan, visible after delay - VISIBILITY BUG ***");
    } else {
        println!("\n*** BUG: Key truly lost - INSERT CORRUPTION ***");
    }
}

#[cfg(feature = "tracing")]
fn run_traced_reproduction(failing_thread: usize, failing_iteration: usize, strategy: &str) {
    let subscriber = tracing_subscriber::registry()
        .with(EnvFilter::new("masstree=debug"))
        .with(fmt::layer().with_target(true).with_thread_ids(true));

    let _guard = tracing::subscriber::set_default(subscriber);

    let key_val: u64 = compute_key(failing_thread, failing_iteration, strategy);
    let key = key_val.to_be_bytes();

    println!("Traced run: inserting key {key_val} (0x{key_val:016x}) with strategy '{strategy}'");

    let tree = MassTree::<u64>::new();
    let guard = tree.guard();

    // Insert keys up to and including the failing one
    for t in 0..NUM_THREADS {
        let max_i = if t == failing_thread {
            failing_iteration + 1
        } else {
            failing_iteration.min(OPS_PER_THREAD)
        };

        for i in 0..max_i {
            let kv = compute_key(t, i, strategy);
            let k = kv.to_be_bytes();
            let _ = tree.insert_with_guard(&k, kv, &guard);
        }
    }

    println!("Tree len after inserts: {}", tree.len());

    let result = tree.get_with_guard(&key, &guard);
    println!("get() result: {result:?}");

    if result.is_none() {
        println!("Key still missing in traced run!");
    } else {
        println!("Key found in traced run (race did not reproduce)");
    }
}

#[cfg(not(feature = "tracing"))]
fn run_traced_reproduction(_failing_thread: usize, _failing_iteration: usize, _strategy: &str) {
    println!("(tracing feature not enabled - rebuild with --features tracing)");
}

/// Compute key value based on strategy
fn compute_key(thread: usize, iteration: usize, strategy: &str) -> u64 {
    match strategy {
        "interleaved" => {
            // Keys interleave across threads: 0, 10000, 20000, 1, 10001, 20001, ...
            (thread * 10_000 + iteration) as u64
        }
        "sequential" => {
            // All threads write to same key space sequentially
            (thread * OPS_PER_THREAD + iteration) as u64
        }
        "reverse" => {
            // Reverse order to trigger different split patterns
            let base = (thread * 10_000) as u64;
            base + (OPS_PER_THREAD - 1 - iteration) as u64
        }
        "random" => {
            // Pseudo-random but deterministic
            let seed = (thread as u64 * 0x9E37_79B9_7F4A_7C15) ^ (iteration as u64);
            seed.wrapping_mul(0x517C_C1B7_2722_0A95) % 100_000
        }
        _ => (thread * 10_000 + iteration) as u64,
    }
}

struct Failure {
    thread_id: usize,
    iteration: usize,
    key_val: u64,
    tree_len: usize,
    found_via_scan: bool,
    found_after_join: bool,
    found_after_delay: bool,
    tree: Arc<MassTree<u64>>,
}

fn run_single_iteration(_run: u64, strategy: &str) -> Option<Failure> {
    let tree = Arc::new(MassTree::<u64>::new());
    let found_bug = Arc::new(AtomicBool::new(false));
    let missing_key = Arc::new(AtomicU64::new(u64::MAX));
    let missing_thread = Arc::new(AtomicUsize::new(usize::MAX));
    let missing_i = Arc::new(AtomicUsize::new(usize::MAX));

    // Pre-compute to avoid closure capture issues
    let strategy_owned = strategy.to_string();

    let handles: Vec<_> = (0..NUM_THREADS)
        .map(|t| {
            let tree = Arc::clone(&tree);
            let found_bug = Arc::clone(&found_bug);
            let missing_key = Arc::clone(&missing_key);
            let missing_thread = Arc::clone(&missing_thread);
            let missing_i = Arc::clone(&missing_i);
            let strat = strategy_owned.clone();

            thread::spawn(move || {
                let guard = tree.guard();

                for i in 0..OPS_PER_THREAD {
                    let key_val = compute_key(t, i, &strat);
                    let key = key_val.to_be_bytes();

                    // Insert the key
                    let insert_result = tree.insert_with_guard(&key, key_val, &guard);

                    if insert_result.is_err() {
                        continue;
                    }

                    // Immediately try to get it back
                    let get_result = tree.get_with_guard(&key, &guard);

                    if get_result.is_none() {
                        // Double-check with a memory fence
                        std::sync::atomic::fence(Ordering::SeqCst);
                        let retry = tree.get_with_guard(&key, &guard);

                        if retry.is_none() {
                            // Record the first failure
                            if !found_bug.swap(true, Ordering::SeqCst) {
                                missing_key.store(key_val, Ordering::SeqCst);
                                missing_thread.store(t, Ordering::SeqCst);
                                missing_i.store(i, Ordering::SeqCst);
                            }
                            break;
                        }
                    }
                }
            })
        })
        .collect();

    for h in handles {
        h.join().unwrap();
    }

    if found_bug.load(Ordering::SeqCst) {
        let key_val = missing_key.load(Ordering::SeqCst);
        let thread_id = missing_thread.load(Ordering::SeqCst);
        let iteration = missing_i.load(Ordering::SeqCst);
        let key = key_val.to_be_bytes();

        // Check via scan (different traversal path)
        let found_via_scan = scan_for_key(&tree, key_val);

        // Check via get AFTER all threads joined
        let guard = tree.guard();
        let found_after_join = tree.get_with_guard(&key, &guard).is_some();
        drop(guard);

        // Wait a bit and try again
        thread::sleep(Duration::from_millis(1));
        let guard = tree.guard();
        let found_after_delay = tree.get_with_guard(&key, &guard).is_some();
        drop(guard);

        Some(Failure {
            thread_id,
            iteration,
            key_val,
            tree_len: tree.len(),
            found_via_scan,
            found_after_join,
            found_after_delay,
            tree,
        })
    } else {
        None
    }
}

fn scan_for_key(tree: &MassTree<u64>, target: u64) -> bool {
    let guard = tree.guard();
    let target_bytes = target.to_be_bytes();

    let mut found = false;
    tree.scan(
        RangeBound::Included(&target_bytes),
        RangeBound::Included(&target_bytes),
        |key: &[u8], value| {
            if key == target_bytes && *value == target {
                found = true;
            }
            false
        },
        &guard,
    );

    found
}

fn diagnose_around_key(tree: &MassTree<u64>, target: u64) {
    let guard = tree.guard();

    println!("\n--- Diagnosis ---");

    let key = target.to_be_bytes();
    let get_result = tree.get_with_guard(&key, &guard);
    println!("get() now returns: {get_result:?}");

    // Scan keys around the target
    let start_key = target.saturating_sub(5).to_be_bytes();
    let end_key = (target + 10).to_be_bytes();

    println!("\nKeys around target (via scan):");
    let mut count = 0;
    tree.scan(
        RangeBound::Included(&start_key),
        RangeBound::Included(&end_key),
        |key_bytes: &[u8], value| {
            if count >= 20 {
                return false;
            }

            let mut arr = [0u8; 8];
            if key_bytes.len() == 8 {
                arr.copy_from_slice(key_bytes);
            }
            let key_val = u64::from_be_bytes(arr);

            let marker = if key_val == target { " <-- TARGET" } else { "" };
            println!("  key=0x{key_val:016x} ({key_val:>6}), value={value}{marker}");

            count += 1;
            true
        },
        &guard,
    );

    if count == 0 {
        println!("  (no keys found in scan range)");
    }

    // Check neighboring keys via get
    println!("\nNeighboring keys via get():");
    for offset in -3i64..=3 {
        let check_val = (target.cast_signed() + offset).cast_unsigned();
        let check_key = check_val.to_be_bytes();
        let result = tree.get_with_guard(&check_key, &guard);
        let marker = if offset == 0 { " <-- TARGET" } else { "" };
        println!("  key=0x{check_val:016x} ({check_val:>6}): {result:?}{marker}");
    }

    // Try multiple gets with delays to see if it eventually appears
    println!("\nRetry get() with delays:");
    for delay_ms in [0, 1, 5, 10, 50] {
        if delay_ms > 0 {
            thread::sleep(Duration::from_millis(delay_ms));
        }
        let result = tree.get_with_guard(&key, &guard);
        println!("  after {delay_ms}ms: {result:?}");
        if result.is_some() {
            println!("  ^ Key appeared after {delay_ms}ms delay!");
            break;
        }
    }
}
