//! Debug binary for concurrent write benchmarks (05 and 06).
//!
//! Diagnoses hangs/deadlocks in `MassTree` concurrent writes.
//!
//! Run with:
//! ```bash
//! RUST_LOG=masstree=debug cargo run --features tracing
//! ```

#![allow(dead_code)]
#![allow(clippy::unwrap_used)]
#![allow(clippy::indexing_slicing)]
#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::too_many_lines
)]

use masstree::MassTree;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::thread;
use std::time::{Duration, Instant};

// =============================================================================
// Key Generation (from bench_utils.rs)
// =============================================================================

const MULTIPLIERS: [u64; 4] = [
    1,
    0x517c_c1b7_2722_0a95,
    0x9e37_79b9_7f4a_7c15,
    0xbf58_476d_1ce4_e5b9,
];

fn keys<const K: usize>(n: usize) -> Vec<[u8; K]> {
    assert!(K.is_multiple_of(8), "key size must be a multiple of 8");
    assert!((8..=32).contains(&K), "key size must be 8..=32");

    let chunks = K / 8;
    let mut out = Vec::with_capacity(n);

    for i in 0..n {
        let mut key = [0u8; K];
        (0..chunks).for_each(|c| {
            let v = (i as u64).wrapping_mul(MULTIPLIERS[c]);
            let bytes = v.to_be_bytes();
            let start = c * 8;
            key[start..start + 8].copy_from_slice(&bytes);
        });
        out.push(key);
    }
    out
}

fn setup_masstree<const K: usize>(keys: &[[u8; K]]) -> MassTree<u64> {
    let tree = MassTree::new();
    for (i, key) in keys.iter().enumerate() {
        let _ = tree.insert(key, i as u64);
    }
    tree
}

// =============================================================================
// Thread progress tracking for hang detection
// =============================================================================

struct ThreadProgress {
    /// Current operation index for each thread
    current_op: Vec<AtomicUsize>,
    /// Current key being processed by each thread
    current_key: Vec<AtomicU64>,
    /// Last time each thread made progress
    last_progress_ms: Vec<AtomicU64>,
    /// Whether each thread is done
    done: Vec<AtomicBool>,
    /// Start time
    start: Instant,
}

impl ThreadProgress {
    fn new(num_threads: usize) -> Self {
        Self {
            current_op: (0..num_threads).map(|_| AtomicUsize::new(0)).collect(),
            current_key: (0..num_threads).map(|_| AtomicU64::new(0)).collect(),
            last_progress_ms: (0..num_threads).map(|_| AtomicU64::new(0)).collect(),
            done: (0..num_threads).map(|_| AtomicBool::new(false)).collect(),
            start: Instant::now(),
        }
    }

    fn update(&self, thread_id: usize, op: usize, key: u64) {
        self.current_op[thread_id].store(op, Ordering::Relaxed);
        self.current_key[thread_id].store(key, Ordering::Relaxed);
        self.last_progress_ms[thread_id]
            .store(self.start.elapsed().as_millis() as u64, Ordering::Relaxed);
    }

    fn mark_done(&self, thread_id: usize) {
        self.done[thread_id].store(true, Ordering::Relaxed);
    }

    fn report_stuck(&self, timeout_ms: u64) -> Vec<(usize, usize, u64, u64)> {
        let now_ms = self.start.elapsed().as_millis() as u64;
        let mut stuck = Vec::new();

        for i in 0..self.done.len() {
            if self.done[i].load(Ordering::Relaxed) {
                continue;
            }
            let last = self.last_progress_ms[i].load(Ordering::Relaxed);
            if now_ms.saturating_sub(last) > timeout_ms {
                stuck.push((
                    i,
                    self.current_op[i].load(Ordering::Relaxed),
                    self.current_key[i].load(Ordering::Relaxed),
                    now_ms - last,
                ));
            }
        }
        stuck
    }

    fn all_done(&self) -> bool {
        self.done.iter().all(|d| d.load(Ordering::Relaxed))
    }
}

// =============================================================================
// Benchmark 05: Concurrent Writes - Disjoint Ranges
// =============================================================================

fn run_05_disjoint_writes(threads: usize, ops_per_thread: usize) {
    println!("\n{}", "=".repeat(80));
    println!("05: DISJOINT WRITES ({threads} threads, {ops_per_thread} ops/thread)");
    println!("{}", "=".repeat(80));

    let tree = Arc::new(MassTree::<u64>::new());
    let progress = Arc::new(ThreadProgress::new(threads));
    let stop_watchdog = Arc::new(AtomicBool::new(false));

    // Watchdog thread to detect hangs
    let watchdog = {
        let progress = Arc::clone(&progress);
        let stop = Arc::clone(&stop_watchdog);
        thread::spawn(move || {
            while !stop.load(Ordering::Relaxed) {
                thread::sleep(Duration::from_millis(500));
                let stuck = progress.report_stuck(2000); // 2 second timeout
                for (tid, op, key, stall_ms) in &stuck {
                    eprintln!(
                        "!!! STUCK: Thread {tid} at op {op} key=0x{key:016x} for {stall_ms}ms"
                    );
                }
                if progress.all_done() {
                    break;
                }
            }
        })
    };

    let start = Instant::now();

    let handles: Vec<_> = (0..threads)
        .map(|t| {
            let tree = Arc::clone(&tree);
            let progress = Arc::clone(&progress);
            thread::spawn(move || {
                let guard = tree.guard();
                let base = t * ops_per_thread;

                eprintln!(
                    "[T{:02}] Start: keys {}..{}",
                    t,
                    base,
                    base + ops_per_thread - 1
                );

                for i in 0..ops_per_thread {
                    let key_val = (base + i) as u64;
                    let key = key_val.to_be_bytes();

                    // Update progress BEFORE the operation
                    progress.update(t, i, key_val);

                    let op_start = Instant::now();
                    let result = tree.insert_with_guard(&key, key_val, &guard);
                    let op_elapsed = op_start.elapsed();

                    // Log slow operations
                    if op_elapsed > Duration::from_millis(100) {
                        eprintln!("[T{t:02}] SLOW op {i} key=0x{key_val:016x} took {op_elapsed:?}");
                    }

                    // Log every 10000th op (less spam for large runs)
                    if i % 10000 == 0 && i > 0 {
                        eprintln!("[T{t:02}] op {i}/{ops_per_thread}");
                    }

                    if let Err(e) = result {
                        eprintln!("[T{t:02}] ERROR op {i} key=0x{key_val:016x}: {e:?}");
                    }
                }

                progress.mark_done(t);
                eprintln!("[T{t:02}] DONE");
            })
        })
        .collect();

    for h in handles {
        let _ = h.join();
    }

    stop_watchdog.store(true, Ordering::Relaxed);
    let _ = watchdog.join();

    let elapsed = start.elapsed();
    println!(
        "05 DONE: {} ops in {:?} ({:.0} ops/sec), tree.len()={}",
        threads * ops_per_thread,
        elapsed,
        (threads * ops_per_thread) as f64 / elapsed.as_secs_f64(),
        tree.len()
    );
}

// =============================================================================
// Benchmark 06: Concurrent Writes - High Contention
// =============================================================================

fn run_06_contention_writes(threads: usize, ops_per_thread: usize, key_space: usize) {
    println!("\n{}", "=".repeat(80));
    println!(
        "06: CONTENTION WRITES ({threads} threads, {ops_per_thread} ops/thread, {key_space} keys)"
    );
    println!("{}", "=".repeat(80));

    let keys: Arc<Vec<[u8; 8]>> = Arc::new(keys::<8>(key_space));
    let tree = Arc::new(setup_masstree::<8>(keys.as_ref()));
    eprintln!("Pre-populated {} keys", tree.len());

    let progress = Arc::new(ThreadProgress::new(threads));
    let stop_watchdog = Arc::new(AtomicBool::new(false));
    let counter = Arc::new(AtomicUsize::new(0));

    // Watchdog thread
    let watchdog = {
        let progress = Arc::clone(&progress);
        let stop = Arc::clone(&stop_watchdog);
        thread::spawn(move || {
            while !stop.load(Ordering::Relaxed) {
                thread::sleep(Duration::from_millis(500));
                let stuck = progress.report_stuck(2000);
                for (tid, op, key, stall_ms) in &stuck {
                    eprintln!(
                        "!!! STUCK: Thread {tid} at op {op} key=0x{key:016x} for {stall_ms}ms"
                    );
                }
                if progress.all_done() {
                    break;
                }
            }
        })
    };

    let start = Instant::now();

    let handles: Vec<_> = (0..threads)
        .map(|t| {
            let tree = Arc::clone(&tree);
            let keys = Arc::clone(&keys);
            let counter = Arc::clone(&counter);
            let progress = Arc::clone(&progress);
            thread::spawn(move || {
                let guard = tree.guard();
                let mut state = (t as u64).wrapping_mul(0x517c_c1b7_2722_0a95);

                eprintln!(
                    "[T{:02}] Start: {} ops over {} keys",
                    t,
                    ops_per_thread,
                    keys.len()
                );

                for op in 0..ops_per_thread {
                    state = state
                        .wrapping_mul(6_364_136_223_846_793_005)
                        .wrapping_add(1);
                    let idx = (state as usize) % keys.len();
                    let key = &keys[idx];
                    let ikey = u64::from_be_bytes(*key);
                    let val = counter.fetch_add(1, Ordering::Relaxed) as u64;

                    progress.update(t, op, ikey);

                    let op_start = Instant::now();
                    let result = tree.insert_with_guard(key, val, &guard);
                    let op_elapsed = op_start.elapsed();

                    if op_elapsed > Duration::from_millis(100) {
                        eprintln!("[T{t:02}] SLOW op {op} key=0x{ikey:016x} took {op_elapsed:?}");
                    }

                    if op % 500 == 0 {
                        eprintln!("[T{t:02}] op {op}/{ops_per_thread}");
                    }

                    if let Err(e) = result {
                        eprintln!("[T{t:02}] ERROR op {op} key=0x{ikey:016x}: {e:?}");
                    }
                }

                progress.mark_done(t);
                eprintln!("[T{t:02}] DONE");
            })
        })
        .collect();

    for h in handles {
        let _ = h.join();
    }

    stop_watchdog.store(true, Ordering::Relaxed);
    let _ = watchdog.join();

    let elapsed = start.elapsed();
    println!(
        "06 DONE: {} ops in {:?} ({:.0} ops/sec), tree.len()={}",
        threads * ops_per_thread,
        elapsed,
        (threads * ops_per_thread) as f64 / elapsed.as_secs_f64(),
        tree.len()
    );
}

// =============================================================================
// Main
// =============================================================================

// =============================================================================
// Verification Test - Debug insert/get failures
// =============================================================================

fn run_verification_test() {
    use masstree::MassTree24;

    println!("\n{}", "=".repeat(80));
    println!("VERIFICATION TEST (4 threads, 500 ops/thread)");
    println!("{}", "=".repeat(80));

    for run in 0..20 {
        let tree = Arc::new(MassTree24::<u64>::new());
        let failures = Arc::new(AtomicUsize::new(0));

        #[expect(clippy::cast_sign_loss)]
        let handles: Vec<_> = (0..4)
            .map(|t| {
                let tree = Arc::clone(&tree);
                let failures = Arc::clone(&failures);
                thread::spawn(move || {
                    let guard = tree.guard();
                    for i in 0..500 {
                        let key_val = (t * 10000 + i) as u64;
                        let key = key_val.to_be_bytes();

                        let insert_result = tree.insert_with_guard(&key, key_val, &guard);

                        // Immediate verification - try multiple times
                        let get1 = tree.get_with_guard(&key, &guard);
                        if get1.is_none() {
                            // Retry with delays
                            for _ in 0..100 { std::hint::spin_loop(); }
                            let get2 = tree.get_with_guard(&key, &guard);
                            std::thread::sleep(std::time::Duration::from_micros(100));
                            let get3 = tree.get_with_guard(&key, &guard);

                            // Log tree state
                            let tree_len = tree.len();

                            failures.fetch_add(1, Ordering::Relaxed);
                            eprintln!(
                                "!!! VERIFY FAIL: t={t} i={i} key=0x{key_val:016x} insert={insert_result:?} get1={get1:?} get2={get2:?} get3={get3:?} tree_len={tree_len}"
                            );
                            // Stop this thread to reduce noise
                            return;
                        }
                    }
                })
            })
            .collect();

        for h in handles {
            h.join().unwrap();
        }

        let fail_count = failures.load(Ordering::Relaxed);
        if fail_count > 0 {
            eprintln!(
                "Run {}: {} verification failures, tree.len()={}",
                run,
                fail_count,
                tree.len()
            );

            return; // Stop on first failure
        }

        println!("Run {}: OK (tree.len()={})", run, tree.len());
    }

    println!("\nAll 20 runs passed!");
}

#[expect(clippy::cast_sign_loss)]
fn run_parent_wait_benchmark() {
    println!("\n{}", "=".repeat(80));
    println!("PARENT WAIT STATS BENCHMARK");
    println!("{}", "=".repeat(80));

    // Reset counters
    masstree::reset_debug_counters();

    let threads = 32;
    let ops_per_thread = 100_000; // 3.2M total (matches benchmark)

    let tree = Arc::new(masstree::MassTree24::<u64>::new());
    let start = Instant::now();

    let handles: Vec<_> = (0..threads)
        .map(|t| {
            let tree = Arc::clone(&tree);
            thread::spawn(move || {
                let guard = tree.guard();
                let base = t * ops_per_thread;
                for i in 0..ops_per_thread {
                    let key_val = (base + i) as u64;
                    let key = key_val.to_be_bytes();
                    let _ = tree.insert_with_guard(&key, key_val, &guard);
                }
            })
        })
        .collect();

    for h in handles {
        h.join().unwrap();
    }

    let elapsed = start.elapsed();
    let stats = masstree::get_parent_wait_stats();
    let debug = masstree::get_all_debug_counters();

    println!("\nRun completed in {elapsed:?}");
    println!("Tree size: {}", tree.len());
    println!("\n=== Parent Wait Stats ===");
    println!("Hits:        {}", stats.hits);
    println!("Total spins: {}", stats.total_spins);
    println!("Max spins:   {}", stats.max_spins);
    println!("Avg spins:   {:.2}", stats.avg_spins);
    println!(
        "Total wait:  {:.2} us ({:.2} ms)",
        stats.total_us,
        stats.total_us / 1000.0
    );
    println!(
        "Max wait:    {:.2} us ({:.2} ms)",
        stats.max_us,
        stats.max_us / 1000.0
    );
    println!("Avg wait:    {:.2} us", stats.avg_us);
    println!("\n=== Debug Counters ===");
    println!("Splits:      {}", debug.split);
    println!("CAS success: {}", debug.cas_insert_success);
    println!("CAS retry:   {}", debug.cas_insert_retry);
    println!("CAS fallbk:  {}", debug.cas_insert_fallback);
    println!("Locked ins:  {}", debug.locked_insert);
    println!("B-link adv:  {}", debug.advance_blink);
    println!("Wrong leaf:  {}", debug.wrong_leaf_insert);

    // Check if elapsed >> parent_wait - indicates other bottleneck
    let elapsed_ms = elapsed.as_secs_f64() * 1000.0;
    let unexplained_ms = elapsed_ms - (stats.total_us / 1000.0);
    if unexplained_ms > 100.0 && stats.total_us > 0.0 {
        println!(
            "\n!!! WARNING: {:.1}ms unexplained (elapsed={:.1}ms, parent_wait={:.1}ms)",
            unexplained_ms,
            elapsed_ms,
            stats.total_us / 1000.0
        );
    }
}

fn main() {
    masstree::init_tracing();

    eprintln!("MassTree Verification Test");
    eprintln!("===========================");
    eprintln!();

    // Run parent wait stats benchmark
    for run in 1..=10 {
        println!("\n\n>>> Run {run}/10 <<<");
        run_parent_wait_benchmark();
    }
}
