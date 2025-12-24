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

fn main() {
    masstree::init_tracing();

    eprintln!("MassTree Concurrent Write Hang Detector");
    eprintln!("========================================");
    eprintln!("Watchdog will report any thread stuck for >2 seconds.");
    eprintln!();

    // Run the exact benchmark config that shows variance: 8 threads, 50k ops
    eprintln!("=== EXACT BENCHMARK CONFIG: 8 threads, 50k ops/thread ===");
    for run in 1..=10 {
        eprintln!("\n--- Run {run}/10 ---");
        run_05_disjoint_writes(8, 50_000);
    }

    eprintln!("\nAll tests completed!");
}
