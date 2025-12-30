//! MassTree vs scc::TreeIndex Comparison
//!
//! Compares performance of MassTree24 against scc::TreeIndex for key workloads.
//!
//! ## Running
//!
//! ```bash
//! cargo run --release --bin tree_compare --features mimalloc -- -j6 -d3
//! cargo run --release --bin tree_compare --features mimalloc -- -j1 -d3  # single-threaded
//! ```

#![allow(clippy::cast_possible_truncation)]
#![allow(clippy::cast_sign_loss)]
#![allow(clippy::unwrap_used)]
#![allow(missing_docs)]

use masstree::MassTree24;
use scc::TreeIndex;
use std::env;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::thread;
use std::time::{Duration, Instant};

// =============================================================================
// Configuration
// =============================================================================

struct Config {
    threads: usize,
    duration_secs: u64,
}

impl Config {
    fn parse() -> Self {
        let args: Vec<String> = env::args().collect();
        let mut threads = 1;
        let mut duration_secs = 3;

        let mut i = 1;
        while i < args.len() {
            if args[i].starts_with("-j") {
                if args[i].len() > 2 {
                    threads = args[i][2..].parse().unwrap_or(1);
                } else if i + 1 < args.len() {
                    i += 1;
                    threads = args[i].parse().unwrap_or(1);
                }
            } else if args[i].starts_with("-d") {
                if args[i].len() > 2 {
                    duration_secs = args[i][2..].parse().unwrap_or(3);
                } else if i + 1 < args.len() {
                    i += 1;
                    duration_secs = args[i].parse().unwrap_or(3);
                }
            }
            i += 1;
        }

        Self {
            threads,
            duration_secs,
        }
    }
}

// =============================================================================
// Key Generation (matches C++ mttest)
// =============================================================================

#[inline]
fn key8(n: u64) -> [u8; 8] {
    let mut buf = [b'0'; 8];
    let mut val = n;
    let mut i = 8;
    if val == 0 {
        return buf;
    }
    while val > 0 && i > 0 {
        i -= 1;
        buf[i] = b'0' + (val % 10) as u8;
        val /= 10;
    }
    buf
}

// =============================================================================
// Test: Sequential Insert then Read (rw3 pattern)
// =============================================================================

fn bench_seq_insert_read_masstree(threads: usize, duration: Duration) -> (u64, Duration) {
    let tree = Arc::new(MassTree24::<u64>::new());
    let timeout = Arc::new(AtomicBool::new(false));
    let total_ops = Arc::new(AtomicU64::new(0));

    // Timeout thread
    let timeout_clone = Arc::clone(&timeout);
    let timeout_handle = thread::spawn(move || {
        thread::sleep(duration);
        timeout_clone.store(true, Ordering::Relaxed);
    });

    let start = Instant::now();

    let handles: Vec<_> = (0..threads)
        .map(|_| {
            let tree = Arc::clone(&tree);
            let timeout = Arc::clone(&timeout);
            let total_ops = Arc::clone(&total_ops);
            thread::spawn(move || {
                // Put phase
                let mut n = 0u64;
                while !timeout.load(Ordering::Relaxed) {
                    let key = key8(n);
                    let _ = tree.insert(&key, n + 1);
                    n += 1;
                }
                // Get phase
                for i in 0..n {
                    let key = key8(i);
                    std::hint::black_box(tree.get(&key));
                }
                total_ops.fetch_add(n * 2, Ordering::Relaxed);
            })
        })
        .collect();

    for h in handles {
        h.join().unwrap();
    }
    timeout_handle.join().unwrap();

    let elapsed = start.elapsed();
    (total_ops.load(Ordering::Relaxed), elapsed)
}

fn bench_seq_insert_read_treeindex(threads: usize, duration: Duration) -> (u64, Duration) {
    let tree: Arc<TreeIndex<[u8; 8], u64>> = Arc::new(TreeIndex::new());
    let timeout = Arc::new(AtomicBool::new(false));
    let total_ops = Arc::new(AtomicU64::new(0));

    // Timeout thread
    let timeout_clone = Arc::clone(&timeout);
    let timeout_handle = thread::spawn(move || {
        thread::sleep(duration);
        timeout_clone.store(true, Ordering::Relaxed);
    });

    let start = Instant::now();

    let handles: Vec<_> = (0..threads)
        .map(|_| {
            let tree = Arc::clone(&tree);
            let timeout = Arc::clone(&timeout);
            let total_ops = Arc::clone(&total_ops);
            thread::spawn(move || {
                // Put phase
                let mut n = 0u64;
                while !timeout.load(Ordering::Relaxed) {
                    let key = key8(n);
                    let _ = tree.insert_sync(key, n + 1);
                    n += 1;
                }
                // Get phase
                let guard = scc::Guard::new();
                for i in 0..n {
                    let key = key8(i);
                    std::hint::black_box(tree.peek(&key, &guard));
                }
                total_ops.fetch_add(n * 2, Ordering::Relaxed);
            })
        })
        .collect();

    for h in handles {
        h.join().unwrap();
    }
    timeout_handle.join().unwrap();

    let elapsed = start.elapsed();
    (total_ops.load(Ordering::Relaxed), elapsed)
}

// =============================================================================
// Test: Read-heavy (90% reads, 10% writes on pre-populated tree)
// =============================================================================

fn bench_read_heavy_masstree(
    threads: usize,
    duration: Duration,
    initial_size: u64,
) -> (u64, Duration) {
    let tree = Arc::new(MassTree24::<u64>::new());

    // Pre-populate
    for i in 0..initial_size {
        let key = key8(i);
        let _ = tree.insert(&key, i + 1);
    }

    let timeout = Arc::new(AtomicBool::new(false));
    let total_ops = Arc::new(AtomicU64::new(0));

    let timeout_clone = Arc::clone(&timeout);
    let timeout_handle = thread::spawn(move || {
        thread::sleep(duration);
        timeout_clone.store(true, Ordering::Relaxed);
    });

    let start = Instant::now();

    let handles: Vec<_> = (0..threads)
        .map(|tid| {
            let tree = Arc::clone(&tree);
            let timeout = Arc::clone(&timeout);
            let total_ops = Arc::clone(&total_ops);
            thread::spawn(move || {
                let mut ops = 0u64;
                let mut i = tid as u64;
                while !timeout.load(Ordering::Relaxed) {
                    let key = key8(i % initial_size);
                    if i % 10 == 0 {
                        // 10% writes
                        let _ = tree.insert(&key, i);
                    } else {
                        // 90% reads
                        std::hint::black_box(tree.get(&key));
                    }
                    ops += 1;
                    i += 1;
                }
                total_ops.fetch_add(ops, Ordering::Relaxed);
            })
        })
        .collect();

    for h in handles {
        h.join().unwrap();
    }
    timeout_handle.join().unwrap();

    let elapsed = start.elapsed();
    (total_ops.load(Ordering::Relaxed), elapsed)
}

fn bench_read_heavy_treeindex(
    threads: usize,
    duration: Duration,
    initial_size: u64,
) -> (u64, Duration) {
    let tree: Arc<TreeIndex<[u8; 8], u64>> = Arc::new(TreeIndex::new());

    // Pre-populate
    for i in 0..initial_size {
        let key = key8(i);
        let _ = tree.insert_sync(key, i + 1);
    }

    let timeout = Arc::new(AtomicBool::new(false));
    let total_ops = Arc::new(AtomicU64::new(0));

    let timeout_clone = Arc::clone(&timeout);
    let timeout_handle = thread::spawn(move || {
        thread::sleep(duration);
        timeout_clone.store(true, Ordering::Relaxed);
    });

    let start = Instant::now();

    let handles: Vec<_> = (0..threads)
        .map(|tid| {
            let tree = Arc::clone(&tree);
            let timeout = Arc::clone(&timeout);
            let total_ops = Arc::clone(&total_ops);
            thread::spawn(move || {
                let guard = scc::Guard::new();
                let mut ops = 0u64;
                let mut i = tid as u64;
                while !timeout.load(Ordering::Relaxed) {
                    let key = key8(i % initial_size);
                    if i % 10 == 0 {
                        // 10% writes
                        let _ = tree.insert_sync(key, i);
                    } else {
                        // 90% reads
                        std::hint::black_box(tree.peek(&key, &guard));
                    }
                    ops += 1;
                    i += 1;
                }
                total_ops.fetch_add(ops, Ordering::Relaxed);
            })
        })
        .collect();

    for h in handles {
        h.join().unwrap();
    }
    timeout_handle.join().unwrap();

    let elapsed = start.elapsed();
    (total_ops.load(Ordering::Relaxed), elapsed)
}

// =============================================================================
// Test: Write-only (insert throughput)
// =============================================================================

fn bench_write_only_masstree(threads: usize, duration: Duration) -> (u64, Duration) {
    let tree = Arc::new(MassTree24::<u64>::new());
    let timeout = Arc::new(AtomicBool::new(false));
    let total_ops = Arc::new(AtomicU64::new(0));

    let timeout_clone = Arc::clone(&timeout);
    let timeout_handle = thread::spawn(move || {
        thread::sleep(duration);
        timeout_clone.store(true, Ordering::Relaxed);
    });

    let start = Instant::now();

    let handles: Vec<_> = (0..threads)
        .map(|tid| {
            let tree = Arc::clone(&tree);
            let timeout = Arc::clone(&timeout);
            let total_ops = Arc::clone(&total_ops);
            thread::spawn(move || {
                let mut ops = 0u64;
                // Each thread uses different key range to reduce contention
                let base = (tid as u64) * 100_000_000;
                while !timeout.load(Ordering::Relaxed) {
                    let key = key8(base + ops);
                    let _ = tree.insert(&key, ops + 1);
                    ops += 1;
                }
                total_ops.fetch_add(ops, Ordering::Relaxed);
            })
        })
        .collect();

    for h in handles {
        h.join().unwrap();
    }
    timeout_handle.join().unwrap();

    let elapsed = start.elapsed();
    (total_ops.load(Ordering::Relaxed), elapsed)
}

fn bench_write_only_treeindex(threads: usize, duration: Duration) -> (u64, Duration) {
    let tree: Arc<TreeIndex<[u8; 8], u64>> = Arc::new(TreeIndex::new());
    let timeout = Arc::new(AtomicBool::new(false));
    let total_ops = Arc::new(AtomicU64::new(0));

    let timeout_clone = Arc::clone(&timeout);
    let timeout_handle = thread::spawn(move || {
        thread::sleep(duration);
        timeout_clone.store(true, Ordering::Relaxed);
    });

    let start = Instant::now();

    let handles: Vec<_> = (0..threads)
        .map(|tid| {
            let tree = Arc::clone(&tree);
            let timeout = Arc::clone(&timeout);
            let total_ops = Arc::clone(&total_ops);
            thread::spawn(move || {
                let mut ops = 0u64;
                let base = (tid as u64) * 100_000_000;
                while !timeout.load(Ordering::Relaxed) {
                    let key = key8(base + ops);
                    let _ = tree.insert_sync(key, ops + 1);
                    ops += 1;
                }
                total_ops.fetch_add(ops, Ordering::Relaxed);
            })
        })
        .collect();

    for h in handles {
        h.join().unwrap();
    }
    timeout_handle.join().unwrap();

    let elapsed = start.elapsed();
    (total_ops.load(Ordering::Relaxed), elapsed)
}

// =============================================================================
// Reporting
// =============================================================================

fn report(test: &str, impl_name: &str, threads: usize, elapsed: Duration, total_ops: u64) {
    let ops_per_sec = total_ops as f64 / elapsed.as_secs_f64();
    println!(
        "{:<20} {:>12} {:>2}t: {:>8.2}M ops/sec ({} ops in {:.2}s)",
        test,
        impl_name,
        threads,
        ops_per_sec / 1_000_000.0,
        total_ops,
        elapsed.as_secs_f64()
    );
}

fn compare(test: &str, threads: usize, masstree: (u64, Duration), treeindex: (u64, Duration)) {
    let mt_rate = masstree.0 as f64 / masstree.1.as_secs_f64();
    let ti_rate = treeindex.0 as f64 / treeindex.1.as_secs_f64();
    let ratio = mt_rate / ti_rate;

    report(test, "MassTree24", threads, masstree.1, masstree.0);
    report(test, "TreeIndex", threads, treeindex.1, treeindex.0);

    let winner = if ratio > 1.0 { "MassTree" } else { "TreeIndex" };
    let factor = if ratio > 1.0 { ratio } else { 1.0 / ratio };
    println!("  => {} is {:.2}x faster\n", winner, factor);
}

// =============================================================================
// Main
// =============================================================================

fn main() {
    let config = Config::parse();
    let duration = Duration::from_secs(config.duration_secs);
    let threads = config.threads;

    println!("===============================================================================");
    println!("MassTree24 vs scc::TreeIndex Comparison");
    println!("Threads: {}, Duration: {}s", threads, config.duration_secs);
    println!("===============================================================================\n");

    // Test 1: Sequential Insert + Read (rw3 pattern)
    println!("--- Test 1: Sequential Insert then Read (rw3 pattern) ---");
    let mt = bench_seq_insert_read_masstree(threads, duration);
    let ti = bench_seq_insert_read_treeindex(threads, duration);
    compare("seq_insert_read", threads, mt, ti);

    // Test 2: Read-heavy (90% read, 10% write)
    println!("--- Test 2: Read-Heavy (90% read, 10% write, 1M keys) ---");
    let mt = bench_read_heavy_masstree(threads, duration, 1_000_000);
    let ti = bench_read_heavy_treeindex(threads, duration, 1_000_000);
    compare("read_heavy_90_10", threads, mt, ti);

    // Test 3: Write-only (disjoint key ranges)
    println!("--- Test 3: Write-Only (disjoint key ranges) ---");
    let mt = bench_write_only_masstree(threads, duration);
    let ti = bench_write_only_treeindex(threads, duration);
    compare("write_only", threads, mt, ti);

    println!("===============================================================================");
}
