//! Guard overhead comparison: seize vs crossbeam-epoch
//!
//! This benchmark measures the cost of guard creation/destruction per operation.
//!
//! ## Running
//!
//! ```bash
//! cargo run --release --bin guard_overhead --features mimalloc
//! ```

#![allow(clippy::unwrap_used)]

use masstree::MassTree24Inline;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;
use std::thread;
use std::time::{Duration, Instant};

const DURATION_SECS: u64 = 3;
const THREADS: usize = 6;

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

/// MassTree with guard-per-operation (current behavior)
fn bench_masstree_guard_per_op() -> (u64, Duration) {
    let tree = Arc::new(MassTree24Inline::<u64>::new());
    let timeout = Arc::new(AtomicBool::new(false));
    let total_ops = Arc::new(AtomicU64::new(0));

    let timeout_clone = Arc::clone(&timeout);
    thread::spawn(move || {
        thread::sleep(Duration::from_secs(DURATION_SECS));
        timeout_clone.store(true, Ordering::Relaxed);
    });

    let start = Instant::now();

    let handles: Vec<_> = (0..THREADS)
        .map(|_| {
            let tree = Arc::clone(&tree);
            let timeout = Arc::clone(&timeout);
            let total_ops = Arc::clone(&total_ops);
            thread::spawn(move || {
                let mut n = 0u64;
                while !timeout.load(Ordering::Relaxed) {
                    let key = key8(n);
                    // Uses insert() which creates a guard internally
                    let _ = tree.insert(&key, n + 1);
                    n += 1;
                }
                total_ops.fetch_add(n, Ordering::Relaxed);
            })
        })
        .collect();

    for h in handles {
        h.join().unwrap();
    }

    (total_ops.load(Ordering::Relaxed), start.elapsed())
}

/// MassTree with guard reuse (amortized guard cost)
fn bench_masstree_guard_reuse() -> (u64, Duration) {
    let tree = Arc::new(MassTree24Inline::<u64>::new());
    let timeout = Arc::new(AtomicBool::new(false));
    let total_ops = Arc::new(AtomicU64::new(0));

    let timeout_clone = Arc::clone(&timeout);
    thread::spawn(move || {
        thread::sleep(Duration::from_secs(DURATION_SECS));
        timeout_clone.store(true, Ordering::Relaxed);
    });

    let start = Instant::now();

    let handles: Vec<_> = (0..THREADS)
        .map(|_| {
            let tree = Arc::clone(&tree);
            let timeout = Arc::clone(&timeout);
            let total_ops = Arc::clone(&total_ops);
            thread::spawn(move || {
                // Create guard ONCE per batch
                let guard = tree.guard();
                let mut n = 0u64;
                while !timeout.load(Ordering::Relaxed) {
                    let key = key8(n);
                    // Uses insert_with_guard() which reuses the guard
                    let _ = tree.insert_with_guard(&key, n + 1, &guard);
                    n += 1;
                }
                total_ops.fetch_add(n, Ordering::Relaxed);
            })
        })
        .collect();

    for h in handles {
        h.join().unwrap();
    }

    (total_ops.load(Ordering::Relaxed), start.elapsed())
}

/// Pure seize guard creation/destruction overhead
fn bench_seize_guard_only() -> (u64, Duration) {
    let collector = seize::Collector::new();
    let timeout = Arc::new(AtomicBool::new(false));
    let total_ops = Arc::new(AtomicU64::new(0));

    let timeout_clone = Arc::clone(&timeout);
    thread::spawn(move || {
        thread::sleep(Duration::from_secs(DURATION_SECS));
        timeout_clone.store(true, Ordering::Relaxed);
    });

    let start = Instant::now();

    std::thread::scope(|s| {
        for _ in 0..THREADS {
            let timeout = Arc::clone(&timeout);
            let total_ops = Arc::clone(&total_ops);
            let collector = &collector;
            s.spawn(move || {
                let mut n = 0u64;
                while !timeout.load(Ordering::Relaxed) {
                    // Create and drop guard every operation
                    let _guard = collector.enter();
                    n += 1;
                }
                total_ops.fetch_add(n, Ordering::Relaxed);
            });
        }
    });

    (total_ops.load(Ordering::Relaxed), start.elapsed())
}

/// Pure crossbeam-epoch pin/unpin overhead
fn bench_crossbeam_guard_only() -> (u64, Duration) {
    let timeout = Arc::new(AtomicBool::new(false));
    let total_ops = Arc::new(AtomicU64::new(0));

    let timeout_clone = Arc::clone(&timeout);
    thread::spawn(move || {
        thread::sleep(Duration::from_secs(DURATION_SECS));
        timeout_clone.store(true, Ordering::Relaxed);
    });

    let start = Instant::now();

    let handles: Vec<_> = (0..THREADS)
        .map(|_| {
            let timeout = Arc::clone(&timeout);
            let total_ops = Arc::clone(&total_ops);
            thread::spawn(move || {
                let mut n = 0u64;
                while !timeout.load(Ordering::Relaxed) {
                    // Just create and drop guard
                    let _guard = crossbeam_epoch::pin();
                    n += 1;
                }
                total_ops.fetch_add(n, Ordering::Relaxed);
            })
        })
        .collect();

    for h in handles {
        h.join().unwrap();
    }

    (total_ops.load(Ordering::Relaxed), start.elapsed())
}

fn report(name: &str, ops: u64, elapsed: Duration) {
    let secs = elapsed.as_secs_f64();
    let ops_per_sec = ops as f64 / secs;
    let ns_per_op = 1_000_000_000.0 / ops_per_sec;
    println!(
        "{:30} {:8.2}M ops/sec ({:6.1}ns/op)",
        name,
        ops_per_sec / 1_000_000.0,
        ns_per_op
    );
}

fn main() {
    println!("Guard Overhead Benchmark ({} threads, {}s duration)", THREADS, DURATION_SECS);
    println!("=============================================================");

    // Pure guard overhead
    println!("\n--- Pure Guard Creation Overhead ---");
    let (ops, elapsed) = bench_crossbeam_guard_only();
    report("crossbeam-epoch::pin()", ops, elapsed);

    let (ops, elapsed) = bench_seize_guard_only();
    report("seize::Collector::enter()", ops, elapsed);

    // MassTree with different patterns
    println!("\n--- MassTree Insert Patterns ---");
    let (ops, elapsed) = bench_masstree_guard_per_op();
    report("insert() [guard per op]", ops, elapsed);

    let (ops, elapsed) = bench_masstree_guard_reuse();
    report("insert_with_guard() [reused]", ops, elapsed);

    println!();
}
