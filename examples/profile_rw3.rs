//! Profile rw3 put phase (sequential inserts)
//!
//! Usage:
//! ```bash
//! # Build with profiling symbols
//! cargo build --profile profiling --example profile_rw3 --features mimalloc
//!
//! # Run with perf (single-threaded for cleaner profile)
//! perf record -g --call-graph dwarf -F 997 ./target/profiling/examples/profile_rw3
//!
//! # Multi-threaded (matches rw3 workload)
//! perf record -g --call-graph dwarf -F 997 ./target/profiling/examples/profile_rw3 -j12
//!
//! # Analyze
//! perf report --stdio --sort symbol -g none --percent-limit 2
//!
//! # Flamegraph
//! perf script | stackcollapse-perf.pl | flamegraph.pl > rw3_put.svg
//! ```

#![allow(
    clippy::unwrap_used,
    clippy::indexing_slicing,
    reason = "fail fast in tests"
)]
#![expect(clippy::needless_collect, reason = "necessary for handles")]
#![expect(clippy::cast_precision_loss)]

use masstree::MassTree15Inline;
use std::hint::black_box;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Barrier};
use std::thread;
use std::time::{Duration, Instant};

/// C++ compatible `quick_istr`: decimal string with minimum length
struct QuickIstr {
    buf: [u8; 24],
    len: usize,
}

impl QuickIstr {
    const fn with_minlen(n: u64, minlen: usize) -> Self {
        let mut buf = [b'0'; 24];
        let mut val = n;
        let mut pos = 23;

        // Write digits right-to-left
        loop {
            buf[pos] = b'0' + (val % 10) as u8;
            val /= 10;
            if val == 0 {
                break;
            }
            pos -= 1;
        }

        // Calculate actual length
        let digit_len = 24 - pos;
        let len = if digit_len < minlen {
            minlen
        } else {
            digit_len
        };
        let start = 24 - len;

        // Shift to beginning
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

const fn key8(n: u64) -> QuickIstr {
    QuickIstr::with_minlen(n, 8)
}

static TIMEOUT: AtomicBool = AtomicBool::new(false);

fn main() {
    let args: Vec<String> = std::env::args().collect();

    // Parse arguments
    let mut threads = 1;
    let mut duration_secs = 10;
    let mut ops: Option<u64> = None;

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
            _ => {}
        }
        i += 1;
    }

    println!("=== rw3 PUT phase profiling ===");
    println!("Threads: {threads}");

    if let Some(n) = ops {
        println!("Mode: Fixed ops ({n} per thread)");
        run_fixed_ops(threads, n);
    } else {
        println!("Mode: Duration-based ({duration_secs}s)");
        run_duration_based(threads, Duration::from_secs(duration_secs));
    }
}

/// Run for fixed number of operations (better for profiling)
fn run_fixed_ops(threads: usize, ops_per_thread: u64) {
    let tree = Arc::new(MassTree15Inline::<u64>::new());
    let barrier = Arc::new(Barrier::new(threads));

    let start = Instant::now();

    let handles: Vec<_> = (0..threads)
        .map(|tid| {
            let tree = Arc::clone(&tree);
            let barrier = Arc::clone(&barrier);
            thread::spawn(move || {
                // Pin to core if possible
                if let Some(core_ids) = core_affinity::get_core_ids()
                    && !core_ids.is_empty()
                {
                    let core = core_ids[tid % core_ids.len()];
                    core_affinity::set_for_current(core);
                }

                let guard = tree.guard();
                barrier.wait();

                // Sequential inserts (rw3 pattern)
                for i in 0..ops_per_thread {
                    let key = key8(i);
                    let _ = black_box(tree.insert_with_guard(key.as_bytes(), i + 1, &guard));
                }

                ops_per_thread
            })
        })
        .collect();

    let total_ops: u64 = handles.into_iter().map(|h| h.join().unwrap()).sum();
    let elapsed = start.elapsed();

    let mops = total_ops as f64 / elapsed.as_secs_f64() / 1_000_000.0;
    println!(
        "Completed: {} ops in {:.2}s = {:.2} Mops/s",
        total_ops,
        elapsed.as_secs_f64(),
        mops
    );
}

/// Run for fixed duration (matches mttest behavior)
fn run_duration_based(threads: usize, duration: Duration) {
    let tree = Arc::new(MassTree15Inline::<u64>::new());
    let barrier = Arc::new(Barrier::new(threads));

    // Timer thread
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
            thread::spawn(move || {
                // Pin to core if possible
                if let Some(core_ids) = core_affinity::get_core_ids()
                    && !core_ids.is_empty()
                {
                    let core = core_ids[tid % core_ids.len()];
                    core_affinity::set_for_current(core);
                }

                let guard = tree.guard();
                barrier.wait();

                let mut n = 0u64;
                while !TIMEOUT.load(Ordering::Relaxed) {
                    let key = key8(n);
                    let _ = black_box(tree.insert_with_guard(key.as_bytes(), n + 1, &guard));
                    n += 1;
                }

                n
            })
        })
        .collect();

    let total_ops: u64 = handles.into_iter().map(|h| h.join().unwrap()).sum();
    let elapsed = start.elapsed();

    let mops = total_ops as f64 / elapsed.as_secs_f64() / 1_000_000.0;
    println!(
        "Completed: {} ops in {:.2}s = {:.2} Mops/s",
        total_ops,
        elapsed.as_secs_f64(),
        mops
    );
}
