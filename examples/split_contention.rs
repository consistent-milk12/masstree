//! Test to measure split contention on sequential inserts
//!
//! Run with: `cargo run --example split_contention --release --features mimalloc`

#![allow(
    clippy::cast_precision_loss,
    clippy::unwrap_used,
    clippy::indexing_slicing
)]

use masstree::MassTree15Inline;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Barrier};
use std::thread;
use std::time::{Duration, Instant};

const fn key8(n: u64) -> [u8; 8] {
    let mut buf = [b'0'; 8];
    let mut i = 8;
    let mut m = n;

    if m == 0 {
        return buf;
    }

    while m > 0 && i > 0 {
        i -= 1;
        buf[i] = b'0' + (m % 10) as u8;
        m /= 10;
    }

    buf
}

fn main() {
    for threads in [1, 2, 4, 6] {
        run_test(threads);
    }
}

fn run_test(threads: usize) {
    let duration = Duration::from_secs(1);

    let tree = Arc::new(MassTree15Inline::<u64>::new());
    let barrier = Arc::new(Barrier::new(threads));
    let total_ops = Arc::new(AtomicU64::new(0));

    #[expect(unused_variables)]
    let handles: Vec<_> = (0..threads)
        .map(|_tid| {
            let tree = Arc::clone(&tree);
            let barrier = Arc::clone(&barrier);
            let total_ops = Arc::clone(&total_ops);
            thread::spawn(move || {
                let guard = tree.guard();
                barrier.wait();

                let start = Instant::now();
                let deadline = start + duration;
                let mut n = 0u64;

                while Instant::now() < deadline {
                    let key = key8(n);
                    let _ = tree.insert_with_guard(&key, n + 1, &guard);
                    n += 1;
                }

                total_ops.fetch_add(n, Ordering::Relaxed);
                n
            })
        })
        .collect();

    let total = total_ops.load(Ordering::Relaxed);
    let tree_size = tree.len();

    println!("=== {threads} threads ===");
    println!("  Total ops: {total}");
    println!("  Tree size: {tree_size}");
    println!("  Ops/sec: {:.2}M", total as f64 / 1_000_000.0);
    println!(
        "  Contention ratio: {:.1}x",
        total as f64 / tree_size as f64
    );
    println!(
        "  Per-thread: {:.2}M ops/sec",
        (total as f64 / threads as f64) / 1_000_000.0
    );
    println!();
}
