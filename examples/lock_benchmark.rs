//! Benchmark raw lock performance to isolate contention overhead
//!
//! Run with: cargo run --example lock_benchmark --release

use std::hint::spin_loop;
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::{Arc, Barrier};
use std::thread;
use std::time::{Duration, Instant};

const LOCK_BIT: u32 = 1 << 8;

/// Simple spinlock matching Rust Masstree's approach
struct SimpleSpinlock {
    value: AtomicU32,
}

impl SimpleSpinlock {
    fn new() -> Self {
        Self {
            value: AtomicU32::new(0),
        }
    }

    #[inline]
    fn lock(&self) {
        loop {
            let v = self.value.load(Ordering::Relaxed);
            if (v & LOCK_BIT) == 0 {
                if self
                    .value
                    .compare_exchange_weak(v, v | LOCK_BIT, Ordering::Acquire, Ordering::Relaxed)
                    .is_ok()
                {
                    return;
                }
            }
            spin_loop();
        }
    }

    #[inline]
    fn lock_with_yield(&self) {
        let mut spins = 0u32;
        loop {
            let v = self.value.load(Ordering::Relaxed);
            if (v & LOCK_BIT) == 0 {
                if self
                    .value
                    .compare_exchange_weak(v, v | LOCK_BIT, Ordering::Acquire, Ordering::Relaxed)
                    .is_ok()
                {
                    return;
                }
            }
            spins += 1;
            if spins < 4 {
                for _ in 0..spins {
                    spin_loop();
                }
            } else {
                thread::yield_now();
                spins = 0;
            }
        }
    }

    #[inline]
    fn unlock(&self) {
        self.value.fetch_and(!LOCK_BIT, Ordering::Release);
    }
}

fn bench_lock(name: &str, threads: usize, use_yield: bool) {
    let lock = Arc::new(SimpleSpinlock::new());
    let counter = Arc::new(AtomicU32::new(0));
    let barrier = Arc::new(Barrier::new(threads));
    let duration = Duration::from_secs(1);

    let handles: Vec<_> = (0..threads)
        .map(|_| {
            let lock = Arc::clone(&lock);
            let counter = Arc::clone(&counter);
            let barrier = Arc::clone(&barrier);
            thread::spawn(move || {
                barrier.wait();
                let start = Instant::now();
                let mut local = 0u64;

                while Instant::now() - start < duration {
                    if use_yield {
                        lock.lock_with_yield();
                    } else {
                        lock.lock();
                    }
                    // Critical section: increment counter
                    counter.fetch_add(1, Ordering::Relaxed);
                    lock.unlock();
                    local += 1;
                }
                local
            })
        })
        .collect();

    let mut total = 0u64;
    for h in handles {
        total += h.join().unwrap();
    }

    println!(
        "{} ({} threads): {:.2}M ops/sec, {:.2}M per thread",
        name,
        threads,
        total as f64 / 1_000_000.0,
        (total as f64 / threads as f64) / 1_000_000.0
    );
}

fn main() {
    println!("=== Lock Contention Benchmark ===\n");

    println!("Simple spinlock (no yield):");
    for t in [1, 2, 4, 6] {
        bench_lock("  spinlock", t, false);
    }

    println!("\nSpinlock with yield:");
    for t in [1, 2, 4, 6] {
        bench_lock("  spinlock+yield", t, true);
    }
}
