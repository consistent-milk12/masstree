//! C++ Masstree Compatibility Benchmarks
//!
//! Direct port of C++ mttest benchmark suite. Reports ops/sec like C++.
//!
//! ## Running
//!
//! ```bash
//! # Run all tests with 6 threads, 3 second duration
//! cargo bench --bench cpp_comp --features mimalloc -- -j6 -d3
//! # (or: cargo run --release --bin cpp_comp --features mimalloc -- -j6 -d3)
//!
//! # Run specific test
//! cargo bench --bench cpp_comp --features mimalloc -- -j6 -d3 rw3
//! # (or: cargo run --release --bin cpp_comp --features mimalloc -- -j6 -d3 rw3)
//!
//! # Compare with C++ (in reference/ directory)
//! ./mttest -j6 -d3 rw3
//! ```

#![allow(clippy::cast_possible_truncation)]
#![allow(clippy::cast_sign_loss)]
#![allow(clippy::indexing_slicing)]
#![allow(clippy::unwrap_used)]
#![allow(clippy::pedantic)]

use masstree::{MassTree24, MassTree24Inline};
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
    tests: Vec<String>,
}

impl Config {
    fn parse() -> Self {
        let args: Vec<String> = env::args().collect();
        let mut threads = 1;
        let mut duration_secs = 3;
        let mut tests = Vec::new();

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
            } else if !args[i].starts_with('-') {
                tests.push(args[i].clone());
            }
            i += 1;
        }

        if tests.is_empty() {
            tests = vec![
                "rw1".into(),
                "rw2".into(),
                "rw2g90".into(),
                "rw2g98".into(),
                "rw3".into(),
                "rw4".into(),
                "same".into(),
                "wscale".into(),
                "rscale".into(),
                "uscale".into(),
                "rw1long".into(),
                "rwsmall24".into(),
            ];
        }

        Self {
            threads,
            duration_secs,
            tests,
        }
    }
}

// =============================================================================
// C++ compatible LCG RNG (from kvrandom.hh)
// =============================================================================

struct KvRandom {
    seed: u32,
}

impl KvRandom {
    const A: u32 = 1_664_525;
    const C: u32 = 1_013_904_223;
    const FIRST_SEED: u64 = 31949;

    const fn new(seed: u64) -> Self {
        Self { seed: seed as u32 }
    }

    const fn seed(&mut self, seed: u64) {
        self.seed = seed as u32;
    }

    #[inline]
    const fn lcg_step(&mut self) -> u32 {
        self.seed = self.seed.wrapping_mul(Self::A).wrapping_add(Self::C);
        self.seed
    }

    const fn rand(&mut self) -> u32 {
        self.lcg_step();
        let x0 = self.lcg_step();
        self.lcg_step();
        let x1 = self.lcg_step();
        (x0 >> 15) | ((x1 & 0x7FFE) << 16)
    }

    fn bernoulli(&mut self, p: f64) -> bool {
        (f64::from(self.rand()) / f64::from(0x7FFF_FFFF)) < p
    }

    const fn uniform(&mut self, max: u32) -> u32 {
        self.rand() % max
    }
}

// =============================================================================
// Key generation (ASCII decimal like C++ quick_istr)
// =============================================================================

#[inline]
fn quick_istr<const N: usize>(mut n: u64) -> [u8; N] {
    let mut buf = [b'0'; N];
    let mut i = N;
    if n == 0 {
        return buf;
    }
    while n > 0 && i > 0 {
        i -= 1;
        buf[i] = b'0' + (n % 10) as u8;
        n /= 10;
    }
    buf
}

#[inline]
fn key8(n: u64) -> [u8; 8] {
    quick_istr::<8>(n)
}

#[inline]
fn key_var(n: u64) -> Vec<u8> {
    if n == 0 {
        return vec![b'0'];
    }
    let mut buf = Vec::with_capacity(20);
    let mut m = n;
    while m > 0 {
        buf.push(b'0' + (m % 10) as u8);
        m /= 10;
    }
    buf.reverse();
    buf
}

// =============================================================================
// Result reporting
// =============================================================================

fn report(test_name: &str, threads: usize, duration: Duration, ops: u64) {
    let secs = duration.as_secs_f64();
    let ops_per_sec = ops as f64 / secs;
    let m_ops = ops_per_sec / 1_000_000.0;
    println!("{test_name:12} {threads:2} threads: {m_ops:8.2}M ops/sec ({ops} ops in {secs:.2}s)");
}

// =============================================================================
// RW1: Random keys - insert then shuffled read
// =============================================================================

fn run_rw1(threads: usize, duration_secs: u64) {
    let tree = Arc::new(MassTree24::<u64>::new());
    let timeout = Arc::new(AtomicBool::new(false));
    let total_ops = Arc::new(AtomicU64::new(0));

    let start = Instant::now();

    let handles: Vec<_> = (0..threads)
        .map(|tid| {
            let tree = Arc::clone(&tree);
            let timeout = Arc::clone(&timeout);
            let total_ops = Arc::clone(&total_ops);
            thread::spawn(move || {
                let seed = KvRandom::FIRST_SEED + (tid % 48) as u64;
                let mut rng = KvRandom::new(seed);

                // Put phase
                let mut puts = 0u64;
                while !timeout.load(Ordering::Relaxed) {
                    let x = rng.rand();
                    let key = key_var(u64::from(x));
                    let _ = tree.insert(&key, u64::from(x + 1));
                    puts += 1;
                }

                // Reconstruct + shuffle (matches C++ `kvtest_rw1_seed`: regenerate keys, then shuffle)
                rng.seed(seed);
                let mut keys = Vec::with_capacity(puts as usize);
                for _ in 0..puts {
                    keys.push(rng.rand());
                }
                for i in 0..keys.len() {
                    let j = rng.uniform(keys.len() as u32) as usize;
                    keys.swap(i, j);
                }

                // Get phase
                let mut gets = 0u64;
                for x in keys {
                    let key = key_var(u64::from(x));
                    std::hint::black_box(tree.get(&key));
                    gets += 1;
                }

                total_ops.fetch_add(puts + gets, Ordering::Relaxed);
            })
        })
        .collect();

    thread::sleep(Duration::from_secs(duration_secs));
    timeout.store(true, Ordering::Relaxed);

    for h in handles {
        h.join().unwrap();
    }

    report(
        "rw1",
        threads,
        start.elapsed(),
        total_ops.load(Ordering::Relaxed),
    );
}

// =============================================================================
// RW2: Interleaved insert/read
// =============================================================================

fn run_rw2_internal(test_name: &str, threads: usize, duration_secs: u64, get_frac: f64) {
    let tree = Arc::new(MassTree24::<u64>::new());
    let timeout = Arc::new(AtomicBool::new(false));
    let total_ops = Arc::new(AtomicU64::new(0));

    let timeout_clone = Arc::clone(&timeout);
    thread::spawn(move || {
        thread::sleep(Duration::from_secs(duration_secs));
        timeout_clone.store(true, Ordering::Relaxed);
    });

    let start = Instant::now();

    let handles: Vec<_> = (0..threads)
        .map(|tid| {
            let tree = Arc::clone(&tree);
            let timeout = Arc::clone(&timeout);
            let total_ops = Arc::clone(&total_ops);
            thread::spawn(move || {
                let seed = KvRandom::FIRST_SEED + (tid % 48) as u64;
                let mut rng = KvRandom::new(seed);
                let offset = rng.rand();
                const C: u32 = 2_654_435_761;

                let mut puts = 0u64;
                let mut gets = 0u64;

                while !timeout.load(Ordering::Relaxed) {
                    if puts == 0 || !rng.bernoulli(get_frac) {
                        let x = (offset.wrapping_add(puts as u32)).wrapping_mul(C);
                        let key = key_var(u64::from(x));
                        let _ = tree.insert(&key, u64::from(x + 1));
                        puts += 1;
                    } else {
                        let idx = rng.uniform(puts as u32);
                        let x = (offset.wrapping_add(idx)).wrapping_mul(C);
                        let key = key_var(u64::from(x));
                        std::hint::black_box(tree.get(&key));
                        gets += 1;
                    }
                }

                total_ops.fetch_add(puts + gets, Ordering::Relaxed);
            })
        })
        .collect();

    for h in handles {
        h.join().unwrap();
    }

    report(
        test_name,
        threads,
        start.elapsed(),
        total_ops.load(Ordering::Relaxed),
    );
}

fn run_rw2(threads: usize, duration_secs: u64) {
    run_rw2_internal("rw2", threads, duration_secs, 0.5);
}

fn run_rw2g90(threads: usize, duration_secs: u64) {
    run_rw2_internal("rw2g90", threads, duration_secs, 0.9);
}

fn run_rw2g98(threads: usize, duration_secs: u64) {
    run_rw2_internal("rw2g98", threads, duration_secs, 0.98);
}

// =============================================================================
// RW3: Sequential 8-byte keys
// =============================================================================

fn run_rw3(threads: usize, duration_secs: u64) {
    // Use inline storage - no Arc allocation per insert (matches C++)
    let tree = Arc::new(MassTree24Inline::<u64>::new());
    let timeout = Arc::new(AtomicBool::new(false));
    let total_ops = Arc::new(AtomicU64::new(0));

    let timeout_clone = Arc::clone(&timeout);
    thread::spawn(move || {
        thread::sleep(Duration::from_secs(duration_secs));
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

    report(
        "rw3",
        threads,
        start.elapsed(),
        total_ops.load(Ordering::Relaxed),
    );
}

// =============================================================================
// RW3_DISJOINT: Sequential 8-byte keys with disjoint ranges per thread
// =============================================================================

fn run_rw3_disjoint(threads: usize, duration_secs: u64) {
    let tree = Arc::new(MassTree24Inline::<u64>::new());
    let timeout = Arc::new(AtomicBool::new(false));
    let total_ops = Arc::new(AtomicU64::new(0));

    let timeout_clone = Arc::clone(&timeout);
    thread::spawn(move || {
        thread::sleep(Duration::from_secs(duration_secs));
        timeout_clone.store(true, Ordering::Relaxed);
    });

    let start = Instant::now();

    let handles: Vec<_> = (0..threads)
        .map(|tid| {
            let tree = Arc::clone(&tree);
            let timeout = Arc::clone(&timeout);
            let total_ops = Arc::clone(&total_ops);
            thread::spawn(move || {
                // Each thread uses disjoint key range
                let base = (tid as u64) * 10_000_000;

                // Put phase
                let mut n = 0u64;
                while !timeout.load(Ordering::Relaxed) {
                    let key = key8(base + n);
                    let _ = tree.insert(&key, n + 1);
                    n += 1;
                }

                // Get phase
                for i in 0..n {
                    let key = key8(base + i);
                    std::hint::black_box(tree.get(&key));
                }

                total_ops.fetch_add(n * 2, Ordering::Relaxed);
            })
        })
        .collect();

    for h in handles {
        h.join().unwrap();
    }

    report(
        "rw3_disjoint",
        threads,
        start.elapsed(),
        total_ops.load(Ordering::Relaxed),
    );
}

// =============================================================================
// RW3_MTTEST: Exact C++ mttest methodology (per-thread timing, sum rates)
// =============================================================================

fn run_rw3_mttest(threads: usize, duration_secs: u64) {
    use std::sync::Mutex;

    // Use inline storage - no Arc allocation per insert (matches C++)
    let tree = Arc::new(MassTree24Inline::<u64>::new());
    let timeout = Arc::new(AtomicBool::new(false));
    // Collect per-thread results: (ops, elapsed_secs)
    let results: Arc<Mutex<Vec<(u64, f64)>>> = Arc::new(Mutex::new(Vec::new()));

    let timeout_clone = Arc::clone(&timeout);
    thread::spawn(move || {
        thread::sleep(Duration::from_secs(duration_secs));
        timeout_clone.store(true, Ordering::Relaxed);
    });

    let handles: Vec<_> = (0..threads)
        .map(|tid| {
            let tree = Arc::clone(&tree);
            let timeout = Arc::clone(&timeout);
            let results = Arc::clone(&results);
            thread::spawn(move || {
                // C++ pattern: each thread times itself
                let t0 = Instant::now();

                // Put phase
                let mut n = 0u64;
                while !timeout.load(Ordering::Relaxed) {
                    let key = key8(n);
                    let _ = tree.insert(&key, n + 1);
                    n += 1;
                }

                let t1 = Instant::now();

                // Get phase
                for i in 0..n {
                    let key = key8(i);
                    std::hint::black_box(tree.get(&key));
                }

                let t2 = Instant::now();

                let total_ops = n * 2;
                let elapsed = (t2 - t0).as_secs_f64();
                let ops_per_sec = total_ops as f64 / elapsed;

                // Report per-thread stats (like C++ does)
                println!(
                    "{}: {{\"thread\":{},\"puts\":{},\"puts_per_sec\":{:.5},\"gets\":{},\"gets_per_sec\":{:.5},\"ops\":{},\"ops_per_sec\":{:.5}}}",
                    tid, tid, n, n as f64 / (t1 - t0).as_secs_f64(),
                    n, n as f64 / (t2 - t1).as_secs_f64(),
                    total_ops, ops_per_sec
                );

                results.lock().unwrap().push((total_ops, elapsed));
            })
        })
        .collect();

    for h in handles {
        h.join().unwrap();
    }

    // Sum per-thread rates (C++ methodology)
    let results = results.lock().unwrap();
    let sum_rates: f64 = results
        .iter()
        .map(|(ops, elapsed)| *ops as f64 / elapsed)
        .sum();
    let total_ops: u64 = results.iter().map(|(ops, _)| ops).sum();
    let max_elapsed: f64 = results.iter().map(|(_, e)| *e).fold(0.0, f64::max);

    println!();
    println!("=== METHODOLOGY COMPARISON ===");
    println!(
        "C++ style (sum per-thread rates): {:.2}M ops/sec",
        sum_rates / 1_000_000.0
    );
    println!(
        "Rust style (total/wall-clock):    {:.2}M ops/sec",
        total_ops as f64 / max_elapsed / 1_000_000.0
    );
    println!();
}

// =============================================================================
// RW3_W15: Sequential ASCII keys with WIDTH=15 (matches C++ default)
// =============================================================================

fn run_rw3_w15(threads: usize, duration_secs: u64) {
    let tree = Arc::new(MassTree24::<u64>::new());
    let timeout = Arc::new(AtomicBool::new(false));
    let total_ops = Arc::new(AtomicU64::new(0));

    let timeout_clone = Arc::clone(&timeout);
    thread::spawn(move || {
        thread::sleep(Duration::from_secs(duration_secs));
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

    report(
        "rw3_w15",
        threads,
        start.elapsed(),
        total_ops.load(Ordering::Relaxed),
    );
}

// =============================================================================
// RW3_BIN: Sequential 8-byte BINARY keys (for comparison)
// =============================================================================

fn run_rw3_bin(threads: usize, duration_secs: u64) {
    let tree = Arc::new(MassTree24::<u64>::new());
    let timeout = Arc::new(AtomicBool::new(false));
    let total_ops = Arc::new(AtomicU64::new(0));

    let timeout_clone = Arc::clone(&timeout);
    thread::spawn(move || {
        thread::sleep(Duration::from_secs(duration_secs));
        timeout_clone.store(true, Ordering::Relaxed);
    });

    let start = Instant::now();

    let handles: Vec<_> = (0..threads)
        .map(|_| {
            let tree = Arc::clone(&tree);
            let timeout = Arc::clone(&timeout);
            let total_ops = Arc::clone(&total_ops);
            thread::spawn(move || {
                // Put phase - BINARY keys like original benchmarks
                let mut n = 0u64;
                while !timeout.load(Ordering::Relaxed) {
                    let key = n.to_be_bytes(); // Binary, not ASCII!
                    let _ = tree.insert(&key, n + 1);
                    n += 1;
                }

                // Get phase
                for i in 0..n {
                    let key = i.to_be_bytes();
                    std::hint::black_box(tree.get(&key));
                }

                total_ops.fetch_add(n * 2, Ordering::Relaxed);
            })
        })
        .collect();

    for h in handles {
        h.join().unwrap();
    }

    report(
        "rw3_bin",
        threads,
        start.elapsed(),
        total_ops.load(Ordering::Relaxed),
    );
}

// =============================================================================
// RW4: Reverse sequential
// =============================================================================

fn run_rw4(threads: usize, duration_secs: u64) {
    const TOP: u64 = 2_147_483_647;
    let tree = Arc::new(MassTree24::<u64>::new());
    let timeout = Arc::new(AtomicBool::new(false));
    let total_ops = Arc::new(AtomicU64::new(0));

    let timeout_clone = Arc::clone(&timeout);
    thread::spawn(move || {
        thread::sleep(Duration::from_secs(duration_secs));
        timeout_clone.store(true, Ordering::Relaxed);
    });

    let start = Instant::now();

    let handles: Vec<_> = (0..threads)
        .map(|_| {
            let tree = Arc::clone(&tree);
            let timeout = Arc::clone(&timeout);
            let total_ops = Arc::clone(&total_ops);
            thread::spawn(move || {
                let mut n = 0u64;
                while !timeout.load(Ordering::Relaxed) {
                    let key = key8(TOP - n);
                    let _ = tree.insert(&key, n + 1);
                    n += 1;
                }

                for i in 0..n {
                    let key = key8(TOP - i);
                    std::hint::black_box(tree.get(&key));
                }

                total_ops.fetch_add(n * 2, Ordering::Relaxed);
            })
        })
        .collect();

    for h in handles {
        h.join().unwrap();
    }

    report(
        "rw4",
        threads,
        start.elapsed(),
        total_ops.load(Ordering::Relaxed),
    );
}

// =============================================================================
// SAME: Hot spot - 10 keys
// =============================================================================

fn run_same(threads: usize, duration_secs: u64) {
    let tree = Arc::new(MassTree24::<u64>::new());
    let timeout = Arc::new(AtomicBool::new(false));
    let total_ops = Arc::new(AtomicU64::new(0));

    let timeout_clone = Arc::clone(&timeout);
    thread::spawn(move || {
        thread::sleep(Duration::from_secs(duration_secs));
        timeout_clone.store(true, Ordering::Relaxed);
    });

    let start = Instant::now();

    let handles: Vec<_> = (0..threads)
        .map(|tid| {
            let tree = Arc::clone(&tree);
            let timeout = Arc::clone(&timeout);
            let total_ops = Arc::clone(&total_ops);
            thread::spawn(move || {
                let seed = KvRandom::FIRST_SEED + (tid % 48) as u64;
                let mut rng = KvRandom::new(seed);

                let mut n = 0u64;
                while !timeout.load(Ordering::Relaxed) {
                    let x = rng.uniform(10);
                    let key = key_var(u64::from(x));
                    let _ = tree.insert(&key, u64::from(x + 1));
                    n += 1;
                }

                total_ops.fetch_add(n, Ordering::Relaxed);
            })
        })
        .collect();

    for h in handles {
        h.join().unwrap();
    }

    report(
        "same",
        threads,
        start.elapsed(),
        total_ops.load(Ordering::Relaxed),
    );
}

// =============================================================================
// WSCALE: Pure random writes
// =============================================================================

fn run_wscale(threads: usize, duration_secs: u64) {
    let tree = Arc::new(MassTree24::<u64>::new());
    let timeout = Arc::new(AtomicBool::new(false));
    let total_ops = Arc::new(AtomicU64::new(0));

    let timeout_clone = Arc::clone(&timeout);
    thread::spawn(move || {
        thread::sleep(Duration::from_secs(duration_secs));
        timeout_clone.store(true, Ordering::Relaxed);
    });

    let start = Instant::now();

    let handles: Vec<_> = (0..threads)
        .map(|tid| {
            let tree = Arc::clone(&tree);
            let timeout = Arc::clone(&timeout);
            let total_ops = Arc::clone(&total_ops);
            thread::spawn(move || {
                let seed = KvRandom::FIRST_SEED + (tid % 48) as u64;
                let mut rng = KvRandom::new(seed);

                let mut n = 0u64;
                while !timeout.load(Ordering::Relaxed) {
                    let x = u64::from(rng.rand());
                    let key = key_var(x);
                    let _ = tree.insert(&key, x + 1);
                    n += 1;
                }

                total_ops.fetch_add(n, Ordering::Relaxed);
            })
        })
        .collect();

    for h in handles {
        h.join().unwrap();
    }

    report(
        "wscale",
        threads,
        start.elapsed(),
        total_ops.load(Ordering::Relaxed),
    );
}

// =============================================================================
// RSCALE: Pure random reads on pre-populated tree
// =============================================================================

fn run_rscale(threads: usize, duration_secs: u64) {
    const PARTSZ: usize = 500_000;
    let tree = Arc::new(MassTree24::<u64>::new());
    let nseqkeys = (PARTSZ * threads) as u64;

    // Pre-populate (like ruscale_init)
    println!("rscale: pre-populating {nseqkeys} keys...");
    for part in 0..threads {
        let seed = KvRandom::FIRST_SEED + (part % 48) as u64;
        let mut rng = KvRandom::new(seed);
        let first_key = PARTSZ * part;

        let mut keys: Vec<usize> = (0..PARTSZ).map(|i| i + first_key).collect();
        for i in 0..PARTSZ {
            let j = rng.uniform(PARTSZ as u32) as usize;
            keys.swap(i, j);
        }

        for &k in &keys {
            let key = key_var(k as u64);
            let _ = tree.insert(&key, k as u64 + 1);
        }
    }

    let timeout = Arc::new(AtomicBool::new(false));
    let total_ops = Arc::new(AtomicU64::new(0));

    let timeout_clone = Arc::clone(&timeout);
    thread::spawn(move || {
        thread::sleep(Duration::from_secs(duration_secs));
        timeout_clone.store(true, Ordering::Relaxed);
    });

    let start = Instant::now();

    let handles: Vec<_> = (0..threads)
        .map(|tid| {
            let tree = Arc::clone(&tree);
            let timeout = Arc::clone(&timeout);
            let total_ops = Arc::clone(&total_ops);
            thread::spawn(move || {
                let seed = KvRandom::FIRST_SEED + (tid % 48) as u64;
                let mut rng = KvRandom::new(seed);

                let mut n = 0u64;
                while !timeout.load(Ordering::Relaxed) {
                    let x = u64::from(rng.rand()) % nseqkeys;
                    let key = key_var(x);
                    std::hint::black_box(tree.get(&key));
                    n += 1;
                }

                total_ops.fetch_add(n, Ordering::Relaxed);
            })
        })
        .collect();

    for h in handles {
        h.join().unwrap();
    }

    report(
        "rscale",
        threads,
        start.elapsed(),
        total_ops.load(Ordering::Relaxed),
    );
}

// =============================================================================
// USCALE: Pure random updates
// =============================================================================

fn run_uscale(threads: usize, duration_secs: u64) {
    const PARTSZ: usize = 500_000;
    let tree = Arc::new(MassTree24::<u64>::new());
    let nseqkeys = (PARTSZ * threads) as u64;

    // Pre-populate
    println!("uscale: pre-populating {nseqkeys} keys...");
    for i in 0..nseqkeys {
        let key = key_var(i);
        let _ = tree.insert(&key, i + 1);
    }

    let timeout = Arc::new(AtomicBool::new(false));
    let total_ops = Arc::new(AtomicU64::new(0));

    let timeout_clone = Arc::clone(&timeout);
    thread::spawn(move || {
        thread::sleep(Duration::from_secs(duration_secs));
        timeout_clone.store(true, Ordering::Relaxed);
    });

    let start = Instant::now();

    let handles: Vec<_> = (0..threads)
        .map(|tid| {
            let tree = Arc::clone(&tree);
            let timeout = Arc::clone(&timeout);
            let total_ops = Arc::clone(&total_ops);
            thread::spawn(move || {
                let seed = KvRandom::FIRST_SEED + tid as u64;
                let mut rng = KvRandom::new(seed);

                let mut n = 0u64;
                while !timeout.load(Ordering::Relaxed) {
                    let x = u64::from(rng.rand()) % nseqkeys;
                    let key = key_var(x);
                    let _ = tree.insert(&key, x + 1);
                    n += 1;
                }

                total_ops.fetch_add(n, Ordering::Relaxed);
            })
        })
        .collect();

    for h in handles {
        h.join().unwrap();
    }

    report(
        "uscale",
        threads,
        start.elapsed(),
        total_ops.load(Ordering::Relaxed),
    );
}

// =============================================================================
// RW1LONG: String keys
// =============================================================================

fn run_rw1long(threads: usize, duration_secs: u64) {
    const FORMATS: [&str; 4] = ["user", "machine", "opening", "fartparade"];

    let tree = Arc::new(MassTree24::<u64>::new());
    let timeout = Arc::new(AtomicBool::new(false));
    let total_ops = Arc::new(AtomicU64::new(0));

    let start = Instant::now();

    let handles: Vec<_> = (0..threads)
        .map(|tid| {
            let tree = Arc::clone(&tree);
            let timeout = Arc::clone(&timeout);
            let total_ops = Arc::clone(&total_ops);
            thread::spawn(move || {
                let seed = KvRandom::FIRST_SEED + (tid % 48) as u64;
                let mut rng = KvRandom::new(seed);

                // Put phase
                let mut puts = 0u64;
                while !timeout.load(Ordering::Relaxed) {
                    let x = rng.rand();
                    let fmt = rng.uniform(4) as usize;
                    let key = format!("{}{}", FORMATS[fmt], x).into_bytes();
                    let _ = tree.insert(&key, u64::from(x + 1));
                    puts += 1;
                }

                // Reconstruct + shuffle (C++ regenerates keys after put phase, then shuffles)
                rng.seed(seed);
                let mut keys = Vec::with_capacity(puts as usize);
                for _ in 0..puts {
                    let x = rng.rand();
                    let fmt = rng.uniform(4) as usize;
                    keys.push((fmt, x));
                }
                for i in 0..keys.len() {
                    let j = rng.uniform(keys.len() as u32) as usize;
                    keys.swap(i, j);
                }

                // Get phase
                let mut gets = 0u64;
                for (fmt, x) in keys {
                    let key = format!("{}{}", FORMATS[fmt], x).into_bytes();
                    std::hint::black_box(tree.get(&key));
                    gets += 1;
                }

                total_ops.fetch_add(puts + gets, Ordering::Relaxed);
            })
        })
        .collect();

    thread::sleep(Duration::from_secs(duration_secs));
    timeout.store(true, Ordering::Relaxed);

    for h in handles {
        h.join().unwrap();
    }

    report(
        "rw1long",
        threads,
        start.elapsed(),
        total_ops.load(Ordering::Relaxed),
    );
}

// =============================================================================
// RWSMALL24: 24 keys, 87.5% reads
// =============================================================================

fn run_rwsmall24(threads: usize, duration_secs: u64) {
    const NKEYS: u32 = 24;

    let tree = Arc::new(MassTree24::<u64>::new());
    let timeout = Arc::new(AtomicBool::new(false));
    let total_ops = Arc::new(AtomicU64::new(0));

    let timeout_clone = Arc::clone(&timeout);
    thread::spawn(move || {
        thread::sleep(Duration::from_secs(duration_secs));
        timeout_clone.store(true, Ordering::Relaxed);
    });

    let start = Instant::now();

    let handles: Vec<_> = (0..threads)
        .map(|tid| {
            let tree = Arc::clone(&tree);
            let timeout = Arc::clone(&timeout);
            let total_ops = Arc::clone(&total_ops);
            thread::spawn(move || {
                let seed = KvRandom::FIRST_SEED + (tid % 48) as u64;
                let mut rng = KvRandom::new(seed);

                let mut n = 0u64;
                while !timeout.load(Ordering::Relaxed) {
                    let x = rng.uniform(NKEYS << 3);
                    let key_idx = x >> 3;
                    let key = key_var(u64::from(key_idx));

                    if (x & 7) != 0 {
                        std::hint::black_box(tree.get(&key));
                    } else {
                        let _ = tree.insert(&key, n);
                    }
                    n += 1;
                }

                total_ops.fetch_add(n, Ordering::Relaxed);
            })
        })
        .collect();

    for h in handles {
        h.join().unwrap();
    }

    report(
        "rwsmall24",
        threads,
        start.elapsed(),
        total_ops.load(Ordering::Relaxed),
    );
}

// =============================================================================
// Main
// =============================================================================

fn main() {
    let config = Config::parse();

    println!(
        "cpp_comp: {} threads, {}s duration",
        config.threads, config.duration_secs
    );
    println!("-------------------------------------------");

    for test in &config.tests {
        match test.as_str() {
            "rw1" => run_rw1(config.threads, config.duration_secs),
            "rw2" => run_rw2(config.threads, config.duration_secs),
            "rw2g90" => run_rw2g90(config.threads, config.duration_secs),
            "rw2g98" => run_rw2g98(config.threads, config.duration_secs),
            "rw3" => run_rw3(config.threads, config.duration_secs),
            "rw3_disjoint" => run_rw3_disjoint(config.threads, config.duration_secs),
            "rw3_mttest" => run_rw3_mttest(config.threads, config.duration_secs),
            "rw3_w15" => run_rw3_w15(config.threads, config.duration_secs),
            "rw3_bin" => run_rw3_bin(config.threads, config.duration_secs),
            "rw4" => run_rw4(config.threads, config.duration_secs),
            "same" => run_same(config.threads, config.duration_secs),
            "wscale" => run_wscale(config.threads, config.duration_secs),
            "rscale" => run_rscale(config.threads, config.duration_secs),
            "uscale" => run_uscale(config.threads, config.duration_secs),
            "rw1long" => run_rw1long(config.threads, config.duration_secs),
            "rwsmall24" => run_rwsmall24(config.threads, config.duration_secs),
            _ => eprintln!("Unknown test: {test}"),
        }
    }
}
