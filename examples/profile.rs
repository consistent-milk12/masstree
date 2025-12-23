//! Profiling binary for use with perf/valgrind/callgrind.
//!
//! # Usage
//!
//! ```bash
//! # Build with optimizations + debug symbols
//! cargo build --release --example profile
//!
//! # Run with perf (recommended for concurrency profiling)
//! perf record -g ./target/release/examples/profile contention
//! perf report
//!
//! # Run with callgrind (single-threaded only)
//! valgrind --tool=callgrind ./target/release/examples/profile key
//!
//! # Analyze callgrind results
//! callgrind_annotate callgrind.out.<pid> --auto=yes
//! kcachegrind callgrind.out.<pid>
//! ```
//!
//! # Workloads
//!
//! - `key` (default): Key operations
//! - `permuter`: Permuter operations
//! - `contention`: 32-thread contention writes (for perf profiling)
//! - `all`: All workloads

use std::hint::black_box;
use std::sync::Arc;
use std::thread;

use masstree::MassTree;
use masstree::key::Key;
use masstree::permuter::Permuter;

const ITERATIONS: usize = 10_000;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let workload = args.get(1).map_or("key", String::as_str);

    match workload {
        "key" => run_key_workload(),
        "permuter" => run_permuter_workload(),
        "contention" => run_contention_workload(),
        "all" => {
            run_key_workload();
            run_permuter_workload();
            run_contention_workload();
        }
        _ => {
            eprintln!("Unknown workload: {workload}");
            eprintln!("Available: key, permuter, contention, all");
            std::process::exit(1);
        }
    }
}

/// Key workload: construction, traversal, comparison
#[inline(never)]
fn run_key_workload() {
    eprintln!("Running Key workload ({ITERATIONS} iterations)...");

    // Workload 1: Key construction with varying lengths
    for _ in 0..ITERATIONS {
        let keys: [&[u8]; 8] = [
            b"a",
            b"abcd",
            b"abcdefgh",
            b"abcdefghijkl",
            b"abcdefghijklmnop",
            b"abcdefghijklmnopqrst",
            b"abcdefghijklmnopqrstuvwx",
            b"abcdefghijklmnopqrstuvwxyz012345",
        ];
        for k in &keys {
            black_box(Key::new(black_box(*k)));
        }
    }

    // Workload 2: Multi-layer traversal
    for _ in 0..ITERATIONS {
        let mut key = Key::new(black_box(b"username:session:token:value!"));
        while key.has_suffix() {
            black_box(key.ikey());
            key.shift();
        }
        black_box(key.ikey());
    }

    // Workload 3: Key comparison (simulates B+tree node search)
    let search_key = Key::new(b"hello world!");
    let stored_ikeys: [u64; 15] = [
        u64::from_be_bytes(*b"aardvark"),
        u64::from_be_bytes(*b"badger__"),
        u64::from_be_bytes(*b"cat_____"),
        u64::from_be_bytes(*b"dog_____"),
        u64::from_be_bytes(*b"elephant"),
        u64::from_be_bytes(*b"fox_____"),
        u64::from_be_bytes(*b"giraffe_"),
        u64::from_be_bytes(*b"hello wo"),
        u64::from_be_bytes(*b"iguana__"),
        u64::from_be_bytes(*b"jaguar__"),
        u64::from_be_bytes(*b"kangaroo"),
        u64::from_be_bytes(*b"lion____"),
        u64::from_be_bytes(*b"mouse___"),
        u64::from_be_bytes(*b"newt____"),
        u64::from_be_bytes(*b"owl_____"),
    ];

    for _ in 0..ITERATIONS {
        for &stored in &stored_ikeys {
            black_box(search_key.compare(black_box(stored), 12));
        }
    }

    // Workload 4: read_ikey at various offsets
    let data = b"0123456789abcdef0123456789ABCDEF0123456789abcdef0123456789ABCDEF";
    for _ in 0..ITERATIONS {
        for offset in (0..64).step_by(8) {
            black_box(Key::read_ikey(black_box(data), black_box(offset)));
        }
    }

    eprintln!("Key workload complete.");
}

/// Permuter workload: insert, remove, search patterns
#[inline(never)]
fn run_permuter_workload() {
    eprintln!("Running Permuter workload ({ITERATIONS} iterations)...");

    // Workload 1: Build full node (sorted insertions)
    for _ in 0..ITERATIONS {
        let mut p: Permuter<15> = Permuter::empty();
        for i in 0..15 {
            let _ = p.insert_from_back(black_box(i));
        }
        black_box(p);
    }

    // Workload 2: Build full node (reverse insertions - worst case)
    for _ in 0..ITERATIONS {
        let mut p: Permuter<15> = Permuter::empty();
        for _ in 0..15 {
            let _ = p.insert_from_back(black_box(0));
        }
        black_box(p);
    }

    // Workload 3: Scan all slots (range query pattern)
    let p: Permuter<15> = Permuter::make_sorted(15);
    for _ in 0..ITERATIONS {
        let mut sum = 0usize;
        for i in 0..15 {
            sum = sum.wrapping_add(p.get(black_box(i)));
        }
        black_box(sum);
    }

    // Workload 4: Insert/remove cycle (update pattern)
    for _ in 0..ITERATIONS {
        let mut p: Permuter<15> = Permuter::make_sorted(10);
        p.remove(black_box(3));
        p.remove(black_box(5));
        let _ = p.insert_from_back(black_box(2));
        let _ = p.insert_from_back(black_box(4));
        black_box(p);
    }

    // Workload 5: Exchange operations (rebalancing)
    for _ in 0..ITERATIONS {
        let mut p: Permuter<15> = Permuter::make_sorted(15);
        p.exchange(black_box(0), black_box(7));
        p.exchange(black_box(1), black_box(8));
        p.exchange(black_box(2), black_box(9));
        black_box(p);
    }

    eprintln!("Permuter workload complete.");
}

// =============================================================================
// Key generation (from bench_utils)
// =============================================================================

const MULTIPLIERS: [u64; 4] = [
    1,
    0x517c_c1b7_2722_0a95,
    0x9e37_79b9_7f4a_7c15,
    0xbf58_476d_1ce4_e5b9,
];

fn keys<const K: usize>(n: usize) -> Vec<[u8; K]> {
    assert!(K % 8 == 0, "key size must be a multiple of 8");
    let chunks = K / 8;
    let mut out = Vec::with_capacity(n);

    for i in 0..n {
        let mut key = [0u8; K];
        for c in 0..chunks {
            let v = (i as u64).wrapping_mul(MULTIPLIERS[c]);
            let bytes = v.to_be_bytes();
            let start = c * 8;
            key[start..start + 8].copy_from_slice(&bytes);
        }
        out.push(key);
    }
    out
}

fn setup_masstree<const K: usize>(keys: &[[u8; K]]) -> MassTree<u64> {
    let mut tree = MassTree::new();
    for (i, key) in keys.iter().enumerate() {
        let _ = tree.insert(key, i as u64);
    }
    tree
}

/// Contention workload: 32 threads hammering 1000 keys
/// This is the workload where MassTree has high variance
#[inline(never)]
fn run_contention_workload() {
    const THREADS: usize = 32;
    const OPS_PER_THREAD: usize = 50_000;
    const KEY_SPACE: usize = 1000;

    eprintln!(
        "Running Contention workload ({} threads, {} ops/thread, {} keys)...",
        THREADS, OPS_PER_THREAD, KEY_SPACE
    );

    let keys: Arc<Vec<[u8; 8]>> = Arc::new(keys::<8>(KEY_SPACE));
    let tree = Arc::new(setup_masstree::<8>(keys.as_ref()));

    eprintln!("Pre-populated {} keys", tree.len());

    let start = std::time::Instant::now();

    let handles: Vec<_> = (0..THREADS)
        .map(|t| {
            let tree = Arc::clone(&tree);
            let keys = Arc::clone(&keys);
            thread::spawn(move || {
                let guard = tree.guard();
                let mut state = (t as u64).wrapping_mul(0x517c_c1b7_2722_0a95);

                for _ in 0..OPS_PER_THREAD {
                    // LCG random
                    state = state
                        .wrapping_mul(6_364_136_223_846_793_005)
                        .wrapping_add(1);
                    let idx = (state as usize) % keys.len();
                    let key = &keys[idx];
                    let val = state;

                    let _ = black_box(tree.insert_with_guard(key, val, &guard));
                }
            })
        })
        .collect();

    for h in handles {
        let _ = h.join();
    }

    let elapsed = start.elapsed();
    let total_ops = THREADS * OPS_PER_THREAD;
    eprintln!(
        "Contention workload complete: {} ops in {:?} ({:.0} ops/sec)",
        total_ops,
        elapsed,
        total_ops as f64 / elapsed.as_secs_f64()
    );
}
