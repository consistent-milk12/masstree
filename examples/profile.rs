//! Profiling binary for use with valgrind/callgrind.
//!
//! # Usage
//!
//! ```bash
//! # Build with optimizations + debug symbols
//! cargo build --release --example profile
//!
//! # Run with callgrind
//! valgrind --tool=callgrind ./target/release/examples/profile [workload]
//!
//! # Analyze results
//! callgrind_annotate callgrind.out.<pid> --auto=yes
//!
//! # Or use kcachegrind for GUI
//! kcachegrind callgrind.out.<pid>
//! ```
//!
//! # Workloads
//!
//! - `key` (default): Key operations
//! - `permuter`: Permuter operations
//! - `all`: Both workloads

use std::hint::black_box;

use masstree::key::Key;
use masstree::permuter::Permuter;

const ITERATIONS: usize = 10_000;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let workload = args.get(1).map(String::as_str).unwrap_or("key");

    match workload {
        "key" => run_key_workload(),
        "permuter" => run_permuter_workload(),
        "all" => {
            run_key_workload();
            run_permuter_workload();
        }
        _ => {
            eprintln!("Unknown workload: {workload}");
            eprintln!("Available: key, permuter, all");
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
