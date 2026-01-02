//! Iai-callgrind benchmarks for [`MassTree`] core operations.
//!
//! Uses instruction counting for deterministic, reproducible measurements.
//! Perfect for CI regression detection.
//!
//! ## Running
//!
//! ```bash
//! # Requires valgrind installed
//! cargo bench --bench iai_masstree
//!
//! # With callgrind output for kcachegrind
//! cargo bench --bench iai_masstree -- --callgrind-args="--dump-instr=yes"
//! ```
//!
//! ## Output location
//!
//! Results are in `target/iai/<benchmark_name>/`

#![allow(missing_docs)]
#![allow(clippy::indexing_slicing)]

use iai_callgrind::{
    Callgrind, LibraryBenchmarkConfig, library_benchmark, library_benchmark_group, main,
};
use masstree::MassTree24;
use std::hint::black_box;

// =============================================================================
// Setup: Pre-built trees for benchmarking
// =============================================================================

/// Small tree (1K keys) - fits in L1/L2 cache
fn setup_tree_1k() -> (MassTree24<u64>, Vec<[u8; 8]>) {
    let tree = MassTree24::new();
    let keys: Vec<[u8; 8]> = (0u64..1_000).map(u64::to_be_bytes).collect();
    {
        let guard = tree.guard();
        for (i, key) in keys.iter().enumerate() {
            let _ = tree.insert_with_guard(key, i as u64, &guard);
        }
    }
    (tree, keys)
}

/// Medium tree (100K keys) - exceeds L2, fits in L3
fn setup_tree_100k() -> (MassTree24<u64>, Vec<[u8; 8]>) {
    let tree = MassTree24::new();
    let keys: Vec<[u8; 8]> = (0u64..100_000).map(u64::to_be_bytes).collect();
    {
        let guard = tree.guard();
        for (i, key) in keys.iter().enumerate() {
            let _ = tree.insert_with_guard(key, i as u64, &guard);
        }
    }
    (tree, keys)
}

/// Long keys (32B) - triggers multi-layer traversal
fn setup_tree_long_keys() -> (MassTree24<u64>, Vec<[u8; 32]>) {
    let tree = MassTree24::new();
    let keys: Vec<[u8; 32]> = (0u64..10_000)
        .map(|i| {
            let mut key = [0u8; 32];
            key[..8].copy_from_slice(&i.to_be_bytes());
            // Add some variation in later bytes
            key[24..].copy_from_slice(&i.wrapping_mul(0x517c_c1b7_2722_0a95).to_be_bytes());
            key
        })
        .collect();
    {
        let guard = tree.guard();
        for (i, key) in keys.iter().enumerate() {
            let _ = tree.insert_with_guard(key, i as u64, &guard);
        }
    }
    (tree, keys)
}

// =============================================================================
// GET Benchmarks
// =============================================================================

#[library_benchmark]
#[bench::small(setup_tree_1k())]
fn get_8b_1k((tree, keys): (MassTree24<u64>, Vec<[u8; 8]>)) -> u64 {
    let guard = tree.guard();
    let mut sum = 0u64;
    // Access 100 keys spread across the tree
    for i in (0..1000).step_by(10) {
        if let Some(v) = tree.get_ref(&keys[i], &guard) {
            sum += *v;
        }
    }
    black_box(sum)
}

#[library_benchmark]
#[bench::medium(setup_tree_100k())]
fn get_8b_100k((tree, keys): (MassTree24<u64>, Vec<[u8; 8]>)) -> u64 {
    let guard = tree.guard();
    let mut sum = 0u64;
    // Access 100 keys spread across the tree
    for i in (0..100_000).step_by(1000) {
        if let Some(v) = tree.get_ref(&keys[i], &guard) {
            sum += *v;
        }
    }
    black_box(sum)
}

#[library_benchmark]
#[bench::long_keys(setup_tree_long_keys())]
fn get_32b((tree, keys): (MassTree24<u64>, Vec<[u8; 32]>)) -> u64 {
    let guard = tree.guard();
    let mut sum = 0u64;
    // Access 100 keys
    for i in (0..10_000).step_by(100) {
        if let Some(v) = tree.get_ref(&keys[i], &guard) {
            sum += *v;
        }
    }
    black_box(sum)
}

// =============================================================================
// INSERT Benchmarks
// =============================================================================

// Insert into empty tree - measures allocation + insert path
#[library_benchmark]
fn insert_empty_tree() -> MassTree24<u64> {
    let tree = MassTree24::new();
    {
        let guard = tree.guard();
        for i in 0u64..100 {
            let key = i.to_be_bytes();
            let _ = tree.insert_with_guard(&key, i, &guard);
        }
    }
    black_box(tree)
}

// Insert sequential keys - best case, no rebalancing needed
#[library_benchmark]
fn insert_sequential() -> MassTree24<u64> {
    let tree = MassTree24::new();
    {
        let guard = tree.guard();
        for i in 0u64..1000 {
            let key = i.to_be_bytes();
            let _ = tree.insert_with_guard(&key, i, &guard);
        }
    }
    black_box(tree)
}

// Insert with splits - pre-fill then add more to trigger splits
#[library_benchmark]
#[bench::splits(setup_tree_1k())]
fn insert_causing_splits((tree, _): (MassTree24<u64>, Vec<[u8; 8]>)) -> MassTree24<u64> {
    {
        let guard = tree.guard();
        // Insert 100 more keys interspersed with existing ones
        for i in 0u64..100 {
            let key = (i * 10 + 5).to_be_bytes(); // Keys between existing ones
            let _ = tree.insert_with_guard(&key, i + 1000, &guard);
        }
    }
    black_box(tree)
}

// Update existing keys (no structural changes)
#[library_benchmark]
#[bench::updates(setup_tree_1k())]
fn insert_update_existing((tree, keys): (MassTree24<u64>, Vec<[u8; 8]>)) {
    let guard = tree.guard();
    for (i, key) in keys.iter().take(100).enumerate() {
        let _ = tree.insert_with_guard(key, (i + 9999) as u64, &guard);
    }
    black_box(());
}

// Insert long keys (32B) - multi-layer creation
#[library_benchmark]
fn insert_long_keys() -> MassTree24<u64> {
    let tree = MassTree24::new();
    {
        let guard = tree.guard();
        for i in 0u64..500 {
            let mut key = [0u8; 32];
            key[..8].copy_from_slice(&i.to_be_bytes());
            key[24..].copy_from_slice(&i.wrapping_mul(0x517c_c1b7_2722_0a95).to_be_bytes());
            let _ = tree.insert_with_guard(&key, i, &guard);
        }
    }
    black_box(tree)
}

// =============================================================================
// Mixed Operations
// =============================================================================

#[library_benchmark]
#[bench::mixed(setup_tree_1k())]
fn mixed_get_insert((tree, keys): (MassTree24<u64>, Vec<[u8; 8]>)) -> u64 {
    let guard = tree.guard();
    let mut sum = 0u64;

    for i in 0..100 {
        // Read existing
        if let Some(v) = tree.get_ref(&keys[i * 10], &guard) {
            sum += *v;
        }
        // Insert new
        let new_key = (1000 + i as u64).to_be_bytes();
        let _ = tree.insert_with_guard(&new_key, i as u64, &guard);
    }
    black_box(sum)
}

// =============================================================================
// Benchmark Groups
// =============================================================================

library_benchmark_group!(
    name = get_benchmarks;
    benchmarks = get_8b_1k, get_8b_100k, get_32b
);

library_benchmark_group!(
    name = insert_benchmarks;
    benchmarks =
        insert_empty_tree,
        insert_sequential,
        insert_causing_splits,
        insert_update_existing,
        insert_long_keys
);

library_benchmark_group!(
    name = mixed_benchmarks;
    benchmarks = mixed_get_insert
);

main!(
    config = LibraryBenchmarkConfig::default()
        .tool(Callgrind::with_args(["--collect-atstart=yes"]));
    library_benchmark_groups =
        get_benchmarks,
        insert_benchmarks,
        mixed_benchmarks
);
