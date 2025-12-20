//! Rigorous comparison benchmarks: `MassTree` vs `BTreeMap`
//!
//! **Methodology:**
//! - Identical key generation for both data structures
//! - Pre-allocated inputs to avoid measuring allocation
//! - Same access patterns (sequential, random, mixed)
//! - Consistent value types (u64, String)
//! - Multiple tree sizes to capture scaling behavior
//!
//! Run with: `cargo bench --bench comparison`
//! With mimalloc: `cargo bench --bench comparison --features mimalloc`

#![expect(clippy::type_complexity)]
#![expect(clippy::indexing_slicing)]

// Use alternative allocator if feature is enabled
#[cfg(feature = "mimalloc")]
#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;

use divan::{Bencher, black_box};
use masstree::tree::{MassTree, MassTreeIndex};
use std::collections::BTreeMap;

fn main() {
    divan::main();
}

// =============================================================================
// Key Generation Helpers (shared between MassTree and BTreeMap)
// =============================================================================

/// Generate sequential keys as byte vectors (`MassTree` format)
fn sequential_keys_bytes(n: usize) -> Vec<Vec<u8>> {
    (0..n).map(|i| (i as u64).to_be_bytes().to_vec()).collect()
}

/// Generate sequential keys as strings (`BTreeMap` format)
#[expect(dead_code)]
fn sequential_keys_strings(n: usize) -> Vec<String> {
    (0..n).map(|i| format!("{:016x}", i as u64)).collect()
}

/// Generate pseudo-random shuffled indices
fn shuffled_indices(n: usize, seed: usize) -> Vec<usize> {
    let mut indices: Vec<usize> = (0..n).collect();
    // Deterministic shuffle for reproducibility
    for i in 0..indices.len() {
        let j = ((i.wrapping_mul(seed)).wrapping_add(17)) % indices.len();
        indices.swap(i, j);
    }
    indices
}

// =============================================================================
// SINGLE INSERT: Empty Tree
// =============================================================================

#[divan::bench_group(name = "01_single_insert_empty")]
mod single_insert_empty {
    use super::{BTreeMap, Bencher, MassTree, MassTreeIndex, black_box};

    /// `MassTree`: Insert into empty tree (Arc mode)
    #[divan::bench]
    fn masstree_arc(bencher: Bencher) {
        bencher
            .with_inputs(|| (MassTree::<u64>::new(), b"hello___".to_vec()))
            .bench_local_values(|(mut tree, key)| {
                let _ = tree.insert(black_box(&key), black_box(42u64));
                tree
            });
    }

    /// `MassTree`: Insert into empty tree (Copy/Index mode)
    #[divan::bench]
    fn masstree_copy(bencher: Bencher) {
        bencher
            .with_inputs(|| (MassTreeIndex::<u64>::new(), b"hello___".to_vec()))
            .bench_local_values(|(mut tree, key)| {
                let _ = tree.insert(black_box(&key), black_box(42u64));
                tree
            });
    }

    /// `BTreeMap`: Insert into empty tree
    #[divan::bench]
    fn btreemap(bencher: Bencher) {
        bencher
            .with_inputs(|| (BTreeMap::<String, u64>::new(), "hello___".to_string()))
            .bench_local_values(|(mut tree, key)| {
                tree.insert(black_box(key), black_box(42u64));
                tree
            });
    }

    /// `BTreeMap`: Insert into empty tree (Vec<u8> key for fairer comparison)
    #[divan::bench]
    fn btreemap_bytes(bencher: Bencher) {
        bencher
            .with_inputs(|| (BTreeMap::<Vec<u8>, u64>::new(), b"hello___".to_vec()))
            .bench_local_values(|(mut tree, key)| {
                tree.insert(black_box(key), black_box(42u64));
                tree
            });
    }
}

// =============================================================================
// SINGLE INSERT: Into Populated Tree
// =============================================================================

#[divan::bench_group(name = "02_single_insert_populated")]
mod single_insert_populated {
    use super::{BTreeMap, Bencher, MassTree, MassTreeIndex, black_box};

    const SIZES: &[usize] = &[10, 100, 1000];

    fn setup_masstree(n: usize) -> MassTree<u64> {
        let mut tree = MassTree::new();
        for i in 0..n {
            let key = (i as u64).to_be_bytes();
            let _ = tree.insert(&key, i as u64);
        }
        tree
    }

    fn setup_masstree_index(n: usize) -> MassTreeIndex<u64> {
        let mut tree = MassTreeIndex::new();
        for i in 0..n {
            let key = (i as u64).to_be_bytes();
            let _ = tree.insert(&key, i as u64);
        }
        tree
    }

    fn setup_btreemap(n: usize) -> BTreeMap<Vec<u8>, u64> {
        let mut tree = BTreeMap::new();
        for i in 0..n {
            let key = (i as u64).to_be_bytes().to_vec();
            tree.insert(key, i as u64);
        }
        tree
    }

    #[divan::bench(args = SIZES)]
    fn masstree_arc(bencher: Bencher, n: usize) {
        // Key that doesn't exist in the tree
        let new_key = (n as u64 + 1).to_be_bytes();
        bencher
            .with_inputs(|| setup_masstree(n))
            .bench_local_values(|mut tree| {
                let _ = tree.insert(black_box(&new_key), black_box(9999u64));
                tree
            });
    }

    #[divan::bench(args = SIZES)]
    fn masstree_copy(bencher: Bencher, n: usize) {
        let new_key = (n as u64 + 1).to_be_bytes();
        bencher
            .with_inputs(|| setup_masstree_index(n))
            .bench_local_values(|mut tree| {
                let _ = tree.insert(black_box(&new_key), black_box(9999u64));
                tree
            });
    }

    #[divan::bench(args = SIZES)]
    fn btreemap(bencher: Bencher, n: usize) {
        let new_key = (n as u64 + 1).to_be_bytes().to_vec();
        bencher
            .with_inputs(|| (setup_btreemap(n), new_key.clone()))
            .bench_local_values(|(mut tree, key)| {
                tree.insert(black_box(key), black_box(9999u64));
                tree
            });
    }
}

// =============================================================================
// SINGLE GET: Hit
// =============================================================================

#[divan::bench_group(name = "03_single_get_hit")]
mod single_get_hit {
    use super::{BTreeMap, Bencher, MassTree, MassTreeIndex, black_box};

    const SIZES: &[usize] = &[10, 100, 1000];

    fn setup_masstree(n: usize) -> MassTree<u64> {
        let mut tree = MassTree::new();
        for i in 0..n {
            let key = (i as u64).to_be_bytes();
            let _ = tree.insert(&key, i as u64);
        }
        tree
    }

    fn setup_masstree_index(n: usize) -> MassTreeIndex<u64> {
        let mut tree = MassTreeIndex::new();
        for i in 0..n {
            let key = (i as u64).to_be_bytes();
            let _ = tree.insert(&key, i as u64);
        }
        tree
    }

    fn setup_btreemap(n: usize) -> BTreeMap<Vec<u8>, u64> {
        let mut tree = BTreeMap::new();
        for i in 0..n {
            let key = (i as u64).to_be_bytes().to_vec();
            tree.insert(key, i as u64);
        }
        tree
    }

    #[divan::bench(args = SIZES)]
    fn masstree_arc(bencher: Bencher, n: usize) {
        let tree = setup_masstree(n);
        let target_key = ((n / 2) as u64).to_be_bytes();
        bencher.bench_local(|| tree.get(black_box(&target_key)));
    }

    #[divan::bench(args = SIZES)]
    fn masstree_copy(bencher: Bencher, n: usize) {
        let tree = setup_masstree_index(n);
        let target_key = ((n / 2) as u64).to_be_bytes();
        bencher.bench_local(|| tree.get(black_box(&target_key)));
    }

    #[divan::bench(args = SIZES)]
    fn btreemap(bencher: Bencher, n: usize) {
        let tree = setup_btreemap(n);
        let target_key = ((n / 2) as u64).to_be_bytes().to_vec();
        bencher.bench_local(|| tree.get(black_box(&target_key)));
    }
}

// =============================================================================
// SINGLE GET: Miss
// =============================================================================

#[divan::bench_group(name = "04_single_get_miss")]
mod single_get_miss {
    use super::{BTreeMap, Bencher, MassTree, black_box};

    const SIZES: &[usize] = &[10, 100, 1000];

    fn setup_masstree(n: usize) -> MassTree<u64> {
        let mut tree = MassTree::new();
        for i in 0..n {
            let key = (i as u64).to_be_bytes();
            let _ = tree.insert(&key, i as u64);
        }
        tree
    }

    fn setup_btreemap(n: usize) -> BTreeMap<Vec<u8>, u64> {
        let mut tree = BTreeMap::new();
        for i in 0..n {
            let key = (i as u64).to_be_bytes().to_vec();
            tree.insert(key, i as u64);
        }
        tree
    }

    #[divan::bench(args = SIZES)]
    fn masstree(bencher: Bencher, n: usize) {
        let tree = setup_masstree(n);
        // Key that doesn't exist
        let missing_key = ((n + 1000) as u64).to_be_bytes();
        bencher.bench_local(|| tree.get(black_box(&missing_key)));
    }

    #[divan::bench(args = SIZES)]
    fn btreemap(bencher: Bencher, n: usize) {
        let tree = setup_btreemap(n);
        let missing_key = ((n + 1000) as u64).to_be_bytes().to_vec();
        bencher.bench_local(|| tree.get(black_box(&missing_key)));
    }
}

// =============================================================================
// BATCH INSERT: Sequential Keys
// =============================================================================

#[divan::bench_group(name = "05_batch_insert_sequential")]
mod batch_insert_sequential {
    use super::{BTreeMap, Bencher, MassTree, MassTreeIndex, sequential_keys_bytes};

    const SIZES: &[usize] = &[10, 50, 100, 500];

    #[divan::bench(args = SIZES)]
    fn masstree_arc(bencher: Bencher, n: usize) {
        let keys = sequential_keys_bytes(n);
        bencher
            .with_inputs(|| (MassTree::<u64>::new(), keys.clone()))
            .bench_local_values(|(mut tree, keys)| {
                for (i, key) in keys.iter().enumerate() {
                    let _ = tree.insert(key, i as u64);
                }
                tree
            });
    }

    #[divan::bench(args = SIZES)]
    fn masstree_copy(bencher: Bencher, n: usize) {
        let keys = sequential_keys_bytes(n);
        bencher
            .with_inputs(|| (MassTreeIndex::<u64>::new(), keys.clone()))
            .bench_local_values(|(mut tree, keys)| {
                for (i, key) in keys.iter().enumerate() {
                    let _ = tree.insert(key, i as u64);
                }
                tree
            });
    }

    #[divan::bench(args = SIZES)]
    fn btreemap(bencher: Bencher, n: usize) {
        let keys = sequential_keys_bytes(n);
        bencher
            .with_inputs(|| (BTreeMap::<Vec<u8>, u64>::new(), keys.clone()))
            .bench_local_values(|(mut tree, keys)| {
                for (i, key) in keys.into_iter().enumerate() {
                    tree.insert(key, i as u64);
                }
                tree
            });
    }
}

// =============================================================================
// BATCH INSERT: Random Order
// =============================================================================

#[divan::bench_group(name = "06_batch_insert_random")]
mod batch_insert_random {
    use super::{
        BTreeMap, Bencher, MassTree, MassTreeIndex, sequential_keys_bytes, shuffled_indices,
    };

    const SIZES: &[usize] = &[10, 50, 100, 500];

    #[divan::bench(args = SIZES)]
    fn masstree_arc(bencher: Bencher, n: usize) {
        let keys = sequential_keys_bytes(n);
        let order = shuffled_indices(n, 31337);
        bencher
            .with_inputs(|| (MassTree::<u64>::new(), keys.clone(), order.clone()))
            .bench_local_values(|(mut tree, keys, order)| {
                for &i in &order {
                    let _ = tree.insert(&keys[i], i as u64);
                }
                tree
            });
    }

    #[divan::bench(args = SIZES)]
    fn masstree_copy(bencher: Bencher, n: usize) {
        let keys = sequential_keys_bytes(n);
        let order = shuffled_indices(n, 31337);
        bencher
            .with_inputs(|| (MassTreeIndex::<u64>::new(), keys.clone(), order.clone()))
            .bench_local_values(|(mut tree, keys, order)| {
                for &i in &order {
                    let _ = tree.insert(&keys[i], i as u64);
                }
                tree
            });
    }

    #[divan::bench(args = SIZES)]
    fn btreemap(bencher: Bencher, n: usize) {
        let keys = sequential_keys_bytes(n);
        let order = shuffled_indices(n, 31337);
        bencher
            .with_inputs(|| (BTreeMap::<Vec<u8>, u64>::new(), keys.clone(), order.clone()))
            .bench_local_values(|(mut tree, keys, order)| {
                for &i in &order {
                    tree.insert(keys[i].clone(), i as u64);
                }
                tree
            });
    }
}

// =============================================================================
// BATCH GET: Sequential Access
// =============================================================================

#[divan::bench_group(name = "07_batch_get_sequential")]
mod batch_get_sequential {
    use super::{BTreeMap, Bencher, MassTree, black_box, sequential_keys_bytes};

    const SIZES: &[usize] = &[10, 50, 100, 500];

    fn setup_masstree(n: usize) -> (MassTree<u64>, Vec<Vec<u8>>) {
        let mut tree = MassTree::new();
        let keys = sequential_keys_bytes(n);
        for (i, key) in keys.iter().enumerate() {
            let _ = tree.insert(key, i as u64);
        }
        (tree, keys)
    }

    fn setup_btreemap(n: usize) -> (BTreeMap<Vec<u8>, u64>, Vec<Vec<u8>>) {
        let mut tree = BTreeMap::new();
        let keys = sequential_keys_bytes(n);
        for (i, key) in keys.iter().enumerate() {
            tree.insert(key.clone(), i as u64);
        }
        (tree, keys)
    }

    #[divan::bench(args = SIZES)]
    fn masstree(bencher: Bencher, n: usize) {
        let (tree, keys) = setup_masstree(n);
        bencher.bench_local(|| {
            let mut sum = 0u64;
            for key in &keys {
                if let Some(v) = tree.get(key) {
                    sum += *v;
                }
            }
            black_box(sum)
        });
    }

    #[divan::bench(args = SIZES)]
    fn btreemap(bencher: Bencher, n: usize) {
        let (tree, keys) = setup_btreemap(n);
        bencher.bench_local(|| {
            let mut sum = 0u64;
            for key in &keys {
                if let Some(&v) = tree.get(key) {
                    sum += v;
                }
            }
            black_box(sum)
        });
    }
}

// =============================================================================
// BATCH GET: Random Access
// =============================================================================

#[divan::bench_group(name = "08_batch_get_random")]
mod batch_get_random {
    use super::{BTreeMap, Bencher, MassTree, black_box, sequential_keys_bytes, shuffled_indices};

    const SIZES: &[usize] = &[10, 50, 100, 500];

    fn setup_masstree(n: usize) -> (MassTree<u64>, Vec<Vec<u8>>) {
        let mut tree = MassTree::new();
        let keys = sequential_keys_bytes(n);
        for (i, key) in keys.iter().enumerate() {
            let _ = tree.insert(key, i as u64);
        }
        (tree, keys)
    }

    fn setup_btreemap(n: usize) -> (BTreeMap<Vec<u8>, u64>, Vec<Vec<u8>>) {
        let mut tree = BTreeMap::new();
        let keys = sequential_keys_bytes(n);
        for (i, key) in keys.iter().enumerate() {
            tree.insert(key.clone(), i as u64);
        }
        (tree, keys)
    }

    #[divan::bench(args = SIZES)]
    fn masstree(bencher: Bencher, n: usize) {
        let (tree, keys) = setup_masstree(n);
        let order = shuffled_indices(n, 31337);
        bencher.bench_local(|| {
            let mut sum = 0u64;
            for &i in &order {
                if let Some(v) = tree.get(&keys[i]) {
                    sum += *v;
                }
            }
            black_box(sum)
        });
    }

    #[divan::bench(args = SIZES)]
    fn btreemap(bencher: Bencher, n: usize) {
        let (tree, keys) = setup_btreemap(n);
        let order = shuffled_indices(n, 31337);
        bencher.bench_local(|| {
            let mut sum = 0u64;
            for &i in &order {
                if let Some(&v) = tree.get(&keys[i]) {
                    sum += v;
                }
            }
            black_box(sum)
        });
    }
}

// =============================================================================
// UPDATE EXISTING KEY
// =============================================================================

#[divan::bench_group(name = "09_update_existing")]
mod update_existing {
    use super::{BTreeMap, Bencher, MassTree, black_box};

    const SIZES: &[usize] = &[10, 100, 1000];

    fn setup_masstree(n: usize) -> MassTree<u64> {
        let mut tree = MassTree::new();
        for i in 0..n {
            let key = (i as u64).to_be_bytes();
            let _ = tree.insert(&key, i as u64);
        }
        tree
    }

    fn setup_btreemap(n: usize) -> BTreeMap<Vec<u8>, u64> {
        let mut tree = BTreeMap::new();
        for i in 0..n {
            let key = (i as u64).to_be_bytes().to_vec();
            tree.insert(key, i as u64);
        }
        tree
    }

    #[divan::bench(args = SIZES)]
    fn masstree(bencher: Bencher, n: usize) {
        let target_key = ((n / 2) as u64).to_be_bytes();
        bencher
            .with_inputs(|| setup_masstree(n))
            .bench_local_values(|mut tree| {
                let _ = tree.insert(black_box(&target_key), black_box(9999u64));
                tree
            });
    }

    #[divan::bench(args = SIZES)]
    fn btreemap(bencher: Bencher, n: usize) {
        let target_key = ((n / 2) as u64).to_be_bytes().to_vec();
        bencher
            .with_inputs(|| (setup_btreemap(n), target_key.clone()))
            .bench_local_values(|(mut tree, key)| {
                tree.insert(black_box(key), black_box(9999u64));
                tree
            });
    }
}

// =============================================================================
// STRING KEYS (Variable Length)
// =============================================================================

#[divan::bench_group(name = "10_string_keys")]
mod string_keys {
    use super::{BTreeMap, Bencher, MassTree, black_box};

    /// Generate string keys of varying lengths
    fn string_keys(n: usize, prefix: &str) -> Vec<String> {
        (0..n).map(|i| format!("{prefix}{i:08x}")).collect()
    }

    const SIZES: &[usize] = &[50, 100, 500];

    #[divan::bench(args = SIZES)]
    fn masstree_insert(bencher: Bencher, n: usize) {
        let keys = string_keys(n, "key_prefix_");
        bencher
            .with_inputs(|| (MassTree::<u64>::new(), keys.clone()))
            .bench_local_values(|(mut tree, keys)| {
                for (i, key) in keys.iter().enumerate() {
                    let _ = tree.insert(key.as_bytes(), i as u64);
                }
                tree
            });
    }

    #[divan::bench(args = SIZES)]
    fn btreemap_insert(bencher: Bencher, n: usize) {
        let keys = string_keys(n, "key_prefix_");
        bencher
            .with_inputs(|| (BTreeMap::<String, u64>::new(), keys.clone()))
            .bench_local_values(|(mut tree, keys)| {
                for (i, key) in keys.into_iter().enumerate() {
                    tree.insert(key, i as u64);
                }
                tree
            });
    }

    #[divan::bench(args = SIZES)]
    fn masstree_get(bencher: Bencher, n: usize) {
        let keys = string_keys(n, "key_prefix_");
        let mut tree = MassTree::<u64>::new();
        for (i, key) in keys.iter().enumerate() {
            let _ = tree.insert(key.as_bytes(), i as u64);
        }
        let target = &keys[n / 2];
        bencher.bench_local(|| tree.get(black_box(target.as_bytes())));
    }

    #[divan::bench(args = SIZES)]
    fn btreemap_get(bencher: Bencher, n: usize) {
        let keys = string_keys(n, "key_prefix_");
        let mut tree = BTreeMap::<String, u64>::new();
        for (i, key) in keys.iter().enumerate() {
            tree.insert(key.clone(), i as u64);
        }
        let target = &keys[n / 2];
        bencher.bench_local(|| tree.get(black_box(target)));
    }
}

// =============================================================================
// LONG KEYS (Multi-layer for MassTree)
// =============================================================================

#[divan::bench_group(name = "11_long_keys")]
mod long_keys {
    use super::{BTreeMap, Bencher, MassTree, black_box};

    /// Generate long keys that span multiple 8-byte slices (multiple `MassTree` layers)
    fn long_keys(n: usize) -> Vec<Vec<u8>> {
        (0..n)
            .map(|i| {
                // 24-byte key: "prefix__" (8) + hex (8) + "__suffix" (8)
                format!("prefix__{i:016x}__suffix").into_bytes()
            })
            .collect()
    }

    const SIZES: &[usize] = &[50, 100, 500];

    #[divan::bench(args = SIZES)]
    fn masstree_insert(bencher: Bencher, n: usize) {
        let keys = long_keys(n);
        bencher
            .with_inputs(|| (MassTree::<u64>::new(), keys.clone()))
            .bench_local_values(|(mut tree, keys)| {
                for (i, key) in keys.iter().enumerate() {
                    let _ = tree.insert(key, i as u64);
                }
                tree
            });
    }

    #[divan::bench(args = SIZES)]
    fn btreemap_insert(bencher: Bencher, n: usize) {
        let keys = long_keys(n);
        bencher
            .with_inputs(|| (BTreeMap::<Vec<u8>, u64>::new(), keys.clone()))
            .bench_local_values(|(mut tree, keys)| {
                for (i, key) in keys.into_iter().enumerate() {
                    tree.insert(key, i as u64);
                }
                tree
            });
    }

    #[divan::bench(args = SIZES)]
    fn masstree_get(bencher: Bencher, n: usize) {
        let keys = long_keys(n);
        let mut tree = MassTree::<u64>::new();
        for (i, key) in keys.iter().enumerate() {
            let _ = tree.insert(key, i as u64);
        }
        let target = &keys[n / 2];
        bencher.bench_local(|| tree.get(black_box(target)));
    }

    #[divan::bench(args = SIZES)]
    fn btreemap_get(bencher: Bencher, n: usize) {
        let keys = long_keys(n);
        let mut tree = BTreeMap::<Vec<u8>, u64>::new();
        for (i, key) in keys.iter().enumerate() {
            tree.insert(key.clone(), i as u64);
        }
        let target = &keys[n / 2];
        bencher.bench_local(|| tree.get(black_box(target)));
    }
}

// =============================================================================
// MIXED WORKLOAD: 50/50 Read/Write
// =============================================================================

#[divan::bench_group(name = "12_mixed_workload_50_50")]
mod mixed_workload_50_50 {
    use super::{BTreeMap, Bencher, MassTree, black_box};

    const SIZES: &[usize] = &[100, 500, 1000];

    fn setup_masstree(n: usize) -> MassTree<u64> {
        let mut tree = MassTree::new();
        for i in 0..n {
            let key = (i as u64).to_be_bytes();
            let _ = tree.insert(&key, i as u64);
        }
        tree
    }

    fn setup_btreemap(n: usize) -> BTreeMap<Vec<u8>, u64> {
        let mut tree = BTreeMap::new();
        for i in 0..n {
            let key = (i as u64).to_be_bytes().to_vec();
            tree.insert(key, i as u64);
        }
        tree
    }

    #[divan::bench(args = SIZES)]
    fn masstree(bencher: Bencher, n: usize) {
        let read_keys: Vec<[u8; 8]> = (0..10)
            .map(|i| ((i * n / 10) as u64).to_be_bytes())
            .collect();
        bencher
            .with_inputs(|| (setup_masstree(n), n as u64))
            .bench_local_values(|(mut tree, mut counter)| {
                // 10 reads
                let mut sum = 0u64;
                for key in &read_keys {
                    if let Some(v) = tree.get(key) {
                        sum += *v;
                    }
                }
                // 10 writes
                for _ in 0..10 {
                    let key = counter.to_be_bytes();
                    let _ = tree.insert(&key, counter);
                    counter += 1;
                }
                black_box(sum);
                (tree, counter)
            });
    }

    #[divan::bench(args = SIZES)]
    fn btreemap(bencher: Bencher, n: usize) {
        let read_keys: Vec<Vec<u8>> = (0..10)
            .map(|i| ((i * n / 10) as u64).to_be_bytes().to_vec())
            .collect();
        bencher
            .with_inputs(|| (setup_btreemap(n), n as u64))
            .bench_local_values(|(mut tree, mut counter)| {
                // 10 reads
                let mut sum = 0u64;
                for key in &read_keys {
                    if let Some(&v) = tree.get(key) {
                        sum += v;
                    }
                }
                // 10 writes
                for _ in 0..10 {
                    let key = counter.to_be_bytes().to_vec();
                    tree.insert(key, counter);
                    counter += 1;
                }
                black_box(sum);
                (tree, counter)
            });
    }
}

// =============================================================================
// MIXED WORKLOAD: 90/10 Read-Heavy
// =============================================================================

#[divan::bench_group(name = "13_mixed_workload_90_10")]
mod mixed_workload_90_10 {
    use super::{BTreeMap, Bencher, MassTree, black_box};

    const SIZES: &[usize] = &[100, 500, 1000];

    fn setup_masstree(n: usize) -> MassTree<u64> {
        let mut tree = MassTree::new();
        for i in 0..n {
            let key = (i as u64).to_be_bytes();
            let _ = tree.insert(&key, i as u64);
        }
        tree
    }

    fn setup_btreemap(n: usize) -> BTreeMap<Vec<u8>, u64> {
        let mut tree = BTreeMap::new();
        for i in 0..n {
            let key = (i as u64).to_be_bytes().to_vec();
            tree.insert(key, i as u64);
        }
        tree
    }

    #[divan::bench(args = SIZES)]
    fn masstree(bencher: Bencher, n: usize) {
        let read_keys: Vec<[u8; 8]> = (0..18)
            .map(|i| ((i * n / 18) as u64).to_be_bytes())
            .collect();
        bencher
            .with_inputs(|| (setup_masstree(n), n as u64))
            .bench_local_values(|(mut tree, mut counter)| {
                // 18 reads (90%)
                let mut sum = 0u64;
                for key in &read_keys {
                    if let Some(v) = tree.get(key) {
                        sum += *v;
                    }
                }
                // 2 writes (10%)
                for _ in 0..2 {
                    let key = counter.to_be_bytes();
                    let _ = tree.insert(&key, counter);
                    counter += 1;
                }
                black_box(sum);
                (tree, counter)
            });
    }

    #[divan::bench(args = SIZES)]
    fn btreemap(bencher: Bencher, n: usize) {
        let read_keys: Vec<Vec<u8>> = (0..18)
            .map(|i| ((i * n / 18) as u64).to_be_bytes().to_vec())
            .collect();
        bencher
            .with_inputs(|| (setup_btreemap(n), n as u64))
            .bench_local_values(|(mut tree, mut counter)| {
                // 18 reads (90%)
                let mut sum = 0u64;
                for key in &read_keys {
                    if let Some(&v) = tree.get(key) {
                        sum += v;
                    }
                }
                // 2 writes (10%)
                for _ in 0..2 {
                    let key = counter.to_be_bytes().to_vec();
                    tree.insert(key, counter);
                    counter += 1;
                }
                black_box(sum);
                (tree, counter)
            });
    }
}

// =============================================================================
// SCALING ANALYSIS: Insert Latency vs Tree Size
// =============================================================================

#[divan::bench_group(name = "14_scaling_insert")]
mod scaling_insert {
    use super::{BTreeMap, Bencher, MassTree, black_box};

    const SIZES: &[usize] = &[100, 500, 1000, 5000, 10000];

    fn setup_masstree(n: usize) -> MassTree<u64> {
        let mut tree = MassTree::new();
        for i in 0..n {
            let key = (i as u64).to_be_bytes();
            let _ = tree.insert(&key, i as u64);
        }
        tree
    }

    fn setup_btreemap(n: usize) -> BTreeMap<Vec<u8>, u64> {
        let mut tree = BTreeMap::new();
        for i in 0..n {
            let key = (i as u64).to_be_bytes().to_vec();
            tree.insert(key, i as u64);
        }
        tree
    }

    #[divan::bench(args = SIZES)]
    fn masstree(bencher: Bencher, n: usize) {
        let new_key = (n as u64).to_be_bytes();
        bencher
            .with_inputs(|| setup_masstree(n))
            .bench_local_values(|mut tree| {
                let _ = tree.insert(black_box(&new_key), black_box(9999u64));
                tree
            });
    }

    #[divan::bench(args = SIZES)]
    fn btreemap(bencher: Bencher, n: usize) {
        let new_key = (n as u64).to_be_bytes().to_vec();
        bencher
            .with_inputs(|| (setup_btreemap(n), new_key.clone()))
            .bench_local_values(|(mut tree, key)| {
                tree.insert(black_box(key), black_box(9999u64));
                tree
            });
    }
}

// =============================================================================
// SCALING ANALYSIS: Get Latency vs Tree Size
// =============================================================================

#[divan::bench_group(name = "15_scaling_get")]
mod scaling_get {
    use super::{BTreeMap, Bencher, MassTree, black_box};

    const SIZES: &[usize] = &[100, 500, 1000, 5000, 10000];

    fn setup_masstree(n: usize) -> MassTree<u64> {
        let mut tree = MassTree::new();
        for i in 0..n {
            let key = (i as u64).to_be_bytes();
            let _ = tree.insert(&key, i as u64);
        }
        tree
    }

    fn setup_btreemap(n: usize) -> BTreeMap<Vec<u8>, u64> {
        let mut tree = BTreeMap::new();
        for i in 0..n {
            let key = (i as u64).to_be_bytes().to_vec();
            tree.insert(key, i as u64);
        }
        tree
    }

    #[divan::bench(args = SIZES)]
    fn masstree(bencher: Bencher, n: usize) {
        let tree = setup_masstree(n);
        let target_key = ((n / 2) as u64).to_be_bytes();
        bencher.bench_local(|| tree.get(black_box(&target_key)));
    }

    #[divan::bench(args = SIZES)]
    fn btreemap(bencher: Bencher, n: usize) {
        let tree = setup_btreemap(n);
        let target_key = ((n / 2) as u64).to_be_bytes().to_vec();
        bencher.bench_local(|| tree.get(black_box(&target_key)));
    }
}

// =============================================================================
// PER-OPERATION COST: Amortized Insert
// =============================================================================

#[divan::bench_group(name = "16_amortized_insert")]
mod amortized_insert {
    use super::{BTreeMap, Bencher, MassTree, sequential_keys_bytes};

    /// Measures cost per insert by dividing total time by count
    /// This captures the true average including split amortization
    const SIZES: &[usize] = &[100, 500, 1000];

    #[divan::bench(args = SIZES)]
    fn masstree_per_op(bencher: Bencher, n: usize) {
        let keys = sequential_keys_bytes(n);
        bencher
            .counter(divan::counter::ItemsCount::new(n))
            .with_inputs(|| (MassTree::<u64>::new(), keys.clone()))
            .bench_local_values(|(mut tree, keys)| {
                for (i, key) in keys.iter().enumerate() {
                    let _ = tree.insert(key, i as u64);
                }
                tree
            });
    }

    #[divan::bench(args = SIZES)]
    fn btreemap_per_op(bencher: Bencher, n: usize) {
        let keys = sequential_keys_bytes(n);
        bencher
            .counter(divan::counter::ItemsCount::new(n))
            .with_inputs(|| (BTreeMap::<Vec<u8>, u64>::new(), keys.clone()))
            .bench_local_values(|(mut tree, keys)| {
                for (i, key) in keys.into_iter().enumerate() {
                    tree.insert(key, i as u64);
                }
                tree
            });
    }
}

// =============================================================================
// PER-OPERATION COST: Amortized Get
// =============================================================================

#[divan::bench_group(name = "17_amortized_get")]
mod amortized_get {
    use super::{BTreeMap, Bencher, MassTree, black_box, sequential_keys_bytes};

    const SIZES: &[usize] = &[100, 500, 1000];

    fn setup_masstree(n: usize) -> (MassTree<u64>, Vec<Vec<u8>>) {
        let mut tree = MassTree::new();
        let keys = sequential_keys_bytes(n);
        for (i, key) in keys.iter().enumerate() {
            let _ = tree.insert(key, i as u64);
        }
        (tree, keys)
    }

    fn setup_btreemap(n: usize) -> (BTreeMap<Vec<u8>, u64>, Vec<Vec<u8>>) {
        let mut tree = BTreeMap::new();
        let keys = sequential_keys_bytes(n);
        for (i, key) in keys.iter().enumerate() {
            tree.insert(key.clone(), i as u64);
        }
        (tree, keys)
    }

    #[divan::bench(args = SIZES)]
    fn masstree_per_op(bencher: Bencher, n: usize) {
        let (tree, keys) = setup_masstree(n);
        bencher
            .counter(divan::counter::ItemsCount::new(n))
            .bench_local(|| {
                let mut sum = 0u64;
                for key in &keys {
                    if let Some(v) = tree.get(key) {
                        sum += *v;
                    }
                }
                black_box(sum)
            });
    }

    #[divan::bench(args = SIZES)]
    fn btreemap_per_op(bencher: Bencher, n: usize) {
        let (tree, keys) = setup_btreemap(n);
        bencher
            .counter(divan::counter::ItemsCount::new(n))
            .bench_local(|| {
                let mut sum = 0u64;
                for key in &keys {
                    if let Some(&v) = tree.get(key) {
                        sum += v;
                    }
                }
                black_box(sum)
            });
    }
}

// =============================================================================
// MEMORY ALLOCATION: Pre-allocated vs Dynamic
// =============================================================================

#[divan::bench_group(name = "18_allocation_patterns")]
mod allocation_patterns {
    use super::{BTreeMap, Bencher, MassTree};

    /// `BTreeMap` with pre-sized hint (not directly possible, but shows baseline)
    #[divan::bench]
    fn btreemap_from_iter(bencher: Bencher) {
        let items: Vec<(Vec<u8>, u64)> =
            (0..100u64).map(|i| (i.to_be_bytes().to_vec(), i)).collect();
        bencher
            .with_inputs(|| items.clone())
            .bench_local_values(|items| {
                let tree: BTreeMap<Vec<u8>, u64> = items.into_iter().collect();
                tree
            });
    }

    /// `MassTree` individual inserts (for comparison)
    #[divan::bench]
    fn masstree_individual(bencher: Bencher) {
        bencher
            .with_inputs(MassTree::<u64>::new)
            .bench_local_values(|mut tree| {
                for i in 0..100u64 {
                    let key = i.to_be_bytes();
                    let _ = tree.insert(&key, i);
                }
                tree
            });
    }
}
