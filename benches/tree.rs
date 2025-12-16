//! Benchmarks for `MassTree` and `MassTreeIndex` using Divan.
//!
//! Run with: `cargo bench --bench tree`
#![expect(clippy::cast_possible_truncation)]

use divan::{Bencher, black_box};
use masstree::tree::{MassTree, MassTreeIndex};
use std::sync::Arc;

fn main() {
    divan::main();
}

// =============================================================================
// Construction
// =============================================================================

#[divan::bench_group]
mod construction {
    use super::{MassTree, MassTreeIndex};

    #[divan::bench]
    fn new_masstree() -> MassTree<u64> {
        MassTree::new()
    }

    #[divan::bench]
    fn default_masstree() -> MassTree<u64> {
        MassTree::default()
    }

    #[divan::bench]
    fn new_masstree_index() -> MassTreeIndex<u64> {
        MassTreeIndex::new()
    }

    #[divan::bench]
    fn default_masstree_index() -> MassTreeIndex<u64> {
        MassTreeIndex::default()
    }
}

// =============================================================================
// Insert Operations
// =============================================================================

#[divan::bench_group]
mod insert {
    use super::{Arc, Bencher, MassTree, black_box};

    #[divan::bench]
    fn insert_single(bencher: Bencher) {
        bencher
            .with_inputs(MassTree::<u64>::new)
            .bench_local_values(|mut tree| {
                let _ = tree.insert(black_box(b"hello"), black_box(42u64));
                tree
            });
    }

    #[divan::bench]
    fn insert_into_existing(bencher: Bencher) {
        bencher
            .with_inputs(|| {
                let mut tree = MassTree::<u64>::new();
                for i in 0..10u64 {
                    let key = i.to_be_bytes();
                    let _ = tree.insert(&key, i);
                }
                tree
            })
            .bench_local_values(|mut tree| {
                let _ = tree.insert(black_box(b"newkey!!"), black_box(999u64));
                tree
            });
    }

    #[divan::bench]
    fn insert_update_existing(bencher: Bencher) {
        bencher
            .with_inputs(|| {
                let mut tree = MassTree::<u64>::new();
                let _ = tree.insert(b"key", 1u64);
                tree
            })
            .bench_local_values(|mut tree| {
                let old = tree.insert(black_box(b"key"), black_box(2u64));
                let _ = black_box(old);
                tree
            });
    }

    #[divan::bench]
    fn insert_arc(bencher: Bencher) {
        bencher
            .with_inputs(MassTree::<u64>::new)
            .bench_local_values(|mut tree| {
                let _ = tree.insert_arc(black_box(b"hello"), black_box(Arc::new(42u64)));
                tree
            });
    }

    // Insert with varying key lengths
    #[divan::bench(args = [1, 2, 4, 6, 8])]
    fn insert_key_len(bencher: Bencher, len: usize) {
        let key: Vec<u8> = (0..len).map(|i| (i as u8) + b'a').collect();

        bencher
            .with_inputs(MassTree::<u64>::new)
            .bench_local_values(|mut tree| {
                let _ = tree.insert(black_box(&key), black_box(42u64));

                tree
            });
    }
}

// =============================================================================
// Get Operations
// =============================================================================

#[divan::bench_group]
mod get {
    use super::{Bencher, MassTree, black_box};

    fn setup_tree(n: usize) -> MassTree<u64> {
        let mut tree = MassTree::new();
        for i in 0..n {
            let key = (i as u64).to_be_bytes();
            let _ = tree.insert(&key, i as u64);
        }
        tree
    }

    #[divan::bench]
    fn get_from_empty(bencher: Bencher) {
        let tree = MassTree::<u64>::new();
        bencher.bench_local(|| tree.get(black_box(b"missing")));
    }

    #[divan::bench]
    fn get_hit(bencher: Bencher) {
        let mut tree = MassTree::<u64>::new();
        let _ = tree.insert(b"hello", 42u64);
        bencher.bench_local(|| tree.get(black_box(b"hello")));
    }

    #[divan::bench]
    fn get_miss(bencher: Bencher) {
        let mut tree = MassTree::<u64>::new();
        let _ = tree.insert(b"hello", 42u64);
        bencher.bench_local(|| tree.get(black_box(b"world")));
    }

    #[divan::bench(args = [5, 10, 15])]
    fn get_from_single_leaf(bencher: Bencher, n: usize) {
        let tree = setup_tree(n);
        let key = ((n / 2) as u64).to_be_bytes();
        bencher.bench_local(|| tree.get(black_box(&key)));
    }

    #[divan::bench(args = [20, 50, 100])]
    fn get_from_multi_leaf(bencher: Bencher, n: usize) {
        let tree = setup_tree(n);
        let key = ((n / 2) as u64).to_be_bytes();
        bencher.bench_local(|| tree.get(black_box(&key)));
    }

    // Get with varying key lengths
    #[divan::bench(args = [1, 2, 4, 6, 8])]
    fn get_key_len(bencher: Bencher, len: usize) {
        let key: Vec<u8> = (0..len).map(|i| i as u8 + b'a').collect();
        let mut tree = MassTree::<u64>::new();
        let _ = tree.insert(&key, 42u64);
        bencher.bench_local(|| tree.get(black_box(&key)));
    }
}

// =============================================================================
// Index Mode (Copy values)
// =============================================================================

#[divan::bench_group]
mod index_mode {
    use super::{Bencher, MassTreeIndex, black_box};

    #[divan::bench]
    fn insert(bencher: Bencher) {
        bencher
            .with_inputs(MassTreeIndex::<u64>::new)
            .bench_local_values(|mut tree| {
                let _ = tree.insert(black_box(b"hello"), black_box(42u64));
                tree
            });
    }

    #[divan::bench]
    fn get_hit(bencher: Bencher) {
        let mut tree = MassTreeIndex::<u64>::new();
        let _ = tree.insert(b"hello", 42u64);
        bencher.bench_local(|| tree.get(black_box(b"hello")));
    }

    #[divan::bench]
    fn get_miss(bencher: Bencher) {
        let mut tree = MassTreeIndex::<u64>::new();
        let _ = tree.insert(b"hello", 42u64);
        bencher.bench_local(|| tree.get(black_box(b"world")));
    }
}

// =============================================================================
// Insert Batches (triggering splits)
// =============================================================================

#[divan::bench_group]
mod batch_insert {
    use super::{Bencher, MassTree};

    #[divan::bench(args = [10, 20, 50, 100])]
    fn sequential_insert(bencher: Bencher, n: usize) {
        bencher
            .with_inputs(MassTree::<u64>::new)
            .bench_local_values(|mut tree| {
                for i in 0..n {
                    let key = (i as u64).to_be_bytes();
                    let _ = tree.insert(&key, i as u64);
                }
                tree
            });
    }

    #[divan::bench(args = [10, 20, 50, 100])]
    fn reverse_insert(bencher: Bencher, n: usize) {
        bencher
            .with_inputs(MassTree::<u64>::new)
            .bench_local_values(|mut tree| {
                for i in (0..n).rev() {
                    let key = (i as u64).to_be_bytes();
                    let _ = tree.insert(&key, i as u64);
                }
                tree
            });
    }

    #[divan::bench(args = [10, 20, 50, 100])]
    fn random_insert(bencher: Bencher, n: usize) {
        // Pre-generate shuffled keys for reproducibility
        let mut keys: Vec<u64> = (0..n as u64).collect();
        // Simple shuffle using deterministic pattern
        for i in 0..keys.len() {
            let j = (i * 7 + 3) % keys.len();
            keys.swap(i, j);
        }

        bencher
            .with_inputs(|| (MassTree::<u64>::new(), keys.clone()))
            .bench_local_values(|(mut tree, keys)| {
                for &k in &keys {
                    let key = k.to_be_bytes();
                    let _ = tree.insert(&key, k);
                }
                tree
            });
    }
}

// =============================================================================
// Get Batches
// =============================================================================

#[divan::bench_group]
mod batch_get {
    use super::{Bencher, MassTree, black_box};

    fn setup_tree(n: usize) -> MassTree<u64> {
        let mut tree = MassTree::new();
        for i in 0..n {
            let key = (i as u64).to_be_bytes();
            let _ = tree.insert(&key, i as u64);
        }
        tree
    }

    #[divan::bench(args = [10, 50, 100])]
    fn sequential_get_all(bencher: Bencher, n: usize) {
        let tree = setup_tree(n);
        bencher.bench_local(|| {
            let mut sum = 0u64;
            for i in 0..n {
                let key = (i as u64).to_be_bytes();
                if let Some(v) = tree.get(&key) {
                    sum += *v;
                }
            }
            black_box(sum)
        });
    }

    #[divan::bench(args = [10, 50, 100])]
    fn random_get_all(bencher: Bencher, n: usize) {
        let tree = setup_tree(n);
        // Pre-generate shuffled access pattern
        let mut indices: Vec<usize> = (0..n).collect();
        for i in 0..indices.len() {
            let j = (i * 7 + 3) % indices.len();
            indices.swap(i, j);
        }

        bencher.bench_local(|| {
            let mut sum = 0u64;
            for &i in &indices {
                let key = (i as u64).to_be_bytes();
                if let Some(v) = tree.get(&key) {
                    sum += *v;
                }
            }
            black_box(sum)
        });
    }
}

// =============================================================================
// Mixed Workloads
// =============================================================================

#[divan::bench_group]
mod workload {
    use super::{Bencher, MassTree, black_box};

    #[divan::bench]
    fn read_heavy_90_10(bencher: Bencher) {
        // 90% reads, 10% writes
        bencher
            .with_inputs(|| {
                let mut tree = MassTree::<u64>::new();
                for i in 0..100u64 {
                    let key = i.to_be_bytes();
                    let _ = tree.insert(&key, i);
                }
                tree
            })
            .bench_local_values(|mut tree| {
                // 9 reads
                for i in 0..9u64 {
                    let key = (i * 10).to_be_bytes();
                    black_box(tree.get(&key));
                }
                // 1 write
                let key = 200u64.to_be_bytes();
                let _ = tree.insert(&key, 200);
                tree
            });
    }

    #[divan::bench]
    fn write_heavy_10_90(bencher: Bencher) {
        // 10% reads, 90% writes
        bencher
            .with_inputs(|| {
                let mut tree = MassTree::<u64>::new();
                for i in 0..10u64 {
                    let key = i.to_be_bytes();
                    let _ = tree.insert(&key, i);
                }
                (tree, 10u64)
            })
            .bench_local_values(|(mut tree, mut counter)| {
                // 1 read
                let read_key = 5u64.to_be_bytes();
                black_box(tree.get(&read_key));

                // 9 writes
                for _ in 0..9 {
                    let key = counter.to_be_bytes();
                    let _ = tree.insert(&key, counter);
                    counter += 1;
                }
                (tree, counter)
            });
    }

    #[divan::bench]
    fn update_existing(bencher: Bencher) {
        // All operations are updates to existing keys
        bencher
            .with_inputs(|| {
                let mut tree = MassTree::<u64>::new();
                for i in 0..50u64 {
                    let key = i.to_be_bytes();
                    let _ = tree.insert(&key, i);
                }
                tree
            })
            .bench_local_values(|mut tree| {
                for i in 0..10u64 {
                    let key = (i * 5).to_be_bytes();
                    let _ = tree.insert(&key, i * 100);
                }
                tree
            });
    }
}

// =============================================================================
// Scaling Analysis
// =============================================================================

#[divan::bench_group]
mod scaling {
    use super::{Bencher, MassTree, black_box};

    fn setup_tree(n: usize) -> MassTree<u64> {
        let mut tree = MassTree::new();
        for i in 0..n {
            let key = (i as u64).to_be_bytes();
            let _ = tree.insert(&key, i as u64);
        }
        tree
    }

    #[divan::bench(args = [100, 500, 1000])]
    fn insert_into_n(bencher: Bencher, n: usize) {
        bencher
            .with_inputs(|| setup_tree(n))
            .bench_local_values(|mut tree| {
                let key = (n as u64).to_be_bytes();
                let _ = tree.insert(&key, n as u64);
                tree
            });
    }

    #[divan::bench(args = [100, 500, 1000])]
    fn get_from_n(bencher: Bencher, n: usize) {
        let tree = setup_tree(n);
        let key = ((n / 2) as u64).to_be_bytes();
        bencher.bench_local(|| tree.get(black_box(&key)));
    }
}

// =============================================================================
// Arc vs Copy Comparison
// =============================================================================

#[divan::bench_group(name = "arc_vs_copy")]
mod arc_vs_copy {
    use super::{Bencher, MassTree, MassTreeIndex, black_box};

    #[divan::bench]
    fn arc_insert_100(bencher: Bencher) {
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

    #[divan::bench]
    fn copy_insert_100(bencher: Bencher) {
        bencher
            .with_inputs(MassTreeIndex::<u64>::new)
            .bench_local_values(|mut tree| {
                for i in 0..100u64 {
                    let key = i.to_be_bytes();
                    let _ = tree.insert(&key, i);
                }
                tree
            });
    }

    #[divan::bench]
    fn arc_get_100(bencher: Bencher) {
        let mut tree = MassTree::<u64>::new();
        for i in 0..100u64 {
            let key = i.to_be_bytes();
            let _ = tree.insert(&key, i);
        }

        bencher.bench_local(|| {
            let mut sum = 0u64;
            for i in 0..100u64 {
                let key = i.to_be_bytes();
                if let Some(v) = tree.get(&key) {
                    sum += *v;
                }
            }
            black_box(sum)
        });
    }

    #[divan::bench]
    fn copy_get_100(bencher: Bencher) {
        let mut tree = MassTreeIndex::<u64>::new();
        for i in 0..100u64 {
            let key = i.to_be_bytes();
            let _ = tree.insert(&key, i);
        }

        bencher.bench_local(|| {
            let mut sum = 0u64;
            for i in 0..100u64 {
                let key = i.to_be_bytes();
                if let Some(v) = tree.get(&key) {
                    sum += v;
                }
            }
            black_box(sum)
        });
    }
}
