//! Benchmarks for key search operations using Divan.
//!
//! Run with: `cargo bench --bench ksearch`
//! With mimalloc: `cargo bench --bench ksearch --features mimalloc`

#![expect(clippy::indexing_slicing, reason = "fail fast in tests")]
#![expect(clippy::cast_possible_truncation, reason = "reasonable for benches")]
#![expect(clippy::cast_sign_loss, reason = "reasonable for benches")]

// Use alternative allocator if feature is enabled
#[cfg(feature = "mimalloc")]
#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;

use divan::{Bencher, black_box};
use masstree::internode::InternodeNode;
use masstree::ksearch::{
    lower_bound_by, lower_bound_leaf, lower_bound_leaf_ikey, lower_bound_linear_by, upper_bound_by,
    upper_bound_internode, upper_bound_internode_direct, upper_bound_linear_by,
};
use masstree::leaf::{LeafNode, LeafValue};
use masstree::permuter::Permuter;

/// Type alias for the slot type used in benchmarks.
type Slot = LeafValue<u64>;

fn main() {
    divan::main();
}

// =============================================================================
// Setup Helpers
// =============================================================================

fn setup_leaf_with_keys(keys: &[u64]) -> Box<LeafNode<Slot, 15>> {
    let leaf = LeafNode::<Slot, 15>::new();
    let mut perm = leaf.permutation();

    for (i, &ikey) in keys.iter().enumerate() {
        let slot = perm.insert_from_back(i);
        leaf.assign_value(slot, ikey, 8, i as u64);
    }
    leaf.set_permutation(perm);
    leaf
}

fn setup_internode_with_keys(keys: &[u64]) -> Box<InternodeNode<Slot, 15>> {
    let node = InternodeNode::<Slot, 15>::new(0);

    for (i, &ikey) in keys.iter().enumerate() {
        node.set_ikey(i, ikey);
        node.set_child(i, (i + 1) as *mut u8);
    }
    node.set_child(keys.len(), (keys.len() + 1) as *mut u8);
    node.set_nkeys(keys.len() as u8);
    node
}

fn setup_sorted_permuter(n: usize) -> Permuter<15> {
    Permuter::<15>::make_sorted(n)
}

// =============================================================================
// Lower Bound (Binary Search)
// =============================================================================

#[divan::bench_group]
mod lower_bound_binary {
    use super::{Bencher, black_box, lower_bound_by, setup_sorted_permuter};

    #[divan::bench(args = [1, 5, 10, 15])]
    fn find_existing(bencher: Bencher, size: usize) {
        // Create sorted keys
        let keys: Vec<u64> = (0..size).map(|i| ((i + 1) as u64) << 56).collect();
        let search_key = keys[size / 2]; // Search for middle element
        let perm = setup_sorted_permuter(size);

        bencher.bench_local(|| {
            let cmp = |slot: usize| search_key.cmp(&keys[slot]);
            lower_bound_by::<15, _>(black_box(size), perm, cmp)
        });
    }

    #[divan::bench(args = [1, 5, 10, 15])]
    fn find_missing(bencher: Bencher, size: usize) {
        let keys: Vec<u64> = (0..size).map(|i| ((i + 1) as u64) << 56).collect();
        // Search for key between two existing keys
        let search_key = 0x0150_0000_0000_0000u64;
        let perm = setup_sorted_permuter(size);

        bencher.bench_local(|| {
            let cmp = |slot: usize| search_key.cmp(&keys[slot]);
            lower_bound_by::<15, _>(black_box(size), perm, cmp)
        });
    }

    #[divan::bench(args = [1, 5, 10, 15])]
    fn find_first(bencher: Bencher, size: usize) {
        let keys: Vec<u64> = (0..size).map(|i| ((i + 1) as u64) << 56).collect();
        let search_key = keys[0]; // Search for first element
        let perm = setup_sorted_permuter(size);

        bencher.bench_local(|| {
            let cmp = |slot: usize| search_key.cmp(&keys[slot]);
            lower_bound_by::<15, _>(black_box(size), perm, cmp)
        });
    }

    #[divan::bench(args = [1, 5, 10, 15])]
    fn find_last(bencher: Bencher, size: usize) {
        let keys: Vec<u64> = (0..size).map(|i| ((i + 1) as u64) << 56).collect();
        let search_key = keys[size - 1]; // Search for last element
        let perm = setup_sorted_permuter(size);

        bencher.bench_local(|| {
            let cmp = |slot: usize| search_key.cmp(&keys[slot]);
            lower_bound_by::<15, _>(black_box(size), perm, cmp)
        });
    }
}

// =============================================================================
// Lower Bound (Linear Search)
// =============================================================================

#[divan::bench_group]
mod lower_bound_linear {
    use super::{Bencher, black_box, lower_bound_linear_by, setup_sorted_permuter};

    #[divan::bench(args = [1, 5, 10, 15])]
    fn find_existing(bencher: Bencher, size: usize) {
        let keys: Vec<u64> = (0..size).map(|i| ((i + 1) as u64) << 56).collect();
        let search_key = keys[size / 2];
        let perm = setup_sorted_permuter(size);

        bencher.bench_local(|| {
            let cmp = |slot: usize| search_key.cmp(&keys[slot]);
            lower_bound_linear_by::<15, _>(black_box(size), perm, cmp)
        });
    }

    #[divan::bench(args = [1, 5, 10, 15])]
    fn find_missing(bencher: Bencher, size: usize) {
        let keys: Vec<u64> = (0..size).map(|i| ((i + 1) as u64) << 56).collect();
        let search_key = 0x0150_0000_0000_0000u64;
        let perm = setup_sorted_permuter(size);

        bencher.bench_local(|| {
            let cmp = |slot: usize| search_key.cmp(&keys[slot]);
            lower_bound_linear_by::<15, _>(black_box(size), perm, cmp)
        });
    }
}

// =============================================================================
// Upper Bound (Binary Search)
// =============================================================================

#[divan::bench_group]
mod upper_bound_binary {
    use super::{Bencher, black_box, setup_sorted_permuter, upper_bound_by};

    #[divan::bench(args = [1, 5, 10, 15])]
    fn route_to_child(bencher: Bencher, size: usize) {
        let keys: Vec<u64> = (0..size).map(|i| ((i + 1) as u64) << 56).collect();
        // Search for key that routes to middle child
        let search_key = 0x0350_0000_0000_0000u64;
        let perm = setup_sorted_permuter(size);

        bencher.bench_local(|| {
            let cmp = |slot: usize| search_key.cmp(&keys[slot]);
            upper_bound_by::<15, _>(black_box(size), perm, cmp)
        });
    }

    #[divan::bench(args = [1, 5, 10, 15])]
    fn route_to_first_child(bencher: Bencher, size: usize) {
        let keys: Vec<u64> = (0..size).map(|i| ((i + 1) as u64) << 56).collect();
        let search_key = 0x0050_0000_0000_0000u64; // Less than all keys
        let perm = setup_sorted_permuter(size);

        bencher.bench_local(|| {
            let cmp = |slot: usize| search_key.cmp(&keys[slot]);
            upper_bound_by::<15, _>(black_box(size), perm, cmp)
        });
    }

    #[divan::bench(args = [1, 5, 10, 15])]
    fn route_to_last_child(bencher: Bencher, size: usize) {
        let keys: Vec<u64> = (0..size).map(|i| ((i + 1) as u64) << 56).collect();
        let search_key = 0xFF00_0000_0000_0000u64; // Greater than all keys
        let perm = setup_sorted_permuter(size);

        bencher.bench_local(|| {
            let cmp = |slot: usize| search_key.cmp(&keys[slot]);
            upper_bound_by::<15, _>(black_box(size), perm, cmp)
        });
    }
}

// =============================================================================
// Upper Bound (Linear Search)
// =============================================================================

#[divan::bench_group]
mod upper_bound_linear {
    use super::{Bencher, black_box, setup_sorted_permuter, upper_bound_linear_by};

    #[divan::bench(args = [1, 5, 10, 15])]
    fn route_to_child(bencher: Bencher, size: usize) {
        let keys: Vec<u64> = (0..size).map(|i| ((i + 1) as u64) << 56).collect();
        let search_key = 0x0350_0000_0000_0000u64;
        let perm = setup_sorted_permuter(size);

        bencher.bench_local(|| {
            let cmp = |slot: usize| search_key.cmp(&keys[slot]);
            upper_bound_linear_by::<15, _>(black_box(size), perm, cmp)
        });
    }
}

// =============================================================================
// Leaf Search (integrated with LeafNode)
// =============================================================================

#[divan::bench_group]
mod leaf_search {
    use super::{
        Bencher, black_box, lower_bound_leaf, lower_bound_leaf_ikey, setup_leaf_with_keys,
    };

    #[divan::bench(args = [1, 5, 10, 15])]
    fn lower_bound_existing(bencher: Bencher, size: usize) {
        let keys: Vec<u64> = (0..size).map(|i| ((i + 1) as u64) << 56).collect();
        let leaf = setup_leaf_with_keys(&keys);
        let search_ikey = keys[size / 2];

        bencher.bench_local(|| {
            lower_bound_leaf(black_box(search_ikey), black_box(8), black_box(&leaf))
        });
    }

    #[divan::bench(args = [1, 5, 10, 15])]
    fn lower_bound_missing(bencher: Bencher, size: usize) {
        let keys: Vec<u64> = (0..size).map(|i| ((i + 1) as u64) << 56).collect();
        let leaf = setup_leaf_with_keys(&keys);
        let search_ikey = 0x0150_0000_0000_0000u64;

        bencher.bench_local(|| {
            lower_bound_leaf(black_box(search_ikey), black_box(8), black_box(&leaf))
        });
    }

    #[divan::bench(args = [1, 5, 10, 15])]
    fn lower_bound_ikey_only(bencher: Bencher, size: usize) {
        let keys: Vec<u64> = (0..size).map(|i| ((i + 1) as u64) << 56).collect();
        let leaf = setup_leaf_with_keys(&keys);
        let search_ikey = keys[size / 2];

        bencher.bench_local(|| lower_bound_leaf_ikey(black_box(search_ikey), black_box(&leaf)));
    }
}

// =============================================================================
// Internode Search (integrated with InternodeNode)
// =============================================================================

#[divan::bench_group]
mod internode_search {
    use super::{
        Bencher, black_box, setup_internode_with_keys, upper_bound_internode,
        upper_bound_internode_direct,
    };

    #[divan::bench(args = [1, 5, 10, 15])]
    fn upper_bound_route(bencher: Bencher, size: usize) {
        let keys: Vec<u64> = (0..size).map(|i| ((i + 1) as u64) << 56).collect();
        let node = setup_internode_with_keys(&keys);
        let search_ikey = 0x0350_0000_0000_0000u64;

        bencher.bench_local(|| upper_bound_internode(black_box(search_ikey), black_box(&node)));
    }

    #[divan::bench(args = [1, 5, 10, 15])]
    fn upper_bound_direct(bencher: Bencher, size: usize) {
        let keys: Vec<u64> = (0..size).map(|i| ((i + 1) as u64) << 56).collect();
        let node = setup_internode_with_keys(&keys);
        let search_ikey = 0x0350_0000_0000_0000u64;

        bencher
            .bench_local(|| upper_bound_internode_direct(black_box(search_ikey), black_box(&node)));
    }
}

// =============================================================================
// Binary vs Linear Comparison
// =============================================================================

#[divan::bench_group(name = "binary_vs_linear")]
mod comparison {
    use super::{
        Bencher, black_box, lower_bound_by, lower_bound_linear_by, setup_sorted_permuter,
        upper_bound_by, upper_bound_linear_by,
    };

    const SIZE: usize = 15;

    fn setup_keys() -> Vec<u64> {
        (0..SIZE).map(|i| ((i + 1) as u64) << 56).collect()
    }

    #[divan::bench]
    fn binary_lower_bound(bencher: Bencher) {
        let keys = setup_keys();
        let search_key = keys[SIZE / 2];
        let perm = setup_sorted_permuter(SIZE);

        bencher.bench_local(|| {
            let cmp = |slot: usize| search_key.cmp(&keys[slot]);
            lower_bound_by::<15, _>(black_box(SIZE), perm, cmp)
        });
    }

    #[divan::bench]
    fn linear_lower_bound(bencher: Bencher) {
        let keys = setup_keys();
        let search_key = keys[SIZE / 2];
        let perm = setup_sorted_permuter(SIZE);

        bencher.bench_local(|| {
            let cmp = |slot: usize| search_key.cmp(&keys[slot]);
            lower_bound_linear_by::<15, _>(black_box(SIZE), perm, cmp)
        });
    }

    #[divan::bench]
    fn binary_upper_bound(bencher: Bencher) {
        let keys = setup_keys();
        let search_key = 0x0550_0000_0000_0000u64;
        let perm = setup_sorted_permuter(SIZE);

        bencher.bench_local(|| {
            let cmp = |slot: usize| search_key.cmp(&keys[slot]);
            upper_bound_by::<15, _>(black_box(SIZE), perm, cmp)
        });
    }

    #[divan::bench]
    fn linear_upper_bound(bencher: Bencher) {
        let keys = setup_keys();
        let search_key = 0x0550_0000_0000_0000u64;
        let perm = setup_sorted_permuter(SIZE);

        bencher.bench_local(|| {
            let cmp = |slot: usize| search_key.cmp(&keys[slot]);
            upper_bound_linear_by::<15, _>(black_box(SIZE), perm, cmp)
        });
    }
}

// =============================================================================
// Realistic Workloads
// =============================================================================

#[divan::bench_group]
mod workload {
    use super::{
        Bencher, black_box, lower_bound_leaf, setup_internode_with_keys, setup_leaf_with_keys,
        upper_bound_internode_direct,
    };

    #[divan::bench]
    fn full_leaf_lookup_hit(bencher: Bencher) {
        let keys: Vec<u64> = (0..15).map(|i| ((i + 1) as u64) << 56).collect();
        let leaf = setup_leaf_with_keys(&keys);
        let search_ikey = 0x0800_0000_0000_0000u64; // Key exists

        bencher.bench_local(|| {
            let pos = lower_bound_leaf(black_box(search_ikey), 8, &leaf);
            black_box(pos.is_found());
            pos
        });
    }

    #[divan::bench]
    fn full_leaf_lookup_miss(bencher: Bencher) {
        let keys: Vec<u64> = (0..15).map(|i| ((i + 1) as u64) << 56).collect();
        let leaf = setup_leaf_with_keys(&keys);
        let search_ikey = 0x0850_0000_0000_0000u64; // Key doesn't exist

        bencher.bench_local(|| {
            let pos = lower_bound_leaf(black_box(search_ikey), 8, &leaf);
            black_box(pos.is_found());
            pos
        });
    }

    #[divan::bench]
    fn internode_traversal_step(bencher: Bencher) {
        let keys: Vec<u64> = (0..15).map(|i| ((i + 1) as u64) << 56).collect();
        let node = setup_internode_with_keys(&keys);
        let search_ikey = 0x0550_0000_0000_0000u64;

        bencher.bench_local(|| {
            let child_idx = upper_bound_internode_direct(black_box(search_ikey), &node);
            let child = node.child(child_idx);
            black_box(child);
            child_idx
        });
    }
}
