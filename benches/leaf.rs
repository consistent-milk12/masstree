//! Benchmarks for `LeafNode` using Divan.
//!
//! Run with: `cargo bench --bench leaf`
//! With mimalloc: `cargo bench --bench leaf --features mimalloc`

// Use alternative allocator if feature is enabled
#[cfg(feature = "mimalloc")]
#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;

use divan::{Bencher, black_box};
use masstree::leaf::{LeafNode, LeafValue};
use masstree::permuter::Permuter;
use seize::Collector;

/// Type alias for the slot type used in benchmarks.
type Slot = LeafValue<u64>;

fn main() {
    divan::main();
}

// =============================================================================
// Construction
// =============================================================================

#[divan::bench_group]
mod construction {
    use super::{LeafNode, Slot};

    #[divan::bench]
    fn new_leaf() -> Box<LeafNode<Slot, 15>> {
        LeafNode::new()
    }

    #[divan::bench]
    fn default_leaf() -> Box<LeafNode<Slot, 15>> {
        Box::<LeafNode<Slot, 15>>::default()
    }
}

// =============================================================================
// Accessors (hot path for lookups)
// =============================================================================

#[divan::bench_group]
mod accessors {
    use super::{Bencher, LeafNode, Slot, black_box};

    fn setup_leaf_with_entries(n: usize) -> Box<LeafNode<Slot, 15>> {
        let leaf = LeafNode::<Slot, 15>::new();
        let mut perm = leaf.permutation();

        for i in 0..n {
            let slot = perm.insert_from_back(i);
            let ikey = (i as u64) << 56; // Spread keys apart
            leaf.assign_arc(slot, ikey, 8, std::sync::Arc::new(i as u64));
        }
        leaf.set_permutation(perm);
        leaf
    }

    #[divan::bench]
    fn size(bencher: Bencher) {
        let leaf = setup_leaf_with_entries(10);
        bencher.bench_local(|| black_box(&leaf).size());
    }

    #[divan::bench]
    fn is_empty(bencher: Bencher) {
        let leaf = setup_leaf_with_entries(10);
        bencher.bench_local(|| black_box(&leaf).is_empty());
    }

    #[divan::bench]
    fn permutation(bencher: Bencher) {
        let leaf = setup_leaf_with_entries(10);
        bencher.bench_local(|| black_box(&leaf).permutation());
    }

    #[divan::bench(args = [0, 5, 10, 14])]
    fn ikey(bencher: Bencher, slot: usize) {
        let leaf = setup_leaf_with_entries(15);
        bencher.bench_local(|| black_box(&leaf).ikey(black_box(slot)));
    }

    #[divan::bench(args = [0, 5, 10, 14])]
    fn keylenx(bencher: Bencher, slot: usize) {
        let leaf = setup_leaf_with_entries(15);
        bencher.bench_local(|| black_box(&leaf).keylenx(black_box(slot)));
    }

    #[divan::bench(args = [0, 5, 10, 14])]
    fn leaf_value_ptr(bencher: Bencher, slot: usize) {
        let leaf = setup_leaf_with_entries(15);
        bencher.bench_local(|| black_box(black_box(&leaf).leaf_value_ptr(black_box(slot))));
    }

    #[divan::bench(args = [0, 5, 10, 14])]
    fn try_clone_arc(bencher: Bencher, slot: usize) {
        let leaf = setup_leaf_with_entries(15);
        bencher.bench_local(|| {
            // SAFETY: benchmark context, no concurrent access
            black_box(unsafe { black_box(&leaf).try_clone_arc(black_box(slot)) })
        });
    }
}

// =============================================================================
// Value Operations
// =============================================================================

#[divan::bench_group]
mod value_ops {
    use super::{Bencher, Collector, LeafNode, Slot, black_box};
    use std::sync::Arc;

    #[divan::bench]
    fn assign_arc(bencher: Bencher) {
        bencher
            .with_inputs(|| {
                let leaf = LeafNode::<Slot, 15>::new();
                let mut perm = leaf.permutation();
                let slot = perm.insert_from_back(0);
                leaf.set_permutation(perm);
                (leaf, slot)
            })
            .bench_local_values(|(leaf, slot)| {
                leaf.assign_arc(slot, black_box(0x1234_5678_9ABC_DEF0), 8, Arc::new(42u64));
                leaf
            });
    }

    #[divan::bench]
    fn swap_value(bencher: Bencher) {
        let collector = Collector::new();
        bencher
            .with_inputs(|| {
                let leaf = LeafNode::<Slot, 15>::new();
                let mut perm = leaf.permutation();
                let slot = perm.insert_from_back(0);
                leaf.assign_arc(slot, 0x1234_5678_9ABC_DEF0, 8, Arc::new(1u64));
                leaf.set_permutation(perm);
                (leaf, slot)
            })
            .bench_local_values(|(leaf, slot)| {
                let guard = collector.enter();
                // SAFETY: benchmark context, no concurrent access
                let old = unsafe { leaf.swap_value(slot, Arc::new(2u64), &guard) };
                black_box(old);
                leaf
            });
    }
}

// =============================================================================
// Permutation Operations
// =============================================================================

#[divan::bench_group]
mod permutation {
    use super::{Bencher, LeafNode, Permuter, Slot};

    #[divan::bench]
    fn set_permutation(bencher: Bencher) {
        bencher
            .with_inputs(|| {
                let leaf = LeafNode::<Slot, 15>::new();
                let mut perm = Permuter::<15>::empty();
                for i in 0..10 {
                    let _ = perm.insert_from_back(i);
                }
                (leaf, perm)
            })
            .bench_local_values(|(leaf, perm): (Box<LeafNode<Slot, 15>>, Permuter<15>)| {
                leaf.set_permutation(perm);
                leaf
            });
    }

    #[divan::bench]
    fn permutation_load(bencher: Bencher) {
        let leaf = LeafNode::<Slot, 15>::new();
        let mut perm = Permuter::<15>::empty();
        for i in 0..10 {
            let _ = perm.insert_from_back(i);
        }
        leaf.set_permutation(perm);

        bencher.bench_local(|| leaf.permutation());
    }
}

// =============================================================================
// Navigation (B-link tree traversal)
// =============================================================================

#[divan::bench_group]
mod navigation {
    use super::{Bencher, LeafNode, Slot, black_box};

    #[divan::bench]
    fn next_raw(bencher: Bencher) {
        let leaf = LeafNode::<Slot, 15>::new();
        bencher.bench_local(|| black_box(&leaf).next_raw());
    }

    #[divan::bench]
    fn prev(bencher: Bencher) {
        let leaf = LeafNode::<Slot, 15>::new();
        bencher.bench_local(|| black_box(&leaf).prev());
    }

    #[divan::bench]
    fn parent(bencher: Bencher) {
        let leaf = LeafNode::<Slot, 15>::new();
        bencher.bench_local(|| black_box(&leaf).parent());
    }
}

// =============================================================================
// Split Operations
// =============================================================================

#[divan::bench_group]
mod split {
    use super::{Bencher, Collector, LeafNode, Slot, black_box};
    use masstree::leaf::SplitUtils;
    use std::sync::Arc;

    fn setup_full_leaf() -> Box<LeafNode<Slot, 15>> {
        let leaf = LeafNode::<Slot, 15>::new();
        let mut perm = leaf.permutation();

        for i in 0..15 {
            let slot = perm.insert_from_back(i);
            let ikey = (i as u64) << 56;
            leaf.assign_arc(slot, ikey, 8, Arc::new(i as u64));
        }
        leaf.set_permutation(perm);
        leaf
    }

    #[divan::bench]
    fn calculate_split_point_middle(bencher: Bencher) {
        let leaf = setup_full_leaf();
        let insert_pos = 7;
        let insert_ikey = 0x0700_0000_0000_0000u64;
        bencher.bench_local(|| {
            SplitUtils::calculate_split_point(
                black_box(&leaf),
                black_box(insert_pos),
                black_box(insert_ikey),
            )
        });
    }

    #[divan::bench]
    fn calculate_split_point_sequential(bencher: Bencher) {
        let leaf = setup_full_leaf();
        // Simulate sequential insert at the end
        let insert_pos = 15;
        let insert_ikey = 0xFF00_0000_0000_0000u64;
        bencher.bench_local(|| {
            SplitUtils::calculate_split_point(
                black_box(&leaf),
                black_box(insert_pos),
                black_box(insert_ikey),
            )
        });
    }

    #[divan::bench]
    fn split_into(bencher: Bencher) {
        let collector = Collector::new();
        bencher
            .with_inputs(|| {
                let leaf = setup_full_leaf();
                (leaf, 8) // Split at position 8
            })
            .bench_local_values(|(leaf, split_pos)| {
                let guard = collector.enter();
                // SAFETY: benchmark context, no concurrent access
                let result = unsafe { leaf.split_into(split_pos, &guard) };
                black_box(result);
                leaf
            });
    }

    #[divan::bench]
    fn split_all_to_right(bencher: Bencher) {
        let collector = Collector::new();
        bencher
            .with_inputs(setup_full_leaf)
            .bench_local_values(|leaf| {
                let guard = collector.enter();
                // SAFETY: benchmark context, no concurrent access
                let result = unsafe { leaf.split_all_to_right(&guard) };
                black_box(result);
                leaf
            });
    }
}

// =============================================================================
// Slot-0 Rule
// =============================================================================

#[divan::bench_group]
mod slot0_rule {
    use super::{Bencher, LeafNode, Slot, black_box};
    use std::sync::Arc;

    #[divan::bench]
    fn can_reuse_slot0_no_prev(bencher: Bencher) {
        let leaf = LeafNode::<Slot, 15>::new();
        let ikey = 0x1234_5678_9ABC_DEF0u64;
        bencher.bench_local(|| black_box(&leaf).can_reuse_slot0(black_box(ikey)));
    }

    #[divan::bench]
    fn can_reuse_slot0_with_prev(bencher: Bencher) {
        let leaf = LeafNode::<Slot, 15>::new();
        // Simulate having a predecessor (non-null prev)
        leaf.set_prev(std::ptr::NonNull::dangling().as_ptr());

        // Set slot 0 with an ikey
        let mut perm = leaf.permutation();
        let slot = perm.insert_from_back(0);
        leaf.assign_arc(slot, 0x1234_5678_0000_0000, 8, Arc::new(0u64));
        leaf.set_permutation(perm);

        // Test with same ikey prefix
        let same_ikey = 0x1234_5678_0000_0000u64;
        bencher.bench_local(|| black_box(&leaf).can_reuse_slot0(black_box(same_ikey)));
    }

    #[divan::bench]
    fn can_insert_directly_empty(bencher: Bencher) {
        let leaf = LeafNode::<Slot, 15>::new();
        let ikey = 0x1234_5678_9ABC_DEF0u64;
        bencher.bench_local(|| black_box(&leaf).can_insert_directly(black_box(ikey)));
    }

    #[divan::bench]
    fn can_insert_directly_half_full(bencher: Bencher) {
        let leaf = LeafNode::<Slot, 15>::new();
        let mut perm = leaf.permutation();

        for i in 0..7 {
            let slot = perm.insert_from_back(i);
            let ikey = (i as u64) << 56;
            leaf.assign_arc(slot, ikey, 8, Arc::new(i as u64));
        }
        leaf.set_permutation(perm);

        let ikey = 0x0800_0000_0000_0000u64;
        bencher.bench_local(|| black_box(&leaf).can_insert_directly(black_box(ikey)));
    }
}

// =============================================================================
// Suffix Operations (COW SuffixBag)
// =============================================================================

#[divan::bench_group]
mod suffix_ops {
    use super::{Bencher, Collector, LeafNode, Slot, black_box};
    use std::sync::Arc;

    fn setup_leaf_with_suffix() -> Box<LeafNode<Slot, 15>> {
        let collector = Collector::new();
        let guard = collector.enter();

        let leaf = LeafNode::<Slot, 15>::new();
        let mut perm = leaf.permutation();
        let slot = perm.insert_from_back(0);
        leaf.assign_arc(slot, 0x1234_5678_9ABC_DEF0, 8, Arc::new(42u64));
        // SAFETY: benchmark context
        unsafe { leaf.assign_ksuf(slot, b"suffix_data", &guard) };
        leaf.set_permutation(perm);
        leaf
    }

    #[divan::bench]
    fn has_ksuf(bencher: Bencher) {
        let leaf = setup_leaf_with_suffix();
        bencher.bench_local(|| black_box(&leaf).has_ksuf(0));
    }

    #[divan::bench]
    fn ksuf(bencher: Bencher) {
        let leaf = setup_leaf_with_suffix();
        bencher.bench_local(|| black_box(&leaf).ksuf(0));
    }

    #[divan::bench]
    fn ksuf_equals_match(bencher: Bencher) {
        let leaf = setup_leaf_with_suffix();
        bencher.bench_local(|| black_box(&leaf).ksuf_equals(0, b"suffix_data"));
    }

    #[divan::bench]
    fn ksuf_equals_mismatch(bencher: Bencher) {
        let leaf = setup_leaf_with_suffix();
        bencher.bench_local(|| black_box(&leaf).ksuf_equals(0, b"different"));
    }

    #[divan::bench]
    fn assign_ksuf(bencher: Bencher) {
        let collector = Collector::new();
        bencher
            .with_inputs(|| {
                let leaf = LeafNode::<Slot, 15>::new();
                let mut perm = leaf.permutation();
                let slot = perm.insert_from_back(0);
                leaf.assign_arc(slot, 0x1234_5678_9ABC_DEF0, 8, Arc::new(42u64));
                leaf.set_permutation(perm);
                leaf
            })
            .bench_local_values(|leaf| {
                let guard = collector.enter();
                // SAFETY: benchmark context
                unsafe { leaf.assign_ksuf(0, b"new_suffix_data", &guard) };
                leaf
            });
    }
}

// =============================================================================
// Version Operations
// =============================================================================

#[divan::bench_group]
mod version {
    use super::{Bencher, LeafNode, Slot, black_box};

    #[divan::bench]
    fn version_value(bencher: Bencher) {
        let leaf = LeafNode::<Slot, 15>::new();
        bencher.bench_local(|| black_box(&leaf).version().value());
    }

    #[divan::bench]
    fn is_root(bencher: Bencher) {
        let leaf = LeafNode::<Slot, 15>::new_root();
        bencher.bench_local(|| black_box(&leaf).version().is_root());
    }
}
