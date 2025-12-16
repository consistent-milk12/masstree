//! Benchmarks for `InternodeNode` using Divan.
//!
//! Run with: `cargo bench --bench internode`
#![expect(clippy::cast_possible_truncation)]

use divan::{Bencher, black_box};
use masstree::internode::InternodeNode;
use std::ptr;

fn main() {
    divan::main();
}

// =============================================================================
// Construction
// =============================================================================

#[divan::bench_group]
mod construction {
    use super::InternodeNode;

    #[divan::bench]
    fn new_height_0() -> Box<InternodeNode<u64, 15>> {
        InternodeNode::new(0)
    }

    #[divan::bench]
    fn new_height_5() -> Box<InternodeNode<u64, 15>> {
        InternodeNode::new(5)
    }

    #[divan::bench]
    fn default_internode() -> Box<InternodeNode<u64, 15>> {
        Box::<InternodeNode<u64, 15>>::default()
    }
}

// =============================================================================
// Accessors (hot path for traversal)
// =============================================================================

#[divan::bench_group]
mod accessors {
    use super::{Bencher, InternodeNode, black_box};

    fn setup_internode_with_keys(n: usize) -> Box<InternodeNode<u64, 15>> {
        let mut node = InternodeNode::<u64, 15>::new(0);

        for i in 0..n {
            let ikey = ((i + 1) as u64) << 56;
            node.set_ikey(i, ikey);
            // Set dummy child pointers
            node.set_child(i, (i + 1) as *mut u8);
        }
        node.set_child(n, (n + 1) as *mut u8); // Rightmost child
        node.set_nkeys(n as u8);
        node
    }

    #[divan::bench]
    fn nkeys(bencher: Bencher) {
        let node = setup_internode_with_keys(10);
        bencher.bench_local(|| black_box(&node).nkeys());
    }

    #[divan::bench]
    fn size(bencher: Bencher) {
        let node = setup_internode_with_keys(10);
        bencher.bench_local(|| black_box(&node).size());
    }

    #[divan::bench]
    fn is_full(bencher: Bencher) {
        let node = setup_internode_with_keys(10);
        bencher.bench_local(|| black_box(&node).is_full());
    }

    #[divan::bench]
    fn height(bencher: Bencher) {
        let node = setup_internode_with_keys(10);
        bencher.bench_local(|| black_box(&node).height());
    }

    #[divan::bench]
    fn children_are_leaves(bencher: Bencher) {
        let node = setup_internode_with_keys(10);
        bencher.bench_local(|| black_box(&node).children_are_leaves());
    }

    #[divan::bench(args = [0, 5, 10, 14])]
    fn ikey(bencher: Bencher, idx: usize) {
        let node = setup_internode_with_keys(15);
        bencher.bench_local(|| black_box(&node).ikey(black_box(idx)));
    }

    #[divan::bench(args = [0, 5, 10, 15])]
    fn child(bencher: Bencher, idx: usize) {
        let node = setup_internode_with_keys(15);
        bencher.bench_local(|| black_box(&node).child(black_box(idx)));
    }
}

// =============================================================================
// Key Comparison (used in binary search during traversal)
// =============================================================================

#[divan::bench_group]
mod compare {
    use super::{Bencher, InternodeNode, black_box};

    fn setup_internode() -> Box<InternodeNode<u64, 15>> {
        let mut node = InternodeNode::<u64, 15>::new(0);
        let keys: [u64; 5] = [
            0x1000_0000_0000_0000,
            0x2000_0000_0000_0000,
            0x3000_0000_0000_0000,
            0x4000_0000_0000_0000,
            0x5000_0000_0000_0000,
        ];

        for (i, &ikey) in keys.iter().enumerate() {
            node.set_ikey(i, ikey);
            node.set_child(i, (i + 1) as *mut u8);
        }
        node.set_child(5, 6 as *mut u8);
        node.set_nkeys(5);
        node
    }

    #[divan::bench]
    fn compare_key_equal(bencher: Bencher) {
        let node = setup_internode();
        let search_ikey = 0x3000_0000_0000_0000u64;
        bencher.bench_local(|| node.compare_key(black_box(search_ikey), 2));
    }

    #[divan::bench]
    fn compare_key_less(bencher: Bencher) {
        let node = setup_internode();
        let search_ikey = 0x1500_0000_0000_0000u64;
        bencher.bench_local(|| node.compare_key(black_box(search_ikey), 2));
    }

    #[divan::bench]
    fn compare_key_greater(bencher: Bencher) {
        let node = setup_internode();
        let search_ikey = 0x4500_0000_0000_0000u64;
        bencher.bench_local(|| node.compare_key(black_box(search_ikey), 2));
    }
}

// =============================================================================
// Modification Operations
// =============================================================================

#[divan::bench_group]
mod modify {
    use super::{Bencher, InternodeNode, black_box};

    #[divan::bench]
    fn set_ikey(bencher: Bencher) {
        bencher
            .with_inputs(|| InternodeNode::<u64, 15>::new(0))
            .bench_local_values(|mut node| {
                node.set_ikey(7, black_box(0x1234_5678_9ABC_DEF0));
                node
            });
    }

    #[divan::bench]
    fn set_child(bencher: Bencher) {
        bencher
            .with_inputs(|| InternodeNode::<u64, 15>::new(0))
            .bench_local_values(|mut node| {
                node.set_child(7, black_box(0xDEAD_BEEF as *mut u8));
                node
            });
    }

    #[divan::bench]
    fn assign(bencher: Bencher) {
        bencher
            .with_inputs(|| {
                let mut node = InternodeNode::<u64, 15>::new(0);
                node.set_nkeys(5);
                node
            })
            .bench_local_values(|mut node| {
                node.assign(
                    3,
                    black_box(0x1234_5678_9ABC_DEF0),
                    black_box(0xDEAD as *mut u8),
                );
                node
            });
    }

    #[divan::bench]
    fn set_nkeys(bencher: Bencher) {
        bencher
            .with_inputs(|| InternodeNode::<u64, 15>::new(0))
            .bench_local_values(|mut node| {
                node.set_nkeys(black_box(10));
                node
            });
    }
}

// =============================================================================
// Insert Key and Child
// =============================================================================

#[divan::bench_group]
mod insert {
    use super::{Bencher, InternodeNode, black_box, ptr};

    fn setup_internode_with_space(n: usize) -> Box<InternodeNode<u64, 15>> {
        let mut node = InternodeNode::<u64, 15>::new(0);

        for i in 0..n {
            let ikey = ((i + 1) as u64) << 56;
            node.set_ikey(i, ikey);
            node.set_child(i, (i + 1) as *mut u8);
        }
        node.set_child(n, (n + 1) as *mut u8);
        node.set_nkeys(n as u8);
        node
    }

    #[divan::bench]
    fn insert_at_front(bencher: Bencher) {
        bencher
            .with_inputs(|| setup_internode_with_space(10))
            .bench_local_values(|mut node| {
                node.insert_key_and_child(
                    0,
                    black_box(0x0050_0000_0000_0000),
                    black_box(ptr::null_mut()),
                );
                node
            });
    }

    #[divan::bench]
    fn insert_at_middle(bencher: Bencher) {
        bencher
            .with_inputs(|| setup_internode_with_space(10))
            .bench_local_values(|mut node| {
                node.insert_key_and_child(
                    5,
                    black_box(0x0550_0000_0000_0000),
                    black_box(ptr::null_mut()),
                );
                node
            });
    }

    #[divan::bench]
    fn insert_at_back(bencher: Bencher) {
        bencher
            .with_inputs(|| setup_internode_with_space(10))
            .bench_local_values(|mut node| {
                node.insert_key_and_child(
                    10,
                    black_box(0xFF00_0000_0000_0000),
                    black_box(ptr::null_mut()),
                );
                node
            });
    }
}

// =============================================================================
// Split Operations
// =============================================================================

#[divan::bench_group]
mod split {
    use super::{Bencher, InternodeNode, black_box, ptr};

    fn setup_full_internode() -> Box<InternodeNode<u64, 15>> {
        let mut node = InternodeNode::<u64, 15>::new(0);

        for i in 0..15 {
            let ikey = ((i + 1) as u64) << 56;
            node.set_ikey(i, ikey);
            node.set_child(i, (i + 1) as *mut u8);
        }
        node.set_child(15, 16 as *mut u8);
        node.set_nkeys(15);
        node
    }

    #[divan::bench]
    fn split_insert_left(bencher: Bencher) {
        bencher
            .with_inputs(|| {
                let left = setup_full_internode();
                let right = InternodeNode::<u64, 15>::new(0);
                (left, right)
            })
            .bench_local_values(|(mut left, mut right)| {
                let (popup, went_left) = left.split_into(
                    &mut right,
                    3, // Insert goes to left
                    black_box(0x0350_0000_0000_0000),
                    black_box(ptr::null_mut()),
                );
                black_box((popup, went_left));
                (left, right)
            });
    }

    #[divan::bench]
    fn split_insert_middle(bencher: Bencher) {
        bencher
            .with_inputs(|| {
                let left = setup_full_internode();
                let right = InternodeNode::<u64, 15>::new(0);
                (left, right)
            })
            .bench_local_values(|(mut left, mut right)| {
                let (popup, went_left) = left.split_into(
                    &mut right,
                    8, // Insert becomes popup key
                    black_box(0x0850_0000_0000_0000),
                    black_box(ptr::null_mut()),
                );
                black_box((popup, went_left));
                (left, right)
            });
    }

    #[divan::bench]
    fn split_insert_right(bencher: Bencher) {
        bencher
            .with_inputs(|| {
                let left = setup_full_internode();
                let right = InternodeNode::<u64, 15>::new(0);
                (left, right)
            })
            .bench_local_values(|(mut left, mut right)| {
                let (popup, went_left) = left.split_into(
                    &mut right,
                    12, // Insert goes to right
                    black_box(0x0C50_0000_0000_0000),
                    black_box(ptr::null_mut()),
                );
                black_box((popup, went_left));
                (left, right)
            });
    }
}

// =============================================================================
// Navigation
// =============================================================================

#[divan::bench_group]
mod navigation {
    use super::{Bencher, InternodeNode, black_box};

    #[divan::bench]
    fn parent(bencher: Bencher) {
        let node = InternodeNode::<u64, 15>::new(0);
        bencher.bench_local(|| black_box(&node).parent());
    }

    #[divan::bench]
    fn is_root(bencher: Bencher) {
        let node = InternodeNode::<u64, 15>::new(0);
        bencher.bench_local(|| black_box(&node).is_root());
    }

    #[divan::bench]
    fn set_parent(bencher: Bencher) {
        bencher
            .with_inputs(|| InternodeNode::<u64, 15>::new(0))
            .bench_local_values(|mut node| {
                node.set_parent(black_box(0xDEAD_BEEF as *mut u8));
                node
            });
    }
}

// =============================================================================
// Shift Operations
// =============================================================================

#[divan::bench_group]
mod shift {
    use super::{Bencher, InternodeNode};

    fn setup_source_internode() -> Box<InternodeNode<u64, 15>> {
        let mut node = InternodeNode::<u64, 15>::new(0);
        for i in 0..10 {
            let ikey = ((i + 1) as u64) << 56;
            node.set_ikey(i, ikey);
            node.set_child(i, (i + 1) as *mut u8);
        }
        node.set_child(10, 11 as *mut u8);
        node.set_nkeys(10);
        node
    }

    #[divan::bench(args = [1, 3, 5, 7])]
    fn shift_from(bencher: Bencher, count: usize) {
        bencher
            .with_inputs(|| {
                let src = setup_source_internode();
                let dst = InternodeNode::<u64, 15>::new(0);
                (src, dst)
            })
            .bench_local_values(|(src, mut dst)| {
                dst.shift_from(0, &src, 0, count);
                (src, dst)
            });
    }
}
