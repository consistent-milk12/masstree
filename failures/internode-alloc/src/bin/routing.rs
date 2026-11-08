//! PERF-12/13: layout arithmetic and isolated routing measurements.
//! Includes the production scalar source; this is not an end-to-end tree benchmark.
use std::hint::black_box;
use std::time::{Duration, Instant};

pub use masstree::{LeafNode15, TreeInternode};
mod leaf15 {
    pub const WIDTH_15: usize = 15;
}
mod policy {
    pub use masstree::LeafPolicy;
}
#[path = "../../../../src/ksearch/scalar.rs"]
mod scalar;

use masstree::InternodeNode;
use scalar::Scalar;

#[inline]
fn count(node: &InternodeNode, target: u64) -> usize {
    (0..node.nkeys())
        .map(|i| usize::from(node.ikey_relaxed(i) <= target))
        .sum()
}

#[inline]
fn binary(node: &InternodeNode, target: u64) -> usize {
    let mut lo = 0;
    let mut hi = node.nkeys();
    while lo < hi {
        let mid = lo + (hi - lo) / 2;
        if node.ikey_relaxed(mid) <= target {
            lo = mid + 1;
        } else {
            hi = mid;
        }
    }
    lo
}

fn node(size: usize, base: u64) -> Box<InternodeNode> {
    let mut storage = Box::<InternodeNode>::new_uninit();
    // SAFETY: The aligned box owns the entire uninitialized node. No pointers
    // escape; routing below reads only the initialized key array and count.
    let node = unsafe {
        InternodeNode::init_at(storage.as_mut_ptr(), 0);
        storage.assume_init()
    };
    for i in 0..size {
        node.set_ikey(i, base + (i as u64 + 1) * 4);
    }
    node.set_nkeys(size as u8);
    node
}

fn measure<const METHOD: u8>(nodes: &[Box<InternodeNode>], queries: &[(usize, u64)]) -> Duration {
    let start = Instant::now();
    let mut total = 0;
    for _ in 0..32 {
        for &(index, target) in queries {
            let node = black_box(&*nodes[index]);
            let target = black_box(target);
            total += match METHOD {
                0 => Scalar::upper_bound_internode_scalar(target, node),
                1 => count(node, target),
                _ => binary(node, target),
            };
        }
    }
    black_box(total);
    start.elapsed()
}

fn main() {
    assert_eq!(size_of::<InternodeNode>(), 320);
    let width14_raw: usize = 16 + 14 * 8 + 15 * 8;
    println!(
        "WIDTH=15: {} B; hypothetical WIDTH=14: {} B (layout arithmetic only)",
        size_of::<InternodeNode>(),
        width14_raw.next_multiple_of(64)
    );
    // Exercise the other source-included scalar helper outside the timings.
    let leaf = LeafNode15::<masstree::BoxPolicy<u64>>::new();
    assert_eq!(Scalar::find_ikey_matches_leaf15(1, &leaf), 0);

    for size in 0..=15 {
        let node = node(size, 0);
        for target in 0..=64 {
            let expected = (1..=size).filter(|&i| i as u64 * 4 <= target).count();
            assert_eq!(
                Scalar::upper_bound_internode_scalar(target, &*node),
                expected
            );
            assert_eq!(count(&node, target), expected);
            assert_eq!(binary(&node, target), expected);
        }
    }

    for size in [3, 7, 15] {
        let nodes: Vec<_> = (0..256).map(|i| node(size, i * 64)).collect();
        for pattern in ["uniform", "equal", "left", "right"] {
            let mut state = 0x1234_5678_9abc_def0u64;
            let queries: Vec<_> = (0..32768)
                .map(|_| {
                    state ^= state << 13;
                    state ^= state >> 7;
                    state ^= state << 17;
                    let index = (state & 255) as usize;
                    let offset = match pattern {
                        "uniform" => (state >> 8) % ((size as u64 + 1) * 4),
                        "equal" => ((state >> 8) % size as u64 + 1) * 4,
                        "left" => 0,
                        _ => 64,
                    };
                    (index, index as u64 * 64 + offset)
                })
                .collect();
            let mut samples = [Vec::new(), Vec::new(), Vec::new()];
            for round in 0..7 {
                // Rotate order to avoid always timing one method first.
                for j in 0..3 {
                    let method = (round + j) % 3;
                    let elapsed = match method {
                        0 => measure::<0>(&nodes, &queries),
                        1 => measure::<1>(&nodes, &queries),
                        _ => measure::<2>(&nodes, &queries),
                    };
                    samples[method].push(elapsed);
                }
            }
            for sample in &mut samples {
                sample.sort();
            }
            println!(
                "n={size:2} {pattern:7}: scalar={:?}, count={:?}, binary={:?}",
                samples[0][3], samples[1][3], samples[2][3]
            );
        }
    }
}
