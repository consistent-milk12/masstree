//! A narrow current-tree layout experiment; it does not benchmark the redesign.
use masstree::{MassTree15Inline, RangeBound};
use std::hint::black_box;
use std::time::Instant;

const KEYS: usize = 100_000;
const OPS: usize = 2_000_000;
const SCAN: usize = 32;

#[derive(Clone, Copy)]
struct Op {
    kind: u64,
    index: usize,
    value: u64,
}

fn next_random(state: &mut u64) -> u64 {
    *state ^= *state << 13;
    *state ^= *state >> 7;
    *state ^= *state << 17;
    *state
}

fn main() {
    let mut random = 42;
    let trace: Vec<_> = (0..OPS)
        .map(|step| Op {
            kind: next_random(&mut random) % 10,
            index: next_random(&mut random) as usize % (KEYS - SCAN),
            value: step as u64 + KEYS as u64,
        })
        .collect();
    let mut expected: Vec<u64> = (0..KEYS as u64).collect();
    let mut expected_sum = 0_u64;
    let mut scan_items = 0;
    let mut trace_hash = 0xcbf2_9ce4_8422_2325_u64;
    for op in &trace {
        for word in [op.kind, op.index as u64, op.value] {
            trace_hash = (trace_hash ^ word).wrapping_mul(0x100_0000_01b3);
        }
        match op.kind {
            0..=5 => expected_sum = expected_sum.wrapping_add(expected[op.index]),
            6..=7 => {
                expected_sum = expected_sum.wrapping_add(expected[op.index]);
                expected[op.index] = op.value;
            }
            _ => {
                for &value in &expected[op.index..op.index + SCAN] {
                    expected_sum = expected_sum.wrapping_add(value);
                    scan_items += 1;
                }
            }
        }
    }
    let mut insertion: Vec<usize> = (0..KEYS).collect();
    for i in (1..KEYS).rev() {
        let j = next_random(&mut random) as usize % (i + 1);
        insertion.swap(i, j);
    }
    let variant = if cfg!(feature = "sidecar-suffix") {
        "sidecar"
    } else if cfg!(feature = "small-suffix-capacity") {
        "embedded256"
    } else {
        "embedded512"
    };
    println!("variant,trial,keys,operations,scan_items,trace_hash,checksum,seconds,ops_per_second");
    for trial in 0..5 {
        let tree = MassTree15Inline::<u64>::new();
        let guard = tree.guard();
        for &i in &insertion {
            assert_eq!(
                tree.insert_with_guard(&(i as u64).to_be_bytes(), i as u64, &guard),
                None
            );
        }
        for i in 0..KEYS {
            assert_eq!(
                tree.get_with_guard(&(i as u64).to_be_bytes(), &guard),
                Some(i as u64)
            );
        }
        let mut sum = 0_u64;
        let mut observed_scan_items = 0;
        let start = Instant::now();
        for op in &trace {
            let key = (op.index as u64).to_be_bytes();
            match op.kind {
                0..=5 => sum = sum.wrapping_add(tree.get_with_guard(&key, &guard).unwrap()),
                6..=7 => {
                    sum = sum.wrapping_add(
                        tree.insert_value_with_guard(&key, op.value, &guard)
                            .unwrap(),
                    );
                }
                _ => {
                    let end = (op.index as u64 + SCAN as u64).to_be_bytes();
                    tree.scan(
                        RangeBound::Included(&key),
                        RangeBound::Excluded(&end),
                        |_, value| {
                            sum = sum.wrapping_add(value);
                            observed_scan_items += 1;
                            true
                        },
                        &guard,
                    );
                }
            }
        }
        let elapsed = start.elapsed().as_secs_f64();
        black_box(sum);
        assert_eq!(sum, expected_sum, "trace checksum mismatch");
        assert_eq!(observed_scan_items, scan_items, "scan membership mismatch");
        for (i, value) in expected.iter().enumerate() {
            assert_eq!(
                tree.get_with_guard(&(i as u64).to_be_bytes(), &guard),
                Some(*value)
            );
        }
        println!(
            "{variant},{trial},{KEYS},{OPS},{scan_items},{trace_hash:016x},{sum},{elapsed:.9},{:.3}",
            OPS as f64 / elapsed
        );
    }
}
