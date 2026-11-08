//! Promoted-layer regression across keyed, reference, values and mixed scans.
use std::collections::{BTreeMap, VecDeque};

use masstree::{MassTree, MassTree15, RangeBound};

fn main() {
    for count in [40, 300] {
        let model: BTreeMap<_, _> = (0..count)
            .map(|i: u64| ([b"PREFIX00".as_slice(), &i.to_be_bytes()].concat(), i))
            .collect();
        let tree: MassTree<u64> = model.iter().map(|(k, &v)| (k, v)).collect();
        let boxed: MassTree15<u64> = model.iter().map(|(k, &v)| (k, v)).collect();
        let guard = tree.guard();
        let boxed_guard = boxed.guard();
        let keys: Vec<_> = model.keys().cloned().collect();
        assert_eq!(tree.len(), model.len());
        assert_eq!(tree.iter(&guard).count(), model.len());

        for idx in [0, 5, 15, 16, 20, 30, 39] {
            let start = keys[idx].as_slice();
            let lower = RangeBound::Included(start);
            let upper = RangeBound::Unbounded;
            let expected: Vec<_> = model
                .range(keys[idx].clone()..)
                .map(|(k, &v)| (k.clone(), v))
                .collect();
            let actual: Vec<_> = tree
                .range(lower, upper, &guard)
                .map(|e| (e.key, e.value))
                .collect();
            assert_eq!(actual, expected, "range at {idx}/{count}");
            let mut actual: Vec<_> = tree
                .range(lower, upper, &guard)
                .rev()
                .map(|e| (e.key, e.value))
                .collect();
            actual.reverse();
            assert_eq!(actual, expected, "reverse at {idx}/{count}");
            let actual: Vec<_> = tree
                .range(RangeBound::Excluded(start), upper, &guard)
                .map(|e| (e.key, e.value))
                .collect();
            assert_eq!(actual, expected[1..], "excluded at {idx}/{count}");

            for batch in [false, true] {
                let mut actual = Vec::new();
                let visit = |key: &[u8], value| {
                    actual.push((key.to_vec(), value));
                    true
                };
                if batch {
                    tree.scan_intra_leaf_batch(lower, upper, visit, &guard);
                } else {
                    tree.scan(lower, upper, visit, &guard);
                }
                assert_eq!(actual, expected, "keyed callback at {idx}/{count}");
            }
            macro_rules! check_ref_scan {
                ($method:ident) => {{
                    let mut actual = Vec::new();
                    boxed.$method(
                        lower,
                        upper,
                        |key, value| {
                            actual.push((key.to_vec(), *value));
                            true
                        },
                        &boxed_guard,
                    );
                    assert_eq!(actual, expected, "{} at {idx}/{count}", stringify!($method));
                }};
            }
            check_ref_scan!(scan_ref);
            check_ref_scan!(scan_batch_ref);
            check_ref_scan!(scan_intra_leaf_batch_ref);
            let mut values = Vec::new();
            tree.scan_values(
                lower,
                upper,
                |v| {
                    values.push(v);
                    true
                },
                &guard,
            );
            assert_eq!(values, expected.iter().map(|(_, v)| *v).collect::<Vec<_>>());
            let mut prefix = Vec::new();
            tree.scan_prefix(
                start,
                |key, value| {
                    prefix.push((key.to_vec(), value));
                    true
                },
                &guard,
            );
            assert_eq!(prefix, expected[..1]);

            let mut mixed = tree.range(lower, upper, &guard);
            let mut pending: VecDeque<_> = expected.into();
            while let Some(first) = pending.pop_front() {
                let actual = mixed.next().unwrap();
                assert_eq!((actual.key, actual.value), first);
                if let Some(last) = pending.pop_back() {
                    let actual = mixed.next_back().unwrap();
                    assert_eq!((actual.key, actual.value), last);
                }
            }
            assert!(mixed.next().is_none());
            assert!(mixed.next_back().is_none());
        }
        println!("promoted scans passed for {count} keys");
    }
}
