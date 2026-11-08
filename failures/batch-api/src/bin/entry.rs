//! Compare immediate Entry mutations with the previous two-descent workflow.
use std::time::{Duration, Instant};

use masstree::MassTree15;

fn update(keys: &[Vec<u8>], entry: bool) -> Duration {
    let tree: MassTree15<u64> = keys.iter().map(|key| (key, 1)).collect();
    let guard = tree.guard();
    let start = Instant::now();
    for key in keys {
        if entry {
            let value = tree.entry_with_guard(key, &guard).and_modify(|v| v + 1).or_insert(0);
            std::hint::black_box(value);
        } else {
            let value = tree.get_with_guard(key, &guard).map_or(0, |v| *v + 1);
            std::hint::black_box(tree.insert_value_with_guard(key, value, &guard));
        }
    }
    let elapsed = start.elapsed();
    assert_eq!(tree.len(), keys.len());
    for key in keys {
        assert_eq!(tree.get(key), Some(2));
    }
    elapsed
}

fn main() {
    for prefix in [b"".as_slice(), b"01234567"] {
        let mut keys: Vec<_> = (0..20_000u64)
            .map(|i| [prefix, &i.to_be_bytes()].concat())
            .collect();
        let mut seed = 0x1234_5678_9abc_def0u64;
        for i in (1..keys.len()).rev() {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            keys.swap(i, seed as usize % (i + 1));
        }
        let mut reference = Vec::new();
        let mut entry = Vec::new();
        for round in 0..7 {
            if round % 2 == 0 {
                reference.push(update(&keys, false));
                entry.push(update(&keys, true));
            } else {
                entry.push(update(&keys, true));
                reference.push(update(&keys, false));
            }
        }
        reference.sort();
        entry.sort();
        println!("prefix_len={}: two_descents={:?} entry={:?}", prefix.len(), reference[3], entry[3]);
    }
}
