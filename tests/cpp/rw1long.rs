//! RW1LONG: Random put/get with long variable-length keys.
//!
//! Port of `kvtest_rw1long` from C++ `kvtest.hh`.
//!
//! Pattern:
//! - Keys like "user123", "machine456", "opening789", "fartparade000"
//! - Tests multi-layer trie behavior

#![allow(clippy::indexing_slicing, clippy::unwrap_used)]

use masstree::MassTree15Inline;
use rand::{Rng, SeedableRng, rngs::StdRng, seq::SliceRandom};
use std::sync::Arc;
use std::thread;

const SEED: u64 = 31949;
const N: usize = 50_000;
const FORMATS: [&str; 4] = ["user", "machine", "opening", "fartparade"];

fn make_key(format_idx: usize, value: u32) -> String {
    format!("{}{}", FORMATS[format_idx % 4], value)
}

#[test]
fn rw1long_single_thread() {
    let tree: MassTree15Inline<u64> = MassTree15Inline::new();
    let guard = tree.guard();
    let mut rng = StdRng::seed_from_u64(SEED);

    // Put phase
    let mut keys: Vec<(usize, u32)> = Vec::with_capacity(N);

    for _ in 0..N {
        let x: u32 = rng.random();
        let fmt: usize = rng.random_range(0..4);
        let key = make_key(fmt, x);

        tree.insert_with_guard(key.as_bytes(), u64::from(x) + 1, &guard)
            .unwrap();
        keys.push((fmt, x));
    }

    // Shuffle
    keys.shuffle(&mut rng);

    // Get phase
    for (fmt, x) in &keys {
        let key = make_key(*fmt, *x);
        let val = tree.get_with_guard(key.as_bytes(), &guard);
        assert_eq!(val, Some(u64::from(*x) + 1), "key {key} mismatch");
    }
}

#[test]
fn rw1long_concurrent() {
    let tree = Arc::new(MassTree15Inline::<u64>::new());
    let num_threads = 4;
    let per_thread = N / num_threads;

    let handles: Vec<_> = (0..num_threads)
        .map(|tid| {
            let tree = Arc::clone(&tree);
            thread::spawn(move || {
                let guard = tree.guard();
                let mut rng = StdRng::seed_from_u64(SEED + tid as u64);

                let mut keys: Vec<(usize, u32)> = Vec::with_capacity(per_thread);
                for _ in 0..per_thread {
                    let x: u32 = rng.random();
                    let fmt: usize = rng.random_range(0..4);
                    let key = make_key(fmt, x);
                    let _ = tree.insert_with_guard(key.as_bytes(), u64::from(x) + 1, &guard);
                    keys.push((fmt, x));
                }

                keys.shuffle(&mut rng);

                for (fmt, x) in &keys {
                    let key = make_key(*fmt, *x);
                    let val = tree.get_with_guard(key.as_bytes(), &guard);
                    assert_eq!(val, Some(u64::from(*x) + 1));
                }
            })
        })
        .collect();

    for h in handles {
        h.join().unwrap();
    }
}
