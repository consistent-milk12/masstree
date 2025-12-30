//! YCSBK: YCSB-style keys (user + 18 random digits).
//!
//! Port of `kvtest_ycsbk` from C++ `kvtest.hh`.
//!
//! Pattern:
//! - Keys like "user000000000000000000" (4 + 18 = 22 bytes)
//! - Tests multi-layer trie with fixed-length keys

use masstree::MassTree24Inline;
use rand::{Rng, SeedableRng, rngs::StdRng, seq::SliceRandom};
use std::sync::Arc;
use std::thread;

const SEED: u64 = 31949;
const N: usize = 50_000;

fn make_ycsb_key(rng: &mut StdRng) -> String {
    let mut key = String::with_capacity(22);
    key.push_str("user");
    for _ in 0..18 {
        key.push((b'0' + rng.random_range(0..10u8)) as char);
    }
    key
}

#[test]
fn ycsbk_single_thread() {
    let tree: MassTree24Inline<u64> = MassTree24Inline::new();
    let guard = tree.guard();
    let mut rng = StdRng::seed_from_u64(SEED);

    // Put phase
    let mut keys: Vec<String> = Vec::with_capacity(N);
    for i in 0..N {
        let key = make_ycsb_key(&mut rng);
        let val: u32 = rng.random();
        tree.insert_with_guard(key.as_bytes(), val as u64, &guard)
            .unwrap();
        keys.push(key);
    }

    // Get phase with same seed to regenerate values
    let mut rng = StdRng::seed_from_u64(SEED);
    for key in &keys {
        let _regenerated_key = make_ycsb_key(&mut rng); // consume to keep rng in sync
        let expected_val: u32 = rng.random();
        let val = tree.get_with_guard(key.as_bytes(), &guard);
        assert_eq!(val, Some(expected_val as u64), "key {} mismatch", key);
    }
}

#[test]
fn ycsbk_concurrent() {
    let tree = Arc::new(MassTree24Inline::<u64>::new());
    let num_threads = 4;
    let per_thread = N / num_threads;

    let handles: Vec<_> = (0..num_threads)
        .map(|tid| {
            let tree = Arc::clone(&tree);
            thread::spawn(move || {
                let guard = tree.guard();
                let mut rng = StdRng::seed_from_u64(SEED + tid as u64);

                let mut keys: Vec<String> = Vec::with_capacity(per_thread);
                for _ in 0..per_thread {
                    let key = make_ycsb_key(&mut rng);
                    let val: u32 = rng.random();
                    let _ = tree.insert_with_guard(key.as_bytes(), val as u64, &guard);
                    keys.push(key);
                }

                // Shuffle and verify presence
                keys.shuffle(&mut rng);
                for key in &keys {
                    let val = tree.get_with_guard(key.as_bytes(), &guard);
                    assert!(val.is_some(), "key {} not found", key);
                }
            })
        })
        .collect();

    for h in handles {
        h.join().unwrap();
    }
}
