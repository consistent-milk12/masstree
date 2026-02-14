//! RW16: 16-byte padded keys (multi-layer).
//!
//! Port of `kvtest_rw16` from C++ `kvtest.hh`.
//!
//! Pattern:
//! - Keys are 16 bytes: zero-padded integers
//! - Forces multi-layer trie traversal (keys > 8 bytes)
//! - Tests layer creation and traversal

#![allow(
    clippy::unwrap_used,
    clippy::cast_sign_loss,
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap
)]

use masstree::MassTree15Inline;
use rand::{RngExt, SeedableRng, rngs::StdRng, seq::SliceRandom};
use std::sync::Arc;
use std::thread;

const SEED: u64 = 31949;
const N: usize = 20_000;

fn make_key16(x: i32) -> [u8; 16] {
    let mut key = [b'0'; 16];
    let s = format!("{:016}", x.abs());
    key.copy_from_slice(s.as_bytes());
    key
}

const fn make_val16(x: i32) -> u64 {
    (x.wrapping_add(1)) as u64
}

#[test]
fn rw16_single_thread() {
    let tree: MassTree15Inline<u64> = MassTree15Inline::new();
    let guard = tree.guard();
    let mut rng = StdRng::seed_from_u64(SEED);

    // Put phase
    let mut keys: Vec<i32> = Vec::with_capacity(N);
    for _ in 0..N {
        let x: i32 = rng.random();
        let key = make_key16(x);
        let val = make_val16(x);
        tree.insert_with_guard(&key, val, &guard);
        keys.push(x);
    }

    // Shuffle for get phase
    keys.shuffle(&mut rng);

    // Get phase
    for x in &keys {
        let key = make_key16(*x);
        let expected = make_val16(*x);
        let val = tree.get_with_guard(&key, &guard);
        assert_eq!(val, Some(expected), "key {x:016} mismatch");
    }
}

#[test]
fn rw16_concurrent() {
    let tree = Arc::new(MassTree15Inline::<u64>::new());
    let num_threads = 4;
    let per_thread = N / num_threads;

    let handles: Vec<_> = (0..num_threads)
        .map(|tid| {
            let tree = Arc::clone(&tree);
            thread::spawn(move || {
                let guard = tree.guard();
                let mut rng = StdRng::seed_from_u64(SEED + tid as u64);

                // Put phase
                let mut keys: Vec<i32> = Vec::with_capacity(per_thread);

                for _ in 0..per_thread {
                    let x: i32 = rng.random();
                    let key = make_key16(x);
                    let val = make_val16(x);
                    let _ = tree.insert_with_guard(&key, val, &guard);
                    keys.push(x);
                }

                // Shuffle and get
                keys.shuffle(&mut rng);

                for x in &keys {
                    let key = make_key16(*x);
                    let val = tree.get_with_guard(&key, &guard);
                    assert!(val.is_some(), "key {x:016} not found");
                }
            })
        })
        .collect();

    for h in handles {
        h.join().unwrap();
    }
}

#[test]
fn rw16_sequential_keys() {
    // Test with sequential 16-byte keys
    let tree: MassTree15Inline<u64> = MassTree15Inline::new();
    let guard = tree.guard();

    (0..N as i32).for_each(|i| {
        let key = make_key16(i);
        tree.insert_with_guard(&key, i as u64, &guard);
    });

    // Verify all
    for i in 0..N as i32 {
        let key = make_key16(i);
        let val = tree.get_with_guard(&key, &guard);
        assert_eq!(val, Some(i as u64));
    }
}
