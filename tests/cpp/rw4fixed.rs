//! RW4FIXED: Sequentially decreasing 8-byte keys.
//!
//! Port of `kvtest_rw4fixed` from C++ `kvtest.hh`.
//!
//! Pattern:
//! - Insert keys from TOP down to TOP-N
//! - Get all keys and verify values
//! - Tests reverse sequential insertion

#![allow(clippy::unwrap_used, clippy::cast_sign_loss)]

use masstree::MassTree15Inline as MassTree24Inline;
use std::sync::Arc;
use std::thread;

const TOP: u64 = 99_999_999;
const N: u64 = 50_000;

#[test]
fn rw4fixed_single_thread() {
    let tree: MassTree24Inline<u64> = MassTree24Inline::new();
    let guard = tree.guard();

    // Put phase - insert from top down
    for n in 0..N {
        let key_val = TOP - n;
        let key = key_val.to_be_bytes();
        tree.insert_with_guard(&key, n + 1, &guard).unwrap();
    }

    // Get phase - verify all keys
    for n in 0..N {
        let key_val = TOP - n;
        let key = key_val.to_be_bytes();
        let val = tree.get_with_guard(&key, &guard);
        assert_eq!(val, Some(n + 1), "key {key_val} mismatch");
    }
}

#[test]
fn rw4fixed_concurrent() {
    let tree = Arc::new(MassTree24Inline::<u64>::new());
    let num_threads = 4;
    let per_thread = N / num_threads as u64;

    // Concurrent put phase
    let handles: Vec<_> = (0..num_threads)
        .map(|tid| {
            let tree = Arc::clone(&tree);
            thread::spawn(move || {
                let guard = tree.guard();
                let start = tid as u64 * per_thread;
                let end = start + per_thread;

                for n in start..end {
                    let key_val = TOP - n;
                    let key = key_val.to_be_bytes();
                    let _ = tree.insert_with_guard(&key, n + 1, &guard);
                }
            })
        })
        .collect();

    for h in handles {
        h.join().unwrap();
    }

    // Verify phase
    let guard = tree.guard();
    for n in 0..N {
        let key_val = TOP - n;
        let key = key_val.to_be_bytes();
        let val = tree.get_with_guard(&key, &guard);
        assert_eq!(val, Some(n + 1), "key {key_val} mismatch");
    }
}
