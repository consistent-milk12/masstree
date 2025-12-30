//! TRI1: Triangle pattern overwrites.
//!
//! Port of `kvtest_tri1` from C++ `kvtest.hh`.
//!
//! Pattern:
//! - For each x in 0..limit:
//!   - For y in 0..=x: put(initial_pos + y*incr, x - y)
//! - Final state: key k = initial_pos + i*incr has value (limit - 1 - i)
//! - Tests many overwrites to same keys

use masstree::MassTree24Inline;
use std::sync::Arc;
use std::thread;

const LIMIT: usize = 500;
const INITIAL_POS: u64 = 0;

#[test]
fn tri1_single_thread() {
    let tree: MassTree24Inline<u64> = MassTree24Inline::new();
    let guard = tree.guard();
    let incr = 1u64;

    // Triangle insert pattern
    for x in 0..LIMIT {
        for y in 0..=x {
            let z = x - y;
            let key = (INITIAL_POS + y as u64 * incr).to_be_bytes();
            tree.insert_with_guard(&key, z as u64, &guard).unwrap();
        }
    }

    // Verify final state
    // Key at position i should have value (LIMIT - 1 - i)
    for i in 0..LIMIT {
        let key = (INITIAL_POS + i as u64 * incr).to_be_bytes();
        let val = tree.get_with_guard(&key, &guard);
        let expected = (LIMIT - 1 - i) as u64;
        assert_eq!(val, Some(expected), "key {} mismatch", i);
    }
}

#[test]
fn tri1_check() {
    // Pre-populate with triangle pattern, then verify
    let tree: MassTree24Inline<u64> = MassTree24Inline::new();
    let guard = tree.guard();
    let incr = 1u64;

    // Insert
    for x in 0..LIMIT {
        for y in 0..=x {
            let z = x - y;
            let key = (INITIAL_POS + y as u64 * incr).to_be_bytes();
            tree.insert_with_guard(&key, z as u64, &guard).unwrap();
        }
    }

    // Check phase (like kvtest_tri1_check)
    for x in 0..LIMIT {
        let key = (INITIAL_POS + x as u64 * incr).to_be_bytes();
        let expected = (LIMIT - 1 - x) as u64;
        let val = tree.get_with_guard(&key, &guard);
        assert_eq!(val, Some(expected), "check: key {} mismatch", x);
    }
}

#[test]
fn tri1_concurrent() {
    let tree = Arc::new(MassTree24Inline::<u64>::new());
    let num_threads = 4;

    // Each thread works on separate key ranges
    let handles: Vec<_> = (0..num_threads)
        .map(|tid| {
            let tree = Arc::clone(&tree);
            thread::spawn(move || {
                let guard = tree.guard();
                let incr = num_threads as u64;
                let initial = tid as u64;
                let limit = LIMIT / num_threads;

                // Triangle pattern for this thread's keys
                for x in 0..limit {
                    for y in 0..=x {
                        let z = x - y;
                        let key = (initial + y as u64 * incr).to_be_bytes();
                        let _ = tree.insert_with_guard(&key, z as u64, &guard);
                    }
                }

                // Verify
                for i in 0..limit {
                    let key = (initial + i as u64 * incr).to_be_bytes();
                    let expected = (limit - 1 - i) as u64;
                    let val = tree.get_with_guard(&key, &guard);
                    assert_eq!(val, Some(expected), "tid {}: key {} mismatch", tid, i);
                }
            })
        })
        .collect();

    for h in handles {
        h.join().unwrap();
    }
}

#[test]
fn tri1_stress_overwrites() {
    // Stress test: many overwrites to same key
    let tree: MassTree24Inline<u64> = MassTree24Inline::new();
    let guard = tree.guard();

    let key = 12345u64.to_be_bytes();

    // Write many times to same key
    for i in 0..10_000u64 {
        tree.insert_with_guard(&key, i, &guard).unwrap();
    }

    // Should have last value
    let val = tree.get_with_guard(&key, &guard);
    assert_eq!(val, Some(9999));
}
