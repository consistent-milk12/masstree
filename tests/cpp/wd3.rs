//! WD3: Write-delete-verify cycle.
//!
//! Port of `kvtest_wd3` from C++ `kvtest.hh`.
//!
//! Pattern:
//! - Insert a range of keys
//! - Verify all keys exist
//! - Remove all keys
//! - Verify all keys are gone
//! - Repeat for multiple rounds

#![allow(clippy::unwrap_used)]

use masstree::MassTree15Inline;
use std::sync::Arc;
use std::thread;

const NK: u64 = 5_000; // Keys per thread
const ROUNDS: usize = 3;

fn make_key(prefix: &[u8], val: u64) -> Vec<u8> {
    let mut key = prefix.to_vec();
    key.extend_from_slice(&val.to_be_bytes());
    key
}

#[test]
fn wd3_single_thread() {
    let tree: MassTree15Inline<u64> = MassTree15Inline::new();
    let guard = tree.guard();
    let prefix = b"test";

    for round in 0..ROUNDS {
        // Insert phase
        for i in 0..NK {
            let key = make_key(prefix, i);
            tree.insert_with_guard(&key, i, &guard).unwrap();
        }

        // Verify all present
        for i in 0..NK {
            let key = make_key(prefix, i);
            let val = tree.get_with_guard(&key, &guard);
            assert_eq!(val, Some(i), "round {round}: key {i} should exist");
        }

        // Remove phase
        for i in 0..NK {
            let key = make_key(prefix, i);
            let result = tree.remove_with_guard(&key, &guard);
            assert!(result.is_ok(), "round {round}: remove {i} failed");
        }

        // Verify all gone
        for i in 0..NK {
            let key = make_key(prefix, i);
            let val = tree.get_with_guard(&key, &guard);
            assert!(val.is_none(), "round {round}: key {i} should be gone");
        }
    }
}

#[test]
fn wd3_concurrent() {
    let tree = Arc::new(MassTree15Inline::<u64>::new());
    let num_threads = 4;

    let handles: Vec<_> = (0..num_threads)
        .map(|tid| {
            let tree = Arc::clone(&tree);
            thread::spawn(move || {
                let guard = tree.guard();
                // Each thread has its own prefix to avoid conflicts
                let prefix = format!("t{tid:02}");

                for round in 0..ROUNDS {
                    // Insert phase
                    for i in 0..NK {
                        let key = make_key(prefix.as_bytes(), i);
                        tree.insert_with_guard(&key, i, &guard).unwrap();
                    }

                    // Verify all present
                    for i in 0..NK {
                        let key = make_key(prefix.as_bytes(), i);
                        let val = tree.get_with_guard(&key, &guard);
                        assert_eq!(
                            val,
                            Some(i),
                            "tid {tid}, round {round}: key {i} should exist"
                        );
                    }

                    // Remove phase
                    for i in 0..NK {
                        let key = make_key(prefix.as_bytes(), i);
                        let _ = tree.remove_with_guard(&key, &guard);
                    }

                    // Verify all gone
                    for i in 0..NK {
                        let key = make_key(prefix.as_bytes(), i);
                        let val = tree.get_with_guard(&key, &guard);
                        assert!(
                            val.is_none(),
                            "tid {tid}, round {round}: key {i} should be gone"
                        );
                    }
                }
            })
        })
        .collect();

    for h in handles {
        h.join().unwrap();
    }
}

#[test]
fn wd3_with_prefix() {
    // Test with variable length prefixes (multi-layer keys)
    let tree: MassTree15Inline<u64> = MassTree15Inline::new();
    let guard = tree.guard();
    let prefix = b"longerprefix/path/to/";

    for round in 0..2 {
        // Insert
        for i in 0..1000u64 {
            let key = make_key(prefix, i);
            tree.insert_with_guard(&key, i, &guard).unwrap();
        }

        // Verify
        for i in 0..1000u64 {
            let key = make_key(prefix, i);
            let val = tree.get_with_guard(&key, &guard);
            assert_eq!(val, Some(i), "round {round}: key {i} mismatch");
        }

        // Remove
        for i in 0..1000u64 {
            let key = make_key(prefix, i);
            let _ = tree.remove_with_guard(&key, &guard);
        }

        // Verify gone
        for i in 0..1000u64 {
            let key = make_key(prefix, i);
            let val = tree.get_with_guard(&key, &guard);
            assert!(val.is_none(), "round {round}: key {i} should be gone");
        }
    }
}
