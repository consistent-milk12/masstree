//! Concurrent operation regression tests.
//!
//! These tests stress concurrent get/insert operations to catch memory safety
//! issues, race conditions, and correctness bugs.
//!
//! Run with: `cargo test --test concurrent_regression`
//! Run with release: `cargo test --test concurrent_regression --release`

#![allow(clippy::pedantic)]
#![expect(clippy::unwrap_used)]

use masstree::MassTree;
use std::collections::HashSet;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::thread;

// =============================================================================
// Basic Concurrent Insert
// =============================================================================

#[test]
fn concurrent_insert_2_threads_disjoint_keys() {
    let tree = Arc::new(MassTree::<u64>::new());

    let handles: Vec<_> = (0..2)
        .map(|t| {
            let tree = Arc::clone(&tree);
            thread::spawn(move || {
                let guard = tree.guard();
                for i in 0..100 {
                    let key = ((t * 1000 + i) as u64).to_be_bytes();
                    let _ = tree.insert_with_guard(&key, (t * 1000 + i) as u64, &guard);
                }
            })
        })
        .collect();

    for h in handles {
        h.join().unwrap();
    }

    assert_eq!(tree.len(), 200);
}

#[test]
fn concurrent_insert_4_threads_disjoint_keys() {
    let tree = Arc::new(MassTree::<u64>::new());

    let handles: Vec<_> = (0..4)
        .map(|t| {
            let tree = Arc::clone(&tree);
            thread::spawn(move || {
                let guard = tree.guard();
                for i in 0..100 {
                    let key = ((t * 1000 + i) as u64).to_be_bytes();
                    let _ = tree.insert_with_guard(&key, (t * 1000 + i) as u64, &guard);
                }
            })
        })
        .collect();

    for h in handles {
        h.join().unwrap();
    }

    assert_eq!(tree.len(), 400);
}

#[test]
fn concurrent_insert_8_threads_disjoint_keys() {
    let tree = Arc::new(MassTree::<u64>::new());

    let handles: Vec<_> = (0..8)
        .map(|t| {
            let tree = Arc::clone(&tree);
            thread::spawn(move || {
                let guard = tree.guard();
                for i in 0..100 {
                    let key = ((t * 1000 + i) as u64).to_be_bytes();
                    let _ = tree.insert_with_guard(&key, (t * 1000 + i) as u64, &guard);
                }
            })
        })
        .collect();

    for h in handles {
        h.join().unwrap();
    }

    assert_eq!(tree.len(), 800);
}

// =============================================================================
// Concurrent Insert with Contention (Same Key Ranges)
// =============================================================================

#[test]
fn concurrent_insert_2_threads_overlapping_keys() {
    let tree = Arc::new(MassTree::<u64>::new());
    let success_count = Arc::new(AtomicUsize::new(0));

    let handles: Vec<_> = (0..2)
        .map(|t| {
            let tree = Arc::clone(&tree);
            let success_count = Arc::clone(&success_count);
            thread::spawn(move || {
                let guard = tree.guard();
                for i in 0..100 {
                    // Overlapping key range
                    let key = (i as u64).to_be_bytes();
                    if tree.insert_with_guard(&key, t as u64, &guard).is_ok() {
                        success_count.fetch_add(1, Ordering::Relaxed);
                    }
                }
            })
        })
        .collect();

    for h in handles {
        h.join().unwrap();
    }

    // All keys should be present (some may have been updated)
    assert_eq!(tree.len(), 100);
}

#[test]
fn concurrent_insert_4_threads_overlapping_keys() {
    let tree = Arc::new(MassTree::<u64>::new());

    let handles: Vec<_> = (0..4)
        .map(|t| {
            let tree = Arc::clone(&tree);
            thread::spawn(move || {
                let guard = tree.guard();
                for i in 0..50 {
                    let key = (i as u64).to_be_bytes();
                    let _ = tree.insert_with_guard(&key, t as u64, &guard);
                }
            })
        })
        .collect();

    for h in handles {
        h.join().unwrap();
    }

    assert_eq!(tree.len(), 50);
}

// =============================================================================
// Concurrent Read + Write
// =============================================================================

#[test]
fn concurrent_read_write_2_threads() {
    let tree = Arc::new(MassTree::<u64>::new());

    // Pre-populate
    {
        let guard = tree.guard();
        for i in 0..100 {
            let key = (i as u64).to_be_bytes();
            let _ = tree.insert_with_guard(&key, i as u64, &guard);
        }
    }

    let tree1 = Arc::clone(&tree);
    let writer = thread::spawn(move || {
        let guard = tree1.guard();
        for i in 100..200 {
            let key = (i as u64).to_be_bytes();
            let _ = tree1.insert_with_guard(&key, i as u64, &guard);
        }
    });

    let tree2 = Arc::clone(&tree);
    let reader = thread::spawn(move || {
        let guard = tree2.guard();
        let mut found = 0;
        for i in 0..100 {
            let key = (i as u64).to_be_bytes();
            if tree2.get_with_guard(&key, &guard).is_some() {
                found += 1;
            }
        }
        found
    });

    writer.join().unwrap();
    let found = reader.join().unwrap();

    // Reader should find all pre-populated keys
    assert_eq!(found, 100);
    assert_eq!(tree.len(), 200);
}

#[test]
fn concurrent_read_write_4_threads() {
    let tree = Arc::new(MassTree::<u64>::new());

    // Pre-populate
    {
        let guard = tree.guard();
        for i in 0..1000 {
            let key = (i as u64).to_be_bytes();
            let _ = tree.insert_with_guard(&key, i as u64, &guard);
        }
    }

    let handles: Vec<_> = (0..4)
        .map(|t| {
            let tree = Arc::clone(&tree);
            thread::spawn(move || {
                let guard = tree.guard();
                if t % 2 == 0 {
                    // Writer
                    for i in 0..100 {
                        let key = ((1000 + t * 100 + i) as u64).to_be_bytes();
                        let _ = tree.insert_with_guard(&key, i as u64, &guard);
                    }
                } else {
                    // Reader
                    for i in 0..1000 {
                        let key = (i as u64).to_be_bytes();
                        let _ = tree.get_with_guard(&key, &guard);
                    }
                }
            })
        })
        .collect();

    for h in handles {
        h.join().unwrap();
    }

    // 1000 pre-populated + 2 writers * 100 each
    assert_eq!(tree.len(), 1200);
}

// =============================================================================
// Stress Tests
// =============================================================================

#[test]
fn stress_concurrent_insert_many_keys() {
    let tree = Arc::new(MassTree::<u64>::new());

    let handles: Vec<_> = (0..4)
        .map(|t| {
            let tree = Arc::clone(&tree);
            thread::spawn(move || {
                let guard = tree.guard();
                for i in 0..500 {
                    let key = ((t * 10000 + i) as u64).to_be_bytes();
                    let _ = tree.insert_with_guard(&key, i as u64, &guard);
                }
            })
        })
        .collect();

    for h in handles {
        h.join().unwrap();
    }

    assert_eq!(tree.len(), 2000);

    // Verify all keys are readable
    let guard = tree.guard();
    for t in 0..4 {
        for i in 0..500 {
            let key = ((t * 10000 + i) as u64).to_be_bytes();
            assert!(
                tree.get_with_guard(&key, &guard).is_some(),
                "Key {} not found",
                t * 10000 + i
            );
        }
    }
}

#[test]
fn stress_concurrent_insert_sequential_keys() {
    // Sequential keys cause more splits
    let tree = Arc::new(MassTree::<u64>::new());

    let handles: Vec<_> = (0..4)
        .map(|t| {
            let tree = Arc::clone(&tree);
            thread::spawn(move || {
                let guard = tree.guard();
                // Interleaved sequential: 0,4,8,12... / 1,5,9,13... etc
                for i in 0..250 {
                    let key = ((i * 4 + t) as u64).to_be_bytes();
                    let _ = tree.insert_with_guard(&key, i as u64, &guard);
                }
            })
        })
        .collect();

    for h in handles {
        h.join().unwrap();
    }

    assert_eq!(tree.len(), 1000);
}

#[test]
fn stress_repeated_runs() {
    // Run multiple times to catch intermittent issues
    for run in 0..10 {
        let tree = Arc::new(MassTree::<u64>::new());

        let handles: Vec<_> = (0..4)
            .map(|t| {
                let tree = Arc::clone(&tree);
                thread::spawn(move || {
                    let guard = tree.guard();
                    for i in 0..100 {
                        let key = ((t * 1000 + i) as u64).to_be_bytes();
                        let _ = tree.insert_with_guard(&key, i as u64, &guard);
                    }
                })
            })
            .collect();

        for h in handles {
            h.join().unwrap();
        }

        assert_eq!(tree.len(), 400, "Failed on run {}", run);
    }
}

// =============================================================================
// Split-Heavy Tests (trigger many splits)
// =============================================================================

#[test]
fn concurrent_insert_triggers_splits() {
    // Insert enough keys to trigger multiple splits
    let tree = Arc::new(MassTree::<u64>::new());

    let handles: Vec<_> = (0..2)
        .map(|t| {
            let tree = Arc::clone(&tree);
            thread::spawn(move || {
                let guard = tree.guard();
                // Each thread inserts 200 keys, enough to trigger splits
                for i in 0..200 {
                    let key = ((t * 10000 + i) as u64).to_be_bytes();
                    let _ = tree.insert_with_guard(&key, i as u64, &guard);
                }
            })
        })
        .collect();

    for h in handles {
        h.join().unwrap();
    }

    assert_eq!(tree.len(), 400);

    // Verify all keys
    let guard = tree.guard();
    for t in 0..2 {
        for i in 0..200 {
            let key = ((t * 10000 + i) as u64).to_be_bytes();
            assert!(tree.get_with_guard(&key, &guard).is_some());
        }
    }
}

#[test]
fn concurrent_insert_adjacent_keys_same_leaf() {
    // Keys that likely land in the same leaf
    let tree = Arc::new(MassTree::<u64>::new());

    let handles: Vec<_> = (0..4)
        .map(|t| {
            let tree = Arc::clone(&tree);
            thread::spawn(move || {
                let guard = tree.guard();
                // All threads insert to same key range (0-99)
                // This maximizes contention on the same leaves
                for i in 0..25 {
                    let key = ((t * 25 + i) as u64).to_be_bytes();
                    let _ = tree.insert_with_guard(&key, (t * 25 + i) as u64, &guard);
                }
            })
        })
        .collect();

    for h in handles {
        h.join().unwrap();
    }

    assert_eq!(tree.len(), 100);
}

// =============================================================================
// Long Key Tests (multi-layer)
// =============================================================================

#[test]
fn concurrent_insert_long_keys() {
    let tree = Arc::new(MassTree::<u64>::new());

    let handles: Vec<_> = (0..4)
        .map(|t| {
            let tree = Arc::clone(&tree);
            thread::spawn(move || {
                let guard = tree.guard();
                for i in 0..50 {
                    // 24-byte key spanning multiple layers
                    let key = format!("thread{:02}_key_{:08}", t, i);
                    let _ = tree.insert_with_guard(key.as_bytes(), i as u64, &guard);
                }
            })
        })
        .collect();

    for h in handles {
        h.join().unwrap();
    }

    assert_eq!(tree.len(), 200);
}

// =============================================================================
// Verification Tests
// =============================================================================

#[test]
fn concurrent_insert_all_values_correct() {
    let tree = Arc::new(MassTree::<u64>::new());

    let handles: Vec<_> = (0..4)
        .map(|t| {
            let tree = Arc::clone(&tree);
            thread::spawn(move || {
                let guard = tree.guard();
                for i in 0..100 {
                    let key = ((t * 1000 + i) as u64).to_be_bytes();
                    let value = (t * 1000 + i) as u64;
                    let _ = tree.insert_with_guard(&key, value, &guard);
                }
            })
        })
        .collect();

    for h in handles {
        h.join().unwrap();
    }

    // Verify all values are correct
    let guard = tree.guard();
    for t in 0..4usize {
        for i in 0..100usize {
            let key = ((t * 1000 + i) as u64).to_be_bytes();
            let expected = (t * 1000 + i) as u64;
            let actual = tree.get_with_guard(&key, &guard);
            assert_eq!(
                actual.map(|v| *v),
                Some(expected),
                "Wrong value for key {}",
                t * 1000 + i
            );
        }
    }
}

#[test]
fn concurrent_insert_no_duplicates() {
    let tree = Arc::new(MassTree::<u64>::new());

    let handles: Vec<_> = (0..4)
        .map(|t| {
            let tree = Arc::clone(&tree);
            thread::spawn(move || {
                let guard = tree.guard();
                for i in 0..100 {
                    let key = ((t * 1000 + i) as u64).to_be_bytes();
                    let _ = tree.insert_with_guard(&key, i as u64, &guard);
                }
            })
        })
        .collect();

    for h in handles {
        h.join().unwrap();
    }

    // Collect all keys and check for duplicates
    let guard = tree.guard();
    let mut seen = HashSet::new();
    for t in 0..4 {
        for i in 0..100 {
            let key = (t * 1000 + i) as u64;
            if tree.get_with_guard(&key.to_be_bytes(), &guard).is_some() {
                assert!(seen.insert(key), "Duplicate key found: {}", key);
            }
        }
    }

    assert_eq!(seen.len(), 400);
}

// =============================================================================
// Edge Cases
// =============================================================================

#[test]
fn concurrent_insert_empty_then_read() {
    let tree = Arc::new(MassTree::<u64>::new());

    let tree1 = Arc::clone(&tree);
    let writer = thread::spawn(move || {
        let guard = tree1.guard();
        for i in 0..100 {
            let key = (i as u64).to_be_bytes();
            let _ = tree1.insert_with_guard(&key, i as u64, &guard);
        }
    });

    // Reader starts immediately on empty tree
    let tree2 = Arc::clone(&tree);
    let reader = thread::spawn(move || {
        let guard = tree2.guard();
        let mut attempts = 0;
        let mut found = 0;
        while attempts < 1000 {
            for i in 0..100 {
                let key = (i as u64).to_be_bytes();
                if tree2.get_with_guard(&key, &guard).is_some() {
                    found += 1;
                }
            }
            attempts += 1;
            if found >= 100 {
                break;
            }
        }
        found
    });

    writer.join().unwrap();
    reader.join().unwrap();

    assert_eq!(tree.len(), 100);
}

#[test]
fn single_key_concurrent_updates() {
    // Multiple threads updating the same key
    let tree = Arc::new(MassTree::<u64>::new());
    let update_count = Arc::new(AtomicUsize::new(0));

    // Insert initial value
    {
        let guard = tree.guard();
        let _ = tree.insert_with_guard(&[0u8; 8], 0, &guard);
    }

    let handles: Vec<_> = (0..8)
        .map(|t| {
            let tree = Arc::clone(&tree);
            let update_count = Arc::clone(&update_count);
            thread::spawn(move || {
                let guard = tree.guard();
                for _ in 0..100 {
                    let _ = tree.insert_with_guard(&[0u8; 8], t as u64, &guard);
                    update_count.fetch_add(1, Ordering::Relaxed);
                }
            })
        })
        .collect();

    for h in handles {
        h.join().unwrap();
    }

    // Should still be exactly 1 key
    assert_eq!(tree.len(), 1);
    // All updates should have completed
    assert_eq!(update_count.load(Ordering::Relaxed), 800);
}
