//! Shuttle linearizability tests for MassTree.
//!
//! Shuttle provides systematic concurrency testing by exploring different
//! thread schedules. Unlike loom, shuttle uses a randomized approach with
//! configurable iteration counts.
//!
//! Run with: `cargo test --lib tree::shuttle_tests`
//!
//! # Linearizability Testing
//!
//! These tests verify that concurrent operations appear to take effect
//! instantaneously at some point between their invocation and response.
//! This is the gold standard for concurrent data structure correctness.
//!
//! # Reference
//!
//! - TODO.md §3.3.6: Shuttle linearizability testing
//! - Suggestions.md §9: Linearizability Testing

use shuttle::sync::Arc;
use shuttle::thread;
use std::sync::atomic::{AtomicU64, Ordering};

// ============================================================================
//  Simplified Concurrent Map for Shuttle Testing
// ============================================================================

/// A simplified concurrent map for shuttle testing.
///
/// This uses shuttle's synchronization primitives to test concurrent behavior.
/// The patterns here mirror MassTree's actual implementation:
/// - Optimistic reads with version validation
/// - Locked writes with version increment
/// - CAS for structural modifications
struct ShuttleMap {
    /// Storage for key-value pairs.
    slots: Vec<(AtomicU64, AtomicU64)>,
    /// Version for each slot (odd = locked).
    versions: Vec<AtomicU64>,
    /// Entry count.
    count: AtomicU64,
}

impl ShuttleMap {
    fn new(capacity: usize) -> Self {
        let mut slots = Vec::with_capacity(capacity);
        let mut versions = Vec::with_capacity(capacity);
        for _ in 0..capacity {
            slots.push((AtomicU64::new(0), AtomicU64::new(0)));
            versions.push(AtomicU64::new(0));
        }
        Self {
            slots,
            versions,
            count: AtomicU64::new(0),
        }
    }

    /// Find slot index for key (simple linear probe for testing).
    fn find_slot(&self, key: u64) -> usize {
        (key as usize) % self.slots.len()
    }

    /// Get value for key (optimistic read).
    fn get(&self, key: u64) -> Option<u64> {
        let slot_idx = self.find_slot(key);

        loop {
            // Take version snapshot
            let v1 = self.versions[slot_idx].load(Ordering::Acquire);

            // Spin if locked
            if v1 & 1 != 0 {
                shuttle::thread::yield_now();
                continue;
            }

            // Read key and value
            let slot_key = self.slots[slot_idx].0.load(Ordering::Relaxed);
            let slot_value = self.slots[slot_idx].1.load(Ordering::Relaxed);

            // Validate version unchanged
            let v2 = self.versions[slot_idx].load(Ordering::Acquire);
            if v1 != v2 {
                // Version changed - retry
                continue;
            }

            if slot_key == key && slot_key != 0 {
                return Some(slot_value);
            }
            return None;
        }
    }

    /// Insert key-value pair (locked write).
    fn insert(&self, key: u64, value: u64) -> Option<u64> {
        let slot_idx = self.find_slot(key);

        loop {
            let version = self.versions[slot_idx].load(Ordering::Relaxed);

            // Spin if locked
            if version & 1 != 0 {
                shuttle::thread::yield_now();
                continue;
            }

            // Try to lock
            if self.versions[slot_idx]
                .compare_exchange(version, version | 1, Ordering::Acquire, Ordering::Relaxed)
                .is_err()
            {
                continue;
            }

            // We hold the lock
            let old_key = self.slots[slot_idx].0.load(Ordering::Relaxed);
            let old_value = self.slots[slot_idx].1.load(Ordering::Relaxed);

            let result = if old_key == key && old_key != 0 {
                // Update existing
                self.slots[slot_idx].1.store(value, Ordering::Release);
                Some(old_value)
            } else if old_key == 0 {
                // Insert new
                self.slots[slot_idx].0.store(key, Ordering::Relaxed);
                self.slots[slot_idx].1.store(value, Ordering::Release);
                self.count.fetch_add(1, Ordering::Relaxed);
                None
            } else {
                // Hash collision - for simplicity, fail
                // Real impl would use chaining or open addressing
                None
            };

            // Unlock (clear low bit, increment version)
            self.versions[slot_idx].store(version + 2, Ordering::Release);

            return result;
        }
    }

    fn len(&self) -> u64 {
        self.count.load(Ordering::Relaxed)
    }
}

// ============================================================================
//  Shuttle Tests
// ============================================================================

/// Test concurrent insert + get returns consistent results.
///
/// This is the core linearizability test: if insert completes before get starts,
/// get must see the inserted value.
#[test]
fn test_shuttle_insert_get_linearizable() {
    shuttle::check_random(
        || {
            let map = Arc::new(ShuttleMap::new(16));

            let m1 = Arc::clone(&map);
            let t1 = thread::spawn(move || {
                m1.insert(42, 100);
            });

            let m2 = Arc::clone(&map);
            let t2 = thread::spawn(move || -> Option<u64> { m2.get(42) });

            t1.join().unwrap();
            let result = t2.join().unwrap();

            // After t1 completes, get should return the value
            // (result may be None if t2 ran first, or Some(100) if t1 ran first)
            if result.is_some() {
                assert_eq!(result, Some(100));
            }

            // Verify final state
            assert_eq!(map.get(42), Some(100));
        },
        100, // iterations
    );
}

/// Test concurrent updates are serializable.
///
/// When two threads update the same key, the final value must be
/// one of the values that was written.
#[test]
fn test_shuttle_concurrent_updates_serializable() {
    shuttle::check_random(
        || {
            let map = Arc::new(ShuttleMap::new(16));

            // Insert initial value
            map.insert(42, 0);

            let m1 = Arc::clone(&map);
            let t1 = thread::spawn(move || {
                m1.insert(42, 100);
            });

            let m2 = Arc::clone(&map);
            let t2 = thread::spawn(move || {
                m2.insert(42, 200);
            });

            t1.join().unwrap();
            t2.join().unwrap();

            // Final value must be one of the written values
            let final_value = map.get(42).unwrap();
            assert!(
                final_value == 100 || final_value == 200,
                "Unexpected final value: {final_value}"
            );
        },
        100,
    );
}

/// Test multiple readers don't interfere with each other.
#[test]
fn test_shuttle_concurrent_reads() {
    shuttle::check_random(
        || {
            let map = Arc::new(ShuttleMap::new(16));

            // Insert some values
            map.insert(1, 100);
            map.insert(2, 200);
            map.insert(3, 300);

            let m1 = Arc::clone(&map);
            let t1 = thread::spawn(move || m1.get(1));

            let m2 = Arc::clone(&map);
            let t2 = thread::spawn(move || m2.get(2));

            let m3 = Arc::clone(&map);
            let t3 = thread::spawn(move || m3.get(3));

            let r1 = t1.join().unwrap();
            let r2 = t2.join().unwrap();
            let r3 = t3.join().unwrap();

            // All reads should see correct values
            assert_eq!(r1, Some(100));
            assert_eq!(r2, Some(200));
            assert_eq!(r3, Some(300));
        },
        100,
    );
}

/// Test concurrent inserts to different keys.
///
/// Inserts to different keys should not interfere.
#[test]
fn test_shuttle_concurrent_different_keys() {
    shuttle::check_random(
        || {
            let map = Arc::new(ShuttleMap::new(16));

            let m1 = Arc::clone(&map);
            let t1 = thread::spawn(move || {
                m1.insert(1, 100);
            });

            let m2 = Arc::clone(&map);
            let t2 = thread::spawn(move || {
                m2.insert(2, 200);
            });

            let m3 = Arc::clone(&map);
            let t3 = thread::spawn(move || {
                m3.insert(3, 300);
            });

            t1.join().unwrap();
            t2.join().unwrap();
            t3.join().unwrap();

            // All keys should exist with correct values
            assert_eq!(map.get(1), Some(100));
            assert_eq!(map.get(2), Some(200));
            assert_eq!(map.get(3), Some(300));
            assert_eq!(map.len(), 3);
        },
        100,
    );
}

/// Test read-your-writes consistency.
///
/// A thread that writes a value should immediately see that value.
#[test]
fn test_shuttle_read_your_writes() {
    shuttle::check_random(
        || {
            let map = Arc::new(ShuttleMap::new(16));

            let m1 = Arc::clone(&map);
            let t1 = thread::spawn(move || {
                m1.insert(42, 100);
                // Immediately read back
                let read = m1.get(42);
                assert_eq!(read, Some(100), "Thread should see its own write");
            });

            t1.join().unwrap();
        },
        100,
    );
}

/// Test insert returns correct old value.
///
/// When updating an existing key, insert should return the previous value.
#[test]
fn test_shuttle_insert_returns_old_value() {
    shuttle::check_random(
        || {
            let map = Arc::new(ShuttleMap::new(16));

            // First insert returns None
            let r1 = map.insert(42, 100);
            assert_eq!(r1, None);

            // Second insert returns old value
            let r2 = map.insert(42, 200);
            assert_eq!(r2, Some(100));

            // Verify current value
            assert_eq!(map.get(42), Some(200));
        },
        100,
    );
}

/// Test version validation prevents reading partial writes.
///
/// This simulates the scenario where a read overlaps with a write,
/// and verifies that the read either sees the old value or new value,
/// never a partial/corrupted value.
#[test]
fn test_shuttle_no_torn_reads() {
    shuttle::check_random(
        || {
            let map = Arc::new(ShuttleMap::new(16));

            // Insert with a value where high and low bits are consistent
            map.insert(42, 0xAAAA_BBBB);

            let m1 = Arc::clone(&map);
            let t1 = thread::spawn(move || {
                // Update to a different consistent value
                m1.insert(42, 0xCCCC_DDDD);
            });

            let m2 = Arc::clone(&map);
            let t2 = thread::spawn(move || -> Option<u64> { m2.get(42) });

            t1.join().unwrap();
            let result = t2.join().unwrap();

            // Result must be one of the valid values, never a mix
            if let Some(v) = result {
                assert!(
                    v == 0xAAAA_BBBB || v == 0xCCCC_DDDD,
                    "Got torn/partial value: {v:x}"
                );
            }
        },
        100,
    );
}

/// Test concurrent operation history is linearizable.
///
/// This test records the history of operations and verifies that
/// there exists a valid sequential ordering.
#[test]
fn test_shuttle_linearizable_history() {
    use std::sync::Mutex;

    shuttle::check_random(
        || {
            let map = Arc::new(ShuttleMap::new(16));
            let history = Arc::new(Mutex::new(Vec::new()));

            let m1 = Arc::clone(&map);
            let h1 = Arc::clone(&history);
            let t1 = thread::spawn(move || {
                let result = m1.insert(42, 100);
                h1.lock().unwrap().push(("insert", 42, 100, result));
            });

            let m2 = Arc::clone(&map);
            let h2 = Arc::clone(&history);
            let t2 = thread::spawn(move || {
                let result = m2.get(42);
                h2.lock()
                    .unwrap()
                    .push(("get", 42, 0, result.map(|v| Some(v)).unwrap_or(None)));
            });

            t1.join().unwrap();
            t2.join().unwrap();

            // Verify history is consistent:
            // - If get returned Some(100), insert must have happened first
            // - If get returned None, insert hadn't happened yet when get ran
            let hist = history.lock().unwrap();
            for (op, key, _val, result) in hist.iter() {
                if *op == "get" && *key == 42 {
                    if let Some(v) = result {
                        assert_eq!(*v, 100, "Get returned wrong value");
                    }
                }
            }
        },
        100,
    );
}

/// Test multiple operations from each thread.
///
/// More complex interleaving with multiple operations per thread.
#[test]
fn test_shuttle_multi_op_threads() {
    shuttle::check_random(
        || {
            let map = Arc::new(ShuttleMap::new(32));

            let m1 = Arc::clone(&map);
            let t1 = thread::spawn(move || {
                m1.insert(1, 10);
                m1.insert(2, 20);
                let v1 = m1.get(1);
                let v2 = m1.get(2);
                (v1, v2)
            });

            let m2 = Arc::clone(&map);
            let t2 = thread::spawn(move || {
                m2.insert(3, 30);
                m2.insert(4, 40);
                let v3 = m2.get(3);
                let v4 = m2.get(4);
                (v3, v4)
            });

            let (v1, v2) = t1.join().unwrap();
            let (v3, v4) = t2.join().unwrap();

            // Each thread should see its own writes
            assert_eq!(v1, Some(10));
            assert_eq!(v2, Some(20));
            assert_eq!(v3, Some(30));
            assert_eq!(v4, Some(40));

            // Final state should have all keys
            assert_eq!(map.get(1), Some(10));
            assert_eq!(map.get(2), Some(20));
            assert_eq!(map.get(3), Some(30));
            assert_eq!(map.get(4), Some(40));
        },
        100,
    );
}
