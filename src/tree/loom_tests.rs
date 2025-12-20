//! Loom tests for MassTree concurrent operations.
//!
//! Loom provides deterministic concurrency testing by exploring all possible
//! thread interleavings. This catches subtle race conditions that random
//! testing might miss.
//!
//! Run with: `RUSTFLAGS="--cfg loom" cargo test --lib tree::loom_tests`
//!
//! NOTE: Loom tests are expensive - they explore all interleavings.
//! Keep the number of operations small to avoid state explosion.

use loom::sync::Arc;
use loom::sync::atomic::{AtomicPtr, AtomicU64, AtomicUsize, Ordering};
use loom::thread;

// ============================================================================
//  Simplified Concurrent Map for Loom Testing
// ============================================================================

/// A simplified concurrent map for loom testing.
///
/// This tests the core CAS semantics without the full MassTree complexity.
/// The actual MassTree uses these same patterns for root updates and
/// slot modifications.
struct LoomMap {
    /// Root pointer (CAS updates on split).
    root: AtomicPtr<LoomNode>,
    /// Entry count.
    count: AtomicUsize,
}

/// Simplified node for loom testing.
struct LoomNode {
    /// Single key-value pair for simplicity.
    key: AtomicU64,
    value: AtomicU64,
    /// Version for optimistic concurrency.
    version: AtomicU64,
}

impl LoomNode {
    fn new() -> Self {
        Self {
            key: AtomicU64::new(0),
            value: AtomicU64::new(0),
            version: AtomicU64::new(0),
        }
    }

    fn boxed() -> Box<Self> {
        Box::new(Self::new())
    }
}

impl LoomMap {
    fn new() -> Self {
        let node = Box::into_raw(LoomNode::boxed());
        Self {
            root: AtomicPtr::new(node),
            count: AtomicUsize::new(0),
        }
    }

    /// Get value for key (optimistic read with version validation).
    ///
    /// Uses SeqCst ordering for strictest guarantees under loom.
    fn get(&self, key: u64) -> Option<u64> {
        loop {
            let root = self.root.load(Ordering::SeqCst);
            // SAFETY: root is valid (we allocated it)
            let node = unsafe { &*root };

            // Take version snapshot with SeqCst for strict ordering
            let v1 = node.version.load(Ordering::SeqCst);

            // Spin if node is locked (odd version = locked)
            if v1 & 1 != 0 {
                loom::thread::yield_now();
                continue;
            }

            // Read key and value with SeqCst
            let node_key = node.key.load(Ordering::SeqCst);
            let node_value = node.value.load(Ordering::SeqCst);

            // Validate version unchanged
            let v2 = node.version.load(Ordering::SeqCst);
            if v1 != v2 {
                // Version changed during read - retry
                loom::thread::yield_now();
                continue;
            }

            if node_key == key && node_key != 0 {
                return Some(node_value);
            }
            return None;
        }
    }

    /// Insert key-value pair (locked write with version increment).
    ///
    /// Uses SeqCst ordering for strictest guarantees under loom.
    fn insert(&self, key: u64, value: u64) -> Option<u64> {
        loop {
            let root = self.root.load(Ordering::SeqCst);
            // SAFETY: root is valid
            let node = unsafe { &*root };

            // Try to acquire "lock" via version CAS (odd = locked)
            let version = node.version.load(Ordering::SeqCst);
            if version & 1 != 0 {
                // Already locked - spin
                loom::thread::yield_now();
                continue;
            }

            // Try to lock (set low bit) using SeqCst
            if node
                .version
                .compare_exchange(version, version | 1, Ordering::SeqCst, Ordering::SeqCst)
                .is_err()
            {
                loom::thread::yield_now();
                continue;
            }

            // We hold the lock - read current state
            let old_key = node.key.load(Ordering::SeqCst);
            let old_value = node.value.load(Ordering::SeqCst);

            let result = if old_key == key && old_key != 0 {
                // Update existing
                node.value.store(value, Ordering::SeqCst);
                Some(old_value)
            } else if old_key == 0 {
                // Insert new
                node.key.store(key, Ordering::SeqCst);
                node.value.store(value, Ordering::SeqCst);
                self.count.fetch_add(1, Ordering::SeqCst);
                None
            } else {
                // Different key - in real impl would search/split
                // For simplicity, just fail
                None
            };

            // Unlock (clear low bit, increment version)
            node.version.store(version + 2, Ordering::SeqCst);

            return result;
        }
    }

    /// CAS root pointer (used in split propagation).
    fn cas_root(&self, expected: *mut LoomNode, new: *mut LoomNode) -> Result<(), *mut LoomNode> {
        self.root
            .compare_exchange(expected, new, Ordering::SeqCst, Ordering::SeqCst)
            .map(|_| ())
    }
}

impl Drop for LoomMap {
    fn drop(&mut self) {
        let root = self.root.load(Ordering::Relaxed);
        if !root.is_null() {
            // SAFETY: We own this pointer
            unsafe {
                drop(Box::from_raw(root));
            }
        }
    }
}

// ============================================================================
//  Loom Tests
// ============================================================================

/// Test concurrent insert + get on same key.
///
/// Verifies that:
/// 1. Get never sees partial writes
/// 2. Version validation correctly detects concurrent modifications
#[test]
fn test_loom_concurrent_insert_get() {
    loom::model(|| {
        let map = Arc::new(LoomMap::new());

        let m1 = Arc::clone(&map);
        let t1 = thread::spawn(move || {
            m1.insert(42, 100);
        });

        let m2 = Arc::clone(&map);
        let t2 = thread::spawn(move || {
            // Get may return None (before insert) or Some(100) (after insert)
            let result = m2.get(42);
            if let Some(v) = result {
                assert_eq!(v, 100, "Got unexpected value");
            }
        });

        t1.join().unwrap();
        t2.join().unwrap();

        // After both threads complete, key should exist
        assert_eq!(map.get(42), Some(100));
    });
}

/// Test concurrent inserts to same key (update race).
///
/// Verifies that:
/// 1. Mutual exclusion via version lock works
/// 2. Final value is one of the inserted values
#[test]
fn test_loom_concurrent_insert_insert() {
    loom::model(|| {
        let map = Arc::new(LoomMap::new());

        // First, insert the key so updates can happen
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

        // Final value should be one of the inserted values
        let final_value = map.get(42).unwrap();
        assert!(
            final_value == 100 || final_value == 200,
            "Unexpected final value: {final_value}"
        );
    });
}

/// Test concurrent get operations (should never block each other).
///
/// Verifies that multiple readers can proceed without blocking.
#[test]
fn test_loom_concurrent_get_get() {
    loom::model(|| {
        let map = Arc::new(LoomMap::new());

        // Insert a value first
        map.insert(42, 999);

        let m1 = Arc::clone(&map);
        let t1 = thread::spawn(move || m1.get(42));

        let m2 = Arc::clone(&map);
        let t2 = thread::spawn(move || m2.get(42));

        let r1 = t1.join().unwrap();
        let r2 = t2.join().unwrap();

        // Both should see the same value
        assert_eq!(r1, Some(999));
        assert_eq!(r2, Some(999));
    });
}

/// Test CAS root update (simulates split propagation).
///
/// Verifies that only one CAS succeeds when two threads race.
#[test]
fn test_loom_cas_root() {
    loom::model(|| {
        let map = Arc::new(LoomMap::new());

        let original_root = map.root.load(Ordering::Relaxed);
        let new_node1 = Box::into_raw(LoomNode::boxed());
        let new_node2 = Box::into_raw(LoomNode::boxed());

        let m1 = Arc::clone(&map);
        let t1 = thread::spawn(move || m1.cas_root(original_root, new_node1));

        let m2 = Arc::clone(&map);
        let or2 = original_root;
        let t2 = thread::spawn(move || m2.cas_root(or2, new_node2));

        let r1 = t1.join().unwrap();
        let r2 = t2.join().unwrap();

        // Exactly one should succeed
        let success_count = [r1.is_ok(), r2.is_ok()]
            .iter()
            .filter(|&&x| x)
            .count();
        assert_eq!(success_count, 1, "Exactly one CAS should succeed");

        // Clean up the node that wasn't installed
        // SAFETY: One of new_node1/new_node2 was not installed
        unsafe {
            if r1.is_err() {
                drop(Box::from_raw(new_node1));
            }
            if r2.is_err() {
                drop(Box::from_raw(new_node2));
            }
        }
    });
}

/// Test version validation detects concurrent modification.
///
/// Verifies that get() retries when version changes.
#[test]
fn test_loom_version_validation() {
    loom::model(|| {
        let map = Arc::new(LoomMap::new());

        // Insert initial value
        map.insert(42, 100);

        let m1 = Arc::clone(&map);
        let t1 = thread::spawn(move || {
            // Update value while get is reading
            m1.insert(42, 200);
        });

        let m2 = Arc::clone(&map);
        let t2 = thread::spawn(move || {
            // Get should see either 100 or 200, never partial
            let result = m2.get(42);
            if let Some(v) = result {
                assert!(v == 100 || v == 200, "Got partial/invalid value: {v}");
            }
        });

        t1.join().unwrap();
        t2.join().unwrap();
    });
}

/// Test count tracking under concurrent inserts.
#[test]
fn test_loom_count_tracking() {
    loom::model(|| {
        let map = Arc::new(LoomMap::new());

        // Insert should increment count exactly once
        map.insert(42, 100);

        assert_eq!(map.count.load(Ordering::Relaxed), 1);

        // Update should not increment count
        map.insert(42, 200);

        assert_eq!(map.count.load(Ordering::Relaxed), 1);
    });
}
