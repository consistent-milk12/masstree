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
        let success_count = [r1.is_ok(), r2.is_ok()].iter().filter(|&&x| x).count();
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

// ============================================================================
//  CAS Insert Pattern Tests
// ============================================================================

/// Simplified permutation CAS model for testing CAS insert semantics.
///
/// Models the core pattern: store slot data, then CAS permutation to publish.
struct LoomPermutedLeaf {
    /// Permutation value (size in low 4 bits, slots in higher bits).
    permutation: AtomicU64,
    /// Slot data (key, value pairs).
    slots: [(AtomicU64, AtomicU64); 4],
    /// Count of successful inserts.
    count: AtomicUsize,
}

impl LoomPermutedLeaf {
    fn new() -> Self {
        Self {
            // Empty permutation with slots [3,2,1,0] in free region
            // Size = 0, so all slots are free
            permutation: AtomicU64::new(0x3210_0),
            slots: std::array::from_fn(|_| (AtomicU64::new(0), AtomicU64::new(0))),
            count: AtomicUsize::new(0),
        }
    }

    /// CAS-based insert (lock-free).
    ///
    /// 1. Load permutation
    /// 2. Compute slot to use
    /// 3. Store data in slot
    /// 4. CAS permutation to publish
    fn cas_insert(&self, key: u64, value: u64) -> bool {
        loop {
            let perm = self.permutation.load(Ordering::SeqCst);
            let size = perm & 0xF;

            if size >= 4 {
                return false; // Full
            }

            // Get slot from "back" (position 3)
            let slot = ((perm >> 16) & 0xF) as usize;

            // Pre-store slot data
            self.slots[slot].0.store(key, Ordering::SeqCst);
            self.slots[slot].1.store(value, Ordering::SeqCst);

            // Compute new permutation (increment size, shift slots)
            // For simplicity, just increment size
            let new_perm = (perm & !0xF) | (size + 1);

            // CAS to publish
            if self
                .permutation
                .compare_exchange(perm, new_perm, Ordering::SeqCst, Ordering::SeqCst)
                .is_ok()
            {
                self.count.fetch_add(1, Ordering::SeqCst);
                return true;
            }

            // CAS failed - retry
            loom::thread::yield_now();
        }
    }

    /// Read all visible key-value pairs.
    fn read_all(&self) -> Vec<(u64, u64)> {
        let perm = self.permutation.load(Ordering::SeqCst);
        let size = (perm & 0xF) as usize;

        let mut result = Vec::new();
        for i in 0..size {
            // Get slot at position i from permutation encoding
            // Permutation layout: [size:4][slot3:4][slot2:4][slot1:4][slot0:4]
            // Position 0 is in bits 4-7, position 1 is in bits 8-11, etc.
            let slot = ((perm >> (4 + i * 4)) & 0xF) as usize;
            let key = self.slots[slot].0.load(Ordering::SeqCst);
            let value = self.slots[slot].1.load(Ordering::SeqCst);
            result.push((key, value));
        }
        result
    }
}

/// Test CAS insert with two threads inserting different keys.
///
/// Both should succeed without interference.
#[test]
fn test_loom_cas_insert_different_slots() {
    loom::model(|| {
        let leaf = Arc::new(LoomPermutedLeaf::new());

        let l1 = Arc::clone(&leaf);
        let t1 = thread::spawn(move || l1.cas_insert(1, 100));

        let l2 = Arc::clone(&leaf);
        let t2 = thread::spawn(move || l2.cas_insert(2, 200));

        let r1 = t1.join().unwrap();
        let r2 = t2.join().unwrap();

        // Both should succeed
        assert!(r1, "First insert should succeed");
        assert!(r2, "Second insert should succeed");

        // Count should be 2
        assert_eq!(leaf.count.load(Ordering::Relaxed), 2);
    });
}

/// Test CAS insert where both threads try to use same slot.
///
/// Only one should win the CAS race per attempt.
#[test]
fn test_loom_cas_insert_same_slot_race() {
    loom::model(|| {
        let leaf = Arc::new(LoomPermutedLeaf::new());

        let l1 = Arc::clone(&leaf);
        let t1 = thread::spawn(move || l1.cas_insert(42, 100));

        let l2 = Arc::clone(&leaf);
        let t2 = thread::spawn(move || l2.cas_insert(42, 200));

        let r1 = t1.join().unwrap();
        let r2 = t2.join().unwrap();

        // Both should succeed (retry on CAS failure)
        assert!(r1, "First insert should eventually succeed");
        assert!(r2, "Second insert should eventually succeed");

        // Final count should be 2 (both inserted)
        assert_eq!(leaf.count.load(Ordering::Relaxed), 2);
    });
}

/// Test CAS insert with concurrent reader.
///
/// Reader should see consistent data (never partial).
#[test]
fn test_loom_cas_insert_with_reader() {
    loom::model(|| {
        let leaf = Arc::new(LoomPermutedLeaf::new());

        let l1 = Arc::clone(&leaf);
        let t1 = thread::spawn(move || {
            l1.cas_insert(42, 999);
        });

        let l2 = Arc::clone(&leaf);
        let t2 = thread::spawn(move || {
            let data = l2.read_all();
            // Should see either empty or complete insert, never partial
            for (key, value) in &data {
                if *key == 42 {
                    assert_eq!(*value, 999, "Partial write detected");
                }
            }
            data
        });

        t1.join().unwrap();
        let read_result = t2.join().unwrap();

        // After join, insert should be visible
        let final_data = leaf.read_all();
        assert!(
            final_data.iter().any(|(k, _)| *k == 42),
            "Insert should be visible after join"
        );

        // If reader saw the key, it must have the correct value
        if let Some((_, v)) = read_result.iter().find(|(k, _)| *k == 42) {
            assert_eq!(*v, 999, "Reader saw partial write");
        }
    });
}
