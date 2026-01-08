use loom::sync::Arc;
use loom::sync::atomic::{AtomicU8, AtomicU64, AtomicUsize, Ordering};
use loom::thread;
use std::array as StdArray;

use super::WIDTH;

/// Simplified internode for loom testing.
///
/// Uses loom atomics to enable deterministic interleaving exploration.
struct LoomInternode {
    nkeys: AtomicU8,
    ikey0: [AtomicU64; WIDTH],
}

impl LoomInternode {
    fn new() -> Self {
        Self {
            nkeys: AtomicU8::new(0),
            ikey0: StdArray::from_fn(|_| AtomicU64::new(0)),
        }
    }

    fn nkeys(&self) -> usize {
        self.nkeys.load(Ordering::Acquire) as usize
    }

    fn set_nkeys(&self, n: u8) {
        self.nkeys.store(n, Ordering::Release);
    }

    fn ikey(&self, idx: usize) -> u64 {
        self.ikey0[idx].load(Ordering::Acquire)
    }

    fn set_ikey(&self, idx: usize, key: u64) {
        self.ikey0[idx].store(key, Ordering::Release);
    }

    fn find_insert_position(&self, insert_ikey: u64) -> usize {
        let n: usize = self.nkeys();

        for i in 0..n {
            if self.ikey(i) >= insert_ikey {
                return i;
            }
        }

        n
    }

    fn insert_key(&self, pos: usize, key: u64) {
        let n = self.nkeys();

        for i in (pos..n).rev() {
            let k = self.ikey(i);
            self.set_ikey(i + 1, k);
        }

        self.set_ikey(pos, key);
        self.set_nkeys((n + 1) as u8);
    }
}

#[test]
fn test_loom_find_position_concurrent_reads() {
    loom::model(|| {
        let node = Arc::new(LoomInternode::new());

        // Setup: insert keys 10, 20, 30
        node.set_ikey(0, 10);
        node.set_ikey(1, 20);
        node.set_ikey(2, 30);
        node.set_nkeys(3);

        let n1 = Arc::clone(&node);
        let t1 = thread::spawn(move || n1.find_insert_position(25));

        let n2 = Arc::clone(&node);
        let t2 = thread::spawn(move || n2.find_insert_position(15));

        let pos1 = t1.join().unwrap();
        let pos2 = t2.join().unwrap();

        assert!(pos1 <= 3, "pos1={} should be <= 3", pos1);
        assert!(pos2 <= 3, "pos2={} should be <= 3", pos2);
    });
}

#[test]
fn test_loom_find_position_during_insert() {
    loom::model(|| {
        let node = Arc::new(LoomInternode::new());

        node.set_ikey(0, 20);
        node.set_nkeys(1);

        let results = Arc::new(AtomicUsize::new(0));

        let n1 = Arc::clone(&node);
        let t1 = thread::spawn(move || {
            n1.insert_key(0, 10);
        });

        let n2 = Arc::clone(&node);
        let r2 = Arc::clone(&results);
        let t2 = thread::spawn(move || {
            let pos = n2.find_insert_position(15);
            r2.store(pos, Ordering::Relaxed);
        });

        t1.join().unwrap();
        t2.join().unwrap();

        let pos = results.load(Ordering::Relaxed);
        assert!(pos <= 2, "pos={} should be <= 2", pos);
    });
}

#[test]
fn test_loom_concurrent_reads_different_keys() {
    loom::model(|| {
        let node = Arc::new(LoomInternode::new());

        // Setup: insert keys 10, 20, 30, 40
        node.set_ikey(0, 10);
        node.set_ikey(1, 20);
        node.set_ikey(2, 30);
        node.set_ikey(3, 40);
        node.set_nkeys(4);

        let n1 = Arc::clone(&node);
        let t1 = thread::spawn(move || n1.find_insert_position(5)); // Before all

        let n2 = Arc::clone(&node);
        let t2 = thread::spawn(move || n2.find_insert_position(25)); // Middle

        let n3 = Arc::clone(&node);
        let t3 = thread::spawn(move || n3.find_insert_position(50)); // After all

        let pos1 = t1.join().unwrap();
        let pos2 = t2.join().unwrap();
        let pos3 = t3.join().unwrap();

        // All should get deterministic results (no concurrent writes)
        assert_eq!(pos1, 0, "5 should go at position 0");
        assert_eq!(pos2, 2, "25 should go at position 2");
        assert_eq!(pos3, 4, "50 should go at position 4");
    });
}
