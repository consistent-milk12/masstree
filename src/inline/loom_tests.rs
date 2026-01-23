//! Loom concurrency tests for inline suffix storage.
//!
//! Tests the concurrent read/write pattern that was causing data races:
//! - Writers clear slot metadata while holding the lock
//! - Readers access slot metadata without the lock (using OCC validation)

use std::array as StdArray;

use loom::sync::Arc;
use loom::sync::atomic::{AtomicU32, Ordering};
use loom::thread;

/// Packed slot metadata (offset in high 16 bits, len in low 16 bits).
/// Uses `u16::MAX` as offset to indicate empty slot.
const EMPTY_PACKED: u32 = (u16::MAX as u32) << 16;

/// Simplified inline suffix bag for loom testing.
///
/// Tests the core concurrent access pattern:
/// - Slot metadata is stored in AtomicU32
/// - Readers load with Acquire ordering
/// - Writers store with Release ordering
struct LoomInlineSuffixBag<const WIDTH: usize> {
    slots: [AtomicU32; WIDTH],
}

impl<const WIDTH: usize> LoomInlineSuffixBag<WIDTH> {
    fn new() -> Self {
        Self {
            slots: StdArray::from_fn(|_| AtomicU32::new(EMPTY_PACKED)),
        }
    }

    /// Check if slot has a suffix (reader operation).
    fn has_suffix(&self, slot: usize) -> bool {
        let packed = self.slots[slot].load(Ordering::Acquire);
        let offset = (packed >> 16) as u16;
        offset != u16::MAX
    }

    /// Get slot metadata (reader operation).
    /// Returns (offset, len) or None if empty.
    fn get_meta(&self, slot: usize) -> Option<(u16, u16)> {
        let packed = self.slots[slot].load(Ordering::Acquire);
        let offset = (packed >> 16) as u16;
        let len = packed as u16;

        if offset == u16::MAX {
            None
        } else {
            Some((offset, len))
        }
    }

    /// Assign suffix metadata (writer operation - requires lock).
    fn assign(&self, slot: usize, offset: u16, len: u16) {
        let packed = ((offset as u32) << 16) | (len as u32);
        self.slots[slot].store(packed, Ordering::Release);
    }

    /// Clear slot (writer operation - requires lock).
    fn clear(&self, slot: usize) {
        self.slots[slot].store(EMPTY_PACKED, Ordering::Release);
    }
}

/// Test concurrent reads don't race with each other.
#[test]
fn test_loom_concurrent_reads() {
    loom::model(|| {
        let bag: Arc<LoomInlineSuffixBag<4>> = Arc::new(LoomInlineSuffixBag::new());

        // Setup: assign suffixes to slots 0 and 1
        bag.assign(0, 0, 10); // offset=0, len=10
        bag.assign(1, 10, 20); // offset=10, len=20

        let b1 = Arc::clone(&bag);
        let b2 = Arc::clone(&bag);

        let t1 = thread::spawn(move || {
            let has0 = b1.has_suffix(0);
            let has1 = b1.has_suffix(1);
            (has0, has1)
        });

        let t2 = thread::spawn(move || {
            let meta0 = b2.get_meta(0);
            let meta1 = b2.get_meta(1);
            (meta0, meta1)
        });

        let (has0, has1) = t1.join().unwrap();
        let (meta0, meta1) = t2.join().unwrap();

        // Both slots should have suffixes
        assert!(has0, "slot 0 should have suffix");
        assert!(has1, "slot 1 should have suffix");
        assert_eq!(meta0, Some((0, 10)));
        assert_eq!(meta1, Some((10, 20)));
    });
}

/// Test reader observes consistent state during writer clear.
///
/// This is the exact pattern that was causing the data race:
/// - Writer clears a slot (under lock)
/// - Reader reads the slot (without lock, uses OCC validation)
#[test]
fn test_loom_read_during_clear() {
    loom::model(|| {
        let bag: Arc<LoomInlineSuffixBag<4>> = Arc::new(LoomInlineSuffixBag::new());

        // Setup: slot 0 has a suffix
        bag.assign(0, 0, 10);

        let b_writer = Arc::clone(&bag);
        let b_reader = Arc::clone(&bag);

        // Writer clears slot 0
        let writer = thread::spawn(move || {
            b_writer.clear(0);
        });

        // Reader reads slot 0
        let reader = thread::spawn(move || {
            let meta = b_reader.get_meta(0);
            // Reader sees either:
            // - Some((0, 10)) - before clear
            // - None - after clear
            // Both are valid, no torn reads
            meta
        });

        writer.join().unwrap();
        let result = reader.join().unwrap();

        // Result is either the original value or None (cleared)
        assert!(
            result == Some((0, 10)) || result.is_none(),
            "unexpected result: {:?}",
            result
        );
    });
}

/// Test reader observes consistent state during writer assign.
#[test]
fn test_loom_read_during_assign() {
    loom::model(|| {
        let bag: Arc<LoomInlineSuffixBag<4>> = Arc::new(LoomInlineSuffixBag::new());

        let b_writer = Arc::clone(&bag);
        let b_reader = Arc::clone(&bag);

        // Writer assigns to slot 0
        let writer = thread::spawn(move || {
            b_writer.assign(0, 100, 50);
        });

        // Reader reads slot 0
        let reader = thread::spawn(move || b_reader.get_meta(0));

        writer.join().unwrap();
        let result = reader.join().unwrap();

        // Result is either:
        // - None (before assign)
        // - Some((100, 50)) (after assign)
        // No torn reads allowed
        assert!(
            result.is_none() || result == Some((100, 50)),
            "unexpected result: {:?}",
            result
        );
    });
}

/// Test multiple writers and readers.
#[test]
fn test_loom_multiple_writers_readers() {
    loom::model(|| {
        let bag: Arc<LoomInlineSuffixBag<4>> = Arc::new(LoomInlineSuffixBag::new());

        // Setup
        bag.assign(0, 0, 10);
        bag.assign(1, 10, 20);

        let b1 = Arc::clone(&bag);
        let b2 = Arc::clone(&bag);
        let b3 = Arc::clone(&bag);

        // Writer 1: clear slot 0
        let w1 = thread::spawn(move || {
            b1.clear(0);
        });

        // Writer 2: assign to slot 1
        let w2 = thread::spawn(move || {
            b2.assign(1, 100, 5);
        });

        // Reader: read both slots
        let reader = thread::spawn(move || {
            let m0 = b3.get_meta(0);
            let m1 = b3.get_meta(1);
            (m0, m1)
        });

        w1.join().unwrap();
        w2.join().unwrap();
        let (m0, m1) = reader.join().unwrap();

        // Slot 0: either original or cleared
        assert!(
            m0 == Some((0, 10)) || m0.is_none(),
            "slot 0 unexpected: {:?}",
            m0
        );

        // Slot 1: either original or updated
        assert!(
            m1 == Some((10, 20)) || m1 == Some((100, 5)),
            "slot 1 unexpected: {:?}",
            m1
        );
    });
}

/// Test the full read-clear-read pattern.
///
/// Simulates:
/// 1. Reader starts reading (may see old value)
/// 2. Writer clears slot
/// 3. Reader reads again (may see new value)
#[test]
fn test_loom_read_clear_read() {
    loom::model(|| {
        let bag: Arc<LoomInlineSuffixBag<4>> = Arc::new(LoomInlineSuffixBag::new());

        // Setup
        bag.assign(0, 0, 10);

        let b_writer = Arc::clone(&bag);
        let b_reader = Arc::clone(&bag);

        let writer = thread::spawn(move || {
            b_writer.clear(0);
        });

        let reader = thread::spawn(move || {
            let before = b_reader.get_meta(0);
            // Yield to allow interleaving
            loom::thread::yield_now();
            let after = b_reader.get_meta(0);
            (before, after)
        });

        writer.join().unwrap();
        let (before, after) = reader.join().unwrap();

        // Valid combinations:
        // - (Some, Some): reader ran entirely before writer
        // - (Some, None): writer ran between reads
        // - (None, None): writer ran before first read
        // Invalid: (None, Some) - can't see cleared then uncleared
        let valid = matches!(
            (before, after),
            (Some((0, 10)), Some((0, 10))) | (Some((0, 10)), None) | (None, None)
        );

        assert!(
            valid,
            "invalid state transition: before={:?}, after={:?}",
            before, after
        );
    });
}
