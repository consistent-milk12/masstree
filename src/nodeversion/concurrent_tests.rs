//! Concurrent tests for NodeVersion.
//!
//! These tests verify the atomic operations work correctly under contention.
//! Guarded with `#[cfg(not(miri))]` because Miri doesn't support multi-threading well.

use super::*;
use std::sync::Arc;
use std::thread;
use std::time::Duration;

#[test]
fn test_concurrent_lock_unlock() {
    let version = Arc::new(NodeVersion::new(true));
    let iterations = 1000;
    let num_threads = 4;

    let handles: Vec<_> = (0..num_threads)
        .map(|_| {
            let v = Arc::clone(&version);
            thread::spawn(move || {
                for _ in 0..iterations {
                    let mut guard = v.lock();
                    guard.mark_insert();
                    // Guard drops here, releasing lock
                }
            })
        })
        .collect();

    for h in handles {
        h.join().expect("thread panicked");
    }

    // All threads completed successfully
    assert!(!version.is_locked());
    assert!(!version.is_dirty());
}

#[test]
fn test_stable_spins_on_dirty() {
    use std::sync::atomic::AtomicBool;

    let version = Arc::new(NodeVersion::new(true));
    let writer_done = Arc::new(AtomicBool::new(false));

    // Writer thread: hold lock with dirty bit for a while
    let v_writer = Arc::clone(&version);
    let done = Arc::clone(&writer_done);
    let writer = thread::spawn(move || {
        let mut guard = v_writer.lock();
        guard.mark_insert();

        // Hold lock for 50ms
        thread::sleep(Duration::from_millis(50));

        // Guard drops, clearing dirty bits
        drop(guard);
        done.store(true, Ordering::Release);
    });

    // Give writer time to acquire lock
    thread::sleep(Duration::from_millis(10));

    // Reader thread: stable() should spin until dirty clears
    let v_reader = Arc::clone(&version);
    let reader = thread::spawn(move || {
        let stable_v = v_reader.stable();
        // If we get here, dirty bits must be clear
        assert_eq!(stable_v & DIRTY_MASK, 0);
    });

    writer.join().expect("writer panicked");
    reader.join().expect("reader panicked");
}

#[test]
fn test_try_lock_for_timeout() {
    let version = Arc::new(NodeVersion::new(true));

    // Hold lock in another thread
    let v = Arc::clone(&version);
    let holder = thread::spawn(move || {
        let guard = v.lock();
        // Hold for 200ms
        thread::sleep(Duration::from_millis(200));
        drop(guard);
    });

    // Give holder time to acquire
    thread::sleep(Duration::from_millis(10));

    // Try to acquire with short timeout - should fail
    let result = version.try_lock_for(Duration::from_millis(50));
    assert!(result.is_none(), "should have timed out");

    holder.join().expect("holder panicked");

    // Now try_lock should succeed
    assert!(version.try_lock().is_some());
}

#[test]
fn test_guard_unlocks_on_panic() {
    let version = Arc::new(NodeVersion::new(true));
    let v = Arc::clone(&version);

    let handle = thread::spawn(move || {
        let mut guard = v.lock();
        guard.mark_insert();
        panic!("intentional panic");
    });

    // Wait for thread to panic
    let result = handle.join();
    assert!(result.is_err(), "should have panicked");

    // Lock should be released due to Drop during unwind
    assert!(!version.is_locked());

    // We can acquire the lock again
    let _guard = version.lock();
    assert!(version.is_locked());
}

#[test]
fn test_version_increment_under_contention() {
    let version = Arc::new(NodeVersion::new(true));
    let iterations = 100;
    let num_threads = 4;

    let initial = version.stable();

    let handles: Vec<_> = (0..num_threads)
        .map(|_| {
            let v = Arc::clone(&version);
            thread::spawn(move || {
                for _ in 0..iterations {
                    let mut guard = v.lock();
                    guard.mark_insert();
                }
            })
        })
        .collect();

    for h in handles {
        h.join().expect("thread panicked");
    }

    // Version should have changed
    assert!(version.has_changed(initial));
}

#[test]
fn test_try_lock_under_contention() {
    let version = Arc::new(NodeVersion::new(true));
    let success_count = Arc::new(std::sync::atomic::AtomicUsize::new(0));
    let num_threads = 8;

    let handles: Vec<_> = (0..num_threads)
        .map(|_| {
            let v = Arc::clone(&version);
            let count = Arc::clone(&success_count);
            thread::spawn(move || {
                for _ in 0..100 {
                    if let Some(mut guard) = v.try_lock() {
                        count.fetch_add(1, Ordering::Relaxed);
                        guard.mark_insert();
                    }
                    // Small sleep to increase contention
                    std::hint::spin_loop();
                }
            })
        })
        .collect();

    for h in handles {
        h.join().expect("thread panicked");
    }

    // At least some try_lock calls should have succeeded.
    // Note: On fast machines, all calls may succeed (no contention),
    // which is fine - it means the lock is working correctly.
    let successes = success_count.load(Ordering::Relaxed);
    assert!(successes > 0, "no try_lock succeeded");

    // Version should have changed after all the inserts
    assert!(version.has_changed(ISLEAF_BIT));
}
