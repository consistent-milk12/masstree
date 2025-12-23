//! Rigorous stress tests for MassTree concurrent operations.
//!
//! These tests are designed to expose race conditions through:
//! - Multi-layer keys (>8 bytes) to test layer traversal
//! - High thread counts (8, 16+ threads)
//! - Large key volumes (10k+ keys)
//! - Various key patterns (sequential, random, interleaved)
//! - Mixed read/write workloads
//! - Repeated runs for intermittent bugs
//!
//! Run all stress tests:
//! ```bash
//! cargo nextest run --features mimalloc --test stress_tests --release
//! ```
//!
//! Run specific category:
//! ```bash
//! cargo nextest run --features mimalloc --test stress_tests multilayer --release
//! cargo nextest run --features mimalloc --test stress_tests high_thread --release
//! ```

#![allow(clippy::pedantic)]
#![expect(clippy::unwrap_used)]
#![allow(clippy::panic)]

mod common;

use masstree::{MassTree, get_debug_counters, reset_debug_counters};
use std::collections::HashSet;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::thread;

// =============================================================================
// Test Configuration
// =============================================================================

/// Report debug counters if any B-link issues detected
fn report_debug_counters(test_name: &str) {
    let (blink_should_follow, search_not_found) = get_debug_counters();
    if blink_should_follow > 0 || search_not_found > 0 {
        eprintln!(
            "\n*** {} - DIAGNOSTIC ***\n\
             B-link followed: {} times\n\
             Search NotFound: {} times\n",
            test_name, blink_should_follow, search_not_found
        );
    }
}

/// Verify all keys are findable, panic with details if any missing
fn verify_all_keys<F>(tree: &MassTree<u64>, key_gen: F, count: usize, test_name: &str)
where
    F: Fn(usize) -> Vec<u8>,
{
    let guard = tree.guard();
    let mut missing = Vec::new();

    for i in 0..count {
        let key = key_gen(i);
        if tree.get_with_guard(&key, &guard).is_none() {
            missing.push(i);
        }
    }

    if !missing.is_empty() {
        let sample: Vec<_> = missing.iter().take(20).collect();
        panic!(
            "{}: Missing {} keys (showing first 20): {:?}\n\
             tree.len()={}, expected={}",
            test_name,
            missing.len(),
            sample,
            tree.len(),
            count
        );
    }
}

// =============================================================================
// MULTI-LAYER KEY TESTS (>8 bytes)
// =============================================================================

/// 16-byte keys: 2 layers
#[test]
fn multilayer_16byte_keys_4_threads() {
    common::init_tracing();
    reset_debug_counters();

    const NUM_THREADS: usize = 4;
    const KEYS_PER_THREAD: usize = 500;
    const TOTAL_KEYS: usize = NUM_THREADS * KEYS_PER_THREAD;

    let tree = Arc::new(MassTree::<u64>::new());
    let verify_failures = Arc::new(AtomicUsize::new(0));

    let handles: Vec<_> = (0..NUM_THREADS)
        .map(|t| {
            let tree = Arc::clone(&tree);
            let verify_failures = Arc::clone(&verify_failures);
            thread::spawn(move || {
                let guard = tree.guard();
                for i in 0..KEYS_PER_THREAD {
                    // 16-byte key: "T00___0000000000" format (exactly 16 bytes)
                    let key = format!("T{:02}___{:010}", t, i);
                    debug_assert_eq!(key.len(), 16);

                    let _ = tree.insert_with_guard(key.as_bytes(), (t * 10000 + i) as u64, &guard);

                    // Immediate verification
                    if tree.get_with_guard(key.as_bytes(), &guard).is_none() {
                        verify_failures.fetch_add(1, Ordering::Relaxed);
                    }
                }
            })
        })
        .collect();

    for h in handles {
        h.join().unwrap();
    }

    let fail_count = verify_failures.load(Ordering::Relaxed);
    if fail_count > 0 {
        eprintln!(
            "multilayer_16byte: {} immediate verification failures",
            fail_count
        );
    }

    // Final verification - use same format as insert!
    let guard = tree.guard();
    let mut missing = Vec::new();
    for t in 0..NUM_THREADS {
        for i in 0..KEYS_PER_THREAD {
            let key = format!("T{:02}___{:010}", t, i);
            if tree.get_with_guard(key.as_bytes(), &guard).is_none() {
                missing.push((t, i));
            }
        }
    }

    report_debug_counters("multilayer_16byte_keys_4_threads");

    if !missing.is_empty() {
        panic!(
            "multilayer_16byte: Missing {} keys: {:?}",
            missing.len(),
            &missing[..missing.len().min(20)]
        );
    }

    assert_eq!(tree.len(), TOTAL_KEYS);
}

/// 24-byte keys: 3 layers
#[test]
fn multilayer_24byte_keys_4_threads() {
    common::init_tracing();
    reset_debug_counters();

    const NUM_THREADS: usize = 4;
    const KEYS_PER_THREAD: usize = 500;
    const TOTAL_KEYS: usize = NUM_THREADS * KEYS_PER_THREAD;

    let tree = Arc::new(MassTree::<u64>::new());
    let verify_failures = Arc::new(AtomicUsize::new(0));

    let handles: Vec<_> = (0..NUM_THREADS)
        .map(|t| {
            let tree = Arc::clone(&tree);
            let verify_failures = Arc::clone(&verify_failures);
            thread::spawn(move || {
                let guard = tree.guard();
                for i in 0..KEYS_PER_THREAD {
                    // 24-byte key: "thread_XX_key_0000000000" format
                    let key = format!("thread_{:02}_key_{:010}", t, i);
                    debug_assert_eq!(key.len(), 24);

                    let _ = tree.insert_with_guard(key.as_bytes(), (t * 10000 + i) as u64, &guard);

                    if tree.get_with_guard(key.as_bytes(), &guard).is_none() {
                        verify_failures.fetch_add(1, Ordering::Relaxed);
                    }
                }
            })
        })
        .collect();

    for h in handles {
        h.join().unwrap();
    }

    let fail_count = verify_failures.load(Ordering::Relaxed);
    if fail_count > 0 {
        eprintln!(
            "multilayer_24byte: {} immediate verification failures",
            fail_count
        );
    }

    let guard = tree.guard();
    let mut missing = Vec::new();
    for t in 0..NUM_THREADS {
        for i in 0..KEYS_PER_THREAD {
            let key = format!("thread_{:02}_key_{:010}", t, i);
            if tree.get_with_guard(key.as_bytes(), &guard).is_none() {
                missing.push((t, i));
            }
        }
    }

    report_debug_counters("multilayer_24byte_keys_4_threads");

    if !missing.is_empty() {
        panic!(
            "multilayer_24byte: Missing {} keys: {:?}",
            missing.len(),
            &missing[..missing.len().min(20)]
        );
    }

    assert_eq!(tree.len(), TOTAL_KEYS);
}

/// 32-byte keys: 4 layers
#[test]
fn multilayer_32byte_keys_4_threads() {
    common::init_tracing();
    reset_debug_counters();

    const NUM_THREADS: usize = 4;
    const KEYS_PER_THREAD: usize = 500;
    const TOTAL_KEYS: usize = NUM_THREADS * KEYS_PER_THREAD;

    let tree = Arc::new(MassTree::<u64>::new());
    let verify_failures = Arc::new(AtomicUsize::new(0));

    let handles: Vec<_> = (0..NUM_THREADS)
        .map(|t| {
            let tree = Arc::clone(&tree);
            let verify_failures = Arc::clone(&verify_failures);
            thread::spawn(move || {
                let guard = tree.guard();
                for i in 0..KEYS_PER_THREAD {
                    // 32-byte key
                    let key = format!("prefix_{:02}_middle_{:010}_ends", t, i);
                    debug_assert_eq!(key.len(), 32);

                    let _ = tree.insert_with_guard(key.as_bytes(), (t * 10000 + i) as u64, &guard);

                    if tree.get_with_guard(key.as_bytes(), &guard).is_none() {
                        verify_failures.fetch_add(1, Ordering::Relaxed);
                    }
                }
            })
        })
        .collect();

    for h in handles {
        h.join().unwrap();
    }

    let fail_count = verify_failures.load(Ordering::Relaxed);
    if fail_count > 0 {
        eprintln!(
            "multilayer_32byte: {} immediate verification failures",
            fail_count
        );
    }

    // Final verification - use same format as insert!
    let guard = tree.guard();
    let mut missing = Vec::new();
    for t in 0..NUM_THREADS {
        for i in 0..KEYS_PER_THREAD {
            let key = format!("prefix_{:02}_middle_{:010}_ends", t, i);
            if tree.get_with_guard(key.as_bytes(), &guard).is_none() {
                missing.push((t, i));
            }
        }
    }

    report_debug_counters("multilayer_32byte_keys_4_threads");

    if !missing.is_empty() {
        panic!(
            "multilayer_32byte: Missing {} keys: {:?}",
            missing.len(),
            &missing[..missing.len().min(20)]
        );
    }

    assert_eq!(tree.len(), TOTAL_KEYS);
}

/// Mixed key lengths: stress layer boundary handling
#[test]
fn multilayer_mixed_lengths() {
    common::init_tracing();
    reset_debug_counters();

    const NUM_THREADS: usize = 4;
    const KEYS_PER_THREAD: usize = 400;

    let tree = Arc::new(MassTree::<u64>::new());
    let verify_failures = Arc::new(AtomicUsize::new(0));

    let handles: Vec<_> = (0..NUM_THREADS)
        .map(|t| {
            let tree = Arc::clone(&tree);
            let verify_failures = Arc::clone(&verify_failures);
            thread::spawn(move || {
                let guard = tree.guard();
                for i in 0..KEYS_PER_THREAD {
                    // Rotate through different key lengths: 8, 16, 24, 32 bytes
                    let key = match i % 4 {
                        0 => format!("{:08}", t * 10000 + i), // 8 bytes (1 layer)
                        1 => format!("k{:02}_{:010}", t, i),  // 16 bytes (2 layers)
                        2 => format!("key_{:02}_middle_{:010}", t, i), // 24 bytes (3 layers)
                        _ => format!("prefix_{:02}_mid_{:010}_end!", t, i), // 32 bytes (4 layers)
                    };

                    let _ = tree.insert_with_guard(key.as_bytes(), (t * 10000 + i) as u64, &guard);

                    if tree.get_with_guard(key.as_bytes(), &guard).is_none() {
                        verify_failures.fetch_add(1, Ordering::Relaxed);
                    }
                }
            })
        })
        .collect();

    for h in handles {
        h.join().unwrap();
    }

    let fail_count = verify_failures.load(Ordering::Relaxed);
    report_debug_counters("multilayer_mixed_lengths");

    if fail_count > 0 {
        panic!(
            "multilayer_mixed: {} immediate verification failures",
            fail_count
        );
    }

    assert_eq!(tree.len(), NUM_THREADS * KEYS_PER_THREAD);
}

// =============================================================================
// HIGH THREAD COUNT TESTS
// =============================================================================

#[test]
fn high_thread_8_threads_8byte_keys() {
    common::init_tracing();
    reset_debug_counters();

    const NUM_THREADS: usize = 8;
    const KEYS_PER_THREAD: usize = 500;
    const TOTAL_KEYS: usize = NUM_THREADS * KEYS_PER_THREAD;

    let tree = Arc::new(MassTree::<u64>::new());
    let verify_failures = Arc::new(AtomicUsize::new(0));

    let handles: Vec<_> = (0..NUM_THREADS)
        .map(|t| {
            let tree = Arc::clone(&tree);
            let verify_failures = Arc::clone(&verify_failures);
            thread::spawn(move || {
                let guard = tree.guard();
                for i in 0..KEYS_PER_THREAD {
                    let key_val = (t * 10000 + i) as u64;
                    let key = key_val.to_be_bytes();

                    let _ = tree.insert_with_guard(&key, key_val, &guard);

                    if tree.get_with_guard(&key, &guard).is_none() {
                        verify_failures.fetch_add(1, Ordering::Relaxed);
                    }
                }
            })
        })
        .collect();

    for h in handles {
        h.join().unwrap();
    }

    let fail_count = verify_failures.load(Ordering::Relaxed);
    report_debug_counters("high_thread_8_threads_8byte_keys");

    if fail_count > 0 {
        panic!(
            "high_thread_8: {} immediate verification failures",
            fail_count
        );
    }

    verify_all_keys(
        &tree,
        |i| {
            let t = i / KEYS_PER_THREAD;
            let idx = i % KEYS_PER_THREAD;
            ((t * 10000 + idx) as u64).to_be_bytes().to_vec()
        },
        TOTAL_KEYS,
        "high_thread_8_threads",
    );

    assert_eq!(tree.len(), TOTAL_KEYS);
}

#[test]
fn high_thread_16_threads_8byte_keys() {
    common::init_tracing();
    reset_debug_counters();

    const NUM_THREADS: usize = 16;
    const KEYS_PER_THREAD: usize = 250;
    const TOTAL_KEYS: usize = NUM_THREADS * KEYS_PER_THREAD;

    let tree = Arc::new(MassTree::<u64>::new());
    let verify_failures = Arc::new(AtomicUsize::new(0));

    let handles: Vec<_> = (0..NUM_THREADS)
        .map(|t| {
            let tree = Arc::clone(&tree);
            let verify_failures = Arc::clone(&verify_failures);
            thread::spawn(move || {
                let guard = tree.guard();
                for i in 0..KEYS_PER_THREAD {
                    let key_val = (t * 10000 + i) as u64;
                    let key = key_val.to_be_bytes();

                    let _ = tree.insert_with_guard(&key, key_val, &guard);

                    if tree.get_with_guard(&key, &guard).is_none() {
                        verify_failures.fetch_add(1, Ordering::Relaxed);
                    }
                }
            })
        })
        .collect();

    for h in handles {
        h.join().unwrap();
    }

    let fail_count = verify_failures.load(Ordering::Relaxed);
    report_debug_counters("high_thread_16_threads_8byte_keys");

    if fail_count > 0 {
        panic!(
            "high_thread_16: {} immediate verification failures",
            fail_count
        );
    }

    assert_eq!(tree.len(), TOTAL_KEYS);
}

#[test]
fn high_thread_8_threads_24byte_keys() {
    common::init_tracing();
    reset_debug_counters();

    const NUM_THREADS: usize = 8;
    const KEYS_PER_THREAD: usize = 400;
    const TOTAL_KEYS: usize = NUM_THREADS * KEYS_PER_THREAD;

    let tree = Arc::new(MassTree::<u64>::new());
    let verify_failures = Arc::new(AtomicUsize::new(0));

    let handles: Vec<_> = (0..NUM_THREADS)
        .map(|t| {
            let tree = Arc::clone(&tree);
            let verify_failures = Arc::clone(&verify_failures);
            thread::spawn(move || {
                let guard = tree.guard();
                for i in 0..KEYS_PER_THREAD {
                    let key = format!("thread_{:02}_key_{:010}", t, i);
                    debug_assert_eq!(key.len(), 24);

                    let _ = tree.insert_with_guard(key.as_bytes(), (t * 10000 + i) as u64, &guard);

                    if tree.get_with_guard(key.as_bytes(), &guard).is_none() {
                        verify_failures.fetch_add(1, Ordering::Relaxed);
                    }
                }
            })
        })
        .collect();

    for h in handles {
        h.join().unwrap();
    }

    let fail_count = verify_failures.load(Ordering::Relaxed);
    report_debug_counters("high_thread_8_threads_24byte_keys");

    if fail_count > 0 {
        panic!(
            "high_thread_8_24byte: {} immediate verification failures",
            fail_count
        );
    }

    assert_eq!(tree.len(), TOTAL_KEYS);
}

// =============================================================================
// LARGE VOLUME TESTS
// =============================================================================

#[test]
fn large_volume_10k_keys_4_threads() {
    common::init_tracing();
    reset_debug_counters();

    const NUM_THREADS: usize = 4;
    const KEYS_PER_THREAD: usize = 2500;
    const TOTAL_KEYS: usize = NUM_THREADS * KEYS_PER_THREAD;

    let tree = Arc::new(MassTree::<u64>::new());
    let verify_failures = Arc::new(AtomicUsize::new(0));

    let handles: Vec<_> = (0..NUM_THREADS)
        .map(|t| {
            let tree = Arc::clone(&tree);
            let verify_failures = Arc::clone(&verify_failures);
            thread::spawn(move || {
                let guard = tree.guard();
                for i in 0..KEYS_PER_THREAD {
                    let key_val = (t * 100000 + i) as u64;
                    let key = key_val.to_be_bytes();

                    let _ = tree.insert_with_guard(&key, key_val, &guard);

                    if tree.get_with_guard(&key, &guard).is_none() {
                        verify_failures.fetch_add(1, Ordering::Relaxed);
                    }
                }
            })
        })
        .collect();

    for h in handles {
        h.join().unwrap();
    }

    let fail_count = verify_failures.load(Ordering::Relaxed);
    report_debug_counters("large_volume_10k_keys_4_threads");

    if fail_count > 0 {
        panic!(
            "large_volume_10k: {} immediate verification failures",
            fail_count
        );
    }

    assert_eq!(tree.len(), TOTAL_KEYS);
}

#[test]
fn large_volume_20k_keys_8_threads() {
    common::init_tracing();
    reset_debug_counters();

    const NUM_THREADS: usize = 8;
    const KEYS_PER_THREAD: usize = 2500;
    const TOTAL_KEYS: usize = NUM_THREADS * KEYS_PER_THREAD;

    let tree = Arc::new(MassTree::<u64>::new());
    let verify_failures = Arc::new(AtomicUsize::new(0));

    let handles: Vec<_> = (0..NUM_THREADS)
        .map(|t| {
            let tree = Arc::clone(&tree);
            let verify_failures = Arc::clone(&verify_failures);
            thread::spawn(move || {
                let guard = tree.guard();
                for i in 0..KEYS_PER_THREAD {
                    let key_val = (t * 100000 + i) as u64;
                    let key = key_val.to_be_bytes();

                    let _ = tree.insert_with_guard(&key, key_val, &guard);

                    if tree.get_with_guard(&key, &guard).is_none() {
                        verify_failures.fetch_add(1, Ordering::Relaxed);
                    }
                }
            })
        })
        .collect();

    for h in handles {
        h.join().unwrap();
    }

    let fail_count = verify_failures.load(Ordering::Relaxed);
    report_debug_counters("large_volume_20k_keys_8_threads");

    if fail_count > 0 {
        panic!(
            "large_volume_20k: {} immediate verification failures",
            fail_count
        );
    }

    assert_eq!(tree.len(), TOTAL_KEYS);
}

// =============================================================================
// KEY PATTERN TESTS
// =============================================================================

/// Sequential keys cause maximum splits
#[test]
fn pattern_sequential_keys_high_splits() {
    common::init_tracing();
    reset_debug_counters();

    const NUM_THREADS: usize = 4;
    const KEYS_PER_THREAD: usize = 500;

    let tree = Arc::new(MassTree::<u64>::new());
    let verify_failures = Arc::new(AtomicUsize::new(0));

    let handles: Vec<_> = (0..NUM_THREADS)
        .map(|t| {
            let tree = Arc::clone(&tree);
            let verify_failures = Arc::clone(&verify_failures);
            thread::spawn(move || {
                let guard = tree.guard();
                // Interleaved sequential: thread 0 gets 0,4,8..., thread 1 gets 1,5,9...
                for i in 0..KEYS_PER_THREAD {
                    let key_val = (i * NUM_THREADS + t) as u64;
                    let key = key_val.to_be_bytes();

                    let _ = tree.insert_with_guard(&key, key_val, &guard);

                    if tree.get_with_guard(&key, &guard).is_none() {
                        verify_failures.fetch_add(1, Ordering::Relaxed);
                    }
                }
            })
        })
        .collect();

    for h in handles {
        h.join().unwrap();
    }

    let fail_count = verify_failures.load(Ordering::Relaxed);
    report_debug_counters("pattern_sequential_keys_high_splits");

    if fail_count > 0 {
        panic!(
            "pattern_sequential: {} immediate verification failures",
            fail_count
        );
    }

    assert_eq!(tree.len(), NUM_THREADS * KEYS_PER_THREAD);
}

/// Reverse sequential keys
#[test]
fn pattern_reverse_sequential() {
    common::init_tracing();
    reset_debug_counters();

    const NUM_THREADS: usize = 4;
    const KEYS_PER_THREAD: usize = 500;
    const MAX_KEY: usize = NUM_THREADS * KEYS_PER_THREAD;

    let tree = Arc::new(MassTree::<u64>::new());
    let verify_failures = Arc::new(AtomicUsize::new(0));

    let handles: Vec<_> = (0..NUM_THREADS)
        .map(|t| {
            let tree = Arc::clone(&tree);
            let verify_failures = Arc::clone(&verify_failures);
            thread::spawn(move || {
                let guard = tree.guard();
                for i in 0..KEYS_PER_THREAD {
                    // Reverse order
                    let key_val = (MAX_KEY - 1 - (i * NUM_THREADS + t)) as u64;
                    let key = key_val.to_be_bytes();

                    let _ = tree.insert_with_guard(&key, key_val, &guard);

                    if tree.get_with_guard(&key, &guard).is_none() {
                        verify_failures.fetch_add(1, Ordering::Relaxed);
                    }
                }
            })
        })
        .collect();

    for h in handles {
        h.join().unwrap();
    }

    let fail_count = verify_failures.load(Ordering::Relaxed);
    report_debug_counters("pattern_reverse_sequential");

    if fail_count > 0 {
        panic!(
            "pattern_reverse: {} immediate verification failures",
            fail_count
        );
    }

    assert_eq!(tree.len(), NUM_THREADS * KEYS_PER_THREAD);
}

/// Pseudo-random keys using simple LCG
#[test]
fn pattern_pseudorandom_keys() {
    common::init_tracing();
    reset_debug_counters();

    const NUM_THREADS: usize = 4;
    const KEYS_PER_THREAD: usize = 500;

    let tree = Arc::new(MassTree::<u64>::new());
    let verify_failures = Arc::new(AtomicUsize::new(0));
    let inserted_keys = Arc::new(std::sync::Mutex::new(HashSet::new()));

    let handles: Vec<_> = (0..NUM_THREADS)
        .map(|t| {
            let tree = Arc::clone(&tree);
            let verify_failures = Arc::clone(&verify_failures);
            let inserted_keys = Arc::clone(&inserted_keys);
            thread::spawn(move || {
                let guard = tree.guard();
                // Simple LCG: x = (a*x + c) mod m
                let mut rng_state = (t as u64 + 1).wrapping_mul(0x9E3779B97F4A7C15);
                let mut thread_keys = Vec::with_capacity(KEYS_PER_THREAD);

                for _ in 0..KEYS_PER_THREAD {
                    rng_state = rng_state
                        .wrapping_mul(0x5851F42D4C957F2D)
                        .wrapping_add(t as u64);
                    let key_val = rng_state;
                    let key = key_val.to_be_bytes();

                    let _ = tree.insert_with_guard(&key, key_val, &guard);
                    thread_keys.push(key_val);

                    if tree.get_with_guard(&key, &guard).is_none() {
                        verify_failures.fetch_add(1, Ordering::Relaxed);
                    }
                }

                // Store keys for verification
                inserted_keys.lock().unwrap().extend(thread_keys);
            })
        })
        .collect();

    for h in handles {
        h.join().unwrap();
    }

    let fail_count = verify_failures.load(Ordering::Relaxed);
    report_debug_counters("pattern_pseudorandom_keys");

    if fail_count > 0 {
        panic!(
            "pattern_pseudorandom: {} immediate verification failures",
            fail_count
        );
    }

    // Verify all inserted keys
    let guard = tree.guard();
    let keys = inserted_keys.lock().unwrap();
    let mut missing = 0;
    for &key_val in keys.iter() {
        if tree
            .get_with_guard(&key_val.to_be_bytes(), &guard)
            .is_none()
        {
            missing += 1;
        }
    }

    if missing > 0 {
        panic!(
            "pattern_pseudorandom: {} keys missing in final verification",
            missing
        );
    }

    assert_eq!(tree.len(), keys.len());
}

/// Same key prefix, different suffixes (tests layer contention)
#[test]
fn pattern_shared_prefix() {
    common::init_tracing();
    reset_debug_counters();

    const NUM_THREADS: usize = 4;
    const KEYS_PER_THREAD: usize = 500;
    const TOTAL_KEYS: usize = NUM_THREADS * KEYS_PER_THREAD;

    let tree = Arc::new(MassTree::<u64>::new());
    let verify_failures = Arc::new(AtomicUsize::new(0));

    let handles: Vec<_> = (0..NUM_THREADS)
        .map(|t| {
            let tree = Arc::clone(&tree);
            let verify_failures = Arc::clone(&verify_failures);
            thread::spawn(move || {
                let guard = tree.guard();
                for i in 0..KEYS_PER_THREAD {
                    // All keys share "prefix__" (8 bytes), differ in suffix
                    let key = format!("prefix__{:02}_{:07}", t, i);
                    debug_assert_eq!(key.len(), 18);

                    let _ = tree.insert_with_guard(key.as_bytes(), (t * 10000 + i) as u64, &guard);

                    if tree.get_with_guard(key.as_bytes(), &guard).is_none() {
                        verify_failures.fetch_add(1, Ordering::Relaxed);
                    }
                }
            })
        })
        .collect();

    for h in handles {
        h.join().unwrap();
    }

    let fail_count = verify_failures.load(Ordering::Relaxed);
    report_debug_counters("pattern_shared_prefix");

    if fail_count > 0 {
        panic!(
            "pattern_shared_prefix: {} immediate verification failures",
            fail_count
        );
    }

    assert_eq!(tree.len(), TOTAL_KEYS);
}

// =============================================================================
// MIXED READ/WRITE TESTS
// =============================================================================

/// Heavy read load during writes
#[test]
fn mixed_heavy_reads_during_writes() {
    common::init_tracing();
    reset_debug_counters();

    const NUM_WRITERS: usize = 2;
    const NUM_READERS: usize = 6;
    const KEYS_PER_WRITER: usize = 500;
    const TOTAL_KEYS: usize = NUM_WRITERS * KEYS_PER_WRITER;

    let tree = Arc::new(MassTree::<u64>::new());
    let write_complete = Arc::new(AtomicUsize::new(0));
    let read_success = Arc::new(AtomicUsize::new(0));

    // Spawn writers
    let writer_handles: Vec<_> = (0..NUM_WRITERS)
        .map(|t| {
            let tree = Arc::clone(&tree);
            let write_complete = Arc::clone(&write_complete);
            thread::spawn(move || {
                let guard = tree.guard();
                for i in 0..KEYS_PER_WRITER {
                    let key_val = (t * 10000 + i) as u64;
                    let key = key_val.to_be_bytes();
                    let _ = tree.insert_with_guard(&key, key_val, &guard);
                }
                write_complete.fetch_add(1, Ordering::Release);
            })
        })
        .collect();

    // Spawn readers that continuously read
    let reader_handles: Vec<_> = (0..NUM_READERS)
        .map(|_| {
            let tree = Arc::clone(&tree);
            let write_complete = Arc::clone(&write_complete);
            let read_success = Arc::clone(&read_success);
            thread::spawn(move || {
                let guard = tree.guard();
                let mut local_success = 0;

                // Keep reading until all writers are done
                while write_complete.load(Ordering::Acquire) < NUM_WRITERS {
                    for t in 0..NUM_WRITERS {
                        for i in 0..KEYS_PER_WRITER {
                            let key_val = (t * 10000 + i) as u64;
                            let key = key_val.to_be_bytes();
                            if tree.get_with_guard(&key, &guard).is_some() {
                                local_success += 1;
                            }
                        }
                    }
                }

                read_success.fetch_add(local_success, Ordering::Relaxed);
            })
        })
        .collect();

    for h in writer_handles {
        h.join().unwrap();
    }
    for h in reader_handles {
        h.join().unwrap();
    }

    report_debug_counters("mixed_heavy_reads_during_writes");

    // Final verification - all keys must be present
    let guard = tree.guard();
    let mut missing = Vec::new();
    for t in 0..NUM_WRITERS {
        for i in 0..KEYS_PER_WRITER {
            let key_val = (t * 10000 + i) as u64;
            let key = key_val.to_be_bytes();
            if tree.get_with_guard(&key, &guard).is_none() {
                missing.push(key_val);
            }
        }
    }

    if !missing.is_empty() {
        panic!(
            "mixed_heavy_reads: {} keys missing: {:?}",
            missing.len(),
            &missing[..missing.len().min(20)]
        );
    }

    assert_eq!(tree.len(), TOTAL_KEYS);
}

/// Continuous read/write with verification
#[test]
fn mixed_continuous_readwrite() {
    common::init_tracing();
    reset_debug_counters();

    const NUM_THREADS: usize = 8;
    const OPS_PER_THREAD: usize = 500;

    let tree = Arc::new(MassTree::<u64>::new());
    let verify_failures = Arc::new(AtomicUsize::new(0));

    let handles: Vec<_> = (0..NUM_THREADS)
        .map(|t| {
            let tree = Arc::clone(&tree);
            let verify_failures = Arc::clone(&verify_failures);
            thread::spawn(move || {
                let guard = tree.guard();

                for i in 0..OPS_PER_THREAD {
                    let key_val = (t * 10000 + i) as u64;
                    let key = key_val.to_be_bytes();

                    // Write
                    let _ = tree.insert_with_guard(&key, key_val, &guard);

                    // Immediate read-back
                    if tree.get_with_guard(&key, &guard).is_none() {
                        verify_failures.fetch_add(1, Ordering::Relaxed);
                    }

                    // Read some other thread's key (may or may not exist)
                    let other_t = (t + 1) % NUM_THREADS;
                    let other_key = ((other_t * 10000 + i) as u64).to_be_bytes();
                    let _ = tree.get_with_guard(&other_key, &guard);
                }
            })
        })
        .collect();

    for h in handles {
        h.join().unwrap();
    }

    let fail_count = verify_failures.load(Ordering::Relaxed);
    report_debug_counters("mixed_continuous_readwrite");

    if fail_count > 0 {
        panic!(
            "mixed_continuous: {} immediate verification failures",
            fail_count
        );
    }

    assert_eq!(tree.len(), NUM_THREADS * OPS_PER_THREAD);
}

// =============================================================================
// REPEATED RUN TESTS (catch intermittent bugs)
// =============================================================================

#[test]
fn repeated_10_runs_4_threads_8byte() {
    common::init_tracing();

    for run in 0..10 {
        reset_debug_counters();

        let tree = Arc::new(MassTree::<u64>::new());
        let verify_failures = Arc::new(AtomicUsize::new(0));

        let handles: Vec<_> = (0..4)
            .map(|t| {
                let tree = Arc::clone(&tree);
                let verify_failures = Arc::clone(&verify_failures);
                thread::spawn(move || {
                    let guard = tree.guard();
                    for i in 0..500 {
                        let key_val = (t * 10000 + i) as u64;
                        let key = key_val.to_be_bytes();

                        let _ = tree.insert_with_guard(&key, key_val, &guard);

                        if tree.get_with_guard(&key, &guard).is_none() {
                            verify_failures.fetch_add(1, Ordering::Relaxed);
                        }
                    }
                })
            })
            .collect();

        for h in handles {
            h.join().unwrap();
        }

        let fail_count = verify_failures.load(Ordering::Relaxed);
        if fail_count > 0 {
            let (blink, notfound) = get_debug_counters();
            panic!(
                "repeated_10_runs: run {} failed with {} verification failures\n\
                 blink_follow={}, search_notfound={}",
                run, fail_count, blink, notfound
            );
        }

        assert_eq!(tree.len(), 2000, "Failed on run {}", run);
    }
}

#[test]
fn repeated_10_runs_4_threads_24byte() {
    common::init_tracing();

    for run in 0..10 {
        reset_debug_counters();

        let tree = Arc::new(MassTree::<u64>::new());
        let verify_failures = Arc::new(AtomicUsize::new(0));

        let handles: Vec<_> = (0..4)
            .map(|t| {
                let tree = Arc::clone(&tree);
                let verify_failures = Arc::clone(&verify_failures);
                thread::spawn(move || {
                    let guard = tree.guard();
                    for i in 0..500 {
                        let key = format!("thread_{:02}_key_{:010}", t, i);

                        let _ =
                            tree.insert_with_guard(key.as_bytes(), (t * 10000 + i) as u64, &guard);

                        if tree.get_with_guard(key.as_bytes(), &guard).is_none() {
                            verify_failures.fetch_add(1, Ordering::Relaxed);
                        }
                    }
                })
            })
            .collect();

        for h in handles {
            h.join().unwrap();
        }

        let fail_count = verify_failures.load(Ordering::Relaxed);
        if fail_count > 0 {
            let (blink, notfound) = get_debug_counters();
            panic!(
                "repeated_10_runs_24byte: run {} failed with {} verification failures\n\
                 blink_follow={}, search_notfound={}",
                run, fail_count, blink, notfound
            );
        }

        assert_eq!(tree.len(), 2000, "Failed on run {}", run);
    }
}

#[test]
fn repeated_20_runs_8_threads_mixed() {
    common::init_tracing();

    for run in 0..20 {
        reset_debug_counters();

        let tree = Arc::new(MassTree::<u64>::new());
        let verify_failures = Arc::new(AtomicUsize::new(0));

        let handles: Vec<_> = (0..8)
            .map(|t| {
                let tree = Arc::clone(&tree);
                let verify_failures = Arc::clone(&verify_failures);
                thread::spawn(move || {
                    let guard = tree.guard();
                    for i in 0..250 {
                        // Alternate between 8-byte and 24-byte keys
                        let (key_bytes, key_val): (Vec<u8>, u64) = if i % 2 == 0 {
                            let v = (t * 10000 + i) as u64;
                            (v.to_be_bytes().to_vec(), v)
                        } else {
                            let k = format!("thread_{:02}_key_{:010}", t, i);
                            (k.into_bytes(), (t * 10000 + i) as u64)
                        };

                        let _ = tree.insert_with_guard(&key_bytes, key_val, &guard);

                        if tree.get_with_guard(&key_bytes, &guard).is_none() {
                            verify_failures.fetch_add(1, Ordering::Relaxed);
                        }
                    }
                })
            })
            .collect();

        for h in handles {
            h.join().unwrap();
        }

        let fail_count = verify_failures.load(Ordering::Relaxed);
        if fail_count > 0 {
            let (blink, notfound) = get_debug_counters();
            panic!(
                "repeated_20_runs_mixed: run {} failed with {} verification failures\n\
                 blink_follow={}, search_notfound={}",
                run, fail_count, blink, notfound
            );
        }

        assert_eq!(tree.len(), 2000, "Failed on run {}", run);
    }
}

// =============================================================================
// EXTREME STRESS TESTS (for CI or extended testing)
// =============================================================================

/// Long-running stress test - run with --ignored for extended testing
#[test]
#[ignore]
fn extreme_100_runs_stress() {
    common::init_tracing();

    for run in 0..100 {
        reset_debug_counters();

        let tree = Arc::new(MassTree::<u64>::new());
        let verify_failures = Arc::new(AtomicUsize::new(0));

        let handles: Vec<_> = (0..8)
            .map(|t| {
                let tree = Arc::clone(&tree);
                let verify_failures = Arc::clone(&verify_failures);
                thread::spawn(move || {
                    let guard = tree.guard();
                    for i in 0..1000 {
                        let key = format!("run{:03}_t{:02}_k{:06}", run, t, i);

                        let _ =
                            tree.insert_with_guard(key.as_bytes(), (t * 10000 + i) as u64, &guard);

                        if tree.get_with_guard(key.as_bytes(), &guard).is_none() {
                            verify_failures.fetch_add(1, Ordering::Relaxed);
                        }
                    }
                })
            })
            .collect();

        for h in handles {
            h.join().unwrap();
        }

        let fail_count = verify_failures.load(Ordering::Relaxed);
        if fail_count > 0 {
            let (blink, notfound) = get_debug_counters();
            panic!(
                "extreme_100_runs: run {} failed with {} verification failures\n\
                 blink_follow={}, search_notfound={}",
                run, fail_count, blink, notfound
            );
        }

        if run % 10 == 0 {
            eprintln!("extreme_100_runs: completed run {}/100", run);
        }

        assert_eq!(tree.len(), 8000, "Failed on run {}", run);
    }

    eprintln!("extreme_100_runs: ALL 100 RUNS PASSED");
}

/// Massive key count test
#[test]
#[ignore]
fn extreme_100k_keys() {
    common::init_tracing();
    reset_debug_counters();

    const NUM_THREADS: usize = 16;
    const KEYS_PER_THREAD: usize = 6250;
    const TOTAL_KEYS: usize = NUM_THREADS * KEYS_PER_THREAD; // 100,000

    let tree = Arc::new(MassTree::<u64>::new());
    let verify_failures = Arc::new(AtomicUsize::new(0));

    let handles: Vec<_> = (0..NUM_THREADS)
        .map(|t| {
            let tree = Arc::clone(&tree);
            let verify_failures = Arc::clone(&verify_failures);
            thread::spawn(move || {
                let guard = tree.guard();
                for i in 0..KEYS_PER_THREAD {
                    let key_val = (t * 1000000 + i) as u64;
                    let key = key_val.to_be_bytes();

                    let _ = tree.insert_with_guard(&key, key_val, &guard);

                    // Verify every 100th key to reduce overhead
                    if i % 100 == 0 {
                        if tree.get_with_guard(&key, &guard).is_none() {
                            verify_failures.fetch_add(1, Ordering::Relaxed);
                        }
                    }
                }
            })
        })
        .collect();

    for h in handles {
        h.join().unwrap();
    }

    let fail_count = verify_failures.load(Ordering::Relaxed);
    report_debug_counters("extreme_100k_keys");

    if fail_count > 0 {
        panic!("extreme_100k: {} sampled verification failures", fail_count);
    }

    assert_eq!(tree.len(), TOTAL_KEYS);
    eprintln!("extreme_100k_keys: PASSED with {} keys", TOTAL_KEYS);
}
