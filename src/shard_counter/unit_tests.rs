use super::{CACHE_LINE_SIZE, PaddedCounter, SHARDS, ShardedCounter};
use std::{sync::Arc, thread::Scope};

#[test]
fn test_new_counter_is_zero() {
    let counter = ShardedCounter::new();
    assert_eq!(counter.load(), 0);
}

#[test]
fn test_single_thread_increment() {
    let counter = ShardedCounter::new();

    for _ in 0..1000 {
        counter.increment();
    }

    assert_eq!(counter.load(), 1000);
}

#[test]
fn test_single_thread_decrement() {
    let counter = ShardedCounter::new();

    for _ in 0..1000 {
        counter.increment();
    }

    for _ in 0..300 {
        counter.decrement();
    }

    assert_eq!(counter.load(), 700);
}

#[test]
fn test_concurrent_increments() {
    let counter = Arc::new(ShardedCounter::new());
    let threads: usize = 8;
    let increments_per_thread: usize = 10_000;

    std::thread::scope(|s: &Scope<'_, '_>| {
        for _ in 0..threads {
            let counter = Arc::clone(&counter);

            s.spawn(move || {
                for _ in 0..increments_per_thread {
                    counter.increment();
                }
            });
        }
    });

    assert_eq!(counter.load(), threads * increments_per_thread);
}

#[test]
fn test_concurrent_mixed() {
    let counter: Arc<ShardedCounter> = Arc::new(ShardedCounter::new());

    // Pre-populate
    for _ in 0..5000 {
        counter.increment();
    }

    std::thread::scope(|s: &Scope<'_, '_>| {
        for _ in 0..4 {
            let counter: Arc<ShardedCounter> = Arc::clone(&counter);

            s.spawn(move || {
                for _ in 0..1000 {
                    counter.increment();
                }
            });
        }

        for _ in 0..4 {
            let counter: Arc<ShardedCounter> = Arc::clone(&counter);

            s.spawn(move || {
                for _ in 0..1000 {
                    counter.decrement();
                }
            });
        }
    });

    // 5000 + (4 * 1000) - (4 * 1000) = 5000 + 4000 - 4000 = 5000
    assert_eq!(counter.load(), 5000);
}

#[test]
fn test_reset() {
    let counter = ShardedCounter::new();

    for _ in 0..1000 {
        counter.increment();
    }

    counter.reset();

    assert_eq!(counter.load(), 0);
}

#[test]
fn test_shard_index_is_cached() {
    // Same thread always gets the same shard index
    let shard1 = ShardedCounter::shard_index();
    let shard2 = ShardedCounter::shard_index();
    let shard3 = ShardedCounter::shard_index();

    assert_eq!(shard1, shard2);
    assert_eq!(shard2, shard3);
}

#[test]
fn test_shard_index_in_valid_range() {
    // Shard index must be < SHARDS
    let index = ShardedCounter::shard_index();
    assert!(index < SHARDS, "shard index {index} >= SHARDS ({SHARDS})");
}

#[test]
fn test_cache_line_alignment() {
    // Verify each PaddedCounter is aligned to CACHE_LINE_SIZE
    assert_eq!(
        std::mem::align_of::<PaddedCounter>(),
        CACHE_LINE_SIZE,
        "PaddedCounter alignment should be {CACHE_LINE_SIZE}"
    );
}

#[test]
fn test_padded_counter_size() {
    // Verify PaddedCounter is at least CACHE_LINE_SIZE
    // (it may be larger due to alignment padding)
    assert!(
        std::mem::size_of::<PaddedCounter>() >= CACHE_LINE_SIZE,
        "PaddedCounter size should be >= {CACHE_LINE_SIZE}"
    );
}

#[test]
fn test_sharded_counter_is_send_sync() {
    // Compile-time check that ShardedCounter is Send + Sync
    fn assert_send_sync<T: Send + Sync>() {}
    assert_send_sync::<ShardedCounter>();
}

#[test]
fn test_add_positive_and_negative() {
    let counter = ShardedCounter::new();

    counter.add(100);
    assert_eq!(counter.load(), 100);

    counter.add(-30);
    assert_eq!(counter.load(), 70);
}

#[test]
fn test_stress_large_scale() {
    // Stress test with 1 million operations across many threads
    let counter: Arc<ShardedCounter> = Arc::new(ShardedCounter::new());
    let threads: usize = 16;
    let ops_per_thread: usize = 100_000;

    std::thread::scope(|s: &Scope<'_, '_>| {
        for _ in 0..threads {
            let counter: Arc<ShardedCounter> = Arc::clone(&counter);

            s.spawn(move || {
                for _ in 0..ops_per_thread {
                    counter.increment();
                }
            });
        }
    });

    assert_eq!(counter.load(), threads * ops_per_thread);
}

#[test]
fn test_stress_mixed_operations() {
    // Stress test with mixed increment/decrement across many threads
    let counter: Arc<ShardedCounter> = Arc::new(ShardedCounter::new());
    let threads: usize = 16;
    let ops_per_thread: usize = 50_000;

    // Pre-populate to avoid negative intermediate states affecting the test
    let initial: usize = threads * ops_per_thread;
    for _ in 0..initial {
        counter.increment();
    }

    std::thread::scope(|s: &Scope<'_, '_>| {
        // Half threads increment, half decrement
        for i in 0..threads {
            let counter: Arc<ShardedCounter> = Arc::clone(&counter);

            s.spawn(move || {
                if i % 2 == 0 {
                    for _ in 0..ops_per_thread {
                        counter.increment();
                    }
                } else {
                    for _ in 0..ops_per_thread {
                        counter.decrement();
                    }
                }
            });
        }
    });

    // Net change is zero (8 threads increment, 8 decrement)
    assert_eq!(counter.load(), initial);
}

#[test]
fn test_shard_distribution_across_threads() {
    // Verify that different threads get distributed across shards
    use std::collections::HashSet;
    use std::sync::Mutex;

    let observed_shards: Arc<Mutex<HashSet<usize>>> = Arc::new(Mutex::new(HashSet::new()));
    let threads: usize = 32; // Spawn more threads than shards to increase coverage

    std::thread::scope(|s: &Scope<'_, '_>| {
        for _ in 0..threads {
            let observed: Arc<Mutex<HashSet<usize>>> = Arc::clone(&observed_shards);

            s.spawn(move || {
                let index: usize = ShardedCounter::shard_index();
                observed.lock().unwrap().insert(index);
            });
        }
    });

    let shards_hit: usize = observed_shards.lock().unwrap().len();

    // With 32 threads and 16 shards, we expect good distribution.
    // Statistically, we should hit most shards. Require at least 8 (50%).
    // This is a probabilistic test, but with FxHash it should be reliable.
    assert!(
        shards_hit >= 8,
        "Poor shard distribution: only {shards_hit}/{SHARDS} shards hit by {threads} threads"
    );
}

#[test]
fn test_sharded_counter_memory_layout() {
    // Verify the overall ShardedCounter size is as expected
    let expected_size: usize = SHARDS * CACHE_LINE_SIZE;

    assert_eq!(
        std::mem::size_of::<ShardedCounter>(),
        expected_size,
        "ShardedCounter should be {expected_size} bytes ({SHARDS} shards × {CACHE_LINE_SIZE} bytes)"
    );
}

#[test]
fn test_default_trait() {
    let counter: ShardedCounter = ShardedCounter::default();
    assert_eq!(counter.load(), 0);
}

#[test]
fn test_debug_output() {
    let counter = ShardedCounter::new();
    counter.increment();
    counter.increment();
    counter.increment();

    let debug_str: String = format!("{:?}", counter);

    assert!(
        debug_str.contains("ShardedCounter"),
        "Debug output should contain type name"
    );
    assert!(
        debug_str.contains("total"),
        "Debug output should contain 'total'"
    );
    assert!(debug_str.contains("3"), "Debug output should contain count");
}
