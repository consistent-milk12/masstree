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
