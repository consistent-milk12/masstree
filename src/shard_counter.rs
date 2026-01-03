//! Sharded counter for high-throughput concurrent counting.
//!
//! This module provides [`ShardedCounter`], a counter optimized for concurrent
//! increment/decrement operations by distributing updates across multiple
//! cache-line-aligned shards.
//!
//! # Performance Characteristics
//!
//! - **Increment/Decrement**: O(1), minimal contention (threads hit different shards)
//! - **Load (sum)**: O(SHARDS), requires reading all shards
//!
//! # Consistency Model
//!
//! Unlike a single [`AtomicUsize`](std::sync::AtomicUsize) `load()` is not linearizable , it reads
//! multiple independent atomics sequentially. During concurrent mutations:
//! - The returned value may be slightly stale or inconsistent
//! - After all mutating threads have joined/quiesced, `load()` returns the exact count
//!
//! This is acceptable for [`MassTree`]'s `len()` which is documented as approximate
//! during concurrent operations.
//!
//! TODO: I have to look into this more for the optimal number of shards.
//! CPU cores may be a point of optimization, and a custom config builder pattern
//! can be considered for more advanced use cases.
//!
//! # Implementation Details
//! - Uses 16 shards
//! - Each shard is explictly aligned to 128 bytes via `#[repr(C, align(128))]`
//! - Thread shard index is cached in thread-local storage for fast access
//! - Relaxed ordering is sufficient since the counter is used for approximate counts

use std::cell::Cell;
use std::sync::atomic::{AtomicIsize, AtomicUsize, Ordering};

/// Number of shards in the counter.
///
/// NOTE: 16 shards provide good distribution for systems up to 16 cores.
/// With more cores, some threads will share shards, but contention
/// is still greatly reduced compared to a single counter.
const SHARDS: usize = 16;

/// Cache line size for alignment.
const CACHE_LINE_SIZE: usize = 128;

#[repr(C, align(128))]
struct PaddedCounter {
    /// The actual counter value.
    /// Using [`AtomicIsize`] to support temporary negative values in individual shards
    /// during concurrent increment/decrement operations.
    value: AtomicIsize,
}
