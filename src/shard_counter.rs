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
//! Unlike a single [`AtomicUsize`], `load()` is not linearizable—it reads
//! multiple independent atomics sequentially. During concurrent mutations:
//! - The returned value may be slightly stale or inconsistent
//! - After all mutating threads have joined/quiesced, `load()` returns the exact count
//!
//! This is acceptable for [`MassTree`](crate::MassTree) `len()` which is documented as approximate
//! during concurrent operations.
//!
//! # Implementation Details
//! - Uses 16 shards
//! - Each shard is explicitly aligned to 128 bytes via `#[repr(C, align(128))]`
//! - Thread shard index is cached in thread-local storage for fast access
//! - Relaxed ordering is sufficient since the counter is used for approximate counts

use rustc_hash::FxHasher;
use static_assertions::{const_assert, const_assert_eq};
use std::cell::Cell;
use std::fmt as StdFmt;
use std::hash::{Hash, Hasher};
use std::sync::atomic::AtomicIsize;
use std::sync::atomic::Ordering as AtomicOrdering;
use std::thread::{self as StdThread, ThreadId};

/// Number of shards in the counter.
///
/// NOTE: 16 shards provide good distribution for systems up to 16 cores.
/// With more cores, some threads will share shards, but contention
/// is still greatly reduced compared to a single counter.
const SHARDS: usize = 16;

/// Cache line size for alignment.
const CACHE_LINE_SIZE: usize = 128;

/// A single cache-line-aligned counter shard.
///
/// The explicit alignment ensures that each shard occupies its own cache line(s),
/// eliminating false sharing between threads updating different shards.
///
/// `#[repr(C)]` alone does not guarantee alignment of array elements,
/// `align(128)` is required to ensure each [`PaddedCounter`] in the array
/// starts at a 128-byte boundary.
#[derive(Debug)]
#[repr(C, align(128))]
struct PaddedCounter {
    /// The actual counter value.
    /// Using [`AtomicIsize`] to support temporary negative values in individual shards
    /// during concurrent increment/decrement operations.
    value: AtomicIsize,
}

impl PaddedCounter {
    /// Create a new padded counter initialized to zero.
    const fn new() -> Self {
        Self {
            value: AtomicIsize::new(0),
        }
    }
}

// Compile-time layout verification.
// PaddedCounter must be cache-line aligned (128 bytes) to prevent false sharing.
const_assert_eq!(std::mem::align_of::<PaddedCounter>(), CACHE_LINE_SIZE);
const_assert!(std::mem::size_of::<PaddedCounter>() >= CACHE_LINE_SIZE);

thread_local! {
    /// Thread-local cached shard index.
    ///
    /// Using [`Cell<Option<usize>>`] to cache the shard index after first computation.
    /// This avoids calling thread::current().id() on every increment/decrement.
    static CACHED_SHARD: Cell<Option<usize>> = const { Cell::new(None) };
}

/// A sharded counter optimized for concurrent increment/decrement operations.
///
/// NOTE: Send + Sync is automatically derived because [`ShardedCounter`] only contains
/// [`AtomicIsize`] fields, which are Send + Sync. No manual Send + Sync needed.
pub struct ShardedCounter {
    shards: [PaddedCounter; SHARDS],
}

impl ShardedCounter {
    /// Create a new sharded counter initialized to zero.
    #[inline]
    #[must_use]
    pub const fn new() -> Self {
        Self {
            shards: [
                PaddedCounter::new(),
                PaddedCounter::new(),
                PaddedCounter::new(),
                PaddedCounter::new(),
                PaddedCounter::new(),
                PaddedCounter::new(),
                PaddedCounter::new(),
                PaddedCounter::new(),
                PaddedCounter::new(),
                PaddedCounter::new(),
                PaddedCounter::new(),
                PaddedCounter::new(),
                PaddedCounter::new(),
                PaddedCounter::new(),
                PaddedCounter::new(),
                PaddedCounter::new(),
            ],
        }
    }

    /// Compute the shard index for the current thread.
    ///
    /// This will be called only once per thread, and the result will be cached.
    /// So #[cold] is a reasonable choice.
    #[cold]
    #[expect(clippy::cast_possible_truncation, reason = "Intentional")]
    fn compute_shard_index() -> usize {
        let thread_id: ThreadId = StdThread::current().id();
        let mut hasher: FxHasher = FxHasher::default();
        thread_id.hash(&mut hasher);

        (hasher.finish() as usize) % SHARDS
    }

    /// Get the shard index for the current thread (cached).
    ///
    /// The first call computes and caches the index; subsequent calls
    /// return the cached value with minimal overhead.
    #[inline(always)]
    fn shard_index() -> usize {
        CACHED_SHARD.with(|cell: &Cell<Option<usize>>| {
            cell.get().unwrap_or_else(|| {
                let index: usize = Self::compute_shard_index();
                cell.set(Some(index));

                index
            })
        })
    }

    /// Get a reference to the shard for the current thread.
    #[inline(always)]
    #[expect(clippy::indexing_slicing, reason = "INVARIANT: index < SHARDS")]
    fn get_shard(&self) -> &AtomicIsize {
        let index: usize = Self::shard_index();

        // SAFETY: `index < SHARDS` - coz modulo in `compute_shard_index`
        &self.shards[index].value
    }

    /// Increment the counter by 1.
    ///
    /// This operation is lock-free and optimized for concurrent access.
    /// Different threads will typically hit different shards, minimizing contention.
    ///
    /// # Ordering
    /// Uses [`AtomicOrdering::Relaxed`]. If increment is needed to be synchronized with
    /// other operations, it will be necessary to use external synchronization.
    #[inline(always)]
    pub fn increment(&self) {
        self.get_shard().fetch_add(1, AtomicOrdering::Relaxed);
    }

    /// Decrement the counter by 1.
    ///
    /// This operation is lock-free and optimized for concurrent access.
    ///
    /// NOTE: Individual shards may temporarily go negative during concurrent operations
    /// (e.g., if a decrement races ahead of a pending increment on the same shard).
    /// The total sum will be correct once all operations complete.
    ///
    /// # Ordering
    /// Uses [`AtomicOrdering::Relaxed`].
    #[inline(always)]
    pub fn decrement(&self) {
        self.get_shard().fetch_sub(1, AtomicOrdering::Relaxed);
    }

    /// Add a provided value to the counter.
    #[allow(dead_code, reason = "Public API")]
    #[inline(always)]
    pub fn add(&self, val: isize) {
        self.get_shard().fetch_add(val, AtomicOrdering::Relaxed);
    }

    /// Load the current counter value by summing all shards.
    ///
    /// # Consistency
    /// This operation reads all shards sequentially and is not linearizable.
    /// During concurrent mutations, the result may be:
    /// - Slightly stale (missing recent increments)
    /// - Temporarily inconsistent (seeing some but not all concurrent changes)
    ///
    /// After all mutating have joined/quiesced, this returns the exact count.
    ///
    /// # Returns
    /// The approximate total count as `usize`.
    ///
    /// # Panics
    /// Debug builds will panic if the total is negative (indicating a bug where
    /// decrements exceeded increments). Release builds return 0 in this case.
    ///
    /// # Performance
    /// `O(SHARDS) = O(16)` reads. This is more expensive than a single atomic load.
    pub fn load(&self) -> usize {
        let mut total: isize = 0;

        for shard in &self.shards {
            total += shard.value.load(AtomicOrdering::Relaxed);
        }

        debug_assert!(
            total >= 0,
            "ShardedCounter total is negative ({total}): more decrements than increments"
        );

        if total >= 0 { total.cast_unsigned() } else { 0 }
    }

    /// Reset the counter to zero.
    ///
    /// # Thread Safety
    #[allow(dead_code, reason = "Public API")]
    #[inline]
    pub fn reset(&self) {
        for shard in &self.shards {
            shard.value.store(0, AtomicOrdering::Relaxed);
        }
    }
}

impl Default for ShardedCounter {
    #[inline]
    fn default() -> Self {
        Self::new()
    }
}

impl StdFmt::Debug for ShardedCounter {
    fn fmt(&self, f: &mut StdFmt::Formatter<'_>) -> StdFmt::Result {
        f.debug_struct("ShardedCounter")
            .field("total", &self.load())
            .finish()
    }
}

#[cfg(test)]
mod unit_tests;
