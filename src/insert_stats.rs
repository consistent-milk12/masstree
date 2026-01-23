//! Insert retry statistics for debugging contention issues.
//!
//! Enable with the `insert-stats` feature flag.
//! Statistics are collected via atomic counters with minimal overhead.
//!
//! # Usage
//!
//! ```rust,ignore
//! // Run benchmark with stats enabled
//! cargo bench --bench mttest --features "mimalloc insert-stats" -- rw3 -j12 -d10
//!
//! // After benchmark, print stats
//! masstree::insert_stats::print_stats();
//! masstree::insert_stats::reset_stats();
//! ```

use std::{
    fmt::{self as StdFmt, Display, Formatter},
    sync::atomic::{AtomicU64, Ordering},
};

/// Global statistics counters for insert operations.
#[derive(Debug)]
pub struct InsertStats {
    /// Total insert attempts (calls to `insert_concurrent_generic`)
    pub attempts: AtomicU64,

    /// Validation failures (version or permutation changed after lock)
    pub validation_failures: AtomicU64,

    /// Membership validation failures (key should be in different leaf)
    pub membership_failures: AtomicU64,

    /// Null pointer retries (slot found but value was null)
    pub null_ptr_retries: AtomicU64,

    /// Split retries (leaf was full, needed split)
    pub split_retries: AtomicU64,

    /// Successful inserts (new key inserted)
    pub successful_inserts: AtomicU64,

    /// Successful updates (existing key updated)
    pub successful_updates: AtomicU64,

    /// Hop limit exceeded (fell back to full traversal)
    pub hop_limit_exceeded: AtomicU64,

    /// Deleted layer retries
    pub deleted_layer_retries: AtomicU64,
}

impl InsertStats {
    const fn new() -> Self {
        Self {
            attempts: AtomicU64::new(0),
            validation_failures: AtomicU64::new(0),
            membership_failures: AtomicU64::new(0),
            null_ptr_retries: AtomicU64::new(0),
            split_retries: AtomicU64::new(0),
            successful_inserts: AtomicU64::new(0),
            successful_updates: AtomicU64::new(0),
            hop_limit_exceeded: AtomicU64::new(0),
            deleted_layer_retries: AtomicU64::new(0),
        }
    }

    /// Reset all counters to zero.
    pub fn reset(&self) {
        self.attempts.store(0, Ordering::Relaxed);
        self.validation_failures.store(0, Ordering::Relaxed);
        self.membership_failures.store(0, Ordering::Relaxed);
        self.null_ptr_retries.store(0, Ordering::Relaxed);
        self.split_retries.store(0, Ordering::Relaxed);
        self.successful_inserts.store(0, Ordering::Relaxed);
        self.successful_updates.store(0, Ordering::Relaxed);
        self.hop_limit_exceeded.store(0, Ordering::Relaxed);
        self.deleted_layer_retries.store(0, Ordering::Relaxed);
    }

    /// Get a snapshot of current stats.
    pub fn snapshot(&self) -> StatsSnapshot {
        StatsSnapshot {
            attempts: self.attempts.load(Ordering::Relaxed),
            validation_failures: self.validation_failures.load(Ordering::Relaxed),
            membership_failures: self.membership_failures.load(Ordering::Relaxed),
            null_ptr_retries: self.null_ptr_retries.load(Ordering::Relaxed),
            split_retries: self.split_retries.load(Ordering::Relaxed),
            successful_inserts: self.successful_inserts.load(Ordering::Relaxed),
            successful_updates: self.successful_updates.load(Ordering::Relaxed),
            hop_limit_exceeded: self.hop_limit_exceeded.load(Ordering::Relaxed),
            deleted_layer_retries: self.deleted_layer_retries.load(Ordering::Relaxed),
        }
    }
}

/// A point-in-time snapshot of insert statistics.
#[derive(Debug, Clone, Copy)]
pub struct StatsSnapshot {
    /// Total insert attempts (calls to `insert_concurrent_generic`).
    pub attempts: u64,

    /// Validation failures (version or permutation changed after lock).
    pub validation_failures: u64,

    /// Membership validation failures (key should be in different leaf).
    pub membership_failures: u64,

    /// Null pointer retries (slot found but value was null).
    pub null_ptr_retries: u64,

    /// Split retries (leaf was full, needed split).
    pub split_retries: u64,

    /// Successful inserts (new key inserted).
    pub successful_inserts: u64,

    /// Successful updates (existing key updated).
    pub successful_updates: u64,

    /// Hop limit exceeded (fell back to full traversal).
    pub hop_limit_exceeded: u64,

    /// Deleted layer retries.
    pub deleted_layer_retries: u64,
}

impl StatsSnapshot {
    /// Total retries (all failure types)
    #[must_use]
    pub const fn total_retries(&self) -> u64 {
        self.validation_failures
            + self.membership_failures
            + self.null_ptr_retries
            + self.split_retries
            + self.hop_limit_exceeded
            + self.deleted_layer_retries
    }

    /// Total successful operations
    #[must_use]
    pub const fn total_successful(&self) -> u64 {
        self.successful_inserts + self.successful_updates
    }

    /// Retry rate as a percentage
    #[must_use]
    #[expect(clippy::cast_precision_loss)]
    pub fn retry_rate(&self) -> f64 {
        if self.attempts == 0 {
            0.0
        } else {
            (self.total_retries() as f64 / self.attempts as f64) * 100.0
        }
    }
}

impl Display for StatsSnapshot {
    fn fmt(&self, f: &mut Formatter<'_>) -> StdFmt::Result {
        writeln!(f, "Insert Statistics:")?;
        writeln!(f, "  Attempts:             {:>12}", self.attempts)?;
        writeln!(f, "  Successful inserts:   {:>12}", self.successful_inserts)?;
        writeln!(f, "  Successful updates:   {:>12}", self.successful_updates)?;
        writeln!(
            f,
            "  Validation failures:  {:>12}",
            self.validation_failures
        )?;
        writeln!(
            f,
            "  Membership failures:  {:>12}",
            self.membership_failures
        )?;
        writeln!(f, "  Null ptr retries:     {:>12}", self.null_ptr_retries)?;
        writeln!(f, "  Split retries:        {:>12}", self.split_retries)?;
        writeln!(f, "  Hop limit exceeded:   {:>12}", self.hop_limit_exceeded)?;
        writeln!(
            f,
            "  Deleted layer retries:{:>12}",
            self.deleted_layer_retries
        )?;
        writeln!(f, "  ---")?;
        writeln!(f, "  Total retries:        {:>12}", self.total_retries())?;
        writeln!(f, "  Retry rate:           {:>11.2}%", self.retry_rate())?;
        Ok(())
    }
}

/// Global statistics instance.
pub static STATS: InsertStats = InsertStats::new();

/// Print current statistics to stdout.
pub fn print_stats() {
    println!("{}", STATS.snapshot());
}

/// Reset all statistics to zero.
pub fn reset_stats() {
    STATS.reset();
}

/// Get a snapshot of current statistics.
pub fn get_stats() -> StatsSnapshot {
    STATS.snapshot()
}

// Inline helper functions for recording stats (compile to no-ops when feature disabled)

/// Record an insert attempt.
#[inline(always)]
pub fn record_attempt() {
    STATS.attempts.fetch_add(1, Ordering::Relaxed);
}

/// Record a validation failure (version or permutation changed).
#[inline(always)]
pub fn record_validation_failure() {
    STATS.validation_failures.fetch_add(1, Ordering::Relaxed);
}

/// Record a membership failure (key should be in different leaf).
#[inline(always)]
pub fn record_membership_failure() {
    STATS.membership_failures.fetch_add(1, Ordering::Relaxed);
}

/// Record a null pointer retry (slot found but value was null).
#[inline(always)]
pub fn record_null_ptr_retry() {
    STATS.null_ptr_retries.fetch_add(1, Ordering::Relaxed);
}

/// Record a split retry (leaf was full, needed split).
#[inline(always)]
pub fn record_split_retry() {
    STATS.split_retries.fetch_add(1, Ordering::Relaxed);
}

/// Record a successful insert (new key inserted).
#[inline(always)]
pub fn record_successful_insert() {
    STATS.successful_inserts.fetch_add(1, Ordering::Relaxed);
}

/// Record a successful update (existing key updated).
#[inline(always)]
pub fn record_successful_update() {
    STATS.successful_updates.fetch_add(1, Ordering::Relaxed);
}

/// Record a hop limit exceeded (fell back to full traversal).
#[inline(always)]
pub fn record_hop_limit_exceeded() {
    STATS.hop_limit_exceeded.fetch_add(1, Ordering::Relaxed);
}

/// Record a deleted layer retry.
#[inline(always)]
pub fn record_deleted_layer_retry() {
    STATS.deleted_layer_retries.fetch_add(1, Ordering::Relaxed);
}
