//! Debug counters for concurrent operations.
//!
//! Lightweight atomic counters for diagnosing concurrent behavior.

use std::sync::atomic::{AtomicU64, Ordering::Relaxed};

// ============================================================================
// Lightweight Debug Counters (near-zero overhead)
// ============================================================================

/// Atomic counter for B-link navigation issues.
pub static BLINK_SHOULD_FOLLOW_COUNT: AtomicU64 = AtomicU64::new(0);

/// Atomic counter for total `NotFound` results in `search_leaf_concurrent`.
pub static SEARCH_NOT_FOUND_COUNT: AtomicU64 = AtomicU64::new(0);

/// Atomic counter for successful CAS inserts.
pub static CAS_INSERT_SUCCESS_COUNT: AtomicU64 = AtomicU64::new(0);

/// Atomic counter for CAS insert retries (contention).
pub static CAS_INSERT_RETRY_COUNT: AtomicU64 = AtomicU64::new(0);

/// Atomic counter for CAS insert fallbacks to locked path.
pub static CAS_INSERT_FALLBACK_COUNT: AtomicU64 = AtomicU64::new(0);

/// Atomic counter for locked insert completions.
pub static LOCKED_INSERT_COUNT: AtomicU64 = AtomicU64::new(0);

/// Atomic counter for leaf splits.
pub static SPLIT_COUNT: AtomicU64 = AtomicU64::new(0);

/// Atomic counter for `advance_to_key` B-link follows.
pub static ADVANCE_BLINK_COUNT: AtomicU64 = AtomicU64::new(0);

/// Atomic counter for keys inserted into wrong leaf.
pub static WRONG_LEAF_INSERT_COUNT: AtomicU64 = AtomicU64::new(0);

/// Reset debug counters (call before test).
pub fn reset_debug_counters() {
    BLINK_SHOULD_FOLLOW_COUNT.store(0, Relaxed);
    SEARCH_NOT_FOUND_COUNT.store(0, Relaxed);
    CAS_INSERT_SUCCESS_COUNT.store(0, Relaxed);
    CAS_INSERT_RETRY_COUNT.store(0, Relaxed);
    CAS_INSERT_FALLBACK_COUNT.store(0, Relaxed);
    LOCKED_INSERT_COUNT.store(0, Relaxed);
    SPLIT_COUNT.store(0, Relaxed);
    ADVANCE_BLINK_COUNT.store(0, Relaxed);
    WRONG_LEAF_INSERT_COUNT.store(0, Relaxed);
}

/// Debug counter values.
#[derive(Debug, Clone, Copy)]
pub struct DebugCounters {
    /// B-link should have been followed in `get()`
    pub blink_should_follow: u64,
    /// Total `NotFound` results
    pub search_not_found: u64,
    /// Successful CAS inserts
    pub cas_insert_success: u64,
    /// CAS insert retries
    pub cas_insert_retry: u64,
    /// CAS insert fallbacks to locked
    pub cas_insert_fallback: u64,
    /// Locked insert completions
    pub locked_insert: u64,
    /// Leaf splits
    pub split: u64,
    /// B-links followed in `advance_to_key`
    pub advance_blink: u64,
    /// Keys inserted into wrong leaf
    pub wrong_leaf_insert: u64,
}

/// Get debug counter values (legacy).
pub fn get_debug_counters() -> (u64, u64) {
    (
        BLINK_SHOULD_FOLLOW_COUNT.load(Relaxed),
        SEARCH_NOT_FOUND_COUNT.load(Relaxed),
    )
}

/// Get all debug counter values.
pub fn get_all_debug_counters() -> DebugCounters {
    DebugCounters {
        blink_should_follow: BLINK_SHOULD_FOLLOW_COUNT.load(Relaxed),
        search_not_found: SEARCH_NOT_FOUND_COUNT.load(Relaxed),
        cas_insert_success: CAS_INSERT_SUCCESS_COUNT.load(Relaxed),
        cas_insert_retry: CAS_INSERT_RETRY_COUNT.load(Relaxed),
        cas_insert_fallback: CAS_INSERT_FALLBACK_COUNT.load(Relaxed),
        locked_insert: LOCKED_INSERT_COUNT.load(Relaxed),
        split: SPLIT_COUNT.load(Relaxed),
        advance_blink: ADVANCE_BLINK_COUNT.load(Relaxed),
        wrong_leaf_insert: WRONG_LEAF_INSERT_COUNT.load(Relaxed),
    }
}
