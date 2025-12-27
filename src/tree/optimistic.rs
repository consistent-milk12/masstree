//! Debug counters for concurrent operations.
//!
//! Lightweight atomic counters for diagnosing concurrent behavior.
//! All counters are gated behind the `tracing` feature.

#[cfg(feature = "tracing")]
use std::sync::atomic::{AtomicU64, Ordering::Relaxed};

// ============================================================================
// Lightweight Debug Counters (only when tracing is enabled)
// ============================================================================

/// Atomic counter for B-link navigation issues.
#[cfg(feature = "tracing")]
pub static BLINK_SHOULD_FOLLOW_COUNT: AtomicU64 = AtomicU64::new(0);

/// Atomic counter for total `NotFound` results in `search_leaf_concurrent`.
#[cfg(feature = "tracing")]
pub static SEARCH_NOT_FOUND_COUNT: AtomicU64 = AtomicU64::new(0);

/// Atomic counter for successful CAS inserts.
#[cfg(feature = "tracing")]
pub static CAS_INSERT_SUCCESS_COUNT: AtomicU64 = AtomicU64::new(0);

/// Atomic counter for CAS insert retries (contention).
#[cfg(feature = "tracing")]
pub static CAS_INSERT_RETRY_COUNT: AtomicU64 = AtomicU64::new(0);

/// Atomic counter for CAS insert fallbacks to locked path.
#[cfg(feature = "tracing")]
pub static CAS_INSERT_FALLBACK_COUNT: AtomicU64 = AtomicU64::new(0);

/// Atomic counter for locked insert completions.
#[cfg(feature = "tracing")]
pub static LOCKED_INSERT_COUNT: AtomicU64 = AtomicU64::new(0);

/// Atomic counter for leaf splits.
#[cfg(feature = "tracing")]
pub static SPLIT_COUNT: AtomicU64 = AtomicU64::new(0);

/// Atomic counter for `advance_to_key` B-link follows.
#[cfg(feature = "tracing")]
pub static ADVANCE_BLINK_COUNT: AtomicU64 = AtomicU64::new(0);

/// Atomic counter for keys inserted into wrong leaf.
#[cfg(feature = "tracing")]
pub static WRONG_LEAF_INSERT_COUNT: AtomicU64 = AtomicU64::new(0);

/// Atomic counter for B-link advance anomalies (cycle or limit hit).
#[cfg(feature = "tracing")]
pub static BLINK_ADVANCE_ANOMALY_COUNT: AtomicU64 = AtomicU64::new(0);

// ============================================================================
// Parent-Wait Instrumentation (for variance analysis)
// ============================================================================

/// Number of times we entered the parent-wait loop (NULL parent on non-layer-root).
#[cfg(feature = "tracing")]
pub static PARENT_WAIT_HIT_COUNT: AtomicU64 = AtomicU64::new(0);

/// Total spin iterations across all parent-wait events.
#[cfg(feature = "tracing")]
pub static PARENT_WAIT_TOTAL_SPINS: AtomicU64 = AtomicU64::new(0);

/// Maximum spins in any single parent-wait event.
#[cfg(feature = "tracing")]
pub static PARENT_WAIT_MAX_SPINS: AtomicU64 = AtomicU64::new(0);

/// Total nanoseconds spent in parent-wait loops.
#[cfg(feature = "tracing")]
pub static PARENT_WAIT_TOTAL_NS: AtomicU64 = AtomicU64::new(0);

/// Maximum nanoseconds in any single parent-wait event.
#[cfg(feature = "tracing")]
pub static PARENT_WAIT_MAX_NS: AtomicU64 = AtomicU64::new(0);

/// Reset debug counters (call before test).
#[cfg(feature = "tracing")]
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
    BLINK_ADVANCE_ANOMALY_COUNT.store(0, Relaxed);
    PARENT_WAIT_HIT_COUNT.store(0, Relaxed);
    PARENT_WAIT_TOTAL_SPINS.store(0, Relaxed);
    PARENT_WAIT_MAX_SPINS.store(0, Relaxed);
    PARENT_WAIT_TOTAL_NS.store(0, Relaxed);
    PARENT_WAIT_MAX_NS.store(0, Relaxed);
}

/// Debug counter values.
#[cfg(feature = "tracing")]
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
    /// B-link advance anomalies (cycle or limit hit)
    pub blink_advance_anomaly: u64,
    /// Parent-wait loop hits (NULL parent on non-layer-root)
    pub parent_wait_hits: u64,
    /// Total spins in parent-wait loops
    pub parent_wait_total_spins: u64,
    /// Max spins in single parent-wait
    pub parent_wait_max_spins: u64,
    /// Total ns in parent-wait loops
    pub parent_wait_total_ns: u64,
    /// Max ns in single parent-wait
    pub parent_wait_max_ns: u64,
}

/// Get debug counter values (legacy).
#[cfg(feature = "tracing")]
pub fn get_debug_counters() -> (u64, u64) {
    (
        BLINK_SHOULD_FOLLOW_COUNT.load(Relaxed),
        SEARCH_NOT_FOUND_COUNT.load(Relaxed),
    )
}

/// Get all debug counter values.
#[cfg(feature = "tracing")]
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
        blink_advance_anomaly: BLINK_ADVANCE_ANOMALY_COUNT.load(Relaxed),
        parent_wait_hits: PARENT_WAIT_HIT_COUNT.load(Relaxed),
        parent_wait_total_spins: PARENT_WAIT_TOTAL_SPINS.load(Relaxed),
        parent_wait_max_spins: PARENT_WAIT_MAX_SPINS.load(Relaxed),
        parent_wait_total_ns: PARENT_WAIT_TOTAL_NS.load(Relaxed),
        parent_wait_max_ns: PARENT_WAIT_MAX_NS.load(Relaxed),
    }
}

/// Get parent-wait statistics for variance analysis.
#[cfg(feature = "tracing")]
#[derive(Debug, Clone, Copy)]
pub struct ParentWaitStats {
    /// Number of parent-wait events
    pub hits: u64,
    /// Total spin iterations
    pub total_spins: u64,
    /// Max spins in single event
    pub max_spins: u64,
    /// Average spins per event (0 if no hits)
    pub avg_spins: f64,
    /// Total wait time in microseconds
    pub total_us: f64,
    /// Max wait time in microseconds
    pub max_us: f64,
    /// Average wait time per event in microseconds
    pub avg_us: f64,
}

/// Get parent-wait statistics summary.
#[cfg(feature = "tracing")]
#[expect(clippy::cast_precision_loss)]
pub fn get_parent_wait_stats() -> ParentWaitStats {
    let hits = PARENT_WAIT_HIT_COUNT.load(Relaxed);
    let total_spins = PARENT_WAIT_TOTAL_SPINS.load(Relaxed);
    let max_spins = PARENT_WAIT_MAX_SPINS.load(Relaxed);
    let total_ns = PARENT_WAIT_TOTAL_NS.load(Relaxed);
    let max_ns = PARENT_WAIT_MAX_NS.load(Relaxed);

    ParentWaitStats {
        hits,
        total_spins,
        max_spins,
        avg_spins: if hits > 0 {
            total_spins as f64 / hits as f64
        } else {
            0.0
        },
        total_us: total_ns as f64 / 1000.0,
        max_us: max_ns as f64 / 1000.0,
        avg_us: if hits > 0 {
            (total_ns as f64 / hits as f64) / 1000.0
        } else {
            0.0
        },
    }
}
