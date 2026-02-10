//! Value retirement wrapper.
//!
//! This module provides [`BatchedRetire`], a thin wrapper around seize's retirement
//! that provides a consistent API for retirement and flushing across the codebase.

use seize::{Guard, LocalGuard};

/// Value retirement utilities.
///
/// Provides a consistent API for value pointer retirement. Delegates
/// directly to seize's `defer_retire`, which already performs internal batching.
#[derive(Debug)]
pub struct BatchedRetire;

impl BatchedRetire {
    /// Flush pending retirements to start reclamation process.
    ///
    /// This triggers seize to process any accumulated retirements in the
    /// thread-local batch. Note that actual memory reclamation won't happen
    /// until the guard is dropped, but this allows reclamation to proceed
    /// more quickly once the guard is released.
    ///
    /// # When to Use
    ///
    /// - After completing a batch of mutations before starting a long read
    /// - At natural boundaries in your workload
    /// - When you want more predictable reclamation timing
    #[inline(always)]
    pub fn flush(guard: &LocalGuard<'_>) {
        guard.flush();
    }
}

// ============================================================================
//  Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_flush() {
        let collector = seize::Collector::new();
        let guard = collector.enter();
        BatchedRetire::flush(&guard);
    }
}
