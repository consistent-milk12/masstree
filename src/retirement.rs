//! Value retirement wrapper.
//!
//! This module provides [`BatchedRetire`], a thin wrapper around seize's retirement
//! that provides a consistent API for value pointer retirement across the codebase.
//!
//! # Current Implementation
//!
//! Delegates directly to `seize::Guard::defer_retire`. Seize already performs
//! internal batching, so this module just encapsulates the `S::cleanup_value_ptr`
//! pattern used at all value retirement call sites.
//!
//! # Usage
//!
//! ```rust,ignore
//! use crate::retirement::BatchedRetire;
//! use seize::LocalGuard;
//!
//! // For value cleanup (uses S::cleanup_value_ptr)
//! if S::NEEDS_RETIREMENT {
//!     unsafe {
//!         BatchedRetire::defer_value::<S>(old_ptr, guard);
//!     }
//! }
//! ```

use seize::{Guard, LocalGuard};

use crate::slot::ValueSlot;

/// Value retirement utilities.
///
/// Provides a consistent API for value pointer retirement. Delegates
/// directly to seize's `defer_retire`, which already performs internal batching.
#[derive(Debug)]
pub struct BatchedRetire;

impl BatchedRetire {
    /// Defer retirement of a value using `S::cleanup_value_ptr`.
    ///
    /// This is the main entry point for value retirement in the tree.
    ///
    /// # Safety
    ///
    /// - `ptr` must be a valid value pointer for slot type `S`
    /// - `ptr` must not be accessed after this call
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// if S::NEEDS_RETIREMENT {
    ///     unsafe {
    ///         BatchedRetire::defer_value::<S>(old_ptr, guard);
    ///     }
    /// }
    /// ```
    #[inline(always)]
    pub unsafe fn defer_value<S: ValueSlot>(ptr: *mut u8, guard: &LocalGuard<'_>) {
        // SAFETY: Caller guarantees ptr is valid for S::cleanup_value_ptr
        unsafe {
            guard.defer_retire(ptr, |p: *mut u8, _| {
                S::cleanup_value_ptr(p);
            });
        }
    }

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
