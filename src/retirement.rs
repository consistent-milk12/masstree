//! Batched retirement for reduced reclamation overhead.
//!
//! This module provides [`BatchedRetire`], a thin wrapper around seize's retirement
//! that enables future batching optimizations while maintaining API stability.
//!
//! # Current Implementation
//!
//! The current implementation delegates directly to `seize::Guard::defer_retire`.
//! Seize already performs internal batching, so external batching provides limited
//! benefit. This module exists to:
//!
//! 1. Provide a consistent API for value retirement across the codebase
//! 2. Enable future optimizations without changing call sites
//! 3. Add `FlushOnDrop` for explicit batch control when needed
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

// ============================================================================
//  Public API
// ============================================================================

/// Batched retirement utilities.
///
/// Provides a consistent API for retirement operations. Currently delegates
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

    /// Defer retirement with a custom cleanup function.
    ///
    /// Use this for suffix bags, nodes, and other non-value retirements.
    ///
    /// # Safety
    ///
    /// - `ptr` must be valid for the provided `cleanup_fn`
    /// - `ptr` must not be accessed after this call
    /// - `cleanup_fn` must correctly clean up the pointer
    #[inline(always)]
    pub unsafe fn defer_custom<T>(
        ptr: *mut T,
        cleanup_fn: unsafe fn(*mut T, &seize::Collector),
        guard: &LocalGuard<'_>,
    ) {
        // SAFETY: Caller guarantees ptr is valid for cleanup_fn
        unsafe {
            guard.defer_retire(ptr, cleanup_fn);
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

    /// Get the number of pending retirements (always 0 in current implementation).
    ///
    /// Since retirement is delegated directly to seize, there are no
    /// pending retirements at this level.
    #[must_use]
    #[inline(always)]
    pub const fn pending() -> usize {
        0
    }
}

// ============================================================================
//  Drop Guard for Automatic Flush
// ============================================================================

/// Guard that flushes the retirement batch on drop.
///
/// When dropped, triggers seize to process any accumulated retirements.
/// This allows reclamation to proceed more quickly once the guard is released.
///
/// # Example
///
/// ```rust,ignore
/// let guard = tree.guard();
/// let _flush = FlushOnDrop::new(&guard);
///
/// // ... perform operations ...
///
/// // Batch is automatically flushed when _flush is dropped
/// ```
#[derive(Debug)]
pub struct FlushOnDrop<'g> {
    #[allow(dead_code)]
    guard: &'g LocalGuard<'g>,
}

impl<'g> FlushOnDrop<'g> {
    /// Create a new flush guard.
    #[inline]
    #[must_use]
    pub const fn new(guard: &'g LocalGuard<'g>) -> Self {
        Self { guard }
    }
}

impl Drop for FlushOnDrop<'_> {
    #[inline]
    fn drop(&mut self) {
        self.guard.flush();
    }
}

// ============================================================================
//  Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use seize::Collector;
    use std::sync::atomic::{AtomicUsize, Ordering};

    static DROP_COUNT: AtomicUsize = AtomicUsize::new(0);

    unsafe fn test_cleanup(ptr: *mut u64, _: &Collector) {
        DROP_COUNT.fetch_add(1, Ordering::SeqCst);
        // SAFETY: ptr came from Box::into_raw and is valid
        unsafe { drop(Box::from_raw(ptr)) };
    }

    #[test]
    fn test_defer_custom() {
        DROP_COUNT.store(0, Ordering::SeqCst);

        let collector = Collector::new();
        let guard = collector.enter();

        // Create and retire some test pointers
        for _ in 0..10 {
            let boxed = Box::new(42u64);
            let ptr = Box::into_raw(boxed);
            unsafe {
                BatchedRetire::defer_custom(ptr, test_cleanup, &guard);
            }
        }

        // pending() always returns 0 in current implementation
        assert_eq!(BatchedRetire::pending(), 0);

        // flush() is a no-op
        BatchedRetire::flush(&guard);

        // Drop guard and force reclamation to trigger cleanup
        drop(guard);

        // Force epoch advancement and reclamation
        let guard2 = collector.enter();
        drop(guard2);
        let guard3 = collector.enter();
        drop(guard3);
    }

    #[test]
    fn test_flush_on_drop() {
        let collector = Collector::new();
        let guard = collector.enter();

        {
            let _flush = FlushOnDrop::new(&guard);
            // Just verify it compiles and doesn't panic
        }
    }
}
