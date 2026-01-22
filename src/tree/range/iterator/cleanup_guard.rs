//! Filepath: `src/tree/range/iterator/cleanup_guard.rs`
//!
//! RAII guard for cleaning up output pointers.

use std::marker::PhantomData;

use crate::ref_value_slot::RefValueSlot;
use crate::slot::ValueSlot;

/// Scope guard for cleaning up output pointers on drop (including panics).
///
/// When converting `S::Output` to a raw pointer for zero-copy iteration,
/// we must ensure the pointer is cleaned up even if the visitor panics.
pub(super) struct CleanupGuard<S: ValueSlot> {
    ptr: *mut u8,
    _marker: PhantomData<S>,
}

impl<S: ValueSlot> Drop for CleanupGuard<S> {
    fn drop(&mut self) {
        // SAFETY: ptr was created by S::output_to_raw
        unsafe { S::cleanup_output_raw(self.ptr) };
    }
}

impl<S: ValueSlot> CleanupGuard<S> {
    /// Execute a closure with a borrowed reference to the output value.
    ///
    /// Converts the output to a raw pointer, borrows it as `&S::Value`,
    /// calls the closure, then cleans up the pointer (even on panic).
    ///
    /// # Safety
    ///
    /// This is safe because:
    /// - `S::output_to_raw` returns a properly aligned pointer
    /// - The guard ensures cleanup via Drop
    /// - The reference lifetime is bounded by the closure scope
    #[inline]
    pub fn with_output_ref<R, F>(output: &S::Output, f: F) -> R
    where
        S: RefValueSlot,
        F: FnOnce(&S::Value) -> R,
    {
        let ptr: *mut u8 = S::output_to_raw(output);
        let _guard = Self {
            ptr,
            _marker: PhantomData,
        };

        // SAFETY: ptr is properly aligned (guaranteed by output_to_raw)
        // and valid for the duration of this scope (guard ensures cleanup)
        let value_ref: &S::Value = unsafe { &*ptr.cast::<S::Value>() };
        f(value_ref)
    }
}
