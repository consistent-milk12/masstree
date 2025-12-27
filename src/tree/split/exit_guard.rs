//! Filepath: `src/tree/split/exit_guard.rs`
//!
//! Adopted from `sdd` crate.
//! Simple scope-exit callback for non-lock cleanup scenarios
//!
//! While [`PropagationContext`] handles the complex lock lifetime unification,
//! [`ExitGuard`] provides a simpler pattern for callback-on-drop cleanup.

use std::mem::{ManuallyDrop, forget};
use std::ops::{Deref, DerefMut};

/// RAII guard that calls a closure on drop.
/// Call `.forget()` to skip the callback on success.
pub struct ExitGuard<T, F: FnOnce(T)> {
    inner: ManuallyDrop<(T, F)>,
}

impl<T, F: FnOnce(T)> ExitGuard<T, F> {
    #[inline]
    pub(crate) const fn new(captured: T, f: F) -> Self {
        Self {
            inner: ManuallyDrop::new((captured, f)),
        }
    }

    /// Forgets the guard, the callback will NOT be called.
    #[inline(always)]
    pub(crate) fn forget(mut self) {
        // SAFETY: We're about to forget self, so inner won't be dropped twice
        unsafe {
            ManuallyDrop::drop(&mut self.inner);
        }

        forget(self);
    }
}

impl<T, F: FnOnce(T)> Drop for ExitGuard<T, F> {
    #[inline]
    fn drop(&mut self) {
        let (captured, f) = unsafe { ManuallyDrop::take(&mut self.inner) };

        f(captured);
    }
}

impl<T, F: FnOnce(T)> Deref for ExitGuard<T, F> {
    type Target = T;

    #[inline]
    fn deref(&self) -> &Self::Target {
        &self.inner.0
    }
}

impl<T, F: FnOnce(T)> DerefMut for ExitGuard<T, F> {
    #[inline]
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.inner.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::cell::Cell;

    #[test]
    fn test_runs_on_drop() {
        let ran = Cell::new(false);
        {
            let _guard = ExitGuard::new((), |()| ran.set(true));
        }
        assert!(ran.get());
    }

    #[test]
    fn test_forget_skips_callback() {
        let ran = Cell::new(false);
        {
            let guard = ExitGuard::new((), |()| ran.set(true));
            guard.forget();
        }
        assert!(!ran.get());
    }

    #[test]
    fn test_captures_value() {
        let result = Cell::new(0);
        {
            let guard = ExitGuard::new(42, |v| result.set(v));
            assert_eq!(*guard, 42);
        }
        assert_eq!(result.get(), 42);
    }

    #[test]
    fn test_deref_mut() {
        let result = Cell::new(0);
        {
            let mut guard = ExitGuard::new(10, |v| result.set(v));
            *guard = 20; // Modify captured value
        }
        assert_eq!(result.get(), 20);
    }
}
