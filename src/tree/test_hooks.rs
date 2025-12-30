//! Test hooks for deterministic concurrency testing.
//!
//! These hooks allow tests to inject barriers/callbacks at specific points
//! in the split protocol to force specific interleavings.
//!
//! # Usage
//!
//! ```rust,ignore
//! use std::sync::Barrier;
//!
//! let barrier = Arc::new(Barrier::new(2));
//! let b = barrier.clone();
//!
//! // Pause after freezing permutation during split
//! set_after_freeze_hook(Box::new(move || {
//!     b.wait(); // Synchronize with another thread
//! }));
//! ```

#![expect(clippy::unwrap_used, reason = "Fail fast in tests")]

use std::sync::{Mutex, OnceLock};

/// Hook type: a boxed closure that takes no arguments.
pub type TestHook = Box<dyn Fn() + Send + Sync>;

/// Hook called immediately after freeze succeeds in split.
///
/// This is the point where the split has:
/// - Acquired the leaf lock
/// - Called `mark_split()`
/// - Successfully frozen the permutation
///
/// The split has NOT yet moved any entries.
static AFTER_FREEZE_HOOK: OnceLock<Mutex<Option<TestHook>>> = OnceLock::new();

/// Set the hook called after freeze succeeds.
///
/// # Panics
/// Panics if the hook was already set and not cleared.
pub fn set_after_freeze_hook(hook: TestHook) {
    let cell = AFTER_FREEZE_HOOK.get_or_init(|| Mutex::new(None));
    let mut guard = cell.lock().unwrap();

    assert!(
        !guard.is_some(),
        "AFTER_FREEZE_HOOK already set; call clear_after_freeze_hook first"
    );

    *guard = Some(hook);
}

/// Clear the after-freeze hook.
///
/// # Panics
///
/// Panics if the internal mutex is poisoned.
pub fn clear_after_freeze_hook() {
    if let Some(cell) = AFTER_FREEZE_HOOK.get() {
        let mut guard = cell.lock().unwrap();
        *guard = None;
    }
}

/// Call the after-freeze hook if set.
///
/// # Panics
///
/// Panics if the internal mutex is poisoned.
pub fn call_after_freeze_hook() {
    if let Some(cell) = AFTER_FREEZE_HOOK.get() {
        let guard = cell.lock().unwrap();
        if let Some(ref hook) = *guard {
            hook();
        }
    }
}

/// Clear all test hooks.
///
/// Should be called in test teardown to avoid cross-test interference.
pub fn clear_all_hooks() {
    clear_after_freeze_hook();
}
