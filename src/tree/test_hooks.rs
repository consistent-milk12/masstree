//! Test hooks for deterministic concurrency testing.
//!
//! These hooks allow tests to inject barriers/callbacks at specific points
//! in the CAS insert and split protocols to force specific interleavings.
//!
//! # Usage
//!
//! ```rust,ignore
//! use std::sync::Barrier;
//!
//! // Set up a barrier to synchronize two threads
//! let barrier = Arc::new(Barrier::new(2));
//! let b1 = barrier.clone();
//! let b2 = barrier.clone();
//!
//! // Thread 1: CAS insert - pause before publishing
//! set_before_cas_publish_hook(Box::new(move || {
//!     b1.wait(); // Wait for thread 2 to freeze
//!     b1.wait(); // Wait for thread 2 to complete split
//! }));
//!
//! // Thread 2: Split - pause after freezing
//! set_after_freeze_hook(Box::new(move || {
//!     b2.wait(); // Signal thread 1 can proceed
//! }));
//! ```
//!
//! # Safety
//!
//! These hooks are only available in test builds (`#[cfg(test)]`).
//! They use `OnceLock<Mutex<Option<...>>>` to allow setting/clearing hooks.

#![expect(clippy::unwrap_used, reason = "Fail fast in tests")]

use std::sync::{Mutex, OnceLock};

/// Hook type: a boxed closure that takes no arguments.
pub type TestHook = Box<dyn Fn() + Send + Sync>;

/// Hook called just before `cas_permutation_raw()` in CAS insert.
///
/// This is the point where the CAS insert has:
/// - Claimed a slot (NULL→ptr CAS succeeded)
/// - Written key metadata (ikey, keylenx)
/// - Is about to publish via permutation CAS
///
/// A split that freezes at this point will cause the CAS to fail.
static BEFORE_CAS_PUBLISH_HOOK: OnceLock<Mutex<Option<TestHook>>> = OnceLock::new();

/// Hook called immediately after freeze succeeds in split.
///
/// This is the point where the split has:
/// - Acquired the leaf lock
/// - Called `mark_split()`
/// - Successfully frozen the permutation
///
/// The split has NOT yet moved any entries.
static AFTER_FREEZE_HOOK: OnceLock<Mutex<Option<TestHook>>> = OnceLock::new();

/// Set the hook called before CAS permutation publish.
///
/// # Panics
/// Panics if the hook was already set and not cleared.
pub fn set_before_cas_publish_hook(hook: TestHook) {
    let cell = BEFORE_CAS_PUBLISH_HOOK.get_or_init(|| Mutex::new(None));
    let mut guard = cell.lock().unwrap();
    assert!(
        !guard.is_some(),
        "BEFORE_CAS_PUBLISH_HOOK already set; call clear_before_cas_publish_hook first"
    );
    *guard = Some(hook);
}

/// Clear the before-CAS-publish hook.
///
/// # Panics
///
/// Panics if the internal mutex is poisoned.
pub fn clear_before_cas_publish_hook() {
    if let Some(cell) = BEFORE_CAS_PUBLISH_HOOK.get() {
        let mut guard = cell.lock().unwrap();
        *guard = None;
    }
}

/// Call the before-CAS-publish hook if set.
pub(super) fn call_before_cas_publish_hook() {
    if let Some(cell) = BEFORE_CAS_PUBLISH_HOOK.get() {
        let guard = cell.lock().unwrap();
        if let Some(ref hook) = *guard {
            hook();
        }
    }
}

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
    clear_before_cas_publish_hook();
    clear_after_freeze_hook();
}
