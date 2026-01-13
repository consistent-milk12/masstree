//! Branch prediction hints for performance-critical paths.
//!
//! These functions provide hints to the compiler about branch probability,
//! matching C++'s `__builtin_expect()` / `likely()` / `unlikely()` macros.
//!
//! # Usage
//!
//! ```ignore
//! use crate::hints::{likely, unlikely};
//!
//! if likely(common_case) {
//!     // fast path - compiler optimizes for this
//! }
//!
//! if unlikely(error_condition) {
//!     // slow path - compiler deprioritizes this
//! }
//! ```
//!
//! # Implementation
//!
//! Uses the `#[cold]` attribute on a helper function to signal branch rarity.
//! This is a stable Rust pattern that achieves similar effects to `__builtin_expect`.

/// Cold path marker - never actually called, just influences codegen.
#[inline(always)]
#[cold]
const fn cold_path() {}

/// Hint that a condition is likely to be true.
///
/// Use this for common/fast-path conditions to guide branch prediction.
///
/// # Example
///
/// ```ignore
/// if likely(!version.has_changed()) {
///     // fast path: no concurrent modification
/// } else {
///     // slow path: retry
/// }
/// ```
#[inline(always)]
#[must_use]
pub const fn likely(b: bool) -> bool {
    if !b {
        cold_path();
    }
    b
}

/// Hint that a condition is unlikely to be true.
///
/// Use this for error conditions and rare branches.
///
/// # Example
///
/// ```ignore
/// if unlikely(slot >= WIDTH) {
///     return Err(OutOfBounds);
/// }
/// ```
#[inline(always)]
#[must_use]
pub const fn unlikely(b: bool) -> bool {
    if b {
        cold_path();
    }

    b
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_likely_true() {
        assert!(likely(true));
        assert!(!likely(false));
    }

    #[test]
    fn test_unlikely_true() {
        assert!(unlikely(true));
        assert!(!unlikely(false));
    }
}
