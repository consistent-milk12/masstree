//! Zero-cost tracing helpers.
//!
//! When the `tracing` feature is enabled, these macros forward to the `tracing` crate.
//! When disabled (default), they compile to no-ops with zero runtime overhead.
//!
//! # Usage
//!
//! ```bash
//! # Normal build - no tracing overhead
//! cargo build --release
//!
//! # Debug build with tracing enabled
//! cargo test --features tracing
//!
//! # Run specific test with tracing
//! RUST_LOG=masstree::tree::locked=trace cargo test --features tracing stress_concurrent
//! ```

#![allow(unused_macros, unused_imports)]

/// Trace-level logging (most verbose). Compiles to no-op without `tracing` feature.
#[cfg(feature = "tracing")]
macro_rules! trace_log {
    ($($arg:tt)*) => {
        tracing::trace!($($arg)*)
    };
}

#[cfg(not(feature = "tracing"))]
macro_rules! trace_log {
    ($($arg:tt)*) => {
        // Completely empty - zero cost
    };
}

/// Debug-level logging. Compiles to no-op without `tracing` feature.
#[cfg(feature = "tracing")]
macro_rules! debug_log {
    ($($arg:tt)*) => {
        tracing::debug!($($arg)*)
    };
}

#[cfg(not(feature = "tracing"))]
macro_rules! debug_log {
    ($($arg:tt)*) => {};
}

/// Warn-level logging. Compiles to no-op without `tracing` feature.
#[cfg(feature = "tracing")]
macro_rules! warn_log {
    ($($arg:tt)*) => {
        tracing::warn!($($arg)*)
    };
}

#[cfg(not(feature = "tracing"))]
macro_rules! warn_log {
    ($($arg:tt)*) => {};
}

/// Error-level logging. Compiles to no-op without `tracing` feature.
#[cfg(feature = "tracing")]
macro_rules! error_log {
    ($($arg:tt)*) => {
        tracing::error!($($arg)*)
    };
}

#[cfg(not(feature = "tracing"))]
macro_rules! error_log {
    ($($arg:tt)*) => {};
}

// Export macros for use within crate
pub(crate) use debug_log;
pub(crate) use error_log;
pub(crate) use trace_log;
pub(crate) use warn_log;
