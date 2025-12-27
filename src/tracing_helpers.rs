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
//!
//! # Function Instrumentation
//!
//! Use `#[cfg_attr(feature = "tracing", tracing::instrument)]` to add entry/exit tracing
//! to functions. This compiles to a no-op when tracing is disabled.
//!
//! ```ignore
//! #[cfg_attr(feature = "tracing", tracing::instrument(level = "debug", skip(self, guard)))]
//! fn reach_leaf_concurrent_generic(&self, start: *const u8, key: &Key<'_>, guard: &Guard) -> *mut L {
//!     // Function body - entry/exit times will be logged
//! }
//! ```
//!
//! Common `#[instrument]` options:
//! - `level = "trace|debug|info|warn|error"` - Set the tracing level
//! - `skip(arg1, arg2)` - Skip logging specific arguments (useful for large/non-Debug types)
//! - `skip_all` - Skip all arguments
//! - `fields(custom = "value")` - Add custom fields to the span
//! - `name = "custom_name"` - Override the span name (default is function name)
//! - `err` - Log if the function returns an error

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
