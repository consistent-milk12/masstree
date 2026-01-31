//! Error types for `MassTree` operations.
//!
//! This module previously contained fallible allocation error types.
//! Those have been removed - allocations are now infallible (abort on OOM).
//!
//! User-facing errors like `InsertError` and `RemoveError` are defined
//! in `src/tree.rs`.
