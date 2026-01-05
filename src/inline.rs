//! True-inline value storage for leaf nodes.
//!
//! This module provides infrastructure for storing small [`Copy`] values
//! directly in leaf nodes without heap allocation.
//!
//! # Modules
//!
//! - [`bits`]: The [`InlineBits`](bits::InlineBits) trait for value encoding/decoding
//! - [`sentinel`]: Sentinel pointer utilities for distinguishing inline vs heap values
//! - [`leaf15_true`]: True-inline leaf node with WIDTH=15

pub mod bits;
pub mod leaf15_true;
pub mod sentinel;
