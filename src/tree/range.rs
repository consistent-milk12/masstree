//! Filepath: src/tree/range/mod.rs
//!
//! Range scan module for [`crate::MassTree`].
//!
//! This module implements ordered range iteration over the tree.
//!
//! # Module Organization
//!
//! - `cursor_key`: Mutable key buffer for scan position tracking
//! - `scan_state`: Shared state, signed leaf position and layer stack
//! - `helper`: Forward scan helper and lower bound search
//! - `scan_ctx`: Direction-parametric readers and iterator/batch drivers
//! - `iterator`: [`RangeIter`] and [`ScanEntry`] types
//! - `api`: Public API methods on [`crate::MassTreeGeneric`]
//!
//! # Public API
//!
//! The main public types are:
//! - [`RangeBound`]: Start/end bound specification
//! - [`ScanEntry`]: Key-value entry returned by iterator
//! - [`RangeIter`]: Iterator over a key range
//!
//! # Example
//!
//! ```ignore
//! use masstree::{MassTree, RangeBound};
//!
//! let tree = MassTree::<String>::new();
//! // ... insert entries ...
//!
//! let guard = tree.guard();
//! for entry in tree.range(
//!     RangeBound::Included(b"start"),
//!     RangeBound::Excluded(b"end"),
//!     &guard
//! ) {
//!     println!("{:?} -> {}", entry.key, entry.value);
//! }
//!

mod api;
mod api_ref;
mod batch_common;
mod cursor_key;
mod direction;
mod forward_ctx;
mod helper;
mod iterator;
mod reverse_ctx;
mod scan_ctx;
mod scan_state;
mod traversal;

// Re-export public types
pub use iterator::{KeysIter, RangeBound, RangeIter, ScanEntry, ValuesIter};
