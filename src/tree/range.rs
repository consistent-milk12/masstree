//! Filepath: src/tree/range/mod.rs
//!
//! Range scan module for [`MassTree`].
//!
//! This module implements ordered range iteration over the tree.
//!
//! # Module Organization
//!
//! - `cursor_key`: Mutable key buffer for scan position tracking
//! - `scan_state`: State machine types ([`ScanState`], [`ScanStackElement`], etc.)
//! - `helper`: Forward scan helper and lower bound search
//! - `find`: Core algorithm (`find_initial`, `find_next`, `find_retry`)
//! - `iterator`: [`RangeIter`] and [`ScanEntry`] types
//! - `api`: Public API methods on [`MassTreeGeneric`]
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
#[allow(
    dead_code,
    reason = "scaffolding for future features (reverse iteration, layer optimization)"
)]
mod cursor_key;
#[allow(dead_code, reason = "scaffolding for future features")]
mod find;
#[allow(
    dead_code,
    reason = "scaffolding for future features (reverse iteration)"
)]
mod helper;
mod iterator;
#[allow(dead_code, reason = "scaffolding for future features")]
mod scan_state;

// Re-export public types
pub use iterator::{KeysIter, RangeBound, RangeIter, ScanEntry, ValuesIter};
