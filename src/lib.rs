//! # `MassTree`
//!
//! A concurrent ordered map based on a trie of B+trees.
//!
//! This crate implements Masstree, combining tries and B+trees:
//! - Trie structure for variable-length key prefixes (8-byte chunks)
//! - B+tree at each trie node for the current 8-byte slice
//! - Cache-friendly: 8-byte key slices fit in registers
//!
//! ## Status: Alpha
//!
//! **Not production ready.** Core concurrent operations work but memory
//! reclamation is incomplete and range scans/deletion are not implemented.
//!
//! | Feature | Status |
//! |---------|--------|
//! | Concurrent get | Works (lock-free, version-validated) |
//! | Concurrent insert | Works (CAS fast path + locked fallback) |
//! | Split propagation | Works (leaf and internode) |
//! | Memory reclamation | Partial (nodes not freed until tree drop) |
//! | Range scans | Not implemented |
//! | Deletion | Not implemented |
//!
//! ## Thread Safety
//!
//! `MassTree<V>` is `Send + Sync` when `V: Send + Sync`. Concurrent access
//! requires using the guard-based API:
//!
//! ```rust
//! use masstree::MassTree;
//!
//! let tree: MassTree<u64> = MassTree::new();
//! let guard = tree.guard();
//!
//! // Concurrent get (lock-free)
//! let value = tree.get_with_guard(b"key", &guard);
//!
//! // Concurrent insert (fine-grained locking)
//! let old = tree.insert_with_guard(b"key", 42, &guard);
//! ```
//!
//! The non-guard methods (`get`, `insert`) exist for convenience but require
//! `&mut self` for insert, making them unsuitable for concurrent use.
//!
//! ## Key Constraints
//!
//! - Keys must be 0-256 bytes. Longer keys will panic.
//! - Keys are byte slices (`&[u8]`), not generic types.
//!
//! ## Design
//!
//! Keys are split into 8-byte slices. Each slice is handled by a B+tree.
//! When a key is longer than 8 bytes, the B+tree leaf points to another
//! layer (another B+tree for the next 8 bytes).
//!
//! ## Value Storage
//!
//! - **`MassTree<V>`**: Stores values as `Arc<V>`. Returns `Arc<V>` on get.
//! - **`MassTreeIndex<V: Copy>`**: Convenience wrapper that copies values.
//!   Note: Currently wraps `MassTree<V>` internally; true inline storage is
//!   planned for a future release.

#![deny(missing_docs)]
#![warn(clippy::pedantic)]
#![warn(clippy::nursery)]
// We use extensive benchmarking to verify #[inline(always)] placement is correct.
#![allow(clippy::inline_always)]

pub mod alloc;
pub mod internode;
pub mod key;
pub mod ksearch;
pub mod leaf;
pub mod nodeversion;
pub mod ordering;
pub mod permuter;
pub mod slot;
pub mod suffix;
pub mod tree;

// Re-export main types for convenience
pub use alloc::{ArenaAllocator, NodeAllocator, SeizeAllocator};
pub use slot::ValueSlot;
pub use suffix::{PermutationProvider, SuffixBag};
pub use tree::{MassTree, MassTreeIndex};
