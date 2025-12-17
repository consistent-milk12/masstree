//! # `MassTree`
//!
//! A high-performance key-value store based on a trie of B+trees.
//!
//! This crate implements Masstree, combining tries and B+trees:
//! - Trie structure for variable-length key prefixes (8-byte chunks)
//! - B+tree at each trie node for the current 8-byte slice
//! - Cache-friendly: 8-byte key slices fit in registers
//!
//! ## Current Status: Phase 1 (Single-Threaded)
//!
//! This implementation is **single-threaded only**. The concurrent features
//! (optimistic reads, CAS-based locking, epoch-based reclamation) are planned
//! for Phase 3 but not yet implemented.
//!
//! **Phase 1 constraints:**
//! - Keys must be 0-8 bytes (longer keys return `KeyTooLong` error)
//! - Single-layer only (no trie traversal for multi-slice keys)
//! - `MassTree` is `!Send` and `!Sync` to prevent accidental concurrent use
//!
//! ## Design
//!
//! Keys are split into 8-byte slices. Each slice is handled by a B+tree.
//! When a key is longer than 8 bytes, the B+tree leaf points to another
//! layer (another B+tree for the next 8 bytes). Layer traversal will be
//! implemented in Phase 2.
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

pub mod internode;
pub mod key;
pub mod ksearch;
pub mod leaf;
pub mod nodeversion;
pub mod permuter;
pub mod suffix;
pub mod tree;

// Re-export main types for convenience
pub use suffix::{PermutationProvider, SuffixBag};
pub use tree::{MassTree, MassTreeIndex};
