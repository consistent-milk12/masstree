//! # Masstree
//!
//! A high-performance concurrent key-value store based on a trie of B+trees.
//!
//! Masstree combines the strengths of tries and B+trees:
//! - Trie structure for variable-length key prefixes (8-byte chunks)
//! - B+tree at each trie node for the current 8-byte slice
//! - Optimistic concurrency control for reads (no locks)
//! - Fine-grained locking for writes
//!
//! ## Design
//!
//! Keys are split into 8-byte slices. Each slice is handled by a B+tree.
//! When a key is longer than 8 bytes, the B+tree leaf points to another
//! layer (another B+tree for the next 8 bytes).
//!
//! ## Performance
//!
//! - Lookups: Lock-free using version numbers
//! - Updates: Lock only affected nodes
//! - Cache-friendly: 8-byte key slices fit in registers

pub mod key;
