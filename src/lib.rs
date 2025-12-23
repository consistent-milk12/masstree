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

// Global allocator selection (enabled via features)
#[cfg(feature = "mimalloc")]
use mimalloc::MiMalloc;

#[cfg(feature = "mimalloc")]
#[global_allocator]
static GLOBAL: MiMalloc = MiMalloc;

/// Initialize tracing subscriber when the `tracing` feature is enabled.
///
/// Call this at the start of tests or `main()` to enable logging.
/// Respects `RUST_LOG` env var (e.g., `RUST_LOG=masstree=debug`).
/// If `RUST_LOG` is not set, defaults to "warn" level.
///
/// Safe to call multiple times - only the first call takes effect.
///
/// # Example
///
/// ```rust,ignore
/// masstree::init_tracing();
/// // Now tracing macros will output
/// ```
#[cfg(feature = "tracing")]
pub fn init_tracing() {
    use std::sync::Once;
    static INIT: Once = Once::new();

    INIT.call_once(|| {
        use tracing_subscriber::{EnvFilter, fmt, prelude::*};
        let filter =
            EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("warn"));
        // Use try_init to avoid panic if tests set their own subscriber
        let _ = tracing_subscriber::registry()
            .with(fmt::layer().compact().with_thread_ids(true))
            .with(filter)
            .try_init();
    });
}

/// No-op when tracing feature is disabled.
#[cfg(not(feature = "tracing"))]
pub fn init_tracing() {}

pub mod alloc;
pub mod freeze;
pub mod internode;
pub mod key;
pub mod ksearch;
pub mod leaf;
pub mod nodeversion;
pub mod ordering;
pub mod permuter;
pub mod prefetch;
pub mod slot;
pub mod suffix;
mod tracing_helpers;
pub mod tree;

// Re-export freeze types for convenience
pub use freeze::{AlreadyFrozen, FreezeGuard, Frozen, LeafFreezeUtils};

// Re-export main types for convenience
pub use alloc::{ArenaAllocator, NodeAllocator, SeizeAllocator};
pub use slot::ValueSlot;
pub use suffix::{PermutationProvider, SuffixBag};
pub use tree::{MassTree, MassTreeIndex};

// Re-export debug counters for diagnosis (lightweight, always-on)
pub use tree::{
    ADVANCE_BLINK_COUNT, BLINK_SHOULD_FOLLOW_COUNT, CAS_INSERT_FALLBACK_COUNT,
    CAS_INSERT_RETRY_COUNT, CAS_INSERT_SUCCESS_COUNT, DebugCounters, LOCKED_INSERT_COUNT,
    SEARCH_NOT_FOUND_COUNT, SPLIT_COUNT, WRONG_LEAF_INSERT_COUNT, get_all_debug_counters,
    get_debug_counters, reset_debug_counters,
};
