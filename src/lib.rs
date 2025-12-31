//! # `MassTree`
//!
//! A concurrent ordered map based on a trie of B+trees.
//!
//! This crate implements Masstree, combining tries and B+trees:
//! - Trie structure for variable-length key prefixes (8-byte chunks)
//! - B+tree at each trie node for the current 8-byte slice
//! - Cache-friendly: 8-byte key slices fit in registers
//!
//! ## Status: Beta (v0.2.0)
//!
//! Core operations work. Range scans implemented. Deletion planned for v0.3.0.
//!
//! | Feature | Status |
//! |---------|--------|
//! | Concurrent get | Lock-free, version-validated |
//! | Concurrent insert | Fine-grained leaf locking |
//! | Split propagation | Leaf and internode |
//! | Range scans | `scan`, `scan_ref`, `scan_prefix` |
//! | Memory reclamation | Partial (values freed, nodes at tree drop) |
//! | Deletion | Planned (v0.3.0) |
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
//!
//! ## Feature Flags
//!
//! ### `tracing`
//!
//! Enables structured logging for debugging concurrent operations.
//! See the crate-level documentation for details.
//!
//! ### `mimalloc`
//!
//! Uses mimalloc as the global allocator for improved performance.

#![deny(missing_docs)]
#![warn(clippy::pedantic)]
#![warn(clippy::nursery)]
// We use extensive benchmarking to verify #[inline(always)] placement is correct.
#![allow(clippy::inline_always)]
#![allow(clippy::multiple_crate_versions)]

// Global allocator selection (enabled via features)
#[cfg(feature = "mimalloc")]
use mimalloc::MiMalloc;

#[cfg(feature = "mimalloc")]
#[global_allocator]
static GLOBAL: MiMalloc = MiMalloc;

/// Initialize tracing with file and console output.
///
/// Writes NDJSON logs to `logs/masstree.jsonl` for structured analysis.
/// Also outputs to console unless `MASSTREE_LOG_CONSOLE=0` is set.
///
/// Safe to call multiple times - only the first call takes effect.
///
/// # Environment Variables
///
/// - `RUST_LOG`: Filter directives (default: `masstree=info`). Example: `masstree=debug,masstree::tree::locked=trace`
/// - `MASSTREE_LOG_DIR`: Log directory (default: `logs/`)
/// - `MASSTREE_LOG_CONSOLE`: Set to `0` to disable console output
///
/// # Example
///
/// ```bash
/// # Run with debug logging to file, no console spam
/// RUST_LOG=masstree=debug MASSTREE_LOG_CONSOLE=0 cargo run --features tracing
///
/// # Analyze logs with jq
/// cat logs/masstree.jsonl | jq 'select(.fields.ikey != null)'
/// ```
#[cfg(feature = "tracing")]
pub fn init_tracing() {
    use std::env;
    use std::sync::Once;
    static INIT: Once = Once::new();

    // Store the guard in a static to keep the non-blocking writer alive.
    // Leaking is intentional - we want logs to flush on process exit.
    static mut GUARD: Option<tracing_appender::non_blocking::WorkerGuard> = None;

    INIT.call_once(|| {
        use tracing_subscriber::{EnvFilter, Layer, layer::SubscriberExt, util::SubscriberInitExt};

        // Configuration from environment
        let log_dir = env::var("MASSTREE_LOG_DIR").unwrap_or_else(|_| "logs".to_string());
        let console_enabled = false;
        let filter_str = env::var("RUST_LOG").unwrap_or_else(|_| "masstree=info".to_string());

        // Create log directory
        let _ = std::fs::create_dir_all(&log_dir);

        // File appender - non-rotating, writes to masstree.jsonl (NDJSON)
        let file_appender = tracing_appender::rolling::never(&log_dir, "masstree.jsonl");
        let (non_blocking, guard) = tracing_appender::non_blocking(file_appender);

        // Store guard to prevent dropping (logs would be lost)
        // SAFETY: This is only called once due to Once::call_once, and we never read GUARD
        // after this point except implicitly when the process exits.
        #[allow(static_mut_refs)]
        unsafe {
            GUARD = Some(guard);
        }

        // File layer - JSON format for structured analysis
        let file_layer = tracing_subscriber::fmt::layer()
            .with_writer(non_blocking)
            .with_thread_ids(true)
            .with_thread_names(true)
            .with_target(true)
            .with_file(true)
            .with_line_number(true)
            .json()
            .with_filter(
                EnvFilter::try_new(&filter_str).unwrap_or_else(|_| EnvFilter::new("info")),
            );

        // Console layer - compact format for human reading
        let console_layer = if console_enabled {
            Some(
                tracing_subscriber::fmt::layer()
                    .with_thread_ids(true)
                    .compact()
                    .with_filter(
                        EnvFilter::try_new(&filter_str).unwrap_or_else(|_| EnvFilter::new("trace")),
                    ),
            )
        } else {
            None
        };

        // Use try_init to avoid panic if tests set their own subscriber
        let _ = tracing_subscriber::registry()
            .with(file_layer)
            .with(console_layer)
            .try_init();
    });
}

/// No-op when tracing feature is disabled.
#[cfg(not(feature = "tracing"))]
pub const fn init_tracing() {}

pub mod alloc24;
pub mod alloc_trait;
pub mod internode;
pub mod key;
pub mod ksearch;
pub mod leaf24;
pub mod leaf_trait;
pub mod link;
pub mod nodeversion;
pub mod ordering;
pub mod permuter;
pub mod permuter24;
pub mod prefetch;
pub mod slot;
pub mod suffix;
pub mod tree;
pub mod value;

// Re-export freeze types for convenience

// Re-export leaf node traits for generic tree operations
pub use leaf_trait::{TreeInternode, TreeLeafNode, TreePermutation};

// Re-export allocator trait for generic tree operations
pub use alloc_trait::NodeAllocatorGeneric;

// Re-export Permuter24 types
pub use permuter24::{AtomicPermuter24, Permuter24, WIDTH_24};

// Re-export LeafNode24 types
pub use leaf24::{
    LeafNode24, MODSTATE_DELETED_LAYER, MODSTATE_INSERT, MODSTATE_REMOVE,
    WIDTH_24 as LEAF24_WIDTH,
};

// Re-export allocator24 types
pub use alloc24::SeizeAllocator24;

// Re-export value types
pub use value::{InsertTarget, LeafValue, LeafValueIndex, SplitPoint};

// Re-export link utilities
pub use link::{is_marked, mark_ptr, unmark_ptr};

// Re-export main types for convenience
pub use slot::ValueSlot;
pub use suffix::{InlineSuffixBag, PermutationProvider, SuffixBag};
pub use tree::RemoveError;
pub use tree::{KeysIter, RangeBound, RangeIter, ScanEntry, ValuesIter};
pub use tree::{MassTree, MassTree24, MassTree24Inline, MassTreeGeneric, MassTreeIndex};
