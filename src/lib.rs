//! # [`MassTree`]
//!
//! A high-performance concurrent ordered map for Rust. Stores keys as `&[u8]` and
//! supports variable-length keys by building a trie of B+trees.
//!
//! ## Features
//!
//! - Ordered map for byte keys (lexicographic ordering)
//! - Lock-free reads with version validation
//! - Concurrent inserts and deletes with fine-grained leaf locking
//! - Zero-copy range scans with `scan_ref` and `scan_prefix`
//! - Memory reclamation via epoch-based deferred cleanup
//! - Lazy leaf coalescing for deleted entries
//! - high-performance inline variant [`MassTree15Inline`], all benchmark results are based on it.
//!
//! ## Quick Start
//!
//! ```rust,ignore
//! use masstree::MassTree;
//!
//! let tree: MassTree<u64> = MassTree::new();
//! let guard = tree.guard();
//!
//! // Insert
//! tree.insert_with_guard(b"hello", 123, &guard);
//!
//! // Point lookup (lock-free)
//! assert_eq!(tree.get_ref(b"hello", &guard), Some(&123));
//!
//! // Remove
//! tree.remove_with_guard(b"hello", &guard).unwrap();
//!
//! // Range scan (zero-copy)
//! use masstree::RangeBound;
//! tree.scan_ref(RangeBound::Included(b"a"), RangeBound::Excluded(b"z"), |key, value| {
//!     println!("{:?} -> {}", key, value);
//!     true // continue scanning
//! }, &guard);
//! ```
//!
//! ## Thread Safety
//!
//! `MassTree<V>` is `Send + Sync` when `V: Send + Sync`. Concurrent access
//! requires using the guard-based API. The guard ties operations to an epoch
//! for safe memory reclamation.
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
//! ## Key Constraints
//!
//! - Keys must be 0-256 bytes. Longer keys will panic.
//! - Keys are byte slices (`&[u8]`), not generic types.

#![deny(missing_docs)]
#![warn(clippy::pedantic)]
#![warn(clippy::nursery)]
// We use extensive benchmarking to verify #[inline(always)] placement is correct.
#![allow(clippy::inline_always)]
#![allow(clippy::multiple_crate_versions)]
// Allocation code casts *mut u8 to node pointers after allocating with correct Layout.
// The memory is always properly aligned before casting.
#![allow(clippy::cast_ptr_alignment)]

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
    use std::env as StdEnv;
    use std::fs as StdFs;
    use std::sync::OnceLock;
    use tracing_subscriber::{EnvFilter, Layer, layer::SubscriberExt, util::SubscriberInitExt};

    // Store the guard in a static to keep the non-blocking writer alive.
    // The guard ensures logs are flushed on process exit.
    static GUARD: OnceLock<tracing_appender::non_blocking::WorkerGuard> = OnceLock::new();

    // OnceLock::get_or_init ensures this runs exactly once
    GUARD.get_or_init(|| {
        // Configuration from environment
        let log_dir = StdEnv::var("MASSTREE_LOG_DIR").unwrap_or_else(|_| "logs".to_string());
        let console_enabled = false;
        let filter_str = StdEnv::var("RUST_LOG").unwrap_or_else(|_| "masstree=info".to_string());

        // Create log directory
        let _ = StdFs::create_dir_all(&log_dir);

        // File appender - non-rotating, writes to masstree.jsonl (NDJSON)
        let file_appender = tracing_appender::rolling::never(&log_dir, "masstree.jsonl");
        let (non_blocking, guard) = tracing_appender::non_blocking(file_appender);

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

        // Return the guard to be stored in OnceLock
        guard
    });
}

/// No-op when tracing feature is disabled.
#[cfg(not(feature = "tracing"))]
pub const fn init_tracing() {}

pub mod alloc15;
mod alloc_common;
pub mod alloc_trait;
pub mod hints;
pub mod inline;
pub mod internode;
pub mod key;
pub mod ksearch;
pub mod leaf15;
pub mod leaf_trait;
pub mod link;
pub mod node_pool;
pub mod nodeversion;
pub mod ordering;
pub mod permuter;
pub mod prefetch;
pub mod ref_value_slot;
mod retirement;
mod shard_counter;
pub mod slot;
pub mod suffix;
pub mod tree;
pub mod value;

#[cfg(feature = "insert-stats")]
pub mod insert_stats;

// Note: AllocError/AllocKind/AllocResult removed - allocations are now infallible
pub use retirement::{BatchedRetire, FlushOnDrop};

// Re-export leaf node traits for generic tree operations
pub use leaf_trait::{TreeInternode, TreeLeafNode, TreePermutation};

// Re-export allocator trait for generic tree operations
pub use alloc_trait::NodeAllocatorGeneric;

// Re-export Permuter types
pub use permuter::{AtomicPermuter, AtomicPermuter15, Permuter, Permuter15};

// Re-export leaf node types
pub use leaf15::{LeafNode15, MODSTATE_DELETED_LAYER, MODSTATE_INSERT, MODSTATE_REMOVE, WIDTH_15};

// Re-export internode and version types (for debugging)
pub use internode::InternodeNode;
pub use nodeversion::NodeVersion;

// Re-export allocator types
pub use alloc15::{SeizeAllocator15, SeizeAllocator15TrueInline};

// Re-export inline types for true-inline storage
pub use inline::bits::InlineBits;
pub use inline::leaf15_true::LeafNode15TrueInline;
pub use slot::true_inline::TrueInlineSlot;

// Re-export value types
pub use value::{InsertTarget, LeafValue, LeafValueIndex, SplitPoint};

// Re-export link utilities
pub use link::Linker;

// Re-export main types for convenience
pub use ref_value_slot::RefValueSlot;
pub use slot::ValueSlot;
pub use suffix::{InlineSuffixBag, PermutationProvider, SuffixBag};
pub use tree::{InsertError, RemoveError};
pub use tree::{KeysIter, RangeBound, RangeIter, ScanEntry, ValuesIter};
pub use tree::{MassTree, MassTree15, MassTree15Inline, MassTreeGeneric};

// Re-export batch insert types
pub use tree::{BatchEntry, BatchInsertResult, batch};
