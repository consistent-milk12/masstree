//! Filepath: src/tree.rs
//! `MassTree` - A high-performance concurrent trie of B+trees.
//!
//! This module provides the main `MassTree<V>` and `MassTreeIndex<V>` types.

use std::fmt as StdFmt;
use std::marker::PhantomData;
use std::sync::atomic::{AtomicPtr, AtomicUsize, Ordering as AtomicOrdering};

use crate::alloc_lockfree::LockFreeAllocator;
use crate::alloc_pool::PoolAllocator24;
use crate::alloc24::SeizeAllocator24;
use crate::leaf24::LeafNode24;
use crate::slot::ValueSlot;
use crate::value::{LeafValue, LeafValueIndex};
use seize::Collector;

mod generic;
mod index;
mod optimistic;
mod range;
pub mod remove;
mod split;

#[cfg(test)]
pub mod test_hooks;

pub use index::MassTreeIndex;
pub use range::{KeysIter, RangeBound, RangeIter, ScanEntry, ValuesIter};
pub use remove::RemoveError;

// Re-export debug counters (only when tracing is enabled)
#[cfg(feature = "tracing")]
pub use optimistic::{
    ADVANCE_BLINK_COUNT, BLINK_ADVANCE_ANOMALY_COUNT, BLINK_SHOULD_FOLLOW_COUNT, DebugCounters,
    LOCKED_INSERT_COUNT, PARENT_WAIT_HIT_COUNT, PARENT_WAIT_MAX_NS, PARENT_WAIT_MAX_SPINS,
    PARENT_WAIT_TOTAL_NS, PARENT_WAIT_TOTAL_SPINS, ParentWaitStats, SEARCH_NOT_FOUND_COUNT,
    SPLIT_COUNT, WRONG_LEAF_INSERT_COUNT, get_all_debug_counters, get_debug_counters,
    get_parent_wait_stats, reset_debug_counters,
};

// ============================================================================
//  InsertError
// ============================================================================

/// Errors that can occur during insert operations.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum InsertError {
    /// Leaf node is full and cannot accept more keys.
    /// Caller should trigger a split.
    LeafFull,

    /// Memory allocation failed.
    AllocationFailed,

    /// Split required (generic path).
    /// Leaf is full and needs to be split.
    SplitRequired,

    /// Layer creation required (generic path).
    /// Key conflict requires creating a new sublayer.
    LayerCreationRequired,

    /// Split operation failed (generic path).
    /// Internal error during split - should not happen in normal operation.
    SplitFailed,

    /// Split propagation to parent failed (generic path).
    /// Parent internode is full and needs cascading split.
    SplitPropagationRequired,
}

impl StdFmt::Display for InsertError {
    fn fmt(&self, f: &mut StdFmt::Formatter<'_>) -> StdFmt::Result {
        match self {
            Self::LeafFull => write!(f, "leaf node is full"),

            Self::AllocationFailed => write!(f, "memory allocation failed"),

            Self::SplitRequired => {
                write!(f, "split required (leaf full)")
            }

            Self::LayerCreationRequired => {
                write!(f, "layer creation required (key conflict)")
            }

            Self::SplitFailed => {
                write!(f, "split operation failed")
            }

            Self::SplitPropagationRequired => {
                write!(f, "split propagation required (parent full)")
            }
        }
    }
}

impl std::error::Error for InsertError {}

// ============================================================================
//  MassTreeGeneric - Generic over Leaf Type
// ============================================================================

use crate::alloc_trait::NodeAllocatorGeneric;
use crate::leaf_trait::TreeLeafNode;

/// A high-performance generic trie of B+trees.
///
/// This is the generic version that abstracts over the leaf node type.
/// Use [`MassTree<V>`] for the standard WIDTH=24 implementation.
///
/// # Type Parameters
///
/// - `V` - The value type to store
/// - `L` - Leaf node type (must implement [`TreeLeafNode`])
/// - `A` - Allocator type (must implement [`NodeAllocatorGeneric`])
///
/// # Example
///
/// ```ignore
/// use masstree::{MassTreeGeneric, LeafNode24, PoolAllocator24};
///
/// // Create a WIDTH=24 tree explicitly
/// let tree: MassTreeGeneric<u64, LeafNode24<_>, PoolAllocator24<_>> =
///     MassTreeGeneric::new();
/// ```
pub struct MassTreeGeneric<S, L, A>
where
    S: ValueSlot,
    L: TreeLeafNode<S>,
    A: NodeAllocatorGeneric<S, L>,
{
    /// Memory reclamation collector for safe concurrent access.
    collector: Collector,

    /// Node allocator for leaf and internode allocation.
    allocator: A,

    /// Atomic root pointer for concurrent access.
    ///
    /// Points to either a leaf node or an internode.
    /// The node type is determined by the node's version field.
    root_ptr: AtomicPtr<u8>,

    /// Number of key-value pairs in the tree (atomic for concurrent access).
    count: AtomicUsize,

    /// Marker to indicate slot and leaf types.
    _marker: PhantomData<(S, L)>,
}

impl<S, L, A> StdFmt::Debug for MassTreeGeneric<S, L, A>
where
    S: ValueSlot,
    L: TreeLeafNode<S>,
    A: NodeAllocatorGeneric<S, L>,
{
    fn fmt(&self, f: &mut StdFmt::Formatter<'_>) -> StdFmt::Result {
        f.debug_struct("MassTreeGeneric")
            .field("root_ptr", &self.root_ptr.load(AtomicOrdering::Relaxed))
            .field("count", &self.count.load(AtomicOrdering::Relaxed))
            .field("width", &L::WIDTH)
            .finish_non_exhaustive()
    }
}

/// Result of searching a leaf for insert position (generic version).
#[derive(Debug)]
pub(crate) enum InsertSearchResultGeneric {
    /// Key exists at this slot.
    Found { slot: usize },

    /// Key not found, insert at logical position.
    NotFound { logical_pos: usize },

    /// Same ikey but different suffix - need to create layer.
    Conflict { slot: usize },

    /// Found layer pointer - descend into sublayer.
    Layer { slot: usize },
}

impl<S, L, A> Drop for MassTreeGeneric<S, L, A>
where
    S: ValueSlot,
    L: TreeLeafNode<S>,
    A: NodeAllocatorGeneric<S, L>,
{
    fn drop(&mut self) {
        // No concurrent access is possible here (Drop requires unique access).
        //
        // Step 1: Process all deferred retirements (suffix bags, etc.)
        // This MUST be called before teardown_tree to ensure any objects
        // retired via defer_retire() are reclaimed before we free nodes.
        //
        // SAFETY: &mut self guarantees no threads are active with guards.
        unsafe { self.collector.reclaim_all() };

        // Step 2: Free all nodes via allocator traversal.
        let root: *mut u8 = self.root_ptr.load(AtomicOrdering::Acquire);
        self.allocator.teardown_tree(root);
    }
}

// Send + Sync for MassTreeGeneric when V: Send + Sync
//
// The struct uses:
// - Collector (Send + Sync)
// - A (Send + Sync via trait bound)
// - AtomicPtr<u8> (Send + Sync)
// - AtomicUsize (Send + Sync)
// - PhantomData<(V, L)> inherits from V, L (both have Send + Sync bounds)
//
// We explicitly verify this compiles via the test below.

// ============================================================================
//  Type Aliases for MassTreeGeneric
// ============================================================================

/// The main [`MassTree`] type alias using WIDTH=24 nodes with Arc-based storage.
///
/// This is a type alias for [`MassTreeGeneric`] with:
/// - `LeafValue<V>` for Arc-based value storage
/// - `LeafNode24<LeafValue<V>>` for leaf nodes (24 slots per node)
/// - `SeizeAllocator24<LeafValue<V>>` for memory management
///
/// VALUES ARE STORED AS `Arc<V>` - each insert allocates. For small `Copy` types
/// like `u64`, consider [`MassTree24Inline`] which stores values inline.
///
/// # Example
///
/// ```ignore
/// use masstree::MassTree;
///
/// let tree: MassTree<u64> = MassTree::new();
/// let guard = tree.guard();
/// tree.insert_with_guard(b"key", 42, &guard).unwrap();
/// ```
pub type MassTree<V> =
    MassTreeGeneric<LeafValue<V>, LeafNode24<LeafValue<V>>, SeizeAllocator24<LeafValue<V>>>;

/// Alias for [`MassTree`] (WIDTH=24, Arc-based storage).
///
/// Provided for backwards compatibility and explicit naming.
pub type MassTree24<V> = MassTree<V>;

/// High-performance inline storage variant for `Copy` types.
///
/// This is a type alias for [`MassTreeGeneric`] with:
/// - `LeafValueIndex<V>` for inline value storage (NO heap allocation per insert)
/// - `LeafNode24<LeafValueIndex<V>>` for leaf nodes (24 slots per node)
/// - `SeizeAllocator24<LeafValueIndex<V>>` for memory management
///
/// **Use this for small, `Copy` types** like `u64`, `i32`, `*const T`, etc.
/// Values are stored directly in leaf nodes without `Arc` overhead.
///
/// # Performance
///
/// For `u64` values, this eliminates ~30-50ns of heap allocation overhead per insert,
/// making it competitive with other inline-storage structures like `scc::TreeIndex`.
///
/// # Example
///
/// ```ignore
/// use masstree::MassTree24Inline;
///
/// let tree: MassTree24Inline<u64> = MassTree24Inline::new();
/// let guard = tree.guard();
/// tree.insert_with_guard(b"key", 42, &guard).unwrap();
/// assert_eq!(tree.get_with_guard(b"key", &guard), Some(42)); // Returns u64 directly!
/// ```
pub type MassTree24Inline<V> = MassTreeGeneric<
    LeafValueIndex<V>,
    LeafNode24<LeafValueIndex<V>>,
    SeizeAllocator24<LeafValueIndex<V>>,
>;

/// [`MassTree`] with lock-free allocation tracking.
///
/// Uses atomic CAS operations instead of mutex for tracking allocations.
/// This eliminates contention on the allocation hot path for better scaling.
pub type MassTreeLockFree<V> = MassTreeGeneric<
    LeafValue<V>,
    LeafNode24<LeafValue<V>>,
    LockFreeAllocator<LeafNode24<LeafValue<V>>, LeafValue<V>>,
>;

/// [`MassTree`] with thread-local pool allocator.
///
/// Uses O(1) allocation from lock-free free lists for higher throughput.
/// Best for write-heavy workloads where allocation cost dominates.
pub type MassTreePooled<V> =
    MassTreeGeneric<LeafValue<V>, LeafNode24<LeafValue<V>>, PoolAllocator24<LeafValue<V>>>;

/// [`MassTree24Inline`] with thread-local pool allocator.
///
/// Combines inline value storage with pool allocation for maximum throughput.
pub type MassTreePooledInline<V> = MassTreeGeneric<
    LeafValueIndex<V>,
    LeafNode24<LeafValueIndex<V>>,
    PoolAllocator24<LeafValueIndex<V>>,
>;

// ============================================================================
//  Constructor implementations for type aliases
// ============================================================================

impl<V: Send + Sync + 'static> MassTree<V> {
    /// Create a new empty `MassTree`.
    #[must_use]
    #[inline(always)]
    pub fn new() -> Self {
        let allocator = SeizeAllocator24::new();
        Self::with_allocator(allocator)
    }
}

impl<V: Send + Sync + 'static> Default for MassTree<V> {
    fn default() -> Self {
        Self::new()
    }
}

impl<V: Copy + Send + Sync + 'static> MassTree24Inline<V> {
    /// Create a new empty `MassTree24Inline`.
    #[must_use]
    #[inline(always)]
    pub fn new() -> Self {
        let allocator = SeizeAllocator24::new();
        Self::with_allocator(allocator)
    }
}

impl<V: Copy + Send + Sync + 'static> Default for MassTree24Inline<V> {
    fn default() -> Self {
        Self::new()
    }
}

impl<V: Send + Sync + 'static> MassTreeLockFree<V> {
    /// Create a new empty `MassTreeLockFree`.
    #[must_use]
    #[inline(always)]
    pub fn new() -> Self {
        let allocator = LockFreeAllocator::new();
        Self::with_allocator(allocator)
    }
}

impl<V: Send + Sync + 'static> Default for MassTreeLockFree<V> {
    fn default() -> Self {
        Self::new()
    }
}

impl<V: Send + Sync + 'static> MassTreePooled<V> {
    /// Create a new empty `MassTreePooled`.
    #[must_use]
    #[inline(always)]
    pub fn new() -> Self {
        let allocator = PoolAllocator24::new();
        Self::with_allocator(allocator)
    }
}

impl<V: Send + Sync + 'static> Default for MassTreePooled<V> {
    fn default() -> Self {
        Self::new()
    }
}

impl<V: Copy + Send + Sync + 'static> MassTreePooledInline<V> {
    /// Create a new empty `MassTreePooledInline`.
    #[must_use]
    #[inline(always)]
    pub fn new() -> Self {
        let allocator = PoolAllocator24::new();
        Self::with_allocator(allocator)
    }
}

impl<V: Copy + Send + Sync + 'static> Default for MassTreePooledInline<V> {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
//  Tests
// ============================================================================

#[cfg(test)]
#[expect(clippy::unwrap_used, reason = "Fail fast in tests")]
#[expect(clippy::cast_possible_truncation, reason = "reasonable in tests")]
#[expect(clippy::cast_sign_loss, reason = "reasonable in tests")]
#[expect(clippy::items_after_statements, reason = "doesn't matter in tests")]
#[expect(clippy::type_complexity, reason = "doesn't matter in tests")]
mod unit_tests;
