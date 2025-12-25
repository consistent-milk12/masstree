//! Filepath: src/tree.rs
//! `MassTree` - A high-performance concurrent trie of B+trees.
//!
//! This module provides the main `MassTree<V>` and `MassTreeIndex<V>` types.

#![allow(dead_code, reason = "Major Refactor in process")]

use std::fmt as StdFmt;
use std::marker::PhantomData;
use std::sync::Arc;
use std::sync::atomic::{AtomicPtr, AtomicUsize, Ordering as AtomicOrdering};

use crate::value::LeafValue;
use parking_lot::{Condvar, Mutex};
use seize::Collector;

mod generic;
mod index;
mod optimistic;

#[cfg(test)]
pub mod test_hooks;

pub use index::MassTreeIndex;

// Re-export debug counters
pub use optimistic::{
    ADVANCE_BLINK_COUNT, BLINK_SHOULD_FOLLOW_COUNT, CAS_INSERT_FALLBACK_COUNT,
    CAS_INSERT_RETRY_COUNT, CAS_INSERT_SUCCESS_COUNT, DebugCounters, LOCKED_INSERT_COUNT,
    SEARCH_NOT_FOUND_COUNT, SPLIT_COUNT, WRONG_LEAF_INSERT_COUNT, get_all_debug_counters,
    get_debug_counters, reset_debug_counters,
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

    /// Root-level split required.
    /// Concurrent root updates require changing `MassTree.root` to `AtomicPtr`.
    /// Fall back to single-threaded insert path.
    RootSplitRequired,

    /// Parent internode is full and needs splitting.
    /// Recursive internode split not yet fully implemented concurrently.
    /// Fall back to single-threaded insert path.
    ParentSplitRequired,

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

            Self::RootSplitRequired => {
                write!(f, "root-level split required (use single-threaded insert)")
            }

            Self::ParentSplitRequired => {
                write!(f, "parent split required (use single-threaded insert)")
            }

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
#[allow(unused_imports)] // Will be used in submodule refactoring
use crate::leaf_trait::{TreeInternode, TreeLeafNode};

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
/// use masstree::{MassTreeGeneric, LeafNode24, SeizeAllocator24};
///
/// // Create a WIDTH=24 tree explicitly
/// let tree: MassTreeGeneric<u64, LeafNode24<_>, SeizeAllocator24<_>> =
///     MassTreeGeneric::new();
/// ```
pub struct MassTreeGeneric<V, L, A>
where
    L: TreeLeafNode<LeafValue<V>>,
    A: NodeAllocatorGeneric<LeafValue<V>, L>,
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

    /// Condition variable for parent-set notification.
    ///
    /// When a node's parent pointer is set during split propagation,
    /// waiters are notified to wake up instead of spinning.
    parent_set_condvar: Condvar,

    /// Mutex paired with the condvar (required by [`parking_lot`] API).
    parent_set_mutex: Mutex<()>,

    /// Marker to indicate V and L must be Send + Sync for concurrent access.
    _marker: PhantomData<(V, L)>,
}

impl<V, L, A> StdFmt::Debug for MassTreeGeneric<V, L, A>
where
    L: TreeLeafNode<LeafValue<V>>,
    A: NodeAllocatorGeneric<LeafValue<V>, L>,
{
    fn fmt(&self, f: &mut StdFmt::Formatter<'_>) -> StdFmt::Result {
        f.debug_struct("MassTreeGeneric")
            .field("root_ptr", &self.root_ptr.load(AtomicOrdering::Relaxed))
            .field("count", &self.count.load(AtomicOrdering::Relaxed))
            .field("width", &L::WIDTH)
            .finish_non_exhaustive()
    }
}

/// Result of a CAS insert attempt (generic version).
#[derive(Debug)]
#[allow(dead_code)]
pub(crate) enum CasInsertResultGeneric<V> {
    /// CAS insert succeeded.
    Success(Option<Arc<V>>),
    /// Key already exists - need locked update.
    ExistsNeedLock { slot: usize },
    /// Leaf is full - need locked split.
    FullNeedLock,
    /// Layer creation needed.
    LayerNeedLock { slot: usize },
    /// Slot-0 violation - need locked path.
    Slot0NeedLock,
    /// High contention - fall back to locked path.
    ContentionFallback,
}

/// Result of searching a leaf for insert position (generic version).
#[derive(Debug)]
#[allow(dead_code)]
pub(crate) enum InsertSearchResultGeneric {
    /// Key exists at this slot.
    Found { slot: usize },
    /// Key not found, insert at logical position.
    NotFound { logical_pos: usize },
    /// Same ikey but different suffix - need to create layer.
    Conflict { slot: usize },
    /// Found layer pointer - descend into sublayer.
    Layer { slot: usize, shift_amount: usize },
}

impl<V, L, A> Drop for MassTreeGeneric<V, L, A>
where
    L: TreeLeafNode<LeafValue<V>>,
    A: NodeAllocatorGeneric<LeafValue<V>, L>,
{
    fn drop(&mut self) {
        // No concurrent access is possible here (Drop requires unique access).
        // Load the root pointer and delegate teardown to the allocator.
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

/// The main [`MassTree`] type alias using WIDTH=24 nodes.
///
/// This is a type alias for [`MassTreeGeneric`] with:
/// - `LeafNode24<LeafValue<V>>` for leaf nodes (24 slots per node)
/// - `SeizeAllocator24<LeafValue<V>>` for memory management
///
/// WIDTH=24 nodes use u128 permutation (vs u64 for WIDTH=15), providing
/// 60% more capacity per node and significantly fewer splits under load.
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
pub type MassTree<V> = MassTreeGeneric<
    V,
    crate::leaf24::LeafNode24<LeafValue<V>>,
    crate::alloc24::SeizeAllocator24<LeafValue<V>>,
>;

/// Alias for [`MassTree`] (WIDTH=24 implementation).
///
/// Provided for backwards compatibility and explicit naming.
pub type MassTree24<V> = MassTree<V>;

// ============================================================================
//  Constructor implementations for type aliases
// ============================================================================

impl<V: Send + Sync + 'static> MassTree<V> {
    /// Create a new empty `MassTree`.
    #[must_use]
    #[inline(always)]
    pub fn new() -> Self {
        let allocator = crate::alloc24::SeizeAllocator24::new();
        Self::with_allocator(allocator)
    }
}

impl<V: Send + Sync + 'static> Default for MassTree<V> {
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
mod tests {
    use super::*;

    // ========================================================================
    // Send/Sync Verification
    // ========================================================================
    //
    // MassTree<V> is Send + Sync when V: Send + Sync.
    //
    // The struct uses:
    // - AtomicPtr<u8> for root_ptr (Send + Sync)
    // - AtomicUsize for count (Send + Sync)
    // - PhantomData<V> which inherits Send/Sync from V
    //
    // This enables concurrent access via Arc<MassTree<V>>.

    fn _assert_send_sync()
    where
        MassTree<u64>: Send + Sync,
    {
    }

    // ========================================================================
    //  MassTree Basic Tests
    // ========================================================================

    #[test]
    fn test_new_tree_is_empty() {
        let tree: MassTree<u64> = MassTree::new();

        assert!(tree.is_empty());
        assert_eq!(tree.len(), 0);
    }

    #[test]
    fn test_get_on_empty_tree() {
        let tree: MassTree<u64> = MassTree::new();

        assert!(tree.get(b"hello").is_none());
        assert!(tree.get(b"").is_none());
        assert!(tree.get(b"any key").is_none());
    }

    // ========================================================================
    //  MassTreeIndex Basic Tests
    // ========================================================================

    #[test]
    fn test_index_new_is_empty() {
        let tree: MassTreeIndex<u64> = MassTreeIndex::new();

        assert!(tree.is_empty());
        assert_eq!(tree.len(), 0);
    }

    #[test]
    fn test_index_get_on_empty() {
        let tree: MassTreeIndex<u64> = MassTreeIndex::new();

        assert!(tree.get(b"hello").is_none());
    }

    // ========================================================================
    //  Key Handling Tests
    // ========================================================================

    #[test]
    fn test_get_with_various_key_lengths() {
        let tree: MassTree<u64> = MassTree::new();

        // All should return None on empty tree
        assert!(tree.get(b"").is_none());
        assert!(tree.get(b"a").is_none());
        assert!(tree.get(b"ab").is_none());
        assert!(tree.get(b"abc").is_none());
        assert!(tree.get(b"abcd").is_none());
        assert!(tree.get(b"abcde").is_none());
        assert!(tree.get(b"abcdef").is_none());
        assert!(tree.get(b"abcdefg").is_none());
        assert!(tree.get(b"abcdefgh").is_none()); // Exactly 8 bytes
    }

    #[test]
    fn test_get_missing_long_keys() {
        let tree: MassTree<u64> = MassTree::new();

        // Long keys should return None on empty tree (not inserted)
        assert!(tree.get(b"abcdefghi").is_none()); // 9 bytes
        assert!(tree.get(b"hello world").is_none()); // 11 bytes
        assert!(tree.get(b"this is a long key").is_none());
    }

    // ========================================================================
    //  Default Trait Tests
    // ========================================================================

    #[test]
    fn test_default_trait() {
        let tree: MassTree<u64> = MassTree::default();

        assert!(tree.is_empty());
    }

    #[test]
    fn test_index_default_trait() {
        let tree: MassTreeIndex<u64> = MassTreeIndex::default();

        assert!(tree.is_empty());
    }

    // ========================================================================
    //  Get After Insert Tests (using public API)
    // ========================================================================

    #[test]
    fn test_get_after_insert() {
        let tree: MassTree<u64> = MassTree::new();

        tree.insert(b"hello", 42).unwrap();

        let result = tree.get(b"hello");
        assert!(result.is_some());
        assert_eq!(*result.unwrap(), 42);
    }

    #[test]
    fn test_get_wrong_key_after_insert() {
        let tree: MassTree<u64> = MassTree::new();

        tree.insert(b"hello", 42).unwrap();

        assert!(tree.get(b"world").is_none());
        assert!(tree.get(b"hell").is_none());
        assert!(tree.get(b"helloX").is_none());
    }

    #[test]
    fn test_get_multiple_keys() {
        let tree: MassTree<u64> = MassTree::new();

        // Use the insert API to add multiple keys
        tree.insert(b"aaa", 100).unwrap();
        tree.insert(b"bbb", 200).unwrap();
        tree.insert(b"ccc", 300).unwrap();

        // Test retrieval
        assert_eq!(*tree.get(b"aaa").unwrap(), 100);
        assert_eq!(*tree.get(b"bbb").unwrap(), 200);
        assert_eq!(*tree.get(b"ccc").unwrap(), 300);

        // Non-existent keys
        assert!(tree.get(b"ddd").is_none());
        assert!(tree.get(b"aab").is_none());
    }

    // ========================================================================
    //  Index Mode Get Tests (using public API)
    // ========================================================================

    #[test]
    fn test_index_get_after_insert() {
        let mut tree: MassTreeIndex<u64> = MassTreeIndex::new();

        tree.insert(b"test", 999).unwrap();

        // Get returns V directly (not Arc<V>)
        let result = tree.get(b"test");
        assert_eq!(result, Some(999));
    }

    // ========================================================================
    //  Edge Case Tests (using public API)
    // ========================================================================

    #[test]
    fn test_insert_and_get_empty_key() {
        let tree: MassTree<u64> = MassTree::new();

        tree.insert(b"", 123).unwrap();
        assert_eq!(*tree.get(b"").unwrap(), 123);
    }

    #[test]
    fn test_insert_and_get_max_inline_key() {
        let tree: MassTree<u64> = MassTree::new();

        // Insert 8-byte key (max inline size)
        let key_bytes = b"12345678";
        tree.insert(key_bytes, 888).unwrap();

        assert_eq!(*tree.get(key_bytes).unwrap(), 888);
    }

    #[test]
    fn test_insert_and_get_binary_key() {
        let tree: MassTree<u64> = MassTree::new();

        // Binary key with null bytes
        let key_bytes = &[0x00, 0x01, 0x02, 0x00, 0xFF];
        tree.insert(key_bytes, 777).unwrap();

        assert_eq!(*tree.get(key_bytes).unwrap(), 777);
    }

    // ========================================================================
    //  Insert Tests
    // ========================================================================

    #[test]
    fn test_insert_into_empty_tree() {
        let tree: MassTree<u64> = MassTree::new();

        let result = tree.insert(b"hello", 42);

        assert!(result.is_ok());
        assert!(result.unwrap().is_none()); // No old value
        assert!(!tree.is_empty());
        assert_eq!(tree.len(), 1);
    }

    #[test]
    fn test_insert_and_get() {
        let tree: MassTree<u64> = MassTree::new();

        tree.insert(b"hello", 42).unwrap();

        let value = tree.get(b"hello");
        assert!(value.is_some());
        assert_eq!(*value.unwrap(), 42);
    }

    #[test]
    fn test_insert_multiple_keys() {
        let tree: MassTree<u64> = MassTree::new();

        tree.insert(b"aaa", 100).unwrap();
        tree.insert(b"bbb", 200).unwrap();
        tree.insert(b"ccc", 300).unwrap();

        assert_eq!(tree.len(), 3);
        assert_eq!(*tree.get(b"aaa").unwrap(), 100);
        assert_eq!(*tree.get(b"bbb").unwrap(), 200);
        assert_eq!(*tree.get(b"ccc").unwrap(), 300);
    }

    #[test]
    fn test_insert_updates_existing() {
        let tree: MassTree<u64> = MassTree::new();

        // First insert
        let old = tree.insert(b"key", 100).unwrap();
        assert!(old.is_none());

        // Update
        let old = tree.insert(b"key", 200).unwrap();
        assert_eq!(*old.unwrap(), 100);

        // Verify new value
        assert_eq!(*tree.get(b"key").unwrap(), 200);
        assert_eq!(tree.len(), 1); // Still one key
    }

    #[test]
    fn test_insert_returns_old_arc() {
        let tree: MassTree<String> = MassTree::new();

        tree.insert(b"key", "first".to_string()).unwrap();

        let old = tree.insert(b"key", "second".to_string()).unwrap();
        assert_eq!(*old.unwrap(), "first");
    }

    #[test]
    fn test_insert_ascending_order() {
        let tree: MassTree<u64> = MassTree::new();

        for i in 0..10 {
            let key = format!("key{i:02}");
            tree.insert(key.as_bytes(), i as u64).unwrap();
        }

        assert_eq!(tree.len(), 10);

        // Verify all values
        for i in 0..10 {
            let key = format!("key{i:02}");
            assert_eq!(*tree.get(key.as_bytes()).unwrap(), i as u64);
        }
    }

    #[test]
    fn test_insert_descending_order() {
        let tree: MassTree<u64> = MassTree::new();

        for i in (0..10).rev() {
            let key = format!("key{i:02}");
            tree.insert(key.as_bytes(), i as u64).unwrap();
        }

        assert_eq!(tree.len(), 10);

        for i in 0..10 {
            let key = format!("key{i:02}");
            assert_eq!(*tree.get(key.as_bytes()).unwrap(), i as u64);
        }
    }

    #[test]
    fn test_insert_random_order() {
        let tree: MassTree<u64> = MassTree::new();

        let keys = ["dog", "cat", "bird", "fish", "ant", "bee"];

        for (i, key) in keys.iter().enumerate() {
            tree.insert(key.as_bytes(), i as u64).unwrap();
        }

        assert_eq!(tree.len(), 6);

        for (i, key) in keys.iter().enumerate() {
            assert_eq!(*tree.get(key.as_bytes()).unwrap(), i as u64);
        }
    }

    #[test]
    fn test_insert_triggers_split() {
        let tree: MassTree<u64> = MassTree::new();

        // Fill the leaf (24 slots for WIDTH=24)
        for i in 0..24 {
            let key = format!("{i:02}");
            tree.insert(key.as_bytes(), i as u64).unwrap();
        }

        assert_eq!(tree.len(), 24);

        // 25th insert should trigger a split
        let result = tree.insert(b"overflow", 999);
        assert!(result.is_ok());
        assert!(result.unwrap().is_none()); // New key, no old value

        assert_eq!(tree.len(), 25);

        // Verify the new key is accessible
        let value = tree.get(b"overflow");
        assert_eq!(*value.unwrap(), 999);

        // Verify all original keys are still accessible
        for i in 0..24 {
            let key = format!("{i:02}");
            let value = tree.get(key.as_bytes());

            assert!(value.is_some(), "Key {key:?} not found after split");
            assert_eq!(*value.unwrap(), i as u64);
        }
    }

    #[test]
    fn test_insert_at_capacity() {
        let tree: MassTree<u64> = MassTree::new();

        // Insert exactly WIDTH keys (24 for WIDTH=24)
        for i in 0..24 {
            let key = format!("{i:02}");
            assert!(tree.insert(key.as_bytes(), i as u64).is_ok());
        }

        assert_eq!(tree.len(), 24);
    }

    #[test]
    fn test_insert_error_display() {
        let err = InsertError::LeafFull;
        assert_eq!(format!("{err}"), "leaf node is full");

        let err = InsertError::AllocationFailed;
        assert_eq!(format!("{err}"), "memory allocation failed");
    }

    #[test]
    fn test_insert_max_key_length() {
        let tree: MassTree<u64> = MassTree::new();

        // Exactly 8 bytes should work
        let max_key = b"12345678";
        let result = tree.insert(max_key, 42);
        assert!(result.is_ok());
        assert_eq!(*tree.get(max_key).unwrap(), 42);
    }

    #[test]
    fn test_slot0_reuse_no_predecessor() {
        let tree: MassTree<u64> = MassTree::new();

        // Root leaf has no predecessor, so slot 0 should be usable
        tree.insert(b"first", 1).unwrap();

        // Verify the value was stored
        assert_eq!(*tree.get(b"first").unwrap(), 1);
    }

    #[test]
    fn test_slot0_reuse_same_ikey() {
        let tree: MassTree<u64> = MassTree::new();

        // Insert key with ikey X
        tree.insert(b"aaa", 1).unwrap();

        // Insert another key with same ikey prefix (within 8 bytes)
        // This should allow slot 0 reuse if needed
        tree.insert(b"aab", 2).unwrap();

        assert_eq!(*tree.get(b"aaa").unwrap(), 1);
        assert_eq!(*tree.get(b"aab").unwrap(), 2);
    }

    #[test]
    fn test_permutation_updates_correctly() {
        let tree: MassTree<u64> = MassTree::new();

        // Insert keys in reverse order
        tree.insert(b"ccc", 3).unwrap();
        tree.insert(b"bbb", 2).unwrap();
        tree.insert(b"aaa", 1).unwrap();

        // Verify size
        assert_eq!(tree.len(), 3);

        // Verify get still works (relies on correct permutation)
        assert_eq!(*tree.get(b"aaa").unwrap(), 1);
        assert_eq!(*tree.get(b"bbb").unwrap(), 2);
        assert_eq!(*tree.get(b"ccc").unwrap(), 3);
    }

    #[test]
    fn test_index_insert_basic() {
        let mut tree: MassTreeIndex<u64> = MassTreeIndex::new();

        let result = tree.insert(b"key", 42);

        assert!(result.is_ok());
        assert!(result.unwrap().is_none());
        assert_eq!(tree.get(b"key"), Some(42));
    }

    #[test]
    fn test_index_insert_update() {
        let mut tree: MassTreeIndex<u64> = MassTreeIndex::new();

        tree.insert(b"key", 100).unwrap();

        let old = tree.insert(b"key", 200).unwrap();

        assert_eq!(old, Some(100)); // Old value returned directly (not Arc)
        assert_eq!(tree.get(b"key"), Some(200));
    }

    #[test]
    fn test_insert_empty_key() {
        let tree: MassTree<u64> = MassTree::new();

        tree.insert(b"", 42).unwrap();

        assert_eq!(*tree.get(b"").unwrap(), 42);
    }

    #[test]
    fn test_insert_max_inline_key() {
        let tree: MassTree<u64> = MassTree::new();

        let key = b"12345678"; // Exactly 8 bytes

        tree.insert(key, 888).unwrap();

        assert_eq!(*tree.get(key).unwrap(), 888);
    }

    #[test]
    fn test_insert_binary_keys() {
        let tree: MassTree<u64> = MassTree::new();

        let keys = [
            vec![0x00, 0x01, 0x02],
            vec![0xFF, 0xFE, 0xFD],
            vec![0x00, 0x00, 0x00],
            vec![0xFF, 0xFF, 0xFF],
        ];

        for (i, key) in keys.iter().enumerate() {
            tree.insert(key, i as u64).unwrap();
        }

        for (i, key) in keys.iter().enumerate() {
            assert_eq!(*tree.get(key).unwrap(), i as u64);
        }
    }

    #[test]
    fn test_insert_similar_keys() {
        let tree: MassTree<u64> = MassTree::new();

        tree.insert(b"test", 1).unwrap();
        tree.insert(b"test1", 2).unwrap();
        tree.insert(b"test12", 3).unwrap();
        tree.insert(b"tes", 4).unwrap();

        assert_eq!(*tree.get(b"test").unwrap(), 1);
        assert_eq!(*tree.get(b"test1").unwrap(), 2);
        assert_eq!(*tree.get(b"test12").unwrap(), 3);
        assert_eq!(*tree.get(b"tes").unwrap(), 4);
    }

    // ========================================================================
    //  Differential Testing
    // ========================================================================

    #[test]
    fn test_differential_small_sequential() {
        use std::collections::BTreeMap;

        let tree: MassTree<u64> = MassTree::new();
        let mut oracle: BTreeMap<Vec<u8>, u64> = BTreeMap::new();

        for i in 0..10 {
            let key = vec![b'a' + i as u8];
            let tree_old = tree.insert(&key, i as u64).unwrap();
            let oracle_old = oracle.insert(key.clone(), i as u64);

            assert_eq!(
                tree_old.map(|arc| *arc),
                oracle_old,
                "Insert mismatch for key {key:?}"
            );
        }
    }

    #[test]
    fn test_differential_updates() {
        use std::collections::BTreeMap;

        let tree: MassTree<u64> = MassTree::new();
        let mut oracle: BTreeMap<Vec<u8>, u64> = BTreeMap::new();

        let key = b"key".to_vec();

        for value in [100, 200, 300] {
            let tree_old = tree.insert(&key, value).unwrap();
            let oracle_old = oracle.insert(key.clone(), value);
            assert_eq!(tree_old.map(|arc| *arc), oracle_old);
        }

        assert_eq!(*tree.get(&key).unwrap(), *oracle.get(&key).unwrap());
    }

    #[test]
    fn test_differential_interleaved() {
        use std::collections::BTreeMap;

        let tree: MassTree<u64> = MassTree::new();
        let mut oracle: BTreeMap<Vec<u8>, u64> = BTreeMap::new();

        tree.insert(b"a", 1).unwrap();
        oracle.insert(b"a".to_vec(), 1);
        assert_eq!(
            tree.get(b"a").map(|a| *a),
            oracle.get(b"a".as_slice()).copied()
        );

        tree.insert(b"b", 2).unwrap();
        oracle.insert(b"b".to_vec(), 2);
        assert_eq!(
            tree.get(b"a").map(|a| *a),
            oracle.get(b"a".as_slice()).copied()
        );
        assert_eq!(
            tree.get(b"b").map(|a| *a),
            oracle.get(b"b".as_slice()).copied()
        );

        tree.insert(b"a", 10).unwrap();
        oracle.insert(b"a".to_vec(), 10);
        assert_eq!(
            tree.get(b"a").map(|a| *a),
            oracle.get(b"a".as_slice()).copied()
        );
    }

    #[test]
    fn test_differential_get_missing() {
        use std::collections::BTreeMap;

        let tree: MassTree<u64> = MassTree::new();
        let oracle: BTreeMap<Vec<u8>, u64> = BTreeMap::new();

        assert_eq!(
            tree.get(b"missing").map(|a| *a),
            oracle.get(b"missing".as_slice()).copied()
        );

        tree.insert(b"present", 42).unwrap();

        assert_eq!(tree.get(b"missing").map(|a| *a), None);
        assert_eq!(*tree.get(b"present").unwrap(), 42);
    }

    // ========================================================================
    //  Split Tests
    // ========================================================================

    #[test]
    fn test_insert_100_keys() {
        let tree: MassTree<u64> = MassTree::new();

        // Insert 100 unique keys (will trigger multiple splits)
        for i in 0..100 {
            let key = format!("{i:08}"); // 8-byte keys
            tree.insert(key.as_bytes(), i as u64).unwrap();
        }

        // Verify all keys are retrievable
        for i in 0..100 {
            let key = format!("{i:08}");
            let value = tree.get(key.as_bytes());
            assert!(value.is_some(), "Key {i} not found");
            assert_eq!(*value.unwrap(), i as u64);
        }

        // Verify len() works after multiple splits
        assert_eq!(tree.len(), 100);
    }

    #[test]
    fn test_len_multi_level_tree() {
        let tree: MassTree<u64> = MassTree::new();

        // Insert enough keys to create multiple levels
        for i in 0..200 {
            let key = format!("{i:08}");
            tree.insert(key.as_bytes(), i as u64).unwrap();
        }

        // Verify len() returns correct count
        assert_eq!(tree.len(), 200);

        // Insert more and verify again
        for i in 200..500 {
            let key = format!("{i:08}");
            tree.insert(key.as_bytes(), i as u64).unwrap();
        }
        assert_eq!(tree.len(), 500);
    }

    #[test]
    fn test_insert_1000_sequential() {
        use std::collections::BTreeMap;

        let tree: MassTree<u64> = MassTree::new();
        let mut oracle: BTreeMap<Vec<u8>, u64> = BTreeMap::new();

        // Insert 1000 keys sequentially
        for i in 0..1000 {
            let key = format!("{i:08}");
            tree.insert(key.as_bytes(), i as u64).unwrap();
            oracle.insert(key.into_bytes(), i as u64);
        }

        // Differential check: all keys match oracle
        for (key, expected) in &oracle {
            let actual = tree.get(key);
            assert!(actual.is_some(), "Key {key:?} not found in tree");
            assert_eq!(*actual.unwrap(), *expected);
        }
    }

    #[test]
    fn test_insert_1000_random() {
        use std::collections::BTreeMap;

        let tree: MassTree<u64> = MassTree::new();
        let mut oracle: BTreeMap<Vec<u8>, u64> = BTreeMap::new();

        // Insert 1000 keys in "random" order (deterministic pattern)
        for i in 0..1000 {
            // Mix up the order using modular arithmetic
            let idx = (i * 997) % 1000;
            let key = format!("{idx:08}");
            tree.insert(key.as_bytes(), idx as u64).unwrap();
            oracle.insert(key.into_bytes(), idx as u64);
        }

        // Differential check
        for (key, expected) in &oracle {
            let actual = tree.get(key);
            assert!(actual.is_some(), "Key {key:?} not found");
            assert_eq!(*actual.unwrap(), *expected);
        }
    }

    #[test]
    fn test_split_preserves_values() {
        let tree: MassTree<String> = MassTree::new();

        // Insert values with distinct content
        let test_values: Vec<(&[u8], String)> = vec![
            (b"aaa", "first value".to_string()),
            (b"bbb", "second value".to_string()),
            (b"ccc", "third value".to_string()),
            (b"ddd", "fourth value".to_string()),
            (b"eee", "fifth value".to_string()),
        ];

        for (key, value) in &test_values {
            tree.insert(key, value.clone()).unwrap();
        }

        // Fill to trigger split
        for i in 0..20 {
            let key = format!("k{i:02}");
            tree.insert(key.as_bytes(), format!("value{i}")).unwrap();
        }

        // Verify original values survived split
        for (key, expected) in &test_values {
            let actual = tree.get(key);
            assert!(actual.is_some(), "Key {key:?} lost after split");
            assert_eq!(*actual.unwrap(), *expected);
        }
    }

    #[test]
    fn test_split_with_updates() {
        let tree: MassTree<u64> = MassTree::new();

        // Insert and immediately update some keys
        for i in 0..30 {
            let key = format!("{i:04}");
            tree.insert(key.as_bytes(), i as u64).unwrap();
        }

        // Update every other key
        for i in (0..30).step_by(2) {
            let key = format!("{i:04}");
            let old = tree.insert(key.as_bytes(), (i + 1000) as u64).unwrap();
            assert_eq!(*old.unwrap(), i as u64);
        }

        // Verify
        for i in 0..30 {
            let key = format!("{i:04}");
            let expected = if i % 2 == 0 { i + 1000 } else { i };
            assert_eq!(*tree.get(key.as_bytes()).unwrap(), expected as u64);
        }
    }

    #[test]
    fn test_multiple_splits_create_deeper_tree() {
        // With WIDTH=24, multiple splits needed for 100 keys
        let tree: MassTree<u64> = MassTree::new();

        for i in 0..100 {
            let key = format!("{i:08}");
            tree.insert(key.as_bytes(), i as u64).unwrap();
        }

        // Verify all keys accessible
        for i in 0..100 {
            let key = format!("{i:08}");
            assert!(tree.get(key.as_bytes()).is_some(), "Key {i} not found");
        }

        // Verify count
        assert_eq!(tree.len(), 100);
    }

    #[test]
    fn test_split_empty_right_edge_case() {
        // Test the sequential optimization edge case
        let tree: MassTree<u64> = MassTree::new();

        // Fill leaf completely with sequential keys (rightmost optimization)
        for i in 0..24 {
            tree.insert(&[i as u8], i as u64).unwrap();
        }

        // 25th insert at end should trigger split
        tree.insert(&[24u8], 24).unwrap();

        // Verify all keys
        for i in 0..25 {
            assert_eq!(*tree.get(&[i as u8]).unwrap(), i as u64);
        }
    }

    #[test]
    fn test_split_index_mode() {
        // Test splits work with MassTreeIndex too
        let mut tree: MassTreeIndex<u64> = MassTreeIndex::new();

        for i in 0..50 {
            let key = format!("{i:02}");
            tree.insert(key.as_bytes(), i as u64).unwrap();
        }

        for i in 0..50 {
            let key = format!("{i:02}");
            assert_eq!(tree.get(key.as_bytes()), Some(i as u64));
        }
    }

    #[test]
    fn test_split_maintains_key_order() {
        let tree: MassTree<u64> = MassTree::new();

        // Insert in reverse order
        for i in (0..50).rev() {
            let key = format!("{i:02}");
            tree.insert(key.as_bytes(), i as u64).unwrap();
        }

        // Keys should still be retrievable
        for i in 0..50 {
            let key = format!("{i:02}");
            let value = tree.get(key.as_bytes());
            assert!(value.is_some(), "Key {i:02} not found");
            assert_eq!(*value.unwrap(), i as u64);
        }
    }

    // ========================================================================
    //  Layer Tests
    // ========================================================================

    #[test]
    fn test_layer_creation_same_prefix() {
        let tree: MassTree<u64> = MassTree::new();

        // Insert "hello world!" (12 bytes)
        tree.insert(b"hello world!", 1).unwrap();

        // Insert "hello worm" (10 bytes) - same 8-byte prefix "hello wo"
        tree.insert(b"hello worm", 2).unwrap();

        // Both should be retrievable
        assert_eq!(tree.get(b"hello world!").map(|v| *v), Some(1));
        assert_eq!(tree.get(b"hello worm").map(|v| *v), Some(2));
    }

    #[test]
    fn test_deep_layer_chain() {
        let tree: MassTree<u64> = MassTree::new();

        // Two 24-byte keys with 16 bytes common prefix
        let key1 = b"aaaaaaaabbbbbbbbXXXXXXXX"; // 24 bytes
        let key2 = b"aaaaaaaabbbbbbbbYYYYYYYY"; // 24 bytes, differs at byte 16

        tree.insert(key1, 1).unwrap();
        tree.insert(key2, 2).unwrap();

        // Should create 2 intermediate twig nodes:
        // Layer 0: ikey="aaaaaaaa" → twig
        // Layer 1: ikey="bbbbbbbb" → twig
        // Layer 2: ikey="XXXXXXXX" and "YYYYYYYY" (different, final leaf)

        assert_eq!(tree.get(key1).map(|v| *v), Some(1));
        assert_eq!(tree.get(key2).map(|v| *v), Some(2));
    }

    #[test]
    fn test_layer_with_suffixes() {
        let tree: MassTree<u64> = MassTree::new();

        // Keys that create layers AND have suffixes
        let key1 = b"prefixAArest1"; // 13 bytes
        let key2 = b"prefixAArest2"; // 13 bytes, same first 8, different suffix

        tree.insert(key1, 1).unwrap();
        tree.insert(key2, 2).unwrap();

        assert_eq!(tree.get(key1).map(|v| *v), Some(1));
        assert_eq!(tree.get(key2).map(|v| *v), Some(2));
        assert_eq!(tree.get(b"prefixAArest3"), None);
    }

    #[test]
    fn test_insert_into_existing_layer() {
        let tree: MassTree<u64> = MassTree::new();

        // Create a layer
        tree.insert(b"hello world!", 1).unwrap();
        tree.insert(b"hello worm", 2).unwrap();

        // Insert another key with same prefix, different continuation
        tree.insert(b"hello wonder", 3).unwrap();

        assert_eq!(tree.get(b"hello world!").map(|v| *v), Some(1));
        assert_eq!(tree.get(b"hello worm").map(|v| *v), Some(2));
        assert_eq!(tree.get(b"hello wonder").map(|v| *v), Some(3));
    }

    #[test]
    fn test_update_in_layer() {
        let tree: MassTree<u64> = MassTree::new();

        tree.insert(b"hello world!", 1).unwrap();
        tree.insert(b"hello worm", 2).unwrap();

        // Update existing key in layer
        let old = tree.insert(b"hello world!", 100).unwrap();
        assert_eq!(old.map(|v| *v), Some(1));

        assert_eq!(tree.get(b"hello world!").map(|v| *v), Some(100));
        assert_eq!(tree.get(b"hello worm").map(|v| *v), Some(2));
    }

    #[test]
    fn test_get_nonexistent_in_layer() {
        let tree: MassTree<u64> = MassTree::new();

        tree.insert(b"hello world!", 1).unwrap();
        tree.insert(b"hello worm", 2).unwrap();

        // Same prefix but different suffix that wasn't inserted
        assert_eq!(tree.get(b"hello worst"), None);
        assert_eq!(tree.get(b"hello wo"), None); // Exact 8 bytes (partial match)
    }

    #[test]
    fn test_long_key_insert_and_get() {
        let tree: MassTree<u64> = MassTree::new();

        // Keys longer than 8 bytes without common prefix
        let key1 = b"abcdefghijklmnop"; // 16 bytes
        let key2 = b"zyxwvutsrqponmlk"; // 16 bytes, different prefix

        tree.insert(key1, 100).unwrap();
        tree.insert(key2, 200).unwrap();

        assert_eq!(tree.get(key1).map(|v| *v), Some(100));
        assert_eq!(tree.get(key2).map(|v| *v), Some(200));
    }

    #[test]
    fn test_mixed_short_and_long_keys() {
        let tree: MassTree<u64> = MassTree::new();

        // Mix of short (<=8) and long (>8) keys
        tree.insert(b"short", 1).unwrap();
        tree.insert(b"exactly8", 2).unwrap();
        tree.insert(b"this is longer", 3).unwrap();
        tree.insert(b"another long key", 4).unwrap();

        assert_eq!(tree.get(b"short").map(|v| *v), Some(1));
        assert_eq!(tree.get(b"exactly8").map(|v| *v), Some(2));
        assert_eq!(tree.get(b"this is longer").map(|v| *v), Some(3));
        assert_eq!(tree.get(b"another long key").map(|v| *v), Some(4));
    }

    #[test]
    fn test_layer_differential_vs_btreemap() {
        use std::collections::BTreeMap;

        let tree: MassTree<u64> = MassTree::new();
        let mut oracle: BTreeMap<Vec<u8>, u64> = BTreeMap::new();

        // Test with keys that will create layers
        let keys = [
            b"hello world!".to_vec(),
            b"hello worm".to_vec(),
            b"hello wonder".to_vec(),
            b"goodbye world".to_vec(),
            b"goodbye friend".to_vec(),
            b"short".to_vec(),
            b"a".to_vec(),
        ];

        for (i, key) in keys.iter().enumerate() {
            tree.insert(key, i as u64).unwrap();
            oracle.insert(key.clone(), i as u64);
        }

        // Verify all oracle entries
        for (key, expected) in &oracle {
            let actual = tree.get(key).map(|v| *v);
            assert_eq!(actual, Some(*expected), "mismatch for key {key:?}");
        }
    }

    #[test]
    fn test_layer_count_tracking() {
        let tree: MassTree<u64> = MassTree::new();

        // Insert keys that create layers
        tree.insert(b"hello world!", 1).unwrap();
        assert_eq!(tree.len(), 1);

        tree.insert(b"hello worm", 2).unwrap();
        assert_eq!(tree.len(), 2);

        tree.insert(b"hello wonder", 3).unwrap();
        assert_eq!(tree.len(), 3);

        // Update shouldn't change count
        tree.insert(b"hello world!", 100).unwrap();
        assert_eq!(tree.len(), 3);
    }

    // ========================================================================
    //  Layer Root Growth Tests
    // ========================================================================

    #[test]
    fn test_layer_growth_beyond_width() {
        // Test that a layer can grow beyond WIDTH entries by promoting to internode
        let tree: MassTree<u64> = MassTree::new();

        // All keys share same 8-byte prefix, forcing them into a layer
        let prefix = b"samepfx!";

        // Insert more than WIDTH (15) keys - will trigger layer root split
        for i in 0..20u64 {
            let mut key = prefix.to_vec();
            key.extend_from_slice(&i.to_be_bytes());
            tree.insert(&key, i).unwrap();
        }

        // All keys must be findable
        for i in 0..20u64 {
            let mut key = prefix.to_vec();
            key.extend_from_slice(&i.to_be_bytes());
            let result = tree.get(&key);
            assert!(result.is_some(), "Key with suffix {i} not found");
            assert_eq!(*result.unwrap(), i);
        }
    }

    #[test]
    fn test_layer_root_becomes_internode() {
        // Test with default WIDTH - insert enough keys to trigger layer split
        let tree: MassTree<u64> = MassTree::new();

        // Force layer creation with suffix keys
        tree.insert(b"prefix00suffix_a", 1).unwrap();
        tree.insert(b"prefix00suffix_b", 2).unwrap();

        // These should go into the layer and eventually trigger split
        // With WIDTH=15, we need >15 keys in the layer to trigger split
        for i in 0..20u64 {
            let key = format!("prefix00suffix_{i:02}");
            tree.insert(key.as_bytes(), i + 100).unwrap();
        }

        // Verify initial keys still findable
        assert_eq!(*tree.get(b"prefix00suffix_a").unwrap(), 1);
        assert_eq!(*tree.get(b"prefix00suffix_b").unwrap(), 2);

        // Verify all numbered keys findable
        for i in 0..20u64 {
            let key = format!("prefix00suffix_{i:02}");
            let result = tree.get(key.as_bytes());
            assert!(result.is_some(), "Key {key} not found");
        }
    }

    #[test]
    fn test_layer_split_preserves_all_keys() {
        // Comprehensive test: insert many keys with shared prefix, verify all retrievable
        let tree: MassTree<u64> = MassTree::new();
        let mut keys_and_values: Vec<(Vec<u8>, u64)> = Vec::new();

        // Create keys that will all go into the same layer (share 8-byte prefix)
        for i in 0..50u64 {
            let mut key = b"testpfx!".to_vec(); // Exactly 8 bytes
            key.extend_from_slice(format!("suffix{i:04}").as_bytes());
            keys_and_values.push((key.clone(), i * 10));
            tree.insert(&key, i * 10).unwrap();
        }

        // Verify all keys
        for (key, expected) in &keys_and_values {
            let result = tree.get(key);
            assert!(result.is_some(), "Key {key:?} not found");
            assert_eq!(*result.unwrap(), *expected);
        }

        assert_eq!(tree.len(), 50);
    }

    // ========================================================================
    //  MassTreeGeneric Tests
    // ========================================================================

    #[test]
    fn test_masstree_generic_new_is_empty() {
        use crate::alloc24::SeizeAllocator24;
        use crate::leaf24::LeafNode24;

        // Create via with_allocator
        let alloc: SeizeAllocator24<LeafValue<u64>> = SeizeAllocator24::new();
        let tree: MassTreeGeneric<
            u64,
            LeafNode24<LeafValue<u64>>,
            SeizeAllocator24<LeafValue<u64>>,
        > = MassTreeGeneric::with_allocator(alloc);

        assert!(tree.is_empty());
        assert_eq!(tree.len(), 0);
    }

    #[test]
    fn test_masstree_generic_debug() {
        use crate::alloc24::SeizeAllocator24;
        use crate::leaf24::LeafNode24;

        let alloc: SeizeAllocator24<LeafValue<u64>> = SeizeAllocator24::new();
        let tree: MassTreeGeneric<
            u64,
            LeafNode24<LeafValue<u64>>,
            SeizeAllocator24<LeafValue<u64>>,
        > = MassTreeGeneric::with_allocator(alloc);

        let debug_str = format!("{tree:?}");
        assert!(debug_str.contains("MassTreeGeneric"));
        assert!(debug_str.contains("width: 24"));
    }

    #[test]
    fn test_masstree_generic_guard() {
        use crate::alloc24::SeizeAllocator24;
        use crate::leaf24::LeafNode24;

        let alloc: SeizeAllocator24<LeafValue<u64>> = SeizeAllocator24::new();
        let tree: MassTreeGeneric<
            u64,
            LeafNode24<LeafValue<u64>>,
            SeizeAllocator24<LeafValue<u64>>,
        > = MassTreeGeneric::with_allocator(alloc);

        // Just verify guard creation doesn't panic
        let _guard = tree.guard();
    }

    // ========================================================================
    //  MassTreeGeneric Send/Sync Tests
    // ========================================================================

    fn _assert_masstree_generic_send_sync()
    where
        MassTreeGeneric<
            u64,
            crate::leaf24::LeafNode24<LeafValue<u64>>,
            crate::alloc24::SeizeAllocator24<LeafValue<u64>>,
        >: Send + Sync,
    {
    }

    // ========================================================================
    //  MassTree24 Type Alias Tests
    // ========================================================================

    #[test]
    fn test_masstree24_new() {
        let tree: MassTree24<u64> = MassTree24::new();
        assert!(tree.is_empty());
        assert_eq!(tree.len(), 0);
    }

    #[test]
    fn test_masstree24_get_empty() {
        let tree: MassTree24<u64> = MassTree24::new();
        let guard = tree.guard();

        // Get on empty tree should return None
        let value = tree.get_with_guard(b"hello", &guard);
        assert!(value.is_none());
    }

    #[test]
    fn test_masstree24_insert_and_get() {
        let tree: MassTree24<u64> = MassTree24::new();
        let guard = tree.guard();

        // Insert a value
        let result = tree.insert_with_guard(b"hello", 42, &guard);
        assert!(result.is_ok());
        assert!(result.unwrap().is_none()); // No old value

        // Get the value back
        let value = tree.get_with_guard(b"hello", &guard);
        assert!(value.is_some());
        assert_eq!(*value.unwrap(), 42);
    }

    #[test]
    fn test_masstree24_multiple_inserts() {
        let tree: MassTree24<u64> = MassTree24::new();
        let guard = tree.guard();

        // Insert multiple values (up to WIDTH=24 without split)
        for i in 0..20u64 {
            let key = format!("key{i:02}");
            let result = tree.insert_with_guard(key.as_bytes(), i * 10, &guard);
            assert!(result.is_ok(), "Failed to insert key {i}");
        }

        assert_eq!(tree.len(), 20);

        // Verify all values
        for i in 0..20u64 {
            let key = format!("key{i:02}");
            let value = tree.get_with_guard(key.as_bytes(), &guard);
            assert!(value.is_some(), "Key {i} not found");
            assert_eq!(*value.unwrap(), i * 10);
        }
    }

    #[test]
    fn test_masstree24_update_existing() {
        let tree: MassTree24<u64> = MassTree24::new();
        let guard = tree.guard();

        // Insert initial value
        tree.insert_with_guard(b"key", 100, &guard).unwrap();
        assert_eq!(*tree.get_with_guard(b"key", &guard).unwrap(), 100);

        // Update the value
        let old = tree.insert_with_guard(b"key", 200, &guard).unwrap();
        assert!(old.is_some());
        assert_eq!(*old.unwrap(), 100);

        // Verify updated value
        assert_eq!(*tree.get_with_guard(b"key", &guard).unwrap(), 200);

        // Count should still be 1
        assert_eq!(tree.len(), 1);
    }

    #[test]
    fn test_masstree_new() {
        let tree: MassTree<u64> = MassTree::new();
        assert!(tree.is_empty());
        assert_eq!(tree.len(), 0);
    }

    #[test]
    fn test_masstree_insert_and_get() {
        let tree: MassTree<u64> = MassTree::new();
        let guard = tree.guard();

        // Insert a value
        let result = tree.insert_with_guard(b"hello", 42, &guard);
        assert!(result.is_ok());
        assert!(result.unwrap().is_none());

        // Get the value back
        let value = tree.get_with_guard(b"hello", &guard);
        assert!(value.is_some());
        assert_eq!(*value.unwrap(), 42);
    }

    #[test]
    fn test_masstree24_split_triggers() {
        // Test that inserting more than 24 keys triggers a split
        let tree: MassTree24<u64> = MassTree24::new();
        let guard = tree.guard();

        // Insert 24 keys (should all succeed, filling the root leaf)
        for i in 0..24u64 {
            let key = format!("key{i:02}");
            let result = tree.insert_with_guard(key.as_bytes(), i, &guard);
            assert!(result.is_ok(), "Failed to insert key {i}: {result:?}");
        }

        assert_eq!(tree.len(), 24);

        // The 25th key should succeed by triggering a split
        let result = tree.insert_with_guard(b"key24", 24, &guard);
        assert!(
            result.is_ok(),
            "25th insert should succeed with split: {result:?}"
        );
        assert_eq!(tree.len(), 25);

        // Verify we can read the 25th key back
        let value = tree.get_with_guard(b"key24", &guard);
        assert!(value.is_some());
        assert_eq!(*value.unwrap(), 24);

        // Verify all original keys are still readable
        for i in 0..24u64 {
            let key = format!("key{i:02}");
            let value = tree.get_with_guard(key.as_bytes(), &guard);
            assert!(value.is_some(), "key{i:02} not found after split");
            assert_eq!(*value.unwrap(), i);
        }
    }

    #[test]
    fn test_masstree24_many_splits() {
        // Test inserting many more keys that require multiple splits
        let tree: MassTree24<u64> = MassTree24::new();
        let guard = tree.guard();

        // Insert 100 keys (should trigger multiple splits)
        for i in 0..100u64 {
            let key = format!("key{i:03}");
            let result = tree.insert_with_guard(key.as_bytes(), i, &guard);
            assert!(result.is_ok(), "Failed to insert key {i}: {result:?}");
        }

        assert_eq!(tree.len(), 100);

        // Verify all keys are readable
        for i in 0..100u64 {
            let key = format!("key{i:03}");
            let value = tree.get_with_guard(key.as_bytes(), &guard);
            assert!(value.is_some(), "key{i:03} not found");
            assert_eq!(*value.unwrap(), i);
        }
    }

    #[test]
    fn test_masstree24_sequential_u64_limited() {
        // Test inserting keys with u64 sequential pattern
        // Uses 10,000 keys to trigger multiple internode splits
        let tree: MassTree24<u64> = MassTree24::new();
        let guard = tree.guard();

        const KEY_COUNT: u64 = 10_000;

        for i in 0u64..KEY_COUNT {
            let key = i.to_be_bytes();
            let result = tree.insert_with_guard(&key, i, &guard);
            assert!(result.is_ok(), "Failed to insert key {i}: {result:?}");
        }

        assert_eq!(tree.len(), KEY_COUNT as usize);

        // Verify all keys are readable
        for i in 0u64..KEY_COUNT {
            let key = i.to_be_bytes();
            let value = tree.get_with_guard(&key, &guard);
            assert!(value.is_some(), "Key {i} not found in final verification");
            assert_eq!(*value.unwrap(), i);
        }
    }

    #[test]
    fn test_masstree24_internode_splits_stress() {
        // Stress test to verify multi-level internode splits work
        // Uses 50,000 keys to trigger deep tree with multiple internode levels
        let tree: MassTree24<u64> = MassTree24::new();
        let guard = tree.guard();

        const KEY_COUNT: u64 = 50_000;

        for i in 0u64..KEY_COUNT {
            let key = i.to_be_bytes();
            let result = tree.insert_with_guard(&key, i, &guard);
            assert!(result.is_ok(), "Failed to insert key {i}: {result:?}");
        }

        assert_eq!(tree.len(), KEY_COUNT as usize);

        // Verify a sample of keys (every 100th)
        for i in (0u64..KEY_COUNT).step_by(100) {
            let key = i.to_be_bytes();
            let value = tree.get_with_guard(&key, &guard);
            assert!(value.is_some(), "Key {i} not found");
            assert_eq!(*value.unwrap(), i);
        }
    }

    /// Regression test for proptest failure: keys = [[52, 0], [52]]
    /// Key [52] was being lost.
    #[test]
    fn test_proptest_regression_two_keys() {
        let tree: MassTree24<u64> = MassTree24::new();

        let key1: &[u8] = &[52, 0];
        let key2: &[u8] = &[52];

        tree.insert(key1, 0).unwrap();
        tree.insert(key2, 1).unwrap();

        // Verify both keys exist
        let result1 = tree.get(key1);
        let result2 = tree.get(key2);

        assert!(result1.is_some(), "Key [52, 0] lost!");
        assert!(result2.is_some(), "Key [52] lost!");
        assert_eq!(*result1.unwrap(), 0);
        assert_eq!(*result2.unwrap(), 1);
        assert_eq!(tree.len(), 2);
    }

    /// Regression test for proptest failure: 9-byte key of all zeros.
    /// Key was being lost after insert.
    #[test]
    fn test_proptest_regression_nine_zeros() {
        let tree: MassTree24<u64> = MassTree24::new();

        let key: &[u8] = &[0, 0, 0, 0, 0, 0, 0, 0, 0];
        tree.insert(key, 42).unwrap();

        let result = tree.get(key);
        assert!(result.is_some(), "9-byte key not found!");
        assert_eq!(*result.unwrap(), 42);
        assert_eq!(tree.len(), 1);
    }

    /// Test two suffix keys with same 8-byte prefix.
    /// This creates a conflict that should create a layer.
    #[test]
    fn test_two_suffix_keys_same_prefix() {
        let tree: MassTree24<u64> = MassTree24::new();

        // Two 16-byte keys with same 8-byte prefix "prefix00"
        let key1: &[u8] = b"prefix0000000000";
        let key2: &[u8] = b"prefix0000000001";

        tree.insert(key1, 1).unwrap();
        tree.insert(key2, 2).unwrap();

        // Both should be retrievable
        let result1 = tree.get(key1);
        let result2 = tree.get(key2);

        assert!(result1.is_some(), "key1 not found!");
        assert!(result2.is_some(), "key2 not found!");
        assert_eq!(*result1.unwrap(), 1);
        assert_eq!(*result2.unwrap(), 2);
        assert_eq!(tree.len(), 2);
    }

    /// Test many suffix keys with same 8-byte prefix.
    /// Tests layer handling as keys accumulate.
    #[test]
    fn test_many_suffix_keys_same_prefix() {
        let tree: MassTree24<u64> = MassTree24::new();

        // 20 keys with same 8-byte prefix "prefix00"
        for i in 0..20u64 {
            let key = format!("prefix00{i:08}"); // 16 bytes each
            tree.insert(key.as_bytes(), i).unwrap();

            // Immediately verify this key
            let result = tree.get(key.as_bytes());
            assert!(
                result.is_some(),
                "Key {i} ('{key}') not found immediately after insert"
            );
            assert_eq!(*result.unwrap(), i);
        }

        // Verify all keys still retrievable at the end
        for i in 0..20u64 {
            let key = format!("prefix00{i:08}");
            let result = tree.get(key.as_bytes());
            assert!(result.is_some(), "Key {i} ('{key}') not found at end");
            assert_eq!(*result.unwrap(), i);
        }

        assert_eq!(tree.len(), 20);
    }
}
