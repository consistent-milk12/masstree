//! Traits for abstracting over leaf node WIDTH variants.
//!
//! This module defines [`TreePermutation`] and [`TreeLeafNode`] traits that
//! enable generic tree operations over both WIDTH=15 and WIDTH=24 leaf nodes.
//!
//! # Design
//!
//! The traits use static dispatch (generics) for zero-cost abstraction:
//! - No vtable overhead
//! - Full monomorphization
//! - Compiler can inline all trait methods
//!
//! # Implementors
//!
//! - [`TreePermutation`]: `Permuter<WIDTH>`, `Permuter24`
//! - [`TreeLeafNode`]: `LeafNode<S, WIDTH>`, `LeafNode24<S>`

use std::cmp::Ordering;
use std::fmt::Debug;

use crate::key::Key;
use crate::nodeversion::NodeVersion;
use crate::slot::ValueSlot;
use seize::LocalGuard;

// ============================================================================
// Re-exports from value.rs for use in generic code
// ============================================================================

pub use crate::value::InsertTarget;
pub use crate::value::SplitPoint;

// ============================================================================
// Split+Insert Types (Atomic Split+Insert Operation)
// ============================================================================

/// Data for inserting a key during a split operation.
///
/// This bundles all the information needed to insert a new key atomically
/// during a leaf split, matching the C++ `split_into()` semantics.
///
/// # C++ Reference
///
/// In C++ Masstree, `split_into()` takes `ka` (key accessor) and performs
/// both the split and the insert atomically. This struct captures the
/// equivalent data.
#[derive(Debug)]
pub struct SplitInsertData<'a> {
    /// The 8-byte key to insert.
    pub ikey: u64,

    /// The keylenx value (0-8 for inline, 64 for suffix, >=128 for layer).
    pub keylenx: u8,

    /// The suffix bytes, if any (present when `keylenx == KSUF_KEYLENX`).
    pub suffix: Option<&'a [u8]>,

    /// The raw value pointer to insert.
    ///
    /// Ownership transfers to the leaf during the split operation.
    /// Created via `ValueSlot::output_consume_to_raw()`.
    pub value_ptr: *mut u8,
}

/// Result of an atomic split+insert operation.
///
/// Contains the split key for parent propagation and information about
/// where the new key was inserted.
#[derive(Debug, Clone, Copy)]
pub struct SplitInsertResult {
    /// The key that separates the left and right leaves.
    ///
    /// This is the first ikey of the right leaf and becomes the
    /// separator key in the parent internode.
    pub split_ikey: u64,

    /// Which leaf received the new key.
    pub insert_target: InsertTarget,
}

// ============================================================================
//  TreePermutation Trait
// ============================================================================

/// Trait for permutation types used in leaf nodes.
///
/// Abstracts over `Permuter<WIDTH>` (u64) and `Permuter24` (u128), enabling
/// generic tree operations that work with both WIDTH=15 and WIDTH=24 nodes.
///
/// # Associated Types
///
/// - `Raw`: The underlying storage type (`u64` or `u128`)
///
/// # Implementors
///
/// - `Permuter<WIDTH>` for WIDTH in 1..=15
/// - `Permuter24` for WIDTH=24
pub trait TreePermutation: Copy + Clone + Eq + Debug + Send + Sync + Sized + 'static {
    /// Raw storage type for atomic operations.
    ///
    /// - `Permuter<WIDTH>`: `u64`
    /// - `Permuter24`: `u128`
    type Raw: Copy + Clone + Eq + Debug + Send + Sync + 'static;

    /// Number of slots this permutation supports.
    const WIDTH: usize;

    // ========================================================================
    //  Construction
    // ========================================================================

    /// Create an empty permutation with size = 0.
    ///
    /// Slots are arranged so `back()` returns slot 0 initially.
    fn empty() -> Self;

    /// Create a sorted permutation with `n` elements in slots `0..n`.
    ///
    /// The permutation will have size `n` with logical positions `0..n`
    /// mapping to physical slots 0..n in order.
    ///
    /// This is used when creating layer nodes during suffix conflict resolution,
    /// where we need a small number of pre-positioned entries.
    ///
    /// # Arguments
    ///
    /// * `n` - Number of elements (`0 <= n <= WIDTH`)
    ///
    /// # Panics
    ///
    /// Panics in debug mode if `n > WIDTH`.
    ///
    /// # Example
    ///
    /// ```ignore
    /// // Create a permutation with 2 sorted entries
    /// let perm = Permuter::<15>::make_sorted(2);
    /// assert_eq!(perm.size(), 2);
    ///
    /// // Position 0 -> Slot 0
    /// assert_eq!(perm.get(0), 0);
    ///
    /// // Position 1 -> Slot 1
    /// assert_eq!(perm.get(1), 1);
    /// ```
    fn make_sorted(n: usize) -> Self;

    /// Create a permutation from a raw storage value.
    ///
    /// Used when loading from atomic storage.
    ///
    /// # Safety Note
    ///
    /// The raw value should be a valid permutation encoding. Invalid values
    /// may cause debug assertions to fail but won't cause undefined behavior.
    fn from_value(raw: Self::Raw) -> Self;

    // ========================================================================
    //  Accessors
    // ========================================================================

    /// Get the raw storage value.
    ///
    /// Used for atomic store/CAS operations.
    fn value(&self) -> Self::Raw;

    /// Get the number of slots in use.
    fn size(&self) -> usize;

    /// Get the physical slot at logical position `i`.
    ///
    /// # Panics
    ///
    /// Debug-panics if `i >= WIDTH`.
    fn get(&self, i: usize) -> usize;

    /// Get the slot at the back (next free slot to allocate).
    ///
    /// Equivalent to `get(WIDTH - 1)`.
    fn back(&self) -> usize;

    /// Get the slot at `back()` with an offset into the free region.
    ///
    /// `back_at_offset(0)` == `back()`.
    ///
    /// # Panics
    ///
    /// Debug-panics if `size() + offset >= WIDTH`.
    fn back_at_offset(&self, offset: usize) -> usize;

    // ========================================================================
    //  Mutation
    // ========================================================================

    /// Allocate a slot from back and insert at position `i`.
    ///
    /// Returns the allocated physical slot index.
    ///
    /// # Panics
    ///
    /// Debug-panics if `i > size()` or `size() >= WIDTH`.
    fn insert_from_back(&mut self, i: usize) -> usize;

    /// Compute insert result without mutation (for CAS operations).
    ///
    /// Returns `(new_permutation, allocated_slot)`.
    ///
    /// This is used in lock-free CAS insert paths where we need to compute
    /// the new permutation value before attempting an atomic CAS.
    fn insert_from_back_immutable(&self, i: usize) -> (Self, usize);

    /// Swap two slots in the free region (positions >= size).
    ///
    /// Used to skip slot 0 when it can't be reused due to `ikey_bound` constraints.
    fn swap_free_slots(&mut self, pos_i: usize, pos_j: usize);

    /// Set the size without changing slot positions.
    fn set_size(&mut self, n: usize);

    /// Remove the slot at logical position `i`.
    ///
    /// The slot is moved to the free region (back) and size is decremented.
    ///
    /// # Panics
    ///
    /// Debug-panics if `i >= size()`.
    fn remove(&mut self, i: usize);
}

// ============================================================================
//  TreeInternode Trait
// ============================================================================

/// Trait for internode types used in a `MassTree`.
///
/// Abstracts over `InternodeNode` for generic tree operations.
/// Internode WIDTH is fixed at 15 (matching leaf WIDTH for optimal B+tree fanout).
///
/// # Note on Generic Parameter
///
/// Unlike leaf nodes, internodes don't store values - only `u64` keys and `*mut u8`
/// child pointers. The internode type is therefore non-generic, reducing code
/// monomorphization (one `InternodeNode` implementation regardless of slot type).
///
/// # Implementors
///
/// - `InternodeNode` (fixed WIDTH=15)
pub trait TreeInternode: Sized + Send + Sync + 'static {
    /// Node width (max number of children).
    const WIDTH: usize;

    // ========================================================================
    //  Construction
    // ========================================================================

    /// Create a new internode with specified height.
    fn new_boxed(height: u32) -> Box<Self>;

    /// Create a new root internode with specified height.
    fn new_root_boxed(height: u32) -> Box<Self>;

    /// Create a new internode sibling for a split operation.
    ///
    /// The new internode is created with a **split-locked** version copied from the
    /// locked parent. This prevents other threads from locking the sibling until
    /// it is installed into the tree and its parent pointer is set.
    ///
    /// # Help-Along Protocol
    ///
    /// This is the internode equivalent of leaf `NodeVersion::new_for_split()`.
    /// The caller MUST call `version().unlock_for_split()` exactly once after:
    /// 1. The sibling is inserted into its parent (grandparent or new root)
    /// 2. The sibling's parent pointer is set
    ///
    /// # C++ Reference
    ///
    /// Matches `next_child->assign_version(*p)` in `masstree_split.hh:234`.
    ///
    /// # Safety
    ///
    /// The `parent_version` must be from a locked node (the parent being split).
    fn new_boxed_for_split(parent_version: &NodeVersion, height: u32) -> Box<Self>;

    // ========================================================================
    //  Version / Locking
    // ========================================================================

    /// Get reference to node version.
    fn version(&self) -> &NodeVersion;

    // ========================================================================
    //  Structure
    // ========================================================================

    /// Get the height of this internode.
    fn height(&self) -> u32;

    /// Check if children are leaves (height == 0).
    fn children_are_leaves(&self) -> bool;

    /// Get number of keys.
    fn nkeys(&self) -> usize;

    /// Get number of keys using Relaxed ordering.
    ///
    /// Use this variant when the value is only needed for hints/speculation
    /// (e.g., prefetch index calculation) and correctness doesn't depend on it.
    fn nkeys_relaxed(&self) -> usize;

    /// Set number of keys.
    fn set_nkeys(&self, n: u8);

    /// Increment nkeys by 1.
    fn inc_nkeys(&self);

    /// Check if this internode is full.
    fn is_full(&self) -> bool;

    // ========================================================================
    //  Keys
    // ========================================================================

    /// Get key at index (Acquire ordering).
    fn ikey(&self, idx: usize) -> u64;

    /// Get key at index using Relaxed ordering.
    ///
    /// # Safety Justification
    ///
    /// Callers must have already established ordering via `version.stable()`
    /// or equivalent fence. The Acquire fence from `stable()` synchronizes
    /// with writer's Release stores, making Relaxed loads safe.
    ///
    /// This is safe because:
    /// 1. Call sites do `stable()` which spins until `DIRTY_MASK == 0`
    /// 2. `stable()` issues `fence(Acquire)` before returning
    /// 3. The fence synchronizes with writer's `Release` stores
    /// 4. Therefore, Relaxed loads see fully-published values
    fn ikey_relaxed(&self, idx: usize) -> u64;

    /// Get raw pointer to the ikey array for SIMD operations.
    ///
    /// Returns a pointer to the first element of the contiguous ikey array.
    /// The pointer is valid for reading `Self::WIDTH` `u64` values.
    ///
    /// # Safety Context
    ///
    /// Caller must ensure:
    /// 1. Memory ordering established via `stable()` before reading
    /// 2. No concurrent writes to the ikey array (ensured by OCC protocol)
    ///
    /// # Implementation Notes
    /// The pointer targets [`AtomicU64`](std::sync::atomic::AtomicU64) storage, but SIMD treats it as plain
    /// `u64`.
    ///
    /// This is safe because:
    /// - [`AtomicU64`](std::sync::atomic::AtomicU64) has identical layout to `u64` (guaranted by Rust)
    /// - Aligned 64-bit loads are atomic on `x86_64`
    /// - Caller has established ordering via `stable()` fence
    fn ikey_ptr(&self) -> *const u64;

    /// Set key at index.
    fn set_ikey(&self, idx: usize, key: u64);

    /// Compare key at position with search key.
    fn compare_key(&self, search_ikey: u64, p: usize) -> Ordering;

    /// Find insert position for a key.
    fn find_insert_position(&self, insert_ikey: u64) -> usize;

    // ========================================================================
    // Children
    // ========================================================================

    /// Get child pointer at index.
    fn child(&self, idx: usize) -> *mut u8;

    /// Set child pointer at index.
    fn set_child(&self, idx: usize, child: *mut u8);

    /// Assign key and right child at position.
    fn assign(&self, p: usize, ikey: u64, right_child: *mut u8);

    /// Insert key and child at position, shifting existing entries.
    fn insert_key_and_child(&self, p: usize, new_ikey: u64, new_child: *mut u8);

    // ========================================================================
    // Navigation
    // ========================================================================

    /// Get parent pointer.
    fn parent(&self) -> *mut u8;

    /// Set parent pointer.
    fn set_parent(&self, parent: *mut u8);

    /// Check if this is a root node.
    fn is_root(&self) -> bool;

    // ========================================================================
    //  Split Support
    // ========================================================================

    /// Shift entries from another internode.
    fn shift_from(&self, dst_pos: usize, src: &Self, src_pos: usize, count: usize);

    /// Split this internode into a new sibling while inserting a key/child.
    ///
    /// This method performs the split AND updates all children's parent pointers
    /// in `new_right` to point to `new_right_ptr`. This is critical for correctness:
    /// parent updates must happen inside `split_into` (before returning) to prevent
    /// races where a thread sees a child with a stale parent pointer.
    ///
    /// # Arguments
    ///
    /// * `new_right` - The new right sibling (pre-allocated, mutable reference)
    /// * `new_right_ptr` - Raw pointer to `new_right` for setting parent pointers
    /// * `insert_pos` - Position where the new key/child should be inserted
    /// * `insert_ikey` - The key to insert
    /// * `insert_child` - The child pointer to insert
    ///
    /// # Returns
    ///
    /// `(popup_key, insert_went_left)` where:
    /// - `popup_key` is the key that goes to the parent
    /// - `insert_went_left` is true if the insert went to the left sibling
    ///
    /// # Safety
    ///
    /// * `new_right_ptr` must point to `new_right`
    /// * The caller must hold the lock on `self`
    #[must_use = "popup_key must be inserted into parent node to complete the split"]
    fn split_into(
        &self,
        new_right: &mut Self,
        new_right_ptr: *mut Self,
        insert_pos: usize,
        insert_ikey: u64,
        insert_child: *mut u8,
    ) -> (u64, bool);
}

// ============================================================================
//  TreeLeafNode Trait
// ============================================================================

/// Trait for leaf node types that can be used in a [`crate::MassTree`].
///
/// Abstracts over `LeafNode<S, WIDTH>` and `LeafNode24<S>`, enabling generic
/// tree operations that work with both WIDTH=15 and WIDTH=24 nodes.
///
/// # Type Parameters
///
/// - `S`: The slot type (e.g., `LeafValue<V>` or `LeafValueIndex<V>`)
///
/// # Associated Types
///
/// - `Perm`: The permutation type for this leaf
/// - `Internode`: The internode type for this tree variant
///
/// # Implementors
///
/// - `LeafNode<S, WIDTH>` for WIDTH in 1..=15
/// - `LeafNode24<S>` for WIDTH=24
pub trait TreeLeafNode<S: ValueSlot>: Sized + Send + Sync + 'static {
    /// The permutation type for this leaf.
    type Perm: TreePermutation;

    /// The internode type for this tree variant.
    type Internode: TreeInternode;

    /// Node width (number of slots).
    const WIDTH: usize;

    /// Split threshold (trigger split when size >= this).
    ///
    /// Default is 80% of WIDTH to reduce split frequency on hot keys.
    /// This leaves headroom for inserts before triggering a split.
    ///
    /// - WIDTH=15: threshold=12 (80%)
    /// - WIDTH=24: threshold=19 (79%)
    const SPLIT_THRESHOLD: usize;

    /// Inline suffix bag capacity in bytes.
    /// Inline suffix storage capacity in bytes.
    ///
    /// Used by the pre-allocation heuristic to predict inline overflow
    /// before acquiring the leaf lock. Override for leaf types with
    /// non-default inline capacity.
    ///
    /// Configurable via `large-suffix-capacity` feature:
    /// - Default: 256 bytes (better cache locality for `deep_trie` workloads)
    /// - Large: 512 bytes (fewer heap allocations, -44% on `deep_trie`)
    #[cfg(not(feature = "large-suffix-capacity"))]
    const INLINE_KSUF_CAPACITY: usize = 256;

    /// Inline suffix storage capacity (large variant, 512 bytes).
    #[cfg(feature = "large-suffix-capacity")]
    const INLINE_KSUF_CAPACITY: usize = 512;

    // ========================================================================
    //  Construction
    // ========================================================================

    /// Create a new leaf node (heap-allocated).
    fn new_boxed() -> Box<Self>;

    /// Create a new root leaf node (heap-allocated).
    fn new_root_boxed() -> Box<Self>;

    /// Create a new leaf node configured as a layer root.
    ///
    /// The returned node has:
    /// - `is_root` flag set via `version.mark_root()`
    /// - `parent` pointer set to null
    ///
    /// Layer roots are used when creating sublayers for keys longer than 8 bytes.
    /// When two keys share the same 8-byte ikey but have different suffixes,
    /// a new layer is created to distinguish them by their next 8-byte chunk.
    fn new_layer_root_boxed() -> Box<Self>;

    // ========================================================================
    //  NodeVersion Operations
    // ========================================================================

    /// Get a reference to the node's version.
    ///
    /// Used for optimistic concurrency control (OCC) and locking.
    fn version(&self) -> &NodeVersion;

    // ========================================================================
    //  Permutation Operations
    // ========================================================================

    /// Load the current permutation with Acquire ordering.
    fn permutation(&self) -> Self::Perm;

    /// Store a new permutation with Release ordering.
    fn set_permutation(&self, perm: Self::Perm);

    /// Store a new permutation with Relaxed ordering.
    ///
    /// Use when caller holds the node lock. The lock's Release fence on unlock
    /// provides necessary synchronization, allowing us to skip Release here.
    fn set_permutation_relaxed(&self, perm: Self::Perm);

    /// Load raw permutation value with Acquire ordering.
    ///
    /// Used for freeze detection without constructing a Permuter.
    fn permutation_raw(&self) -> <Self::Perm as TreePermutation>::Raw;

    // ========================================================================
    //  Key Operations
    // ========================================================================

    /// Get ikey at physical slot using Acquire ordering.
    ///
    /// # Panics
    ///
    /// Debug-panics if `slot >= WIDTH`.
    fn ikey(&self, slot: usize) -> u64;

    /// Get ikey at physical slot using Relaxed ordering.
    ///
    /// # Safety Justification
    ///
    /// Safe to use when caller has already loaded permutation with Acquire
    /// ordering and will validate with OCC at the end of the read operation.
    /// Avoids redundant Acquire fences on each ikey load.
    ///
    /// # Panics
    ///
    /// Debug-panics if `slot >= WIDTH`.
    fn ikey_relaxed(&self, slot: usize) -> u64;

    /// Set ikey at physical slot.
    ///
    /// # Panics
    ///
    /// Debug-panics if `slot >= WIDTH`.
    fn set_ikey(&self, slot: usize, ikey: u64);

    /// Set ikey at physical slot with Relaxed ordering.
    ///
    /// Use when a subsequent Release store (e.g., permutation update)
    /// will serve as the linearization point.
    fn set_ikey_relaxed(&self, slot: usize, ikey: u64);

    /// Get ikey bound (slot 0's ikey for B-link navigation).
    ///
    /// The `ikey_bound` is the smallest ikey in this leaf and is used
    /// for navigating to the correct sibling during splits.
    fn ikey_bound(&self) -> u64;

    /// Find all physical slots with matching ikey, returning a bitmask.
    ///
    /// Returns a `u32` where bit `i` is set if `self.ikey(i) == target_ikey`.
    /// Used for SIMD-accelerated key search.
    ///
    /// The default implementation uses a scalar loop. Implementations may
    /// override with SIMD for better performance.
    #[inline]
    fn find_ikey_matches(&self, target_ikey: u64) -> u32 {
        let mut mask: u32 = 0;
        for slot in 0..Self::WIDTH {
            if self.ikey(slot) == target_ikey {
                mask |= 1 << slot;
            }
        }
        mask
    }

    /// Get keylenx at physical slot.
    ///
    /// Values:
    /// - 0-8: inline key length
    /// - 64 (`KSUF_KEYLENX)`: has suffix
    /// - >=128 (LAYER_KEYLENX): is layer pointer
    fn keylenx(&self, slot: usize) -> u8;

    /// Set keylenx at physical slot.
    fn set_keylenx(&self, slot: usize, keylenx: u8);

    /// Set keylenx at physical slot with Relaxed ordering.
    ///
    /// Use when a subsequent Release store will serve as the linearization point.
    fn set_keylenx_relaxed(&self, slot: usize, keylenx: u8);

    /// Check if slot contains a layer pointer.
    ///
    /// A layer pointer indicates this slot descends into a sublayer
    /// for keys longer than 8 bytes at this level.
    fn is_layer(&self, slot: usize) -> bool;

    /// Check if slot has a suffix.
    fn has_ksuf(&self, slot: usize) -> bool;

    // ========================================================================
    //  Value Operations
    // ========================================================================

    /// Load value pointer at slot.
    ///
    /// Returns raw pointer to either an `Arc<V>` (value mode) or
    /// a sublayer root node (layer mode).
    fn leaf_value_ptr(&self, slot: usize) -> *mut u8;

    /// Store value pointer at slot.
    fn set_leaf_value_ptr(&self, slot: usize, ptr: *mut u8);

    /// Store value pointer at slot with Relaxed ordering.
    ///
    /// Use when a subsequent Release store will serve as the linearization point.
    fn set_leaf_value_ptr_relaxed(&self, slot: usize, ptr: *mut u8);

    /// Update an existing value in place.
    ///
    /// This is an optimization for updates where the slot already contains
    /// a value. For inline storage, this skips the sentinel store since
    /// the sentinel is already in place.
    ///
    /// Default implementation calls `set_leaf_value_ptr`.
    #[inline(always)]
    fn update_leaf_value_in_place(&self, slot: usize, ptr: *mut u8) {
        self.set_leaf_value_ptr(slot, ptr);
    }

    /// CAS value pointer at slot.
    ///
    /// Used in CAS insert path to atomically claim a slot.
    ///
    /// # Errors
    ///
    /// Returns `Err(actual)` containing the actual pointer value if the CAS
    /// failed due to a concurrent modification (the slot's current value
    /// did not match `expected`).
    fn cas_slot_value(
        &self,
        slot: usize,
        expected: *mut u8,
        new_value: *mut u8,
    ) -> Result<(), *mut u8>;

    // ========================================================================
    //  Slot Clearing (for gc_layer)
    // ========================================================================

    /// Clear a slot completely, removing any value or layer pointer.
    ///
    /// Used by `gc_layer` when cleaning up an empty sublayer.
    /// The parent leaf's slot that pointed to the sublayer is cleared.
    ///
    /// # Safety
    ///
    /// The caller must ensure:
    /// - The leaf is locked
    /// - The slot is valid (0..WIDTH)
    /// - Any value/layer at this slot has been or will be properly retired
    fn clear_slot(&self, slot: usize);

    /// Clear a slot and update permutation.
    ///
    /// This is a convenience method that:
    /// 1. Clears the slot contents
    /// 2. Removes the slot from the permutation
    ///
    /// # Safety
    ///
    /// The caller must ensure the leaf is locked.
    fn clear_slot_and_permutation(&self, slot: usize);

    // ========================================================================
    //  Size Operations
    // ========================================================================

    /// Get number of keys in this leaf.
    #[inline(always)]
    fn size(&self) -> usize {
        self.permutation().size()
    }

    /// Check if leaf is empty.
    #[inline(always)]
    fn is_empty(&self) -> bool {
        self.size() == 0
    }

    /// Check if leaf is full.
    #[inline(always)]
    fn is_full(&self) -> bool {
        self.size() >= Self::WIDTH
    }

    // ========================================================================
    //  Navigation (B-link tree pointers)
    // ========================================================================

    /// Get next leaf pointer (with mark bit cleared).
    ///
    /// The next pointer may be marked during splits. This method
    /// returns the clean pointer for following the linked list.
    fn safe_next(&self) -> *mut Self;

    /// Check if next pointer is marked.
    ///
    /// A marked next pointer indicates a split is in progress.
    fn next_is_marked(&self) -> bool;

    /// Set next leaf pointer.
    fn set_next(&self, next: *mut Self);

    /// Mark the next pointer (during split).
    fn mark_next(&self);

    /// Unmark the next pointer.
    fn unmark_next(&self);

    /// Get previous leaf pointer.
    fn prev(&self) -> *mut Self;

    /// Set previous leaf pointer.
    fn set_prev(&self, prev: *mut Self);

    /// Unlink this leaf from the B-link doubly-linked chain.
    ///
    /// Used when removing an empty leaf from the tree.
    ///
    /// # Safety
    ///
    /// - Caller must hold the version lock on this leaf
    /// - `self.prev()` must be non-null (not the leftmost leaf)
    /// - The prev and next pointers must be valid leaves
    unsafe fn unlink_from_chain(&self);

    /// Get parent internode pointer.
    fn parent(&self) -> *mut u8;

    /// Set parent internode pointer.
    fn set_parent(&self, parent: *mut u8);

    // ========================================================================
    //  Slot Assignment Helpers
    // ========================================================================

    /// Check if slot 0 can be reused for a new key.
    ///
    /// Slot 0 stores `ikey_bound()` which must be preserved if this
    /// leaf has a predecessor (prev != null). Slot 0 can only be
    /// reused if the new key has the same ikey as the current bound.
    fn can_reuse_slot0(&self, new_ikey: u64) -> bool;

    // ========================================================================
    //  CAS Insert Support
    // ========================================================================

    /// Store key metadata (`ikey`, `keylenx`) for a CAS insert attempt.
    ///
    /// # Safety
    ///
    /// - The caller must have successfully claimed the slot via `cas_slot_value` and ensured
    ///   the slot still belongs to the CAS attempt (i.e. `leaf_values[slot]` still equals the
    ///   claimed pointer).
    ///
    /// Note: writing key metadata *before* claiming the slot is not safe in this design because
    /// multiple concurrent CAS attempts can overwrite each other's metadata before publish.
    unsafe fn store_key_data_for_cas(&self, slot: usize, ikey: u64, keylenx: u8);

    /// Load the raw slot value pointer atomically.
    ///
    /// Used to verify slot ownership after CAS claim.
    fn load_slot_value(&self, slot: usize) -> *mut u8;

    /// Get the raw next pointer (may be marked).
    ///
    /// Returns the next pointer without unmarking. Use to check
    /// if a split is in progress (marked) or get the raw value.
    fn next_raw(&self) -> *mut Self;

    /// Wait for an in-progress split to complete.
    ///
    /// Spins until the next pointer is unmarked and version is stable.
    fn wait_for_split(&self);

    // ========================================================================
    //  Split Operations
    // ========================================================================

    /// Calculate the optimal split point.
    ///
    /// # Arguments
    ///
    /// * `insert_pos` - Logical position where new key will be inserted
    /// * `insert_ikey` - The key being inserted
    ///
    /// # Returns
    ///
    /// `Some(SplitPoint)` with position and split key, or `None` if split
    /// is not possible (e.g., empty leaf).
    fn calculate_split_point(&self, insert_pos: usize, insert_ikey: u64) -> Option<SplitPoint>;

    /// Split this leaf at `split_pos` using a pre-allocated target.
    ///
    /// Moves entries from `split_pos..size` to `new_leaf_ptr`.
    /// The caller is responsible for allocating and tracking the new leaf.
    ///
    /// # Returns
    ///
    /// `(split_ikey, insert_target)` tuple where:
    /// - `split_ikey` is the first key of the new leaf (separator for parent)
    /// - `insert_target` indicates which leaf should receive the new key
    ///
    /// # Safety
    ///
    /// - Caller must hold the leaf lock (if concurrent)
    /// - `new_leaf_ptr` must point to valid, initialized leaf memory
    /// - The new leaf should be freshly allocated (empty) with split-locked version
    /// - `guard` must be valid
    unsafe fn split_into_preallocated(
        &self,
        split_pos: usize,
        new_leaf_ptr: *mut Self,
        guard: &seize::LocalGuard<'_>,
    ) -> (u64, InsertTarget);

    /// Move ALL entries to a new right leaf.
    ///
    /// Used for the edge case where `split_pos == 0` in post-insert coordinates.
    /// The original leaf becomes empty, and all entries move to the new leaf.
    ///
    /// # Safety
    ///
    /// Same requirements as `split_into_preallocated`.
    unsafe fn split_all_to_right_preallocated(
        &self,
        new_leaf_ptr: *mut Self,
        guard: &seize::LocalGuard<'_>,
    ) -> (u64, InsertTarget);

    /// Atomically split this leaf AND insert a new key.
    ///
    /// This is the key optimization matching C++ Masstree's `split_into()`.
    /// Unlike `split_into_preallocated()` which only moves existing keys,
    /// this method performs the split and insert in one atomic operation.
    ///
    /// # Algorithm (matches C++ `split_into()`)
    ///
    /// Given: `split_pos`, `insert_pos`, existing entries `[0..size)`
    ///
    /// **Case 1: `insert_pos < split_pos`** (new key → LEFT leaf)
    /// - Move `[split_pos, size)` to right leaf
    /// - Insert new key in left leaf at `insert_pos`
    /// - `split_ikey` = right leaf's slot 0
    ///
    /// **Case 2: `insert_pos >= split_pos`** (new key → RIGHT leaf)
    /// - Interleaved loop from `split_pos` to `size`:
    ///   - If position == adjusted insert position: write new key
    ///   - Else: move existing key
    /// - `split_ikey` = right leaf's slot 0 (may be the new key)
    ///
    /// **Case 3: Sequential optimization** (`insert_pos == size`, `split_pos == size`)
    /// - Move 0 keys (left keeps all existing)
    /// - Right leaf gets ONLY the new key at slot 0
    /// - `split_ikey` = new key's ikey
    ///
    /// # Returns
    ///
    /// [`SplitInsertResult`] containing the split key for parent propagation
    /// and information about which leaf received the new key.
    ///
    /// # Critical Invariants
    ///
    /// 1. **Slot-0 / `ikey_bound`**: Right leaf's slot 0 is ALWAYS filled (never empty)
    /// 2. **No-abandon**: After `mark_split()`, propagation must complete
    /// 3. **Memory ordering**: Relaxed stores during move, Release on permutation
    /// 4. **Value ownership**: `value_ptr` ownership transfers to leaf during split
    ///
    /// # Safety
    ///
    /// - Caller must hold the leaf lock
    /// - `new_leaf_ptr` must point to valid, freshly-allocated leaf memory
    /// - The new leaf should be freshly allocated (empty) with split-locked version
    /// - `insert_data.value_ptr` must be a valid pointer from `output_consume_to_raw()`
    /// - `guard` must be valid
    ///
    /// # C++ Reference
    ///
    /// Matches `leaf::split_into()` in `masstree_split.hh:63-142`.
    unsafe fn split_and_insert(
        &self,
        split_pos: usize,
        new_leaf_ptr: *mut Self,
        insert_pos: usize,
        insert_data: SplitInsertData<'_>,
        guard: &seize::LocalGuard<'_>,
    ) -> SplitInsertResult;

    // ========================================================================
    //  Sibling Link Helper (for split)
    // ========================================================================

    /// Link this leaf to a new sibling (B-link tree threading).
    ///
    /// Sets up the doubly-linked list: `self.next = new_sibling`,
    /// `new_sibling.prev = self`, and if there was an old next,
    /// updates `old_next.prev = new_sibling`.
    ///
    /// # Safety
    ///
    /// - `new_sibling` must be a valid pointer to a freshly allocated leaf
    /// - Caller must hold the leaf lock
    unsafe fn link_sibling(&self, new_sibling: *mut Self);

    // ========================================================================
    //  Suffix Operations (for split)
    // ========================================================================

    /// Get suffix at slot (if any).
    ///
    /// Returns `None` if no suffix is stored at this slot.
    fn ksuf(&self, slot: usize) -> Option<&[u8]>;

    /// Assign a suffix to a slot (normal operation).
    ///
    /// Uses the permutation to determine which slots are active.
    /// This is the hot path for normal inserts/updates.
    ///
    /// # Returns
    ///
    /// A raw pointer to the old external suffix bag that must be retired
    /// via [`retire_suffix_bag_ptr`](Self::retire_suffix_bag_ptr) **after
    /// releasing the leaf lock**. Returns null if no retirement is needed
    /// (fast-path inline/in-place assignment, or no previous external bag).
    ///
    /// # Safety
    ///
    /// - Caller must hold the leaf lock
    /// - Slot must be valid
    unsafe fn assign_ksuf(
        &self,
        slot: usize,
        suffix: &[u8],
        guard: &seize::LocalGuard<'_>,
    ) -> *mut u8;

    /// Retire a suffix bag pointer that was returned by [`assign_ksuf`](Self::assign_ksuf).
    ///
    /// This must be called **after releasing the leaf lock** to avoid holding
    /// locks during EBR retirement (which may issue `sys_membarrier` syscalls).
    ///
    /// # Safety
    ///
    /// - `ptr` must be non-null and must have come from `assign_ksuf`
    /// - `guard` must be from this tree's collector
    unsafe fn retire_suffix_bag_ptr(&self, ptr: *mut u8, guard: &seize::LocalGuard<'_>);

    /// Assign a suffix to a slot during node initialization (e.g., splits).
    ///
    /// Unlike `assign_ksuf`, this assumes slots `0..slot` are already filled
    /// sequentially and doesn't rely on the permutation. Used during split
    /// operations when the new node's permutation hasn't been set up yet.
    ///
    /// This matches C++ `assign_ksuf`'s `initializing=true` path in
    /// `masstree_struct.hh:728`.
    ///
    /// # Safety
    ///
    /// - Caller must hold the leaf lock
    /// - Slot must be valid
    /// - Slots 0..slot must already be filled sequentially
    unsafe fn assign_ksuf_init(&self, slot: usize, suffix: &[u8], guard: &seize::LocalGuard<'_>);

    /// Clear suffix at slot.
    ///
    /// # Safety
    ///
    /// - Caller must hold the leaf lock
    unsafe fn clear_ksuf(&self, slot: usize, guard: &seize::LocalGuard<'_>);

    /// Take ownership of the value pointer at slot (for moving during split).
    ///
    /// Returns the pointer and clears the slot. Used when moving entries
    /// between leaves during a split.
    fn take_leaf_value_ptr(&self, slot: usize) -> *mut u8;

    // ========================================================================
    //  Suffix Comparison Operations
    // ========================================================================

    /// Check if a slot's suffix equals the given suffix.
    ///
    /// Returns `false` if:
    /// - Slot has no suffix (`keylenx != KSUF_KEYLENX`)
    /// - Suffix bag is null
    /// - Suffixes don't match
    ///
    /// # Panics
    ///
    /// Panics in debug mode if `slot >= WIDTH`.
    #[must_use]
    fn ksuf_equals(&self, slot: usize, suffix: &[u8]) -> bool;

    /// Compare a slot's suffix with the given suffix.
    ///
    /// Returns `None` if the slot has no suffix.
    /// Returns `Some(Ordering)` if comparison is possible.
    ///
    /// # Panics
    ///
    /// Panics in debug mode if `slot >= WIDTH`.
    #[must_use]
    fn ksuf_compare(&self, slot: usize, suffix: &[u8]) -> Option<Ordering>;

    /// Get the suffix for a slot, or an empty slice if none.
    ///
    /// Convenience wrapper around `ksuf()`.
    ///
    /// # Panics
    ///
    /// Panics in debug mode if `slot >= WIDTH`.
    #[must_use]
    #[inline(always)]
    fn ksuf_or_empty(&self, slot: usize) -> &[u8] {
        self.ksuf(slot).unwrap_or(&[])
    }

    /// Check if a slot's key (ikey + suffix) matches the given full key.
    ///
    /// This compares both the 8-byte ikey and the suffix (if any).
    ///
    /// # Arguments
    ///
    /// * `slot` - Physical slot index
    /// * `ikey` - The 8-byte key to compare
    /// * `suffix` - The suffix to compare (bytes after the first 8)
    ///
    /// # Returns
    ///
    /// `true` if both ikey and suffix match.
    ///
    /// # Panics
    ///
    /// Panics in debug mode if `slot >= WIDTH`.
    #[must_use]
    fn ksuf_matches(&self, slot: usize, ikey: u64, suffix: &[u8]) -> bool;

    /// Check if a slot matches the given key parameters, with layer detection.
    ///
    /// This is the layer-aware version of `ksuf_matches` that returns detailed
    /// match information needed for layer traversal.
    ///
    /// # Arguments
    ///
    /// * `slot` - Physical slot index
    /// * `keylenx` - The keylenx of the search key (0-8 for inline, `KSUF_KEYLENX` for suffix)
    /// * `suffix` - The suffix bytes to match (empty if inline key)
    ///
    /// # Returns
    ///
    /// * `1` - Exact match (ikey, keylenx, and suffix all match)
    /// * `0` - Same ikey but different key (keylenx or suffix mismatch)
    /// * `-8` - Slot is a layer pointer; caller should shift key by 8 bytes and descend
    ///
    /// # Note
    ///
    /// The ikey is assumed to already match (caller should check `leaf.ikey(slot) == ikey`
    /// before calling this method).
    ///
    /// # Panics
    ///
    /// Panics in debug mode if `slot >= WIDTH`.
    #[must_use]
    fn ksuf_match_result(&self, slot: usize, keylenx: u8, suffix: &[u8]) -> i32;

    // ========================================================================
    //  Cache Optimization
    // ========================================================================

    /// Prefetch the leaf node's data into cache.
    ///
    /// Brings the node's key arrays (`ikey0`, `keylenx`) and value pointers
    /// (`leaf_values`) into CPU cache before they're accessed, reducing memory
    /// latency during sequential scanning.
    ///
    /// # C++ Reference
    ///
    /// Matches C++ `leaf::prefetch()` pattern from `masstree_scan.hh:195, 299`.
    fn prefetch(&self);

    /// Prefetch the ikey at the given slot into CPU cache.
    ///
    /// This is used during linear search to hide memory latency by
    /// prefetching future ikeys while processing current ones.
    ///
    /// # Arguments
    ///
    /// * `slot` - Physical slot index (0..WIDTH)
    ///
    /// # Default Implementation
    ///
    /// No-op. Implementations may override with actual prefetch.
    #[inline(always)]
    fn prefetch_ikey(&self, _slot: usize) {
        // Default no-op; implementations may override
    }

    /// Prefetch permutation and ikeys for point lookups.
    ///
    /// Lighter-weight than [`Self::prefetch`] - only fetches cache lines
    /// needed for the search loop. Call before `version.stable()` to
    /// hide memory latency during version spin-wait.
    ///
    /// # Default Implementation
    ///
    /// No-op. Implementations may override with actual prefetch.
    #[inline(always)]
    fn prefetch_for_search(&self) {
        // Default no-op; implementations may override
    }

    /// Prefetch for search, adapting to node size.
    ///
    /// Avoids unnecessary prefetch instructions for small nodes by
    /// only prefetching cache lines that contain valid keys.
    ///
    /// # Arguments
    ///
    /// * `size` - Number of keys in the node
    ///
    /// # Default Implementation
    ///
    /// Falls back to unconditional [`Self::prefetch_for_search`].
    #[inline(always)]
    fn prefetch_for_search_adaptive(&self, size: usize) {
        let _ = size;
        self.prefetch_for_search();
    }

    // ========================================================================
    //  Modification State (modstate) Operations
    // ========================================================================

    /// Get the modification state.
    ///
    /// Returns one of:
    /// - `0` (`MODSTATE_INSERT`): Normal insert mode
    /// - `1` (`MODSTATE_REMOVE`): Node is being removed
    /// - `2` (`MODSTATE_DELETED_LAYER`): Layer has been garbage collected
    ///
    /// # C++ Reference
    ///
    /// Matches `leaf::modstate_` in `masstree_struct.hh:264-270`.
    fn modstate(&self) -> u8;

    /// Set the modification state.
    fn set_modstate(&self, state: u8);

    /// Check if this layer has been deleted (garbage collected).
    ///
    /// This is distinct from `version.is_deleted()`:
    /// - `is_deleted()` means the node itself is removed from the tree
    /// - `deleted_layer()` means the sublayer this node was root of has been gc'd
    ///
    /// When `deleted_layer()` is true, readers should reset their key position
    /// and retry from the main tree root.
    ///
    /// # C++ Reference
    ///
    /// Matches `leaf::deleted_layer()` in `masstree_struct.hh:456-458`.
    fn deleted_layer(&self) -> bool;

    /// Mark this layer as deleted (for `gc_layer`).
    ///
    /// Called when garbage collecting an empty sublayer.
    fn mark_deleted_layer(&self);

    /// Mark this node as being in remove mode.
    ///
    /// Called at the start of a remove operation.
    fn mark_remove(&self);

    /// Check if this node is in remove mode.
    fn is_removing(&self) -> bool;

    // =========================================================================
    //  Empty State (for lazy coalescing)
    // =========================================================================

    /// Check if this leaf is in empty state (modstate == `MODSTATE_EMPTY`).
    ///
    /// Empty state means all keys were removed and the leaf is available
    /// for reuse by insert or cleanup by background coalescing.
    ///
    /// Note: Use `is_empty()` (inherited from trait) to check if permutation
    /// size is 0. Use `is_empty_state()` to check the modstate flag.
    fn is_empty_state(&self) -> bool;

    /// Mark this leaf as empty (all keys removed).
    ///
    /// Called when the last key is removed from a leaf.
    fn mark_empty(&self);

    /// Clear empty state, returning to normal insert mode.
    ///
    /// Called when an empty leaf is being reused for a new insert.
    fn clear_empty_state(&self);
}

// =============================================================================
// LayerCapableLeaf Trait
// =============================================================================

/// Extension trait for Arc-mode leaves that support layer creation.
///
/// This trait adds layer-specific operations needed for handling suffix conflicts
/// in keys longer than 8 bytes. It is separate from [`TreeLeafNode`] because the
/// methods are specific to `LeafValue<V>` mode (Arc-wrapped values).
///
/// # When Layer Creation Occurs
///
/// Layer creation is triggered when:
/// 1. Two keys share the same 8-byte ikey
/// 2. Both have suffixes (bytes beyond the first 8)
/// 3. The suffixes differ
/// 4. Neither slot is already a layer pointer
///
/// This is the "Conflict" case in `InsertSearchResultGeneric`.
///
/// # Implementors
///
/// - `LeafNode24<LeafValue<V>>`
pub trait LayerCapableLeaf<S: ValueSlot>: TreeLeafNode<S>
where
    S::Value: Send + Sync + 'static,
    S::Output: Send + Sync,
{
    /// Try to clone the Arc value from a slot.
    ///
    /// Returns `None` if:
    /// - Slot is empty (null pointer)
    /// - Slot contains a layer pointer (`keylenx >= LAYER_KEYLENX`)
    ///
    /// # Safety Considerations
    ///
    /// This method is safe to call, but the caller should:
    /// - Hold the node lock (for write operations), OR
    /// - Have validated the version (for read operations)
    ///
    /// The returned `Arc<V>` is a new strong reference; the original
    /// slot's reference count is incremented.
    ///
    /// # Panics
    ///
    /// Panics in debug mode if `slot >= WIDTH`.
    fn try_clone_output(&self, slot: usize) -> Option<S::Output>;

    /// Assign a slot from a Key iterator with an Arc value.
    ///
    /// This method sets up a slot with:
    /// - `ikey` from `key.ikey()`
    /// - `keylenx` computed from `key.has_suffix()`:
    ///   - If `key.has_suffix()`: `KSUF_KEYLENX` (64)
    ///   - Otherwise: `key.current_len().min(8)` (0-8)
    /// - Value pointer from `Arc::into_raw(value)`
    /// - Suffix data via `assign_ksuf()` if `key.has_suffix()`
    ///
    /// # Arguments
    ///
    /// * `slot` - Physical slot index (0..WIDTH)
    /// * `key` - The key containing ikey and suffix information
    /// * `value` - The Arc-wrapped value. Must be `Some`; `None` will panic.
    /// * `guard` - Seize guard for deferred suffix bag retirement
    ///
    /// # Safety
    ///
    /// - Caller must hold the node lock
    /// - `guard` must come from this tree's collector
    /// - Slot must be unoccupied or caller must handle cleanup of old value
    ///
    /// # Panics
    ///
    /// - Panics if `value` is `None` (use layer pointer setup methods instead)
    /// - Panics in debug mode if `slot >= WIDTH`
    unsafe fn assign_from_key_arc(
        &self,
        slot: usize,
        key: &Key<'_>,
        value: Option<S::Output>,
        guard: &LocalGuard<'_>,
    );

    /// Assign a suffix using a pre-alloced buffer.
    ///
    /// # Safety
    ///
    /// Check `assign_ksuf`.
    unsafe fn assign_ksuf_prealloc(
        &self,
        slot: usize,
        suffix: &[u8],
        guard: &LocalGuard<'_>,
        _prealloc: Vec<u8>,
    ) -> *mut u8 {
        unsafe { self.assign_ksuf(slot, suffix, guard) }
    }
}

// ============================================================================
//  Tests
// ============================================================================

#[cfg(test)]
mod unit_tests;
