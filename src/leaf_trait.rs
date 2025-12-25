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

use std::fmt::Debug;
use std::sync::Arc;

use crate::key::Key;
use crate::nodeversion::NodeVersion;
use crate::slot::ValueSlot;
use crate::value::LeafValue;
use seize::LocalGuard;

// ============================================================================
// Re-exports from value.rs for use in generic code
// ============================================================================

pub use crate::value::InsertTarget;
pub use crate::value::SplitPoint;

// ============================================================================
//  CAS Permutation Error
// ============================================================================

/// Error returned when a CAS permutation operation fails.
///
/// Contains the current permutation value that caused the CAS to fail.
/// Use `is_frozen()` to check if a split is in progress.
#[derive(Debug, Clone, Copy)]
pub struct CasPermutationError<P: TreePermutation> {
    /// The current permutation value in the node.
    pub current: P,
}

impl<P: TreePermutation> CasPermutationError<P> {
    /// Create a new CAS permutation error.
    #[inline(always)]
    pub const fn new(current: P) -> Self {
        Self { current }
    }

    /// Check if the failure was due to frozen state (split in progress).
    #[inline(always)]
    pub fn is_frozen(&self) -> bool {
        P::is_frozen_raw(self.current.value())
    }

    /// Get the current permutation value.
    #[inline(always)]
    pub const fn current(&self) -> P {
        self.current
    }
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

    // ========================================================================
    //  Freeze Operations
    // ========================================================================

    /// Check if a raw permutation value is frozen.
    ///
    /// A frozen permutation indicates a split is in progress.
    /// CAS insert threads should fall back to the locked path.
    fn is_frozen_raw(raw: Self::Raw) -> bool;

    /// Freeze a raw permutation value.
    ///
    /// Returns a frozen value that will fail any CAS with a valid expected.
    fn freeze_raw(raw: Self::Raw) -> Self::Raw;
}

// ============================================================================
//  FreezeGuardOps Trait
// ============================================================================

/// Operations that freeze guards must support.
///
/// This trait abstracts over `FreezeGuard<'a, S, WIDTH>` (WIDTH=15) and
/// `FreezeGuard24<'a, S>` (WIDTH=24), enabling generic split operations.
///
/// A freeze guard captures a snapshot of the permutation at freeze time and
/// provides panic safety by restoring the original permutation if dropped
/// while still active.
///
/// # Implementors
///
/// - `FreezeGuard<'a, S, WIDTH>` for WIDTH in 1..=15
/// - `FreezeGuard24<'a, S>` for WIDTH=24
pub trait FreezeGuardOps<P: TreePermutation> {
    /// Get the permutation snapshot captured at freeze time.
    ///
    /// This is the authoritative membership for split computation.
    /// It includes all CAS inserts that published before freeze succeeded.
    fn snapshot(&self) -> P;

    /// Get the raw snapshot value.
    ///
    /// Used for debugging/logging and low-level operations.
    fn snapshot_raw(&self) -> P::Raw;

    /// Set whether the guard is active.
    ///
    /// When active, dropping the guard will restore the original permutation
    /// (panic safety). Set to `false` before successful unfreeze to prevent
    /// rollback on normal drop.
    fn set_active(&mut self, active: bool);
}

// ============================================================================
//  TreeInternode Trait
// ============================================================================

/// Trait for internode types used in a `MassTree`.
///
/// Abstracts over `InternodeNode<S, WIDTH>` for different WIDTH values,
/// enabling generic tree operations.
///
/// # Type Parameters
///
/// - `S`: The slot type implementing [`ValueSlot`]
///
/// # Implementors
///
/// - `InternodeNode<S, WIDTH>` for WIDTH in 1..=15
/// - `InternodeNode<S, 24>` for WIDTH=24
pub trait TreeInternode<S: ValueSlot>: Sized + Send + Sync + 'static {
    /// Node width (max number of children).
    const WIDTH: usize;

    // ========================================================================
    //  Construction
    // ========================================================================

    /// Create a new internode with specified height.
    fn new_boxed(height: u32) -> Box<Self>;

    /// Create a new root internode with specified height.
    fn new_root_boxed(height: u32) -> Box<Self>;

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

    /// Set number of keys.
    fn set_nkeys(&self, n: u8);

    /// Increment nkeys by 1.
    fn inc_nkeys(&self);

    /// Check if this internode is full.
    fn is_full(&self) -> bool;

    // ========================================================================
    //  Keys
    // ========================================================================

    /// Get key at index.
    fn ikey(&self, idx: usize) -> u64;

    /// Set key at index.
    fn set_ikey(&self, idx: usize, key: u64);

    /// Compare key at position with search key.
    fn compare_key(&self, search_ikey: u64, p: usize) -> std::cmp::Ordering;

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
    fn split_into(
        &self,
        new_right: &mut Self,
        new_right_ptr: *mut Self,
        insert_pos: usize,
        insert_ikey: u64,
        insert_child: *mut u8,
    ) -> (u64, bool);

    // ========================================================================
    //  Performance
    // ========================================================================

    /// Prefetch the internode's data into cache.
    ///
    /// Brings the node's key and child arrays into CPU cache before they're
    /// accessed, reducing memory latency during traversal.
    fn prefetch(&self);
}

// ============================================================================
//  TreeLeafNode Trait
// ============================================================================

/// Trait for leaf node types that can be used in a [`MassTree`].
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
    type Internode: TreeInternode<S>;

    /// Node width (number of slots).
    const WIDTH: usize;

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

    /// Load raw permutation value with Acquire ordering.
    ///
    /// Used for freeze detection without constructing a Permuter.
    fn permutation_raw(&self) -> <Self::Perm as TreePermutation>::Raw;

    /// Check if permutation is frozen (split in progress).
    ///
    /// Convenience method that checks the raw value.
    #[inline(always)]
    fn is_perm_frozen(&self) -> bool {
        Self::Perm::is_frozen_raw(self.permutation_raw())
    }

    /// Try to load permutation, returning error if frozen.
    ///
    /// Used in CAS insert path to detect ongoing splits.
    ///
    /// # Errors
    /// Fails when trying to load a frozen permutation.
    #[expect(clippy::result_unit_err)]
    fn permutation_try(&self) -> Result<Self::Perm, ()>;

    /// Wait for permutation to unfreeze.
    ///
    /// Spins with progressive backoff until permutation is valid.
    /// May timeout and return empty permutation if stuck too long.
    fn permutation_wait(&self) -> Self::Perm;

    // ========================================================================
    //  Key Operations
    // ========================================================================

    /// Get ikey at physical slot.
    ///
    /// # Panics
    ///
    /// Debug-panics if `slot >= WIDTH`.
    fn ikey(&self, slot: usize) -> u64;

    /// Set ikey at physical slot.
    ///
    /// # Panics
    ///
    /// Debug-panics if `slot >= WIDTH`.
    fn set_ikey(&self, slot: usize, ikey: u64);

    /// Get ikey bound (slot 0's ikey for B-link navigation).
    ///
    /// The `ikey_bound` is the smallest ikey in this leaf and is used
    /// for navigating to the correct sibling during splits.
    fn ikey_bound(&self) -> u64;

    /// Get keylenx at physical slot.
    ///
    /// Values:
    /// - 0-8: inline key length
    /// - 64 (`KSUF_KEYLENX)`: has suffix
    /// - >=128 (LAYER_KEYLENX): is layer pointer
    fn keylenx(&self, slot: usize) -> u8;

    /// Set keylenx at physical slot.
    fn set_keylenx(&self, slot: usize, keylenx: u8);

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
    /// Returns raw pointer to either an Arc<V> (value mode) or
    /// a sublayer root node (layer mode).
    fn leaf_value_ptr(&self, slot: usize) -> *mut u8;

    /// Store value pointer at slot.
    fn set_leaf_value_ptr(&self, slot: usize, ptr: *mut u8);

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

    /// Pre-store slot data for CAS-based insert.
    ///
    /// # Safety
    ///
    /// - `slot` must be in the free region of the current permutation
    /// - No concurrent writer should be modifying this slot
    unsafe fn store_slot_for_cas(&self, slot: usize, ikey: u64, keylenx: u8, value_ptr: *mut u8);

    /// Store key data for a slot after successful CAS claim.
    ///
    /// # Safety
    ///
    /// Caller must have successfully claimed the slot via `cas_slot_value`.
    unsafe fn store_key_data_for_cas(&self, slot: usize, ikey: u64, keylenx: u8);

    /// Clear a slot after failed CAS insert.
    ///
    /// # Safety
    ///
    /// Caller must have already reclaimed/freed the value that was stored.
    unsafe fn clear_slot_for_cas(&self, slot: usize);

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

    /// CAS the permutation from expected to new value.
    ///
    /// The raw permutation value is used for atomic comparison.
    ///
    /// # Errors
    ///
    /// Returns `Err(CasPermutationError)` if:
    /// - The permutation is frozen (split in progress)
    /// - The current permutation value does not match `expected` (concurrent modification)
    ///
    /// # Freeze Safety
    ///
    /// If the permutation is frozen (split in progress), the CAS will fail.
    fn cas_permutation_raw(
        &self,
        expected: Self::Perm,
        new: Self::Perm,
    ) -> Result<(), CasPermutationError<Self::Perm>>;

    // ========================================================================
    //  Split Operations
    // ========================================================================

    /// The freeze guard type for this leaf.
    ///
    /// Used by split operations to atomically freeze the permutation and
    /// capture a snapshot for computing the split.
    type FreezeGuard<'a>: FreezeGuardOps<Self::Perm>
    where
        Self: 'a;

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
    /// Moves entries from `split_pos..size` to `new_leaf`.
    ///
    /// # Returns
    ///
    /// `(new_leaf_box, split_ikey, insert_target)` tuple where:
    /// - `new_leaf_box` is the new right leaf with moved entries
    /// - `split_ikey` is the first key of the new leaf (separator for parent)
    /// - `insert_target` indicates which leaf should receive the new key
    ///
    /// # Safety
    ///
    /// - Caller must hold the leaf lock (if concurrent)
    /// - `new_leaf` must be freshly allocated (empty)
    /// - `guard` must be valid
    unsafe fn split_into_preallocated(
        &self,
        split_pos: usize,
        new_leaf: Box<Self>,
        guard: &seize::LocalGuard<'_>,
    ) -> (Box<Self>, u64, InsertTarget);

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
        new_leaf: Box<Self>,
        guard: &seize::LocalGuard<'_>,
    ) -> (Box<Self>, u64, InsertTarget);

    // ========================================================================
    //  Freeze Operations
    // ========================================================================

    /// Freeze the permutation for split operations.
    ///
    /// Returns a guard that captures the pre-freeze permutation snapshot.
    /// The guard provides panic safety by restoring a valid permutation if
    /// the split is aborted.
    ///
    /// # Preconditions
    ///
    /// - Caller must hold the leaf lock
    /// - Caller must have called `version().mark_split()`
    fn freeze_permutation(&self) -> Self::FreezeGuard<'_>;

    /// Unfreeze the permutation and publish the final split result.
    ///
    /// This consumes the freeze guard and atomically publishes the new
    /// permutation, making the split visible to readers.
    ///
    /// # Arguments
    ///
    /// * `guard` - The freeze guard from `freeze_permutation()`
    /// * `perm` - The new permutation to publish
    fn unfreeze_set_permutation(&self, guard: Self::FreezeGuard<'_>, perm: Self::Perm);

    /// Check if the permutation is currently frozen.
    ///
    /// A frozen permutation indicates a split is in progress.
    fn is_permutation_frozen(&self) -> bool;

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

    /// Assign a suffix to a slot.
    ///
    /// # Safety
    ///
    /// - Caller must hold the leaf lock
    /// - Slot must be valid
    unsafe fn assign_ksuf(&self, slot: usize, suffix: &[u8], guard: &seize::LocalGuard<'_>);

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
    fn ksuf_compare(&self, slot: usize, suffix: &[u8]) -> Option<std::cmp::Ordering>;

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
pub trait LayerCapableLeaf<V: Send + Sync + 'static>: TreeLeafNode<LeafValue<V>> {
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
    fn try_clone_arc(&self, slot: usize) -> Option<Arc<V>>;

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
        value: Option<Arc<V>>,
        guard: &LocalGuard<'_>,
    );
}

// ============================================================================
//  Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::leaf24::LeafNode24;
    use crate::permuter24::Permuter24;
    use crate::value::LeafValue;

    // ========================================================================
    //  TreePermutation Tests
    // ========================================================================

    fn test_permutation_empty<P: TreePermutation>() {
        let p = P::empty();
        assert_eq!(p.size(), 0);
        assert_eq!(p.back(), 0);
    }

    fn test_permutation_insert<P: TreePermutation>() {
        let mut p = P::empty();
        assert_eq!(p.size(), 0);

        let slot = p.insert_from_back(0);
        assert_eq!(slot, 0);
        assert_eq!(p.size(), 1);
        assert_eq!(p.get(0), 0);
    }

    fn test_permutation_insert_immutable<P: TreePermutation>() {
        let p = P::empty();
        let (new_p, slot) = p.insert_from_back_immutable(0);

        // Original unchanged
        assert_eq!(p.size(), 0);

        // New permuter has insert
        assert_eq!(new_p.size(), 1);
        assert_eq!(slot, 0);
        assert_eq!(new_p.get(0), 0);
    }

    fn test_permutation_freeze<P: TreePermutation>() {
        let p = P::empty();
        assert!(!P::is_frozen_raw(p.value()));

        let frozen = P::freeze_raw(p.value());
        assert!(P::is_frozen_raw(frozen));
    }

    fn test_permutation_roundtrip<P: TreePermutation>() {
        let p = P::empty();
        let raw = p.value();
        let p2 = P::from_value(raw);
        assert_eq!(p, p2);
    }

    #[test]
    fn test_permuter24_trait_empty() {
        test_permutation_empty::<Permuter24>();
    }

    #[test]
    fn test_permuter24_trait_insert() {
        test_permutation_insert::<Permuter24>();
    }

    #[test]
    fn test_permuter24_trait_insert_immutable() {
        test_permutation_insert_immutable::<Permuter24>();
    }

    #[test]
    fn test_permuter24_trait_freeze() {
        test_permutation_freeze::<Permuter24>();
    }

    #[test]
    fn test_permuter24_trait_roundtrip() {
        test_permutation_roundtrip::<Permuter24>();
    }

    // ========================================================================
    //  TreeLeafNode Tests
    // ========================================================================

    fn test_leaf_new<L: TreeLeafNode<LeafValue<u64>>>() {
        let leaf: Box<L> = L::new_boxed();
        assert!(leaf.is_empty());
        assert!(!leaf.is_full());
        assert_eq!(leaf.size(), 0);
    }

    fn test_leaf_permutation<L: TreeLeafNode<LeafValue<u64>>>() {
        let leaf: Box<L> = L::new_boxed();
        let perm = leaf.permutation();
        assert_eq!(perm.size(), 0);

        // Insert via permutation
        let mut new_perm = perm;
        let slot = new_perm.insert_from_back(0);
        leaf.set_permutation(new_perm);

        assert_eq!(leaf.size(), 1);
        assert_eq!(leaf.permutation().get(0), slot);
    }

    fn test_leaf_ikey<L: TreeLeafNode<LeafValue<u64>>>() {
        let leaf: Box<L> = L::new_boxed();
        leaf.set_ikey(0, 12345);
        assert_eq!(leaf.ikey(0), 12345);
        assert_eq!(leaf.ikey_bound(), 12345);
    }

    fn test_leaf_keylenx<L: TreeLeafNode<LeafValue<u64>>>() {
        let leaf: Box<L> = L::new_boxed();
        leaf.set_keylenx(1, 8);
        assert_eq!(leaf.keylenx(1), 8);
        assert!(!leaf.is_layer(1));
        assert!(!leaf.has_ksuf(1));

        // Test layer marker
        leaf.set_keylenx(2, 128);
        assert!(leaf.is_layer(2));
    }

    fn test_leaf_linking<L: TreeLeafNode<LeafValue<u64>>>() {
        let leaf1: Box<L> = L::new_boxed();
        let leaf2: Box<L> = L::new_boxed();
        let leaf2_ptr = Box::into_raw(leaf2);

        leaf1.set_next(leaf2_ptr);
        assert_eq!(leaf1.safe_next(), leaf2_ptr);
        assert!(!leaf1.next_is_marked());

        leaf1.mark_next();
        assert!(leaf1.next_is_marked());
        assert_eq!(leaf1.safe_next(), leaf2_ptr);

        leaf1.unmark_next();
        assert!(!leaf1.next_is_marked());

        // Cleanup
        let _ = unsafe { Box::from_raw(leaf2_ptr) };
    }

    fn test_leaf_version<L: TreeLeafNode<LeafValue<u64>>>() {
        let leaf: Box<L> = L::new_boxed();
        let version = leaf.version();

        // Should be unlocked initially
        assert!(!version.is_locked());

        // Can lock (guard unlocks on drop)
        {
            let _guard = version.lock();
            assert!(version.is_locked());
        }
        // Guard dropped, should be unlocked
        assert!(!version.is_locked());
    }

    #[test]
    fn test_leafnode24_trait_new() {
        test_leaf_new::<LeafNode24<LeafValue<u64>>>();
    }

    #[test]
    fn test_leafnode24_trait_permutation() {
        test_leaf_permutation::<LeafNode24<LeafValue<u64>>>();
    }

    #[test]
    fn test_leafnode24_trait_ikey() {
        test_leaf_ikey::<LeafNode24<LeafValue<u64>>>();
    }

    #[test]
    fn test_leafnode24_trait_keylenx() {
        test_leaf_keylenx::<LeafNode24<LeafValue<u64>>>();
    }

    #[test]
    fn test_leafnode24_trait_linking() {
        test_leaf_linking::<LeafNode24<LeafValue<u64>>>();
    }

    #[test]
    fn test_leafnode24_trait_version() {
        test_leaf_version::<LeafNode24<LeafValue<u64>>>();
    }

    // ========================================================================
    //  WIDTH Constant Verification
    // ========================================================================

    #[test]
    fn test_width_constants() {
        // Permutation WIDTH matches leaf WIDTH
        assert_eq!(
            <Permuter24 as TreePermutation>::WIDTH,
            <LeafNode24<LeafValue<u64>> as TreeLeafNode<LeafValue<u64>>>::WIDTH
        );

        // Verify actual values
        assert_eq!(
            <LeafNode24<LeafValue<u64>> as TreeLeafNode<LeafValue<u64>>>::WIDTH,
            24
        );
    }

    // ========================================================================
    //  Generic Function Tests (prove traits enable generic code)
    // ========================================================================

    /// Generic function that works with any permutation type
    fn generic_perm_fill<P: TreePermutation>(count: usize) -> P {
        let mut perm = P::empty();
        for i in 0..count.min(P::WIDTH) {
            perm.insert_from_back(i);
        }
        perm
    }

    /// Generic function that works with any leaf node type
    #[allow(clippy::unnecessary_box_returns)]
    fn generic_leaf_setup<L: TreeLeafNode<LeafValue<u64>>>(ikey: u64) -> Box<L> {
        let leaf = L::new_boxed();
        leaf.set_ikey(0, ikey);
        leaf.set_keylenx(0, 8);

        let mut perm = leaf.permutation();
        perm.insert_from_back(0);
        leaf.set_permutation(perm);

        leaf
    }

    #[test]
    fn test_generic_perm_fill_24() {
        let perm: Permuter24 = generic_perm_fill(10);
        assert_eq!(perm.size(), 10);
    }

    #[test]
    fn test_generic_leaf_setup_24() {
        let leaf: Box<LeafNode24<LeafValue<u64>>> = generic_leaf_setup(42);
        assert_eq!(leaf.ikey(0), 42);
        assert_eq!(leaf.size(), 1);
    }
}
