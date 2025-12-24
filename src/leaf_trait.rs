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

use crate::nodeversion::NodeVersion;
use crate::slot::ValueSlot;

// ============================================================================
// CAS Permutation Error
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
    #[inline]
    pub fn new(current: P) -> Self {
        Self { current }
    }

    /// Check if the failure was due to frozen state (split in progress).
    #[inline]
    pub fn is_frozen(&self) -> bool {
        P::is_frozen_raw(self.current.value())
    }

    /// Get the current permutation value.
    #[inline]
    pub fn current(&self) -> P {
        self.current
    }
}

// ============================================================================
// TreePermutation Trait
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
    // Construction
    // ========================================================================

    /// Create an empty permutation with size = 0.
    ///
    /// Slots are arranged so `back()` returns slot 0 initially.
    fn empty() -> Self;

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
    // Accessors
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
    // Mutation
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
    // Freeze Operations
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
// TreeInternode Trait
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
    // Construction
    // ========================================================================

    /// Create a new internode with specified height.
    fn new_boxed(height: u32) -> Box<Self>;

    /// Create a new root internode with specified height.
    fn new_root_boxed(height: u32) -> Box<Self>;

    // ========================================================================
    // Version / Locking
    // ========================================================================

    /// Get reference to node version.
    fn version(&self) -> &NodeVersion;

    // ========================================================================
    // Structure
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
    // Keys
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
    // Split Support
    // ========================================================================

    /// Shift entries from another internode.
    fn shift_from(&self, dst_pos: usize, src: &Self, src_pos: usize, count: usize);

    /// Split this internode into a new sibling while inserting a key/child.
    ///
    /// # Arguments
    ///
    /// * `new_right` - The new right sibling (pre-allocated)
    /// * `insert_pos` - Position where the new key/child should be inserted
    /// * `insert_ikey` - The key to insert
    /// * `insert_child` - The child pointer to insert
    ///
    /// # Returns
    ///
    /// `(popup_key, insert_went_left)` where:
    /// - `popup_key` is the key that goes to the parent
    /// - `insert_went_left` is true if the insert went to the left sibling
    fn split_into(
        &self,
        new_right: &mut Self,
        insert_pos: usize,
        insert_ikey: u64,
        insert_child: *mut u8,
    ) -> (u64, bool);
}

// ============================================================================
// TreeLeafNode Trait
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
    // Construction
    // ========================================================================

    /// Create a new leaf node (heap-allocated).
    fn new_boxed() -> Box<Self>;

    /// Create a new root leaf node (heap-allocated).
    fn new_root_boxed() -> Box<Self>;

    // ========================================================================
    // NodeVersion Operations
    // ========================================================================

    /// Get a reference to the node's version.
    ///
    /// Used for optimistic concurrency control (OCC) and locking.
    fn version(&self) -> &NodeVersion;

    // ========================================================================
    // Permutation Operations
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
    #[inline]
    fn is_perm_frozen(&self) -> bool {
        Self::Perm::is_frozen_raw(self.permutation_raw())
    }

    /// Try to load permutation, returning error if frozen.
    ///
    /// Used in CAS insert path to detect ongoing splits.
    ///
    /// # Errors
    /// Fails when trying to load a frozen permutation.
    fn permutation_try(&self) -> Result<Self::Perm, ()>;

    /// Wait for permutation to unfreeze.
    ///
    /// Spins with progressive backoff until permutation is valid.
    /// May timeout and return empty permutation if stuck too long.
    fn permutation_wait(&self) -> Self::Perm;

    // ========================================================================
    // Key Operations
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
    /// The ikey_bound is the smallest ikey in this leaf and is used
    /// for navigating to the correct sibling during splits.
    fn ikey_bound(&self) -> u64;

    /// Get keylenx at physical slot.
    ///
    /// Values:
    /// - 0-8: inline key length
    /// - 64 (KSUF_KEYLENX): has suffix
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
    // Value Operations
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
    /// # Returns
    ///
    /// - `Ok(())` if CAS succeeded
    /// - `Err(actual)` with the actual value if CAS failed
    fn cas_slot_value(
        &self,
        slot: usize,
        expected: *mut u8,
        new_value: *mut u8,
    ) -> Result<(), *mut u8>;

    // ========================================================================
    // Size Operations
    // ========================================================================

    /// Get number of keys in this leaf.
    #[inline]
    fn size(&self) -> usize {
        self.permutation().size()
    }

    /// Check if leaf is empty.
    #[inline]
    fn is_empty(&self) -> bool {
        self.size() == 0
    }

    /// Check if leaf is full.
    #[inline]
    fn is_full(&self) -> bool {
        self.size() >= Self::WIDTH
    }

    // ========================================================================
    // Navigation (B-link tree pointers)
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
    // Slot Assignment Helpers
    // ========================================================================

    /// Check if slot 0 can be reused for a new key.
    ///
    /// Slot 0 stores ikey_bound() which must be preserved if this
    /// leaf has a predecessor (prev != null). Slot 0 can only be
    /// reused if the new key has the same ikey as the current bound.
    fn can_reuse_slot0(&self, new_ikey: u64) -> bool;

    // ========================================================================
    // CAS Insert Support
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
    /// Returns `Ok(())` on success, `Err` with the failure info on failure.
    /// The raw permutation value is used for atomic comparison.
    ///
    /// # Freeze Safety
    ///
    /// If the permutation is frozen (split in progress), the CAS will fail.
    fn cas_permutation_raw(
        &self,
        expected: Self::Perm,
        new: Self::Perm,
    ) -> Result<(), CasPermutationError<Self::Perm>>;
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::leaf::{LeafNode, LeafValue};
    use crate::leaf24::LeafNode24;
    use crate::permuter::Permuter;
    use crate::permuter24::Permuter24;

    // ========================================================================
    // TreePermutation Tests
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
    fn test_permuter15_trait_empty() {
        test_permutation_empty::<Permuter<15>>();
    }

    #[test]
    fn test_permuter15_trait_insert() {
        test_permutation_insert::<Permuter<15>>();
    }

    #[test]
    fn test_permuter15_trait_insert_immutable() {
        test_permutation_insert_immutable::<Permuter<15>>();
    }

    #[test]
    fn test_permuter15_trait_freeze() {
        test_permutation_freeze::<Permuter<15>>();
    }

    #[test]
    fn test_permuter15_trait_roundtrip() {
        test_permutation_roundtrip::<Permuter<15>>();
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
    // TreeLeafNode Tests
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
    fn test_leafnode15_trait_new() {
        test_leaf_new::<LeafNode<LeafValue<u64>, 15>>();
    }

    #[test]
    fn test_leafnode15_trait_permutation() {
        test_leaf_permutation::<LeafNode<LeafValue<u64>, 15>>();
    }

    #[test]
    fn test_leafnode15_trait_ikey() {
        test_leaf_ikey::<LeafNode<LeafValue<u64>, 15>>();
    }

    #[test]
    fn test_leafnode15_trait_keylenx() {
        test_leaf_keylenx::<LeafNode<LeafValue<u64>, 15>>();
    }

    #[test]
    fn test_leafnode15_trait_linking() {
        test_leaf_linking::<LeafNode<LeafValue<u64>, 15>>();
    }

    #[test]
    fn test_leafnode15_trait_version() {
        test_leaf_version::<LeafNode<LeafValue<u64>, 15>>();
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
    // WIDTH Constant Verification
    // ========================================================================

    #[test]
    fn test_width_constants() {
        // Permutation WIDTH matches leaf WIDTH
        assert_eq!(
            <Permuter<15> as TreePermutation>::WIDTH,
            <LeafNode<LeafValue<u64>, 15> as TreeLeafNode<LeafValue<u64>>>::WIDTH
        );
        assert_eq!(
            <Permuter24 as TreePermutation>::WIDTH,
            <LeafNode24<LeafValue<u64>> as TreeLeafNode<LeafValue<u64>>>::WIDTH
        );

        // Verify actual values
        assert_eq!(
            <LeafNode<LeafValue<u64>, 15> as TreeLeafNode<LeafValue<u64>>>::WIDTH,
            15
        );
        assert_eq!(
            <LeafNode24<LeafValue<u64>> as TreeLeafNode<LeafValue<u64>>>::WIDTH,
            24
        );
    }

    // ========================================================================
    // Generic Function Tests (prove traits enable generic code)
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
    fn test_generic_perm_fill_15() {
        let perm: Permuter<15> = generic_perm_fill(5);
        assert_eq!(perm.size(), 5);
    }

    #[test]
    fn test_generic_perm_fill_24() {
        let perm: Permuter24 = generic_perm_fill(10);
        assert_eq!(perm.size(), 10);
    }

    #[test]
    fn test_generic_leaf_setup_15() {
        let leaf: Box<LeafNode<LeafValue<u64>, 15>> = generic_leaf_setup(42);
        assert_eq!(leaf.ikey(0), 42);
        assert_eq!(leaf.size(), 1);
    }

    #[test]
    fn test_generic_leaf_setup_24() {
        let leaf: Box<LeafNode24<LeafValue<u64>>> = generic_leaf_setup(42);
        assert_eq!(leaf.ikey(0), 42);
        assert_eq!(leaf.size(), 1);
    }
}
