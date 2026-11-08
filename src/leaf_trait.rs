//! Permutation and internode contracts, plus leaf split data types.

use std::cmp::Ordering;
use std::fmt::Debug;

use crate::nodeversion::NodeVersion;
use crate::policy::LeafPolicy;

// ============================================================================
//  Split Types
// ============================================================================

/// Split point for leaf node splitting.
#[derive(Debug, Clone, Copy)]
pub struct SplitPoint {
    /// Logical position where to split (in post-insert coordinates).
    pub pos: usize,

    /// The ikey that will be the first key of the new (right) leaf.
    pub split_ikey: u64,
}

/// Which leaf to insert into after a split.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InsertTarget {
    /// Insert into the original (left) leaf.
    Left,

    /// Insert into the new (right) leaf.
    Right,
}

// ============================================================================
// Split+Insert Types (Atomic Split+Insert Operation)
// ============================================================================

/// Data for inserting a key during a split operation.
#[derive(Debug)]
pub struct SplitInsertData<'a, P: LeafPolicy> {
    /// The 8-byte key to insert.
    pub ikey: u64,

    /// The keylenx value (0-8 for inline, 64 for suffix, >=128 for layer).
    pub keylenx: u8,

    /// The suffix bytes, if any (present when `keylenx == KSUF_KEYLENX`).
    pub suffix: Option<&'a [u8]>,

    /// The typed value to insert.
    pub value: P::Output,
}

/// Result of an atomic split+insert operation.
#[derive(Debug, Clone, Copy)]
pub struct SplitInsertResult {
    /// The key that separates the left and right leaves.
    pub split_ikey: u64,

    /// Which leaf received the new key.
    pub insert_target: InsertTarget,
}

// ============================================================================
//  TreePermutation Trait
// ============================================================================

/// Trait for permutation types used in leaf nodes.
pub trait TreePermutation: Copy + Clone + Eq + Debug + Send + Sync + Sized + 'static {
    /// Raw storage type for atomic operations.
    type Raw: Copy + Clone + Eq + Debug + Send + Sync + 'static;

    /// Number of slots this permutation supports.
    const WIDTH: usize;

    // ========================================================================
    //  Construction
    // ========================================================================

    /// Create an empty permutation with size = 0.
    fn empty() -> Self;

    /// Create a sorted permutation with `n` elements in slots `0..n`.
    fn make_sorted(n: usize) -> Self;

    /// Create a permutation from a raw storage value.
    fn from_value(raw: Self::Raw) -> Self;

    // ========================================================================
    //  Accessors
    // ========================================================================

    /// Get the raw storage value.
    fn value(&self) -> Self::Raw;

    /// Get the number of slots in use.
    fn size(&self) -> usize;

    /// Get the physical slot at logical position `i`.
    fn get(&self, i: usize) -> usize;

    /// Get the slot at the back (next free slot to allocate).
    fn back(&self) -> usize;

    /// Get the slot at `back()` with an offset into the free region.
    fn back_at_offset(&self, offset: usize) -> usize;

    // ========================================================================
    //  Mutation
    // ========================================================================

    /// Allocate a slot from back and insert at position `i`.
    fn insert_from_back(&mut self, i: usize) -> usize;

    /// Compute insert result without mutation (for CAS operations).
    fn insert_from_back_immutable(&self, i: usize) -> (Self, usize);

    /// Swap two slots in the free region (positions >= size).
    fn swap_free_slots(&mut self, pos_i: usize, pos_j: usize);

    /// Set the size without changing slot positions.
    fn set_size(&mut self, n: usize);

    /// Remove the slot at logical position `i`.
    fn remove(&mut self, i: usize);
}

// ============================================================================
//  TreeInternode Trait
// ============================================================================

/// Trait for internode types used in a `MassTree`.
pub trait TreeInternode: Sized + Send + Sync + 'static {
    /// Node width (max number of children).
    const WIDTH: usize;

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
    fn ikey_relaxed(&self, idx: usize) -> u64;

    /// Get raw pointer to the ikey array for SIMD operations.
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

    /// Split this internode into a new sibling while inserting a key/child.
    ///
    /// # Safety
    ///
    /// Same contract as `InternodeNode::split_into`: the node must be locked,
    /// the sibling exclusively owned, and moved children valid.
    #[must_use = "popup_key must be inserted into parent node to complete the split"]
    unsafe fn split_into(
        &self,
        new_right: &mut Self,
        new_right_ptr: *mut Self,
        insert_pos: usize,
        insert_ikey: u64,
        insert_child: *mut u8,
    ) -> (u64, bool);
}

// ============================================================================
//  Tests
// ============================================================================

#[cfg(test)]
mod unit_tests;
