//! Filepath: src/leaf.rs
//!
//! Leaf node for [`MassTree`].
//!
//! Leaf nodes store the actual key-value pairs, using a permutation array
//! for logical ordering without data movement.

use std::ptr as StdPtr;
use std::sync::Arc;

use crate::{
    nodeversion::NodeVersion,
    permuter::{Permuter, SuffixStorage},
};

/// Special keylenx value indicating key has a suffix.
pub const KSUF_KEYLENX: u8 = 64;

/// Base keylenx value indicating a layer pointer (>= this means layer).
pub const LAYER_KEYLENX: u8 = 128;

/// Modification state values.
#[repr(u8)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ModState {
    /// Node is in insert mode (normal operation).
    Insert = 0,

    /// Node is being removed.
    Remove = 1,

    /// Node's layer has been deleted.
    DeletedLayer = 2,
}

/// Value stored in a leaf slot (default mode with `Arc<V>`).
///
/// Each slot can contain either an Arc-wrapped value or a pointer to a next-layer
/// subtree (for keys longer than 8 bytes at this level).
///
/// # Type Parameter
/// * `V` - The value type stored in the tree (stored as `Arc<V>`)
///
/// # Why `Arc<V>`?
/// Optimistic reads need to return owned values safely.
/// `Arc<V>` allows cheap cloning (refcount) and decouples value
/// lifetime from node lifetime (EBR handles node, `Arc` handles values).
/// More details and rationale can be found in README.md
#[derive(Debug, Default)]
pub enum LeafValue<V> {
    /// Slot is empty (no key assigned).
    #[default]
    Empty,

    /// Slot contains an Arc-wrapped value.
    /// Cloning is cheap (refcount increment), enabling lock-free reads.
    Value(Arc<V>),

    /// Slot contains a pointer to a next-layer subtree.
    /// The pointer is to a `LeafNode` that serves as the root of the sublayer.
    ///
    /// this will be refined to a proper node type.
    Layer(*mut u8),
}

impl<V> LeafValue<V> {
    // ============================================================================
    //  Constructor Methods
    // ============================================================================

    /// Create an empty leaf value.
    #[inline]
    #[must_use]
    pub const fn empty() -> Self {
        Self::Empty
    }

    // ============================================================================
    //  Boolean Methods
    // ============================================================================

    /// Check if this slot is empty.
    #[inline]
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        matches!(self, Self::Empty)
    }

    /// Check if this slot contains a value.
    #[inline]
    #[must_use]
    pub const fn is_value(&self) -> bool {
        matches!(self, Self::Value(_))
    }

    /// Check if this slot contains a layer pointer.
    #[inline]
    #[must_use]
    pub const fn is_layer(&self) -> bool {
        matches!(self, Self::Layer(_))
    }

    // ============================================================================
    //  Safe Try-None Methods
    // ============================================================================

    /// Try to get the Arc value, returning None if not a value variant.
    #[inline]
    #[must_use]
    pub const fn try_as_value(&self) -> Option<&Arc<V>> {
        match self {
            Self::Value(arc) => Some(arc),

            _ => None,
        }
    }

    /// Try to clone the Arc value, returning None if not a value variant.
    #[inline]
    #[must_use]
    pub fn try_clone_arc(&self) -> Option<Arc<V>> {
        match self {
            Self::Value(arc) => Some(Arc::clone(arc)),

            _ => None,
        }
    }

    /// Try to get the layer pointer, returning None if not a Layer variant.
    #[inline]
    #[must_use]
    pub const fn try_as_layer(&self) -> Option<*mut u8> {
        match self {
            Self::Layer(ptr) => Some(*ptr),

            _ => None,
        }
    }

    // ============================================================================
    //  Probable panicky but ergonomic internal use methods
    // ============================================================================

    /// Get the layer pointer.
    ///
    /// WARN: Prefer `try_as_layer()`. This method is
    /// provided for internal use where the variant is known.
    ///
    /// # Panics
    /// Panics if this is not a Layer variant.
    #[inline]
    #[must_use]
    #[expect(clippy::expect_used, reason = "Invariant ensured by caller")]
    pub const fn as_layer(&self) -> *mut u8 {
        self.try_as_layer()
            .expect("LeafValue::as_layer called on non-Layer variant")
    }

    /// Gets a reference to the Arc-wrapped value.
    ///
    ///  WARN: Prefer `try_as_value()`. This was added for internal use where the invariant is known.
    ///
    /// # Panics
    /// Panics if ths is not a Value variant.
    #[inline]
    #[must_use]
    #[expect(clippy::expect_used, reason = "Invariant ensured by caller")]
    pub const fn as_value(&self) -> &Arc<V> {
        self.try_as_value()
            .expect("LeafValue::as_value called on non-Value variant")
    }

    /// Clone the Arc<V> (cheap reference counting increment)
    ///
    ///  WARN: Prefer `try_as_value()`. This was added for internal use where the invariant is known.
    ///
    /// # Panics
    /// Panics if ths is not a Value variant.
    #[inline]
    #[must_use]
    #[expect(clippy::expect_used, reason = "Invariant ensured by caller")]
    pub fn clone_arc(&self) -> Arc<V> {
        self.try_clone_arc()
            .expect("LeafValue::as_value called on non-Value variant")
    }

    /// Get a mutable reference to the Arc-wrapped value.
    ///
    ///  WARN: Prefer checking `is_value()` first if using without invariant checks.
    ///
    /// # Panics
    /// Panics if ths is not a Value variant.
    #[inline]
    #[expect(clippy::panic, reason = "Invariant ensured by caller")]
    pub fn as_value_mut(&mut self) -> &mut Arc<V> {
        match self {
            Self::Value(arc) => arc,

            _ => panic!("LeafValue::as_value_mut called on non-value variant"),
        }
    }
}

impl<V> Clone for LeafValue<V> {
    fn clone(&self) -> Self {
        match self {
            Self::Empty => Self::Empty,

            // This adds some overhead already compared to the raw ptr
            // handling of C++ impl, but it currently seems like the
            // right design decision. The trade off seems justified.
            Self::Value(arc) => Self::Value(Arc::clone(arc)),

            Self::Layer(ptr) => Self::Layer(*ptr),
        }
    }
}

/// Value stored in a leaf slot (index mode with inline `V: Copy`).
///
/// For high-perf index use cases values are small and Copy.
///
/// # Type Parameter
/// `V` - The value type stored inline (must be `Copy`)
#[derive(Clone, Copy, Debug, Default)]
pub enum LeafValueIndex<V: Copy> {
    /// Slot is empty (no key assigned).
    #[default]
    Empty,

    /// Slot contains an inline value (copied on read).
    Value(V),

    /// Slot contains a pointer to a next-layer subtree.
    ///
    /// This will be refined to proper node type.
    Layer(*mut u8),
}

impl<V: Copy> LeafValueIndex<V> {
    // ============================================================================
    //  Constructor Methods
    // ============================================================================

    /// Create an empty leaf value.
    #[inline]
    #[must_use]
    pub const fn empty() -> Self {
        Self::Empty
    }

    // ============================================================================
    //  Boolean Methods
    // ============================================================================

    /// Check if this slot is empty.
    #[inline]
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        matches!(self, Self::Empty)
    }

    /// Check if this slot contains a value.
    #[inline]
    #[must_use]
    pub const fn is_value(&self) -> bool {
        matches!(self, Self::Value(_))
    }

    /// Check if this slot contains a layer pointer.
    #[inline]
    #[must_use]
    pub const fn is_layer(&self) -> bool {
        matches!(self, Self::Layer(_))
    }

    // ============================================================================
    //  Safe Try-None Methods
    // ============================================================================

    /// Try to get the value, returning None if not a Value variant.
    #[inline]
    #[must_use]
    pub const fn try_value(&self) -> Option<V> {
        match self {
            Self::Value(v) => Some(*v),

            _ => None,
        }
    }

    /// Try to get the layer pointer, returning None if not a Layer variant.
    #[inline]
    #[must_use]
    pub const fn try_as_layer(&self) -> Option<*mut u8> {
        match self {
            Self::Layer(ptr) => Some(*ptr),

            _ => None,
        }
    }

    // ============================================================================
    //  Probable panicky but ergonomic internal use methods
    // ============================================================================

    /// Get the value (copied).
    ///
    /// WARN: Prefer `try_value()`. This method is
    /// provided for internal use where the variant is known.
    ///
    /// # Panics
    /// Panics if this is not a Value variant.
    #[inline]
    #[must_use]
    #[expect(clippy::expect_used, reason = "Invariant ensured by caller")]
    pub const fn value(&self) -> V {
        self.try_value()
            .expect("LeafValueIndex::value called on non-Value variant")
    }

    /// Get the layer pointer.
    ///
    /// WARN: Prefer `try_as_layer()`. This method is
    /// provided for internal use where the variant is known.
    ///
    /// # Panics
    /// Panics if this is not a Layer variant.
    #[expect(clippy::expect_used, reason = "Invariant ensured by caller")]
    #[inline]
    #[must_use]
    pub const fn as_layer(&self) -> *mut u8 {
        self.try_as_layer()
            .expect("LeafValueIndex::as_layer called on non-Layer variant")
    }
}

/// A leaf node in the [`MadTree`].
///
/// Stores up to WIDTH key-value pairs, with keys sorted via a permutation array.
/// Leaves are linked for efficient range scans.
///
/// # Type Params
/// * `V` - The value type stored in the tree
/// * `WIDTH` - Number of slots (default: 15, max: 15 for u64 permuter)
///
/// # Invariants
/// - For each slot `s` in `0..permutation.size()`:
///   - `keylenx[s]` describes the key at that slot
///   - `ikey0[s]` contains the 8-byte ikey
///   - `lv[s]` contains the value or layer pointer
/// - Slots are in sorted order by logical position (via permutation)
///
/// # Memory Layout
/// The struct uses `#[repr(C)]` for predictable layout and `align(64)` for
/// cache-line alignment. For WIDTH=15 with V=u64, total size is ~448 bytes (7 cache lines).
#[repr(C, align(64))]
#[derive(Debug)]
pub struct LeafNode<V, const WIDTH: usize = 15> {
    /// Version for optimistic concurrency control.
    version: NodeVersion,

    /// Modification state `(insert, remove, deleted_layer)`.
    modstate: ModState,

    /// Key length/type for each slot.
    /// - 0-8: inline key with that exact length
    /// - 64 (`KSUF_KEYLENX)`: key has suffix
    /// - 128+ (`LAYER_KEYLENX)`: layer pointer
    keylenx: [u8; WIDTH],

    /// Permutation array mapping logical positions to physical slots.
    permutation: Permuter<WIDTH>,

    /// 8-byte keys for each slot (big-endian for lexicographic comparison).
    ikey0: [u64; WIDTH],

    /// Values or layer pointers for each slot.
    leaf_values: [LeafValue<V>; WIDTH],

    /// Pointer to external suffix storage (null if no suffixes or using inline).
    ///
    ///  TODO: Defined in `crate::permuter` as a placeholder,
    ///  this will have to be properly defined and extended in later phases.
    ksuf: *mut SuffixStorage,

    /// Next leaf pointer (LSB used as mark bit during splits).
    next: *mut Self,

    /// Previous leaf pointer.
    prev: *mut Self,

    /// Parent internode pointer.
    parent: *mut u8, // Will be refined to InternodeNode type
}

// Compile-time assertion: WIDTH must be 1..=15
impl<V, const WIDTH: usize> LeafNode<V, WIDTH> {
    const WIDTH_CHECK: () = {
        assert!(WIDTH > 0, "WIDTH must be at least 1");
        assert!(WIDTH <= 15, "WIDTH must be at most 15 (u64 permuter limit)");
    };
}

impl<V, const WIDTH: usize> LeafNode<V, WIDTH> {
    // ============================================================================
    //  Constructor Methods
    // ============================================================================

    /// Create a new leaf node.
    ///
    /// The node is initialized as a leaf + root with empty permutation.
    /// All pointers are set to null.
    ///
    /// # Returns
    /// A boxed leaf node (heap-allocated).
    #[must_use]
    pub fn new() -> Box<Self> {
        // Trigger compile-time WIDTH check
        let _: () = Self::WIDTH_CHECK;

        // SAFETY: We're initializing all fields to valid default values
        Box::new(Self {
            version: NodeVersion::new(true), // true = is_leaf
            modstate: ModState::Insert,
            keylenx: [0; WIDTH],
            permutation: Permuter::empty(),
            ikey0: [0; WIDTH],
            leaf_values: std::array::from_fn(|_| LeafValue::Empty),
            ksuf: StdPtr::null_mut(),
            next: StdPtr::null_mut(),
            prev: StdPtr::null_mut(),
            parent: StdPtr::null_mut(),
        })
    }

    /// Create a new leaf node as the root of a tree/layer.
    ///
    /// Same as `new()` but explicitly marks the node as root.
    #[must_use]
    pub fn new_root() -> Box<Self> {
        let node: Box<Self> = Self::new();
        node.version.mark_root();

        node
    }

    // ============================================================================
    //  NodeVersion Accessors
    // ============================================================================

    /// Get a reference to the node's version.
    #[inline]
    pub const fn version(&self) -> &NodeVersion {
        &self.version
    }

    /// Get a mutable reference to the nodes version.
    #[inline]
    pub const fn version_mut(&mut self) -> &mut NodeVersion {
        &mut self.version
    }

    // ============================================================================
    //  Key Accessors
    // ============================================================================

    /// Get the ikey at the given physical slot.
    ///
    /// # Panics
    /// Panics in debug mode if `slot >= WIDTH`.
    #[inline]
    #[must_use]
    #[expect(
        clippy::indexing_slicing,
        reason = "Slot from Permuter, valid by construction"
    )]
    pub fn ikey(&self, slot: usize) -> u64 {
        debug_assert!(slot < WIDTH, "ikey: slot out of bounds");

        self.ikey0[slot]
    }

    /// Get the keylenx at the given physical slot.
    ///
    /// # Panics
    /// Panics in debug mode if `slot >= WIDTH`.
    #[inline]
    #[must_use]
    #[expect(
        clippy::indexing_slicing,
        reason = "Slot from Permuter, valid by construction"
    )]
    pub fn keylenx(&self, slot: usize) -> u8 {
        debug_assert!(slot < WIDTH, "keylenx: slot out of bounds");

        self.keylenx[slot]
    }

    /// Get the ikey bound (ikey at slot 0, used for B-link tree routing).
    ///
    /// This is the smallest key that could be in this leaf (after splits).
    #[inline]
    #[must_use]
    pub const fn ikey_bound(&self) -> u64 {
        self.ikey0[0]
    }

    /// Check if the given slot contains a layer pointer.
    ///
    /// # Panics
    /// Panics in debug mode if `slot >= WIDTH`.
    #[inline]
    #[must_use]
    pub fn is_layer(&self, slot: usize) -> bool {
        self.keylenx(slot) >= LAYER_KEYLENX
    }

    /// Check if the given slot has a suffix.
    ///
    /// # Panics
    /// Panics in debug mode if `slot >= WIDTH`.
    #[inline]
    #[must_use]
    pub fn has_ksuf(&self, slot: usize) -> bool {
        self.keylenx(slot) == KSUF_KEYLENX
    }

    /// Check if keylenx indicates a layer pointer (static helper).
    #[inline]
    #[must_use]
    pub const fn keylenx_is_layer(keylenx: u8) -> bool {
        keylenx >= LAYER_KEYLENX
    }

    /// Check if keylenx indicates suffix storage (static helper).
    #[inline]
    #[must_use]
    pub const fn keylenx_has_ksuf(keylenx: u8) -> bool {
        keylenx == KSUF_KEYLENX
    }

    // ============================================================================
    //  Value Accessors
    // ============================================================================

    /// Get a reference to the value at the given physical slot.
    ///
    /// # Panics
    /// Panics in debug mode if `slot >= WIDTH`.
    #[inline]
    #[expect(
        clippy::indexing_slicing,
        reason = "Slot from Permuter; valid by construction"
    )]
    pub fn leaf_value(&self, slot: usize) -> &LeafValue<V> {
        debug_assert!(slot < WIDTH, "leaf_value: slot out of bounds");

        &self.leaf_values[slot]
    }

    // ============================================================================
    //  Permutation Accessors
    // ============================================================================

    /// Get the current permutation.
    #[inline]
    #[must_use]
    pub const fn permutation(&self) -> Permuter<WIDTH> {
        self.permutation
    }

    /// Set the permutation.
    #[inline]
    pub const fn set_permutation(&mut self, perm: Permuter<WIDTH>) {
        self.permutation = perm;
    }

    /// Get the number of keys in this leaf.
    #[inline]
    #[must_use]
    pub const fn size(&self) -> usize {
        self.permutation.size()
    }

    /// Check if the leaf is empty.
    #[inline]
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.size() == 0
    }

    /// Check if the leaf is full.
    #[inline]
    #[must_use]
    pub const fn is_full(&self) -> bool {
        self.size() >= WIDTH
    }

    // ============================================================================
    //  Leaf Linking
    // ============================================================================

    // NOTE: The next pointer uses LSB tagging (mark bit during splits).
    // We use `ptr::map_addr` (stable in Rust 1.84+) to preserve pointer provenance
    // when manipulating the address. This is important for Miri and strict provenance.
    // See: https://doc.rust-lang.org/std/ptr/index.html#provenance

    /// Get the next leaf pointer, masking the mark bit.
    ///
    /// The LSB of `next` is used as a mark during splits. This function
    /// returns the actual pointer with the mark bit cleared.
    #[inline]
    #[must_use]
    pub fn safe_next(&self) -> *mut Self {
        // Use map_addr to preserve provenance while clearing the mark bit
        self.next.map_addr(|addr: usize| addr & !1)
    }

    /// Get the raw next pointer (including mark bit).
    #[inline]
    #[must_use]
    pub const fn next_raw(&self) -> *mut Self {
        self.next
    }

    /// Check if the next pointer is marked (split in progress).
    #[inline]
    #[must_use]
    pub fn next_is_marked(&self) -> bool {
        // addr() extracts address without exposing provenance
        (self.next.addr() & 1) != 0
    }

    /// Set the next leaf pointer.
    #[inline]
    pub const fn set_next(&mut self, next: *mut Self) {
        self.next = next;
    }

    /// Mark the next pointer (during split).
    #[inline]
    pub fn mark_next(&mut self) {
        // Use map_addr to preserve provenance while setting the mark bit
        self.next = self.next.map_addr(|addr: usize| addr | 1);
    }

    /// Unmark the next pointer.
    #[inline]
    pub fn unmark_next(&mut self) {
        self.next = self.safe_next();
    }

    /// Get the previous leaf pointer.
    #[inline]
    #[must_use]
    pub const fn prev(&self) -> *mut Self {
        self.prev
    }

    /// Set the previous leaf pointer.
    #[inline]
    pub const fn set_prev(&mut self, prev: *mut Self) {
        self.prev = prev;
    }

    /// Link two leaves after a split (single-threaded version).
    ///
    ///  FIX: This is the single-threaded implementation.
    ///  For concurrent version, this will be replaced with atomic CAS
    ///  operations.
    ///
    /// After splitting `left` into `left` and `right`:
    /// - `right.next` = old `left.next`
    /// - `right.prev` = `left`
    /// - `left.next` = `right`
    /// - `old_next.prev` = `right` (if `old_next` exists)
    ///
    /// # Safety
    /// Caller must ensure:
    /// 1. `old_next` is a valid, non-null pointer to a live `LeafNode<V, WIDTH>`
    /// 2. No concurrent access to any of the nodes (single-threaded guarantee)
    /// 3. Caller holds mutable references to both `left` and `right`
    /// 4. `old_next` is not being deallocated
    pub fn link_split(left: &mut Self, right: &mut Self) {
        let old_next = left.safe_next();

        right.next = old_next;
        right.prev = StdPtr::from_mut::<Self>(left);
        left.next = StdPtr::from_mut::<Self>(right);

        if !old_next.is_null() {
            //  SAFETY:
            //  1. old_next is from left.safe_next(), which came from a valid ptr
            //  2. single-threaded exec ensures no concurrent mod
            //  3. old_next is not being deallocated (caller's responsibility)
            unsafe {
                (*old_next).prev = StdPtr::from_mut::<Self>(right);
            }
        }
    }

    // ============================================================================
    //  Parent Accessors
    // ============================================================================

    // ============================================================================
    //  ModState Accessors
    // ============================================================================

    // ============================================================================
    //  Slot Assignment
    // ============================================================================

    // ============================================================================
    //  Invariant Checker
    // ============================================================================
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_leaf_value_empty() {
        let lv: LeafValue<u64> = LeafValue::empty();
        assert!(lv.is_empty());
        assert!(!lv.is_value());
        assert!(!lv.is_layer());
    }

    #[test]
    fn test_leaf_value_value() {
        let lv: LeafValue<u64> = LeafValue::Value(Arc::new(42));
        assert!(!lv.is_empty());
        assert!(lv.is_value());
        assert!(!lv.is_layer());
        assert_eq!(**lv.as_value(), 42); // Double deref: &Arc<u64> -> u64

        // Test clone_arc
        let cloned: Arc<u64> = lv.clone_arc();

        assert_eq!(*cloned, 42);
        assert_eq!(Arc::strong_count(&cloned), 2); // Original + clone
    }

    #[test]
    fn test_leaf_value_layer() {
        let ptr: *mut u8 = 0xDEAD_BEEF as *mut u8;
        let lv: LeafValue<u64> = LeafValue::Layer(ptr);

        assert!(!lv.is_empty());
        assert!(!lv.is_value());
        assert!(lv.is_layer());
        assert_eq!(lv.as_layer(), ptr);
    }

    #[test]
    fn test_leaf_node_linking() {
        let mut left: Box<LeafNode<u64, 15>> = LeafNode::new();
        let mut right: Box<LeafNode<u64, 15>> = LeafNode::new();

        // Get raw pointers before linking
        let left_ptr: *mut LeafNode<u64> = left.as_mut() as *mut LeafNode<u64, 15>;
        let right_ptr: *mut LeafNode<u64> = right.as_mut() as *mut LeafNode<u64, 15>;

        // Link them
        LeafNode::link_split(&mut left, &mut right);

        assert_eq!(left.safe_next(), right_ptr);
        assert_eq!(right.prev(), left_ptr);
        assert!(right.safe_next().is_null());
    }
}
