//! Filepath: src/leaf.rs
//!
//! Leaf node for [`MassTree`].
//!
//! Leaf nodes store the actual key-value pairs, using a permutation array
//! for logical ordering without data movement.

use std::array as StdArray;
use std::mem as StdMem;
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

// ============================================================================
//  Split Types
// ============================================================================

/// Result of calculating a split point.
///
/// Contains the logical position where to split and the key that will
/// become the first key of the new (right) leaf.
#[derive(Debug, Clone, Copy)]
pub struct SplitPoint {
    /// Logical position where to split (in post-insert coordinates).
    /// Entries from `pos` to end (in post-insert order) go to new leaf.
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

/// Result of a leaf split operation.
///
/// Contains the new leaf and information about where to insert.
#[derive(Debug)]
pub struct LeafSplitResult<V, const WIDTH: usize = 15> {
    /// The new leaf (right sibling).
    pub new_leaf: Box<LeafNode<V, WIDTH>>,

    /// The split key (first key of new leaf).
    pub split_ikey: u64,

    /// Which leaf the new key should go into.
    pub insert_into: InsertTarget,
}

// ============================================================================
//  Split Point Calculation Functions
// ============================================================================

/// Get the ikey at logical position `i` after the new key is inserted.
///
/// This is critical for correct split point adjustment: we must consider
/// the post-insert key stream, not just existing keys.
///
/// # Arguments
///
/// * `leaf` - The leaf being split
/// * `i` - Logical position in post-insert order
/// * `insert_pos` - Where the new key will be inserted
/// * `insert_ikey` - The ikey of the new key
fn ikey_after_insert<V, const WIDTH: usize>(
    leaf: &LeafNode<V, WIDTH>,
    i: usize,
    insert_pos: usize,
    insert_ikey: u64,
) -> u64 {
    use std::cmp::Ordering;

    let perm = leaf.permutation();

    match i.cmp(&insert_pos) {
        Ordering::Less => {
            // Before insert point: use existing key at position i
            leaf.ikey(perm.get(i))
        }

        Ordering::Equal => {
            // At insert point: this is the new key
            insert_ikey
        }

        Ordering::Greater => {
            // After insert point: shifted by 1, so use position i-1
            leaf.ikey(perm.get(i - 1))
        }
    }
}

/// Adjust split position to keep equal ikey0 values together.
///
/// Uses post-insert key stream to correctly handle the case where the
/// inserted key affects which keys should stay together.
///
/// Returns None if all entries have the same ikey (layer case).
fn adjust_split_for_equal_ikeys_post_insert<V, const WIDTH: usize>(
    leaf: &LeafNode<V, WIDTH>,
    mut pos: usize,
    insert_pos: usize,
    insert_ikey: u64,
    post_insert_size: usize,
) -> Option<usize> {
    if post_insert_size <= 1 {
        return Some(pos);
    }

    // Get ikey at current split position (post-insert)
    let split_ikey: u64 = ikey_after_insert(leaf, pos, insert_pos, insert_ikey);

    // Move left while previous entry has same ikey
    while pos > 0 {
        let prev_ikey: u64 = ikey_after_insert(leaf, pos - 1, insert_pos, insert_ikey);

        if prev_ikey != split_ikey {
            break;
        }

        pos -= 1;
    }

    // If we moved to position 0, all entries might have same ikey
    // Try moving right instead
    if pos == 0 {
        let first_ikey: u64 = ikey_after_insert(leaf, 0, insert_pos, insert_ikey);
        pos = 1;

        while pos < post_insert_size {
            let curr_ikey: u64 = ikey_after_insert(leaf, pos, insert_pos, insert_ikey);

            if curr_ikey != first_ikey {
                break;
            }

            pos += 1;
        }

        // If we reached the end, all entries have same ikey
        if pos >= post_insert_size {
            return None; // Layer case - can't split
        }
    }

    Some(pos)
}

/// Calculate the split point for a leaf node.
///
/// Returns the logical position where to split. Entries from `pos` to `size`
/// (in post-insert coordinates) will move to the new leaf.
///
/// # Arguments
///
/// * `leaf` - The leaf node to split
/// * `insert_pos` - Where the new key would be inserted
/// * `insert_ikey` - The ikey of the new key
///
/// # Returns
///
/// `SplitPoint` with position and split key, or None if split is not possible
/// (e.g., all entries have same ikey - would need layer instead).
pub fn calculate_split_point<V, const WIDTH: usize>(
    leaf: &LeafNode<V, WIDTH>,
    insert_pos: usize,
    insert_ikey: u64,
) -> Option<SplitPoint> {
    let perm: Permuter<WIDTH> = leaf.permutation();
    let size: usize = perm.size();

    if size == 0 {
        return None; // Can't split empty leaf
    }

    // Post-insert size is size + 1
    let post_insert_size: usize = size + 1;

    // Default: split in the middle (of post-insert size)
    let mut split_pos: usize = post_insert_size.div_ceil(2);

    // Sequential optimization heuristics
    let is_rightmost: bool = leaf.safe_next().is_null();
    let is_leftmost: bool = leaf.prev().is_null();
    let inserting_at_end: bool = insert_pos >= size;
    let inserting_at_start: bool = insert_pos == 0;

    if is_rightmost && inserting_at_end {
        // Right-sequential: keep left nearly full
        split_pos = post_insert_size - 1;
    } else if is_leftmost && inserting_at_start {
        // Left-sequential: keep right nearly full
        split_pos = 1;
    }

    // Adjust to keep equal ikey0 values together (using post-insert keys)
    split_pos = adjust_split_for_equal_ikeys_post_insert(
        leaf,
        split_pos,
        insert_pos,
        insert_ikey,
        post_insert_size,
    )?;

    // Get the split key (first key of right half, in post-insert order)
    let split_ikey = ikey_after_insert(leaf, split_pos, insert_pos, insert_ikey);

    Some(SplitPoint {
        pos: split_pos,
        split_ikey,
    })
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

/// A leaf node in the [`MassTree`].
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
        let old_next: *mut Self = left.safe_next();

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

    /// Get the parent pointer.
    #[inline]
    #[must_use]
    pub const fn parent(&self) -> *mut u8 {
        self.parent
    }

    /// Set the parent pointer.
    #[inline]
    pub const fn set_parent(&mut self, parent: *mut u8) {
        self.parent = parent;
    }

    // ============================================================================
    //  ModState Accessors
    // ============================================================================

    /// Get the modification state.
    #[inline]
    #[must_use]
    pub const fn modstate(&self) -> ModState {
        self.modstate
    }

    /// Set the modification state.
    #[inline]
    pub const fn set_modstate(&mut self, state: ModState) {
        self.modstate = state;
    }

    // ============================================================================
    //  Slot Assignment
    // ============================================================================

    /// Assign a key-value pair to a physical slot.
    ///
    /// This sets the ikey, keylenx, and value for the slot. It does not update
    /// the permutation, the caller must do that seperately.
    ///
    /// # Parameters
    /// - `slot`: Physical slot index
    /// - `ikey`: 8-byte key (big-endian)
    /// - `key_len`: Actual key length (0-8)
    /// - `value`: The value to store
    ///
    /// # Panics
    /// Panics in debug mode if `slot >= WIDTH`.
    #[expect(
        clippy::indexing_slicing,
        reason = "Slot from Permuter, valid by construction"
    )]
    pub fn assign(&mut self, slot: usize, ikey: u64, keylenx: u8, value: LeafValue<V>) {
        debug_assert!(slot < WIDTH, "assign: slot out of bounds");

        self.ikey0[slot] = ikey;
        self.keylenx[slot] = keylenx;
        self.leaf_values[slot] = value;
    }

    /// Assign a simple value (no suffix, no layer).
    ///
    /// The value is wrapped in `Arc::new()` for the default mode.
    ///
    /// # Parameters
    /// - `slot`: Physical slot index
    /// - `ikey`: 8-byte key (big-endian)
    /// - `key_len`: Actual key length (0-8)
    /// - `value`: The value to store (will be wrapped in Arc)
    pub fn assign_value(&mut self, slot: usize, ikey: u64, key_len: u8, value: V) {
        debug_assert!(
            key_len <= 8,
            "assign_value: key_len must be 0-8 for inline keys"
        );

        self.assign(slot, ikey, key_len, LeafValue::Value(Arc::new(value)));
    }

    /// Assign an already-Arc-wrapped value (for efficiency when Arc is pre-allocated).
    ///
    /// # Parameters
    /// - `slot`: Physical slot index
    /// - `ikey`: 8-byte key (big-endian)
    /// - `key_len`: Actual key length (0-8)
    /// - `value`: The Arc-wrapped value to store
    pub fn assign_arc(&mut self, slot: usize, ikey: u64, key_len: u8, arc_value: Arc<V>) {
        debug_assert!(
            key_len <= 8,
            "assign_arc: key_len must be 0-8 for inline keys"
        );

        self.assign(slot, ikey, key_len, LeafValue::Value(arc_value));
    }

    /// Check if slot 0 can be reused for a new key.
    ///
    /// Slot 0 stores the `ikey_bound` for B-link navigation.
    /// **Correct Rule** (matching C++ reference):
    /// - `prev` is null (no predecessor leaf), **OR**
    /// - New ikey matches current `ikey_bound` (slot 0's ikey)
    ///
    /// # Arguments
    ///
    /// * `new_ikey` - The ikey of the key to insert
    ///
    /// # Returns
    ///
    /// `true` if slot 0 can be reused, `false` otherwise.
    #[inline]
    #[must_use]
    pub const fn can_reuse_slot0(&self, new_ikey: u64) -> bool {
        // Rule 1: No predecessor leaf means slot 0 is always available
        if self.prev().is_null() {
            return true;
        }

        // Rule 2: Same ikey as current ikey_bound (slot 0) is safe
        self.ikey0[0] == new_ikey
    }

    /// Swap a value at a slot, returning the old value.
    ///
    /// Used when updating an existing key.
    ///
    /// # Arguments
    ///
    /// * `slot` - Physical slot index (0..WIDTH)
    /// * `new_value` - The new Arc-wrapped value to store
    ///
    /// # Returns
    ///
    /// The previous value at the slot, or None if slot was empty/layer.
    ///
    /// # Panics
    ///
    /// Panics in debug mode if slot >= WIDTH.
    #[expect(clippy::indexing_slicing, reason = "bounds checked via debug_assert")]
    pub fn swap_value(&mut self, slot: usize, new_value: Arc<V>) -> Option<Arc<V>> {
        debug_assert!(slot < WIDTH, "slot {slot} >= WIDTH {WIDTH}");

        // Plain field access with StdMem::replace
        let old_value: LeafValue<V> =
            StdMem::replace(&mut self.leaf_values[slot], LeafValue::Value(new_value));

        match old_value {
            LeafValue::Value(arc) => Some(arc),

            LeafValue::Layer(_) | LeafValue::Empty => None,
        }
    }

    // ============================================================================
    //  Split Operations
    // ============================================================================

    /// Split this leaf, moving upper half to a new leaf.
    ///
    /// **Important:** The new leaf is returned in a Box. The caller
    /// (`MassTree`) must store it in the arena to ensure pointer stability,
    /// then link the leaves.
    ///
    /// After this operation:
    /// - `self` contains entries from position 0 to split_pos-1
    /// - returned leaf contains entries from `split_pos` to size-1
    /// - Leaf chain pointers are NOT updated here (caller must do it after arena allocation)
    ///
    /// # Arguments
    ///
    /// * `split_pos` - Logical position where to split (in pre-insert coordinates)
    ///
    /// # Returns
    ///
    /// A new leaf containing the upper half, and the split key.
    ///
    /// # Panics
    ///
    /// Panics in debug mode if `split_pos` is 0 or >= size.
    #[expect(
        clippy::indexing_slicing,
        reason = "Indices from permuter, valid by construction"
    )]
    pub fn split_into(&mut self, split_pos: usize) -> LeafSplitResult<V, WIDTH> {
        let mut new_leaf: Box<Self> = Self::new();
        let old_perm: Permuter<WIDTH> = self.permutation();
        let old_size: usize = old_perm.size();

        debug_assert!(
            split_pos > 0 && split_pos < old_size,
            "invalid split_pos {split_pos} for size {old_size}"
        );

        // Copy entries from split_pos to end into new leaf
        let entries_to_move: usize = old_size - split_pos;

        for i in 0..entries_to_move {
            let old_logical_pos: usize = split_pos + i;
            let old_slot: usize = old_perm.get(old_logical_pos);

            // Copy key metadata (plain field access, no atomics)
            let ikey: u64 = self.ikey(old_slot);
            let keylenx: u8 = self.keylenx(old_slot);

            // Allocate slot in new leaf (using natural order for simplicity)
            let new_slot: usize = i;

            new_leaf.ikey0[new_slot] = ikey;
            new_leaf.keylenx[new_slot] = keylenx;

            // Move value (take ownership, leave Empty behind)
            let old_value: LeafValue<V> =
                StdMem::replace(&mut self.leaf_values[old_slot], LeafValue::Empty);
            new_leaf.leaf_values[new_slot] = old_value;
        }

        // Build new leaf's permutation (sorted order, size = entries_to_move)
        let new_perm: Permuter<WIDTH> = Permuter::make_sorted(entries_to_move);
        new_leaf.set_permutation(new_perm);

        // Update old leaf's permutation (just reduce size)
        let mut old_perm_updated: Permuter<WIDTH> = old_perm;
        old_perm_updated.set_size(split_pos);
        self.set_permutation(old_perm_updated);

        // Get split key (first key of new leaf)
        let split_ikey = new_leaf.ikey(new_perm.get(0));

        // Note: Leaf chain linking is done by the caller after arena allocation
        // to ensure the pointer is stable.

        LeafSplitResult {
            new_leaf,
            split_ikey,
            insert_into: InsertTarget::Left, // Caller determines based on insert_pos
        }
    }

    /// Move ALL entries from this leaf to a new right leaf.
    ///
    /// This is used for the left-sequential optimization edge case where
    /// `split_pos == 0` in post-insert coordinates, meaning all existing entries
    /// should go to the right leaf and the new key (being inserted at position 0)
    /// should go into this (left) leaf.
    ///
    /// After this operation:
    /// - `self` is empty (will receive the new key)
    /// - returned leaf contains all entries from `self`
    ///
    /// # Returns
    ///
    /// A new leaf containing all entries, and the split key (first key of new leaf).
    #[expect(
        clippy::indexing_slicing,
        reason = "Indices from permuter, valid by construction"
    )]
    pub fn split_all_to_right(&mut self) -> LeafSplitResult<V, WIDTH> {
        use crate::permuter::Permuter;

        let mut new_leaf = Self::new();
        let old_perm = self.permutation();
        let old_size = old_perm.size();

        debug_assert!(old_size > 0, "Cannot split empty leaf");

        // Move all entries to new leaf
        for i in 0..old_size {
            let old_slot = old_perm.get(i);

            new_leaf.ikey0[i] = self.ikey0[old_slot];
            new_leaf.keylenx[i] = self.keylenx[old_slot];
            new_leaf.leaf_values[i] =
                StdMem::replace(&mut self.leaf_values[old_slot], LeafValue::Empty);
        }

        // Set new leaf's permutation
        let new_perm = Permuter::make_sorted(old_size);
        new_leaf.set_permutation(new_perm);

        // Clear this leaf's permutation (now empty)
        self.set_permutation(Permuter::empty());

        // Split key is first key of new leaf
        let split_ikey = new_leaf.ikey(new_perm.get(0));

        LeafSplitResult {
            new_leaf,
            split_ikey,
            insert_into: InsertTarget::Left, // New key goes into empty left leaf
        }
    }

    // ============================================================================
    //  Invariant Checker
    // ============================================================================

    /// Verify leaf node invariants (debug builds only).
    ///
    /// Checks:
    /// - Permutation is valid
    /// - keylenx values are consistent with lv variants
    /// - ikeys are in sorted order (via permutation)
    ///
    /// # Panics
    /// If any invariant is violated.
    #[cfg(debug_assertions)]
    #[expect(
        clippy::indexing_slicing,
        reason = "Slot from Permuter, valid by construction"
    )]
    pub fn debug_assert_invariants(&self) {
        // Check permutation validity
        self.permutation.debug_assert_valid();

        let size: usize = self.size();

        // Check keylenx/leaf_values consistency for in-use slots
        for i in 0..size {
            let slot: usize = self.permutation.get(i);
            let keylenx: u8 = self.keylenx[slot];
            let leaf_value: &LeafValue<V> = &self.leaf_values[slot];

            // Layer slots must have Layer values
            if keylenx >= LAYER_KEYLENX {
                assert!(
                    leaf_value.is_layer(),
                    "slot {slot} has keylenx but non-Layer value"
                );
            } else if (keylenx > 0) || !leaf_value.is_empty() {
                assert!(
                    leaf_value.is_value() || leaf_value.is_empty(),
                    "slot {slot} has non-layer keylenx but Layer value"
                );
            }
        }

        // Check if ikey ordering (if size > 1)
        if size > 1 {
            for i in 1..size {
                let prev_slot: usize = self.permutation.get(i - 1);
                let curr_slot: usize = self.permutation.get(i);

                let prev_ikey: u64 = self.ikey0[prev_slot];
                let curr_ikey: u64 = self.ikey0[curr_slot];

                assert!(
                    prev_ikey <= curr_ikey,
                    "ikeys are not in sorted order: slot {prev_slot} ({prev_ikey:#x}) > slot {curr_slot} ({curr_ikey:#x})"
                );
            }
        }
    }

    /// No-op in release builds
    #[cfg(not(debug_assertions))]
    #[inline]
    pub fn debug_assert_invariants(&self) {}

    // ============================================================================
    //  Test Helpers
    // ============================================================================

    /// Assign raw values to a slot for testing (including invalid states).
    ///
    /// This allows setting up deliberately inconsistent states to test
    /// invariant checking. NOT for production use.
    #[cfg(test)]
    #[expect(clippy::indexing_slicing, reason = "Only for tests")]
    pub fn assign_raw_for_test(
        &mut self,
        slot: usize,
        ikey: u64,
        keylenx: u8,
        value: LeafValue<V>,
    ) {
        debug_assert!(slot < WIDTH, "assign_raw_for_test: slot out of bounds");

        self.ikey0[slot] = ikey;
        self.keylenx[slot] = keylenx;
        self.leaf_values[slot] = value;
    }
}

impl<V, const WIDTH: usize> Default for LeafNode<V, WIDTH> {
    fn default() -> Self {
        // Trigger compile-time WIDTH check
        let _: () = Self::WIDTH_CHECK;

        Self {
            version: NodeVersion::new(true),
            modstate: ModState::Insert,
            keylenx: [0; WIDTH],
            permutation: Permuter::empty(),
            ikey0: [0; WIDTH],
            leaf_values: StdArray::from_fn(|_| LeafValue::Empty),
            ksuf: StdPtr::null_mut(),
            next: StdPtr::null_mut(),
            prev: StdPtr::null_mut(),
            parent: StdPtr::null_mut(),
        }
    }
}

// ============================================================================
//  Type Aliases
// ============================================================================

/// Standard 15-slot leaf node (default mode with Arc<V>).
pub type LeafNode15<V> = LeafNode<V, 15>;

/// Compact 7-slot leaf node (fits in ~2 cache lines with small V).
pub type LeafNodeCompact<V> = LeafNode<V, 7>;

// ============================================================================
//  Compile-time Size Assertions
// ============================================================================

/// Compile-time size check for `LeafNode<u64, 15>`.
/// Should be around 448 bytes (7 cache lines) after alignment.
///
/// The enum discriminant adds overhead compared to C++ union approach, but keeps
/// type safety. `LeafValue<u64>` is 16 bytes (Arc ptr + discriminant).
const _: () = {
    // Compile-time assertion: ensure node stays cache-friendly
    const SIZE: usize = StdMem::size_of::<LeafNode<u64, 15>>();
    const ALIGN: usize = StdMem::align_of::<LeafNode<u64, 15>>();

    // Should fit in 8 cache lines (512 bytes) at most
    assert!(SIZE <= 512, "LeafNode exceeds 8 cache lines");

    // Should be cache cache-aligned
    assert!(ALIGN == 64, "LeafNode not cache-line-aligned");
};

#[cfg(test)]
mod tests {
    use super::*;

    // ============================================================================
    //  Basic Tests
    // ============================================================================

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
        //  FIXED: Use a real allocation to get a pointer with valid provenance
        let boxed: Box<u8> = Box::new(0xBE);
        let ptr: *mut u8 = Box::into_raw(boxed);

        let lv: LeafValue<u64> = LeafValue::Layer(ptr);

        assert!(!lv.is_empty());
        assert!(!lv.is_value());
        assert!(lv.is_layer());
        assert_eq!(lv.as_layer(), ptr);

        // Clean up the allocation
        //  SAFETY: ptr came from Box::into_raw above
        unsafe { drop(Box::from_raw(ptr)) };
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

    #[test]
    fn test_leaf_node_new() {
        let node: Box<LeafNode<u64, 15>> = LeafNode::new();

        assert!(node.version().is_leaf());
        assert_eq!(node.size(), 0);
        assert!(node.is_empty());
        assert!(!node.is_full());
        assert!(node.safe_next().is_null());
        assert!(node.prev().is_null());
        assert!(node.parent().is_null());
    }

    #[test]
    fn test_leaf_node_new_root() {
        let node: Box<LeafNode<u64, 15>> = LeafNode::new_root();

        assert!(node.version().is_leaf());
        assert!(node.version().is_root());
    }

    #[test]
    fn test_leaf_node_assign() {
        let mut node: Box<LeafNode<u64, 15>> = LeafNode::new();

        // Assign to slot 0 (value is wrapped in Arc internally)
        node.assign_value(0, 0x1234_5678_0000_0000, 4, 100);

        assert_eq!(node.ikey(0), 0x1234_5678_0000_0000);
        assert_eq!(node.keylenx(0), 4);
        assert!(node.leaf_value(0).is_value());
        assert_eq!(**node.leaf_value(0).as_value(), 100); // Double deref for Arc<u64>

        // Test clone_arc for optimistic reads
        let cloned: Arc<u64> = node.leaf_value(0).clone_arc();
        assert_eq!(*cloned, 100);
    }

    #[test]
    fn test_leaf_node_permutation() {
        let mut node: Box<LeafNode<u64, 15>> = LeafNode::new();

        // Initially empty
        assert_eq!(node.permutation().size(), 0);

        // Set a permutation with 3 elements
        let perm: Permuter<15> = Permuter::make_sorted(3);
        node.set_permutation(perm);

        assert_eq!(node.size(), 3);
        assert!(!node.is_empty());
        assert!(!node.is_full());
    }

    #[test]
    fn test_safe_next_masks_mark() {
        let mut node: Box<LeafNode<u64, 15>> = LeafNode::new();

        // Use a real allocation to get valid provenance
        let other_node: Box<LeafNode<u64, 15>> = LeafNode::new();
        let fake_next: *mut LeafNode<u64> = Box::into_raw(other_node);

        node.set_next(fake_next);
        assert_eq!(node.safe_next(), fake_next);
        assert!(!node.next_is_marked());

        node.mark_next();
        assert!(node.next_is_marked());
        assert_eq!(node.safe_next(), fake_next); // Mark bit masked off

        node.unmark_next();
        assert!(!node.next_is_marked());

        // Clean up the allocation
        // SAFETY: fake_next came from Box::into_raw above
        unsafe { drop(Box::from_raw(fake_next)) };
    }

    #[test]
    fn test_keylenx_helpers() {
        assert!(!LeafNode::<u64>::keylenx_is_layer(0));
        assert!(!LeafNode::<u64>::keylenx_is_layer(8));
        assert!(!LeafNode::<u64>::keylenx_is_layer(64));
        assert!(!LeafNode::<u64>::keylenx_is_layer(127));
        assert!(LeafNode::<u64>::keylenx_is_layer(128));
        assert!(LeafNode::<u64>::keylenx_is_layer(255));

        assert!(!LeafNode::<u64>::keylenx_has_ksuf(0));
        assert!(!LeafNode::<u64>::keylenx_has_ksuf(8));
        assert!(LeafNode::<u64>::keylenx_has_ksuf(64));
        assert!(!LeafNode::<u64>::keylenx_has_ksuf(128));
    }

    #[test]
    fn test_compact_leaf_node() {
        let node: Box<LeafNodeCompact<u64>> = LeafNode::new();

        assert_eq!(node.size(), 0);
        // Compact node should work the same way
    }

    #[test]
    fn test_ikey_bound() {
        let mut node: Box<LeafNode<u64, 15>> = LeafNode::new();

        // ikey_bound returns ikey0[0] - use assign_value to set it
        node.assign_value(0, 0xABCD_0000_0000_0000, 4, 42);
        assert_eq!(node.ikey_bound(), 0xABCD_0000_0000_0000);
    }

    #[test]
    fn test_modstate() {
        let mut node: Box<LeafNode<u64, 15>> = LeafNode::new();

        assert_eq!(node.modstate(), ModState::Insert);

        node.set_modstate(ModState::Remove);
        assert_eq!(node.modstate(), ModState::Remove);

        node.set_modstate(ModState::DeletedLayer);
        assert_eq!(node.modstate(), ModState::DeletedLayer);
    }

    // ============================================================================
    //  WIDTH Edge Cases
    // ============================================================================

    #[test]
    fn test_width_1_node() {
        let node: Box<LeafNode<u64, 1>> = LeafNode::new();

        assert_eq!(node.size(), 0);
        assert!(node.is_empty());
        assert!(!node.is_full());

        // Verify single-slot operations work
        assert_eq!(node.ikey(0), 0);
        assert_eq!(node.keylenx(0), 0);
    }

    #[test]
    fn test_width_15_full() {
        let mut node: Box<LeafNode<u64, 15>> = LeafNode::new();
        let perm = Permuter::make_sorted(15);
        node.set_permutation(perm);

        assert_eq!(node.size(), 15);
        assert!(!node.is_empty());
        assert!(node.is_full());
    }

    // ============================================================================
    //  Invariant Tests
    // ============================================================================

    #[test]
    #[should_panic(expected = "ikeys are not in sorted order")]
    #[cfg(debug_assertions)]
    fn test_invariant_unsorted_ikeys() {
        let mut node: Box<LeafNode<u64, 15>> = LeafNode::new();

        // Set up unsorted keys
        node.assign_value(0, 0x2000_0000_0000_0000, 2, 200);
        node.assign_value(1, 0x1000_0000_0000_0000, 2, 100);

        let mut perm = Permuter::empty();
        let _ = perm.insert_from_back(0); // slot 0 at logical pos 0
        let _ = perm.insert_from_back(1); // slot 1 at logical pos 1 (but ikey[1] < ikey[0]!)
        node.set_permutation(perm);

        node.debug_assert_invariants(); // Should panic
    }

    #[test]
    #[should_panic(expected = "has keylenx but non-Layer value")]
    #[cfg(debug_assertions)]
    fn test_invariant_layer_mismatch() {
        let mut node: Box<LeafNode<u64, 15>> = LeafNode::new();

        // Set keylenx to indicate layer but lv is Value (deliberately invalid)
        node.assign_raw_for_test(
            0,
            0x1000_0000_0000_0000,
            LAYER_KEYLENX,
            LeafValue::Value(Arc::new(42)),
        );

        let mut perm = Permuter::empty();
        let _ = perm.insert_from_back(0);
        node.set_permutation(perm);

        node.debug_assert_invariants(); // Should panic
    }

    #[test]
    fn test_invariant_valid_node() {
        let mut node: Box<LeafNode<u64, 15>> = LeafNode::new();

        // Set up correctly sorted keys
        node.assign_value(0, 0x1000_0000_0000_0000, 2, 100);
        node.assign_value(1, 0x2000_0000_0000_0000, 2, 200);
        node.assign_value(2, 0x3000_0000_0000_0000, 2, 300);

        let mut perm = Permuter::empty();
        let _ = perm.insert_from_back(0); // slot 0
        let _ = perm.insert_from_back(1); // slot 1
        let _ = perm.insert_from_back(2); // slot 2
        node.set_permutation(perm);

        // Should not panic
        node.debug_assert_invariants();
    }

    // ============================================================================
    //  LeafValueIndex Tests (Index Mode)
    // ============================================================================

    #[test]
    fn test_leaf_value_index_empty() {
        let lv: LeafValueIndex<u64> = LeafValueIndex::empty();
        assert!(lv.is_empty());
        assert!(!lv.is_value());
        assert!(!lv.is_layer());
    }

    #[test]
    fn test_leaf_value_index_value() {
        let lv: LeafValueIndex<u64> = LeafValueIndex::Value(42);
        assert!(!lv.is_empty());
        assert!(lv.is_value());
        assert!(!lv.is_layer());
        assert_eq!(lv.value(), 42); // Direct copy, no Arc

        // Copy semantics
        let copied: u64 = lv.value();
        assert_eq!(copied, 42);
    }

    #[test]
    fn test_leaf_value_index_layer() {
        // Use a real allocation to get a pointer with valid provenance
        let boxed: Box<u8> = Box::new(0xBE);
        let ptr: *mut u8 = Box::into_raw(boxed);

        let lv: LeafValueIndex<u64> = LeafValueIndex::Layer(ptr);

        assert!(!lv.is_empty());
        assert!(!lv.is_value());
        assert!(lv.is_layer());
        assert_eq!(lv.as_layer(), ptr);

        // Clean up the allocation
        // SAFETY: ptr came from Box::into_raw above
        unsafe { drop(Box::from_raw(ptr)) };
    }

    #[test]
    fn test_leaf_value_index_is_copy() {
        // LeafValueIndex<V: Copy> should itself be Copy
        let lv: LeafValueIndex<u64> = LeafValueIndex::Value(42);
        let lv2: LeafValueIndex<u64> = lv; // Copy, not move

        assert_eq!(lv.value(), 42);
        assert_eq!(lv2.value(), 42);
    }
}
