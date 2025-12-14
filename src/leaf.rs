//! Filepath: src/leaf.rs
//!
//! Leaf node for [`MassTree`].
//!
//! Leaf nodes store the actual key-value pairs, using a permutation array
//! for logical ordering without data movement.

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

    /// Try to get the layer pinter, returning None if not a Layer variant.
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
    #[must_use]
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
/// Leaves are linke for efficient range scans.
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
/// cache-line alignment. For WIDTH=15 with V=u64, total size is ~320 bytes.
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
    lv: [LeafValue<V>; WIDTH],

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
