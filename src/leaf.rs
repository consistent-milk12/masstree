//! Filepath: src/leaf.rs
//!
//! Leaf node for [`MassTree`].
//!
//! Leaf nodes store the actual key-value pairs, using a permutation array
//! for logical ordering without data movement.

use std::sync::Arc;

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
#[derive(Debug)]
pub enum LeafValue<V> {
    /// Slot is empty (no key assigned).
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

/// Value stored in a leaf slot (index mode with inline `V: Copy`).
///
/// For high-perf index use cases values are small and Copy.
///
/// # Type Parameter
/// `V` - The value type stored inline (must be `Copy`)
#[derive(Clone, Copy, Debug)]
pub enum LeafValueIndex<V: Copy> {
    /// Slot is empty (no key assigned).
    Empty,

    /// Slot contains an inline value (copied on read).
    Value(V),

    /// Slot contains a pointer to a next-layer subtree.
    ///
    /// This will be refined to proper node type.
    Layer(*mut u8),
}

impl<V> LeafValue<V> {
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
