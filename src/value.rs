//! Value types for leaf node storage.
//!
//! This module provides the value enums used in leaf nodes:
//! - [`LeafValue<V>`]: Arc-wrapped values for the default storage mode
//! - [`LeafValueIndex<V>`]: Inline copy values for the index storage mode
//!
//! These types are leaf-implementation agnostic and can be used with any WIDTH.

use std::sync::Arc;

// ============================================================================
//  LeafValue<V> - Arc-based storage
// ============================================================================

/// Value stored in a leaf slot (default mode with `Arc<V>`).
///
/// Uses reference counting for cheap cloning on reads.
#[derive(Default)]
pub enum LeafValue<V> {
    /// Slot is empty (no key assigned).
    #[default]
    Empty,

    /// Slot contains an Arc-wrapped value.
    Value(Arc<V>),

    /// Slot contains a pointer to a next-layer subtree.
    Layer(*mut u8),
}

impl<V> LeafValue<V> {
    /// Create an empty leaf value.
    #[must_use]
    #[inline(always)]
    pub const fn empty() -> Self {
        Self::Empty
    }

    /// Check if this slot is empty.
    #[must_use]
    #[inline(always)]
    pub const fn is_empty(&self) -> bool {
        matches!(self, Self::Empty)
    }

    /// Check if this slot contains a value.
    #[must_use]
    #[inline(always)]
    pub const fn is_value(&self) -> bool {
        matches!(self, Self::Value(_))
    }

    /// Check if this slot contains a layer pointer.
    #[must_use]
    #[inline(always)]
    pub const fn is_layer(&self) -> bool {
        matches!(self, Self::Layer(_))
    }

    /// Try to get the Arc value, returning None if not a value variant.
    #[must_use]
    #[inline(always)]
    pub const fn try_as_value(&self) -> Option<&Arc<V>> {
        match self {
            Self::Value(arc) => Some(arc),
            _ => None,
        }
    }

    /// Try to clone the Arc value, returning None if not a value variant.
    #[must_use]
    #[inline(always)]
    pub fn try_clone_arc(&self) -> Option<Arc<V>> {
        match self {
            Self::Value(arc) => Some(Arc::clone(arc)),
            _ => None,
        }
    }

    /// Try to get the layer pointer, returning None if not a Layer variant.
    #[must_use]
    #[inline(always)]
    pub const fn try_as_layer(&self) -> Option<*mut u8> {
        match self {
            Self::Layer(ptr) => Some(*ptr),
            _ => None,
        }
    }

    /// Get the layer pointer.
    ///
    /// # Panics
    /// Panics if this is not a Layer variant.
    #[must_use]
    #[inline(always)]
    #[expect(clippy::expect_used, reason = "Invariant ensured by caller")]
    pub const fn as_layer(&self) -> *mut u8 {
        self.try_as_layer()
            .expect("LeafValue::as_layer called on non-Layer variant")
    }

    /// Gets a reference to the Arc-wrapped value.
    ///
    /// # Panics
    /// Panics if this is not a Value variant.
    #[must_use]
    #[inline(always)]
    #[expect(clippy::expect_used, reason = "Invariant ensured by caller")]
    pub const fn as_value(&self) -> &Arc<V> {
        self.try_as_value()
            .expect("LeafValue::as_value called on non-Value variant")
    }

    /// Clone the Arc<V> (cheap reference counting increment).
    ///
    /// # Panics
    /// Panics if this is not a Value variant.
    #[must_use]
    #[inline(always)]
    #[expect(clippy::expect_used, reason = "Invariant ensured by caller")]
    pub fn clone_arc(&self) -> Arc<V> {
        self.try_clone_arc()
            .expect("LeafValue::clone_arc called on non-Value variant")
    }

    /// Get a mutable reference to the Arc-wrapped value.
    ///
    /// # Panics
    /// Panics if this is not a Value variant.
    #[inline(always)]
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
            Self::Value(arc) => Self::Value(Arc::clone(arc)),
            Self::Layer(ptr) => Self::Layer(*ptr),
        }
    }
}

// SAFETY: LeafValue is safe to send between threads because:
// 1. `Empty` has no data
// 2. `Value(Arc<V>)` is Send when V: Send+Sync
// 3. `Layer(*mut u8)` points to tree-owned nodes protected by OCC
unsafe impl<V: Send + Sync> Send for LeafValue<V> {}
unsafe impl<V: Send + Sync> Sync for LeafValue<V> {}

impl<V> std::fmt::Debug for LeafValue<V> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Empty => write!(f, "Empty"),
            Self::Value(_) => write!(f, "Value(...)"),
            Self::Layer(ptr) => write!(f, "Layer({ptr:?})"),
        }
    }
}

// ============================================================================
//  LeafValueIndex<V> - Inline copy storage
// ============================================================================

/// Value stored in a leaf slot (index mode with inline `V: Copy`).
///
/// For high-performance index use cases where values are small and Copy.
#[derive(Clone, Default)]
pub enum LeafValueIndex<V: Copy> {
    /// Slot is empty (no key assigned).
    #[default]
    Empty,

    /// Slot contains an inline value (copied on read).
    Value(V),

    /// Slot contains a pointer to a next-layer subtree.
    Layer(*mut u8),
}

// SAFETY: Same reasoning as LeafValue
unsafe impl<V: Copy + Send> Send for LeafValueIndex<V> {}
unsafe impl<V: Copy + Sync> Sync for LeafValueIndex<V> {}

impl<V: Copy> LeafValueIndex<V> {
    /// Create an empty leaf value.
    #[must_use]
    #[inline(always)]
    pub const fn empty() -> Self {
        Self::Empty
    }

    /// Check if this slot is empty.
    #[must_use]
    #[inline(always)]
    pub const fn is_empty(&self) -> bool {
        matches!(self, Self::Empty)
    }

    /// Check if this slot contains a value.
    #[must_use]
    #[inline(always)]
    pub const fn is_value(&self) -> bool {
        matches!(self, Self::Value(_))
    }

    /// Check if this slot contains a layer pointer.
    #[must_use]
    #[inline(always)]
    pub const fn is_layer(&self) -> bool {
        matches!(self, Self::Layer(_))
    }

    /// Try to get the value, returning None if not a Value variant.
    #[must_use]
    #[inline(always)]
    pub const fn try_value(&self) -> Option<V> {
        match self {
            Self::Value(v) => Some(*v),
            _ => None,
        }
    }

    /// Try to get the layer pointer, returning None if not a Layer variant.
    #[must_use]
    #[inline(always)]
    pub const fn try_as_layer(&self) -> Option<*mut u8> {
        match self {
            Self::Layer(ptr) => Some(*ptr),
            _ => None,
        }
    }

    /// Get the value (copied).
    ///
    /// # Panics
    /// Panics if this is not a Value variant.
    #[must_use]
    #[inline(always)]
    #[expect(clippy::expect_used, reason = "Invariant ensured by caller")]
    pub const fn value(&self) -> V {
        self.try_value()
            .expect("LeafValueIndex::value called on non-Value variant")
    }

    /// Get the layer pointer.
    ///
    /// # Panics
    /// Panics if this is not a Layer variant.
    #[must_use]
    #[inline(always)]
    #[expect(clippy::expect_used, reason = "Invariant ensured by caller")]
    pub const fn as_layer(&self) -> *mut u8 {
        self.try_as_layer()
            .expect("LeafValueIndex::as_layer called on non-Layer variant")
    }
}

impl<V: Copy> Copy for LeafValueIndex<V> {}

impl<V: Copy> std::fmt::Debug for LeafValueIndex<V> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Empty => write!(f, "Empty"),
            Self::Value(_) => write!(f, "Value(...)"),
            Self::Layer(ptr) => write!(f, "Layer({ptr:?})"),
        }
    }
}

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
