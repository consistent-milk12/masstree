//! Value slot abstraction for [`MassTree`] storage modes.
//!
//! This module provides the [`ValueSlot`] trait that abstracts how values
//! are stored in leaf nodes. The trait is implemented for existing types:
//!
//! - [`LeafValue<V>`]: Arc-based storage (default mode)
//! - [`LeafValueIndex`]: Inline storage (index mode)
//!
//! # Design: Value vs Output
//! The trait distinguishes between:
//! - `Value`: What users provie to `insert()` (e.g., `String`, `u64`)
//! - `Output`: What the tree carries internally and returns (e.g. `Arc<String>`, `u64`)
//!
//! This seperation is criticial for retry loops: when an insert splits a leaf or
//! crates a layer, the tree must retry without re-allocating the [`Arc`]. The
//! `Output` type is created once via `into_output()` and reused across retries.
//!
//! # Storage Nodes
//!
//! | Node    | Slot Type           | Output Type | Allocation      |
//! |---------|---------------------|-------------|-----------------|
//! | Default | `LeafValue<V>`      | `Arc<V>`    | Once per insert |
//! | Index   | `LeafValueIndex<V>` | `V`         | None (copy)     |

use std::mem as StdMem;
use std::sync::Arc;

use crate::value::{LeafValue, LeafValueIndex};

// ================================================================================
//  ValueSlot Trait
// ================================================================================

/// Trait for value slots stored in leaf nodes.
///
/// This trait abstracts the storage strategy for leaf values, enabling a single
/// tree implementation to work with both Arc-based and inline storage.
/// The core complexity is ensuring allocation happens exactly once per insert,
/// even across retries (splits, layer creation). The trait achieves this by
/// separating Value (user input) from Output (retryable handle). A secondary
/// benefit is unifying the implementation via [`TreeCore<S>`] to avoid code duplication.
///
/// # Associated Types
/// - `Value`: The user-facing value type (what users insert)
/// - `Output`: The type carried across retires and returned from `get()`
///
/// # Why `Output` is separate from `Value`
/// Insert operations may retry
pub trait ValueSlot: Default + Sized {
    /// The user facing value type.
    ///
    /// This is what users provide to `insert()`.
    type Value;

    /// The type returned from get operations and carried across retries.
    ///
    /// - For `LeafValue<V>`: `Arc<V>` (cheap clone via refcount)
    /// - For `LeafValueIndex<V>`: `V` (direct copy)
    ///
    /// Mus be [`Clone`] to support returning values from optimistic reads.
    type Output: Clone;

    // ================================================================================
    //  Output Conversion (Critical for Retry Semantics)
    // ================================================================================

    /// Convert a user value into an output handle.
    ///
    /// This is called exactly once per insert attempt. For [`Arc`] mode, this
    /// performs the heap allocation. The returned `Output` is then carried
    /// across any retries (splits, layer creation).
    ///
    /// - For [`LeafValue<V>`]: `Arc::new(value)`
    /// - For [`LeafValueIndex<V>`]: `value` (identity)
    fn into_output(value: Self::Value) -> Self::Output;

    /// Create a slot from an output handle.
    ///
    /// Used when:
    /// - Storing a new value after successful insert position found
    /// - Moving existing values during layer creation
    ///
    /// - For [`LeafValue<V>`]: Wraps the Arc directly (no allocation)
    /// - For [`LeafValueIndex<V>`]: Wraps the value directly
    fn from_output(output: Self::Output) -> Self;

    // ========================================================================
    //  Predicates
    // ========================================================================

    /// Check if slot is empty.
    fn is_empty(&self) -> bool;

    /// Check if slot contains a value.
    fn is_value(&self) -> bool;

    /// Check if slot contains a layer pointer.
    fn is_layer(&self) -> bool;

    // ========================================================================
    //  Extraction
    // ========================================================================

    /// Try to get the output value.
    ///
    /// Returns `Some(Output)` if slot contains a value, `None` otherwise.
    ///
    /// - For `LeafValue<V>`: Returns `Some(Arc::clone(&arc))`
    /// - For `LeafValueIndex<V>`: Returns `Some(value)` (copy)
    fn try_get(&self) -> Option<Self::Output>;

    /// Try to get the layer pointer.
    ///
    /// Returns `Some(ptr)` if slot contains a layer, `None` otherwise.
    fn try_layer(&self) -> Option<*mut u8>;

    // ========================================================================
    //  Construction
    // ========================================================================

    /// Create a slot containing a layer pointer.
    ///
    /// This is a static constructor for creating layer slots directly.
    fn layer(ptr: *mut u8) -> Self;

    // ========================================================================
    //  Mutation
    // ========================================================================

    /// Replace the slot with a layer pointer.
    fn set_layer(&mut self, ptr: *mut u8);

    /// Replace the slot's contents with a new output, returning the old output.
    ///
    /// Used when updating an existing key's value.
    ///
    /// # Arguments
    ///
    /// * `new_output` - The new output to store (already converted via `into_output`)
    ///
    /// # Returns
    ///
    /// * `Some(old_output)` - If slot previously contained a value
    /// * `None` - If slot was empty or contained a layer
    fn swap_output(&mut self, new_output: Self::Output) -> Option<Self::Output>;

    /// Take the slot contents, leaving Empty in place.
    ///
    /// Used during splits to move values without cloning.
    #[must_use]
    fn take(&mut self) -> Self {
        StdMem::take(self)
    }

    /// Cleanup a raw pointer that was stored via `Arc::into_raw` or `Box::into_raw`.
    ///
    /// Called during node teardown (Drop) for non-layer slots.
    ///
    /// # Safety
    ///
    /// - `ptr` must be non-null and have been created by the corresponding
    ///   storage method (`assign_arc` for `LeafValue`, `assign_inline` for `LeafValueIndex`)
    /// - `ptr` must not have been already cleaned up
    /// - Caller must ensure no concurrent access to this pointer
    unsafe fn cleanup_value_ptr(ptr: *mut u8);
}

// ============================================================================
//  ValueSlot impl for LeafValue<V> (Default Arc Mode)
// ============================================================================

impl<V> ValueSlot for LeafValue<V> {
    type Value = V;
    type Output = Arc<V>;

    #[inline(always)]
    fn into_output(value: V) -> Arc<V> {
        Arc::new(value)
    }

    #[inline(always)]
    fn from_output(output: Arc<V>) -> Self {
        Self::Value(output)
    }

    #[inline(always)]
    fn is_empty(&self) -> bool {
        matches!(self, Self::Empty)
    }

    #[inline(always)]
    fn is_value(&self) -> bool {
        matches!(self, Self::Value(_))
    }

    #[inline(always)]
    fn is_layer(&self) -> bool {
        matches!(self, Self::Layer(_))
    }

    #[inline(always)]
    fn try_get(&self) -> Option<Arc<V>> {
        match self {
            Self::Value(arc) => Some(Arc::clone(arc)),

            _ => None,
        }
    }

    #[inline(always)]
    fn try_layer(&self) -> Option<*mut u8> {
        match self {
            Self::Layer(ptr) => Some(*ptr),

            _ => None,
        }
    }

    #[inline(always)]
    fn layer(ptr: *mut u8) -> Self {
        Self::Layer(ptr)
    }

    #[inline(always)]
    fn set_layer(&mut self, ptr: *mut u8) {
        *self = Self::Layer(ptr);
    }

    #[inline(always)]
    fn swap_output(&mut self, new_output: Arc<V>) -> Option<Arc<V>> {
        debug_assert!(
            !self.is_layer(),
            "swap_output called on Layer slot; layer pointer would be lost"
        );

        let old: Self = std::mem::replace(self, Self::Value(new_output));

        match old {
            Self::Value(arc) => Some(arc),

            _ => None,
        }
    }

    #[inline(always)]
    unsafe fn cleanup_value_ptr(ptr: *mut u8) {
        // SAFETY: Caller guarantees ptr came from Arc::into_raw
        unsafe {
            drop(Arc::from_raw(ptr.cast::<V>()));
        }
    }
}

// ============================================================================
//  ValueSlot impl for LeafValueIndex<V: Copy> (True Inline Mode)
// ============================================================================

impl<V: Copy> ValueSlot for LeafValueIndex<V> {
    type Value = V;
    type Output = V; // Returns V directly, no Arc!

    #[inline(always)]
    fn into_output(value: V) -> V {
        value // Identity - no allocation!
    }

    #[inline(always)]
    fn from_output(output: V) -> Self {
        Self::Value(output)
    }

    #[inline(always)]
    fn is_empty(&self) -> bool {
        matches!(self, Self::Empty)
    }

    #[inline(always)]
    fn is_value(&self) -> bool {
        matches!(self, Self::Value(_))
    }

    #[inline(always)]
    fn is_layer(&self) -> bool {
        matches!(self, Self::Layer(_))
    }

    #[inline(always)]
    fn try_get(&self) -> Option<V> {
        match self {
            Self::Value(v) => Some(*v),

            _ => None,
        }
    }

    #[inline(always)]
    fn try_layer(&self) -> Option<*mut u8> {
        match self {
            Self::Layer(ptr) => Some(*ptr),

            _ => None,
        }
    }

    #[inline(always)]
    fn layer(ptr: *mut u8) -> Self {
        Self::Layer(ptr)
    }

    #[inline(always)]
    fn set_layer(&mut self, ptr: *mut u8) {
        *self = Self::Layer(ptr);
    }

    #[inline(always)]
    fn swap_output(&mut self, new_output: V) -> Option<V> {
        debug_assert!(
            !self.is_layer(),
            "swap_output called on Layer slot; layer pointer would be lost"
        );

        let old: Self = std::mem::replace(self, Self::Value(new_output));
        match old {
            Self::Value(v) => Some(v),
            _ => None,
        }
    }

    #[inline(always)]
    unsafe fn cleanup_value_ptr(ptr: *mut u8) {
        // SAFETY: Caller guarantees ptr came from Box::into_raw
        unsafe {
            drop(Box::from_raw(ptr.cast::<V>()));
        }
    }
}

// ============================================================================
//  Tests
// ============================================================================

#[cfg(test)]
#[expect(clippy::unwrap_used, reason = "fail fast in tests")]
mod tests {
    use super::*;
    use std::ptr as StdPtr;

    // ------------------------------------------------------------------------
    //  LeafValue<V> (Arc mode) Tests
    // ------------------------------------------------------------------------

    #[test]
    fn arc_mode_into_output_allocates_once() {
        let output1: Arc<u64> = LeafValue::<u64>::into_output(42);
        let output2: Arc<u64> = Arc::clone(&output1);

        // Both point to same allocation
        assert_eq!(Arc::strong_count(&output1), 2);
        assert_eq!(*output1, 42);
        assert_eq!(*output2, 42);
    }

    #[test]
    fn arc_mode_from_output_no_realloc() {
        let output: Arc<u64> = Arc::new(42);
        let slot: LeafValue<u64> = LeafValue::from_output(Arc::clone(&output));

        // Slot shares the same Arc (refcount = 2)
        assert!(slot.is_value());
        assert_eq!(Arc::strong_count(&output), 2);
    }

    #[test]
    fn arc_mode_swap_output() {
        let mut slot: LeafValue<u64> = LeafValue::from_output(Arc::new(100));

        let old: Option<Arc<u64>> = slot.swap_output(Arc::new(200));
        assert_eq!(*old.unwrap(), 100);
        assert_eq!(*slot.try_get().unwrap(), 200);
    }

    #[test]
    fn arc_mode_swap_output_empty_returns_none() {
        let mut slot: LeafValue<u64> = LeafValue::Empty;

        let old: Option<Arc<u64>> = slot.swap_output(Arc::new(42));
        assert!(old.is_none());
        assert_eq!(*slot.try_get().unwrap(), 42);
    }

    #[test]
    fn arc_mode_predicates() {
        let empty: LeafValue<u64> = LeafValue::Empty;
        assert!(empty.is_empty());
        assert!(!empty.is_value());
        assert!(!empty.is_layer());

        let value: LeafValue<u64> = LeafValue::from_output(Arc::new(42));
        assert!(!value.is_empty());
        assert!(value.is_value());
        assert!(!value.is_layer());

        let mut layer: LeafValue<u64> = LeafValue::Empty;
        let mut dummy: u64 = 0;
        layer.set_layer(StdPtr::addr_of_mut!(dummy).cast());

        assert!(!layer.is_empty());
        assert!(!layer.is_value());
        assert!(layer.is_layer());
    }

    #[test]
    fn arc_mode_take() {
        let mut slot: LeafValue<u64> = LeafValue::from_output(Arc::new(42));
        let taken: LeafValue<u64> = slot.take();

        assert!(slot.is_empty());
        assert!(taken.is_value());
        assert_eq!(*taken.try_get().unwrap(), 42);
    }

    // ------------------------------------------------------------------------
    //  LeafValueIndex<V: Copy> (Inline mode) Tests
    // ------------------------------------------------------------------------

    #[test]
    fn inline_mode_into_output_no_allocation() {
        // into_output is identity for Copy types - no allocation!
        let output: u64 = LeafValueIndex::<u64>::into_output(42);
        assert_eq!(output, 42);
    }

    #[test]
    fn inline_mode_from_output() {
        let slot: LeafValueIndex<u64> = LeafValueIndex::from_output(42);
        assert!(slot.is_value());
        assert_eq!(slot.try_get(), Some(42));
    }

    #[test]
    fn inline_mode_swap_output() {
        let mut slot: LeafValueIndex<u64> = LeafValueIndex::from_output(100);

        let old: Option<u64> = slot.swap_output(200);
        assert_eq!(old, Some(100));
        assert_eq!(slot.try_get(), Some(200));
    }

    #[test]
    fn inline_mode_predicates() {
        let empty: LeafValueIndex<u64> = LeafValueIndex::Empty;
        assert!(empty.is_empty());
        assert!(!empty.is_value());
        assert!(!empty.is_layer());

        let value: LeafValueIndex<u64> = LeafValueIndex::from_output(42);
        assert!(!value.is_empty());
        assert!(value.is_value());
        assert!(!value.is_layer());
    }

    #[test]
    fn inline_mode_is_copy() {
        let slot: LeafValueIndex<u64> = LeafValueIndex::from_output(42);
        let copied: LeafValueIndex<u64> = slot; // Copy, not move

        assert_eq!(slot.try_get(), Some(42));
        assert_eq!(copied.try_get(), Some(42));
    }

    #[test]
    fn inline_mode_take() {
        let mut slot: LeafValueIndex<u64> = LeafValueIndex::from_output(42);
        let taken: LeafValueIndex<u64> = slot.take();

        assert!(slot.is_empty());
        assert_eq!(taken.try_get(), Some(42));
    }

    #[test]
    fn inline_mode_layer() {
        let mut slot: LeafValueIndex<u64> = LeafValueIndex::Empty;
        let mut dummy: u64 = 0;
        let ptr: *mut u8 = StdPtr::addr_of_mut!(dummy).cast();

        slot.set_layer(ptr);

        assert!(slot.is_layer());
        assert_eq!(slot.try_layer(), Some(ptr));
        assert!(slot.try_get().is_none());
    }

    // ------------------------------------------------------------------------
    //  Cross-mode comparison tests
    // ------------------------------------------------------------------------

    #[test]
    fn both_modes_layer_works_same() {
        let mut arc_slot: LeafValue<u64> = LeafValue::Empty;
        let mut inline_slot: LeafValueIndex<u64> = LeafValueIndex::Empty;

        let mut dummy: u64 = 0;
        let ptr: *mut u8 = StdPtr::addr_of_mut!(dummy).cast();

        arc_slot.set_layer(ptr);
        inline_slot.set_layer(ptr);

        assert_eq!(arc_slot.try_layer(), Some(ptr));
        assert_eq!(inline_slot.try_layer(), Some(ptr));
    }
}
