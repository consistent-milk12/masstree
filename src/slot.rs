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

    // ========================================================================
    //  Raw Pointer Operations (for leaf storage)
    // ========================================================================

    /// Convert an output to a raw pointer for storage in leaf nodes.
    ///
    /// This clones the output and converts it to a raw pointer. The caller
    /// is responsible for either calling `output_from_raw` to recover it or
    /// `cleanup_value_ptr` to clean it up.
    ///
    /// - For `LeafValue<V>`: `Arc::into_raw(Arc::clone(&output))`
    /// - For `LeafValueIndex<V>`: `Box::into_raw(Box::new(output))`
    fn output_to_raw(output: &Self::Output) -> *mut u8;

    /// Reconstruct an output from a raw pointer.
    ///
    /// This increments the refcount (for Arc) or copies the value (for Copy types),
    /// then returns an owned Output. The raw pointer remains valid.
    ///
    /// # Safety
    ///
    /// - `ptr` must be non-null
    /// - `ptr` must have been created by `output_to_raw` or equivalent
    /// - For Arc mode: the Arc must still be live (not cleaned up)
    unsafe fn output_from_raw(ptr: *const u8) -> Self::Output;

    /// Convert an output to a raw pointer, consuming the output.
    ///
    /// Unlike `output_to_raw`, this takes ownership and doesn't clone.
    /// More efficient when the output is no longer needed.
    ///
    /// - For `LeafValue<V>`: `Arc::into_raw(output)` directly
    /// - For `LeafValueIndex<V>`: `Box::into_raw(Box::new(output))`
    fn output_consume_to_raw(output: Self::Output) -> *mut u8;

    /// Clean up a raw pointer created by `output_to_raw`.
    ///
    /// This is the counterpart to `output_to_raw` - it frees the memory
    /// allocated when converting an output to a raw pointer.
    ///
    /// - For `LeafValue<V>`: Decrements Arc refcount (drops the cloned Arc)
    /// - For `LeafValueIndex<V>`: Drops the Box
    ///
    /// # Safety
    ///
    /// - `ptr` must have been created by `output_to_raw`
    /// - `ptr` must not have been already cleaned up
    /// - Caller must ensure no references to the pointed-to value exist
    unsafe fn cleanup_output_raw(ptr: *mut u8);
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

        let old: Self = StdMem::replace(self, Self::Value(new_output));

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

    #[inline(always)]
    fn output_to_raw(output: &Arc<V>) -> *mut u8 {
        Arc::into_raw(Arc::clone(output)) as *mut u8
    }

    #[inline(always)]
    unsafe fn output_from_raw(ptr: *const u8) -> Arc<V> {
        // SAFETY: Caller guarantees ptr is valid Arc<V> pointer
        unsafe {
            Arc::increment_strong_count(ptr.cast::<V>());
            Arc::from_raw(ptr.cast::<V>())
        }
    }

    #[inline(always)]
    fn output_consume_to_raw(output: Arc<V>) -> *mut u8 {
        Arc::into_raw(output) as *mut u8
    }

    #[inline(always)]
    unsafe fn cleanup_output_raw(ptr: *mut u8) {
        // SAFETY: Caller guarantees ptr came from output_to_raw (Arc::into_raw)
        unsafe {
            drop(Arc::from_raw(ptr.cast::<V>()));
        }
    }
}

// ============================================================================
//  ValueSlot impl for LeafValueIndex<V: Copy> (Inline Mode)
// ============================================================================
//
// NOTE: We use Box<V> for storage instead of pointer-punning because:
// - `get_ref()` returns `&V`, which requires V to exist at a valid memory address
// - Pointer-punning stores the value IN the pointer bits, with no backing memory
// - Dereferencing a punned pointer causes SIGSEGV
//
// Future optimization: Add `get_copy()` API that returns V by value, then
// pointer-punning could be used for that path while keeping `get_ref()` working.

impl<V: Copy> ValueSlot for LeafValueIndex<V> {
    type Value = V;
    type Output = V; // Returns V directly, no Arc!

    #[inline(always)]
    fn into_output(value: V) -> V {
        value // Identity - no allocation at this stage
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

        let old: Self = StdMem::replace(self, Self::Value(new_output));
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

    #[inline(always)]
    fn output_to_raw(output: &V) -> *mut u8 {
        // Box the value to get a stable pointer that can be dereferenced.
        // This is required for `get_ref()` to work (returns &V).
        Box::into_raw(Box::new(*output)).cast::<u8>()
    }

    #[inline(always)]
    unsafe fn output_from_raw(ptr: *const u8) -> V {
        // SAFETY: Caller guarantees ptr is valid V pointer from Box::into_raw.
        // V is Copy, so we just read the value (don't consume the Box).
        unsafe { *ptr.cast::<V>() }
    }

    #[inline(always)]
    fn output_consume_to_raw(output: V) -> *mut u8 {
        // Box the value to get a stable pointer.
        Box::into_raw(Box::new(output)).cast::<u8>()
    }

    #[inline(always)]
    unsafe fn cleanup_output_raw(ptr: *mut u8) {
        // SAFETY: Caller guarantees ptr came from output_to_raw (Box::into_raw)
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
mod unit_tests;
