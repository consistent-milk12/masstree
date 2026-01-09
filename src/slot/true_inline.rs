//! True-inline slot type for    the [`ValueSlot`] trait.
//!
//! This module provides [`TrueInlineSlot<V>`], a marker type that implements
//! `ValueSlot` for true-inline value storage.

use std::marker::PhantomData;
use std::ptr as StdPtr;

use crate::{inline::bits::InlineBits, slot::ValueSlot};

// ============================================================================
//  TrueInlineSlot<V>
// ============================================================================

/// Marker type for true-inline value storage.
///
/// This type implements `ValueSlot` with:
/// - `NEEDS_RETIREMENT = false` (values are inline, no heap allocation)
/// - `Value = V` and `Output = V` (both are the inline type)
///
/// # Important
///
/// The pointer-oriented methods (`output_to_raw`, `output_from_raw`, etc.) are
/// **not used** for true-inline storage. The tree routes value operations through
/// the leaf value traits instead (`LeafValueStore`, `LeafValueUpdate`, etc.).
///
/// The implementations here return sentinel/dummy values to satisfy the trait,
/// but calling them is a logic error.
#[derive(Clone, Debug)]
pub struct TrueInlineSlot<V: InlineBits> {
    _marker: PhantomData<V>,
}

impl<V: InlineBits> Default for TrueInlineSlot<V> {
    fn default() -> Self {
        Self {
            _marker: PhantomData,
        }
    }
}

impl<V: InlineBits> Copy for TrueInlineSlot<V> {}

/// Magic constant for encoding values in pointer addresses.
/// `XORed` with value bits to avoid null pointer (value 0 would otherwise be null).
pub const ENCODING_MAGIC: u64 = 0x5555_5555_5555_5555;

impl<V: InlineBits> ValueSlot for TrueInlineSlot<V> {
    /// True-inline storage does not need retirement.
    ///
    /// Values are stored in `AtomicU64` slots directly, not behind pointers.
    const NEEDS_RETIREMENT: bool = false;

    type Value = V;
    type Output = V;

    #[inline(always)]
    fn into_output(value: Self::Value) -> Self::Output {
        value
    }

    #[inline(always)]
    fn from_output(output: Self::Output) -> Self {
        // TrueInlineSlot is a marker type - it doesn't store values itself.
        // Values are stored directly in the leaf's inline_values array.
        let _ = output;
        Self {
            _marker: PhantomData,
        }
    }

    /// Not meaningful for true-inline - always returns false.
    ///
    /// [`TrueInlineSlot`] is a marker type; actual emptiness is determined
    /// by checking the leaf's `leaf_values` pointer (null = empty).
    #[inline(always)]
    fn is_empty(&self) -> bool {
        false
    }

    /// Not meaningful for true-inline - always returns true.
    ///
    /// [`TrueInlineSlot`] is a marker type; actual value presence is determined
    /// by checking the leaf's `leaf_values` pointer (sentinel = value).
    #[inline(always)]
    fn is_value(&self) -> bool {
        true
    }

    /// Not meaningful for true-inline - always returns false.
    ///
    /// [`TrueInlineSlot`] is a marker type; actual layer detection is determined
    /// by checking the leaf's `leaf_values` pointer (not null, not sentinel = layer).
    #[inline(always)]
    fn is_layer(&self) -> bool {
        false
    }

    /// Not used for true-inline.
    ///
    /// Value retrieval goes through [`crate::value::traits::LeafValueLoad::try_load_output`] instead.
    #[inline(always)]
    fn try_get(&self) -> Option<Self::Output> {
        // TrueInlineSlot is a marker type - use LeafValueLoad trait instead
        None
    }

    /// Not used for true-inline.
    ///
    /// Layer detection goes through leaf's `leaf_value_ptr()` instead.
    #[inline(always)]
    fn try_layer(&self) -> Option<*mut u8> {
        None
    }

    /// Not used for true-inline - layer pointers go directly to leaf.
    #[inline(always)]
    fn layer(_ptr: *mut u8) -> Self {
        Self {
            _marker: PhantomData,
        }
    }

    /// Not used for true-inline - layer pointers go directly to leaf.
    #[inline(always)]
    fn set_layer(&mut self, _ptr: *mut u8) {
        // No-op - layers are set directly on the leaf
    }

    /// Not used for true-inline.
    ///
    /// Value updates go through [`crate::value::traits::LeafValueUpdate::replace_value_output`] instead.
    #[inline(always)]
    fn swap_output(&mut self, new_output: Self::Output) -> Option<Self::Output> {
        // TrueInlineSlot is a marker type - use LeafValueUpdate trait instead
        let _ = new_output;
        None
    }

    /// Not used for true-inline - no cleanup needed.
    ///
    /// # Safety
    ///
    /// This method should never be called for true-inline storage.
    #[inline(always)]
    unsafe fn cleanup_value_ptr(_ptr: *mut u8) {
        // No-op - inline values don't have heap allocations
        debug_assert!(false, "cleanup_value_ptr called on TrueInlineSlot");
    }

    /// Not used for true-inline - values go through leaf accessor methods.
    ///
    /// # Panics
    ///
    /// Debug-asserts if called (logic error).
    /// Encode value bits in pointer address for true-inline storage.
    ///
    /// We XOR with a magic constant to avoid null pointer (value 0 would be null).
    /// The leaf's `set_leaf_value_ptr` will extract these bits and store them
    /// in `inline_values[slot]`.
    #[inline(always)]
    #[expect(clippy::cast_possible_truncation)]
    fn output_to_raw(output: &Self::Output) -> *mut u8 {
        let bits: u64 = output.to_bits();

        // XOR with magic to avoid null pointer
        let encoded: u64 = bits ^ ENCODING_MAGIC;

        StdPtr::without_provenance_mut(encoded as usize)
    }

    /// Decode value bits from an encoded pointer.
    ///
    /// For true-inline storage, values are encoded in pointer addresses
    /// (`XORed` with [`ENCODING_MAGIC`]). This decodes them back to the value type.
    ///
    /// # Safety
    ///
    /// The pointer must have been created by `output_to_raw` or `output_consume_to_raw`.
    #[inline(always)]
    unsafe fn output_from_raw(ptr: *const u8) -> Self::Output {
        let encoded: u64 = ptr.addr() as u64;
        let bits: u64 = encoded ^ ENCODING_MAGIC;
        V::from_bits(bits)
    }

    /// Encode value bits in pointer for true-inline storage.
    #[inline(always)]
    #[expect(clippy::cast_possible_truncation)]
    fn output_consume_to_raw(output: Self::Output) -> *mut u8 {
        let bits: u64 = output.to_bits();
        // XOR with magic to avoid null pointer
        let encoded: u64 = bits ^ ENCODING_MAGIC;

        StdPtr::without_provenance_mut(encoded as usize)
    }

    /// Not used for true-inline - no cleanup needed.
    ///
    /// # Safety
    ///
    /// This method should never be called for true-inline storage.
    #[inline(always)]
    unsafe fn cleanup_output_raw(_ptr: *mut u8) {
        // No-op - inline values don't have heap allocations
        debug_assert!(false, "cleanup_output_raw called on TrueInlineSlot");
    }
}

// ============================================================================
//  Send/Sync
// ============================================================================

// SAFETY: TrueInlineSlot is just a marker type with PhantomData
unsafe impl<V: InlineBits> Send for TrueInlineSlot<V> {}
unsafe impl<V: InlineBits> Sync for TrueInlineSlot<V> {}
