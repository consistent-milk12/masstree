//! Marker trait for value slots that support reference returns.
//!
//! Only pointer-backed storage modes (Arc, Box) can return `&V` references.
//! True-inline storage cannot, as values are stored as bits in atomic integers.

use crate::slot::ValueSlot;
use crate::value::LeafValue;

// ============================================================================
//  RefValueSlot Marker Trait
// ============================================================================

/// Marker trait for value slots that can return references to stored values.
///
/// Only pointer-backed storage modes implement this trait. True-inline storage
/// cannot return references because values are not stored at stable addresses.
///
/// # API Gating
///
/// APIs that return `&V` (like `get_ref`, `scan_ref`, ref iterators) require
/// `S: RefValueSlot` as a trait bound, preventing compilation for true-inline.
///
/// # Example
///
/// ```ignore
/// impl<S, L, A> MassTreeGeneric<S, L, A>
/// where
///     S: ValueSlot + RefValueSlot,  // Only for pointer-backed storage
///     // ...
/// {
///     pub fn get_ref<'g>(&self, key: &[u8], guard: &'g LocalGuard<'_>) -> Option<&'g S::Value> {
///         // ... implementation ...
///     }
/// }
/// ```
pub trait RefValueSlot: ValueSlot {}

// ============================================================================
//  Implementations for Pointer-Backed Storage
// ============================================================================

/// Arc-based storage supports references (value stored at Arc's heap allocation).
impl<V> RefValueSlot for LeafValue<V> {}

// NOTE: TrueInlineSlot<V> intentionally does NOT implement RefValueSlot.
// This is what prevents get_ref/scan_ref/ref iterators from compiling for true-inline.
