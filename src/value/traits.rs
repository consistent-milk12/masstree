//! Leaf value operation traits for storage-agnostic value access.
//!
//! These traits enable the tree to manipulate values without knowing whether
//! they're stored as pointers ([`Arc`]/[`Box`]) or inline bits.
//!
//! # Design Principle
//! The leaf owns representation, the tree owns concurrency. Value storage details
//! (pointer vs inline) are encapsulated in these leaf-provided methods.

use seize::LocalGuard;

use crate::slot::ValueSlot;

// ============================================================================
//  LeafValueLoad - Read terminal value from slot
// ============================================================================

/// Trait for loading terminal values from leaf slots.
///
/// This is the read-path interface. Readers call this after version validation
/// to obtain a cloned/copied output without knowing the storage representation.
pub trait LeafValueLoad<S: ValueSlot> {
    /// Load the terminal value output from `slot`.
    ///
    /// Returns `Some(output)` if the slot contains a terminal value.
    /// Returns `None` if the slot is empty or contains a layer pointer.
    ///
    /// # Ordering
    /// Caller must have validated version before calling.
    /// Implementation uses appropriate load ordering for the storage mode.
    fn try_load_output(&self, slot: usize) -> Option<S::Output>;
}

// ============================================================================
//  LeafValueStore - Store terminal value into slot
// ============================================================================

/// Trait for storing terminal values into leaf slots.
///
/// This is used during insert to initialize a new slot with a value.
pub trait LeafValueStore<S: ValueSlot> {
    /// Store a terminal value output into `slot`.
    ///
    /// # Preconditions
    /// - Caller holds the leaf lock
    /// - Slot is being initialized (empty -> terminal value)
    ///
    /// # Ordering
    /// Implementation ensures value is visible before the caller
    /// publishes the slot in the permutation.
    fn store_value_output(&self, slot: usize, output: &S::Output, guard: &LocalGuard<'_>);
}

// ============================================================================
//  LeafValueUpdate - Replace terminal value in slot
// ============================================================================

/// Trait for replacing terminal values in leaf slots.
///
/// This is used during update to swap an existing value with a new one.
pub trait LeafValueUpdate<S: ValueSlot> {
    /// Replace the terminal value at `slot`, returning the old output.
    ///
    /// # Preconditions
    ///
    /// - Slot contains a terminal value (not empty, not layer)
    /// - Caller holds the leaf lock
    ///
    /// # Implementation
    ///
    /// - Pointer-backed: swap pointers, retire old (if [`NEEDS_RETIREMENT`])
    /// - True-inline: swap inline bits, no retirement
    fn replace_value_output(
        &self,
        slot: usize,
        new_output: S::Output,
        guard: &LocalGuard<'_>,
    ) -> S::Output;
}

// ============================================================================
//  LeafValueClear - Clear terminal value from slot
// ============================================================================

/// Trait for clearing terminal values from leaf slots.
///
/// This is used during remove to clear a slot's value.
pub trait LeafValueClear<S: ValueSlot> {
    /// Clear the terminal value at `slot`.
    ///
    /// # Preconditions
    ///
    /// - Caller holds the leaf lock
    ///
    /// # Behavior
    ///
    /// - Sets slot to empty state (null pointer)
    /// - Retires old value if pointer-backed (under [`NEEDS_RETIREMENT`] guard)
    fn clear_value_output(&self, slot: usize, guard: &LocalGuard<'_>);
}

// ============================================================================
//  LeafValueTake - Extract and clear terminal value from slot
// ============================================================================

/// Trait for extracting terminal values from leaf slots.
///
/// This is used during layer creation (suffix conflict) to move a value
/// before installing a layer pointer.
pub trait LeafValueTake<S: ValueSlot> {
    /// Extract and clear the terminal value at `slot`, returning the old output.
    ///
    /// # Preconditions
    ///
    /// - Slot contains a terminal value (not empty, not layer)
    /// - Caller holds the leaf lock
    ///
    /// # Behavior
    ///
    /// - Returns the value that was in the slot
    /// - Sets slot to empty state (null pointer)
    /// - Retires old value if pointer-backed (under [`NEEDS_RETIREMENT`] guard)
    ///
    /// Note: The caller will install a layer pointer after this returns.
    fn take_value_output(&self, slot: usize, guard: &LocalGuard<'_>) -> Option<S::Output>;
}

// ============================================================================
//  Composite Trait for Convenience
// ============================================================================

/// Composite trait for all leaf value operations.
///
/// Types implementing this trait support full value lifecycle:
/// load, store, update, clear, take.
pub trait LeafValueOps<S: ValueSlot>:
    LeafValueLoad<S> + LeafValueStore<S> + LeafValueUpdate<S> + LeafValueClear<S> + LeafValueTake<S>
{
}

// Blanket impl
impl<T, S> LeafValueOps<S> for T
where
    S: ValueSlot,
    T: LeafValueLoad<S>
        + LeafValueStore<S>
        + LeafValueUpdate<S>
        + LeafValueClear<S>
        + LeafValueTake<S>,
{
}
