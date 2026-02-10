//! Leaf node storage policy system.
//!
//! This module defines the trait hierarchy that parameterizes [`LeafNode15`] over
//! its value and suffix storage strategy. Two policies are provided:
//!
//! - [`ArcPolicy<V>`]: Heap-allocated `Arc<V>` values, lazy sidecar suffix.
//!   Compact 448-byte leaves. For general-purpose use with any `V: Send + Sync`.
//!
//! - [`InlinePolicy<V>`]: True-inline `Copy` values stored as `u64` bits.
//!   Embedded suffix (896/1152-byte leaves) by default, or sidecar suffix
//!   (~576-byte leaves) with the `sidecar-suffix` feature.
//!   For small types implementing [`InlineBits`].
//!
//! All traits are **sealed** — external crates cannot implement them. The two
//! provided policies cover all supported storage modes.
//!
//! [`LeafNode15`]: crate::leaf15::LeafNode15
//! [`InlineBits`]: crate::inline::InlineBits

#![allow(dead_code)]

mod arc_values;
#[cfg(not(feature = "sidecar-suffix"))]
mod embedded_suffix;
mod inline_values;
mod sidecar_suffix;
#[cfg(test)]
mod tests;

pub use arc_values::ArcValueArray;
#[cfg(not(feature = "sidecar-suffix"))]
pub use embedded_suffix::EmbeddedSuffix;
pub use inline_values::InlineValueArray;
pub use sidecar_suffix::SidecarSuffix;

use std::marker::PhantomData;
use std::sync::Arc;

use seize::{Guard, LocalGuard};

use crate::inline::bits::InlineBits;
use crate::prefetch::prefetch_read;

// ============================================================================
//  Sealed Module
// ============================================================================

mod sealed {
    use crate::inline::bits::InlineBits;

    use super::{ArcPolicy, ArcValueArray, InlinePolicy, InlineValueArray, SidecarSuffix};

    #[cfg(not(feature = "sidecar-suffix"))]
    use super::EmbeddedSuffix;

    /// Prevents external crates from implementing policy traits.
    pub trait Sealed {}

    impl<V: Send + Sync + 'static> Sealed for ArcPolicy<V> {}
    impl<V: InlineBits> Sealed for InlinePolicy<V> {}

    /// Prevents external crates from implementing `ValueArray`.
    pub trait ValueArraySealed {}

    impl<V: Send + Sync + 'static> ValueArraySealed for ArcValueArray<V> {}
    impl<V: InlineBits> ValueArraySealed for InlineValueArray<V> {}

    /// Prevents external crates from implementing `SuffixStore`.
    pub trait SuffixStoreSealed {}

    impl SuffixStoreSealed for SidecarSuffix {}

    #[cfg(not(feature = "sidecar-suffix"))]
    impl SuffixStoreSealed for EmbeddedSuffix {}

    /// Prevents external crates from implementing `RefPolicy`.
    pub trait RefPolicySealed {}

    impl<V: Send + Sync + 'static> RefPolicySealed for ArcPolicy<V> {}
}

// ============================================================================
//  RetireHandle
// ============================================================================

/// Opaque handle for deferred value retirement.
///
/// Produced by [`ValueArray::update_in_place()`] and consumed by
/// [`LeafPolicy::retire_handle()`]. This ensures the old value is captured
/// **before** the slot is overwritten, preventing the retire-new-value bug.
///
/// # Variants
///
/// - `Ptr(*mut u8)`: A raw pointer to an old Arc allocation that needs EBR
///   retirement. Only produced by `ArcValueArray`.
/// - `Noop`: No retirement needed. Always produced by `InlineValueArray`
///   (inline values are `Copy`, no heap allocation to free).
#[derive(Debug, PartialEq, Eq)]
pub enum RetireHandle {
    /// A raw pointer to an old Arc allocation that needs EBR retirement.
    Ptr(*mut u8),

    /// No retirement needed (inline values are Copy).
    Noop,
}

// SAFETY: RetireHandle::Ptr contains a pointer that was obtained from Arc::into_raw
// and will be consumed exactly once by retire_handle(). The pointer is not
// dereferenced outside of the retirement path which runs under EBR protection.
unsafe impl Send for RetireHandle {}

// ============================================================================
//  SlotKind / SlotState — Slot Classification Enums
// ============================================================================

/// Classification of a leaf slot's contents with value extraction.
///
/// Used by scan paths where the output is always consumed. For Arc mode,
/// constructing the `Value` variant increments the Arc refcount.
///
/// # Usage
///
/// ```ignore
/// match leaf.classify_slot(slot) {
///     SlotKind::Empty => { /* retry or skip */ }
///     SlotKind::Layer(ptr) => { /* descend into sublayer */ }
///     SlotKind::Value(output) => { /* consume output */ }
/// }
/// ```
pub enum SlotKind<O> {
    /// Slot is empty (not yet assigned or cleared).
    Empty,

    /// Slot contains a terminal value.
    Value(O),

    /// Slot contains a layer pointer to a sublayer root.
    Layer(*mut u8),
}

/// Lightweight slot classification without value extraction.
///
/// Used by search paths that need to branch on slot state but only
/// extract the value conditionally (e.g., after version validation).
/// This avoids forcing an Arc clone in hot paths where only the
/// branch decision is needed.
///
/// # Usage
///
/// ```ignore
/// match leaf.classify_slot_light(slot) {
///     SlotState::Empty => { /* retry */ }
///     SlotState::Layer(ptr) => { /* descend */ }
///     SlotState::Value => {
///         // Only extract after OCC validation
///         let output = leaf.load_value(slot).unwrap();
///     }
/// }
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SlotState {
    /// Slot is empty.
    Empty,

    /// Slot contains a terminal value (not extracted).
    Value,

    /// Slot contains a layer pointer.
    Layer(*mut u8),
}

// ============================================================================
//  ValueArray<O> — Per-Slot Value Storage Trait
// ============================================================================

/// Abstraction over how terminal values and layer pointers are stored in leaf slots.
///
/// # Implementations
///
/// - [`ArcValueArray<V>`]: `[AtomicPtr<u8>; 15]` storing `Arc<V>` raw pointers.
/// - [`InlineValueArray<V>`]: `[AtomicPtr<u8>; 15]` (state tag) + `[AtomicU64; 15]` (bits).
///
/// # Slot State
///
/// Every slot is in one of three states: **Empty**, **Value**, or **Layer**.
/// The implementation determines how these states are encoded physically.
///
/// # Ordering Contract
///
/// - `load` / `load_layer`: Use `READ_ORD` (Acquire). Caller validates version afterward.
/// - `store` / `store_layer`: Use `WRITE_ORD` (Release). Caller holds lock.
/// - `store_relaxed`: Use `RELAXED`. Caller will publish via permutation CAS.
/// - `move_slot`: Destination uses `WRITE_ORD`. Called during split under lock.
///   Caller must `clear()` the source slot after the move.
///
/// # Safety
///
/// This trait is **sealed** — only `ArcValueArray` and `InlineValueArray` may
/// implement it. Implementations must maintain the ordering contract above.
pub trait ValueArray<O>: sealed::ValueArraySealed + Send + Sync + Sized + 'static {
    /// Create a new array with all slots empty.
    fn new() -> Self;

    // ========================================================================
    //  Slot Classification
    // ========================================================================

    /// Check if a slot is empty (no value, no layer).
    fn is_empty(&self, slot: usize) -> bool;

    /// Check if a slot contains a layer pointer.
    ///
    /// NOTE: The authoritative layer check is `keylenx >= LAYER_KEYLENX`,
    /// which lives on the leaf. This method checks the value storage state
    /// for consistency. The leaf's `is_layer()` method should be preferred
    /// in most contexts; this exists for the value array's internal use
    /// and for the `Drop` path where keylenx is also consulted.
    fn is_layer(&self, slot: usize) -> bool;

    // ========================================================================
    //  Terminal Value Operations
    // ========================================================================

    /// Load the terminal value at a slot.
    ///
    /// Returns `None` if the slot is empty.
    /// Returns `Some(output)` if the slot contains a terminal value.
    ///
    /// # Prechecked Contract
    ///
    /// Caller **must** verify the slot is not a layer via `keylenx < LAYER_KEYLENX`
    /// before calling. This is a **prechecked contract** — `load()` does NOT
    /// check layer status itself. The caller is responsible for the check.
    ///
    /// For `ArcValueArray`, calling `load()` on a layer slot would perform
    /// `Arc::increment_strong_count` on a non-Arc pointer, which is **UB**.
    /// For `InlineValueArray`, layer slots return `None` (safe but misleading).
    ///
    /// For `ArcValueArray`: increments refcount and returns `Arc<V>`.
    /// For `InlineValueArray`: copies the `u64` bits and reconstructs `V`.
    fn load(&self, slot: usize) -> Option<O>;

    /// Store a terminal value at a slot.
    ///
    /// Uses `WRITE_ORD` (Release). Caller must hold the leaf lock.
    fn store(&self, slot: usize, output: &O);

    /// Store a terminal value with Relaxed ordering.
    ///
    /// Used during new slot assignment where the permutation CAS is the
    /// linearization point. The CAS provides the necessary ordering.
    fn store_relaxed(&self, slot: usize, output: &O);

    /// Update an existing terminal value in place, returning a retire handle
    /// for the old value.
    ///
    /// The old value **must** be captured before the store to ensure correct
    /// retirement — reading the slot after overwrite would return the new value.
    ///
    /// For `ArcValueArray`: loads old raw pointer, stores new pointer, returns
    /// `RetireHandle::Ptr(old_ptr)`.
    /// For `InlineValueArray`: overwrites bits, returns `RetireHandle::Noop`.
    ///
    /// Caller must pass the returned handle to [`LeafPolicy::retire_handle()`].
    fn update_in_place(&self, slot: usize, output: &O) -> RetireHandle;

    /// Take the terminal value, leaving the slot empty.
    ///
    /// Returns `None` if the slot was empty or a layer.
    /// Uses `RELAXED` for the swap (caller holds lock).
    fn take(&self, slot: usize) -> Option<O>;

    // ========================================================================
    //  Layer Pointer Operations
    // ========================================================================

    /// Load the raw pointer at a slot without any typed interpretation.
    ///
    /// For `ArcValueArray`: returns the raw `*mut u8` from the `AtomicPtr`.
    /// For `InlineValueArray`: returns the tag pointer (null/sentinel/layer).
    ///
    /// Used for low-level inspection and prefetch paths.
    fn load_raw(&self, slot: usize) -> *mut u8;

    /// Load the layer pointer at a slot.
    ///
    /// Caller must have verified `keylenx >= LAYER_KEYLENX` first.
    /// Returns the raw pointer to the sublayer root node.
    fn load_layer(&self, slot: usize) -> *mut u8;

    /// Store a layer pointer at a slot.
    ///
    /// Caller must also set `keylenx >= LAYER_KEYLENX`.
    fn store_layer(&self, slot: usize, ptr: *mut u8);

    // ========================================================================
    //  Slot Management
    // ========================================================================

    /// Clear a slot (set to empty state).
    fn clear(&self, slot: usize);

    /// Move a slot's contents from `self[src_slot]` to `dst[dst_slot]`.
    ///
    /// Ownership transfer semantics: after this call, `dst[dst_slot]` owns
    /// the value and `self[src_slot]` is in an indeterminate state. The caller
    /// **must** call `self.clear(src_slot)` after the move to reset the source.
    ///
    /// For Arc: loads raw pointer from source, stores to destination (pointer
    /// move, no refcount change).
    /// For inline: loads tag+bits from source, stores tag+bits to destination
    /// (bitwise move).
    /// For layer pointers: loads pointer from source, stores to destination.
    ///
    /// Uses `WRITE_ORD` for destination store. Called during split under lock.
    fn move_slot(&self, dst: &Self, src_slot: usize, dst_slot: usize);

    // ========================================================================
    //  Lifecycle
    // ========================================================================

    /// Clean up a slot's value during leaf Drop.
    ///
    /// For `ArcValueArray`: decrements refcount (`Arc::from_raw` + drop).
    /// For `InlineValueArray`: no-op.
    ///
    /// # Safety
    ///
    /// - Caller has exclusive access (`&mut` via Drop).
    /// - Slot must contain a terminal value (not empty, not layer).
    unsafe fn cleanup(&self, slot: usize);
}

// ============================================================================
//  SuffixStore — Key Suffix Storage Trait
// ============================================================================

/// Abstraction over how key suffixes are stored for a leaf node.
///
/// # Implementations
///
/// - [`SidecarSuffix`]: Lazy-allocated `AtomicPtr<SuffixSidecar>`. Compact leaves
///   (8 bytes in leaf struct). One pointer dereference for suffix access.
///   Used by [`ArcPolicy`] and optionally by [`InlinePolicy`] (`sidecar-suffix` feature).
/// - `EmbeddedSuffix` (default for [`InlinePolicy`]): `InlineSuffixBag` + `AtomicPtr<SuffixBag>`
///   embedded directly in the leaf. Zero-indirection suffix reads. Larger leaves.
///
/// # Ordering Contract
///
/// - Read methods (`get`, `suffix_equals`, `suffix_compare`): safe for concurrent
///   readers under OCC. The suffix bytes are immutable after publication.
/// - Write methods (`assign`, `clear`): caller must hold the leaf lock.
///
/// # Safety
///
/// This trait is **sealed** — only the provided implementations may implement it.
pub trait SuffixStore: sealed::SuffixStoreSealed + Send + Sync + Sized + 'static {
    /// Create a new empty suffix store.
    fn new() -> Self;

    // ========================================================================
    //  Read Operations (OCC-safe, no lock required)
    // ========================================================================

    /// Get the suffix bytes for a slot.
    ///
    /// Returns `None` if the slot has no suffix stored in this store.
    /// Checks inline storage first, then external overflow.
    fn get(&self, slot: usize) -> Option<&[u8]>;

    /// Check if a slot's suffix equals the given bytes.
    ///
    /// Returns `false` if the slot has no suffix.
    fn suffix_equals(&self, slot: usize, suffix: &[u8]) -> bool;

    /// Compare a slot's suffix with the given bytes.
    ///
    /// Returns `None` if the slot has no suffix.
    fn suffix_compare(&self, slot: usize, suffix: &[u8]) -> Option<std::cmp::Ordering>;

    /// Check if external (overflow) storage has been allocated.
    fn has_external(&self) -> bool;

    // ========================================================================
    //  Write Operations (lock required)
    // ========================================================================

    /// Assign a suffix to a slot.
    ///
    /// Tries inline first, then external in-place, then falls back to
    /// drain-and-rebuild. Returns old external bag pointer for deferred
    /// retirement (null if no retirement needed).
    ///
    /// # Safety
    ///
    /// - Caller must hold the leaf lock.
    /// - `guard` must come from this tree's collector.
    unsafe fn assign(
        &self,
        slot: usize,
        suffix: &[u8],
        perm: &impl crate::TreePermutation,
        guard: &LocalGuard<'_>,
    ) -> *mut u8;

    /// Assign a suffix during node initialization (sequential slots 0..slot).
    ///
    /// Unlike `assign`, this does not consult the permutation. Used during
    /// splits where slots are filled sequentially.
    ///
    /// # Safety
    ///
    /// Same as `assign`.
    unsafe fn assign_init(&self, slot: usize, suffix: &[u8], guard: &LocalGuard<'_>);

    /// Assign a suffix using a pre-allocated buffer (reduces critical section time).
    ///
    /// # Safety
    ///
    /// Same as `assign`.
    unsafe fn assign_prealloc(
        &self,
        slot: usize,
        suffix: &[u8],
        perm: &impl crate::TreePermutation,
        guard: &LocalGuard<'_>,
        prealloc: Vec<u8>,
    ) -> *mut u8;

    /// Pre-allocate external overflow storage.
    ///
    /// # Safety
    ///
    /// Caller must hold the leaf lock.
    unsafe fn ensure_external(&self) -> *mut crate::suffix::SuffixBag;

    /// Clear a slot's suffix.
    ///
    /// # Safety
    ///
    /// Caller must hold the leaf lock.
    unsafe fn clear(&self, slot: usize, guard: &LocalGuard<'_>);

    /// Retire a suffix bag pointer returned by `assign`.
    ///
    /// # Safety
    ///
    /// `ptr` must have come from `assign` on this suffix store type.
    unsafe fn retire_bag_ptr(ptr: *mut u8, guard: &LocalGuard<'_>);

    // ========================================================================
    //  Lifecycle
    // ========================================================================

    /// Drop all owned heap allocations (external bag, sidecar).
    ///
    /// Called from the leaf's `Drop` impl with `&mut self` access.
    ///
    /// # Safety
    ///
    /// Caller has exclusive access (no concurrent readers).
    unsafe fn drop_storage(&mut self);

    /// Initialize suffix storage at a zeroed memory location.
    ///
    /// Called by `LeafNode15::init_at()` after `write_bytes(ptr, 0, 1)`.
    /// For `SidecarSuffix`: no-op (null pointer is correct zero state).
    ///
    /// # Safety
    ///
    /// `ptr` must point to valid, zeroed, properly-aligned memory.
    unsafe fn init_at_zero(ptr: *mut Self);
}

// ============================================================================
//  LeafPolicy — Combined Configuration Trait
// ============================================================================

/// Bundles value storage, suffix storage, and type conversion for a leaf node.
///
/// This is the single type parameter on `LeafNode15<P: LeafPolicy>`.
/// Two canonical implementations are provided:
///
/// - [`ArcPolicy<V>`]: Heap-allocated `Arc<V>` values, sidecar suffix, 448-byte leaves.
/// - [`InlinePolicy<V>`]: Inline `Copy` values, embedded suffix (default) or
///   sidecar suffix (`sidecar-suffix` feature).
///
/// # Design
///
/// `LeafPolicy` intentionally bundles value and suffix strategies into one trait
/// rather than parameterizing independently. `ArcPolicy` uses `SidecarSuffix`;
/// `InlinePolicy` uses `EmbeddedSuffix` by default (or `SidecarSuffix` with
/// the `sidecar-suffix` feature). The value array type differs:
/// `ArcValueArray` (120 bytes) vs `InlineValueArray` (240 bytes). Keeping one
/// type parameter simplifies the tree's generic signature from `<S, L, A>` to
/// `<P, A>`.
///
/// # Safety
///
/// This trait is **sealed** — only `ArcPolicy` and `InlinePolicy` may implement it.
pub trait LeafPolicy: sealed::Sealed + Send + Sync + Sized + 'static {
    /// The user-facing value type provided to `insert()`.
    type Value: Send + Sync;

    /// The internal output type returned from `get()` and carried across retries.
    ///
    /// - `ArcPolicy<V>`: `Arc<V>` (cheap clone via refcount increment)
    /// - `InlinePolicy<V>`: `V` (bitwise copy)
    type Output: Clone + Send + Sync;

    /// The value storage array type embedded in the leaf.
    ///
    /// - `ArcPolicy<V>`: `ArcValueArray<V>` wrapping `[AtomicPtr<u8>; 15]`
    /// - `InlinePolicy<V>`: `InlineValueArray<V>` wrapping sentinel + bits arrays
    type Values: ValueArray<Self::Output>;

    /// The suffix storage type embedded in the leaf.
    ///
    /// - `ArcPolicy`: `SidecarSuffix` (8-byte `AtomicPtr<SuffixSidecar>`)
    /// - `InlinePolicy`: `EmbeddedSuffix` (default) or `SidecarSuffix` (`sidecar-suffix` feature)
    type Suffix: SuffixStore;

    /// Whether old value pointers require EBR deferred retirement.
    ///
    /// - `true`: `Arc` mode. Old pointers passed to `guard.defer_retire()`.
    /// - `false`: Inline mode. Values are `Copy`, no heap allocation to free.
    const NEEDS_RETIREMENT: bool;

    /// Whether this policy supports zero-copy reference returns.
    ///
    /// - `true`: Arc mode. Values live at stable addresses behind the pointer.
    /// - `false`: Inline mode. Values are encoded as bits, no stable address.
    const SUPPORTS_REF: bool;

    // ========================================================================
    //  Slot Queries
    // ========================================================================

    /// Check if a slot contains no value (empty or layer).
    ///
    /// Alias for checking slot emptiness from the policy level.
    /// Delegates to the value array's `is_empty()`.
    fn is_value_empty(values: &Self::Values, slot: usize) -> bool;

    /// Prefetch the value at a slot for upcoming access.
    ///
    /// For Arc: prefetches the pointer target into L1 cache.
    /// For inline: no-op (bits are already in the leaf cache line).
    fn prefetch_value(values: &Self::Values, slot: usize);

    /// Load a raw pointer to the value at a slot.
    ///
    /// Used by reference-returning paths that need to defer the
    /// pointer-to-reference conversion until after OCC validation.
    ///
    /// Returns null if the slot is empty.
    ///
    /// # Safety
    ///
    /// Caller must hold a valid guard and validate OCC version after use.
    unsafe fn load_value_ptr(values: &Self::Values, slot: usize) -> *const Self::Value;

    /// Take the value at a slot when converting it to a layer pointer.
    ///
    /// Returns a `RetireHandle` for the displaced value. The caller
    /// is responsible for retiring the handle after installing the layer.
    ///
    /// For Arc: swaps out the pointer, returns `RetireHandle::Ptr`.
    /// For inline: clears the bits, returns `RetireHandle::Noop`.
    fn take_value_for_layer(values: &Self::Values, slot: usize) -> RetireHandle;

    // ========================================================================
    //  Value Conversion
    // ========================================================================

    /// Convert a user value into an output handle.
    ///
    /// Called exactly once per insert. For Arc mode, this allocates the Arc.
    /// The returned output is carried across retries (splits, layer creation).
    fn into_output(value: Self::Value) -> Self::Output;

    /// Clone an output for return to the caller.
    ///
    /// For Arc: `Arc::clone` (refcount increment).
    /// For inline: bitwise copy.
    #[inline(always)]
    fn clone_output(output: &Self::Output) -> Self::Output {
        output.clone()
    }

    // ========================================================================
    //  Retirement
    // ========================================================================

    /// Retire an old value through EBR using a handle captured before mutation.
    ///
    /// The `handle` was produced by [`ValueArray::update_in_place()`] at the
    /// moment the old value was displaced. This design guarantees we always
    /// retire the correct (old) value, not the newly installed one.
    ///
    /// For Arc: defers `Arc::from_raw` + drop via `guard.defer_retire`.
    /// For inline: no-op (handle is always `RetireHandle::Noop`).
    ///
    /// # Safety
    ///
    /// The handle must have been produced by a prior `update_in_place()` call
    /// on the same leaf, and must not have been retired already.
    unsafe fn retire_handle(handle: RetireHandle, guard: &LocalGuard<'_>);

    /// Retire a taken value through EBR.
    ///
    /// Used when `take()` returns `Some(output)` and the value needs deferred
    /// cleanup (e.g., during remove or layer-creation). The caller owns the
    /// `Output` value and passes it here for deferred drop.
    ///
    /// For Arc: converts `Arc<V>` back to raw pointer, defers drop.
    /// For inline: no-op.
    ///
    /// # Safety
    ///
    /// The output must no longer be accessible to concurrent readers after
    /// the current epoch ends.
    unsafe fn retire_output(output: Self::Output, guard: &LocalGuard<'_>);

    // ========================================================================
    //  Reference Access
    // ========================================================================

    /// Load a reference to the value at a slot, without cloning.
    ///
    /// Returns `None` if the slot is empty or a layer.
    /// Only valid when `SUPPORTS_REF == true`.
    ///
    /// # Safety
    ///
    /// - Caller must hold a valid guard preventing deallocation.
    /// - Caller must validate the OCC version after reading.
    /// - The returned reference is valid only as long as the guard is held.
    unsafe fn load_value_ref<'g>(values: &Self::Values, slot: usize) -> Option<&'g Self::Value>;
}

// ============================================================================
//  RefPolicy — Marker Trait for Reference-Returning APIs
// ============================================================================

/// Marker trait for policies that support zero-copy reference returns.
///
/// Only [`ArcPolicy`] implements this trait. [`InlinePolicy`] does not,
/// because inline values have no stable memory address to reference.
///
/// # API Gating
///
/// APIs that return `&V` (like `get_ref`, `scan_ref`, ref iterators) require
/// `P: RefPolicy` as a trait bound, preventing compilation for inline mode.
///
/// ```ignore
/// impl<P, A> MassTreeGeneric<P, A>
/// where
///     P: LeafPolicy + RefPolicy,
///     A: TreeAllocator<P>,
/// {
///     pub fn get_ref<'g>(&self, key: &[u8], guard: &'g LocalGuard<'_>) -> Option<&'g P::Value> {
///         // ...
///     }
/// }
/// ```
pub trait RefPolicy: sealed::RefPolicySealed + LeafPolicy {
    /// Borrow the underlying value from an output handle.
    ///
    /// For `ArcPolicy<V>`: returns `&V` via `Deref`.
    fn output_as_ref(output: &Self::Output) -> &Self::Value;
}

// ============================================================================
//  ArcPolicy<V> — Heap-Allocated Arc Storage
// ============================================================================

/// Arc-based value storage with lazy sidecar suffix.
///
/// Produces compact 448-byte leaves. Values are `Arc<V>` raw pointers stored
/// in `[AtomicPtr<u8>; 15]`. Old values require EBR deferred retirement.
///
/// # When to Use
///
/// - Values are large or non-Copy (e.g., `String`, `Vec<u8>`, structs)
/// - Workloads are primarily integer-key (few/no suffixes)
/// - Memory efficiency matters (small leaf footprint)
///
/// # Type Parameters
///
/// - `V`: The user value type. Must be `Send + Sync + 'static`.
pub struct ArcPolicy<V>(PhantomData<V>);

impl<V: Send + Sync + 'static> LeafPolicy for ArcPolicy<V> {
    type Value = V;
    type Output = Arc<V>;
    type Values = ArcValueArray<V>;
    type Suffix = SidecarSuffix;

    const NEEDS_RETIREMENT: bool = true;
    const SUPPORTS_REF: bool = true;

    #[inline(always)]
    fn is_value_empty(values: &ArcValueArray<V>, slot: usize) -> bool {
        values.is_empty(slot)
    }

    #[inline(always)]
    fn prefetch_value(values: &ArcValueArray<V>, slot: usize) {
        // Prefetch the Arc<V> target into L1 cache for upcoming dereference.
        let ptr: *mut u8 = values.load_raw(slot);
        if !ptr.is_null() {
            prefetch_read(ptr.cast::<V>());
        }
    }

    #[inline(always)]
    unsafe fn load_value_ptr(values: &ArcValueArray<V>, slot: usize) -> *const V {
        // SAFETY: Caller holds a valid guard and will validate OCC version after.
        let ptr: *mut u8 = values.load_raw(slot);
        if ptr.is_null() {
            std::ptr::null()
        } else {
            ptr.cast::<V>()
        }
    }

    #[inline(always)]
    fn take_value_for_layer(values: &ArcValueArray<V>, slot: usize) -> RetireHandle {
        // Load the raw pointer before clearing the slot.
        // This is called under a lock, so Relaxed loads are safe.
        let ptr: *mut u8 = values.load_raw(slot);
        if ptr.is_null() {
            return RetireHandle::Noop;
        }
        // Clear the slot (sets to null).
        values.clear(slot);
        RetireHandle::Ptr(ptr)
    }

    #[inline(always)]
    fn into_output(value: V) -> Arc<V> {
        Arc::new(value)
    }

    #[inline(always)]
    unsafe fn retire_handle(handle: RetireHandle, guard: &LocalGuard<'_>) {
        if let RetireHandle::Ptr(ptr) = handle {
            debug_assert!(!ptr.is_null());
            // SAFETY: ptr was obtained from Arc::into_raw in update_in_place().
            // The guard ensures no readers can still observe the old value
            // after the current epoch completes.
            unsafe {
                guard.defer_retire(ptr.cast::<V>(), |ptr, _| {
                    drop(Arc::from_raw(ptr));
                });
            }
        }
    }

    #[inline(always)]
    unsafe fn retire_output(output: Arc<V>, guard: &LocalGuard<'_>) {
        let ptr: *mut V = Arc::into_raw(output).cast_mut();

        // SAFETY: ptr is a valid Arc<V> pointer. The guard ensures no readers
        // can observe this value after the current epoch completes.
        unsafe {
            guard.defer_retire(ptr, |ptr, _| {
                drop(Arc::from_raw(ptr));
            });
        }
    }

    #[inline(always)]
    unsafe fn load_value_ref<'g>(values: &ArcValueArray<V>, slot: usize) -> Option<&'g V> {
        // SAFETY: Caller guarantees:
        // 1. A valid guard prevents retirement of this value.
        // 2. OCC version will be validated after this load.
        // 3. The slot is known to contain a value (not empty, not layer).
        let ptr: *mut u8 = values.load_raw(slot);

        if ptr.is_null() {
            None
        } else {
            // SAFETY: ptr is a valid Arc<V> raw pointer. The guard prevents
            // retirement. OCC validation ensures we read consistent data.
            Some(unsafe { &*ptr.cast::<V>() })
        }
    }
}

impl<V: Send + Sync + 'static> RefPolicy for ArcPolicy<V> {
    #[inline(always)]
    fn output_as_ref(output: &Arc<V>) -> &V {
        output
    }
}

// ============================================================================
//  InlinePolicy<V> — True-Inline Copy Storage
// ============================================================================

/// Inline value storage with embedded suffix (default) or sidecar suffix.
///
/// Default: 1152-byte leaves (18 cache lines) with embedded `InlineSuffixBag`.
/// With `small-suffix-capacity`: 896-byte leaves (14 cache lines).
/// With `sidecar-suffix` feature: ~576-byte leaves (9 cache lines).
///
/// Values of type `V: InlineBits` (≤ 8 bytes, Copy) are stored as `u64` bits
/// in `[AtomicU64; 15]`. No heap allocation for values. No retirement needed
/// for value updates.
///
/// # When to Use
///
/// - Values are small and Copy (e.g., `u64`, `i32`, `f64`, `bool`)
/// - Workloads benefit from avoiding pointer indirection
///
/// # Type Parameters
///
/// - `V`: The user value type. Must implement [`InlineBits`].
pub struct InlinePolicy<V: InlineBits>(PhantomData<V>);

impl<V: InlineBits> LeafPolicy for InlinePolicy<V> {
    type Value = V;
    type Output = V;
    type Values = InlineValueArray<V>;

    #[cfg(not(feature = "sidecar-suffix"))]
    type Suffix = EmbeddedSuffix;
    #[cfg(feature = "sidecar-suffix")]
    type Suffix = SidecarSuffix;

    const NEEDS_RETIREMENT: bool = false;
    const SUPPORTS_REF: bool = false;

    #[inline(always)]
    fn is_value_empty(values: &InlineValueArray<V>, slot: usize) -> bool {
        values.is_empty(slot)
    }

    #[inline(always)]
    fn prefetch_value(_values: &InlineValueArray<V>, _slot: usize) {
        // No-op: inline values are stored directly in the leaf cache line.
        // No pointer indirection to prefetch.
    }

    #[inline(always)]
    unsafe fn load_value_ptr(_values: &InlineValueArray<V>, _slot: usize) -> *const V {
        // Inline values have no stable address. This is unreachable at compile
        // time because SUPPORTS_REF == false prevents ref-returning paths.
        unreachable!("inline values do not support pointer-based access")
    }

    #[inline(always)]
    fn take_value_for_layer(values: &InlineValueArray<V>, slot: usize) -> RetireHandle {
        // Clear the slot. No retirement needed for inline Copy values.
        values.clear(slot);
        RetireHandle::Noop
    }

    #[inline(always)]
    fn into_output(value: V) -> V {
        value
    }

    #[inline(always)]
    unsafe fn retire_handle(_handle: RetireHandle, _guard: &LocalGuard<'_>) {
        // No-op: inline values are Copy, no heap allocation to free.
        // handle is always RetireHandle::Noop for this policy.
    }

    #[inline(always)]
    unsafe fn retire_output(_output: V, _guard: &LocalGuard<'_>) {
        // No-op: inline values are Copy, no heap allocation to free.
    }

    #[inline(always)]
    unsafe fn load_value_ref<'g>(_values: &InlineValueArray<V>, _slot: usize) -> Option<&'g V> {
        // Inline values have no stable address. This is unreachable at compile
        // time because RefPolicy is not implemented for InlinePolicy.
        unreachable!("inline values do not support reference returns")
    }
}

// NOTE: InlinePolicy<V> intentionally does NOT implement RefPolicy.
// This prevents get_ref/scan_ref/ref iterators from compiling for inline mode.
