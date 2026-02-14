//! True-inline value array implementation.
//!
//! Stores `V: InlineBits` values directly as `u64` bits in `[AtomicU64; 15]`.
//! Uses `InlineSentinel` pointer in a parallel `[AtomicPtr<u8>; 15]` tag array
//! to distinguish slot states.
//!
//! This is the value storage backend for [`InlinePolicy<V>`].
//!
//! # Slot State Encoding
//!
//! ```text
//! tags[slot] == null                → Empty
//! tags[slot] == INLINE_SENTINEL     → Terminal value, bits in values[slot]
//! tags[slot] == other non-null      → Layer pointer
//! ```
//!
//! The sentinel is a pointer to a `static u8`, ensuring strict provenance.
//! No XOR encoding. No fake pointers. Values are stored as raw `u64` bits.
//!
//! [`InlinePolicy<V>`]: super::InlinePolicy

use std::marker::PhantomData;
use std::ptr as StdPtr;
use std::sync::atomic::{AtomicPtr, AtomicU64};

use crate::inline::bits::InlineBits;
use crate::inline::sentinel::InlineSentinel;
use crate::leaf15::WIDTH_15;
use crate::ordering::{READ_ORD, RELAXED, WRITE_ORD};

use super::RetireHandle;
use super::ValueArray;

// ============================================================================
//  InlineValueArray<V>
// ============================================================================

/// Value array with dual storage: state tags + inline bits.
///
/// # Layout
///
/// ```text
/// InlineValueArray<V> {
///     tags:   [AtomicPtr<u8>; 15]  // 120 bytes — slot state discriminator
///     values: [AtomicU64; 15]      // 120 bytes — inline value bits
///     _marker: PhantomData<V>      // 0 bytes
/// }
/// Total: 240 bytes
/// ```
///
/// This is 120 bytes more than `BoxValueArray<V>` (120 bytes), which is the
/// cost of inline value storage. The tradeoff: no heap allocation per insert,
/// no retirement needed for value updates, no pointer indirection for reads.
///
/// # Sentinel
///
/// The sentinel pointer (`InlineSentinel::inline_sentinel_ptr()`) is a real
/// `&'static u8` reference, ensuring strict provenance compliance. It is
/// never dereferenced for payload — its sole purpose is to distinguish
/// terminal value slots from empty slots and layer pointer slots.
#[repr(C)]
pub struct InlineValueArray<V: InlineBits> {
    /// State discriminator tags.
    ///
    /// - `null` → empty slot
    /// - `INLINE_SENTINEL_PTR` → terminal value (bits in `values[slot]`)
    /// - other non-null → layer pointer
    tags: [AtomicPtr<u8>; WIDTH_15],

    /// Inline value bits storage.
    ///
    /// Only meaningful when `tags[slot] == INLINE_SENTINEL_PTR`.
    /// Contains `V::to_bits()` representation.
    values: [AtomicU64; WIDTH_15],

    _marker: PhantomData<V>,
}

// SAFETY: InlineValueArray is Send+Sync when V: Send+Sync (which InlineBits requires).
// The AtomicPtr and AtomicU64 provide thread-safe access. Layer pointers stored
// in tags are protected by the tree's concurrency protocol (OCC + locks).
unsafe impl<V: InlineBits> Send for InlineValueArray<V> {}
unsafe impl<V: InlineBits> Sync for InlineValueArray<V> {}

impl<V: InlineBits> ValueArray<V> for InlineValueArray<V> {
    #[inline(always)]
    fn new() -> Self {
        // SAFETY: InlineValueArray is #[repr(C)]. All fields are zero-safe:
        // - tags: [AtomicPtr<u8>; 15] — null pointers are all-zero-bits
        // - values: [AtomicU64; 15] — zero is all-zero-bits
        // - _marker: PhantomData<V> — ZST, no bytes
        // AtomicPtr/AtomicU64 are #[repr(C)] wrapping UnsafeCell; 0/null is 0.
        unsafe { std::mem::zeroed() }
    }

    // ========================================================================
    //  Slot Classification
    // ========================================================================

    #[inline(always)]
    fn is_empty(&self, slot: usize) -> bool {
        debug_assert!(slot < WIDTH_15, "is_empty: slot {slot} out of bounds");
        self.tags[slot].load(READ_ORD).is_null()
    }

    #[inline(always)]
    fn is_layer(&self, slot: usize) -> bool {
        debug_assert!(slot < WIDTH_15, "is_layer: slot {slot} out of bounds");

        let tag: *mut u8 = self.tags[slot].load(READ_ORD);
        // Layer: non-null and NOT the sentinel
        !tag.is_null() && !InlineSentinel::is_inline_sentinel(tag)
    }

    // ========================================================================
    //  Terminal Value Operations
    // ========================================================================

    #[inline(always)]
    fn load(&self, slot: usize) -> Option<V> {
        debug_assert!(slot < WIDTH_15, "load: slot {slot} out of bounds");

        let tag: *mut u8 = self.tags[slot].load(READ_ORD);

        if InlineSentinel::is_inline_sentinel(tag) {
            // Relaxed is sufficient: the Acquire on the tag above synchronizes
            // with the writer's Release on the tag store. Since the writer
            // stores bits before the tag (with Release), the bits are guaranteed
            // visible once we've Acquired the tag. For in-place updates (bits
            // change, tag stays sentinel), the OCC version Acquire in the caller
            // provides the necessary synchronization.
            let bits: u64 = self.values[slot].load(RELAXED);
            Some(V::from_bits(bits))
        } else {
            // Empty (null) or layer pointer — not a terminal value.
            None
        }
    }

    #[inline(always)]
    fn store(&self, slot: usize, output: &V) {
        debug_assert!(slot < WIDTH_15, "store: slot {slot} out of bounds");

        // Store bits first (Relaxed), then sentinel tag (Release = publication).
        // The Release on the tag guarantees that any reader who Acquires the
        // sentinel will see the prior Relaxed bits store.
        self.values[slot].store(output.to_bits(), RELAXED);
        self.tags[slot].store(InlineSentinel::inline_sentinel_ptr(), WRITE_ORD);
    }

    #[inline(always)]
    fn store_relaxed(&self, slot: usize, output: &V) {
        debug_assert!(slot < WIDTH_15, "store_relaxed: slot {slot} out of bounds");

        // Relaxed ordering: the permutation CAS provides the ordering guarantee.
        self.values[slot].store(output.to_bits(), RELAXED);
        self.tags[slot].store(InlineSentinel::inline_sentinel_ptr(), RELAXED);
    }

    #[inline(always)]
    fn update_in_place(&self, slot: usize, output: &V) -> RetireHandle {
        debug_assert!(
            slot < WIDTH_15,
            "update_in_place: slot {slot} out of bounds"
        );
        debug_assert!(
            InlineSentinel::is_inline_sentinel(self.tags[slot].load(RELAXED)),
            "update_in_place called on non-value slot {slot}"
        );

        // Overwrite bits directly. Sentinel tag is already in place.
        // No retirement needed for inline values — they're Copy.
        self.values[slot].store(output.to_bits(), WRITE_ORD);

        RetireHandle::Noop
    }

    #[inline(always)]
    fn take(&self, slot: usize) -> Option<V> {
        debug_assert!(slot < WIDTH_15, "take: slot {slot} out of bounds");

        let tag: *mut u8 = self.tags[slot].load(RELAXED);

        if InlineSentinel::is_inline_sentinel(tag) {
            let bits: u64 = self.values[slot].load(RELAXED);
            let value: V = V::from_bits(bits);

            // Clear the slot by setting tag to null.
            self.tags[slot].store(StdPtr::null_mut(), RELAXED);

            Some(value)
        } else {
            // Empty or layer — nothing to take.
            None
        }
    }

    // ========================================================================
    //  Layer Pointer Operations
    // ========================================================================

    #[inline(always)]
    fn load_raw(&self, slot: usize) -> *mut u8 {
        debug_assert!(slot < WIDTH_15, "load_raw: slot {slot} out of bounds");
        // Returns the tag pointer: null (empty), sentinel (value), or layer pointer.
        self.tags[slot].load(READ_ORD)
    }

    #[inline(always)]
    fn load_layer(&self, slot: usize) -> *mut u8 {
        debug_assert!(slot < WIDTH_15, "load_layer: slot {slot} out of bounds");
        // Layer pointers are stored directly in the tag array.
        self.tags[slot].load(READ_ORD)
    }

    #[inline(always)]
    fn store_layer(&self, slot: usize, ptr: *mut u8) {
        debug_assert!(slot < WIDTH_15, "store_layer: slot {slot} out of bounds");
        debug_assert!(
            !ptr.is_null(),
            "store_layer: null layer pointer at slot {slot}"
        );
        debug_assert!(
            !InlineSentinel::is_inline_sentinel(ptr),
            "store_layer: sentinel pointer used as layer pointer at slot {slot}"
        );

        // Store layer pointer directly in tag. No bits stored.
        self.tags[slot].store(ptr, WRITE_ORD);
    }

    // ========================================================================
    //  Slot Management
    // ========================================================================

    #[inline(always)]
    fn clear(&self, slot: usize) {
        debug_assert!(slot < WIDTH_15, "clear: slot {slot} out of bounds");
        self.tags[slot].store(StdPtr::null_mut(), WRITE_ORD);
        // values[slot] left as-is — tag null makes it invisible.
    }

    #[inline(always)]
    fn move_slot(&self, dst: &Self, src_slot: usize, dst_slot: usize) {
        debug_assert!(
            src_slot < WIDTH_15,
            "move_slot: src_slot {src_slot} out of bounds"
        );
        debug_assert!(
            dst_slot < WIDTH_15,
            "move_slot: dst_slot {dst_slot} out of bounds"
        );

        let tag: *mut u8 = self.tags[src_slot].load(RELAXED);

        if InlineSentinel::is_inline_sentinel(tag) {
            // Terminal value: copy both tag and bits.
            let bits: u64 = self.values[src_slot].load(RELAXED);
            dst.values[dst_slot].store(bits, WRITE_ORD);
            dst.tags[dst_slot].store(InlineSentinel::inline_sentinel_ptr(), WRITE_ORD);
        } else {
            // Layer pointer or empty: copy tag only.
            dst.tags[dst_slot].store(tag, WRITE_ORD);
        }

        // NOTE: Caller MUST call self.clear(src_slot) after this.
    }

    // ========================================================================
    //  Lifecycle
    // ========================================================================

    #[inline(always)]
    unsafe fn cleanup(&self, _slot: usize) {
        // No-op for inline values.
        // Inline values are Copy — no heap allocation, no refcount, nothing to free.
        // The leaf's Drop impl calls this for completeness but it's a no-op.
    }
}
