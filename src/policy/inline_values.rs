//! True-inline value array implementation.

use std::marker::PhantomData;
use std::ptr as StdPtr;
use std::sync::atomic::{AtomicPtr, AtomicU64, Ordering};

use crate::inline::bits::InlineBits;
use crate::inline::sentinel::InlineSentinel;
use crate::leaf15::WIDTH_15;
use crate::ordering::{READ_ORD, RELAXED, WRITE_ORD};

use super::RetireHandle;
use super::ValueArray;

// ============================================================================
//  InlineValueArray<V>
// ============================================================================

/// Keep each tag/payload pair inside one cache line, including in a leaf.
#[repr(C, align(16))]
struct InlineSlot {
    tag: AtomicPtr<u8>,
    value: AtomicU64,
}

/// Interleaved state tags and inline value bits (240 bytes on 64-bit targets).
#[repr(C)]
pub struct InlineValueArray<V: InlineBits> {
    slots: [InlineSlot; WIDTH_15],
    _marker: PhantomData<V>,
}

#[cfg(target_pointer_width = "64")]
const _: () = {
    use std::mem::{align_of, offset_of, size_of};
    assert!(size_of::<InlineSlot>() == 16);
    assert!(align_of::<InlineSlot>() == 16);
    assert!(offset_of!(InlineSlot, tag) == 0);
    assert!(offset_of!(InlineSlot, value) == 8);
    assert!(size_of::<InlineValueArray<u64>>() == 240);
    assert!(align_of::<InlineValueArray<u64>>() == 16);
};

// SAFETY: InlineValueArray is Send+Sync when V: Send+Sync (which InlineBits requires).
// The AtomicPtr and AtomicU64 provide thread-safe access. Layer pointers stored
// in tags are protected by the tree's concurrency protocol (OCC + locks).
unsafe impl<V: InlineBits> Send for InlineValueArray<V> {}
unsafe impl<V: InlineBits> Sync for InlineValueArray<V> {}

impl<V: InlineBits> ValueArray<V> for InlineValueArray<V> {
    #[inline(always)]
    fn new() -> Self {
        // SAFETY: InlineValueArray is #[repr(C)]. All fields are zero-safe:
        // - each slot: AtomicPtr<u8> + AtomicU64 — null/zero are all-zero-bits
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
        self.slots[slot].tag.load(READ_ORD).is_null()
    }

    #[inline(always)]
    fn is_empty_relaxed(&self, slot: usize) -> bool {
        debug_assert!(
            slot < WIDTH_15,
            "is_empty_relaxed: slot {slot} out of bounds"
        );
        self.slots[slot].tag.load(RELAXED).is_null()
    }

    #[inline(always)]
    fn is_layer(&self, slot: usize) -> bool {
        debug_assert!(slot < WIDTH_15, "is_layer: slot {slot} out of bounds");

        let tag: *mut u8 = self.slots[slot].tag.load(READ_ORD);

        !tag.is_null() && !InlineSentinel::is_inline_sentinel(tag)
    }

    // ========================================================================
    //  Terminal Value Operations
    // ========================================================================

    #[inline(always)]
    fn load(&self, slot: usize) -> Option<V> {
        debug_assert!(slot < WIDTH_15, "load: slot {slot} out of bounds");

        let tag: *mut u8 = self.slots[slot].tag.load(READ_ORD);

        if InlineSentinel::is_inline_sentinel(tag) {
            let bits: u64 = self.slots[slot].value.load(RELAXED);

            Some(V::from_bits(bits))
        } else {
            None
        }
    }

    #[inline(always)]
    fn store(&self, slot: usize, output: &V) {
        debug_assert!(slot < WIDTH_15, "store: slot {slot} out of bounds");

        self.slots[slot].value.store(output.to_bits(), RELAXED);
        self.slots[slot]
            .tag
            .store(InlineSentinel::inline_sentinel_ptr(), WRITE_ORD);
    }

    #[inline(always)]
    fn store_relaxed(&self, slot: usize, output: &V) {
        debug_assert!(slot < WIDTH_15, "store_relaxed: slot {slot} out of bounds");

        self.slots[slot].value.store(output.to_bits(), RELAXED);
        self.slots[slot]
            .tag
            .store(InlineSentinel::inline_sentinel_ptr(), RELAXED);
    }

    type Prepared = u64;

    #[inline(always)]
    fn prepare(output: &V) -> u64 {
        output.to_bits()
    }

    #[inline(always)]
    fn store_prepared(&self, slot: usize, prepared: &u64) {
        debug_assert!(slot < WIDTH_15, "store_prepared: slot {slot} out of bounds");

        self.slots[slot].value.store(*prepared, RELAXED);
        self.slots[slot]
            .tag
            .store(InlineSentinel::inline_sentinel_ptr(), WRITE_ORD);
    }

    #[inline(always)]
    fn store_prepared_relaxed(&self, slot: usize, prepared: &u64) {
        debug_assert!(
            slot < WIDTH_15,
            "store_prepared_relaxed: slot {slot} out of bounds"
        );

        self.slots[slot].value.store(*prepared, RELAXED);
        self.slots[slot]
            .tag
            .store(InlineSentinel::inline_sentinel_ptr(), RELAXED);
    }

    #[inline(always)]
    fn update_in_place(&self, slot: usize, output: &V) -> RetireHandle {
        debug_assert!(
            slot < WIDTH_15,
            "update_in_place: slot {slot} out of bounds"
        );
        debug_assert!(
            InlineSentinel::is_inline_sentinel(self.slots[slot].tag.load(RELAXED)),
            "update_in_place called on non-value slot {slot}"
        );

        self.slots[slot].value.store(output.to_bits(), WRITE_ORD);

        RetireHandle::Noop
    }

    #[inline(always)]
    fn update_in_place_relaxed(&self, slot: usize, output: &V) -> RetireHandle {
        debug_assert!(
            slot < WIDTH_15,
            "update_in_place_relaxed: slot {slot} out of bounds"
        );
        debug_assert!(
            InlineSentinel::is_inline_sentinel(self.slots[slot].tag.load(RELAXED)),
            "update_in_place_relaxed called on non-value slot {slot}"
        );

        self.slots[slot].value.store(output.to_bits(), RELAXED);

        RetireHandle::Noop
    }

    #[inline(always)]
    fn take(&self, slot: usize) -> Option<V> {
        debug_assert!(slot < WIDTH_15, "take: slot {slot} out of bounds");

        let tag: *mut u8 = self.slots[slot].tag.load(RELAXED);

        if InlineSentinel::is_inline_sentinel(tag) {
            let bits: u64 = self.slots[slot].value.load(RELAXED);
            let value: V = V::from_bits(bits);

            self.slots[slot].tag.store(StdPtr::null_mut(), RELAXED);

            Some(value)
        } else {
            None
        }
    }

    // ========================================================================
    //  Layer Pointer Operations
    // ========================================================================

    #[inline(always)]
    fn load_raw(&self, slot: usize) -> *mut u8 {
        debug_assert!(slot < WIDTH_15, "load_raw: slot {slot} out of bounds");

        // SeqCst participates in seize's protected-load total order; the tag
        // may be a sublayer pointer that the caller dereferences.
        self.slots[slot].tag.load(Ordering::SeqCst)
    }

    #[inline(always)]
    fn load_raw_relaxed(&self, slot: usize) -> *mut u8 {
        debug_assert!(
            slot < WIDTH_15,
            "load_raw_relaxed: slot {slot} out of bounds"
        );

        self.slots[slot].tag.load(RELAXED)
    }

    #[inline(always)]
    fn load_layer(&self, slot: usize) -> *mut u8 {
        debug_assert!(slot < WIDTH_15, "load_layer: slot {slot} out of bounds");

        self.slots[slot].tag.load(Ordering::SeqCst)
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

        self.slots[slot].tag.store(ptr, WRITE_ORD);
    }

    // ========================================================================
    //  Relaxed Load (under lock)
    // ========================================================================

    #[inline(always)]
    fn load_relaxed(&self, slot: usize) -> Option<V> {
        debug_assert!(slot < WIDTH_15, "load_relaxed: slot {slot} out of bounds");

        let tag: *mut u8 = self.slots[slot].tag.load(RELAXED);

        if InlineSentinel::is_inline_sentinel(tag) {
            let bits: u64 = self.slots[slot].value.load(RELAXED);

            Some(V::from_bits(bits))
        } else {
            None
        }
    }

    // ========================================================================
    //  Slot Management
    // ========================================================================

    #[inline(always)]
    fn clear(&self, slot: usize) {
        debug_assert!(slot < WIDTH_15, "clear: slot {slot} out of bounds");

        self.slots[slot].tag.store(StdPtr::null_mut(), WRITE_ORD);
    }

    #[inline(always)]
    fn clear_relaxed(&self, slot: usize) {
        debug_assert!(slot < WIDTH_15, "clear_relaxed: slot {slot} out of bounds");

        self.slots[slot].tag.store(StdPtr::null_mut(), RELAXED);
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

        let tag: *mut u8 = self.slots[src_slot].tag.load(RELAXED);

        if InlineSentinel::is_inline_sentinel(tag) {
            let bits: u64 = self.slots[src_slot].value.load(RELAXED);

            dst.slots[dst_slot].value.store(bits, WRITE_ORD);
            dst.slots[dst_slot]
                .tag
                .store(InlineSentinel::inline_sentinel_ptr(), WRITE_ORD);
        } else {
            dst.slots[dst_slot].tag.store(tag, WRITE_ORD);
        }

        // True move: clear the source tag.
        self.slots[src_slot].tag.store(StdPtr::null_mut(), RELAXED);
    }

    #[inline(always)]
    fn move_slot_relaxed(&self, dst: &Self, src_slot: usize, dst_slot: usize) {
        debug_assert!(
            src_slot < WIDTH_15,
            "move_slot_relaxed: src_slot {src_slot} out of bounds"
        );
        debug_assert!(
            dst_slot < WIDTH_15,
            "move_slot_relaxed: dst_slot {dst_slot} out of bounds"
        );

        let tag: *mut u8 = self.slots[src_slot].tag.load(RELAXED);

        if InlineSentinel::is_inline_sentinel(tag) {
            let bits: u64 = self.slots[src_slot].value.load(RELAXED);

            dst.slots[dst_slot].value.store(bits, RELAXED);
            dst.slots[dst_slot]
                .tag
                .store(InlineSentinel::inline_sentinel_ptr(), RELAXED);
        } else {
            dst.slots[dst_slot].tag.store(tag, RELAXED);
        }

        // True move: clear the source tag.
        self.slots[src_slot].tag.store(StdPtr::null_mut(), RELAXED);
    }

    // ========================================================================
    //  Lifecycle
    // ========================================================================

    #[inline(always)]
    unsafe fn cleanup(&self, _slot: usize) {}
}

#[cfg(test)]
mod layout_tests {
    use super::*;

    #[test]
    fn inline_slot_tag_and_value_share_cache_line() {
        let array = InlineValueArray::<u64>::new();
        for slot in 0..WIDTH_15 {
            let tag = (&raw const array.slots[slot].tag).addr();
            let value = (&raw const array.slots[slot].value).addr();
            assert_eq!(tag / 64, value / 64, "slot {slot} spans cache lines");
        }
    }
}
