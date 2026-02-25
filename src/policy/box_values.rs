//! Box-based value array: stores `Box<V>` as raw pointers in `[AtomicPtr<u8>; 15]`.

use std::marker::PhantomData;
use std::ptr as StdPtr;
use std::sync::atomic::AtomicPtr;

use crate::leaf15::WIDTH_15;
use crate::ordering::{READ_ORD, RELAXED, WRITE_ORD};

use super::RetireHandle;
use super::ValueArray;
use super::ValuePtr;

// ============================================================================
//  BoxValueArray<V>
// ============================================================================

/// Value array storing `Box<V>` pointers in `[AtomicPtr<u8>; 15]`.
#[repr(C)]
pub struct BoxValueArray<V> {
    ptrs: [AtomicPtr<u8>; WIDTH_15],
    _marker: PhantomData<V>,
}

impl<V> BoxValueArray<V> {
    /// Load the raw pointer at `slot` without typed interpretation.
    #[inline(always)]
    pub(crate) fn load_raw(&self, slot: usize) -> *mut u8 {
        debug_assert!(slot < WIDTH_15, "load_raw: slot {slot} out of bounds");
        self.ptrs[slot].load(READ_ORD)
    }
}

// SAFETY: AtomicPtr provides thread-safe access; raw pointers are valid
// Box<V> or layer pointers protected by OCC + locks.
unsafe impl<V: Send + Sync> Send for BoxValueArray<V> {}
unsafe impl<V: Send + Sync> Sync for BoxValueArray<V> {}

impl<V: Send + Sync + 'static> ValueArray<ValuePtr<V>> for BoxValueArray<V> {
    #[inline(always)]
    fn new() -> Self {
        // SAFETY: All-zero is valid — AtomicPtr null is zero-bits, PhantomData is ZST.
        unsafe { std::mem::zeroed() }
    }

    // ========================================================================
    //  Slot Classification
    // ========================================================================

    #[inline(always)]
    fn is_empty(&self, slot: usize) -> bool {
        debug_assert!(slot < WIDTH_15, "is_empty: slot {slot} out of bounds");
        self.ptrs[slot].load(READ_ORD).is_null()
    }

    #[inline(always)]
    fn is_layer(&self, slot: usize) -> bool {
        debug_assert!(slot < WIDTH_15, "is_layer: slot {slot} out of bounds");
        // Cannot distinguish values from layers — returns true for any non-null.
        // Authoritative check is `keylenx >= LAYER_KEYLENX` on the leaf.
        !self.ptrs[slot].load(READ_ORD).is_null()
    }

    // ========================================================================
    //  Terminal Value Operations
    // ========================================================================

    #[inline(always)]
    fn load(&self, slot: usize) -> Option<ValuePtr<V>> {
        debug_assert!(slot < WIDTH_15, "load: slot {slot} out of bounds");

        let ptr: *mut u8 = self.ptrs[slot].load(READ_ORD);
        if ptr.is_null() {
            return None;
        }

        // SAFETY: Caller verified keylenx < LAYER_KEYLENX. ptr was stored via
        // Box::into_raw. Valid while caller's EBR guard is held.
        unsafe { Some(ValuePtr::from_raw(ptr.cast::<V>())) }
    }

    #[inline(always)]
    fn store(&self, slot: usize, output: &ValuePtr<V>) {
        debug_assert!(slot < WIDTH_15, "store: slot {slot} out of bounds");
        let ptr: *mut u8 = output.as_ptr().cast::<u8>();
        self.ptrs[slot].store(ptr, WRITE_ORD);
    }

    #[inline(always)]
    fn store_relaxed(&self, slot: usize, output: &ValuePtr<V>) {
        debug_assert!(slot < WIDTH_15, "store_relaxed: slot {slot} out of bounds");
        let ptr: *mut u8 = output.as_ptr().cast::<u8>();
        self.ptrs[slot].store(ptr, RELAXED);
    }

    #[inline(always)]
    fn update_in_place(&self, slot: usize, output: &ValuePtr<V>) -> RetireHandle {
        debug_assert!(
            slot < WIDTH_15,
            "update_in_place: slot {slot} out of bounds"
        );

        let old_ptr: *mut u8 = self.ptrs[slot].load(RELAXED);
        debug_assert!(
            !old_ptr.is_null(),
            "update_in_place called on empty slot {slot}"
        );

        let new_ptr: *mut u8 = output.as_ptr().cast::<u8>();
        self.ptrs[slot].store(new_ptr, WRITE_ORD);

        RetireHandle::Ptr(old_ptr)
    }

    #[inline(always)]
    fn take(&self, slot: usize) -> Option<ValuePtr<V>> {
        debug_assert!(slot < WIDTH_15, "take: slot {slot} out of bounds");

        let old_ptr: *mut u8 = self.ptrs[slot].swap(StdPtr::null_mut(), RELAXED);
        if old_ptr.is_null() {
            return None;
        }

        // SAFETY: ptr was stored via Box::into_raw. Swap to null prevents double-free.
        unsafe { Some(ValuePtr::from_raw(old_ptr.cast::<V>())) }
    }

    // ========================================================================
    //  Layer Pointer Operations
    // ========================================================================

    #[inline(always)]
    fn load_raw(&self, slot: usize) -> *mut u8 {
        self.load_raw(slot)
    }

    #[inline(always)]
    fn load_layer(&self, slot: usize) -> *mut u8 {
        debug_assert!(slot < WIDTH_15, "load_layer: slot {slot} out of bounds");
        self.ptrs[slot].load(READ_ORD)
    }

    #[inline(always)]
    fn store_layer(&self, slot: usize, ptr: *mut u8) {
        debug_assert!(slot < WIDTH_15, "store_layer: slot {slot} out of bounds");
        self.ptrs[slot].store(ptr, WRITE_ORD);
    }

    // ========================================================================
    //  Slot Management
    // ========================================================================

    #[inline(always)]
    fn clear(&self, slot: usize) {
        debug_assert!(slot < WIDTH_15, "clear: slot {slot} out of bounds");
        self.ptrs[slot].store(StdPtr::null_mut(), WRITE_ORD);
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

        let ptr: *mut u8 = self.ptrs[src_slot].load(RELAXED);
        dst.ptrs[dst_slot].store(ptr, WRITE_ORD);
    }

    // ========================================================================
    //  Lifecycle
    // ========================================================================

    #[inline(always)]
    unsafe fn cleanup(&self, slot: usize) {
        debug_assert!(slot < WIDTH_15, "cleanup: slot {slot} out of bounds");

        let ptr: *mut u8 = self.ptrs[slot].load(RELAXED);
        debug_assert!(!ptr.is_null(), "cleanup called on empty slot {slot}");

        // SAFETY: Caller guarantees exclusive access, slot is a terminal value,
        // and the pointer was stored via Box::into_raw.
        unsafe {
            drop(Box::from_raw(ptr.cast::<V>()));
        }
    }
}
