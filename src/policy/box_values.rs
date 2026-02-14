//! Box-based value array implementation.
//!
//! Stores `Box<V>` values as raw pointers in `[AtomicPtr<u8>; 15]`.
//! This is the value storage backend for [`BoxPolicy<V>`].
//!
//! Unlike [`BoxValueArray`], `load()` does **not** perform any atomic
//! reference-count operations. Values are returned as [`ValuePtr<V>`]
//! — a `Copy` wrapper around a raw pointer. Lifetime safety comes from
//! the EBR guard held by the caller: the pointer remains valid as long
//! as the guard is alive.
//!
//! # Slot State Encoding
//!
//! - `null` pointer → Empty
//! - non-null pointer → Terminal value (`Box<V>` raw pointer) or Layer pointer
//!
//! Layer vs value distinction is made by the leaf via `keylenx`.
//! This array does not discriminate — `load()` returns `Some(ValuePtr)` for
//! any non-null pointer. The caller **MUST** check `keylenx < LAYER_KEYLENX`
//! before calling `load()`. Calling `load()` on a layer slot produces a
//! `ValuePtr` to a non-`V` allocation — dereferencing it is **UB**.
//!
//! [`BoxPolicy<V>`]: super::BoxPolicy
//! [`BoxValueArray`]: super::BoxValueArray

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

/// Value array backed by `[AtomicPtr<u8>; 15]`.
///
/// Each slot stores a raw pointer that is either:
/// - `null` → slot is empty
/// - valid `Box<V>` pointer → terminal value
/// - valid node pointer → layer (distinguished by `keylenx` on the leaf)
///
/// # Memory Ownership
///
/// For terminal value slots, this array **owns** the heap allocation
/// produced by `Box::into_raw`. When the leaf is dropped, `cleanup()` must
/// be called for each terminal value slot to reclaim the allocation.
///
/// For layer pointer slots, ownership belongs to the tree's allocator.
/// This array merely stores the pointer.
#[repr(C)]
pub struct BoxValueArray<V> {
    ptrs: [AtomicPtr<u8>; WIDTH_15],
    _marker: PhantomData<V>,
}

impl<V> BoxValueArray<V> {
    /// Load the raw pointer at a slot without any typed interpretation.
    ///
    /// Used by `BoxPolicy::load_value_ref()` for zero-copy reference returns
    /// and by internal classification methods.
    #[inline(always)]
    pub(crate) fn load_raw(&self, slot: usize) -> *mut u8 {
        debug_assert!(slot < WIDTH_15, "load_raw: slot {slot} out of bounds");
        self.ptrs[slot].load(READ_ORD)
    }
}

// SAFETY: BoxValueArray is Send+Sync when V: Send+Sync.
// The AtomicPtr provides thread-safe access. The raw pointers stored
// are valid Box<V> pointers or layer pointers protected by the tree's
// concurrency protocol (OCC + locks).
unsafe impl<V: Send + Sync> Send for BoxValueArray<V> {}
unsafe impl<V: Send + Sync> Sync for BoxValueArray<V> {}

impl<V: Send + Sync + 'static> ValueArray<ValuePtr<V>> for BoxValueArray<V> {
    #[inline(always)]
    fn new() -> Self {
        // SAFETY: BoxValueArray is #[repr(C)]. All fields are zero-safe:
        // - ptrs: [AtomicPtr<u8>; 15] — null pointers are all-zero-bits
        // - _marker: PhantomData<V> — ZST, no bytes
        // AtomicPtr is #[repr(C)] wrapping UnsafeCell<*mut T>; null is 0.
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
        // WARNING: For Box mode, this cannot distinguish values from layers.
        // Returns `true` for ANY non-null pointer, including terminal values.
        // The authoritative layer check is `keylenx >= LAYER_KEYLENX` on the
        // leaf node. This method exists only for the ValueArray trait contract
        // (Drop path, internal consistency checks). Prefer `leaf.is_layer(slot)`.
        !self.ptrs[slot].load(READ_ORD).is_null()
    }

    // ========================================================================
    //  Terminal Value Operations
    // ========================================================================

    #[inline(always)]
    /// # Prechecked Contract
    ///
    /// Caller **must** verify `keylenx < LAYER_KEYLENX` before calling.
    /// This method does NOT check layer status. Calling on a layer slot
    /// produces a `ValuePtr` pointing to a non-`V` allocation — dereferencing
    /// it is **UB**.
    fn load(&self, slot: usize) -> Option<ValuePtr<V>> {
        debug_assert!(slot < WIDTH_15, "load: slot {slot} out of bounds");

        let ptr: *mut u8 = self.ptrs[slot].load(READ_ORD);

        if ptr.is_null() {
            return None;
        }

        // SAFETY: Caller has verified `keylenx < LAYER_KEYLENX` before calling
        // (prechecked contract). ptr is non-null and was stored by
        // into_output (Box::into_raw). We wrap it in ValuePtr without any
        // refcount operation — the pointer is valid as long as the caller's
        // EBR guard is held.
        //
        // WARNING: Calling this on a layer slot produces a ValuePtr to a
        // non-V allocation. The leaf's classify_slot() and load_value()
        // methods enforce the keylenx precondition.
        unsafe { Some(ValuePtr::from_raw(ptr.cast::<V>())) }
    }

    #[inline(always)]
    fn store(&self, slot: usize, output: &ValuePtr<V>) {
        debug_assert!(slot < WIDTH_15, "store: slot {slot} out of bounds");

        // Extract the raw pointer from ValuePtr and store it.
        // The slot takes ownership of the allocation.
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

        // CRITICAL: Capture old pointer BEFORE storing new one.
        // This is the key safety invariant — reading after store would
        // return the new value, causing us to retire the wrong pointer.
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

        // SAFETY: old_ptr was stored by us via Box::into_raw (through into_output).
        // We take ownership of the stored pointer (no refcount change).
        // The swap to null ensures no double-free from the slot side.
        // The caller must ensure the returned ValuePtr is either:
        // 1. Retired via EBR (for concurrent safety), or
        // 2. Used only while the guard is held.
        unsafe { Some(ValuePtr::from_raw(old_ptr.cast::<V>())) }
    }

    // ========================================================================
    //  Layer Pointer Operations
    // ========================================================================

    #[inline(always)]
    fn load_raw(&self, slot: usize) -> *mut u8 {
        // Delegate to inherent method to avoid duplicate logic.
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

        // Load the raw pointer from source. This may be a value or layer pointer.
        // No ownership change — we're transferring the pointer.
        let ptr: *mut u8 = self.ptrs[src_slot].load(RELAXED);
        dst.ptrs[dst_slot].store(ptr, WRITE_ORD);

        // NOTE: Caller MUST call self.clear(src_slot) after this.
        // We don't clear here because the caller may need to copy
        // additional data (keylenx, suffix) before clearing.
    }

    // ========================================================================
    //  Lifecycle
    // ========================================================================

    #[inline(always)]
    unsafe fn cleanup(&self, slot: usize) {
        debug_assert!(slot < WIDTH_15, "cleanup: slot {slot} out of bounds");

        let ptr: *mut u8 = self.ptrs[slot].load(RELAXED);
        debug_assert!(!ptr.is_null(), "cleanup called on empty slot {slot}");

        // SAFETY: Caller guarantees:
        // 1. Exclusive access (&mut via Drop).
        // 2. Slot contains a terminal value (checked via keylenx in leaf Drop).
        // 3. The pointer was stored by us via Box::into_raw.
        unsafe {
            drop(Box::from_raw(ptr.cast::<V>()));
        }
    }
}
