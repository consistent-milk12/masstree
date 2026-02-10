//! Arc-based value array implementation.
//!
//! Stores `Arc<V>` values as raw pointers in `[AtomicPtr<u8>; 15]`.
//! This is the value storage backend for [`ArcPolicy<V>`].
//!
//! # Slot State Encoding
//!
//! - `null` pointer → Empty
//! - non-null pointer → Terminal value (`Arc<V>` raw pointer) or Layer pointer
//!
//! Layer vs value distinction is made by the leaf via `keylenx`.
//! This array does not discriminate — `load()` returns `Some(Arc)` for any
//! non-null pointer. The caller **MUST** check `keylenx < LAYER_KEYLENX`
//! before calling `load()`. Calling `load()` on a layer slot is **UB**
//! (it would `Arc::increment_strong_count` on a non-Arc pointer).
//!
//! [`ArcPolicy<V>`]: super::ArcPolicy

use std::marker::PhantomData;
use std::ptr as StdPtr;
use std::sync::Arc;
use std::sync::atomic::AtomicPtr;

use crate::leaf15::WIDTH_15;
use crate::ordering::{READ_ORD, RELAXED, WRITE_ORD};

use super::RetireHandle;
use super::ValueArray;

// ============================================================================
//  ArcValueArray<V>
// ============================================================================

/// Value array backed by `[AtomicPtr<u8>; 15]`.
///
/// Each slot stores a raw pointer that is either:
/// - `null` → slot is empty
/// - valid `Arc<V>` pointer → terminal value
/// - valid node pointer → layer (distinguished by `keylenx` on the leaf)
///
/// # Memory Ownership
///
/// For terminal value slots, this array **owns** one strong reference count
/// of the `Arc<V>`. When the leaf is dropped, `cleanup()` must be called
/// for each terminal value slot to decrement the refcount.
///
/// For layer pointer slots, ownership belongs to the tree's allocator.
/// This array merely stores the pointer.
#[repr(C)]
pub struct ArcValueArray<V> {
    ptrs: [AtomicPtr<u8>; WIDTH_15],
    _marker: PhantomData<V>,
}

impl<V> ArcValueArray<V> {
    /// Load the raw pointer at a slot without any Arc operations.
    ///
    /// Used by `ArcPolicy::load_value_ref()` for zero-copy reference returns
    /// and by internal classification methods.
    #[inline(always)]
    pub(crate) fn load_raw(&self, slot: usize) -> *mut u8 {
        debug_assert!(slot < WIDTH_15, "load_raw: slot {slot} out of bounds");
        self.ptrs[slot].load(READ_ORD)
    }
}

// SAFETY: ArcValueArray is Send+Sync when V: Send+Sync.
// The AtomicPtr provides thread-safe access. The raw pointers stored
// are valid Arc<V> pointers or layer pointers protected by the tree's
// concurrency protocol (OCC + locks).
unsafe impl<V: Send + Sync> Send for ArcValueArray<V> {}
unsafe impl<V: Send + Sync> Sync for ArcValueArray<V> {}

impl<V: Send + Sync + 'static> ValueArray<Arc<V>> for ArcValueArray<V> {
    #[inline(always)]
    fn new() -> Self {
        // Initialize all slots to null (empty).
        // Using array initialization to avoid per-element loop.
        Self {
            ptrs: std::array::from_fn(|_| AtomicPtr::new(StdPtr::null_mut())),
            _marker: PhantomData,
        }
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
        // WARNING: For Arc mode, this cannot distinguish values from layers.
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
    /// This method does NOT check layer status. Calling on a layer slot is UB.
    fn load(&self, slot: usize) -> Option<Arc<V>> {
        debug_assert!(slot < WIDTH_15, "load: slot {slot} out of bounds");

        let ptr: *mut u8 = self.ptrs[slot].load(READ_ORD);

        if ptr.is_null() {
            return None;
        }

        // SAFETY: Caller has verified `keylenx < LAYER_KEYLENX` before calling
        // (prechecked contract). ptr is non-null and was stored by
        // output_to_raw (Arc::into_raw). We increment the strong count to
        // create a new Arc handle without taking ownership of the stored pointer.
        //
        // WARNING: Calling this on a layer slot is UB — the pointer is not an
        // Arc allocation. The leaf's classify_slot() and load_value() methods
        // enforce the keylenx precondition.
        unsafe {
            Arc::increment_strong_count(ptr.cast::<V>());
            Some(Arc::from_raw(ptr.cast::<V>()))
        }
    }

    #[inline(always)]
    fn store(&self, slot: usize, output: &Arc<V>) {
        debug_assert!(slot < WIDTH_15, "store: slot {slot} out of bounds");

        // Clone the Arc (increment refcount) and convert to raw pointer.
        // The array takes ownership of one strong reference.
        let ptr: *mut u8 = Arc::into_raw(Arc::clone(output)) as *mut u8;
        self.ptrs[slot].store(ptr, WRITE_ORD);
    }

    #[inline(always)]
    fn store_relaxed(&self, slot: usize, output: &Arc<V>) {
        debug_assert!(slot < WIDTH_15, "store_relaxed: slot {slot} out of bounds");

        let ptr: *mut u8 = Arc::into_raw(Arc::clone(output)) as *mut u8;
        self.ptrs[slot].store(ptr, RELAXED);
    }

    #[inline(always)]
    fn update_in_place(&self, slot: usize, output: &Arc<V>) -> RetireHandle {
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

        let new_ptr: *mut u8 = Arc::into_raw(Arc::clone(output)) as *mut u8;
        self.ptrs[slot].store(new_ptr, WRITE_ORD);

        RetireHandle::Ptr(old_ptr)
    }

    #[inline(always)]
    fn take(&self, slot: usize) -> Option<Arc<V>> {
        debug_assert!(slot < WIDTH_15, "take: slot {slot} out of bounds");

        let old_ptr: *mut u8 = self.ptrs[slot].swap(StdPtr::null_mut(), RELAXED);

        if old_ptr.is_null() {
            return None;
        }

        // SAFETY: old_ptr was stored by us via Arc::into_raw.
        // We take ownership of the stored reference (no refcount change).
        // The swap to null ensures no double-free.
        unsafe { Some(Arc::from_raw(old_ptr.cast::<V>())) }
    }

    // ========================================================================
    //  Layer Pointer Operations
    // ========================================================================

    #[inline(always)]
    fn load_raw(&self, slot: usize) -> *mut u8 {
        debug_assert!(slot < WIDTH_15, "load_raw: slot {slot} out of bounds");
        self.ptrs[slot].load(READ_ORD)
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
        // No refcount change — we're transferring ownership.
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
        // 3. The pointer was stored by us via Arc::into_raw.
        unsafe {
            drop(Arc::from_raw(ptr.cast::<V>()));
        }
    }
}
