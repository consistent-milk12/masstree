//! True-inline leaf node with WIDTH=15.
//!
//! This module provides [`LeafNode15TrueInline`], a leaf node variant that stores
//! small [`Copy`] values directly in the leaf rather than behind heap pointers.

use seize::{Guard, LocalGuard};
use static_assertions::const_assert_eq;
use std::array as StdArray;
use std::cell::UnsafeCell;
use std::fmt as StdFmt;
use std::marker::PhantomData;
use std::mem as StdMem;
use std::ptr as StdPtr;
use std::sync::atomic::{AtomicPtr, AtomicU8, AtomicU64};

use super::bits::InlineBits;
use super::sentinel::InlineSentinel;
use crate::Linker;
use crate::alloc_common::BoxAllocator;
use crate::error::{AllocKind, AllocResult};
use crate::nodeversion::NodeVersion;
use crate::ordering::CAS_FAILURE;
use crate::ordering::CAS_SUCCESS;
use crate::ordering::RELAXED;
use crate::ordering::{READ_ORD, WRITE_ORD};
use crate::permuter::{AtomicPermuter15, Permuter15};
use crate::slot::true_inline::TrueInlineSlot;
use crate::suffix::{InlineSuffixBag, SuffixBag};
use crate::value::traits::LeafValueClear;
use crate::value::traits::LeafValueLoad;
use crate::value::traits::LeafValueStore;
use crate::value::traits::LeafValueTake;
use crate::value::traits::LeafValueUpdate;

/// Width constant for [`LeafNode15TrueInline`].
pub const WIDTH_15: usize = 15;

/// Default capacity for inline suffix storage (bytes).
const INLINE_KSUF_CAPACITY: usize = 256;

/// Special keylenx value indicating key has a suffix.
pub const KSUF_KEYLENX: u8 = 64;

/// Base keylenx value indicating a layer pointer (>= this means layer).
pub const LAYER_KEYLENX: u8 = 128;

/// Modification states - insert.
pub const MODSTATE_INSERT: u8 = 0;

/// Modification state - remove.
pub const MODSTATE_REMOVE: u8 = 1;

/// Modification state - `deleted_layer`.
pub const MODSTATE_DELETED_LAYER: u8 = 2;

/// Modification state - empty.
pub const MODSTATE_EMPTY: u8 = 3;

// Always decode as inline value
// Magic constant must match TrueInlineSlot::ENCODING_MAGIC
const ENCODING_MAGIC: u64 = 0x5555_5555_5555_5555;

/// True-inline leaf node with 15 slots.
///
/// # Differences from `LeafNode15<S>`
///
/// - Values are stored in `inline_values: [AtomicU64; 15]` instead of via pointers
/// - Terminal value slots use `INLINE_SENTINEL_PTR` in `leaf_values` (not the actual pointer)
/// - No retirement needed for value updates (just overwrite bits)
/// - Only supports `get_copy()`, not `get_ref()` (no `&V` references)
///
/// # Memory Layout
///
/// Same base layout as `LeafNode15`, plus `inline_values` array (+120 bytes).
///
/// # Concurrency Model
///
/// Uses the same OCC protocol as `LeafNode15`:
/// - Writers hold lock, update `inline_values[slot]` with `WRITE_ORD`, then unlock
/// - Readers use `NodeVersion::stable()` which provides Acquire, then read with `READ_ORD`
#[repr(C, align(64))]
pub struct LeafNode15TrueInline<V: InlineBits> {
    // ========================================================================
    // Cache Line 0: Version + metadata
    // ========================================================================
    version: NodeVersion,
    modstate: AtomicU8,
    _pad0: [u8; 55],

    // ========================================================================
    // Cache Line 1: Permutation
    // ========================================================================
    permutation: AtomicPermuter15,
    _pad1: [u8; 56], // u64 = 8 bytes, 64 - 8 = 56

    // ========================================================================
    // Cache Lines 2+: Keys, keylenx, leaf_values (pointers for layers/sentinel)
    // ========================================================================
    ikey0: [AtomicU64; WIDTH_15],
    keylenx: [AtomicU8; WIDTH_15],
    leaf_values: [AtomicPtr<u8>; WIDTH_15],

    // ========================================================================
    // Inline value storage (the key addition for true-inline)
    // ========================================================================
    /// Inline value payload storage.
    ///
    /// - For terminal value slots: contains `V::to_bits()` encoded value
    /// - For empty/layer slots: contents are undefined (not read)
    ///
    /// Writers store with `WRITE_ORD` under lock.
    /// Readers load with `READ_ORD` after version validation.
    inline_values: [AtomicU64; WIDTH_15],

    // ========================================================================
    // Suffix and link pointers (same as LeafNode15)
    // ========================================================================
    inline_ksuf: UnsafeCell<InlineSuffixBag<WIDTH_15, INLINE_KSUF_CAPACITY>>,
    external_ksuf: AtomicPtr<SuffixBag<WIDTH_15>>,
    next: AtomicPtr<Self>,
    prev: AtomicPtr<Self>,
    parent: AtomicPtr<u8>,

    _marker: PhantomData<V>,
}

impl<V: InlineBits> StdFmt::Debug for LeafNode15TrueInline<V> {
    fn fmt(&self, f: &mut StdFmt::Formatter<'_>) -> StdFmt::Result {
        f.debug_struct("LeafNode15TrueInline")
            .field("size", &self.size())
            .field("is_root", &self.version.is_root())
            .finish_non_exhaustive()
    }
}

// Compile-time layout verification
const_assert_eq!(StdMem::align_of::<LeafNode15TrueInline<u64>>(), 64);

impl<V: InlineBits> LeafNode15TrueInline<V> {
    // ========================================================================
    //  Constructor Methods
    // ========================================================================

    /// Create a new true-inline leaf node.
    #[must_use]
    pub fn new_with_root(is_root: bool) -> Self {
        let version = NodeVersion::new(true);
        if is_root {
            version.mark_root();
        }

        Self {
            version,
            modstate: AtomicU8::new(MODSTATE_INSERT),
            _pad0: [0; 55],
            permutation: AtomicPermuter15::new(),
            _pad1: [0; 56],
            ikey0: StdArray::from_fn(|_| AtomicU64::new(0)),
            keylenx: StdArray::from_fn(|_| AtomicU8::new(0)),
            leaf_values: StdArray::from_fn(|_| AtomicPtr::new(StdPtr::null_mut())),
            inline_values: StdArray::from_fn(|_| AtomicU64::new(0)),
            inline_ksuf: UnsafeCell::new(InlineSuffixBag::new()),
            external_ksuf: AtomicPtr::new(StdPtr::null_mut()),
            next: AtomicPtr::new(StdPtr::null_mut()),
            prev: AtomicPtr::new(StdPtr::null_mut()),
            parent: AtomicPtr::new(StdPtr::null_mut()),
            _marker: PhantomData,
        }
    }

    /// Create a new boxed leaf node.
    #[must_use]
    #[inline(always)]
    pub fn new() -> Box<Self> {
        Box::new(Self::new_with_root(false))
    }

    /// Create a new boxed root leaf node.
    #[must_use]
    #[inline(always)]
    pub fn new_root() -> Box<Self> {
        Box::new(Self::new_with_root(true))
    }

    // ========================================================================
    //  Inline Value Operations (the key new methods)
    // ========================================================================

    /// Get the inline value at the given slot.
    ///
    /// Returns `None` if the slot is empty or contains a layer pointer.
    /// Returns `Some(V)` if the slot contains a terminal inline value.
    ///
    /// # Ordering
    ///
    /// Uses `READ_ORD`. Caller must have validated version before calling.
    ///
    /// # Panics
    ///
    /// Panics in debug builds if `slot >= WIDTH_15`.
    #[inline(always)]
    #[expect(
        clippy::indexing_slicing,
        reason = "slot bounds checked by debug_assert; valid slots from Permuter15"
    )]
    pub fn inline_get(&self, slot: usize) -> Option<V> {
        debug_assert!(slot < WIDTH_15);

        let ptr: *mut u8 = self.leaf_values[slot].load(READ_ORD);

        // Check if this is a terminal inline value (sentinel pointer)
        if InlineSentinel::is_inline_sentinel(ptr) {
            let bits: u64 = self.inline_values[slot].load(READ_ORD);
            Some(V::from_bits(bits))
        } else {
            // Empty (null) or layer pointer
            None
        }
    }

    /// Set the inline value at the given slot.
    ///
    /// # Preconditions
    ///
    /// - Caller holds the leaf lock
    /// - Slot is being initialized or updated (not a layer)
    ///
    /// # Ordering
    ///
    /// Uses `WRITE_ORD`. The unlock provides Release ordering.
    ///
    /// # Panics
    ///
    /// Panics in debug builds if `slot >= WIDTH_15`.
    #[inline(always)]
    #[expect(
        clippy::indexing_slicing,
        reason = "slot bounds checked by debug_assert; valid slots from Permuter15"
    )]
    pub fn inline_set(&self, slot: usize, value: V) {
        debug_assert!(slot < WIDTH_15);

        // Store the value bits
        self.inline_values[slot].store(value.to_bits(), WRITE_ORD);

        // Mark slot as containing an inline value (sentinel pointer)
        self.leaf_values[slot].store(InlineSentinel::inline_sentinel_ptr(), WRITE_ORD);
    }

    /// Swap the inline value at the given slot, returning the old value.
    ///
    /// # Preconditions
    ///
    /// - Caller holds the leaf lock
    /// - Slot contains a terminal inline value (not empty, not layer)
    ///
    /// # Ordering
    ///
    /// Uses `WRITE_ORD` for the new value store.
    /// Uses `READ_ORD` for the old value load (already written, just reading).
    ///
    /// # Panics
    ///
    /// Panics in debug builds if `slot >= WIDTH_15` or if the slot does not
    /// contain an inline sentinel.
    #[inline(always)]
    #[expect(
        clippy::indexing_slicing,
        reason = "slot bounds checked by debug_assert; valid slots from Permuter15"
    )]
    pub fn inline_swap(&self, slot: usize, new_value: V) -> V {
        debug_assert!(slot < WIDTH_15);
        debug_assert!(InlineSentinel::is_inline_sentinel(
            self.leaf_values[slot].load(READ_ORD)
        ));

        // Load old value
        let old_bits: u64 = self.inline_values[slot].load(READ_ORD);
        let old_value = V::from_bits(old_bits);

        // Store new value
        self.inline_values[slot].store(new_value.to_bits(), WRITE_ORD);

        old_value
    }

    /// Clear the inline value at the given slot.
    ///
    /// # Preconditions
    ///
    /// - Caller holds the leaf lock
    ///
    /// # Behavior
    ///
    /// Sets `leaf_values[slot]` to null (marks slot as empty).
    /// Does NOT clear `inline_values[slot]` (undefined after clear anyway).
    ///
    /// # Panics
    ///
    /// Panics in debug builds if `slot >= WIDTH_15`.
    #[inline(always)]
    #[expect(
        clippy::indexing_slicing,
        reason = "slot bounds checked by debug_assert; valid slots from Permuter15"
    )]
    pub fn inline_clear(&self, slot: usize) {
        debug_assert!(slot < WIDTH_15);

        // Mark slot as empty
        self.leaf_values[slot].store(StdPtr::null_mut(), WRITE_ORD);
    }

    /// Take the inline value from a slot, clearing it and returning the value.
    ///
    /// Used during layer creation when a terminal value needs to be moved.
    ///
    /// # Preconditions
    ///
    /// - Caller holds the leaf lock
    /// - Slot contains a terminal inline value
    ///
    /// # Returns
    ///
    /// The value that was stored in the slot.
    ///
    /// # Panics
    ///
    /// Panics in debug builds if `slot >= WIDTH_15` or if the slot does not
    /// contain an inline sentinel.
    #[inline(always)]
    #[expect(
        clippy::indexing_slicing,
        reason = "slot bounds checked by debug_assert; valid slots from Permuter15"
    )]
    pub fn inline_take(&self, slot: usize) -> V {
        debug_assert!(slot < WIDTH_15);
        debug_assert!(InlineSentinel::is_inline_sentinel(
            self.leaf_values[slot].load(READ_ORD)
        ));

        // Load the value
        let bits: u64 = self.inline_values[slot].load(READ_ORD);
        let value = V::from_bits(bits);

        // Clear the slot (layer pointer will be installed separately)
        self.leaf_values[slot].store(StdPtr::null_mut(), WRITE_ORD);

        value
    }

    // ========================================================================
    //  Standard Leaf Accessors (matching LeafNode15)
    // ========================================================================

    /// Get the node version.
    #[inline(always)]
    pub const fn version(&self) -> &NodeVersion {
        &self.version
    }

    /// Get the current permutation size (number of entries).
    #[inline(always)]
    pub fn size(&self) -> usize {
        self.permutation.load(READ_ORD).size()
    }

    /// Get the current permutation.
    #[inline(always)]
    pub fn permutation(&self) -> Permuter15 {
        self.permutation.load(READ_ORD)
    }

    /// Get the raw permutation value.
    #[inline(always)]
    pub fn permutation_raw(&self) -> u64 {
        self.permutation.load_raw(READ_ORD)
    }

    /// Set the permutation.
    #[inline(always)]
    pub fn set_permutation(&self, perm: Permuter15) {
        self.permutation.store(perm, WRITE_ORD);
    }

    /// Get the ikey at the given slot.
    ///
    /// # Panics
    ///
    /// Panics in debug builds if `slot >= WIDTH_15`.
    #[inline(always)]
    #[expect(
        clippy::indexing_slicing,
        reason = "slot bounds checked by debug_assert; valid slots from Permuter15"
    )]
    pub fn ikey(&self, slot: usize) -> u64 {
        debug_assert!(slot < WIDTH_15);
        self.ikey0[slot].load(READ_ORD)
    }

    /// Set the ikey at the given slot.
    ///
    /// # Panics
    ///
    /// Panics in debug builds if `slot >= WIDTH_15`.
    #[inline(always)]
    #[expect(
        clippy::indexing_slicing,
        reason = "slot bounds checked by debug_assert; valid slots from Permuter15"
    )]
    pub fn set_ikey(&self, slot: usize, ikey: u64) {
        debug_assert!(slot < WIDTH_15);
        self.ikey0[slot].store(ikey, WRITE_ORD);
    }

    /// Get the `ikey_bound` (slot 0's ikey).
    #[inline(always)]
    pub fn ikey_bound(&self) -> u64 {
        self.ikey0[0].load(READ_ORD)
    }

    /// Get the keylenx at the given slot.
    ///
    /// # Panics
    ///
    /// Panics in debug builds if `slot >= WIDTH_15`.
    #[inline(always)]
    #[expect(
        clippy::indexing_slicing,
        reason = "slot bounds checked by debug_assert; valid slots from Permuter15"
    )]
    pub fn keylenx(&self, slot: usize) -> u8 {
        debug_assert!(slot < WIDTH_15);
        self.keylenx[slot].load(READ_ORD)
    }

    /// Set the keylenx at the given slot.
    ///
    /// # Panics
    ///
    /// Panics in debug builds if `slot >= WIDTH_15`.
    #[inline(always)]
    #[expect(
        clippy::indexing_slicing,
        reason = "slot bounds checked by debug_assert; valid slots from Permuter15"
    )]
    pub fn set_keylenx(&self, slot: usize, keylenx: u8) {
        debug_assert!(slot < WIDTH_15);
        self.keylenx[slot].store(keylenx, WRITE_ORD);
    }

    /// Get the leaf value pointer at the given slot.
    ///
    /// For true-inline leaves:
    /// - null → empty slot
    /// - encoded value pointer (for inline values)
    /// - layer pointer (for sublayers)
    /// - null (for empty slots)
    ///
    /// For true-inline storage, if the slot contains an inline value (sentinel
    /// stored in `leaf_values`), this returns the encoded value from `inline_values`.
    ///
    /// # Panics
    ///
    /// Panics in debug builds if `slot >= WIDTH_15`.
    #[inline(always)]
    #[expect(clippy::cast_possible_truncation)]
    #[expect(
        clippy::indexing_slicing,
        reason = "slot bounds checked by debug_assert; valid slots from Permuter15"
    )]
    pub fn leaf_value_ptr(&self, slot: usize) -> *mut u8 {
        debug_assert!(slot < WIDTH_15);
        let ptr = self.leaf_values[slot].load(READ_ORD);

        // Check if this is a sentinel (inline value)
        if InlineSentinel::is_inline_sentinel(ptr) {
            let bits: u64 = self.inline_values[slot].load(READ_ORD);
            let encoded: u64 = bits ^ ENCODING_MAGIC;

            StdPtr::without_provenance_mut(encoded as usize)
        } else {
            // Return the actual pointer (null or layer)
            ptr
        }
    }

    /// Set the leaf value pointer at the given slot.
    ///
    /// For true-inline storage, this always decodes the pointer as an encoded
    /// inline value. For layer pointers, use [`Self::set_layer_ptr`] instead.
    ///
    /// # Panics
    ///
    /// Panics in debug builds if `slot >= WIDTH_15`.
    #[inline(always)]
    #[expect(
        clippy::indexing_slicing,
        reason = "slot bounds checked by debug_assert; valid slots from Permuter15"
    )]
    pub fn set_leaf_value_ptr(&self, slot: usize, ptr: *mut u8) {
        debug_assert!(slot < WIDTH_15);

        if ptr.is_null() {
            // Empty slot - just store null
            self.leaf_values[slot].store(ptr, WRITE_ORD);
            return;
        }

        let encoded: u64 = ptr.addr() as u64;
        let bits: u64 = encoded ^ ENCODING_MAGIC;
        self.inline_values[slot].store(bits, WRITE_ORD);
        self.leaf_values[slot].store(InlineSentinel::inline_sentinel_ptr(), WRITE_ORD);
    }

    /// Update an existing inline value in place.
    ///
    /// This is an optimization for updates where the slot already contains
    /// an inline value. Since the sentinel is already in `leaf_values[slot]`,
    /// we only need to update `inline_values[slot]` - saving one atomic store.
    ///
    /// # Panics
    ///
    /// Debug-asserts that `slot < WIDTH_15` and `ptr` is non-null.
    #[inline(always)]
    #[expect(
        clippy::indexing_slicing,
        reason = "slot bounds checked by debug_assert; valid slots from Permuter15"
    )]
    pub fn update_leaf_value_in_place(&self, slot: usize, ptr: *mut u8) {
        debug_assert!(slot < WIDTH_15);
        debug_assert!(!ptr.is_null(), "update_leaf_value_in_place: ptr is null");

        let encoded: u64 = ptr.addr() as u64;
        let bits: u64 = encoded ^ ENCODING_MAGIC;
        // Only update inline_values - sentinel is already in leaf_values
        self.inline_values[slot].store(bits, WRITE_ORD);
    }

    /// Set a layer pointer at the given slot.
    ///
    /// This stores the pointer directly without decoding.
    /// Caller must also set `keylenx = LAYER_KEYLENX`.
    #[inline(always)]
    #[expect(
        clippy::indexing_slicing,
        reason = "slot bounds checked by debug_assert"
    )]
    pub fn set_layer_ptr(&self, slot: usize, ptr: *mut u8) {
        debug_assert!(slot < WIDTH_15);
        self.leaf_values[slot].store(ptr, WRITE_ORD);
    }

    /// Get the next leaf pointer (including mark bit).
    ///
    /// Uses guard protection to ensure the load participates in seize's
    /// total order, making it safe on all architectures.
    #[inline(always)]
    pub fn next_raw(&self, guard: &impl Guard) -> *mut Self {
        guard.protect(&self.next, READ_ORD)
    }

    /// Get the next leaf pointer without guard protection.
    ///
    /// # Safety
    ///
    /// Caller must ensure the next pointer's target won't be retired during use.
    #[inline(always)]
    pub unsafe fn next_raw_unguarded(&self) -> *mut Self {
        self.next.load(READ_ORD)
    }

    /// Get the next leaf pointer (without mark bit).
    ///
    /// Uses guard protection to ensure the load participates in seize's
    /// total order, making it safe on all architectures.
    #[inline(always)]
    pub fn safe_next(&self, guard: &impl Guard) -> *mut Self {
        Linker::unmark_ptr(guard.protect(&self.next, READ_ORD))
    }

    /// Get the next leaf pointer without guard protection.
    ///
    /// # Safety
    ///
    /// Caller must ensure the next pointer's target won't be retired during use.
    #[inline(always)]
    pub unsafe fn safe_next_unguarded(&self) -> *mut Self {
        Linker::unmark_ptr(self.next.load(READ_ORD))
    }

    /// Set the next leaf pointer.
    #[inline(always)]
    pub fn set_next(&self, next: *mut Self) {
        self.next.store(next, WRITE_ORD);
    }

    /// Get the previous leaf pointer.
    ///
    /// Uses guard protection to ensure the load participates in seize's
    /// total order, making it safe on all architectures.
    #[inline(always)]
    pub fn prev(&self, guard: &impl Guard) -> *mut Self {
        guard.protect(&self.prev, READ_ORD)
    }

    /// Get the previous leaf pointer without guard protection.
    ///
    /// # Safety
    ///
    /// Caller must ensure the prev pointer's target won't be retired during use.
    #[inline(always)]
    pub unsafe fn prev_unguarded(&self) -> *mut Self {
        self.prev.load(READ_ORD)
    }

    /// Set the previous leaf pointer.
    #[inline(always)]
    pub fn set_prev(&self, prev: *mut Self) {
        self.prev.store(prev, WRITE_ORD);
    }

    /// Get the parent pointer.
    ///
    /// Uses guard protection to ensure the load participates in seize's
    /// total order, making it safe on all architectures.
    #[inline(always)]
    pub fn parent(&self, guard: &impl Guard) -> *mut u8 {
        guard.protect(&self.parent, READ_ORD)
    }

    /// Get the parent pointer without guard protection.
    ///
    /// # Safety
    ///
    /// Caller must ensure the parent pointer's target won't be retired during use.
    #[inline(always)]
    pub unsafe fn parent_unguarded(&self) -> *mut u8 {
        self.parent.load(READ_ORD)
    }

    /// Set the parent pointer.
    #[inline(always)]
    pub fn set_parent(&self, parent: *mut u8) {
        self.parent.store(parent, WRITE_ORD);
    }

    /// Check if this leaf is empty (has no entries).
    #[inline(always)]
    pub fn is_empty(&self) -> bool {
        self.permutation.load(READ_ORD).size() == 0
    }

    /// Check if this leaf is full.
    #[inline(always)]
    pub fn is_full(&self) -> bool {
        self.size() >= WIDTH_15
    }

    /// Get the modification state.
    #[inline(always)]
    pub fn modstate(&self) -> u8 {
        self.modstate.load(READ_ORD)
    }

    /// Check if this is a deleted layer.
    #[inline(always)]
    pub fn deleted_layer(&self) -> bool {
        self.modstate() == MODSTATE_DELETED_LAYER
    }

    /// Mark this leaf as a deleted layer.
    #[inline(always)]
    pub fn mark_deleted_layer(&self) {
        self.modstate.store(MODSTATE_DELETED_LAYER, WRITE_ORD);
    }

    /// Mark this leaf as empty.
    #[inline(always)]
    pub fn mark_empty(&self) {
        self.modstate.store(MODSTATE_EMPTY, WRITE_ORD);
    }

    /// Clear the empty state (reactivating the leaf).
    #[inline(always)]
    pub fn clear_empty_state(&self) {
        self.modstate.store(MODSTATE_INSERT, WRITE_ORD);
    }

    /// Convert this leaf into a layer root.
    #[inline(always)]
    pub fn make_layer_root(&self) {
        self.set_parent(StdPtr::null_mut());
        self.version.mark_root();
    }

    /// Create a new leaf node configured as a layer root.
    #[inline]
    #[must_use]
    pub fn new_layer_root() -> Box<Self> {
        let node: Box<Self> = Self::new();
        node.make_layer_root();
        node
    }

    // ========================================================================
    //  Additional Key Operations
    // ========================================================================

    /// Get the ikey at the given physical slot using Relaxed ordering.
    #[must_use]
    #[inline(always)]
    #[expect(clippy::indexing_slicing, reason = "bounds checked by debug_assert")]
    pub fn ikey_relaxed(&self, slot: usize) -> u64 {
        debug_assert!(slot < WIDTH_15, "ikey_relaxed: slot out of bounds");
        self.ikey0[slot].load(crate::ordering::RELAXED)
    }

    /// Check if the given slot contains a layer pointer.
    #[must_use]
    #[inline(always)]
    pub fn is_layer(&self, slot: usize) -> bool {
        self.keylenx(slot) >= LAYER_KEYLENX
    }

    /// Check if the given slot has a suffix.
    #[must_use]
    #[inline(always)]
    pub fn has_ksuf(&self, slot: usize) -> bool {
        self.keylenx(slot) == KSUF_KEYLENX
    }

    /// Check if keylenx indicates a layer pointer (static helper).
    #[inline(always)]
    #[must_use]
    pub const fn keylenx_is_layer(keylenx: u8) -> bool {
        keylenx >= LAYER_KEYLENX
    }

    /// Check if keylenx indicates suffix storage (static helper).
    #[must_use]
    #[inline(always)]
    pub const fn keylenx_has_ksuf(keylenx: u8) -> bool {
        keylenx == KSUF_KEYLENX
    }

    // ========================================================================
    //  ModState Additional Operations
    // ========================================================================

    /// Set the modification state.
    #[inline(always)]
    pub fn set_modstate(&self, state: u8) {
        self.modstate.store(state, WRITE_ORD);
    }

    /// Mark this node as being in remove mode.
    #[inline(always)]
    pub fn mark_remove(&self) {
        self.set_modstate(MODSTATE_REMOVE);
    }

    /// Check if this node is in remove mode.
    #[must_use]
    #[inline(always)]
    pub fn is_removing(&self) -> bool {
        self.modstate() == MODSTATE_REMOVE
    }

    /// Check if this leaf is in empty state.
    #[must_use]
    #[inline(always)]
    pub fn is_empty_state(&self) -> bool {
        self.modstate() == MODSTATE_EMPTY
    }

    // ========================================================================
    //  Value Operations
    // ========================================================================

    /// Take the leaf value pointer, leaving null in the slot.
    #[inline(always)]
    #[expect(clippy::indexing_slicing, reason = "bounds checked by debug_assert")]
    pub fn take_leaf_value_ptr(&self, slot: usize) -> *mut u8 {
        debug_assert!(slot < WIDTH_15, "take_leaf_value_ptr: slot out of bounds");
        self.leaf_values[slot].swap(StdPtr::null_mut(), crate::ordering::RELAXED)
    }

    /// Load the current value pointer at a slot.
    #[inline(always)]
    #[expect(clippy::indexing_slicing, reason = "bounds checked by debug_assert")]
    pub fn load_slot_value(&self, slot: usize) -> *mut u8 {
        debug_assert!(slot < WIDTH_15, "load_slot_value: slot out of bounds");
        self.leaf_values[slot].load(READ_ORD)
    }

    /// Atomically claim a slot for CAS insert.
    ///
    /// # Errors
    /// Returns error on [`CAS_FAILURE`]
    #[inline(always)]
    #[expect(clippy::indexing_slicing, reason = "bounds checked by debug_assert")]
    pub fn cas_slot_value(
        &self,
        slot: usize,
        expected: *mut u8,
        new_value: *mut u8,
    ) -> Result<(), *mut u8> {
        debug_assert!(slot < WIDTH_15, "cas_slot_value: slot out of bounds");
        match self.leaf_values[slot].compare_exchange(expected, new_value, CAS_SUCCESS, CAS_FAILURE)
        {
            Ok(_) => Ok(()),

            Err(actual) => Err(actual),
        }
    }

    /// Store key metadata for a CAS insert attempt.
    ///
    /// # Safety
    ///
    /// Caller must have successfully claimed the slot via `cas_slot_value`.
    #[inline(always)]
    #[expect(clippy::indexing_slicing, reason = "bounds checked by debug_assert")]
    pub unsafe fn store_key_data_for_cas(&self, slot: usize, ikey: u64, keylenx: u8) {
        debug_assert!(
            slot < WIDTH_15,
            "store_key_data_for_cas: slot out of bounds"
        );
        self.ikey0[slot].store(ikey, WRITE_ORD);
        self.keylenx[slot].store(keylenx, WRITE_ORD);
    }

    // ========================================================================
    //  Slot Clearing
    // ========================================================================

    /// Clear a slot completely.
    #[inline]
    #[expect(clippy::indexing_slicing)]
    pub fn clear_slot(&self, slot: usize) {
        debug_assert!(slot < WIDTH_15, "clear_slot: slot out of bounds");
        self.keylenx[slot].store(0, WRITE_ORD);
        self.leaf_values[slot].store(StdPtr::null_mut(), WRITE_ORD);
    }

    /// Clear a slot and update permutation.
    pub fn clear_slot_and_permutation(&self, slot: usize) {
        self.clear_slot(slot);
        let mut perm = self.permutation();
        perm.remove(slot);
        self.set_permutation(perm);
    }

    // ========================================================================
    //  Slot Assignment Helpers
    // ========================================================================

    /// Check if slot 0 can be reused for a new key.
    ///
    /// # Safety
    ///
    /// Called under exclusive lock - uses unguarded prev load.
    #[must_use]
    #[inline(always)]
    pub fn can_reuse_slot0(&self, new_ikey: u64) -> bool {
        // SAFETY: Called under exclusive lock - no concurrent retirement.
        if unsafe { self.prev_unguarded() }.is_null() {
            return true;
        }
        self.ikey_bound() == new_ikey
    }

    // ========================================================================
    //  B-Link Navigation
    // ========================================================================

    /// Check if the next pointer is marked (split in progress).
    #[must_use]
    #[inline(always)]
    pub fn next_is_marked(&self) -> bool {
        (self.next.load(READ_ORD).addr() & 1) != 0
    }

    /// Mark the next pointer (during split).
    #[inline(always)]
    pub fn mark_next(&self) {
        let ptr: *mut Self = self.next.load(crate::ordering::RELAXED);
        let marked: *mut Self = ptr.map_addr(|addr: usize| addr | 1);
        self.next.store(marked, WRITE_ORD);
    }

    /// Unmark the next pointer.
    #[inline(always)]
    pub fn unmark_next(&self) {
        // SAFETY: Called during split completion - we hold the lock.
        let ptr: *mut Self = unsafe { self.safe_next_unguarded() };
        self.next.store(ptr, WRITE_ORD);
    }

    /// Wait for an in-progress split to complete.
    pub fn wait_for_split(&self) {
        while self.next_is_marked() {
            if self.version.is_deleted() {
                return;
            }
            for _ in 0..16 {
                std::hint::spin_loop();
                if !self.next_is_marked() {
                    return;
                }
            }
            let _ = self.version.stable();
            if self.version.is_deleted() {
                return;
            }
        }
    }

    /// CAS-based lock of the next pointer for split linking.
    fn lock_next(&self) -> *mut Self {
        loop {
            let next: *mut Self = self.next.load(READ_ORD);
            if Linker::is_marked(next) {
                self.wait_for_split();
                continue;
            }
            let marked: *mut Self = Linker::mark_ptr(next);
            match self
                .next
                .compare_exchange(next, marked, CAS_SUCCESS, CAS_FAILURE)
            {
                Ok(_) => return next,
                Err(_) => std::hint::spin_loop(),
            }
        }
    }

    /// Unlink this leaf from the B-link doubly-linked chain.
    ///
    /// # Safety
    ///
    /// - Caller must hold the version lock on this leaf
    /// - `self.prev()` must be non-null
    pub unsafe fn unlink_from_chain(&self) {
        use std::sync::atomic::Ordering as AtomicOrdering;

        let next: *mut Self = self.lock_next();
        let self_ptr: *mut Self = StdPtr::from_ref(self).cast_mut();
        let marked_self: *mut Self = Linker::mark_ptr(self_ptr);

        let final_prev: *mut Self;
        loop {
            // SAFETY: Called under exclusive lock - no concurrent retirement.
            let prev: *mut Self = unsafe { self.prev_unguarded() };
            debug_assert!(!prev.is_null(), "unlink_from_chain: prev must be non-null");

            // SAFETY: prev is non-null
            match unsafe { &*prev }.next.compare_exchange(
                self_ptr,
                marked_self,
                CAS_SUCCESS,
                CAS_FAILURE,
            ) {
                Ok(_) => {
                    final_prev = prev;
                    break;
                }
                Err(current) => {
                    if Linker::is_marked(current) {
                        // SAFETY: prev is valid
                        unsafe { (*prev).wait_for_split() };
                    }

                    std::hint::spin_loop();
                }
            }
        }

        if !next.is_null() {
            // SAFETY: next is non-null
            unsafe { (*next).set_prev(final_prev) };
        }

        std::sync::atomic::fence(AtomicOrdering::Release);

        // SAFETY: final_prev is non-null
        unsafe { (*final_prev).set_next(next) };
    }

    // ========================================================================
    //  Suffix Operations
    // ========================================================================

    /// Load external suffix bag pointer (reader).
    #[must_use]
    #[inline(always)]
    pub fn external_ksuf_ptr(&self) -> *mut SuffixBag<WIDTH_15> {
        self.external_ksuf.load(READ_ORD)
    }

    /// Check if this leaf has external suffix storage allocated.
    #[must_use]
    #[inline(always)]
    pub fn has_external_ksuf(&self) -> bool {
        !self.external_ksuf_ptr().is_null()
    }

    /// Get the suffix for a slot (checks inline first, then external).
    #[must_use]
    #[inline]
    pub fn ksuf(&self, slot: usize) -> Option<&[u8]> {
        debug_assert!(slot < WIDTH_15, "ksuf: slot out of bounds");

        if !self.has_ksuf(slot) {
            return None;
        }

        // SAFETY: Reader access, concurrent writes require lock
        let inline: &InlineSuffixBag<WIDTH_15, INLINE_KSUF_CAPACITY> =
            unsafe { &*self.inline_ksuf.get() };
        if let Some(suffix) = inline.get(slot) {
            return Some(suffix);
        }

        let ext_ptr: *mut SuffixBag<WIDTH_15> = self.external_ksuf_ptr();
        if ext_ptr.is_null() {
            return None;
        }

        // SAFETY: ext_ptr is non-null
        unsafe { (*ext_ptr).get(slot) }
    }

    /// Get the suffix for a slot, or an empty slice if none.
    #[must_use]
    #[inline(always)]
    pub fn ksuf_or_empty(&self, slot: usize) -> &[u8] {
        self.ksuf(slot).unwrap_or(&[])
    }

    /// Assign a suffix to a slot.
    ///
    /// # Safety
    ///
    /// - Caller must hold lock
    /// - `guard` must come from this tree's collector
    #[expect(clippy::indexing_slicing)]
    pub unsafe fn assign_ksuf(&self, slot: usize, suffix: &[u8], guard: &LocalGuard<'_>) {
        debug_assert!(slot < WIDTH_15, "assign_ksuf: slot out of bounds");
        debug_assert!(
            self.version.is_locked() || self.version.is_unpublished(),
            "assign_ksuf: caller must hold lock"
        );

        // SAFETY: We hold the lock
        let inline: &mut InlineSuffixBag<WIDTH_15, INLINE_KSUF_CAPACITY> =
            unsafe { &mut *self.inline_ksuf.get() };

        if inline.try_assign(slot, suffix) {
            self.keylenx[slot].store(KSUF_KEYLENX, WRITE_ORD);
            return;
        }

        let old_ext: *mut SuffixBag<WIDTH_15> = self.external_ksuf.load(crate::ordering::RELAXED);
        if !old_ext.is_null() {
            // SAFETY: old_ext is non-null
            let bag: &mut SuffixBag<WIDTH_15> = unsafe { &mut *old_ext };
            if bag.try_assign_in_place(slot, suffix) {
                inline.clear(slot);
                self.keylenx[slot].store(KSUF_KEYLENX, WRITE_ORD);
                return;
            }
        }

        // SAFETY: Same preconditions
        unsafe { self.assign_ksuf_slow(slot, suffix, guard) };
    }

    /// Slow path for suffix assignment.
    #[cold]
    #[inline(never)]
    #[expect(clippy::indexing_slicing)]
    unsafe fn assign_ksuf_slow(&self, slot: usize, suffix: &[u8], guard: &LocalGuard<'_>) {
        let perm = self.permutation();
        // SAFETY: Caller holds lock
        let inline: &mut InlineSuffixBag<WIDTH_15, INLINE_KSUF_CAPACITY> =
            unsafe { &mut *self.inline_ksuf.get() };

        // This infallible path aborts on OOM. Use `try_assign_ksuf` for fallible version.
        #[expect(clippy::expect_used)]
        let mut new_bag: SuffixBag<WIDTH_15> = inline
            .drain_to_external(&perm, slot, suffix)
            .expect("OOM: suffix bag allocation failed");

        let old_ext: *mut SuffixBag<WIDTH_15> = self.external_ksuf.load(crate::ordering::RELAXED);
        if !old_ext.is_null() {
            // SAFETY: old_ext is non-null
            let old_bag: &SuffixBag<WIDTH_15> = unsafe { &*old_ext };

            for i in 0..perm.size() {
                let s: usize = perm.get(i);
                if s != slot
                    && let Some(ext_suffix) = old_bag.get(s)
                {
                    new_bag.assign(s, ext_suffix);
                }
            }
        }

        let new_ptr: *mut SuffixBag<WIDTH_15> = Box::into_raw(Box::new(new_bag));
        self.external_ksuf.store(new_ptr, WRITE_ORD);

        if !old_ext.is_null() {
            // SAFETY: old_ext is non-null
            unsafe {
                guard.defer_retire(old_ext, |ptr, _| {
                    drop(Box::from_raw(ptr));
                });
            }
        }

        self.keylenx[slot].store(KSUF_KEYLENX, WRITE_ORD);
    }

    /// Assign a suffix during node initialization (e.g., splits).
    ///
    /// Unlike `assign_ksuf`, this assumes slots `0..slot` are already filled
    /// sequentially and doesn't rely on the permutation.
    ///
    /// # Safety
    /// - Caller must hold lock or node must be unpublished
    /// - `guard` must come from this tree's collector
    /// - Slots 0..slot must already be filled sequentially
    #[expect(clippy::indexing_slicing)]
    pub unsafe fn assign_ksuf_init(&self, slot: usize, suffix: &[u8], guard: &LocalGuard<'_>) {
        debug_assert!(slot < WIDTH_15, "assign_ksuf_init: slot out of bounds");
        debug_assert!(
            self.version.is_locked() || self.version.is_unpublished(),
            "assign_ksuf_init: caller must hold lock"
        );

        // SAFETY: We hold the lock
        let inline: &mut InlineSuffixBag<WIDTH_15, INLINE_KSUF_CAPACITY> =
            unsafe { &mut *self.inline_ksuf.get() };

        if inline.try_assign(slot, suffix) {
            self.keylenx[slot].store(KSUF_KEYLENX, WRITE_ORD);
            return;
        }

        let old_ext: *mut SuffixBag<WIDTH_15> = self.external_ksuf.load(crate::ordering::RELAXED);
        if !old_ext.is_null() {
            let bag: &mut SuffixBag<WIDTH_15> = unsafe { &mut *old_ext };
            if bag.try_assign_in_place(slot, suffix) {
                inline.clear(slot);
                self.keylenx[slot].store(KSUF_KEYLENX, WRITE_ORD);
                return;
            }
        }

        // SAFETY: Same preconditions
        unsafe { self.assign_ksuf_init_slow(slot, suffix, guard) };
    }

    /// Slow path for suffix assignment during initialization.
    #[cold]
    #[inline(never)]
    #[expect(clippy::indexing_slicing)]
    unsafe fn assign_ksuf_init_slow(&self, slot: usize, suffix: &[u8], guard: &LocalGuard<'_>) {
        // SAFETY: Caller holds lock
        let inline: &mut InlineSuffixBag<WIDTH_15, INLINE_KSUF_CAPACITY> =
            unsafe { &mut *self.inline_ksuf.get() };

        // Drain inline using sequential slot iteration (0..slot)
        #[expect(clippy::expect_used)]
        let mut new_bag: SuffixBag<WIDTH_15> = inline
            .drain_to_external_init(slot, suffix)
            .expect("OOM: suffix bag allocation failed");

        let old_ext: *mut SuffixBag<WIDTH_15> = self.external_ksuf.load(crate::ordering::RELAXED);
        if !old_ext.is_null() {
            let old_bag: &SuffixBag<WIDTH_15> = unsafe { &*old_ext };

            for s in 0..slot {
                if let Some(ext_suffix) = old_bag.get(s) {
                    new_bag.assign(s, ext_suffix);
                }
            }
        }

        let new_ptr: *mut SuffixBag<WIDTH_15> = Box::into_raw(Box::new(new_bag));
        self.external_ksuf.store(new_ptr, WRITE_ORD);

        if !old_ext.is_null() {
            unsafe {
                guard.defer_retire(old_ext, |ptr, _| {
                    drop(Box::from_raw(ptr));
                });
            }
        }

        self.keylenx[slot].store(KSUF_KEYLENX, WRITE_ORD);
    }

    /// Try to assign a suffix to a slot, returning error on allocation failure.
    ///
    /// This is the fallible version of `assign_ksuf`. It tries inline storage
    /// first, then falls back to external storage, returning an error if
    /// allocation fails.
    ///
    /// # Safety
    /// - Caller must hold lock and have called `mark_insert()`
    /// - `guard` must come from this tree's collector
    ///
    /// # Errors
    /// Returns `Err(AllocError)` if external bag allocation fails.
    #[expect(clippy::indexing_slicing)]
    pub unsafe fn try_assign_ksuf(
        &self,
        slot: usize,
        suffix: &[u8],
        guard: &LocalGuard<'_>,
    ) -> AllocResult<()> {
        debug_assert!(slot < WIDTH_15, "try_assign_ksuf: slot out of bounds");
        debug_assert!(
            self.version.is_locked() || self.version.is_unpublished(),
            "try_assign_ksuf: caller must hold lock or node must be unpublished"
        );

        // FAST PATH 1: Try inline storage first (no allocation!)
        let inline: &mut InlineSuffixBag<WIDTH_15, INLINE_KSUF_CAPACITY> =
            unsafe { &mut *self.inline_ksuf.get() };

        if inline.try_assign(slot, suffix) {
            self.keylenx[slot].store(KSUF_KEYLENX, WRITE_ORD);
            return Ok(());
        }

        // FAST PATH 2: Try external storage in-place (if exists and has room)
        let old_ext: *mut SuffixBag<WIDTH_15> = self.external_ksuf.load(crate::ordering::RELAXED);
        if !old_ext.is_null() {
            let bag: &mut SuffixBag<WIDTH_15> = unsafe { &mut *old_ext };
            if bag.try_assign_in_place(slot, suffix) {
                inline.clear(slot);
                self.keylenx[slot].store(KSUF_KEYLENX, WRITE_ORD);
                return Ok(());
            }
        }

        // SLOW PATH: Drain inline to external and allocate new external bag (fallible)
        unsafe { self.try_assign_ksuf_slow(slot, suffix, guard) }
    }

    /// Slow path for fallible suffix assignment: allocate/reallocate external bag.
    ///
    /// # Safety
    /// Same as `try_assign_ksuf`.
    ///
    /// # Errors
    /// Returns `Err(AllocError)` if allocation fails.
    #[cold]
    #[inline(never)]
    #[expect(clippy::indexing_slicing)]
    unsafe fn try_assign_ksuf_slow(
        &self,
        slot: usize,
        suffix: &[u8],
        guard: &LocalGuard<'_>,
    ) -> AllocResult<()> {
        debug_assert!(
            self.version.is_locked() || self.version.is_unpublished(),
            "try_assign_ksuf_slow: caller must hold lock or node must be unpublished"
        );

        let perm = self.permutation();
        let inline: &mut InlineSuffixBag<WIDTH_15, INLINE_KSUF_CAPACITY> =
            unsafe { &mut *self.inline_ksuf.get() };

        // Drain inline to a new external bag with the new suffix (FALLIBLE)
        let mut new_bag: SuffixBag<WIDTH_15> = inline.drain_to_external(&perm, slot, suffix)?;

        // Merge with existing external suffixes (if any)
        let old_ext: *mut SuffixBag<WIDTH_15> = self.external_ksuf.load(crate::ordering::RELAXED);
        if !old_ext.is_null() {
            let old_bag: &SuffixBag<WIDTH_15> = unsafe { &*old_ext };

            for i in 0..perm.size() {
                let s: usize = perm.get(i);

                if s != slot
                    && let Some(ext_suffix) = old_bag.get(s)
                {
                    new_bag.assign(s, ext_suffix);
                }
            }
        }

        // Install new external bag (FALLIBLE: use try_box)
        let new_ptr: *mut SuffixBag<WIDTH_15> =
            Box::into_raw(BoxAllocator::try_box_with_kind(new_bag, AllocKind::Suffix)?);
        self.external_ksuf.store(new_ptr, WRITE_ORD);

        // Retire old external bag
        if !old_ext.is_null() {
            unsafe {
                guard.defer_retire(old_ext, |ptr, _| {
                    drop(Box::from_raw(ptr));
                });
            }
        }

        self.keylenx[slot].store(KSUF_KEYLENX, WRITE_ORD);
        Ok(())
    }

    /// Fallible version of `assign_ksuf_init` for node initialization.
    ///
    /// # Errors
    /// Returns `AllocResult::Err` if memory allocation fails.
    ///
    /// # Safety
    /// Same as `assign_ksuf_init`.
    #[expect(clippy::indexing_slicing)]
    pub unsafe fn try_assign_ksuf_init(
        &self,
        slot: usize,
        suffix: &[u8],
        guard: &LocalGuard<'_>,
    ) -> AllocResult<()> {
        debug_assert!(slot < WIDTH_15, "try_assign_ksuf_init: slot out of bounds");
        debug_assert!(
            self.version.is_locked() || self.version.is_unpublished(),
            "try_assign_ksuf_init: caller must hold lock or node must be unpublished"
        );

        // FAST PATH: Try inline storage first
        let inline: &mut InlineSuffixBag<WIDTH_15, INLINE_KSUF_CAPACITY> =
            unsafe { &mut *self.inline_ksuf.get() };

        if inline.try_assign(slot, suffix) {
            self.keylenx[slot].store(KSUF_KEYLENX, WRITE_ORD);
            return Ok(());
        }

        // FAST PATH 2: Try external storage in-place
        let old_ext: *mut SuffixBag<WIDTH_15> = self.external_ksuf.load(crate::ordering::RELAXED);
        if !old_ext.is_null() {
            let bag: &mut SuffixBag<WIDTH_15> = unsafe { &mut *old_ext };
            if bag.try_assign_in_place(slot, suffix) {
                inline.clear(slot);
                self.keylenx[slot].store(KSUF_KEYLENX, WRITE_ORD);
                return Ok(());
            }
        }

        // SLOW PATH: Drain inline to external (fallible)
        unsafe { self.try_assign_ksuf_init_slow(slot, suffix, guard) }
    }

    /// Slow path for fallible suffix assignment during initialization.
    #[cold]
    #[inline(never)]
    #[expect(clippy::indexing_slicing)]
    unsafe fn try_assign_ksuf_init_slow(
        &self,
        slot: usize,
        suffix: &[u8],
        guard: &LocalGuard<'_>,
    ) -> AllocResult<()> {
        debug_assert!(
            self.version.is_locked() || self.version.is_unpublished(),
            "try_assign_ksuf_init_slow: caller must hold lock or node must be unpublished"
        );

        let inline: &mut InlineSuffixBag<WIDTH_15, INLINE_KSUF_CAPACITY> =
            unsafe { &mut *self.inline_ksuf.get() };

        // Drain inline using sequential slot iteration (0..slot) (FALLIBLE)
        let mut new_bag: SuffixBag<WIDTH_15> = inline.drain_to_external_init(slot, suffix)?;

        // Merge with existing external suffixes (if any)
        let old_ext: *mut SuffixBag<WIDTH_15> = self.external_ksuf.load(crate::ordering::RELAXED);
        if !old_ext.is_null() {
            let old_bag: &SuffixBag<WIDTH_15> = unsafe { &*old_ext };

            for s in 0..slot {
                if let Some(ext_suffix) = old_bag.get(s) {
                    new_bag.assign(s, ext_suffix);
                }
            }
        }

        // Install new external bag (FALLIBLE: use try_box)
        let new_ptr: *mut SuffixBag<WIDTH_15> =
            Box::into_raw(BoxAllocator::try_box_with_kind(new_bag, AllocKind::Suffix)?);
        self.external_ksuf.store(new_ptr, WRITE_ORD);

        // Retire old external bag
        if !old_ext.is_null() {
            unsafe {
                guard.defer_retire(old_ext, |ptr, _| {
                    drop(Box::from_raw(ptr));
                });
            }
        }

        self.keylenx[slot].store(KSUF_KEYLENX, WRITE_ORD);
        Ok(())
    }

    /// Clear the suffix from a slot.
    ///
    /// # Safety
    ///
    /// - Caller must hold lock
    #[expect(clippy::indexing_slicing)]
    pub unsafe fn clear_ksuf(&self, slot: usize, _guard: &LocalGuard<'_>) {
        debug_assert!(slot < WIDTH_15, "clear_ksuf: slot out of bounds");
        debug_assert!(
            self.version.is_locked(),
            "clear_ksuf: caller must hold lock"
        );

        // SAFETY: We hold the lock
        let inline: &mut InlineSuffixBag<WIDTH_15, INLINE_KSUF_CAPACITY> =
            unsafe { &mut *self.inline_ksuf.get() };
        inline.clear(slot);

        let ext_ptr: *mut SuffixBag<WIDTH_15> = self.external_ksuf.load(crate::ordering::RELAXED);
        if !ext_ptr.is_null() {
            // SAFETY: We hold the lock
            let bag: &mut SuffixBag<WIDTH_15> = unsafe { &mut *ext_ptr };
            bag.clear(slot);
        }

        self.keylenx[slot].store(0, WRITE_ORD);
    }

    /// Check if a slot's suffix equals the given suffix.
    #[must_use]
    #[inline]
    pub fn ksuf_equals(&self, slot: usize, suffix: &[u8]) -> bool {
        debug_assert!(slot < WIDTH_15, "ksuf_equals: slot out of bounds");

        if !self.has_ksuf(slot) {
            return false;
        }

        // SAFETY: Reader access
        let inline: &InlineSuffixBag<WIDTH_15, INLINE_KSUF_CAPACITY> =
            unsafe { &*self.inline_ksuf.get() };
        if inline.suffix_equals(slot, suffix) {
            return true;
        }

        let ext_ptr: *mut SuffixBag<WIDTH_15> = self.external_ksuf_ptr();
        if ext_ptr.is_null() {
            return false;
        }

        // SAFETY: ext_ptr is non-null
        unsafe { (*ext_ptr).suffix_equals(slot, suffix) }
    }

    /// Compare a slot's suffix with the given suffix.
    #[must_use]
    #[inline]
    pub fn ksuf_compare(&self, slot: usize, suffix: &[u8]) -> Option<std::cmp::Ordering> {
        debug_assert!(slot < WIDTH_15, "ksuf_compare: slot out of bounds");

        if !self.has_ksuf(slot) {
            return None;
        }

        // SAFETY: Reader access
        let inline: &InlineSuffixBag<WIDTH_15, INLINE_KSUF_CAPACITY> =
            unsafe { &*self.inline_ksuf.get() };
        if let Some(cmp) = inline.suffix_compare(slot, suffix) {
            return Some(cmp);
        }

        let ext_ptr: *mut SuffixBag<WIDTH_15> = self.external_ksuf_ptr();
        if ext_ptr.is_null() {
            return None;
        }

        // SAFETY: ext_ptr is non-null
        unsafe { (*ext_ptr).suffix_compare(slot, suffix) }
    }

    /// Check if a slot's key matches the given key.
    #[must_use]
    #[inline]
    pub fn ksuf_matches(&self, slot: usize, ikey: u64, suffix: &[u8]) -> bool {
        debug_assert!(slot < WIDTH_15, "ksuf_matches: slot out of bounds");

        if self.ikey(slot) != ikey {
            return false;
        }

        if suffix.is_empty() {
            !self.has_ksuf(slot)
        } else {
            self.ksuf_equals(slot, suffix)
        }
    }

    /// Match result for layer-aware key comparison.
    ///
    /// Returns:
    /// * 1 - Exact match
    /// * 0 - Same ikey but different key
    /// * -8 - Slot is a layer pointer
    #[must_use]
    #[inline(always)]
    pub fn ksuf_match_result(&self, slot: usize, keylenx: u8, suffix: &[u8]) -> i32 {
        debug_assert!(slot < WIDTH_15, "ksuf_match_result: slot out of bounds");

        let stored_keylenx: u8 = self.keylenx(slot);

        if Self::keylenx_is_layer(stored_keylenx) {
            return -8; // MATCH_RESULT_LAYER
        }

        if !self.has_ksuf(slot) {
            if stored_keylenx == keylenx && suffix.is_empty() {
                return 1; // MATCH_RESULT_EXACT
            }
            return 0; // MATCH_RESULT_MISMATCH
        }

        if suffix.is_empty() {
            return 0; // MATCH_RESULT_MISMATCH
        }

        i32::from(self.ksuf_equals(slot, suffix))
    }

    // ========================================================================
    //  Prefetch Operations
    // ========================================================================

    /// Prefetch the ikey at the given slot into CPU cache.
    #[inline(always)]
    #[expect(clippy::indexing_slicing)]
    pub fn prefetch_ikey(&self, slot: usize) {
        debug_assert!(slot < WIDTH_15);
        crate::prefetch::prefetch_read(&raw const self.ikey0[slot]);
    }

    /// Prefetch leaf node data for range scans.
    #[inline(always)]
    pub fn prefetch(&self) {
        let self_ptr: *const u8 = StdPtr::from_ref::<Self>(self).cast::<u8>();
        // SAFETY: Offsets within struct bounds
        unsafe {
            crate::prefetch::prefetch_read(self_ptr.add(128)); // ikey0[0..7]
            crate::prefetch::prefetch_read(self_ptr.add(192)); // ikey0[8..14] + keylenx
            crate::prefetch::prefetch_read(self_ptr.add(256)); // keylenx + leaf_values
            crate::prefetch::prefetch_read(self_ptr.add(320)); // leaf_values + inline_values
        }
    }

    /// Prefetch for point lookups.
    #[inline(always)]
    pub fn prefetch_for_search(&self) {
        let self_ptr: *const u8 = StdPtr::from_ref::<Self>(self).cast::<u8>();
        // SAFETY: Offsets within struct bounds
        unsafe {
            crate::prefetch::prefetch_read(self_ptr.add(64)); // permutation
            crate::prefetch::prefetch_read(self_ptr.add(128)); // ikey0[0..7]
            crate::prefetch::prefetch_read(self_ptr.add(192)); // ikey0[8..14]
        }
    }

    // ========================================================================
    //  Split Operations
    // ========================================================================

    /// Calculate the optimal split point.
    #[must_use]
    pub fn calculate_split_point(
        &self,
        _insert_pos: usize,
        insert_ikey: u64,
    ) -> Option<crate::value::SplitPoint> {
        use std::cmp::Ordering;

        let perm = self.permutation();
        let size = perm.size();

        if size == 0 {
            return None;
        }

        let mut split_pos = size / 2;
        if split_pos == 0 {
            return None;
        }

        // Adjust for equal ikeys: if keys at split boundary are equal,
        // move split point to keep equal keys together.
        //
        // CRITICAL INVARIANT: All entries with the same ikey MUST end up in
        // the same leaf. Otherwise, routing (which uses ikey comparison) will
        // send lookups to the wrong leaf.
        //
        // The split_ikey becomes the separator in the parent internode:
        // - Keys with ikey < split_ikey route to left leaf
        // - Keys with ikey >= split_ikey route to right leaf
        //
        // So if we have entries with equal ikeys at the boundary, they must
        // ALL go to the right leaf (since routing uses >=).
        while split_pos > 0 && split_pos < size {
            let left_slot = perm.get(split_pos - 1);
            let right_slot = perm.get(split_pos);
            let left_ikey = self.ikey(left_slot);
            let right_ikey = self.ikey(right_slot);

            if left_ikey == right_ikey {
                // Equal ikeys at boundary - must keep them together!
                match insert_ikey.cmp(&left_ikey) {
                    // Insert has same ikey - move split right
                    Ordering::Equal => split_pos += 1,
                    // Insert goes before this group - move split left
                    Ordering::Less | Ordering::Greater => split_pos -= 1,
                }
            } else {
                break;
            }
        }

        if split_pos == 0 || split_pos >= size {
            return None;
        }

        let split_slot = perm.get(split_pos);
        let split_ikey = self.ikey(split_slot);

        Some(crate::value::SplitPoint {
            pos: split_pos,
            split_ikey,
        })
    }

    /// Split this leaf at `split_pos` using a pre-allocated target.
    ///
    /// For true-inline leaves, this copies inline value bits rather than moving pointers.
    ///
    /// # Safety
    ///
    /// - Caller must hold the leaf lock
    /// - `new_leaf_ptr` must point to valid, initialized leaf memory
    /// - `guard` must be valid
    #[expect(clippy::indexing_slicing)]
    pub unsafe fn split_into_preallocated(
        &self,
        split_pos: usize,
        new_leaf_ptr: *mut Self,
        guard: &LocalGuard<'_>,
    ) -> (u64, crate::value::InsertTarget) {
        // CRITICAL (Help-Along Protocol): Initialize new leaf's version for split
        // BEFORE any data is written.
        let split_version = crate::nodeversion::NodeVersion::new_for_split(&self.version);

        // SAFETY: new_leaf is not yet visible to other threads
        unsafe {
            StdPtr::write(
                StdPtr::addr_of!((*new_leaf_ptr).version).cast_mut(),
                split_version,
            );
        }

        let perm = self.permutation();
        let size = perm.size();
        let entries_to_move = size - split_pos;

        // SAFETY: new_leaf_ptr is valid
        let new_leaf = unsafe { &*new_leaf_ptr };

        // Copy entries from old leaf to new leaf
        for i in 0..entries_to_move {
            let old_logical = split_pos + i;
            let old_slot = perm.get(old_logical);
            let new_slot = i; // New leaf uses positions 0..entries_to_move

            // Copy ikey and keylenx
            let ikey = self.ikey(old_slot);
            let keylenx_val = self.keylenx(old_slot);
            new_leaf.set_ikey(new_slot, ikey);
            new_leaf.set_keylenx(new_slot, keylenx_val);

            // Handle value: check if it's inline or layer pointer
            let ptr = self.leaf_values[old_slot].load(READ_ORD);
            if InlineSentinel::is_inline_sentinel(ptr) {
                // Inline value: copy the bits
                let bits = self.inline_values[old_slot].load(READ_ORD);
                new_leaf.inline_values[new_slot].store(bits, WRITE_ORD);
                new_leaf.leaf_values[new_slot]
                    .store(InlineSentinel::inline_sentinel_ptr(), WRITE_ORD);
            } else {
                // Layer pointer: move the pointer
                let moved_ptr = self.take_leaf_value_ptr(old_slot);
                new_leaf.set_leaf_value_ptr(new_slot, moved_ptr);
            }

            // Copy suffix if present
            if keylenx_val == KSUF_KEYLENX {
                if let Some(suffix) = self.ksuf(old_slot) {
                    // SAFETY: We hold lock on both leaves
                    unsafe { new_leaf.assign_ksuf_init(new_slot, suffix, guard) };
                }
                // Clear suffix from old slot
                // SAFETY: We hold lock
                unsafe { self.clear_ksuf(old_slot, guard) };
            }

            // Clear old slot's value (not the suffix, already cleared)
            self.leaf_values[old_slot].store(StdPtr::null_mut(), WRITE_ORD);
        }

        // Set up new leaf's permutation
        let new_perm = Permuter15::make_sorted(entries_to_move);
        new_leaf.set_permutation(new_perm);

        // Update old leaf's permutation
        let mut old_perm_updated = perm;
        old_perm_updated.set_size(split_pos);
        self.set_permutation(old_perm_updated);

        // Get split key (first key of new leaf)
        let split_ikey = new_leaf.ikey(0);

        (split_ikey, crate::value::InsertTarget::Left)
    }

    /// Move ALL entries to a new right leaf.
    ///
    /// # Safety
    ///
    /// Same as `split_into_preallocated`.
    #[expect(clippy::indexing_slicing)]
    pub unsafe fn split_all_to_right_preallocated(
        &self,
        new_leaf_ptr: *mut Self,
        guard: &LocalGuard<'_>,
    ) -> (u64, crate::value::InsertTarget) {
        // CRITICAL (Help-Along Protocol): Initialize new leaf's version for split
        let split_version = crate::nodeversion::NodeVersion::new_for_split(&self.version);

        // SAFETY: new_leaf is not yet visible to other threads
        unsafe {
            StdPtr::write(
                StdPtr::addr_of!((*new_leaf_ptr).version).cast_mut(),
                split_version,
            );
        }

        let perm = self.permutation();
        let size = perm.size();

        // SAFETY: new_leaf_ptr is valid
        let new_leaf = unsafe { &*new_leaf_ptr };

        // Copy all entries to new leaf
        for i in 0..size {
            let old_slot = perm.get(i);
            let new_slot = i;

            let ikey = self.ikey(old_slot);
            let keylenx_val = self.keylenx(old_slot);
            new_leaf.set_ikey(new_slot, ikey);
            new_leaf.set_keylenx(new_slot, keylenx_val);

            let ptr = self.leaf_values[old_slot].load(READ_ORD);
            if InlineSentinel::is_inline_sentinel(ptr) {
                let bits = self.inline_values[old_slot].load(READ_ORD);
                new_leaf.inline_values[new_slot].store(bits, WRITE_ORD);
                new_leaf.leaf_values[new_slot]
                    .store(InlineSentinel::inline_sentinel_ptr(), WRITE_ORD);
            } else {
                let moved_ptr = self.take_leaf_value_ptr(old_slot);
                new_leaf.set_leaf_value_ptr(new_slot, moved_ptr);
            }

            if keylenx_val == KSUF_KEYLENX {
                if let Some(suffix) = self.ksuf(old_slot) {
                    unsafe { new_leaf.assign_ksuf_init(new_slot, suffix, guard) };
                }
                unsafe { self.clear_ksuf(old_slot, guard) };
            }

            self.leaf_values[old_slot].store(StdPtr::null_mut(), WRITE_ORD);
        }

        let new_perm = Permuter15::make_sorted(size);
        new_leaf.set_permutation(new_perm);

        // Old leaf becomes empty
        self.set_permutation(Permuter15::empty());

        let split_ikey = new_leaf.ikey(0);
        (split_ikey, crate::value::InsertTarget::Right)
    }

    /// Link this leaf to a new sibling (B-link tree threading).
    ///
    /// # Safety
    ///
    /// - `new_sibling` must be a valid pointer to a freshly allocated leaf
    /// - Caller must hold the leaf lock
    pub unsafe fn link_sibling(&self, new_sibling: *mut Self) {
        use std::sync::atomic::Ordering as AtomicOrdering;

        // Lock our next pointer
        let old_next = self.lock_next();

        // SAFETY: new_sibling is valid
        let sibling = unsafe { &*new_sibling };

        // Set up sibling's links
        sibling.set_prev(StdPtr::from_ref(self).cast_mut());
        sibling.set_next(old_next);

        // Update old_next's prev pointer
        if !old_next.is_null() {
            // SAFETY: old_next is non-null
            unsafe { (*old_next).set_prev(new_sibling) };
        }

        // Release fence for visibility
        std::sync::atomic::fence(AtomicOrdering::Release);

        // Unlock by storing unmarked new_sibling as next
        self.set_next(new_sibling);
    }

    /// Initialize a leaf node directly at the given pointer.
    ///
    /// # Safety
    ///
    /// - `ptr` must be valid, properly aligned, and point to uninitialized memory
    #[inline]
    pub unsafe fn init_at(ptr: *mut Self, is_root: bool) {
        unsafe {
            StdPtr::write_bytes(ptr, 0, 1);

            let node = &mut *ptr;
            StdPtr::write(&raw mut node.version, NodeVersion::new(true));
            if is_root {
                node.version.mark_root();
            }
            StdPtr::write(&raw mut node.modstate, AtomicU8::new(MODSTATE_INSERT));
            StdPtr::write(&raw mut node.permutation, AtomicPermuter15::new());
            StdPtr::write(
                &raw mut (*ptr).inline_ksuf,
                UnsafeCell::new(InlineSuffixBag::new()),
            );
        }
    }
}

// ============================================================================
//  Leaf Value Trait Implementations for LeafNode15TrueInline<V> (True-Inline)
// ============================================================================

impl<V: InlineBits> LeafValueLoad<TrueInlineSlot<V>> for LeafNode15TrueInline<V> {
    #[inline(always)]
    fn try_load_output(&self, slot: usize) -> Option<V> {
        self.inline_get(slot)
    }
}

impl<V: InlineBits> LeafValueStore<TrueInlineSlot<V>> for LeafNode15TrueInline<V> {
    #[inline(always)]
    fn store_value_output(&self, slot: usize, output: &V, _guard: &LocalGuard<'_>) {
        self.inline_set(slot, *output);
    }
}

impl<V: InlineBits> LeafValueUpdate<TrueInlineSlot<V>> for LeafNode15TrueInline<V> {
    #[inline(always)]
    fn replace_value_output(&self, slot: usize, new_output: V, _guard: &LocalGuard<'_>) -> V {
        self.inline_swap(slot, new_output)
    }
}

impl<V: InlineBits> LeafValueClear<TrueInlineSlot<V>> for LeafNode15TrueInline<V> {
    #[inline(always)]
    fn clear_value_output(&self, slot: usize, _guard: &LocalGuard<'_>) {
        self.inline_clear(slot);
    }
}

impl<V: InlineBits> LeafValueTake<TrueInlineSlot<V>> for LeafNode15TrueInline<V> {
    #[inline(always)]
    fn take_value_output(&self, slot: usize, _guard: &LocalGuard<'_>) -> Option<V> {
        let ptr = self.leaf_value_ptr(slot);

        if !InlineSentinel::is_inline_sentinel(ptr) {
            return None;
        }

        Some(self.inline_take(slot))
    }
}

// SAFETY: LeafNode15TrueInline is safe to send between threads because:
// - All fields use atomic types for concurrent access
// - V is bound by InlineBits which requires Send + Sync
unsafe impl<V: InlineBits> Send for LeafNode15TrueInline<V> {}
unsafe impl<V: InlineBits> Sync for LeafNode15TrueInline<V> {}

// ============================================================================
//  TreeLeafNode Trait Implementation
// ============================================================================

impl<V: InlineBits> crate::leaf_trait::TreeLeafNode<TrueInlineSlot<V>> for LeafNode15TrueInline<V> {
    type Perm = Permuter15;
    // Internodes are non-generic (they don't store values, only keys and child pointers)
    type Internode = crate::internode::InternodeNode;
    const WIDTH: usize = WIDTH_15;
    const SPLIT_THRESHOLD: usize = 12; // 80% of 15

    // Construction
    #[inline]
    fn new_boxed() -> Box<Self> {
        Self::new()
    }

    #[inline]
    fn new_root_boxed() -> Box<Self> {
        Self::new_root()
    }

    #[inline]
    fn new_layer_root_boxed() -> Box<Self> {
        Self::new_layer_root()
    }

    // NodeVersion
    #[inline(always)]
    fn version(&self) -> &crate::nodeversion::NodeVersion {
        Self::version(self)
    }

    // Permutation
    #[inline(always)]
    fn permutation(&self) -> Self::Perm {
        Self::permutation(self)
    }

    #[inline(always)]
    fn set_permutation(&self, perm: Self::Perm) {
        Self::set_permutation(self, perm);
    }

    #[inline(always)]
    fn permutation_raw(&self) -> u64 {
        Self::permutation_raw(self)
    }

    // Key operations
    #[inline(always)]
    fn ikey(&self, slot: usize) -> u64 {
        Self::ikey(self, slot)
    }

    #[inline(always)]
    fn ikey_relaxed(&self, slot: usize) -> u64 {
        Self::ikey_relaxed(self, slot)
    }

    #[inline(always)]
    fn set_ikey(&self, slot: usize, ikey: u64) {
        Self::set_ikey(self, slot, ikey);
    }

    #[inline(always)]
    fn ikey_bound(&self) -> u64 {
        Self::ikey_bound(self)
    }

    #[inline(always)]
    fn keylenx(&self, slot: usize) -> u8 {
        Self::keylenx(self, slot)
    }

    #[inline(always)]
    fn set_keylenx(&self, slot: usize, keylenx: u8) {
        Self::set_keylenx(self, slot, keylenx);
    }

    #[inline(always)]
    fn is_layer(&self, slot: usize) -> bool {
        Self::is_layer(self, slot)
    }

    #[inline(always)]
    fn has_ksuf(&self, slot: usize) -> bool {
        Self::has_ksuf(self, slot)
    }

    // Value operations
    #[inline(always)]
    fn leaf_value_ptr(&self, slot: usize) -> *mut u8 {
        Self::leaf_value_ptr(self, slot)
    }

    #[inline(always)]
    fn set_leaf_value_ptr(&self, slot: usize, ptr: *mut u8) {
        Self::set_leaf_value_ptr(self, slot, ptr);
    }

    #[inline(always)]
    fn update_leaf_value_in_place(&self, slot: usize, ptr: *mut u8) {
        Self::update_leaf_value_in_place(self, slot, ptr);
    }

    #[inline(always)]
    fn cas_slot_value(
        &self,
        slot: usize,
        expected: *mut u8,
        new_value: *mut u8,
    ) -> Result<(), *mut u8> {
        Self::cas_slot_value(self, slot, expected, new_value)
    }

    // Slot clearing
    #[inline]
    fn clear_slot(&self, slot: usize) {
        Self::clear_slot(self, slot);
    }

    fn clear_slot_and_permutation(&self, slot: usize) {
        Self::clear_slot_and_permutation(self, slot);
    }

    // Navigation
    /// # Safety
    ///
    /// Trait methods use unguarded loads - intended for locked operations.
    #[inline(always)]
    fn safe_next(&self) -> *mut Self {
        // SAFETY: Trait methods are called during locked operations.
        unsafe { Self::safe_next_unguarded(self) }
    }

    #[inline(always)]
    fn next_is_marked(&self) -> bool {
        Self::next_is_marked(self)
    }

    #[inline(always)]
    fn set_next(&self, next: *mut Self) {
        Self::set_next(self, next);
    }

    #[inline(always)]
    fn mark_next(&self) {
        Self::mark_next(self);
    }

    #[inline(always)]
    fn unmark_next(&self) {
        Self::unmark_next(self);
    }

    /// # Safety
    ///
    /// Trait methods use unguarded loads - intended for locked operations.
    #[inline(always)]
    fn prev(&self) -> *mut Self {
        // SAFETY: Trait methods are called during locked operations.
        unsafe { Self::prev_unguarded(self) }
    }

    #[inline(always)]
    fn set_prev(&self, prev: *mut Self) {
        Self::set_prev(self, prev);
    }

    unsafe fn unlink_from_chain(&self) {
        // SAFETY: Same preconditions
        unsafe { Self::unlink_from_chain(self) };
    }

    /// # Safety
    ///
    /// Trait methods use unguarded loads - intended for locked operations.
    #[inline(always)]
    fn parent(&self) -> *mut u8 {
        // SAFETY: Trait methods are called during locked operations.
        unsafe { Self::parent_unguarded(self) }
    }

    #[inline(always)]
    fn set_parent(&self, parent: *mut u8) {
        Self::set_parent(self, parent);
    }

    // Slot assignment helpers
    #[inline(always)]
    fn can_reuse_slot0(&self, new_ikey: u64) -> bool {
        Self::can_reuse_slot0(self, new_ikey)
    }

    // CAS insert support
    unsafe fn store_key_data_for_cas(&self, slot: usize, ikey: u64, keylenx: u8) {
        // SAFETY: Same preconditions
        unsafe { Self::store_key_data_for_cas(self, slot, ikey, keylenx) };
    }

    #[inline(always)]
    fn load_slot_value(&self, slot: usize) -> *mut u8 {
        Self::load_slot_value(self, slot)
    }

    /// # Safety
    ///
    /// Trait methods use unguarded loads - intended for locked operations.
    #[inline(always)]
    fn next_raw(&self) -> *mut Self {
        // SAFETY: Trait methods are called during locked operations.
        unsafe { Self::next_raw_unguarded(self) }
    }

    fn wait_for_split(&self) {
        Self::wait_for_split(self);
    }

    // Split operations
    fn calculate_split_point(
        &self,
        insert_pos: usize,
        insert_ikey: u64,
    ) -> Option<crate::value::SplitPoint> {
        Self::calculate_split_point(self, insert_pos, insert_ikey)
    }

    unsafe fn split_into_preallocated(
        &self,
        split_pos: usize,
        new_leaf_ptr: *mut Self,
        guard: &LocalGuard<'_>,
    ) -> (u64, crate::value::InsertTarget) {
        // SAFETY: Same preconditions
        unsafe { Self::split_into_preallocated(self, split_pos, new_leaf_ptr, guard) }
    }

    unsafe fn split_all_to_right_preallocated(
        &self,
        new_leaf_ptr: *mut Self,
        guard: &LocalGuard<'_>,
    ) -> (u64, crate::value::InsertTarget) {
        // SAFETY: Same preconditions
        unsafe { Self::split_all_to_right_preallocated(self, new_leaf_ptr, guard) }
    }

    unsafe fn link_sibling(&self, new_sibling: *mut Self) {
        // SAFETY: Same preconditions
        unsafe { Self::link_sibling(self, new_sibling) };
    }

    // Suffix operations
    fn ksuf(&self, slot: usize) -> Option<&[u8]> {
        Self::ksuf(self, slot)
    }

    unsafe fn assign_ksuf(&self, slot: usize, suffix: &[u8], guard: &LocalGuard<'_>) {
        // SAFETY: Same preconditions
        unsafe { Self::assign_ksuf(self, slot, suffix, guard) };
    }

    unsafe fn assign_ksuf_init(&self, slot: usize, suffix: &[u8], guard: &LocalGuard<'_>) {
        // SAFETY: Same preconditions
        unsafe { Self::assign_ksuf_init(self, slot, suffix, guard) };
    }

    unsafe fn clear_ksuf(&self, slot: usize, guard: &LocalGuard<'_>) {
        // SAFETY: Same preconditions
        unsafe { Self::clear_ksuf(self, slot, guard) };
    }

    #[inline(always)]
    fn take_leaf_value_ptr(&self, slot: usize) -> *mut u8 {
        Self::take_leaf_value_ptr(self, slot)
    }

    // Suffix comparison
    #[inline]
    fn ksuf_equals(&self, slot: usize, suffix: &[u8]) -> bool {
        Self::ksuf_equals(self, slot, suffix)
    }

    #[inline]
    fn ksuf_compare(&self, slot: usize, suffix: &[u8]) -> Option<std::cmp::Ordering> {
        Self::ksuf_compare(self, slot, suffix)
    }

    #[inline]
    fn ksuf_matches(&self, slot: usize, ikey: u64, suffix: &[u8]) -> bool {
        Self::ksuf_matches(self, slot, ikey, suffix)
    }

    #[inline(always)]
    fn ksuf_match_result(&self, slot: usize, keylenx: u8, suffix: &[u8]) -> i32 {
        Self::ksuf_match_result(self, slot, keylenx, suffix)
    }

    // Prefetch
    #[inline(always)]
    fn prefetch(&self) {
        Self::prefetch(self);
    }

    #[inline(always)]
    fn prefetch_ikey(&self, slot: usize) {
        Self::prefetch_ikey(self, slot);
    }

    #[inline(always)]
    fn prefetch_for_search(&self) {
        Self::prefetch_for_search(self);
    }

    // ModState
    #[inline(always)]
    fn modstate(&self) -> u8 {
        Self::modstate(self)
    }

    #[inline(always)]
    fn set_modstate(&self, state: u8) {
        Self::set_modstate(self, state);
    }

    #[inline(always)]
    fn deleted_layer(&self) -> bool {
        Self::deleted_layer(self)
    }

    #[inline(always)]
    fn mark_deleted_layer(&self) {
        Self::mark_deleted_layer(self);
    }

    #[inline(always)]
    fn mark_remove(&self) {
        Self::mark_remove(self);
    }

    #[inline(always)]
    fn is_removing(&self) -> bool {
        Self::is_removing(self)
    }

    #[inline(always)]
    fn is_empty_state(&self) -> bool {
        Self::is_empty_state(self)
    }

    #[inline(always)]
    fn mark_empty(&self) {
        Self::mark_empty(self);
    }

    #[inline(always)]
    fn clear_empty_state(&self) {
        Self::clear_empty_state(self);
    }
}

// ============================================================================
//  LayerCapableLeaf Trait Implementation
// ============================================================================

impl<V: InlineBits> crate::leaf_trait::LayerCapableLeaf<TrueInlineSlot<V>>
    for LeafNode15TrueInline<V>
{
    fn try_clone_output(&self, slot: usize) -> Option<V> {
        // For true-inline, the value is Copy - just load it
        self.inline_get(slot)
    }

    #[expect(clippy::cast_possible_truncation)]
    unsafe fn assign_from_key_arc(
        &self,
        slot: usize,
        key: &crate::key::Key<'_>,
        value: Option<V>,
        guard: &LocalGuard<'_>,
    ) {
        // Store ikey
        self.set_ikey(slot, key.ikey());

        // Store value using inline storage
        if let Some(v) = value {
            self.inline_set(slot, v);
        }

        // Handle suffix if present
        if key.has_suffix() {
            self.set_keylenx(slot, KSUF_KEYLENX);
            // SAFETY: Same preconditions as this function
            // Pass initializing=false: this is a normal insert, not split initialization
            unsafe { self.assign_ksuf(slot, key.suffix(), guard) };
        } else {
            let len = key.current_len().min(8) as u8;
            self.set_keylenx(slot, len);
        }
    }
}

// ============================================================================
//  Drop Implementation
// ============================================================================

impl<V: InlineBits> Drop for LeafNode15TrueInline<V> {
    /// FIXED: Drop the leaf node, cleaning up the external suffix bag.
    /// This was leading to OOM's in benchmarks for [`crate::MassTree15Inline`].
    ///
    /// # What's cleaned up
    /// - `external_ksuf`: Heap-allocated [`SuffixBag`] for overflow suffix storage
    ///
    /// # What's not cleaned up here
    /// - `inline_values`: Just [`AtomicU64`] bits, no heap allocation
    /// - `inline_ksuf`: Embedded in struct, dropped automatically
    /// - Layer Pointers: Ownded by tree, freed during teardown traversal
    /// - Values: [`TrueInlineSlot::NEEDS_RETIREMENT = false`], no cleanup needed
    fn drop(&mut self) {
        // Free external suffix bag if allocated
        let ext_ptr: *mut SuffixBag<WIDTH_15> = self.external_ksuf.load(RELAXED);

        if !ext_ptr.is_null() {
            // SAFETY: ext_ptr came from [`Box::into_raw`] in `assign_ksuf` or
            // `ensure_external_ksuf`. We have `&mut self` so no concurrent access.
            unsafe { drop(Box::from_raw(ext_ptr)) }
        }
    }
}

#[cfg(test)]
mod drop_tests;
