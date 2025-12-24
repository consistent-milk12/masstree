//! Filepath: src/leaf24.rs
//!
//! Leaf node for [`MassTree`] with WIDTH=24 (24 slots).
//!
//! This module provides `LeafNode24`, a leaf node variant optimized for reduced
//! split frequency by using 24 slots instead of the standard 15. The key difference
//! is the use of [`AtomicPermuter24`] (u128) instead of `AtomicU64` for permutation.
//!
//! # Design
//!
//! The 24-slot design requires 5 bits per slot (values 0-23) vs 4 bits for WIDTH=15.
//! Total: 5 (size) + 24×5 (slots) = 125 bits, requiring u128 storage.

use std::fmt as StdFmt;
use std::marker::PhantomData;
use std::ptr as StdPtr;
use std::sync::atomic::{AtomicPtr, AtomicU8, AtomicU64};

use crate::nodeversion::NodeVersion;
use crate::ordering::{READ_ORD, RELAXED, WRITE_ORD};
use crate::permuter24::{AtomicPermuter24, Permuter24};
use crate::slot::ValueSlot;
use crate::suffix::SuffixBag;
use seize::{Guard, LocalGuard};

mod cas;
mod freeze;

pub use cas::CasPermutationFailure24;

/// Special keylenx value indicating key has a suffix.
pub const KSUF_KEYLENX: u8 = 64;

/// Base keylenx value indicating a layer pointer (>= this means layer).
pub const LAYER_KEYLENX: u8 = 128;

/// Width constant for LeafNode24.
pub const WIDTH_24: usize = 24;

/// Modification state values (shared with leaf.rs).
#[repr(u8)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ModState24 {
    /// Node is in insert mode (normal operation).
    Insert = 0,

    /// Node is being removed.
    Remove = 1,

    /// Node's layer has been deleted.
    DeletedLayer = 2,
}

/// Leaf node with 24 slots using u128 permutation.
///
/// # Concurrency Model
///
/// Same as `LeafNode<S, WIDTH>` but uses [`AtomicPermuter24`] for the permutation
/// field, enabling 24 slots instead of 15.
///
/// # Memory Layout
///
/// ```text
/// Cache Line 0 (64 bytes): version + modstate + padding
/// Cache Line 1 (64 bytes): permutation (u128 = 16 bytes) + padding (48 bytes)
/// Cache Lines 2+: keys, keylenx, values (24 slots each)
/// ```
#[repr(C, align(64))]
pub struct LeafNode24<S: ValueSlot> {
    // ========================================================================
    // Cache Line 0: Version + metadata (read-heavy, rarely written)
    // ========================================================================
    /// Version for optimistic concurrency control.
    version: NodeVersion,

    /// Modification state for suffix operations.
    modstate: ModState24,

    /// Padding to fill cache line 0 and separate version from permutation.
    ///
    /// **Purpose**: Eliminate false sharing between `version` and `permutation`.
    /// - `version` is CAS'd during splits (infrequent)
    /// - `permutation` is CAS'd on every CAS insert (frequent)
    _pad0: [u8; 55],

    // ========================================================================
    // Cache Line 1: Permutation (CAS-heavy, isolated for performance)
    // ========================================================================
    /// Permutation using u128 for 24-slot support.
    /// Store is linearization point for new slot visibility.
    permutation: AtomicPermuter24,

    /// Padding to fill cache line 1.
    /// u128 = 16 bytes, so need 64 - 16 = 48 bytes padding.
    _pad1: [u8; 48],

    // ========================================================================
    // Cache Lines 2+: Keys and values (read during search, written on insert)
    // ========================================================================
    /// 8-byte keys for each slot.
    ikey0: [AtomicU64; WIDTH_24],

    /// Key length/type for each slot.
    /// Values 0-8: inline key length
    /// Value 64: has suffix
    /// Value ≥128: is layer
    keylenx: [AtomicU8; WIDTH_24],

    /// Values/layer pointers for each slot.
    /// Stores Arc<V> raw pointer or layer pointer as *mut u8.
    /// Type is determined by keylenx: if < `LAYER_KEYLENX` → Arc<V>, else → layer node.
    leaf_values: [AtomicPtr<u8>; WIDTH_24],

    /// Suffix storage (atomic pointer for concurrent access).
    ksuf: AtomicPtr<SuffixBag<WIDTH_24>>,

    /// Next leaf with mark bit in LSB for split coordination.
    next: AtomicPtr<Self>,

    /// Previous leaf.
    prev: AtomicPtr<Self>,

    /// Parent internode.
    parent: AtomicPtr<u8>,

    /// Phantom for slot type.
    _marker: PhantomData<S>,
}

impl<S: ValueSlot> StdFmt::Debug for LeafNode24<S> {
    fn fmt(&self, f: &mut StdFmt::Formatter<'_>) -> StdFmt::Result {
        f.debug_struct("LeafNode24")
            .field("size", &self.size())
            .field("is_root", &self.version.is_root())
            .field("has_parent", &(!self.parent().is_null()))
            .finish_non_exhaustive()
    }
}

impl<S: ValueSlot> LeafNode24<S> {
    // ============================================================================
    //  Constructor Methods
    // ============================================================================

    /// Create a new leaf node (unboxed).
    #[must_use]
    pub fn new_with_root(is_root: bool) -> Self {
        let version: NodeVersion = NodeVersion::new(true);
        if is_root {
            version.mark_root();
        }

        Self {
            version,
            modstate: ModState24::Insert,
            _pad0: [0; 55],
            permutation: AtomicPermuter24::new(),
            _pad1: [0; 48],
            ikey0: std::array::from_fn(|_| AtomicU64::new(0)),
            keylenx: std::array::from_fn(|_| AtomicU8::new(0)),
            leaf_values: std::array::from_fn(|_| AtomicPtr::new(std::ptr::null_mut())),
            ksuf: AtomicPtr::new(std::ptr::null_mut()),
            next: AtomicPtr::new(std::ptr::null_mut()),
            prev: AtomicPtr::new(std::ptr::null_mut()),
            parent: AtomicPtr::new(std::ptr::null_mut()),
            _marker: PhantomData,
        }
    }

    /// Create a new leaf node (boxed).
    #[must_use]
    pub fn new() -> Box<Self> {
        Box::new(Self::new_with_root(false))
    }

    /// Create a new leaf node as the root of a tree/layer.
    #[must_use]
    pub fn new_root() -> Box<Self> {
        Box::new(Self::new_with_root(true))
    }

    // ============================================================================
    //  NodeVersion Accessors
    // ============================================================================

    /// Get a reference to the node's version.
    #[inline]
    pub const fn version(&self) -> &NodeVersion {
        &self.version
    }

    /// Get a mutable reference to the node's version.
    #[inline]
    pub const fn version_mut(&mut self) -> &mut NodeVersion {
        &mut self.version
    }

    // ============================================================================
    //  Key Accessors
    // ============================================================================

    /// Get the ikey at the given physical slot.
    ///
    /// # Panics
    /// Panics in debug mode if `slot >= WIDTH_24`.
    #[inline(always)]
    #[must_use]
    #[expect(
        clippy::indexing_slicing,
        reason = "Slot from Permuter24, valid by construction"
    )]
    pub fn ikey(&self, slot: usize) -> u64 {
        debug_assert!(slot < WIDTH_24, "ikey: slot out of bounds");

        self.ikey0[slot].load(READ_ORD)
    }

    /// Set the ikey at the given physical slot.
    #[inline(always)]
    #[expect(
        clippy::indexing_slicing,
        reason = "Slot from Permuter24, valid by construction"
    )]
    pub fn set_ikey(&self, slot: usize, ikey: u64) {
        debug_assert!(slot < WIDTH_24, "set_ikey: slot out of bounds");

        self.ikey0[slot].store(ikey, WRITE_ORD);
    }

    /// Load all ikeys into a contiguous buffer for SIMD search.
    #[inline]
    #[must_use]
    #[expect(clippy::indexing_slicing)]
    pub fn load_all_ikeys(&self) -> [u64; WIDTH_24] {
        let mut ikeys = [0u64; WIDTH_24];

        (0..WIDTH_24).for_each(|i| {
            ikeys[i] = self.ikey0[i].load(READ_ORD);
        });

        ikeys
    }

    /// Get the keylenx at the given physical slot.
    #[inline(always)]
    #[must_use]
    #[expect(
        clippy::indexing_slicing,
        reason = "Slot from Permuter24, valid by construction"
    )]
    pub fn keylenx(&self, slot: usize) -> u8 {
        debug_assert!(slot < WIDTH_24, "keylenx: slot out of bounds");

        self.keylenx[slot].load(READ_ORD)
    }

    /// Set the keylenx at the given physical slot.
    #[inline(always)]
    #[expect(
        clippy::indexing_slicing,
        reason = "Slot from Permuter24, valid by construction"
    )]
    pub fn set_keylenx(&self, slot: usize, keylenx: u8) {
        debug_assert!(slot < WIDTH_24, "set_keylenx: slot out of bounds");

        self.keylenx[slot].store(keylenx, WRITE_ORD);
    }

    /// Get the ikey bound (ikey at slot 0, used for B-link tree routing).
    #[inline(always)]
    #[must_use]
    pub fn ikey_bound(&self) -> u64 {
        self.ikey0[0].load(READ_ORD)
    }

    /// Get the `keylenx` bound for this leaf.
    #[inline]
    pub fn keylenx_bound(&self) -> u8 {
        let perm: Permuter24 = self.permutation();

        debug_assert!(perm.size() > 0, "keylenx_bound called on empty_leaf");

        self.keylenx(perm.get(0))
    }

    /// Check if the given slot contains a layer pointer.
    #[inline]
    #[must_use]
    pub fn is_layer(&self, slot: usize) -> bool {
        self.keylenx(slot) >= LAYER_KEYLENX
    }

    /// Check if the given slot has a suffix.
    #[inline(always)]
    #[must_use]
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
    #[inline(always)]
    #[must_use]
    pub const fn keylenx_has_ksuf(keylenx: u8) -> bool {
        keylenx == KSUF_KEYLENX
    }

    // ============================================================================
    //  Suffix Storage Methods
    // ============================================================================

    /// Load suffix bag pointer (reader).
    #[inline]
    #[must_use]
    pub fn ksuf_ptr(&self) -> *mut SuffixBag<WIDTH_24> {
        self.ksuf.load(READ_ORD)
    }

    /// Check if this leaf has suffix storage allocated.
    #[inline]
    #[must_use]
    pub fn has_ksuf_storage(&self) -> bool {
        !self.ksuf_ptr().is_null()
    }

    /// Get the suffix for a slot.
    #[must_use]
    pub fn ksuf(&self, slot: usize) -> Option<&[u8]> {
        debug_assert!(slot < WIDTH_24, "ksuf: slot {slot} >= WIDTH_24 {WIDTH_24}");

        if !self.has_ksuf(slot) {
            return None;
        }

        let ptr = self.ksuf_ptr();
        if ptr.is_null() {
            return None;
        }

        // SAFETY: Caller must ensure suffix bag is stable (lock or version check).
        unsafe { (*ptr).get(slot) }
    }

    /// Get the suffix for a slot, or an empty slice if none.
    #[inline]
    #[must_use]
    pub fn ksuf_or_empty(&self, slot: usize) -> &[u8] {
        self.ksuf(slot).unwrap_or(&[])
    }

    /// Assign a suffix to a slot (copy-on-write).
    ///
    /// # Safety
    /// - Caller must hold lock and have called `mark_insert()`
    /// - `guard` must come from this tree's collector
    #[expect(
        clippy::indexing_slicing,
        reason = "Slot bounds checked via debug_assert"
    )]
    pub unsafe fn assign_ksuf(&self, slot: usize, suffix: &[u8], guard: &LocalGuard<'_>) {
        debug_assert!(
            slot < WIDTH_24,
            "assign_ksuf: slot {slot} >= WIDTH_24 {WIDTH_24}"
        );

        let old_ptr: *mut SuffixBag<WIDTH_24> = self.ksuf.load(RELAXED);
        let mut new_bag: SuffixBag<WIDTH_24> = if old_ptr.is_null() {
            SuffixBag::new()
        } else {
            // SAFETY: old_ptr is non-null and came from Box::into_raw
            unsafe { (*old_ptr).clone() }
        };

        new_bag.assign(slot, suffix);
        let new_ptr: *mut SuffixBag<WIDTH_24> = Box::into_raw(Box::new(new_bag));

        self.ksuf.store(new_ptr, WRITE_ORD);

        if !old_ptr.is_null() {
            // SAFETY: old_ptr is non-null and came from Box::into_raw
            unsafe {
                guard.defer_retire(old_ptr, |ptr, _| {
                    drop(Box::from_raw(ptr));
                });
            }
        }

        self.keylenx[slot].store(KSUF_KEYLENX, WRITE_ORD);
    }

    /// Clear the suffix from a slot (copy-on-write).
    ///
    /// # Safety
    /// - Caller must hold lock and have called `mark_insert()`
    /// - `guard` must come from this tree's collector
    #[expect(
        clippy::indexing_slicing,
        reason = "Slot bounds checked via debug_assert"
    )]
    pub unsafe fn clear_ksuf(&self, slot: usize, guard: &LocalGuard<'_>) {
        debug_assert!(
            slot < WIDTH_24,
            "clear_ksuf: slot {slot} >= WIDTH_24 {WIDTH_24}"
        );

        let old_ptr: *mut SuffixBag<WIDTH_24> = self.ksuf.load(RELAXED);
        if old_ptr.is_null() {
            self.keylenx[slot].store(0, WRITE_ORD);
            return;
        }

        // SAFETY: old_ptr is non-null and came from Box::into_raw
        let mut new_bag: SuffixBag<WIDTH_24> = unsafe { (*old_ptr).clone() };
        new_bag.clear(slot);
        let new_ptr: *mut SuffixBag<WIDTH_24> = Box::into_raw(Box::new(new_bag));

        self.ksuf.store(new_ptr, WRITE_ORD);

        // SAFETY: old_ptr is non-null and came from Box::into_raw
        unsafe {
            guard.defer_retire(old_ptr, |ptr, _| {
                drop(Box::from_raw(ptr));
            });
        }

        self.keylenx[slot].store(0, WRITE_ORD);
    }

    /// Check if a slot's suffix equals the given suffix.
    #[must_use]
    pub fn ksuf_equals(&self, slot: usize, suffix: &[u8]) -> bool {
        debug_assert!(
            slot < WIDTH_24,
            "ksuf_equals: slot {slot} >= WIDTH_24 {WIDTH_24}"
        );

        if !self.has_ksuf(slot) {
            return false;
        }

        let ptr = self.ksuf_ptr();
        if ptr.is_null() {
            return false;
        }

        // SAFETY: Caller must ensure suffix bag is stable
        unsafe { (*ptr).suffix_equals(slot, suffix) }
    }

    /// Compare a slot's suffix with the given suffix.
    #[must_use]
    pub fn ksuf_compare(&self, slot: usize, suffix: &[u8]) -> Option<std::cmp::Ordering> {
        debug_assert!(
            slot < WIDTH_24,
            "ksuf_compare: slot {slot} >= WIDTH_24 {WIDTH_24}"
        );

        if !self.has_ksuf(slot) {
            return None;
        }

        let ptr = self.ksuf_ptr();
        if ptr.is_null() {
            return None;
        }

        // SAFETY: Caller must ensure suffix bag is stable
        unsafe { (*ptr).suffix_compare(slot, suffix) }
    }

    /// Check if a slot's key matches the given key.
    #[must_use]
    pub fn ksuf_matches(&self, slot: usize, ikey: u64, suffix: &[u8]) -> bool {
        debug_assert!(
            slot < WIDTH_24,
            "ksuf_matches: slot {slot} >= WIDTH_24 {WIDTH_24}"
        );

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
    /// * `1` - Exact match
    /// * `0` - Same ikey but different key
    /// * `-8` - Slot is a layer pointer
    #[inline(always)]
    #[must_use]
    #[expect(
        clippy::cast_possible_wrap,
        clippy::cast_possible_truncation,
        reason = "IKEY_SIZE (8) fits in i32"
    )]
    pub fn ksuf_match_result(&self, slot: usize, keylenx: u8, suffix: &[u8]) -> i32 {
        use crate::key::IKEY_SIZE;

        debug_assert!(
            slot < WIDTH_24,
            "ksuf_match_result: slot {slot} >= WIDTH_24 {WIDTH_24}"
        );

        let stored_keylenx: u8 = self.keylenx(slot);

        if Self::keylenx_is_layer(stored_keylenx) {
            return -(IKEY_SIZE as i32);
        }

        if !self.has_ksuf(slot) {
            if stored_keylenx == keylenx && suffix.is_empty() {
                return 1;
            }
            return 0;
        }

        if suffix.is_empty() {
            return 0;
        }

        i32::from(self.ksuf_equals(slot, suffix))
    }

    /// Compact suffix storage.
    ///
    /// # Safety
    /// - The `guard` must be valid and from the same collector as the tree.
    pub unsafe fn compact_ksuf(
        &self,
        exclude_slot: Option<usize>,
        guard: &LocalGuard<'_>,
    ) -> usize {
        let old_ptr: *mut SuffixBag<WIDTH_24> = self.ksuf.load(RELAXED);
        if old_ptr.is_null() {
            return 0;
        }

        let perm = self.permutation();
        // SAFETY: old_ptr is non-null
        let mut new_bag: SuffixBag<WIDTH_24> = unsafe { (*old_ptr).clone() };
        let reclaimed = new_bag.compact_with_permuter(&perm, exclude_slot);
        let new_ptr: *mut SuffixBag<WIDTH_24> = Box::into_raw(Box::new(new_bag));

        self.ksuf.store(new_ptr, WRITE_ORD);

        // SAFETY: old_ptr is non-null
        unsafe {
            guard.defer_retire(old_ptr, |ptr, _| {
                drop(Box::from_raw(ptr));
            });
        }

        reclaimed
    }

    // ============================================================================
    //  Value Accessors
    // ============================================================================

    /// Load leaf value pointer at the given slot.
    #[inline(always)]
    #[expect(
        clippy::indexing_slicing,
        reason = "Slot from Permuter24; valid by construction"
    )]
    pub fn leaf_value_ptr(&self, slot: usize) -> *mut u8 {
        debug_assert!(slot < WIDTH_24, "leaf_value_ptr: slot out of bounds");

        self.leaf_values[slot].load(READ_ORD)
    }

    /// Store leaf value pointer at the given slot.
    #[inline(always)]
    #[expect(
        clippy::indexing_slicing,
        reason = "Slot from Permuter24; valid by construction"
    )]
    pub fn set_leaf_value_ptr(&self, slot: usize, ptr: *mut u8) {
        debug_assert!(slot < WIDTH_24, "set_leaf_value_ptr: slot out of bounds");

        self.leaf_values[slot].store(ptr, WRITE_ORD);
    }

    /// Take the leaf value pointer, leaving null in the slot.
    #[inline(always)]
    #[expect(
        clippy::indexing_slicing,
        reason = "Slot from Permuter24; valid by construction"
    )]
    pub fn take_leaf_value_ptr(&self, slot: usize) -> *mut u8 {
        debug_assert!(slot < WIDTH_24, "take_leaf_value_ptr: slot out of bounds");

        self.leaf_values[slot].swap(StdPtr::null_mut(), RELAXED)
    }

    /// Check if a slot is empty (value pointer is null).
    #[inline(always)]
    #[must_use]
    pub fn is_slot_empty(&self, slot: usize) -> bool {
        self.leaf_value_ptr(slot).is_null()
    }

    // ============================================================================
    //  Permutation Accessors
    // ============================================================================

    /// Load permutation with Acquire ordering.
    #[inline(always)]
    #[must_use]
    pub fn permutation(&self) -> Permuter24 {
        self.permutation.load(READ_ORD)
    }

    /// Store permutation with Release ordering.
    #[inline(always)]
    pub fn set_permutation(&self, perm: Permuter24) {
        self.permutation.store(perm, WRITE_ORD);
    }

    /// Get raw permutation value (for debugging).
    #[inline(always)]
    #[must_use]
    pub fn permutation_raw(&self) -> u128 {
        self.permutation.load_raw(READ_ORD)
    }

    /// Store raw permutation value with Release ordering.
    #[inline(always)]
    pub(crate) fn permutation_store_raw_release(&self, raw: u128) {
        self.permutation.store_raw(raw, WRITE_ORD);
    }

    /// Pre-store slot data for CAS-based insert.
    ///
    /// # Safety
    /// - `slot` is in the free region of the current permutation
    /// - No concurrent writer is modifying this slot
    #[inline]
    #[expect(
        clippy::indexing_slicing,
        reason = "bounds checked by debug_assert"
    )]
    pub unsafe fn store_slot_for_cas(
        &self,
        slot: usize,
        ikey: u64,
        keylenx: u8,
        value_ptr: *mut u8,
    ) {
        debug_assert!(slot < WIDTH_24, "store_slot_for_cas: slot out of bounds");
        self.ikey0[slot].store(ikey, WRITE_ORD);
        self.keylenx[slot].store(keylenx, WRITE_ORD);
        self.leaf_values[slot].store(value_ptr, WRITE_ORD);
    }

    /// Clear a slot after a failed CAS insert.
    ///
    /// # Safety
    /// - Caller must have already reclaimed/freed the value that was stored
    #[inline]
    #[expect(
        clippy::indexing_slicing,
        reason = "bounds checked by debug_assert"
    )]
    pub unsafe fn clear_slot_for_cas(&self, slot: usize) {
        debug_assert!(slot < WIDTH_24, "clear_slot_for_cas: slot out of bounds");
        self.leaf_values[slot].store(std::ptr::null_mut(), WRITE_ORD);
    }

    /// Atomically claim a slot for CAS insert.
    #[inline]
    #[expect(clippy::indexing_slicing, reason = "bounds checked by debug_assert")]
    pub fn cas_slot_value(
        &self,
        slot: usize,
        expected: *mut u8,
        new_value: *mut u8,
    ) -> Result<(), *mut u8> {
        use crate::ordering::{CAS_FAILURE, CAS_SUCCESS};
        debug_assert!(slot < WIDTH_24, "cas_slot_value: slot out of bounds");

        match self.leaf_values[slot].compare_exchange(expected, new_value, CAS_SUCCESS, CAS_FAILURE)
        {
            Ok(_) => Ok(()),
            Err(actual) => Err(actual),
        }
    }

    /// Load the current value pointer at a slot.
    #[inline]
    #[expect(clippy::indexing_slicing, reason = "bounds checked by debug_assert")]
    pub fn load_slot_value(&self, slot: usize) -> *mut u8 {
        debug_assert!(slot < WIDTH_24, "load_slot_value: slot out of bounds");
        self.leaf_values[slot].load(READ_ORD)
    }

    /// Store key data for a slot after successful CAS claim.
    ///
    /// # Safety
    /// - Caller must have successfully claimed the slot via `cas_slot_value`
    #[inline]
    #[expect(
        clippy::indexing_slicing,
        reason = "bounds checked by debug_assert"
    )]
    pub unsafe fn store_key_data_for_cas(&self, slot: usize, ikey: u64, keylenx: u8) {
        debug_assert!(slot < WIDTH_24, "store_key_data_for_cas: slot out of bounds");
        self.ikey0[slot].store(ikey, WRITE_ORD);
        self.keylenx[slot].store(keylenx, WRITE_ORD);
    }

    /// Get the number of keys in this leaf.
    #[inline(always)]
    #[must_use]
    pub fn size(&self) -> usize {
        self.permutation().size()
    }

    /// Check if the leaf is empty.
    #[inline(always)]
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.size() == 0
    }

    /// Check if the leaf is full.
    #[inline]
    #[must_use]
    pub fn is_full(&self) -> bool {
        self.size() >= WIDTH_24
    }

    // ============================================================================
    //  Leaf Linking
    // ============================================================================

    /// Get the next leaf pointer, masking the mark bit.
    #[inline]
    #[must_use]
    pub fn safe_next(&self) -> *mut Self {
        let ptr: *mut Self = self.next.load(READ_ORD);
        ptr.map_addr(|addr: usize| addr & !1)
    }

    /// Get the raw next pointer (including mark bit).
    #[inline]
    #[must_use]
    pub fn next_raw(&self) -> *mut Self {
        self.next.load(READ_ORD)
    }

    /// Check if the next pointer is marked (split in progress).
    #[inline]
    #[must_use]
    pub fn next_is_marked(&self) -> bool {
        (self.next.load(READ_ORD).addr() & 1) != 0
    }

    /// Set the next leaf pointer.
    #[inline]
    pub fn set_next(&self, next: *mut Self) {
        self.next.store(next, WRITE_ORD);
    }

    /// Mark the next pointer (during split).
    #[inline]
    pub fn mark_next(&self) {
        let ptr: *mut Self = self.next.load(RELAXED);
        let marked: *mut Self = ptr.map_addr(|addr: usize| addr | 1);
        self.next.store(marked, WRITE_ORD);
    }

    /// Unmark the next pointer.
    #[inline]
    pub fn unmark_next(&self) {
        let ptr: *mut Self = self.safe_next();
        self.next.store(ptr, WRITE_ORD);
    }

    /// Wait for an in-progress split to complete.
    ///
    /// Spins until the next pointer is unmarked and version is stable.
    #[inline]
    pub fn wait_for_split(&self) {
        const MAX_RETRIES: usize = 1000;
        let mut retries: usize = 0;

        while self.next_is_marked() {
            // Quick check: did marker clear during spin?
            for _ in 0..16 {
                std::hint::spin_loop();
                if !self.next_is_marked() {
                    return;
                }
            }

            // Still marked - wait for version to stabilize
            let _ = self.version.stable();

            retries += 1;
            if retries > MAX_RETRIES {
                // Timeout - proceed anyway
                break;
            }
        }
    }

    /// Get the previous leaf pointer.
    #[inline]
    #[must_use]
    pub fn prev(&self) -> *mut Self {
        self.prev.load(READ_ORD)
    }

    /// Set the previous leaf pointer.
    #[inline]
    pub fn set_prev(&self, prev: *mut Self) {
        self.prev.store(prev, WRITE_ORD);
    }

    // ============================================================================
    //  Parent Accessors
    // ============================================================================

    /// Get the parent pointer.
    #[inline]
    #[must_use]
    pub fn parent(&self) -> *mut u8 {
        self.parent.load(READ_ORD)
    }

    /// Set the parent pointer.
    #[inline]
    pub fn set_parent(&self, parent: *mut u8) {
        self.parent.store(parent, WRITE_ORD);
    }

    // ============================================================================
    //  ModState Accessors
    // ============================================================================

    /// Get the modification state.
    #[inline]
    #[must_use]
    pub const fn modstate(&self) -> ModState24 {
        self.modstate
    }

    /// Set the modification state.
    #[inline]
    pub const fn set_modstate(&mut self, state: ModState24) {
        self.modstate = state;
    }

    // ============================================================================
    //  Slot Assignment
    // ============================================================================

    /// Check if slot 0 can be reused for a new key.
    #[inline]
    #[must_use]
    pub fn can_reuse_slot0(&self, new_ikey: u64) -> bool {
        if self.prev().is_null() {
            return true;
        }

        self.ikey_bound() == new_ikey
    }
}

// ============================================================================
//  Send + Sync
// ============================================================================

// SAFETY: LeafNode24 is safe to send/share between threads when S is.
// The atomic fields handle concurrent access, and the raw pointers are
// protected by the tree's concurrency protocol (version validation, locks).
unsafe impl<S: ValueSlot + Send + Sync> Send for LeafNode24<S> {}
unsafe impl<S: ValueSlot + Send + Sync> Sync for LeafNode24<S> {}

// ============================================================================
//  TreeLeafNode Implementation
// ============================================================================

impl<S: ValueSlot + Send + Sync + 'static> crate::leaf_trait::TreeLeafNode<S> for LeafNode24<S> {
    type Perm = Permuter24;
    type Internode = crate::internode::InternodeNode<S, WIDTH_24>;
    const WIDTH: usize = WIDTH_24;

    #[inline]
    fn new_boxed() -> Box<Self> {
        LeafNode24::new()
    }

    #[inline]
    fn new_root_boxed() -> Box<Self> {
        LeafNode24::new_root()
    }

    #[inline]
    fn version(&self) -> &crate::nodeversion::NodeVersion {
        LeafNode24::version(self)
    }

    #[inline]
    fn permutation(&self) -> Permuter24 {
        LeafNode24::permutation(self)
    }

    #[inline]
    fn set_permutation(&self, perm: Permuter24) {
        LeafNode24::set_permutation(self, perm)
    }

    #[inline]
    fn permutation_raw(&self) -> u128 {
        LeafNode24::permutation_raw(self)
    }

    fn permutation_try(&self) -> Result<Permuter24, ()> {
        LeafNode24::permutation_try(self).map_err(|_| ())
    }

    fn permutation_wait(&self) -> Permuter24 {
        LeafNode24::permutation_wait(self)
    }

    #[inline]
    fn ikey(&self, slot: usize) -> u64 {
        LeafNode24::ikey(self, slot)
    }

    #[inline]
    fn set_ikey(&self, slot: usize, ikey: u64) {
        LeafNode24::set_ikey(self, slot, ikey)
    }

    #[inline]
    fn ikey_bound(&self) -> u64 {
        LeafNode24::ikey_bound(self)
    }

    #[inline]
    fn keylenx(&self, slot: usize) -> u8 {
        LeafNode24::keylenx(self, slot)
    }

    #[inline]
    fn set_keylenx(&self, slot: usize, keylenx: u8) {
        LeafNode24::set_keylenx(self, slot, keylenx)
    }

    #[inline]
    fn is_layer(&self, slot: usize) -> bool {
        LeafNode24::is_layer(self, slot)
    }

    #[inline]
    fn has_ksuf(&self, slot: usize) -> bool {
        LeafNode24::has_ksuf(self, slot)
    }

    #[inline]
    fn leaf_value_ptr(&self, slot: usize) -> *mut u8 {
        LeafNode24::leaf_value_ptr(self, slot)
    }

    #[inline]
    fn set_leaf_value_ptr(&self, slot: usize, ptr: *mut u8) {
        LeafNode24::set_leaf_value_ptr(self, slot, ptr)
    }

    #[inline]
    fn cas_slot_value(
        &self,
        slot: usize,
        expected: *mut u8,
        new_value: *mut u8,
    ) -> Result<(), *mut u8> {
        LeafNode24::cas_slot_value(self, slot, expected, new_value)
    }

    #[inline]
    fn safe_next(&self) -> *mut Self {
        LeafNode24::safe_next(self)
    }

    #[inline]
    fn next_is_marked(&self) -> bool {
        LeafNode24::next_is_marked(self)
    }

    #[inline]
    fn set_next(&self, next: *mut Self) {
        LeafNode24::set_next(self, next)
    }

    #[inline]
    fn mark_next(&self) {
        LeafNode24::mark_next(self)
    }

    #[inline]
    fn unmark_next(&self) {
        LeafNode24::unmark_next(self)
    }

    #[inline]
    fn prev(&self) -> *mut Self {
        LeafNode24::prev(self)
    }

    #[inline]
    fn set_prev(&self, prev: *mut Self) {
        LeafNode24::set_prev(self, prev)
    }

    #[inline]
    fn parent(&self) -> *mut u8 {
        LeafNode24::parent(self)
    }

    #[inline]
    fn set_parent(&self, parent: *mut u8) {
        LeafNode24::set_parent(self, parent)
    }

    #[inline]
    fn can_reuse_slot0(&self, new_ikey: u64) -> bool {
        LeafNode24::can_reuse_slot0(self, new_ikey)
    }

    #[inline]
    unsafe fn store_slot_for_cas(
        &self,
        slot: usize,
        ikey: u64,
        keylenx: u8,
        value_ptr: *mut u8,
    ) {
        // SAFETY: Caller guarantees slot is in free region and no concurrent modification
        unsafe { LeafNode24::store_slot_for_cas(self, slot, ikey, keylenx, value_ptr) }
    }

    #[inline]
    unsafe fn store_key_data_for_cas(&self, slot: usize, ikey: u64, keylenx: u8) {
        // SAFETY: Caller guarantees slot was claimed via cas_slot_value
        unsafe { LeafNode24::store_key_data_for_cas(self, slot, ikey, keylenx) }
    }

    #[inline]
    unsafe fn clear_slot_for_cas(&self, slot: usize) {
        // SAFETY: Caller guarantees value has been reclaimed
        unsafe { LeafNode24::clear_slot_for_cas(self, slot) }
    }

    #[inline]
    fn load_slot_value(&self, slot: usize) -> *mut u8 {
        LeafNode24::load_slot_value(self, slot)
    }

    #[inline]
    fn next_raw(&self) -> *mut Self {
        LeafNode24::next_raw(self)
    }

    #[inline]
    fn wait_for_split(&self) {
        LeafNode24::wait_for_split(self)
    }

    #[inline]
    fn cas_permutation_raw(
        &self,
        expected: Self::Perm,
        new: Self::Perm,
    ) -> Result<(), crate::leaf_trait::CasPermutationError<Self::Perm>> {
        LeafNode24::cas_permutation_raw(self, expected, new)
            .map_err(|failure| {
                crate::leaf_trait::CasPermutationError::new(
                    crate::permuter24::Permuter24::from_value(failure.current_raw())
                )
            })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::leaf::LeafValue;

    #[test]
    fn test_new_leaf24_is_empty() {
        let leaf: Box<LeafNode24<LeafValue<u64>>> = LeafNode24::new();
        assert!(leaf.is_empty());
        assert_eq!(leaf.size(), 0);
        assert!(!leaf.is_full());
    }

    #[test]
    fn test_leaf24_permutation_basic() {
        let leaf: Box<LeafNode24<LeafValue<u64>>> = LeafNode24::new();
        let perm = leaf.permutation();
        assert_eq!(perm.size(), 0);

        // Insert one key
        let mut new_perm = perm;
        let slot = new_perm.insert_from_back(0);
        leaf.set_permutation(new_perm);

        assert_eq!(leaf.size(), 1);
        assert_eq!(leaf.permutation().get(0), slot);
    }

    #[test]
    fn test_leaf24_ikey_operations() {
        let leaf: Box<LeafNode24<LeafValue<u64>>> = LeafNode24::new();
        leaf.set_ikey(0, 12345);
        assert_eq!(leaf.ikey(0), 12345);
        assert_eq!(leaf.ikey_bound(), 12345);
    }

    #[test]
    fn test_leaf24_keylenx_operations() {
        let leaf: Box<LeafNode24<LeafValue<u64>>> = LeafNode24::new();
        leaf.set_keylenx(5, 8);
        assert_eq!(leaf.keylenx(5), 8);
        assert!(!leaf.is_layer(5));
        assert!(!leaf.has_ksuf(5));

        leaf.set_keylenx(10, LAYER_KEYLENX);
        assert!(leaf.is_layer(10));

        leaf.set_keylenx(15, KSUF_KEYLENX);
        assert!(leaf.has_ksuf(15));
    }

    #[test]
    fn test_leaf24_full_at_24_slots() {
        let leaf: Box<LeafNode24<LeafValue<u64>>> = LeafNode24::new();
        let mut perm = leaf.permutation();

        for i in 0..WIDTH_24 {
            let _slot = perm.insert_from_back(i);
        }
        leaf.set_permutation(perm);

        assert!(leaf.is_full());
        assert_eq!(leaf.size(), WIDTH_24);
    }

    #[test]
    fn test_leaf24_linking() {
        let leaf1: Box<LeafNode24<LeafValue<u64>>> = LeafNode24::new();
        let leaf2: Box<LeafNode24<LeafValue<u64>>> = LeafNode24::new();

        let leaf2_ptr: *mut LeafNode24<LeafValue<u64>> = Box::into_raw(leaf2);

        leaf1.set_next(leaf2_ptr);
        assert_eq!(leaf1.safe_next(), leaf2_ptr);
        assert!(!leaf1.next_is_marked());

        leaf1.mark_next();
        assert!(leaf1.next_is_marked());
        assert_eq!(leaf1.safe_next(), leaf2_ptr);

        leaf1.unmark_next();
        assert!(!leaf1.next_is_marked());

        // Cleanup
        let _ = unsafe { Box::from_raw(leaf2_ptr) };
    }

    #[test]
    fn test_leaf24_parent() {
        let leaf: Box<LeafNode24<LeafValue<u64>>> = LeafNode24::new();
        assert!(leaf.parent().is_null());

        let dummy: u64 = 0xDEAD_BEEF;
        let dummy_ptr: *mut u8 = std::ptr::from_ref(&dummy).cast_mut().cast();
        leaf.set_parent(dummy_ptr);
        assert_eq!(leaf.parent(), dummy_ptr);
    }
}
