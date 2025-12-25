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

/// Width constant for [`LeafNode24`].
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
    #[inline]
    #[must_use]
    pub fn new() -> Box<Self> {
        Box::new(Self::new_with_root(false))
    }

    /// Create a new leaf node as the root of a tree/layer.
    #[inline]
    #[must_use]
    pub fn new_root() -> Box<Self> {
        Box::new(Self::new_with_root(true))
    }

    /// Convert this leaf into a layer root.
    ///
    /// Sets up the node to serve as the root of a sub-layer:
    /// - Sets parent pointer to null
    /// - Marks version as root
    ///
    /// NOTE: This matches [`LeafNode::make_layer_root`] in `src/leaf/layer.rs`.
    ///
    /// SAFETY: Caller must ensure this node is not currently part of another tree
    /// structure, or that appropriate synchronization is in place.
    #[inline(always)]
    pub fn make_layer_root(&self) {
        self.set_parent(StdPtr::null_mut());
        self.version.mark_root();
    }

    /// Create a new leaf node configured as a layer root.
    ///
    /// Used when creating sublayers for keys longer than 8 bytes.
    #[inline]
    #[must_use]
    pub fn new_layer_root() -> Box<Self> {
        let node: Box<Self> = Self::new();
        node.make_layer_root();

        node
    }

    // ============================================================================
    //  NodeVersion Accessors
    // ============================================================================

    /// Get a reference to the node's version.
    #[inline(always)]
    pub const fn version(&self) -> &NodeVersion {
        &self.version
    }

    /// Get a mutable reference to the node's version.
    #[inline(always)]
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
    #[must_use]
    #[inline(always)]
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
    #[must_use]
    #[inline(always)]
    #[expect(clippy::indexing_slicing)]
    pub fn load_all_ikeys(&self) -> [u64; WIDTH_24] {
        let mut ikeys = [0u64; WIDTH_24];

        (0..WIDTH_24).for_each(|i| {
            ikeys[i] = self.ikey0[i].load(READ_ORD);
        });

        ikeys
    }

    /// Get the keylenx at the given physical slot.
    #[must_use]
    #[inline(always)]
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
    #[must_use]
    #[inline(always)]
    pub fn ikey_bound(&self) -> u64 {
        self.ikey0[0].load(READ_ORD)
    }

    /// Get the `keylenx` bound for this leaf.
    #[inline(always)]
    pub fn keylenx_bound(&self) -> u8 {
        let perm: Permuter24 = self.permutation();

        debug_assert!(perm.size() > 0, "keylenx_bound called on empty_leaf");

        self.keylenx(perm.get(0))
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

    // ============================================================================
    //  Suffix Storage Methods
    // ============================================================================

    /// Load suffix bag pointer (reader).
    #[must_use]
    #[inline(always)]
    pub fn ksuf_ptr(&self) -> *mut SuffixBag<WIDTH_24> {
        self.ksuf.load(READ_ORD)
    }

    /// Check if this leaf has suffix storage allocated.
    #[must_use]
    #[inline(always)]
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
    #[must_use]
    #[inline(always)]
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

        // FAST PATH: Try in-place assignment if bag exists and has room.
        // This avoids clone + box allocation + defer_retire in the common case.
        // SAFETY: We hold the lock, so no concurrent writers. Readers use version
        // validation and will retry if they see partial state.
        if !old_ptr.is_null() {
            // SAFETY: old_ptr is non-null and came from Box::into_raw.
            // We hold the lock, so we can mutate the bag in place.
            let bag: &mut SuffixBag<WIDTH_24> = unsafe { &mut *old_ptr };
            if bag.try_assign_in_place(slot, suffix) {
                self.keylenx[slot].store(KSUF_KEYLENX, WRITE_ORD);
                return;
            }
        }

        // SLOW PATH: Need to allocate a new bag (either first allocation or bag is full)
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
    #[must_use]
    #[inline(always)]
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
    #[inline(always)]
    #[expect(clippy::indexing_slicing, reason = "bounds checked by debug_assert")]
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
    #[inline(always)]
    #[expect(clippy::indexing_slicing, reason = "bounds checked by debug_assert")]
    pub unsafe fn clear_slot_for_cas(&self, slot: usize) {
        debug_assert!(slot < WIDTH_24, "clear_slot_for_cas: slot out of bounds");
        self.leaf_values[slot].store(std::ptr::null_mut(), WRITE_ORD);
    }

    /// Atomically claim a slot for CAS insert.
    ///
    /// # Errors
    /// Returns error if CAS fails
    #[inline(always)]
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
    #[inline(always)]
    #[expect(clippy::indexing_slicing, reason = "bounds checked by debug_assert")]
    pub fn load_slot_value(&self, slot: usize) -> *mut u8 {
        debug_assert!(slot < WIDTH_24, "load_slot_value: slot out of bounds");
        self.leaf_values[slot].load(READ_ORD)
    }

    /// Store key data for a slot after successful CAS claim.
    ///
    /// # Safety
    /// - Caller must have successfully claimed the slot via `cas_slot_value`
    #[inline(always)]
    #[expect(clippy::indexing_slicing, reason = "bounds checked by debug_assert")]
    pub unsafe fn store_key_data_for_cas(&self, slot: usize, ikey: u64, keylenx: u8) {
        debug_assert!(
            slot < WIDTH_24,
            "store_key_data_for_cas: slot out of bounds"
        );
        self.ikey0[slot].store(ikey, WRITE_ORD);
        self.keylenx[slot].store(keylenx, WRITE_ORD);
    }

    /// Get the number of keys in this leaf.
    #[must_use]
    #[inline(always)]
    pub fn size(&self) -> usize {
        self.permutation().size()
    }

    /// Check if the leaf is empty.
    #[must_use]
    #[inline(always)]
    pub fn is_empty(&self) -> bool {
        self.size() == 0
    }

    /// Check if the leaf is full.
    #[must_use]
    #[inline(always)]
    pub fn is_full(&self) -> bool {
        self.size() >= WIDTH_24
    }

    // ============================================================================
    //  Leaf Linking
    // ============================================================================

    /// Get the next leaf pointer, masking the mark bit.
    #[must_use]
    #[inline(always)]
    pub fn safe_next(&self) -> *mut Self {
        let ptr: *mut Self = self.next.load(READ_ORD);
        ptr.map_addr(|addr: usize| addr & !1)
    }

    /// Get the raw next pointer (including mark bit).
    #[must_use]
    #[inline(always)]
    pub fn next_raw(&self) -> *mut Self {
        self.next.load(READ_ORD)
    }

    /// Check if the next pointer is marked (split in progress).
    #[must_use]
    #[inline(always)]
    pub fn next_is_marked(&self) -> bool {
        (self.next.load(READ_ORD).addr() & 1) != 0
    }

    /// Set the next leaf pointer.
    #[inline(always)]
    pub fn set_next(&self, next: *mut Self) {
        self.next.store(next, WRITE_ORD);
    }

    /// Mark the next pointer (during split).
    #[inline(always)]
    pub fn mark_next(&self) {
        let ptr: *mut Self = self.next.load(RELAXED);
        let marked: *mut Self = ptr.map_addr(|addr: usize| addr | 1);
        self.next.store(marked, WRITE_ORD);
    }

    /// Unmark the next pointer.
    #[inline(always)]
    pub fn unmark_next(&self) {
        let ptr: *mut Self = self.safe_next();
        self.next.store(ptr, WRITE_ORD);
    }

    /// Wait for an in-progress split to complete.
    ///
    /// Spins until the next pointer is unmarked and version is stable.
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

    /// CAS-based lock of the next pointer for split linking.
    ///
    /// Marks the next pointer (LSB) to signal a split is in progress.
    /// Other threads seeing a marked pointer will wait via `wait_for_split`.
    ///
    /// Reference: C++ `btree_leaflink.hh:39-56` (`lock_next`)
    ///
    /// # Returns
    /// The unmarked old next pointer (may be null).
    ///
    /// # Ordering
    /// Uses CAS with AcqRel/Acquire ordering to ensure visibility.
    fn lock_next(&self) -> *mut Self {
        use crate::link::{is_marked, mark_ptr};
        use crate::ordering::{CAS_FAILURE, CAS_SUCCESS};

        loop {
            let next: *mut Self = self.next.load(READ_ORD);

            // Already marked: another split is in progress, wait
            if is_marked(next) {
                self.wait_for_split();
                continue;
            }

            // Try to mark the pointer via CAS
            let marked: *mut Self = mark_ptr(next);
            match self
                .next
                .compare_exchange(next, marked, CAS_SUCCESS, CAS_FAILURE)
            {
                Ok(_) => {
                    #[cfg(feature = "tracing")]
                    tracing::trace!(
                        self_ptr = ?StdPtr::from_ref(self),
                        next = ?next,
                        "lock_next: SUCCESS - marked next pointer"
                    );
                    // Return UNMARKED old next (may be null). We intentionally mark even
                    // `NULL` next pointers to avoid two concurrent splits both "seeing"
                    // `NULL` and racing to publish different siblings, orphaning one.
                    return next;
                }
                Err(_) => {
                    // CAS failed: someone else updated next, retry
                    std::hint::spin_loop();
                }
            }
        }
    }

    /// CAS-based compare-and-swap on the next pointer.
    ///
    /// # Errors
    /// Returns `Err(current_value)` if the CAS failed.
    #[inline]
    #[cfg_attr(not(test), allow(dead_code))]
    fn cas_next(&self, current: *mut Self, new: *mut Self) -> Result<*mut Self, *mut Self> {
        use crate::ordering::{CAS_FAILURE, CAS_SUCCESS};
        self.next
            .compare_exchange(current, new, CAS_SUCCESS, CAS_FAILURE)
    }

    /// Get the previous leaf pointer.
    #[must_use]
    #[inline(always)]
    pub fn prev(&self) -> *mut Self {
        self.prev.load(READ_ORD)
    }

    /// Set the previous leaf pointer.
    #[inline(always)]
    pub fn set_prev(&self, prev: *mut Self) {
        self.prev.store(prev, WRITE_ORD);
    }

    // ============================================================================
    //  Parent Accessors
    // ============================================================================

    /// Get the parent pointer.
    #[must_use]
    #[inline(always)]
    pub fn parent(&self) -> *mut u8 {
        self.parent.load(READ_ORD)
    }

    /// Set the parent pointer.
    #[inline(always)]
    pub fn set_parent(&self, parent: *mut u8) {
        self.parent.store(parent, WRITE_ORD);
    }

    // ============================================================================
    //  ModState Accessors
    // ============================================================================

    /// Get the modification state.
    #[must_use]
    #[inline(always)]
    pub const fn modstate(&self) -> ModState24 {
        self.modstate
    }

    /// Set the modification state.
    #[inline(always)]
    pub const fn set_modstate(&mut self, state: ModState24) {
        self.modstate = state;
    }

    // ============================================================================
    //  Slot Assignment
    // ============================================================================

    /// Check if slot 0 can be reused for a new key.
    #[must_use]
    #[inline(always)]
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
    // Internodes are limited to WIDTH=15 due to 4-bit permutation slots
    type Internode = crate::internode::InternodeNode<S, 15>;
    const WIDTH: usize = WIDTH_24;

    #[inline(always)]
    fn new_boxed() -> Box<Self> {
        Self::new()
    }

    #[inline(always)]
    fn new_root_boxed() -> Box<Self> {
        Self::new_root()
    }

    #[inline]
    fn new_layer_root_boxed() -> Box<Self> {
        Self::new_layer_root()
    }

    #[inline(always)]
    fn version(&self) -> &crate::nodeversion::NodeVersion {
        Self::version(self)
    }

    #[inline(always)]
    fn permutation(&self) -> Permuter24 {
        Self::permutation(self)
    }

    #[inline(always)]
    fn set_permutation(&self, perm: Permuter24) {
        Self::set_permutation(self, perm);
    }

    #[inline(always)]
    fn permutation_raw(&self) -> u128 {
        Self::permutation_raw(self)
    }

    #[inline(always)]
    fn permutation_try(&self) -> Result<Permuter24, ()> {
        Self::permutation_try(self).map_err(|_| ())
    }

    #[inline(always)]
    fn permutation_wait(&self) -> Permuter24 {
        Self::permutation_wait(self)
    }

    #[inline(always)]
    fn ikey(&self, slot: usize) -> u64 {
        Self::ikey(self, slot)
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

    #[inline(always)]
    fn leaf_value_ptr(&self, slot: usize) -> *mut u8 {
        Self::leaf_value_ptr(self, slot)
    }

    #[inline(always)]
    fn set_leaf_value_ptr(&self, slot: usize, ptr: *mut u8) {
        Self::set_leaf_value_ptr(self, slot, ptr);
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

    #[inline(always)]
    fn safe_next(&self) -> *mut Self {
        Self::safe_next(self)
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

    #[inline(always)]
    fn prev(&self) -> *mut Self {
        Self::prev(self)
    }

    #[inline(always)]
    fn set_prev(&self, prev: *mut Self) {
        Self::set_prev(self, prev);
    }

    #[inline(always)]
    fn parent(&self) -> *mut u8 {
        Self::parent(self)
    }

    #[inline(always)]
    fn set_parent(&self, parent: *mut u8) {
        Self::set_parent(self, parent);
    }

    #[inline(always)]
    fn can_reuse_slot0(&self, new_ikey: u64) -> bool {
        Self::can_reuse_slot0(self, new_ikey)
    }

    #[inline(always)]
    unsafe fn store_slot_for_cas(&self, slot: usize, ikey: u64, keylenx: u8, value_ptr: *mut u8) {
        // SAFETY: Caller guarantees slot is in free region and no concurrent modification
        unsafe { Self::store_slot_for_cas(self, slot, ikey, keylenx, value_ptr) }
    }

    #[inline(always)]
    unsafe fn store_key_data_for_cas(&self, slot: usize, ikey: u64, keylenx: u8) {
        // SAFETY: Caller guarantees slot was claimed via cas_slot_value
        unsafe { Self::store_key_data_for_cas(self, slot, ikey, keylenx) }
    }

    #[inline(always)]
    unsafe fn clear_slot_for_cas(&self, slot: usize) {
        // SAFETY: Caller guarantees value has been reclaimed
        unsafe { Self::clear_slot_for_cas(self, slot) }
    }

    #[inline(always)]
    fn load_slot_value(&self, slot: usize) -> *mut u8 {
        Self::load_slot_value(self, slot)
    }

    #[inline(always)]
    fn next_raw(&self) -> *mut Self {
        Self::next_raw(self)
    }

    #[inline(always)]
    fn wait_for_split(&self) {
        Self::wait_for_split(self);
    }

    #[inline(always)]
    fn cas_permutation_raw(
        &self,
        expected: Self::Perm,
        new: Self::Perm,
    ) -> Result<(), crate::leaf_trait::CasPermutationError<Self::Perm>> {
        Self::cas_permutation_raw(self, expected, new).map_err(|failure| {
            crate::leaf_trait::CasPermutationError::new(crate::permuter24::Permuter24::from_value(
                failure.current_raw(),
            ))
        })
    }

    // ========================================================================
    // Split Operations
    // ========================================================================

    type FreezeGuard<'a>
        = freeze::FreezeGuard24<'a, S>
    where
        Self: 'a;

    fn calculate_split_point(
        &self,
        _insert_pos: usize,
        insert_ikey: u64,
    ) -> Option<crate::value::SplitPoint> {
        let perm = self.permutation();
        let size = perm.size();

        if size == 0 {
            return None;
        }

        // Split at midpoint
        let mut split_pos = size / 2;
        if split_pos == 0 {
            return None;
        }

        // Adjust for equal ikeys: if keys at split boundary are equal,
        // move split point to keep equal keys together
        while split_pos > 0 && split_pos < size {
            let left_slot = perm.get(split_pos - 1);
            let right_slot = perm.get(split_pos);
            let left_ikey = self.ikey(left_slot);
            let right_ikey = self.ikey(right_slot);

            if left_ikey == right_ikey {
                // Equal keys - check if insert_ikey matches
                match insert_ikey.cmp(&left_ikey) {
                    std::cmp::Ordering::Equal => {
                        // Insert goes with this group - move split right
                        split_pos += 1;
                    }
                    std::cmp::Ordering::Less => {
                        // Insert goes left - move split left
                        split_pos -= 1;
                    }
                    std::cmp::Ordering::Greater => {
                        // Insert goes right - done
                        break;
                    }
                }
            } else {
                break;
            }
        }

        // Edge case: if split_pos is 0 or size, can't split
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

    unsafe fn split_into_preallocated(
        &self,
        split_pos: usize,
        new_leaf: Box<Self>,
        guard: &seize::LocalGuard<'_>,
    ) -> (Box<Self>, u64, crate::value::InsertTarget) {
        // CRITICAL (Help-Along Protocol): Initialize new leaf's version for split
        // BEFORE any data is written. This creates a locked version with SPLITTING_BIT set.
        // The new leaf will remain locked until propagate_split sets its parent.
        //
        // # Why This Works
        //
        // 1. new_leaf is allocated but not yet linked into the tree
        // 2. We replace its default NodeVersion with a split-locked version
        // 3. After link_sibling(), other threads see new_leaf via B-link chain
        // 4. Those threads call stable() on new_leaf.version, which spins because dirty
        // 5. propagate_split sets parent pointer and calls unlock_for_split
        // 6. Now stable() returns and threads can proceed
        //
        // # Safety
        //
        // We're writing to a field of a Box we own, before it's shared.
        // Using ptr::write because NodeVersion doesn't implement Copy.
        unsafe {
            let split_version = crate::nodeversion::NodeVersion::new_for_split(&self.version);

            std::ptr::write(
                std::ptr::addr_of!(new_leaf.version).cast_mut(),
                split_version,
            );

            #[cfg(feature = "tracing")]
            tracing::debug!(
                self_ptr = ?std::ptr::from_ref(self),
                new_leaf_ptr = ?std::ptr::from_ref(new_leaf.as_ref()),
                split_pos = split_pos,
                new_leaf_is_split_locked = new_leaf.version.is_split_locked(),
                "split_into_preallocated: START (new leaf is split-locked)"
            );

            // Always freeze during split - caller must hold lock
            let freeze_guard = Self::freeze_permutation(self);

            let old_perm: Permuter24 = freeze_guard.snapshot();
            let old_size = old_perm.size();

            debug_assert!(
                split_pos > 0 && split_pos < old_size,
                "invalid split_pos {split_pos} for size {old_size}"
            );

            let entries_to_move = old_size - split_pos;

            // Move entries to new leaf
            for i in 0..entries_to_move {
                let old_logical_pos = split_pos + i;
                let old_slot = old_perm.get(old_logical_pos);
                let new_slot = i;

                let ikey = self.ikey(old_slot);
                let keylenx = self.keylenx(old_slot);

                new_leaf.set_ikey(new_slot, ikey);
                new_leaf.set_keylenx(new_slot, keylenx);

                // Move value pointer
                let old_ptr = self.take_leaf_value_ptr(old_slot);
                new_leaf.set_leaf_value_ptr(new_slot, old_ptr);

                // Migrate suffix if present
                if keylenx == KSUF_KEYLENX {
                    if let Some(suffix) = self.ksuf(old_slot) {
                        // SAFETY: new_leaf is freshly allocated and caller holds lock
                        new_leaf.assign_ksuf(new_slot, suffix, guard);
                    }
                    // SAFETY: caller holds lock
                    self.clear_ksuf(old_slot, guard);
                }
            }

            // Build new leaf's permutation
            let new_perm = Permuter24::make_sorted(entries_to_move);
            new_leaf.set_permutation(new_perm);

            // Update old leaf's permutation
            let mut old_perm_updated = old_perm;
            old_perm_updated.set_size(split_pos);

            // Publish truncated permutation and unfreeze
            Self::unfreeze_set_permutation(self, freeze_guard, old_perm_updated);

            // Get split key from new leaf's first entry
            let split_ikey = new_leaf.ikey(new_perm.get(0));

            #[cfg(feature = "tracing")]
            tracing::debug!(
                self_ptr = ?std::ptr::from_ref(self),
                new_leaf_ptr = ?std::ptr::from_ref(new_leaf.as_ref()),
                split_ikey = format_args!("{:016x}", split_ikey),
                entries_moved = entries_to_move,
                old_size_after = split_pos,
                new_size = new_perm.size(),
                "split_into_preallocated: DONE (new leaf still split-locked)"
            );

            (new_leaf, split_ikey, crate::value::InsertTarget::Left)
        }
    }

    unsafe fn split_all_to_right_preallocated(
        &self,
        new_leaf: Box<Self>,
        guard: &seize::LocalGuard<'_>,
    ) -> (Box<Self>, u64, crate::value::InsertTarget) {
        // CRITICAL (Help-Along Protocol): Initialize new leaf's version for split
        // (same as split_into_preallocated)
        let split_version = crate::nodeversion::NodeVersion::new_for_split(&self.version);
        // SAFETY: new_leaf is not yet visible to other threads, we own the Box.
        unsafe {
            std::ptr::write(
                std::ptr::addr_of!(new_leaf.version).cast_mut(),
                split_version,
            );
        }

        // Always freeze during split - caller must hold lock
        let freeze_guard = Self::freeze_permutation(self);

        let old_perm: Permuter24 = freeze_guard.snapshot();
        let old_size = old_perm.size();

        debug_assert!(old_size > 0, "Cannot split empty leaf");

        // Move all entries to new leaf
        for i in 0..old_size {
            let old_slot = old_perm.get(i);
            let new_slot = i;

            let ikey = self.ikey(old_slot);
            let keylenx = self.keylenx(old_slot);

            new_leaf.set_ikey(new_slot, ikey);
            new_leaf.set_keylenx(new_slot, keylenx);

            let old_ptr = self.take_leaf_value_ptr(old_slot);
            new_leaf.set_leaf_value_ptr(new_slot, old_ptr);

            if keylenx == KSUF_KEYLENX {
                if let Some(suffix) = self.ksuf(old_slot) {
                    // SAFETY: new_leaf is freshly allocated and caller holds lock
                    unsafe { new_leaf.assign_ksuf(new_slot, suffix, guard) };
                }
                // SAFETY: caller holds lock
                unsafe { self.clear_ksuf(old_slot, guard) };
            }
        }

        // New leaf gets all entries
        let new_perm = Permuter24::make_sorted(old_size);
        new_leaf.set_permutation(new_perm);

        // Old leaf becomes empty - unfreeze with empty permutation
        Self::unfreeze_set_permutation(self, freeze_guard, Permuter24::empty());

        // Split key is first key of new leaf
        let split_ikey = new_leaf.ikey(new_perm.get(0));

        (new_leaf, split_ikey, crate::value::InsertTarget::Right)
    }

    #[inline(always)]
    fn freeze_permutation(&self) -> Self::FreezeGuard<'_> {
        Self::freeze_permutation(self)
    }

    #[inline(always)]
    fn unfreeze_set_permutation(&self, guard: Self::FreezeGuard<'_>, perm: Self::Perm) {
        Self::unfreeze_set_permutation(self, guard, perm);
    }

    #[inline(always)]
    fn is_permutation_frozen(&self) -> bool {
        crate::freeze24::Freeze24Utils::is_frozen(self.permutation_raw())
    }

    #[inline(always)]
    unsafe fn link_sibling(&self, new_sibling: *mut Self) {
        // CAS-based link_sibling matching C++ btree_leaflink.hh:56-69 (link_split).
        //
        // The key insight is that we must use CAS to "lock" the next pointer before
        // modifying it. This prevents concurrent splits from clobbering each other's
        // next pointer updates. The mark bit (LSB) signals a split is in progress.
        //
        // Sequence:
        // 1. CAS to mark self.next (lock_next)
        // 2. Set up new_sibling's prev/next pointers
        // 3. Update old_next.prev if non-null
        // 4. Release fence for visibility
        // 5. Store new_sibling into self.next (unmarked), completing the link

        #[cfg(feature = "tracing")]
        tracing::debug!(
            self_ptr = ?StdPtr::from_ref(self),
            new_sibling = ?new_sibling,
            "link_sibling: START (CAS-based)"
        );

        // Step 1: Lock the next pointer via CAS mark
        // This returns the unmarked old_next pointer
        let old_next: *mut Self = self.lock_next();

        #[cfg(feature = "tracing")]
        tracing::trace!(
            self_ptr = ?StdPtr::from_ref(self),
            old_next = ?old_next,
            "link_sibling: locked next pointer"
        );

        // SAFETY: Caller guarantees new_sibling is valid
        unsafe {
            // Step 2: Set up new_sibling's pointers
            (*new_sibling).set_prev(StdPtr::from_ref(self).cast_mut());
            (*new_sibling).set_next(old_next);

            // Step 3: Update old_next.prev if non-null
            if !old_next.is_null() {
                (*old_next).set_prev(new_sibling);
            }
        }

        // Step 4: Release fence ensures new_sibling is fully visible before publishing
        std::sync::atomic::fence(std::sync::atomic::Ordering::Release);

        // Step 5: Store new_sibling (unmarked) - atomically publishes the link
        // This also "unlocks" the next pointer by clearing the mark bit
        <Self as crate::leaf_trait::TreeLeafNode<S>>::set_next(self, new_sibling);

        #[cfg(feature = "tracing")]
        tracing::debug!(
            self_ptr = ?StdPtr::from_ref(self),
            new_sibling = ?new_sibling,
            old_next = ?old_next,
            "link_sibling: DONE (CAS-based)"
        );
    }

    #[inline(always)]
    fn ksuf(&self, slot: usize) -> Option<&[u8]> {
        Self::ksuf(self, slot)
    }

    #[inline(always)]
    unsafe fn assign_ksuf(&self, slot: usize, suffix: &[u8], guard: &seize::LocalGuard<'_>) {
        // SAFETY: Caller guarantees preconditions
        unsafe { Self::assign_ksuf(self, slot, suffix, guard) }
    }

    #[inline(always)]
    unsafe fn clear_ksuf(&self, slot: usize, guard: &seize::LocalGuard<'_>) {
        // SAFETY: Caller guarantees preconditions
        unsafe { Self::clear_ksuf(self, slot, guard) }
    }

    #[inline(always)]
    fn take_leaf_value_ptr(&self, slot: usize) -> *mut u8 {
        Self::take_leaf_value_ptr(self, slot)
    }

    // ========================================================================
    // Suffix Comparison Operations (Trait Delegates)
    // ========================================================================

    #[inline(always)]
    fn ksuf_equals(&self, slot: usize, suffix: &[u8]) -> bool {
        Self::ksuf_equals(self, slot, suffix)
    }

    #[inline(always)]
    fn ksuf_compare(&self, slot: usize, suffix: &[u8]) -> Option<std::cmp::Ordering> {
        Self::ksuf_compare(self, slot, suffix)
    }

    #[inline(always)]
    fn ksuf_or_empty(&self, slot: usize) -> &[u8] {
        Self::ksuf_or_empty(self, slot)
    }

    #[inline(always)]
    fn ksuf_matches(&self, slot: usize, ikey: u64, suffix: &[u8]) -> bool {
        Self::ksuf_matches(self, slot, ikey, suffix)
    }

    #[inline(always)]
    fn ksuf_match_result(&self, slot: usize, keylenx: u8, suffix: &[u8]) -> i32 {
        Self::ksuf_match_result(self, slot, keylenx, suffix)
    }
}

// =============================================================================
// Drop Implementation
// =============================================================================

impl<S: ValueSlot> Drop for LeafNode24<S> {
    /// Drop the leaf node, cleaning up stored values and suffix bag.
    ///
    /// This iterates through all slots and drops any non-null value pointers
    /// that are not layer pointers (keylenx < `LAYER_KEYLENX`). Layer pointers
    /// are owned by the tree and cleaned up during tree teardown.
    #[expect(
        clippy::indexing_slicing,
        reason = "slot iterates 0..WIDTH_24 which matches array size"
    )]
    fn drop(&mut self) {
        for slot in 0..WIDTH_24 {
            let ptr: *mut u8 = self.leaf_values[slot].load(RELAXED);
            if ptr.is_null() {
                continue;
            }

            let keylenx: u8 = self.keylenx[slot].load(RELAXED);
            if keylenx < LAYER_KEYLENX {
                // SAFETY: ptr came from the slot type's storage method
                // (Arc::into_raw for LeafValue, Box::into_raw for LeafValueIndex).
                // We only cleanup non-layer slots (keylenx < LAYER_KEYLENX).
                unsafe {
                    S::cleanup_value_ptr(ptr);
                }
            }
            // Note: Layer pointers are owned by the tree and cleaned up
            // during tree teardown, not here.
        }

        let ksuf_ptr: *mut SuffixBag<WIDTH_24> = self.ksuf.load(RELAXED);
        if !ksuf_ptr.is_null() {
            // SAFETY: ksuf_ptr came from Box::into_raw in assign_ksuf.
            unsafe {
                drop(Box::from_raw(ksuf_ptr));
            }
        }
    }
}

// =============================================================================
// LayerCapableLeaf Implementation
// =============================================================================

impl<V: Send + Sync + 'static> crate::leaf_trait::LayerCapableLeaf<V>
    for LeafNode24<crate::value::LeafValue<V>>
{
    fn try_clone_arc(&self, slot: usize) -> Option<std::sync::Arc<V>> {
        debug_assert!(
            slot < WIDTH_24,
            "try_clone_arc: slot {slot} >= WIDTH_24 {WIDTH_24}"
        );

        // Check for layer pointer - layer pointers are NOT Arc values
        if self.keylenx(slot) >= LAYER_KEYLENX {
            return None;
        }

        let ptr: *mut u8 = self.leaf_value_ptr(slot);
        if ptr.is_null() {
            return None;
        }

        // SAFETY:
        // - ptr is non-null (checked above)
        // - ptr is not a layer pointer (keylenx < LAYER_KEYLENX, checked above)
        // - ptr came from Arc::into_raw during insert
        // - Caller ensures slot is stable (lock or version validation)
        unsafe {
            let value_ptr: *const V = ptr.cast();
            std::sync::Arc::increment_strong_count(value_ptr);
            Some(std::sync::Arc::from_raw(value_ptr))
        }
    }

    unsafe fn assign_from_key_arc(
        &self,
        slot: usize,
        key: &crate::key::Key<'_>,
        value: Option<std::sync::Arc<V>>,
        guard: &seize::LocalGuard<'_>,
    ) {
        debug_assert!(
            slot < WIDTH_24,
            "assign_from_key_arc: slot {slot} >= WIDTH_24 {WIDTH_24}"
        );

        // Calculate inline length (0-8 bytes)
        // current_len() returns the remaining key length at current layer
        #[expect(
            clippy::cast_possible_truncation,
            reason = "current_len() capped at slice length, min(8) ensures <= 8"
        )]
        let inline_len: u8 = key.current_len().min(8) as u8;

        // INVARIANT: value must be Some for layer creation
        // Conflict case always has a value, not a layer pointer.
        // If this panics, caller incorrectly identified a layer pointer as a conflict.
        #[expect(
            clippy::expect_used,
            reason = "invariant: source slot must contain value"
        )]
        let arc: std::sync::Arc<V> = value.expect(
            "assign_from_key_arc: value cannot be None (source slot was not a value); \
             this indicates a bug in conflict detection",
        );

        // Store ikey (8 bytes, big-endian encoded)
        self.set_ikey(slot, key.ikey());

        // Store Arc as raw pointer
        // NOTE: Arc ownership transfers to the slot; the slot now owns one strong reference.
        // The caller must NOT drop `value` again - it's been consumed via into_raw.
        let ptr: *mut u8 = std::sync::Arc::into_raw(arc).cast_mut().cast::<u8>();
        self.set_leaf_value_ptr(slot, ptr);

        // Set keylenx and suffix based on whether key has remaining bytes
        if key.has_suffix() {
            // Key has suffix bytes beyond the 8-byte ikey
            self.set_keylenx(slot, KSUF_KEYLENX);

            // Store suffix in suffix bag
            // SAFETY: Caller guarantees guard is from this tree's collector
            unsafe { self.assign_ksuf(slot, key.suffix(), guard) };
        } else {
            // Inline key (0-8 bytes total, no suffix)
            self.set_keylenx(slot, inline_len);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::value::LeafValue;

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

    #[test]
    fn test_lock_next_null() {
        // lock_next on a leaf with NULL next should return NULL and mark the next pointer.
        // This prevents two concurrent splits both observing NULL and racing to publish
        // different siblings (orphaning one).
        let leaf: Box<LeafNode24<LeafValue<u64>>> = LeafNode24::new();
        assert!(leaf.safe_next().is_null());

        let result = leaf.lock_next();
        assert!(result.is_null());
        // After lock_next with NULL, next is marked (but safe_next still reads as NULL)
        assert!(leaf.next_is_marked());
        assert!(leaf.safe_next().is_null());

        // Unmark should restore a clean NULL next
        leaf.unmark_next();
        assert!(!leaf.next_is_marked());
    }

    #[test]
    fn test_lock_next_marks_and_unmarks() {
        // lock_next should mark the next pointer, returned value is unmarked
        let leaf1: Box<LeafNode24<LeafValue<u64>>> = LeafNode24::new();
        let leaf2: Box<LeafNode24<LeafValue<u64>>> = LeafNode24::new();

        let leaf2_ptr: *mut LeafNode24<LeafValue<u64>> = Box::into_raw(leaf2);
        leaf1.set_next(leaf2_ptr);

        // Before lock_next: not marked
        assert!(!leaf1.next_is_marked());

        // Call lock_next
        let result = leaf1.lock_next();

        // Returned value should be the unmarked pointer
        assert_eq!(result, leaf2_ptr);
        // But the next pointer should now be marked
        assert!(leaf1.next_is_marked());
        // safe_next still returns the unmarked pointer
        assert_eq!(leaf1.safe_next(), leaf2_ptr);

        // "Unlock" by storing the unmarked pointer
        leaf1.set_next(leaf2_ptr);
        assert!(!leaf1.next_is_marked());

        // Cleanup
        let _ = unsafe { Box::from_raw(leaf2_ptr) };
    }

    #[test]
    #[expect(clippy::similar_names)]
    fn test_link_sibling_preserves_existing_next() {
        // Set up chain: A -> B
        // After link A -> C: should have A -> C -> B with correct prev pointers
        use crate::leaf_trait::TreeLeafNode;

        let leaf_a: Box<LeafNode24<LeafValue<u64>>> = LeafNode24::new();
        let leaf_b: Box<LeafNode24<LeafValue<u64>>> = LeafNode24::new();
        let leaf_c: Box<LeafNode24<LeafValue<u64>>> = LeafNode24::new();

        let leaf_a_ptr: *mut LeafNode24<LeafValue<u64>> = Box::into_raw(leaf_a);
        let leaf_b_ptr: *mut LeafNode24<LeafValue<u64>> = Box::into_raw(leaf_b);
        let leaf_c_ptr: *mut LeafNode24<LeafValue<u64>> = Box::into_raw(leaf_c);

        // Reconstruct references
        let leaf_a: &LeafNode24<LeafValue<u64>> = unsafe { &*leaf_a_ptr };
        let leaf_b: &LeafNode24<LeafValue<u64>> = unsafe { &*leaf_b_ptr };
        let leaf_c: &LeafNode24<LeafValue<u64>> = unsafe { &*leaf_c_ptr };

        // Initial chain: A -> B
        leaf_a.set_next(leaf_b_ptr);
        leaf_b.set_prev(leaf_a_ptr);

        // Link C after A: A -> C -> B
        unsafe { TreeLeafNode::<LeafValue<u64>>::link_sibling(leaf_a, leaf_c_ptr) };

        // Verify chain structure
        assert_eq!(leaf_a.safe_next(), leaf_c_ptr, "A.next should be C");
        assert_eq!(leaf_c.safe_next(), leaf_b_ptr, "C.next should be B");
        assert_eq!(leaf_c.prev(), leaf_a_ptr, "C.prev should be A");
        assert_eq!(leaf_b.prev(), leaf_c_ptr, "B.prev should be C");

        // Verify not marked after link
        assert!(!leaf_a.next_is_marked());
        assert!(!leaf_c.next_is_marked());

        // Cleanup
        let _ = unsafe { Box::from_raw(leaf_a_ptr) };
        let _ = unsafe { Box::from_raw(leaf_b_ptr) };
        let _ = unsafe { Box::from_raw(leaf_c_ptr) };
    }

    #[test]
    #[expect(clippy::similar_names)]
    fn test_link_sibling_null_next() {
        // Link sibling when next is NULL: A -> NULL becomes A -> C -> NULL
        use crate::leaf_trait::TreeLeafNode;

        let leaf_a: Box<LeafNode24<LeafValue<u64>>> = LeafNode24::new();
        let leaf_c: Box<LeafNode24<LeafValue<u64>>> = LeafNode24::new();

        let leaf_a_ptr: *mut LeafNode24<LeafValue<u64>> = Box::into_raw(leaf_a);
        let leaf_c_ptr: *mut LeafNode24<LeafValue<u64>> = Box::into_raw(leaf_c);

        let leaf_a: &LeafNode24<LeafValue<u64>> = unsafe { &*leaf_a_ptr };
        let leaf_c: &LeafNode24<LeafValue<u64>> = unsafe { &*leaf_c_ptr };

        // A has no next
        assert!(leaf_a.safe_next().is_null());

        // Link C after A
        unsafe { TreeLeafNode::<LeafValue<u64>>::link_sibling(leaf_a, leaf_c_ptr) };

        // Verify chain
        assert_eq!(leaf_a.safe_next(), leaf_c_ptr, "A.next should be C");
        assert!(leaf_c.safe_next().is_null(), "C.next should be NULL");
        assert_eq!(leaf_c.prev(), leaf_a_ptr, "C.prev should be A");
        assert!(!leaf_a.next_is_marked());

        // Cleanup
        let _ = unsafe { Box::from_raw(leaf_a_ptr) };
        let _ = unsafe { Box::from_raw(leaf_c_ptr) };
    }

    #[test]
    #[expect(clippy::unwrap_used)]
    fn test_cas_next() {
        let leaf1: Box<LeafNode24<LeafValue<u64>>> = LeafNode24::new();
        let leaf2: Box<LeafNode24<LeafValue<u64>>> = LeafNode24::new();

        let leaf2_ptr: *mut LeafNode24<LeafValue<u64>> = Box::into_raw(leaf2);

        // CAS from NULL to leaf2_ptr should succeed
        let result = leaf1.cas_next(std::ptr::null_mut(), leaf2_ptr);
        assert!(result.is_ok());
        assert_eq!(leaf1.safe_next(), leaf2_ptr);

        // CAS with wrong expected should fail
        let result = leaf1.cas_next(std::ptr::null_mut(), std::ptr::null_mut());
        assert!(result.is_err());
        assert_eq!(result.unwrap_err(), leaf2_ptr);

        // Cleanup
        let _ = unsafe { Box::from_raw(leaf2_ptr) };
    }
}
