//! Filepath: src/leaf15.rs
//!
//! Leaf node for [`MassTree`] with WIDTH=15 (15 slots).
//!
//! This module provides `LeafNode15`, a leaf node matching the C++ reference
//! implementation's default `nodeparams<15, 15>`. Uses a u64 permutation
//! (4 bits per slot) for better cache efficiency.
//!
//! # Design
//!
//! The 15-slot design uses 4 bits per slot: 4 (size) + 15×4 (slots) = 64 bits.
//! This matches C++ Masstree and provides better cache locality than WIDTH=24.

#![allow(missing_docs)] // Internal module matching leaf24.rs API

use std::fmt as StdFmt;
use std::marker::PhantomData;
use std::ptr as StdPtr;
use std::sync::atomic::{AtomicPtr, AtomicU64, AtomicU8};

use crate::nodeversion::NodeVersion;
use crate::ordering::{READ_ORD, RELAXED, WRITE_ORD};
use crate::permuter::Permuter15;
use crate::slot::ValueSlot;
use crate::suffix::SuffixBag;
use seize::{Guard, LocalGuard};

/// Special keylenx value indicating key has a suffix.
pub const KSUF_KEYLENX: u8 = 64;

/// Base keylenx value indicating a layer pointer (>= this means layer).
pub const LAYER_KEYLENX: u8 = 128;

/// Width constant for [`LeafNode15`].
pub const WIDTH_15: usize = 15;

/// Modification state values.
#[repr(u8)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ModState15 {
    /// Node is in insert mode (normal operation).
    Insert = 0,

    /// Node is being removed.
    Remove = 1,

    /// Node's layer has been deleted.
    DeletedLayer = 2,
}

/// Leaf node with 15 slots using u64 permutation (matches C++ default).
///
/// # Memory Layout
///
/// ```text
/// Cache Line 0 (64 bytes): version + modstate + permutation (u64) + padding
/// Cache Lines 1+: keys, keylenx, values (15 slots each)
/// ```
///
/// Node size: ~240 bytes (~4 cache lines) vs LeafNode24's ~384 bytes (~6 cache lines).
#[repr(C, align(64))]
pub struct LeafNode15<S: ValueSlot> {
    // ========================================================================
    // Cache Line 0: Version + permutation (hot path together)
    // ========================================================================
    /// Version for optimistic concurrency control.
    version: NodeVersion,

    /// Modification state for suffix operations.
    modstate: ModState15,

    /// Padding after modstate (version is 8 bytes, modstate is 1 byte).
    _pad0: [u8; 47],

    /// Permutation using u64 for 15-slot support (4 bits per slot).
    /// Store is linearization point for new slot visibility.
    permutation: AtomicU64,

    // ========================================================================
    // Cache Lines 1+: Keys and values (read during search, written on insert)
    // ========================================================================
    /// 8-byte keys for each slot.
    ikey0: [AtomicU64; WIDTH_15],

    /// Key length/type for each slot.
    keylenx: [AtomicU8; WIDTH_15],

    /// Values/layer pointers for each slot.
    leaf_values: [AtomicPtr<u8>; WIDTH_15],

    /// Suffix storage.
    ksuf: AtomicPtr<SuffixBag<WIDTH_15>>,

    /// Next leaf with mark bit in LSB for split coordination.
    next: AtomicPtr<Self>,

    /// Previous leaf.
    prev: AtomicPtr<Self>,

    /// Parent internode.
    parent: AtomicPtr<u8>,

    /// Phantom for slot type.
    _marker: PhantomData<S>,
}

impl<S: ValueSlot> StdFmt::Debug for LeafNode15<S> {
    fn fmt(&self, f: &mut StdFmt::Formatter<'_>) -> StdFmt::Result {
        f.debug_struct("LeafNode15")
            .field("size", &self.size())
            .field("is_root", &self.version.is_root())
            .field("has_parent", &(!self.parent().is_null()))
            .finish_non_exhaustive()
    }
}

impl<S: ValueSlot> LeafNode15<S> {
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
            modstate: ModState15::Insert,
            _pad0: [0; 47],
            permutation: AtomicU64::new(Permuter15::empty().value()),
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

    // ============================================================================
    //  NodeVersion Accessors
    // ============================================================================

    #[inline(always)]
    pub const fn version(&self) -> &NodeVersion {
        &self.version
    }

    #[inline(always)]
    pub const fn version_mut(&mut self) -> &mut NodeVersion {
        &mut self.version
    }

    // ============================================================================
    //  Key Accessors
    // ============================================================================

    #[must_use]
    #[inline(always)]
    #[expect(clippy::indexing_slicing, reason = "Slot from Permuter, valid by construction")]
    pub fn ikey(&self, slot: usize) -> u64 {
        debug_assert!(slot < WIDTH_15, "ikey: slot out of bounds");
        self.ikey0[slot].load(READ_ORD)
    }

    #[inline(always)]
    #[expect(clippy::indexing_slicing, reason = "Slot from Permuter, valid by construction")]
    pub fn set_ikey(&self, slot: usize, ikey: u64) {
        debug_assert!(slot < WIDTH_15, "set_ikey: slot out of bounds");
        self.ikey0[slot].store(ikey, WRITE_ORD);
    }

    #[must_use]
    #[inline(always)]
    #[expect(clippy::indexing_slicing)]
    pub fn load_all_ikeys(&self) -> [u64; WIDTH_15] {
        let mut ikeys = [0u64; WIDTH_15];
        (0..WIDTH_15).for_each(|i| {
            ikeys[i] = self.ikey0[i].load(READ_ORD);
        });
        ikeys
    }

    /// Prefetch leaf node data for range scans.
    #[inline(always)]
    pub fn prefetch(&self) {
        use crate::prefetch::prefetch_read;

        let self_ptr: *const u8 = StdPtr::from_ref::<Self>(self).cast::<u8>();

        // Prefetch ikey0 array (15 × 8B = 120 bytes, ~2 cache lines)
        unsafe {
            prefetch_read(self_ptr.add(64));  // ikey0[0..8]
            prefetch_read(self_ptr.add(128)); // ikey0[8..15] + keylenx
            prefetch_read(self_ptr.add(192)); // leaf_values
        }
    }

    #[must_use]
    #[inline(always)]
    #[expect(clippy::indexing_slicing, reason = "Slot from Permuter, valid by construction")]
    pub fn keylenx(&self, slot: usize) -> u8 {
        debug_assert!(slot < WIDTH_15, "keylenx: slot out of bounds");
        self.keylenx[slot].load(READ_ORD)
    }

    #[inline(always)]
    #[expect(clippy::indexing_slicing, reason = "Slot from Permuter, valid by construction")]
    pub fn set_keylenx(&self, slot: usize, keylenx: u8) {
        debug_assert!(slot < WIDTH_15, "set_keylenx: slot out of bounds");
        self.keylenx[slot].store(keylenx, WRITE_ORD);
    }

    #[must_use]
    #[inline(always)]
    pub fn ikey_bound(&self) -> u64 {
        self.ikey0[0].load(READ_ORD)
    }

    #[inline(always)]
    pub fn keylenx_bound(&self) -> u8 {
        let perm: Permuter15 = self.permutation();
        debug_assert!(perm.size() > 0, "keylenx_bound called on empty_leaf");
        self.keylenx(perm.get(0))
    }

    #[must_use]
    #[inline(always)]
    pub fn is_layer(&self, slot: usize) -> bool {
        self.keylenx(slot) >= LAYER_KEYLENX
    }

    #[must_use]
    #[inline(always)]
    pub fn has_ksuf(&self, slot: usize) -> bool {
        self.keylenx(slot) == KSUF_KEYLENX
    }

    #[inline(always)]
    #[must_use]
    pub const fn keylenx_is_layer(keylenx: u8) -> bool {
        keylenx >= LAYER_KEYLENX
    }

    #[must_use]
    #[inline(always)]
    pub const fn keylenx_has_ksuf(keylenx: u8) -> bool {
        keylenx == KSUF_KEYLENX
    }

    // ============================================================================
    //  Suffix Storage Methods
    // ============================================================================

    #[must_use]
    #[inline(always)]
    pub fn ksuf_ptr(&self) -> *mut SuffixBag<WIDTH_15> {
        self.ksuf.load(READ_ORD)
    }

    #[must_use]
    #[inline(always)]
    pub fn has_ksuf_storage(&self) -> bool {
        !self.ksuf_ptr().is_null()
    }

    #[must_use]
    pub fn ksuf(&self, slot: usize) -> Option<&[u8]> {
        debug_assert!(slot < WIDTH_15, "ksuf: slot {slot} >= WIDTH_15");
        if !self.has_ksuf(slot) {
            return None;
        }
        let ptr = self.ksuf_ptr();
        if ptr.is_null() {
            return None;
        }
        unsafe { (*ptr).get(slot) }
    }

    #[must_use]
    #[inline(always)]
    pub fn ksuf_or_empty(&self, slot: usize) -> &[u8] {
        self.ksuf(slot).unwrap_or(&[])
    }

    #[expect(clippy::indexing_slicing, reason = "Slot bounds checked via debug_assert")]
    pub unsafe fn assign_ksuf(&self, slot: usize, suffix: &[u8], guard: &LocalGuard<'_>) {
        debug_assert!(slot < WIDTH_15, "assign_ksuf: slot {slot} >= WIDTH_15");

        let old_ptr: *mut SuffixBag<WIDTH_15> = self.ksuf.load(RELAXED);

        if !old_ptr.is_null() {
            let bag: &mut SuffixBag<WIDTH_15> = unsafe { &mut *old_ptr };
            if bag.try_assign_in_place(slot, suffix) {
                self.keylenx[slot].store(KSUF_KEYLENX, WRITE_ORD);
                return;
            }
        }

        let mut new_bag: SuffixBag<WIDTH_15> = if old_ptr.is_null() {
            SuffixBag::new()
        } else {
            unsafe { (*old_ptr).clone() }
        };

        new_bag.assign(slot, suffix);
        let new_ptr: *mut SuffixBag<WIDTH_15> = Box::into_raw(Box::new(new_bag));

        self.ksuf.store(new_ptr, WRITE_ORD);

        if !old_ptr.is_null() {
            unsafe {
                guard.defer_retire(old_ptr, |ptr, _| {
                    drop(Box::from_raw(ptr));
                });
            }
        }

        self.keylenx[slot].store(KSUF_KEYLENX, WRITE_ORD);
    }

    #[expect(clippy::indexing_slicing, reason = "Slot bounds checked via debug_assert")]
    pub unsafe fn clear_ksuf(&self, slot: usize, guard: &LocalGuard<'_>) {
        debug_assert!(slot < WIDTH_15, "clear_ksuf: slot {slot} >= WIDTH_15");

        let old_ptr: *mut SuffixBag<WIDTH_15> = self.ksuf.load(RELAXED);
        if old_ptr.is_null() {
            self.keylenx[slot].store(0, WRITE_ORD);
            return;
        }

        let mut new_bag: SuffixBag<WIDTH_15> = unsafe { (*old_ptr).clone() };
        new_bag.clear(slot);
        let new_ptr: *mut SuffixBag<WIDTH_15> = Box::into_raw(Box::new(new_bag));

        self.ksuf.store(new_ptr, WRITE_ORD);

        unsafe {
            guard.defer_retire(old_ptr, |ptr, _| {
                drop(Box::from_raw(ptr));
            });
        }

        self.keylenx[slot].store(0, WRITE_ORD);
    }

    #[must_use]
    pub fn ksuf_equals(&self, slot: usize, suffix: &[u8]) -> bool {
        debug_assert!(slot < WIDTH_15, "ksuf_equals: slot {slot} >= WIDTH_15");
        if !self.has_ksuf(slot) {
            return false;
        }
        let ptr = self.ksuf_ptr();
        if ptr.is_null() {
            return false;
        }
        unsafe { (*ptr).suffix_equals(slot, suffix) }
    }

    #[must_use]
    pub fn ksuf_compare(&self, slot: usize, suffix: &[u8]) -> Option<std::cmp::Ordering> {
        debug_assert!(slot < WIDTH_15, "ksuf_compare: slot {slot} >= WIDTH_15");
        if !self.has_ksuf(slot) {
            return None;
        }
        let ptr = self.ksuf_ptr();
        if ptr.is_null() {
            return None;
        }
        unsafe { (*ptr).suffix_compare(slot, suffix) }
    }

    #[must_use]
    pub fn ksuf_matches(&self, slot: usize, ikey: u64, suffix: &[u8]) -> bool {
        debug_assert!(slot < WIDTH_15, "ksuf_matches: slot {slot} >= WIDTH_15");
        if self.ikey(slot) != ikey {
            return false;
        }
        if suffix.is_empty() {
            !self.has_ksuf(slot)
        } else {
            self.ksuf_equals(slot, suffix)
        }
    }

    #[must_use]
    #[inline(always)]
    #[expect(clippy::cast_possible_wrap, clippy::cast_possible_truncation)]
    pub fn ksuf_match_result(&self, slot: usize, keylenx: u8, suffix: &[u8]) -> i32 {
        use crate::key::IKEY_SIZE;

        debug_assert!(slot < WIDTH_15, "ksuf_match_result: slot {slot} >= WIDTH_15");

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

    pub unsafe fn compact_ksuf(
        &self,
        exclude_slot: Option<usize>,
        guard: &LocalGuard<'_>,
    ) -> usize {
        let old_ptr: *mut SuffixBag<WIDTH_15> = self.ksuf.load(RELAXED);
        if old_ptr.is_null() {
            return 0;
        }

        let perm = self.permutation();
        let mut new_bag: SuffixBag<WIDTH_15> = unsafe { (*old_ptr).clone() };
        let reclaimed = new_bag.compact_with_permuter(&perm, exclude_slot);
        let new_ptr: *mut SuffixBag<WIDTH_15> = Box::into_raw(Box::new(new_bag));

        self.ksuf.store(new_ptr, WRITE_ORD);

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

    #[inline(always)]
    #[expect(clippy::indexing_slicing, reason = "Slot from Permuter; valid by construction")]
    pub fn leaf_value_ptr(&self, slot: usize) -> *mut u8 {
        debug_assert!(slot < WIDTH_15, "leaf_value_ptr: slot out of bounds");
        self.leaf_values[slot].load(READ_ORD)
    }

    #[inline(always)]
    #[expect(clippy::indexing_slicing, reason = "Slot from Permuter; valid by construction")]
    pub fn set_leaf_value_ptr(&self, slot: usize, ptr: *mut u8) {
        debug_assert!(slot < WIDTH_15, "set_leaf_value_ptr: slot out of bounds");
        self.leaf_values[slot].store(ptr, WRITE_ORD);
    }

    #[inline(always)]
    #[expect(clippy::indexing_slicing, reason = "Slot from Permuter; valid by construction")]
    pub fn take_leaf_value_ptr(&self, slot: usize) -> *mut u8 {
        debug_assert!(slot < WIDTH_15, "take_leaf_value_ptr: slot out of bounds");
        self.leaf_values[slot].swap(StdPtr::null_mut(), RELAXED)
    }

    #[inline(always)]
    #[must_use]
    pub fn is_slot_empty(&self, slot: usize) -> bool {
        self.leaf_value_ptr(slot).is_null()
    }

    // ============================================================================
    //  Permutation Accessors
    // ============================================================================

    #[inline(always)]
    #[must_use]
    pub fn permutation(&self) -> Permuter15 {
        Permuter15::from_value(self.permutation.load(READ_ORD))
    }

    #[inline(always)]
    pub fn set_permutation(&self, perm: Permuter15) {
        self.permutation.store(perm.value(), WRITE_ORD);
    }

    #[inline(always)]
    #[must_use]
    pub fn permutation_raw(&self) -> u64 {
        self.permutation.load(READ_ORD)
    }

    #[inline(always)]
    #[expect(clippy::indexing_slicing, reason = "bounds checked by debug_assert")]
    pub fn cas_slot_value(
        &self,
        slot: usize,
        expected: *mut u8,
        new_value: *mut u8,
    ) -> Result<(), *mut u8> {
        use crate::ordering::{CAS_FAILURE, CAS_SUCCESS};
        debug_assert!(slot < WIDTH_15, "cas_slot_value: slot out of bounds");

        match self.leaf_values[slot].compare_exchange(expected, new_value, CAS_SUCCESS, CAS_FAILURE)
        {
            Ok(_) => Ok(()),
            Err(actual) => Err(actual),
        }
    }

    #[inline(always)]
    #[expect(clippy::indexing_slicing, reason = "bounds checked by debug_assert")]
    pub fn load_slot_value(&self, slot: usize) -> *mut u8 {
        debug_assert!(slot < WIDTH_15, "load_slot_value: slot out of bounds");
        self.leaf_values[slot].load(READ_ORD)
    }

    #[inline(always)]
    #[expect(clippy::indexing_slicing, reason = "bounds checked by debug_assert")]
    pub unsafe fn store_key_data_for_cas(&self, slot: usize, ikey: u64, keylenx: u8) {
        debug_assert!(slot < WIDTH_15, "store_key_data_for_cas: slot out of bounds");
        self.ikey0[slot].store(ikey, WRITE_ORD);
        self.keylenx[slot].store(keylenx, WRITE_ORD);
    }

    #[must_use]
    #[inline(always)]
    pub fn size(&self) -> usize {
        self.permutation().size()
    }

    #[must_use]
    #[inline(always)]
    pub fn is_empty(&self) -> bool {
        self.size() == 0
    }

    #[must_use]
    #[inline(always)]
    pub fn is_full(&self) -> bool {
        self.size() >= WIDTH_15
    }

    // ============================================================================
    //  Leaf Linking
    // ============================================================================

    #[must_use]
    #[inline(always)]
    pub fn safe_next(&self) -> *mut Self {
        let ptr: *mut Self = self.next.load(READ_ORD);
        ptr.map_addr(|addr: usize| addr & !1)
    }

    #[must_use]
    #[inline(always)]
    pub fn next_raw(&self) -> *mut Self {
        self.next.load(READ_ORD)
    }

    #[must_use]
    #[inline(always)]
    pub fn next_is_marked(&self) -> bool {
        (self.next.load(READ_ORD).addr() & 1) != 0
    }

    #[inline(always)]
    pub fn set_next(&self, next: *mut Self) {
        self.next.store(next, WRITE_ORD);
    }

    #[inline(always)]
    pub fn mark_next(&self) {
        let ptr: *mut Self = self.next.load(RELAXED);
        let marked: *mut Self = ptr.map_addr(|addr: usize| addr | 1);
        self.next.store(marked, WRITE_ORD);
    }

    #[inline(always)]
    pub fn unmark_next(&self) {
        let ptr: *mut Self = self.safe_next();
        self.next.store(ptr, WRITE_ORD);
    }

    pub fn wait_for_split(&self) {
        const MAX_RETRIES: usize = 1000;
        let mut retries: usize = 0;

        while self.next_is_marked() {
            for _ in 0..16 {
                std::hint::spin_loop();
                if !self.next_is_marked() {
                    return;
                }
            }
            let _ = self.version.stable();
            retries += 1;
            if retries > MAX_RETRIES {
                break;
            }
        }
    }

    fn lock_next(&self) -> *mut Self {
        use crate::link::{is_marked, mark_ptr};
        use crate::ordering::{CAS_FAILURE, CAS_SUCCESS};

        loop {
            let next: *mut Self = self.next.load(READ_ORD);
            if is_marked(next) {
                self.wait_for_split();
                continue;
            }
            let marked: *mut Self = mark_ptr(next);
            match self.next.compare_exchange(next, marked, CAS_SUCCESS, CAS_FAILURE) {
                Ok(_) => return next,
                Err(_) => std::hint::spin_loop(),
            }
        }
    }

    #[inline]
    #[cfg_attr(not(test), allow(dead_code))]
    fn cas_next(&self, current: *mut Self, new: *mut Self) -> Result<*mut Self, *mut Self> {
        use crate::ordering::{CAS_FAILURE, CAS_SUCCESS};
        self.next.compare_exchange(current, new, CAS_SUCCESS, CAS_FAILURE)
    }

    #[must_use]
    #[inline(always)]
    pub fn prev(&self) -> *mut Self {
        self.prev.load(READ_ORD)
    }

    #[inline(always)]
    pub fn set_prev(&self, prev: *mut Self) {
        self.prev.store(prev, WRITE_ORD);
    }

    // ============================================================================
    //  Parent Accessors
    // ============================================================================

    #[must_use]
    #[inline(always)]
    pub fn parent(&self) -> *mut u8 {
        self.parent.load(READ_ORD)
    }

    #[inline(always)]
    pub fn set_parent(&self, parent: *mut u8) {
        self.parent.store(parent, WRITE_ORD);
    }

    // ============================================================================
    //  ModState Accessors
    // ============================================================================

    #[must_use]
    #[inline(always)]
    pub const fn modstate(&self) -> ModState15 {
        self.modstate
    }

    #[inline(always)]
    pub const fn set_modstate(&mut self, state: ModState15) {
        self.modstate = state;
    }

    // ============================================================================
    //  Slot Assignment
    // ============================================================================

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

unsafe impl<S: ValueSlot + Send + Sync> Send for LeafNode15<S> {}
unsafe impl<S: ValueSlot + Send + Sync> Sync for LeafNode15<S> {}

// ============================================================================
//  TreeLeafNode Implementation
// ============================================================================

impl<S: ValueSlot + Send + Sync + 'static> crate::leaf_trait::TreeLeafNode<S> for LeafNode15<S> {
    type Perm = Permuter15;
    type Internode = crate::internode::InternodeNode<S, 15>;
    const WIDTH: usize = WIDTH_15;

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
    fn permutation(&self) -> Permuter15 {
        Self::permutation(self)
    }

    #[inline(always)]
    fn set_permutation(&self, perm: Permuter15) {
        Self::set_permutation(self, perm);
    }

    #[inline(always)]
    fn permutation_raw(&self) -> u64 {
        Self::permutation_raw(self)
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

    #[inline]
    fn find_ikey_matches(&self, target_ikey: u64) -> u32 {
        // Linear search for WIDTH=15 (SIMD optimization not implemented yet)
        let perm = self.permutation();
        let mut mask: u32 = 0;
        for i in 0..perm.size() {
            let slot = perm.get(i);
            if self.ikey(slot) == target_ikey {
                mask |= 1 << slot;
            }
        }
        mask
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
    unsafe fn store_key_data_for_cas(&self, slot: usize, ikey: u64, keylenx: u8) {
        unsafe { Self::store_key_data_for_cas(self, slot, ikey, keylenx) }
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

    // ========================================================================
    // Split Operations
    // ========================================================================

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

        let mut split_pos = size / 2;
        if split_pos == 0 {
            return None;
        }

        while split_pos > 0 && split_pos < size {
            let left_slot = perm.get(split_pos - 1);
            let right_slot = perm.get(split_pos);
            let left_ikey = self.ikey(left_slot);
            let right_ikey = self.ikey(right_slot);

            if left_ikey == right_ikey {
                match insert_ikey.cmp(&left_ikey) {
                    std::cmp::Ordering::Equal => split_pos += 1,
                    std::cmp::Ordering::Less => split_pos -= 1,
                    std::cmp::Ordering::Greater => break,
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

    unsafe fn split_into_preallocated(
        &self,
        split_pos: usize,
        new_leaf: Box<Self>,
        guard: &seize::LocalGuard<'_>,
    ) -> (Box<Self>, u64, crate::value::InsertTarget) {
        unsafe {
            let split_version = crate::nodeversion::NodeVersion::new_for_split(&self.version);
            std::ptr::write(
                std::ptr::addr_of!(new_leaf.version).cast_mut(),
                split_version,
            );

            let old_perm: Permuter15 = self.permutation();
            let old_size = old_perm.size();

            debug_assert!(
                split_pos > 0 && split_pos < old_size,
                "invalid split_pos {split_pos} for size {old_size}"
            );

            let entries_to_move = old_size - split_pos;

            for i in 0..entries_to_move {
                let old_logical_pos = split_pos + i;
                let old_slot = old_perm.get(old_logical_pos);
                let new_slot = i;

                let ikey = self.ikey(old_slot);
                let keylenx = self.keylenx(old_slot);

                new_leaf.set_ikey(new_slot, ikey);
                new_leaf.set_keylenx(new_slot, keylenx);

                let old_ptr = self.take_leaf_value_ptr(old_slot);
                new_leaf.set_leaf_value_ptr(new_slot, old_ptr);

                if keylenx == KSUF_KEYLENX {
                    if let Some(suffix) = self.ksuf(old_slot) {
                        new_leaf.assign_ksuf(new_slot, suffix, guard);
                    }
                    self.clear_ksuf(old_slot, guard);
                }
            }

            let new_perm = Permuter15::make_sorted(entries_to_move);
            new_leaf.set_permutation(new_perm);

            let mut old_perm_updated = old_perm;
            old_perm_updated.set_size(split_pos);
            self.set_permutation(old_perm_updated);

            let split_ikey = new_leaf.ikey(new_perm.get(0));

            (new_leaf, split_ikey, crate::value::InsertTarget::Left)
        }
    }

    unsafe fn split_all_to_right_preallocated(
        &self,
        new_leaf: Box<Self>,
        guard: &seize::LocalGuard<'_>,
    ) -> (Box<Self>, u64, crate::value::InsertTarget) {
        let split_version = crate::nodeversion::NodeVersion::new_for_split(&self.version);
        unsafe {
            std::ptr::write(
                std::ptr::addr_of!(new_leaf.version).cast_mut(),
                split_version,
            );
        }

        let old_perm: Permuter15 = self.permutation();
        let old_size = old_perm.size();

        debug_assert!(old_size > 0, "Cannot split empty leaf");

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
                    unsafe { new_leaf.assign_ksuf(new_slot, suffix, guard) };
                }
                unsafe { self.clear_ksuf(old_slot, guard) };
            }
        }

        let new_perm = Permuter15::make_sorted(old_size);
        new_leaf.set_permutation(new_perm);

        self.set_permutation(Permuter15::empty());

        let split_ikey = new_leaf.ikey(new_perm.get(0));

        (new_leaf, split_ikey, crate::value::InsertTarget::Right)
    }

    #[inline(always)]
    unsafe fn link_sibling(&self, new_sibling: *mut Self) {
        let old_next: *mut Self = self.lock_next();

        unsafe {
            (*new_sibling).set_prev(StdPtr::from_ref(self).cast_mut());
            (*new_sibling).set_next(old_next);

            if !old_next.is_null() {
                (*old_next).set_prev(new_sibling);
            }
        }

        std::sync::atomic::fence(std::sync::atomic::Ordering::Release);
        <Self as crate::leaf_trait::TreeLeafNode<S>>::set_next(self, new_sibling);
    }

    #[inline(always)]
    fn ksuf(&self, slot: usize) -> Option<&[u8]> {
        Self::ksuf(self, slot)
    }

    #[inline(always)]
    unsafe fn assign_ksuf(&self, slot: usize, suffix: &[u8], guard: &seize::LocalGuard<'_>) {
        unsafe { Self::assign_ksuf(self, slot, suffix, guard) }
    }

    #[inline(always)]
    unsafe fn clear_ksuf(&self, slot: usize, guard: &seize::LocalGuard<'_>) {
        unsafe { Self::clear_ksuf(self, slot, guard) }
    }

    #[inline(always)]
    fn take_leaf_value_ptr(&self, slot: usize) -> *mut u8 {
        Self::take_leaf_value_ptr(self, slot)
    }

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

    #[inline(always)]
    fn prefetch(&self) {
        Self::prefetch(self);
    }
}

// =============================================================================
// Drop Implementation
// =============================================================================

impl<S: ValueSlot> Drop for LeafNode15<S> {
    #[expect(clippy::indexing_slicing, reason = "slot iterates 0..WIDTH_15 which matches array size")]
    fn drop(&mut self) {
        for slot in 0..WIDTH_15 {
            let ptr: *mut u8 = self.leaf_values[slot].load(RELAXED);
            if ptr.is_null() {
                continue;
            }

            let keylenx: u8 = self.keylenx[slot].load(RELAXED);
            if keylenx < LAYER_KEYLENX {
                unsafe {
                    S::cleanup_value_ptr(ptr);
                }
            }
        }

        let ksuf_ptr: *mut SuffixBag<WIDTH_15> = self.ksuf.load(RELAXED);
        if !ksuf_ptr.is_null() {
            unsafe {
                drop(Box::from_raw(ksuf_ptr));
            }
        }
    }
}

// =============================================================================
// LayerCapableLeaf Implementation
// =============================================================================

impl<V: Send + Sync + 'static> crate::leaf_trait::LayerCapableLeaf<crate::value::LeafValue<V>>
    for LeafNode15<crate::value::LeafValue<V>>
{
    fn try_clone_output(&self, slot: usize) -> Option<std::sync::Arc<V>> {
        debug_assert!(slot < WIDTH_15, "try_clone_arc: slot {slot} >= WIDTH_15");

        if self.keylenx(slot) >= LAYER_KEYLENX {
            return None;
        }

        let ptr: *mut u8 = self.leaf_value_ptr(slot);
        if ptr.is_null() {
            return None;
        }

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
        debug_assert!(slot < WIDTH_15, "assign_from_key_arc: slot {slot} >= WIDTH_15");

        #[expect(clippy::cast_possible_truncation)]
        let inline_len: u8 = key.current_len().min(8) as u8;

        #[expect(clippy::expect_used)]
        let arc: std::sync::Arc<V> = value.expect("assign_from_key_arc: value cannot be None");

        self.set_ikey(slot, key.ikey());
        let ptr: *mut u8 = std::sync::Arc::into_raw(arc).cast_mut().cast::<u8>();
        self.set_leaf_value_ptr(slot, ptr);

        if key.has_suffix() {
            self.set_keylenx(slot, KSUF_KEYLENX);
            unsafe { self.assign_ksuf(slot, key.suffix(), guard) };
        } else {
            self.set_keylenx(slot, inline_len);
        }
    }
}

impl<V: Copy + Send + Sync + 'static>
    crate::leaf_trait::LayerCapableLeaf<crate::value::LeafValueIndex<V>>
    for LeafNode15<crate::value::LeafValueIndex<V>>
{
    fn try_clone_output(&self, slot: usize) -> Option<V> {
        debug_assert!(slot < WIDTH_15, "try_clone_output: slot {slot} >= WIDTH_15");

        if self.keylenx(slot) >= LAYER_KEYLENX {
            return None;
        }

        let ptr: *mut u8 = self.leaf_value_ptr(slot);
        if ptr.is_null() {
            return None;
        }

        unsafe { Some(*ptr.cast::<V>()) }
    }

    unsafe fn assign_from_key_arc(
        &self,
        slot: usize,
        key: &crate::key::Key<'_>,
        value: Option<V>,
        guard: &seize::LocalGuard<'_>,
    ) {
        debug_assert!(slot < WIDTH_15, "assign_from_key_arc: slot {slot} >= WIDTH_15");

        #[expect(clippy::cast_possible_truncation)]
        let inline_len: u8 = key.current_len().min(8) as u8;

        #[expect(clippy::expect_used)]
        let v: V = value.expect("assign_from_key_arc: value cannot be None");

        self.set_ikey(slot, key.ikey());
        let ptr: *mut u8 = Box::into_raw(Box::new(v)).cast::<u8>();
        self.set_leaf_value_ptr(slot, ptr);

        if key.has_suffix() {
            self.set_keylenx(slot, KSUF_KEYLENX);
            unsafe { self.assign_ksuf(slot, key.suffix(), guard) };
        } else {
            self.set_keylenx(slot, inline_len);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::value::LeafValue;

    #[test]
    fn test_new_leaf15_is_empty() {
        let leaf: Box<LeafNode15<LeafValue<u64>>> = LeafNode15::new();
        assert!(leaf.is_empty());
        assert_eq!(leaf.size(), 0);
        assert!(!leaf.is_full());
    }

    #[test]
    fn test_leaf15_full_at_15_slots() {
        let leaf: Box<LeafNode15<LeafValue<u64>>> = LeafNode15::new();
        let mut perm = leaf.permutation();

        for i in 0..WIDTH_15 {
            let _slot = perm.insert_from_back(i);
        }
        leaf.set_permutation(perm);

        assert!(leaf.is_full());
        assert_eq!(leaf.size(), WIDTH_15);
    }

    #[test]
    fn test_leaf15_size() {
        // Verify the node is smaller than LeafNode24 (which is ~512 bytes with alignment)
        // LeafNode15 should be ~320-384 bytes depending on padding
        let leaf15_size = std::mem::size_of::<LeafNode15<LeafValue<u64>>>();
        let leaf24_size = std::mem::size_of::<crate::leaf24::LeafNode24<LeafValue<u64>>>();
        eprintln!("LeafNode15 size: {} bytes", leaf15_size);
        eprintln!("LeafNode24 size: {} bytes", leaf24_size);
        assert!(leaf15_size < leaf24_size, "LeafNode15 should be smaller than LeafNode24");
    }
}
