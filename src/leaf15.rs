//! Filepath: src/leaf15.rs
//!
//! Unified leaf node for [`crate::MassTree`] with WIDTH=15 (15 slots).
//!
//! This module provides [`LeafNode15<P>`], a leaf node parameterized over a
//! [`LeafPolicy`] that determines how values and suffixes are stored.
//!
//! # Design
//!
//! The 15-slot design requires 4 bits per slot (values 0-14).
//! Total: 4 (size) + 15×4 (slots) = 64 bits, fitting in u64.
//!
//! # Policies
//!
//! | Policy | Leaf Size | Cache Lines | Use Case |
//! |--------|-----------|-------------|----------|
//! | `BoxPolicy<V>` | 448 bytes | 7 | General-purpose, non-Copy values |
//! | `InlinePolicy<V>` (default) | 1152 bytes | 18 | Embedded suffix, zero-indirection |
//! | `InlinePolicy<V>` + `small-suffix-capacity` | 896 bytes | 14 | Smaller embedded suffix |
//! | `InlinePolicy<V>` + `sidecar-suffix` | 576 bytes | 9 | Compact leaves, pointer-indirect |

use seize::LocalGuard;
use std::array as StdArray;
use std::cmp::Ordering;
use std::fmt as StdFmt;
use std::hint as StdHint;
use std::mem as StdMem;
use std::ptr as StdPtr;
use std::sync::atomic as StdAtomic;
use std::sync::atomic::{AtomicPtr, AtomicU8, AtomicU64, Ordering as AtomicOrdering};

use crate::Linker;
use crate::internode::InternodeNode;
use crate::key::IKEY_SIZE;
use crate::key::Key;
use crate::ksearch::scalar::Scalar;
use crate::leaf_trait::LayerCapableLeaf;
use crate::leaf_trait::SplitInsertData;
use crate::leaf_trait::SplitInsertResult;
use crate::leaf_trait::TreeLeafNode;
use crate::nodeversion::NodeVersion;
use crate::ordering::{CAS_FAILURE, CAS_SUCCESS, READ_ORD, RELAXED, WRITE_ORD};
use crate::permuter::{AtomicPermuter15, Permuter15};
use crate::policy::{LeafPolicy, RetireHandle, SlotKind, SlotState, SuffixStore, ValueArray};
use crate::prefetch::prefetch_read;
use crate::value::InsertTarget;
use crate::value::SplitPoint;
use seize::Guard;

mod layout;

#[cfg(test)]
mod unit_tests;

/// Special keylenx value indicating key has a suffix.
pub const KSUF_KEYLENX: u8 = 64;

/// Base keylenx value indicating a layer pointer (>= this means layer).
pub const LAYER_KEYLENX: u8 = 128;

/// Width constant for [`LeafNode15`].
pub const WIDTH_15: usize = 15;

/// Return value from [`LeafNode15::ksuf_match_result`] indicating an exact match.
pub const MATCH_RESULT_EXACT: i32 = 1;

/// Return value from [`LeafNode15::ksuf_match_result`] indicating same ikey but different key.
pub const MATCH_RESULT_MISMATCH: i32 = 0;

/// Return value from [`LeafNode15::ksuf_match_result`] indicating a layer pointer.
///
/// This is `-IKEY_SIZE` (i.e., `-8`), signaling the caller should descend into
/// the sublayer rather than treating this as a key match or mismatch.
#[expect(
    clippy::cast_possible_wrap,
    clippy::cast_possible_truncation,
    reason = "Known const behavior"
)]
pub const MATCH_RESULT_LAYER: i32 = -(IKEY_SIZE as i32);

/// Modification state: node is in insert mode (normal operation).
pub const MODSTATE_INSERT: u8 = 0;

/// Modification state: node is being removed.
pub const MODSTATE_REMOVE: u8 = 1;

/// Modification state: node's layer has been deleted.
pub const MODSTATE_DELETED_LAYER: u8 = 2;

/// Modification state: node is empty (all keys removed).
/// Empty nodes can be reused by insert or cleaned up by background task.
pub const MODSTATE_EMPTY: u8 = 3;

/// B-link leaf node with 15 slots, parameterized over [`LeafPolicy`].
///
/// `P` controls value storage (Arc vs inline), suffix storage (sidecar vs
/// embedded), and whether value retirement is needed. All structural
/// operations are implemented once regardless of policy.
///
/// # Layout
///
/// `repr(C, align(64))`: cache-line aligned. Fields are ordered to isolate
/// contention — version (CL0) and permutation (CL1) live on separate cache
/// lines. Size: 448B with `BoxPolicy`, 896B with `InlinePolicy`.
#[repr(C, align(64))]
pub struct LeafNode15<P: LeafPolicy> {
    // — Cache Line 0: read-heavy, rarely written —
    /// OCC version word.
    version: NodeVersion,
    /// Lifecycle state: INSERT(0) / REMOVE(1) / `DELETED_LAYER`(2) / EMPTY(3).
    modstate: AtomicU8,
    _pad0: [u8; 55],

    // — Cache Line 1: CAS-heavy, isolated —
    /// Logical→physical slot mapping. CAS is the insert linearization point.
    permutation: AtomicPermuter15,
    _pad1: [u8; 56],

    // — Cache Lines 2+: keys —
    /// First 8 bytes of each key.
    ikey0: [AtomicU64; WIDTH_15],
    /// Key discriminator: 0–8 = inline length, 64 = has suffix, ≥128 = layer.
    keylenx: [AtomicU8; WIDTH_15],

    // — Policy-specific storage —
    /// Terminal values / layer pointers (layout determined by `P::Values`).
    values: P::Values,
    /// Key suffix storage (layout determined by `P::Suffix`).
    suffix: P::Suffix,

    // — Navigation —
    /// Next leaf (LSB = split mark).
    next: AtomicPtr<Self>,
    /// Previous leaf.
    prev: AtomicPtr<Self>,
    /// Parent internode.
    parent: AtomicPtr<u8>,
}

impl<P: LeafPolicy> StdFmt::Debug for LeafNode15<P> {
    fn fmt(&self, f: &mut StdFmt::Formatter<'_>) -> StdFmt::Result {
        f.debug_struct("LeafNode15")
            .field("size", &self.size())
            .field("is_root", &self.version.is_root())
            .field(
                "has_parent",
                // SAFETY: Debug formatting only reads the atomic parent pointer.
                // No dereference of the parent — just a null check.
                &(!unsafe { self.parent_unguarded() }.is_null()),
            )
            .finish_non_exhaustive()
    }
}

impl<P: LeafPolicy> LeafNode15<P> {
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
            modstate: AtomicU8::new(MODSTATE_INSERT),
            _pad0: [0; 55],
            permutation: AtomicPermuter15::new(),
            _pad1: [0; 56],
            ikey0: StdArray::from_fn(|_| AtomicU64::new(0)),
            keylenx: StdArray::from_fn(|_| AtomicU8::new(0)),
            values: P::Values::new(),
            suffix: P::Suffix::new(),
            next: AtomicPtr::new(StdPtr::null_mut()),
            prev: AtomicPtr::new(StdPtr::null_mut()),
            parent: AtomicPtr::new(StdPtr::null_mut()),
        }
    }

    /// Initialize a leaf node directly at the given pointer.
    ///
    /// Avoids stack allocation and copy by writing directly to the destination.
    /// Used by pool allocators for maximum performance.
    ///
    /// # Safety
    ///
    /// - `ptr` must be valid, properly aligned, and point to uninitialized memory
    /// - `ptr` must have space for `size_of::<Self>()` bytes
    #[inline]
    pub unsafe fn init_at(ptr: *mut Self, is_root: bool) {
        // SAFETY: All operations here are safe because:
        // - ptr is valid and properly sized (caller guarantees)
        // - We have exclusive access to the memory
        // - All writes are to properly aligned fields
        unsafe {
            // Zero the entire struct first (most fields are zero-initialized).
            // For BoxPolicy, zero = null pointers (empty slots).
            // For InlinePolicy, zero = null tags (empty slots) + zero bits.
            // For SidecarSuffix, zero = null sidecar pointer (both policies).
            StdPtr::write_bytes(ptr, 0, 1);

            let node: &mut Self = &mut *ptr;

            // Version: leaf node, optionally root
            StdPtr::write(&raw mut node.version, NodeVersion::new(true));
            if is_root {
                node.version.mark_root();
            }

            // ModState: Insert mode (atomic)
            StdPtr::write(&raw mut node.modstate, AtomicU8::new(MODSTATE_INSERT));

            // Permutation: empty
            StdPtr::write(&raw mut node.permutation, AtomicPermuter15::new());

            // NOTE: values and suffix fields are correctly zero-initialized:
            // - BoxValueArray: all AtomicPtr null (zero)
            // - InlineValueArray: all AtomicPtr null + all AtomicU64 zero
            // - SidecarSuffix: AtomicPtr null (zero) — no-op init:
            P::Suffix::init_at_zero(StdPtr::addr_of_mut!((*ptr).suffix));
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

    /// Inherent alias for [`TreeLeafNode::new_boxed`].
    ///
    /// Allows calling `LeafNode15::<P>::new_boxed()` without importing the trait.
    #[inline(always)]
    #[must_use]
    pub fn new_boxed() -> Box<Self> {
        Self::new()
    }

    /// Inherent alias for [`TreeLeafNode::new_root_boxed`].
    ///
    /// Allows calling `LeafNode15::<P>::new_root_boxed()` without importing the trait.
    #[inline(always)]
    #[must_use]
    pub fn new_root_boxed() -> Box<Self> {
        Self::new_root()
    }

    /// Inherent alias for [`TreeLeafNode::new_layer_root_boxed`].
    ///
    /// Allows calling `LeafNode15::<P>::new_layer_root_boxed()` without importing the trait.
    #[inline(always)]
    #[must_use]
    pub fn new_layer_root_boxed() -> Box<Self> {
        Self::new_layer_root()
    }

    /// Node width (number of slots). Matches [`TreeLeafNode::WIDTH`].
    pub const WIDTH: usize = WIDTH_15;

    // ========================================================================
    //  Value Operations (delegate to P::Values)
    // ========================================================================

    /// Load the terminal value at a slot.
    #[inline(always)]
    pub fn load_value(&self, slot: usize) -> Option<P::Output> {
        self.values.load(slot)
    }

    /// Store a terminal value at a slot.
    #[inline(always)]
    pub fn store_value(&self, slot: usize, output: &P::Output) {
        self.values.store(slot, output);
    }

    /// Store a terminal value with Relaxed ordering.
    #[inline(always)]
    pub fn store_value_relaxed(&self, slot: usize, output: &P::Output) {
        self.values.store_relaxed(slot, output);
    }

    /// Update an existing terminal value in place, returning a retire handle.
    #[inline(always)]
    pub fn update_value_in_place(&self, slot: usize, output: &P::Output) -> RetireHandle {
        self.values.update_in_place(slot, output)
    }

    /// Take the terminal value from a slot, leaving it empty.
    #[inline(always)]
    pub fn take_value(&self, slot: usize) -> Option<P::Output> {
        self.values.take(slot)
    }

    /// Load the layer pointer at a slot.
    #[inline(always)]
    pub fn load_layer(&self, slot: usize) -> *mut u8 {
        self.values.load_layer(slot)
    }

    /// Store a layer pointer at a slot.
    ///
    /// Caller must also set `keylenx >= LAYER_KEYLENX`.
    #[inline(always)]
    pub fn store_layer(&self, slot: usize, ptr: *mut u8) {
        self.values.store_layer(slot, ptr);
    }

    /// Take the value at a slot when converting it to a layer pointer.
    #[inline(always)]
    pub fn take_value_for_layer(&self, slot: usize) -> RetireHandle {
        P::take_value_for_layer(&self.values, slot)
    }

    /// Prefetch the value at a slot to hide memory latency.
    ///
    /// For Arc: prefetches the heap allocation behind the pointer.
    /// For inline: no-op (value is already in the leaf's cache lines).
    #[inline(always)]
    pub fn prefetch_value(&self, slot: usize) {
        P::prefetch_value(&self.values, slot);
    }

    /// Load the typed value pointer at a slot.
    ///
    /// Returns a `*const P::Value` pointing to the underlying data.
    /// For Arc: returns the pointer to the heap-allocated `V`.
    /// For inline: not meaningful (compile-time prevented by `RefPolicy` gating).
    ///
    /// # Safety
    ///
    /// Caller must ensure the slot contains a terminal value, not a layer.
    #[inline(always)]
    pub unsafe fn load_value_ptr(&self, slot: usize) -> *const P::Value {
        // SAFETY: Caller guarantees slot contains a terminal value (not a layer).
        // Delegates to the policy's typed load which reads the raw pointer atomically.
        unsafe { P::load_value_ptr(&self.values, slot) }
    }

    /// Check if a slot is empty (no value, no layer).
    #[inline(always)]
    pub fn is_slot_empty(&self, slot: usize) -> bool {
        self.values.is_empty(slot)
    }

    /// Check if a slot has no value stored.
    ///
    /// Alias for [`is_slot_empty`](Self::is_slot_empty). Phase 3 tree code
    /// uses this name for clarity in value-centric contexts.
    #[inline(always)]
    pub fn is_value_empty(&self, slot: usize) -> bool {
        self.is_slot_empty(slot)
    }

    /// Update an existing terminal value in place (alias).
    ///
    /// Alias for [`update_value_in_place`](Self::update_value_in_place).
    /// Phase 3 tree code uses this shorter name.
    #[inline(always)]
    pub fn update_in_place(&self, slot: usize, value: &P::Output) -> RetireHandle {
        self.update_value_in_place(slot, value)
    }

    /// Load the raw layer pointer at a slot.
    ///
    /// Alias for [`load_layer`](Self::load_layer). Phase 3 tree code uses
    /// this name to communicate that the return type is a raw `*mut u8`.
    #[inline(always)]
    pub fn load_layer_raw(&self, slot: usize) -> *mut u8 {
        self.load_layer(slot)
    }

    /// Load the raw value pointer at a slot without interpretation.
    #[inline(always)]
    pub fn load_value_raw(&self, slot: usize) -> *mut u8 {
        self.values.load_raw(slot)
    }

    /// Clear a slot's value and keylenx.
    #[inline(always)]
    pub fn clear_slot(&self, slot: usize) {
        debug_assert!(slot < WIDTH_15, "clear_slot: slot out of bounds");
        self.values.clear(slot);
        self.keylenx[slot].store(0, WRITE_ORD);
    }

    /// Clear a slot's value, keylenx, and permutation entry.
    ///
    /// Atomically removes the slot from the visible permutation and clears
    /// its data. Used during remove operations.
    #[inline(always)]
    pub fn clear_slot_and_permutation(&self, slot: usize) {
        debug_assert!(
            slot < WIDTH_15,
            "clear_slot_and_permutation: slot out of bounds"
        );
        let mut perm: Permuter15 = self.permutation();
        perm.remove_slot(slot);
        self.set_permutation(perm);
        self.clear_slot(slot);
    }

    // ========================================================================
    //  Slot Classification
    // ========================================================================

    /// Classify a slot's contents, extracting the value if present.
    #[inline(always)]
    pub fn classify_slot(&self, slot: usize) -> SlotKind<P::Output> {
        debug_assert!(slot < WIDTH_15, "classify_slot: slot out of bounds");

        let klx: u8 = self.keylenx[slot].load(READ_ORD);

        if klx >= LAYER_KEYLENX {
            // Layer pointer
            let ptr: *mut u8 = self.values.load_layer(slot);
            return SlotKind::Layer(ptr);
        }

        // Terminal value or empty — delegate to value array
        self.values
            .load(slot)
            .map_or_else(|| SlotKind::Empty, SlotKind::Value)
    }

    /// Lightweight slot classification without value extration.
    #[inline(always)]
    pub fn classify_slot_light(&self, slot: usize) -> SlotState {
        debug_assert!(slot < WIDTH_15, "classify_slot_light: slot out of bounds");

        let klx: u8 = self.keylenx[slot].load(READ_ORD);

        if klx >= LAYER_KEYLENX {
            let ptr: *mut u8 = self.values.load_layer(slot);
            return SlotState::Layer(ptr);
        }

        if self.values.is_empty(slot) {
            SlotState::Empty
        } else {
            SlotState::Value
        }
    }

    // ========================================================================
    //  Value Move (for split operations)
    // ========================================================================

    /// Move a slot's value from this leaf to another leaf.
    #[inline(always)]
    pub fn move_value_to(&self, dst: &Self, src_slot: usize, dst_slot: usize) {
        self.values.move_slot(&dst.values, src_slot, dst_slot);
    }

    /// Clear a slot's value storage (without touching keylenx).
    #[inline(always)]
    pub fn clear_value(&self, slot: usize) {
        self.values.clear(slot);
    }

    // ========================================================================
    //  Suffix Read Operations (OCC-safe, no lock required)
    // ========================================================================

    /// Get the suffix bytes for a slot.
    #[must_use]
    #[inline(always)]
    pub fn ksuf(&self, slot: usize) -> Option<&[u8]> {
        debug_assert!(slot < WIDTH_15, "ksuf: slot {slot} out of bounds");

        if !self.has_ksuf(slot) {
            return None;
        }

        self.suffix.get(slot)
    }

    /// Get the suffix bytes for a slot, or an empty slice if none.
    #[must_use]
    #[inline(always)]
    pub fn ksuf_or_empty(&self, slot: usize) -> &[u8] {
        self.ksuf(slot).unwrap_or(&[])
    }

    /// Check if a slot's suffix equals the given bytes.
    #[must_use]
    #[inline(always)]
    pub fn ksuf_equals(&self, slot: usize, suffix: &[u8]) -> bool {
        debug_assert!(slot < WIDTH_15, "ksuf_equals: slot {slot} out of bounds");

        self.ksuf(slot)
            .map_or_else(|| suffix.is_empty(), |stored: &[u8]| stored == suffix)
    }

    /// Compare a slot's suffix with the given bytes.
    #[must_use]
    #[inline(always)]
    pub fn ksuf_compare(&self, slot: usize, suffix: &[u8]) -> Option<Ordering> {
        debug_assert!(slot < WIDTH_15, "ksuf_compare: slot {slot} out of bounds");

        self.ksuf(slot).map(|stored| stored.cmp(suffix))
    }

    /// Check if a slot's full key matches the given ikey + suffix.
    #[must_use]
    #[inline(always)]
    pub fn ksuf_matches(&self, slot: usize, ikey: u64, suffix: &[u8]) -> bool {
        debug_assert!(slot < WIDTH_15, "ksuf_matches: slot {slot} out of bounds");

        if self.ikey(slot) != ikey {
            return false;
        }

        if suffix.is_empty() {
            !self.has_ksuf(slot)
        } else {
            self.ksuf_equals(slot, suffix)
        }
    }

    /// Match a slot against a key suffix, returning a result code.
    ///
    /// Returns:
    /// - `MATCH_RESULT_EXACT` (1): exact key match
    /// - `MATCH_RESULT_MISMATCH` (0): same ikey, different full key
    /// - `MATCH_RESULT_LAYER` (-8): slot is a layer pointer
    #[must_use]
    #[inline(always)]
    pub fn ksuf_match_result(&self, slot: usize, keylenx: u8, suffix: &[u8]) -> i32 {
        debug_assert!(
            slot < WIDTH_15,
            "ksuf_match_result: slot {slot} out of bounds"
        );

        let stored_keylenx: u8 = self.keylenx(slot);

        if Self::keylenx_is_layer(stored_keylenx) {
            return MATCH_RESULT_LAYER;
        }

        if stored_keylenx != KSUF_KEYLENX {
            if stored_keylenx == keylenx && suffix.is_empty() {
                return MATCH_RESULT_EXACT;
            }

            return MATCH_RESULT_MISMATCH;
        }

        if suffix.is_empty() {
            return MATCH_RESULT_MISMATCH;
        }

        i32::from(self.ksuf_equals(slot, suffix))
    }

    // ========================================================================
    //  Suffix Write Operations (lock required)
    // ========================================================================

    /// Assign a suffix to a slot.
    ///
    /// # Safety
    /// - Caller must hold the leaf lock.
    /// - `guard` must come from this tree's collector.
    #[inline(always)]
    pub unsafe fn assign_ksuf(
        &self,
        slot: usize,
        suffix: &[u8],
        guard: &LocalGuard<'_>,
    ) -> *mut u8 {
        debug_assert!(slot < WIDTH_15, "assign_ksuf: slot {slot} out of bounds");
        debug_assert!(
            self.version.is_locked() || self.version.is_unpublished(),
            "assign_ksuf: caller must hold lock or node must be unpublished"
        );

        if suffix.is_empty() {
            return StdPtr::null_mut();
        }

        let perm: Permuter15 = self.permutation();

        // Delegate to suffix store — it handles inline/external/drain logic.
        // SAFETY: Caller holds lock.
        let retire_ptr: *mut u8 = unsafe { self.suffix.assign(slot, suffix, &perm, guard) };

        // CRITICAL: Publish suffix presence AFTER bytes are written.
        // Readers check keylenx with Acquire, which synchronizes with this
        // Release store, ensuring suffix bytes are visible.
        self.keylenx[slot].store(KSUF_KEYLENX, WRITE_ORD);

        retire_ptr
    }

    /// Assign a suffix during node initialization (sequential slots 0..slot).
    ///
    /// Unlike `assign_ksuf`, this does not return a retire pointer because
    /// the suffix store handles retirement internally during init.
    ///
    /// # Safety
    /// Same as `assign_ksuf`.
    #[inline(always)]
    pub unsafe fn assign_ksuf_init(&self, slot: usize, suffix: &[u8], guard: &LocalGuard<'_>) {
        debug_assert!(
            slot < WIDTH_15,
            "assign_ksuf_init: slot {slot} out of bounds"
        );
        debug_assert!(
            self.version.is_locked() || self.version.is_unpublished(),
            "assign_ksuf_init: caller must hold lock or node must be unpublished"
        );

        if suffix.is_empty() {
            return;
        }

        // SAFETY: Caller holds lock.
        unsafe { self.suffix.assign_init(slot, suffix, guard) };

        // Publish suffix presence.
        self.keylenx[slot].store(KSUF_KEYLENX, WRITE_ORD);
    }

    /// Assign a suffix using a pre-allocated buffer.
    ///
    /// # Safety
    /// Same as `assign_ksuf`.
    #[inline(always)]
    pub unsafe fn assign_ksuf_prealloc(
        &self,
        slot: usize,
        suffix: &[u8],
        guard: &LocalGuard<'_>,
        prealloc: Vec<u8>,
    ) -> *mut u8 {
        debug_assert!(
            slot < WIDTH_15,
            "assign_ksuf_prealloc: slot {slot} out of bounds"
        );
        debug_assert!(
            self.version.is_locked() || self.version.is_unpublished(),
            "assign_ksuf_prealloc: caller must hold lock or node must be unpublished"
        );

        if suffix.is_empty() {
            return StdPtr::null_mut();
        }

        let perm: Permuter15 = self.permutation();

        // SAFETY: Caller holds lock.
        let retire_ptr: *mut u8 = unsafe {
            self.suffix
                .assign_prealloc(slot, suffix, &perm, guard, prealloc)
        };

        self.keylenx[slot].store(KSUF_KEYLENX, WRITE_ORD);

        retire_ptr
    }

    /// Retire a suffix bag pointer returned by `assign_ksuf`.
    ///
    /// # Safety
    /// `ptr` must have come from `assign_ksuf` and must not be null.
    #[inline(always)]
    pub unsafe fn retire_suffix_bag_ptr(ptr: *mut u8, guard: &LocalGuard<'_>) {
        // SAFETY: Delegate to suffix store's static retire method.
        unsafe { P::Suffix::retire_bag_ptr(ptr, guard) };
    }

    /// Clear a slot's suffix.
    ///
    /// # Safety
    /// Caller must hold the leaf lock.
    #[inline(always)]
    pub unsafe fn clear_ksuf(&self, slot: usize, guard: &LocalGuard<'_>) {
        debug_assert!(slot < WIDTH_15, "clear_ksuf: slot {slot} out of bounds");
        debug_assert!(
            self.version.is_locked(),
            "clear_ksuf: caller must hold lock"
        );

        // SAFETY: Caller holds lock.
        unsafe { self.suffix.clear(slot, guard) };

        self.keylenx[slot].store(0, WRITE_ORD);
    }

    /// Check if external (overflow) suffix storage has been allocated.
    #[inline(always)]
    pub fn has_external_ksuf(&self) -> bool {
        self.suffix.has_external()
    }

    /// Pre-allocate external overflow storage.
    ///
    /// # Safety
    /// Caller must hold the leaf lock.
    #[inline(always)]
    pub unsafe fn ensure_external_ksuf(&self) -> *mut crate::suffix::SuffixBag {
        // SAFETY: Caller holds lock.
        unsafe { self.suffix.ensure_external() }
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
    /// Uses Acquire ordering to synchronize with writer's Release stores.
    ///
    /// # Panics
    /// Panics in debug mode if `slot >= WIDTH_15`.
    #[must_use]
    #[inline(always)]
    #[expect(
        clippy::indexing_slicing,
        reason = "Slot from Permuter15, valid by construction"
    )]
    pub fn ikey(&self, slot: usize) -> u64 {
        debug_assert!(slot < WIDTH_15, "ikey: slot out of bounds");

        self.ikey0[slot].load(READ_ORD)
    }

    /// Get the ikey at the given physical slot using Relaxed ordering.
    ///
    /// # Safety Justification
    /// Safe to use Relaxed when:
    /// 1. Caller has already loaded permutation with Acquire ordering, which
    ///    synchronizes with the writer's Release fence after modifications
    /// 2. OCC version validation at the end of the read catches any races
    ///
    /// This avoids redundant Acquire fences on each ikey load (up to 15 per search),
    /// improving read throughput by 10-15%.
    ///
    /// # Panics
    /// Panics in debug mode if `slot >= WIDTH_15`.
    #[must_use]
    #[inline(always)]
    #[expect(
        clippy::indexing_slicing,
        reason = "Slot from Permuter15, valid by construction"
    )]
    pub fn ikey_relaxed(&self, slot: usize) -> u64 {
        debug_assert!(slot < WIDTH_15, "ikey_relaxed: slot out of bounds");

        self.ikey0[slot].load(RELAXED)
    }

    /// Set the ikey at the given physical slot.
    #[inline(always)]
    #[expect(
        clippy::indexing_slicing,
        reason = "Slot from Permuter15, valid by construction"
    )]
    pub fn set_ikey(&self, slot: usize, ikey: u64) {
        debug_assert!(slot < WIDTH_15, "set_ikey: slot out of bounds");

        self.ikey0[slot].store(ikey, WRITE_ORD);
    }

    /// Set the ikey at the given physical slot using Relaxed ordering.
    ///
    /// # Safety Preconditions
    /// Only safe when writing to a node that is:
    /// - Not yet visible to other threads (during split setup), AND
    /// - Protected by version lock on the source node
    #[inline(always)]
    #[expect(
        clippy::indexing_slicing,
        reason = "Slot from Permuter15, valid by construction"
    )]
    pub(crate) fn set_ikey_relaxed(&self, slot: usize, ikey: u64) {
        debug_assert!(slot < WIDTH_15, "set_ikey_relaxed: slot out of bounds");

        self.ikey0[slot].store(ikey, RELAXED);
    }

    /// Prefetch the ikey at the given slot into CPU cache.
    ///
    /// # Safety
    /// The slot must be in range `[0, WIDTH_15)`. No bounds check in release mode.
    #[inline(always)]
    #[expect(clippy::indexing_slicing, reason = "Caller ensures slot is valid")]
    pub fn prefetch_ikey(&self, slot: usize) {
        debug_assert!(slot < WIDTH_15, "prefetch_ikey: slot out of bounds");
        prefetch_read(&raw const self.ikey0[slot]);
    }

    /// Prefetch leaf node data for range scans.
    #[inline(always)]
    pub fn prefetch(&self) {
        let self_ptr: *const u8 = StdPtr::from_ref::<Self>(self).cast::<u8>();
        // SAFETY: self_ptr is derived from a valid reference, offsets are within struct bounds.
        unsafe {
            prefetch_read(self_ptr.add(StdMem::offset_of!(Self, ikey0)));
            prefetch_read(self_ptr.add(StdMem::offset_of!(Self, ikey0) + 64));
            prefetch_read(self_ptr.add(StdMem::offset_of!(Self, values)));
            prefetch_read(self_ptr.add(StdMem::offset_of!(Self, values) + 64));
        }
    }

    /// Prefetch for point lookup (permutation + keys only).
    #[inline(always)]
    pub fn prefetch_for_search(&self) {
        let self_ptr: *const u8 = StdPtr::from_ref::<Self>(self).cast::<u8>();

        // SAFETY: self_ptr is derived from a valid reference. Offsets target
        // permutation and ikey0 fields, which are within struct bounds.
        unsafe {
            prefetch_read(self_ptr.add(StdMem::offset_of!(Self, permutation)));
            prefetch_read(self_ptr.add(StdMem::offset_of!(Self, ikey0)));
            prefetch_read(self_ptr.add(StdMem::offset_of!(Self, ikey0) + 64));
        }
    }

    /// Size-aware prefetch: only fetch cache lines that will be accessed.
    #[inline(always)]
    pub fn prefetch_for_search_adaptive(&self, size: usize) {
        let self_ptr: *const u8 = StdPtr::from_ref(self).cast::<u8>();

        // SAFETY: self_ptr is derived from a valid reference. Offsets target
        // permutation and ikey0 fields, which are within struct bounds.
        // Conditional prefetch avoids fetching cache lines for empty regions.
        unsafe {
            prefetch_read(self_ptr.add(StdMem::offset_of!(Self, permutation)));

            if size > 0 {
                prefetch_read(self_ptr.add(StdMem::offset_of!(Self, ikey0)));
            }

            if size > 8 {
                prefetch_read(self_ptr.add(StdMem::offset_of!(Self, ikey0) + 64));
            }
        }
    }

    /// Get the keylenx at the given physical slot.
    #[must_use]
    #[inline(always)]
    #[expect(
        clippy::indexing_slicing,
        reason = "Slot from Permuter15, valid by construction"
    )]
    pub fn keylenx(&self, slot: usize) -> u8 {
        debug_assert!(slot < WIDTH_15, "keylenx: slot out of bounds");

        self.keylenx[slot].load(READ_ORD)
    }

    /// Set the keylenx at the given physical slot.
    #[inline(always)]
    #[expect(
        clippy::indexing_slicing,
        reason = "Slot from Permuter15, valid by construction"
    )]
    pub fn set_keylenx(&self, slot: usize, keylenx: u8) {
        debug_assert!(slot < WIDTH_15, "set_keylenx: slot out of bounds");

        self.keylenx[slot].store(keylenx, WRITE_ORD);
    }

    /// Set the keylenx at the given physical slot using Relaxed ordering.
    #[inline(always)]
    #[expect(
        clippy::indexing_slicing,
        reason = "Slot from Permuter15, valid by construction"
    )]
    pub(crate) fn set_keylenx_relaxed(&self, slot: usize, keylenx: u8) {
        debug_assert!(slot < WIDTH_15, "set_keylenx_relaxed: slot out of bounds");

        self.keylenx[slot].store(keylenx, RELAXED);
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
        let perm: Permuter15 = self.permutation();

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

    /// Search all 15 ikeys for matches with target, returning a bitmask.
    #[must_use]
    #[inline(always)]
    pub fn find_ikey_matches(&self, target_ikey: u64) -> u32 {
        Scalar::find_ikey_matches_leaf15(target_ikey, self)
    }

    // ============================================================================
    //  Permutation Accessors
    // ============================================================================

    /// Load permutation with Acquire ordering.
    #[inline(always)]
    #[must_use]
    pub fn permutation(&self) -> Permuter15 {
        self.permutation.load(READ_ORD)
    }

    /// Store permutation with Release ordering.
    #[inline(always)]
    pub fn set_permutation(&self, perm: Permuter15) {
        self.permutation.store(perm, WRITE_ORD);
    }

    /// Store permutation with Relaxed ordering.
    ///
    /// # Safety
    /// Caller must hold the lock on this node. Using Relaxed without the
    /// lock would create a data race with concurrent readers.
    #[inline(always)]
    pub fn set_permutation_relaxed(&self, perm: Permuter15) {
        self.permutation.store(perm, RELAXED);
    }

    /// Get raw permutation value (for debugging).
    #[inline(always)]
    #[must_use]
    pub fn permutation_raw(&self) -> u64 {
        self.permutation.load_raw(READ_ORD)
    }

    /// Store key metadata (`ikey`, `keylenx`) for a CAS insert attempt.
    ///
    /// # Safety
    /// - The caller must have successfully claimed the slot via `cas_slot_value` and ensured
    ///   the slot still belongs to the CAS attempt (i.e. `leaf_values[slot]` still equals the
    ///   claimed pointer).
    ///
    /// NOTE: writing key metadata *before* claiming the slot is not safe in this design because
    /// multiple concurrent CAS attempts can overwrite each other's metadata before publish.
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
        self.size() >= WIDTH_15
    }

    // ============================================================================
    //  Leaf Linking
    // ============================================================================

    /// Get the next leaf pointer, masking the mark bit.
    #[must_use]
    #[inline(always)]
    pub fn safe_next(&self, guard: &impl Guard) -> *mut Self {
        let ptr: *mut Self = guard.protect(&self.next, READ_ORD);
        ptr.map_addr(|addr: usize| addr & !1)
    }

    /// Get the next leaf pointer without guard protection.
    ///
    /// # Safety
    /// Caller must ensure the next pointer's target won't be retired during use.
    #[must_use]
    #[inline(always)]
    pub unsafe fn safe_next_unguarded(&self) -> *mut Self {
        let ptr: *mut Self = self.next.load(READ_ORD);
        ptr.map_addr(|addr: usize| addr & !1)
    }

    /// Get the raw next pointer (including mark bit).
    ///
    /// Uses guard protection to ensure the load participates in seize's
    /// total order, making it safe on all architectures.
    #[must_use]
    #[inline(always)]
    pub fn next_raw(&self, guard: &impl Guard) -> *mut Self {
        guard.protect(&self.next, READ_ORD)
    }

    /// Get the raw next pointer without guard protection.
    ///
    /// # Safety
    /// Caller must ensure the next pointer's target won't be retired during use.
    #[must_use]
    #[inline(always)]
    pub unsafe fn next_raw_unguarded(&self) -> *mut Self {
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

    /// Set the next leaf pointer using Relaxed ordering.
    ///
    /// # Safety Preconditions
    /// Only safe when writing to a node that is not yet visible to other threads
    /// The subsequent `set_next(Release)` in `link_sibling` ensures visibility.
    #[inline(always)]
    pub(crate) fn set_next_relaxed(&self, next: *mut Self) {
        self.next.store(next, RELAXED);
    }

    /// Mark the next pointer (during split).
    #[inline(always)]
    pub fn mark_next(&self) {
        let ptr: *mut Self = self.next.load(RELAXED);
        let marked: *mut Self = ptr.map_addr(|addr: usize| addr | 1);
        self.next.store(marked, WRITE_ORD);
    }

    /// Unmark the next pointer.
    ///
    /// # Safety Note
    /// Uses unguarded load internally since we're modifying our own field
    /// during a locked operation, not traversing to a different node.
    #[inline(always)]
    pub fn unmark_next(&self) {
        // SAFETY: We hold the lock on this node during split/unlink operations,
        // so we're not racing with retirement of the next pointer's target.
        let ptr: *mut Self = unsafe { self.safe_next_unguarded() };
        self.next.store(ptr, WRITE_ORD);
    }

    /// Wait for an in-progress split to complete.
    ///
    /// Spins until the next pointer is unmarked, the version is stable,
    /// OR the node is marked as deleted.
    ///
    /// NOTE: A marked next pointer can mean either:
    /// 1. A split is in progress (will be unmarked when split completes)
    /// 2. An unlink is in progress (leaf being deleted, may stay marked)
    ///
    /// We check `is_deleted()` to avoid spinning forever on case 2.
    pub fn wait_for_split(&self) {
        while self.next_is_marked() {
            // Check if node was deleted (unlink marks next but never unmarks)
            if self.version.is_deleted() {
                return;
            }

            // Quick check: did marker clear during spin?
            for _ in 0..16 {
                std::hint::spin_loop();
                if !self.next_is_marked() {
                    return;
                }
            }

            // Still marked - wait for version to stabilize
            let _ = self.version.stable();

            // Re-check deletion after waiting for version
            if self.version.is_deleted() {
                return;
            }
        }
    }

    /// CAS-based lock of the next pointer for split linking.
    fn lock_next(&self) -> *mut Self {
        loop {
            let next: *mut Self = self.next.load(READ_ORD);

            // Already marked: another split is in progress, wait
            if Linker::is_marked(next) {
                self.wait_for_split();
                continue;
            }

            // Try to mark the pointer via CAS
            let marked: *mut Self = Linker::mark_ptr(next);
            match self
                .next
                .compare_exchange(next, marked, CAS_SUCCESS, CAS_FAILURE)
            {
                Ok(_) => {
                    return next;
                }

                Err(_) => {
                    // CAS failed: someone else updated next, retry immediately.
                    StdHint::spin_loop();
                }
            }
        }
    }

    /// Unlink this leaf from the B-link doubly-linked chain.
    ///
    /// # Preconditions
    ///
    /// - Self is locked (caller holds version lock)
    /// - Self has a predecessor (`prev` is non-null)
    ///
    /// # Safety
    /// - Caller must hold the version lock on this leaf
    /// - `self.prev()` must be non-null (not the leftmost leaf)
    /// - The prev and next pointers must be valid leaves
    pub unsafe fn unlink_from_chain(&self) {
        // Step 1: Lock our next pointer (mark it)
        // This prevents concurrent splits from interfering
        let next: *mut Self = self.lock_next();

        // Step 2: CAS prev->next from self to marked(self)
        // This signals to prev that we're unlinking
        let self_ptr: *mut Self = StdPtr::from_ref(self).cast_mut();
        let marked_self: *mut Self = Linker::mark_ptr(self_ptr);

        let final_prev: *mut Self;
        loop {
            // Re-read prev on each iteration (may change if prev splits)
            // SAFETY: Called under exclusive lock - no concurrent retirement.
            let prev: *mut Self = unsafe { self.prev_unguarded() };
            debug_assert!(!prev.is_null(), "unlink_from_chain: prev must be non-null");

            // SAFETY: prev is non-null (checked above) and points to a valid leaf
            // Try to mark prev's next pointer (from self to marked(self))
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
                    // If prev->next is already marked, wait for it to clear
                    // This can happen if prev is splitting
                    if Linker::is_marked(current) {
                        // SAFETY: prev is valid
                        unsafe { (*prev).wait_for_split() };
                    }

                    StdHint::spin_loop();
                }
            }
        }

        // Step 3: Update next->prev to skip over us
        if !next.is_null() {
            // SAFETY: next is non-null (just checked) and points to a valid leaf
            unsafe { (*next).set_prev(final_prev) };
        }

        // Step 4: Release fence for visibility
        StdAtomic::fence(AtomicOrdering::Release);

        // Step 5: Complete unlinking by storing unmarked next into prev->next
        // SAFETY: final_prev is non-null and points to a valid leaf
        unsafe { (*final_prev).set_next(next) };
    }

    /// Get the previous leaf pointer.
    #[must_use]
    #[inline(always)]
    pub fn prev(&self, guard: &impl Guard) -> *mut Self {
        guard.protect(&self.prev, READ_ORD)
    }

    /// Get the previous leaf pointer without guard protection.
    ///
    /// # Safety
    /// Caller must ensure the prev pointer's target won't be retired during use.
    #[must_use]
    #[inline(always)]
    pub unsafe fn prev_unguarded(&self) -> *mut Self {
        self.prev.load(READ_ORD)
    }

    /// Set the previous leaf pointer.
    #[inline(always)]
    pub fn set_prev(&self, prev: *mut Self) {
        self.prev.store(prev, WRITE_ORD);
    }

    /// Set the previous leaf pointer using Relaxed ordering.
    ///
    /// # Safety Preconditions
    /// Only safe when writing to a node that is not yet visible to other threads
    /// The subsequent `set_next(Release)` in `link_sibling` ensures visibility.
    #[inline(always)]
    pub(crate) fn set_prev_relaxed(&self, prev: *mut Self) {
        self.prev.store(prev, RELAXED);
    }

    // ============================================================================
    //  Parent Accessors
    // ============================================================================

    /// Get the parent pointer.
    ///
    /// Uses guard protection to ensure the load participates in seize's
    /// total order, making it safe on all architectures.
    #[must_use]
    #[inline(always)]
    pub fn parent(&self, guard: &impl Guard) -> *mut u8 {
        guard.protect(&self.parent, READ_ORD)
    }

    /// Get the parent pointer without guard protection.
    ///
    /// # Safety
    /// Caller must ensure the parent pointer's target won't be retired during use.
    #[must_use]
    #[inline(always)]
    pub unsafe fn parent_unguarded(&self) -> *mut u8 {
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
    ///
    /// Returns one of:
    /// - `MODSTATE_INSERT` (0): Normal insert mode
    /// - `MODSTATE_REMOVE` (1): Node is being removed
    /// - `MODSTATE_DELETED_LAYER` (2): Layer has been garbage collected
    #[must_use]
    #[inline(always)]
    pub fn modstate(&self) -> u8 {
        self.modstate.load(AtomicOrdering::Acquire)
    }

    /// Set the modification state.
    #[inline(always)]
    pub fn set_modstate(&self, state: u8) {
        self.modstate.store(state, AtomicOrdering::Release);
    }

    /// Check if this layer has been deleted (garbage collected).
    #[must_use]
    #[inline(always)]
    pub fn deleted_layer(&self) -> bool {
        self.modstate() == MODSTATE_DELETED_LAYER
    }

    /// Mark this layer as deleted (for `gc_layer`).
    #[inline(always)]
    pub fn mark_deleted_layer(&self) {
        self.set_modstate(MODSTATE_DELETED_LAYER);
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

    // ============================================================================
    //  Empty State (for lazy coalescing)
    // ============================================================================

    /// Check if this leaf is in empty state (modstate == `MODSTATE_EMPTY`).
    #[must_use]
    #[inline(always)]
    pub fn is_empty_state(&self) -> bool {
        self.modstate() == MODSTATE_EMPTY
    }

    /// Mark this leaf as empty (all keys removed).
    #[inline(always)]
    pub fn mark_empty(&self) {
        self.set_modstate(MODSTATE_EMPTY);
    }

    /// Clear empty state, returning to normal insert mode.
    #[inline(always)]
    pub fn clear_empty_state(&self) {
        self.set_modstate(MODSTATE_INSERT);
    }

    // ============================================================================
    //  Slot Assignment
    // ============================================================================

    /// Check if slot 0 can be reused for a new key.
    ///
    /// # Safety
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
    //  Split Operations
    // ========================================================================

    /// Calculate the optimal split point for this leaf.
    ///
    /// Includes sequential access optimization (forward and reverse)
    /// and equal-ikey adjustment to maintain routing invariants.
    pub fn calculate_split_point(&self, insert_pos: usize, insert_ikey: u64) -> Option<SplitPoint> {
        let perm = self.permutation();
        let size = perm.size();

        if size == 0 {
            return None;
        }

        // Check for forward-sequential pattern: appending to rightmost leaf.
        // SAFETY: Called during insert with lock held, unguarded access is safe.
        //
        // Guard: skip this optimization when the last existing entry shares
        // insert_ikey. A forward-sequential split would set split_ikey = insert_ikey,
        // routing lookups for that ikey to the right leaf while the matching
        // entry remains stranded on the left. Fall through to midpoint split
        // which places both entries on the same side.
        if insert_pos == size && unsafe { self.next_raw_unguarded() }.is_null() {
            let last_slot = perm.get(size - 1);
            if self.ikey(last_slot) != insert_ikey {
                return Some(SplitPoint {
                    pos: size,
                    split_ikey: insert_ikey,
                });
            }
        }

        // Check for reverse-sequential pattern: prepending to leftmost leaf.
        // SAFETY: Called during insert with lock held, unguarded access is safe.
        //
        // Guard: skip when split_ikey would equal insert_ikey (same stranding issue).
        if insert_pos == 0 && unsafe { self.prev_unguarded() }.is_null() && size > 1 {
            let split_slot = perm.get(1);
            let split_ikey = self.ikey(split_slot);
            if split_ikey != insert_ikey {
                return Some(SplitPoint { pos: 1, split_ikey });
            }
        }

        // Default: midpoint split
        let mut split_pos: usize = size / 2;

        if split_pos == 0 {
            split_pos = 1;
        }

        // Equal-ikey adjustment: keep equal ikeys together in the same leaf.
        while split_pos > 0 && split_pos < size {
            let left_slot = perm.get(split_pos - 1);
            let right_slot = perm.get(split_pos);
            let left_ikey = self.ikey(left_slot);
            let right_ikey = self.ikey(right_slot);

            if left_ikey == right_ikey {
                match insert_ikey.cmp(&left_ikey) {
                    Ordering::Equal => {
                        split_pos += 1;
                    }
                    Ordering::Less | Ordering::Greater => {
                        split_pos -= 1;
                    }
                }
            } else {
                break;
            }
        }

        // Post-adjustment guard: when insert_pos == split_pos, the new entry
        // lands at slot 0 of the right leaf, making split_ikey = insert_ikey.
        // If the last left entry also has ikey == insert_ikey, lookups for that
        // ikey route right while the existing entry is stranded on the left.
        //
        // Fix: move the boundary left so all entries with insert_ikey end up
        // on the right side. Left entries then have strictly smaller ikeys.
        if insert_pos == split_pos && split_pos > 0 {
            let last_left_slot = perm.get(split_pos - 1);
            if self.ikey(last_left_slot) == insert_ikey {
                split_pos -= 1;
                while split_pos > 0 {
                    let check_slot = perm.get(split_pos - 1);
                    if self.ikey(check_slot) != insert_ikey {
                        break;
                    }
                    split_pos -= 1;
                }
            }
        }

        if split_pos == 0 || split_pos >= size {
            return None;
        }

        let split_slot = perm.get(split_pos);
        let split_ikey = self.ikey(split_slot);

        Some(SplitPoint {
            pos: split_pos,
            split_ikey,
        })
    }

    /// Split entries from position `split_pos..size` into a new leaf.
    ///
    /// Returns the split key (first ikey of right leaf) and which leaf
    /// should receive a subsequent insert.
    ///
    /// # Safety
    ///
    /// - Caller holds the lock on `self`.
    /// - `new_leaf_ptr` is valid, properly aligned, and points to an
    ///   initialized (via `new_with_root(false)` or `init_at`) but empty leaf.
    /// - `new_leaf_ptr` is not yet visible to other threads.
    pub unsafe fn split_into_preallocated(
        &self,
        split_pos: usize,
        new_leaf_ptr: *mut Self,
        guard: &LocalGuard<'_>,
    ) -> (u64, InsertTarget) {
        let split_version = NodeVersion::new_for_split(&self.version);

        // SAFETY: new_leaf is not yet visible to other threads.
        unsafe {
            StdPtr::write(
                StdPtr::addr_of!((*new_leaf_ptr).version).cast_mut(),
                split_version,
            );
        }

        let perm: Permuter15 = self.permutation();
        let size: usize = perm.size();
        let entries_to_move: usize = size - split_pos;

        // SAFETY: new_leaf_ptr is valid (caller guarantees).
        let new_leaf: &Self = unsafe { &*new_leaf_ptr };

        // Pre-allocate external suffix storage if source has it.
        if self.has_external_ksuf() {
            let _ = unsafe { new_leaf.ensure_external_ksuf() };
        }

        // Move entries from self[split_pos..size] to new_leaf[0..entries_to_move].
        for i in 0..entries_to_move {
            let old_logical: usize = split_pos + i;
            let old_slot: usize = perm.get(old_logical);
            let new_slot: usize = i;

            let ikey: u64 = self.ikey(old_slot);
            let keylenx_val: u8 = self.keylenx(old_slot);
            new_leaf.set_ikey(new_slot, ikey);
            new_leaf.set_keylenx(new_slot, keylenx_val);

            // Move value: ONE call, no if/else for storage mode.
            self.values.move_slot(&new_leaf.values, old_slot, new_slot);

            // Copy suffix if present.
            if keylenx_val == KSUF_KEYLENX {
                if let Some(suffix) = self.ksuf(old_slot) {
                    // SAFETY: We hold lock on both leaves.
                    unsafe { new_leaf.assign_ksuf_init(new_slot, suffix, guard) };
                }
                // SAFETY: We hold lock.
                unsafe { self.clear_ksuf(old_slot, guard) };
            }

            // Clear source slot value (required after move_slot).
            self.values.clear(old_slot);
        }

        // Update old leaf: truncate permutation.
        let mut old_perm = perm;
        old_perm.set_size(split_pos);
        self.set_permutation(old_perm);

        // Update new leaf: sequential permutation for moved entries.
        let new_perm: Permuter15 = Permuter15::make_sorted(entries_to_move);
        new_leaf.set_permutation_relaxed(new_perm);

        let split_ikey: u64 = new_leaf.ikey(0);
        (split_ikey, InsertTarget::Left)
    }

    /// Split ALL entries to the right leaf (used for reverse-sequential pattern).
    ///
    /// # Safety
    ///
    /// Same as `split_into_preallocated`.
    pub unsafe fn split_all_to_right_preallocated(
        &self,
        new_leaf_ptr: *mut Self,
        guard: &LocalGuard<'_>,
    ) -> (u64, InsertTarget) {
        let split_version = NodeVersion::new_for_split(&self.version);
        // SAFETY: new_leaf_ptr is not yet visible to other threads (caller guarantee).
        // Writing the version field initializes the new leaf before publication.
        unsafe {
            StdPtr::write(
                StdPtr::addr_of!((*new_leaf_ptr).version).cast_mut(),
                split_version,
            );
        }

        let perm: Permuter15 = self.permutation();
        let size: usize = perm.size();
        // SAFETY: new_leaf_ptr is valid and properly aligned (caller guarantee).
        let new_leaf: &Self = unsafe { &*new_leaf_ptr };

        if self.has_external_ksuf() {
            // SAFETY: new_leaf is not yet visible; pre-allocating suffix storage
            // before entries are moved ensures suffix writes won't fail.
            let _ = unsafe { new_leaf.ensure_external_ksuf() };
        }

        for i in 0..size {
            let old_slot: usize = perm.get(i);
            let new_slot: usize = i;

            let ikey: u64 = self.ikey(old_slot);
            let keylenx_val: u8 = self.keylenx(old_slot);
            new_leaf.set_ikey(new_slot, ikey);
            new_leaf.set_keylenx(new_slot, keylenx_val);

            self.values.move_slot(&new_leaf.values, old_slot, new_slot);

            if keylenx_val == KSUF_KEYLENX {
                if let Some(suffix) = self.ksuf(old_slot) {
                    // SAFETY: We hold lock on self; new_leaf is not yet visible.
                    // Suffix data is copied to new_leaf's storage.
                    unsafe { new_leaf.assign_ksuf_init(new_slot, suffix, guard) };
                }
                // SAFETY: We hold lock on self. Clearing old suffix after copy
                // retires the old bag pointer via the guard for deferred reclamation.
                unsafe { self.clear_ksuf(old_slot, guard) };
            }

            self.values.clear(old_slot);
        }

        self.set_permutation(Permuter15::empty());

        let new_perm: Permuter15 = Permuter15::make_sorted(size);
        new_leaf.set_permutation_relaxed(new_perm);

        let split_ikey: u64 = new_leaf.ikey(0);
        (split_ikey, InsertTarget::Left)
    }

    /// Atomic split + insert: split entries and insert a new key in one operation.
    ///
    /// # Safety
    ///
    /// Same as `split_into_preallocated`.
    #[expect(clippy::too_many_lines)]
    pub unsafe fn split_and_insert(
        &self,
        split_pos: usize,
        new_leaf_ptr: *mut Self,
        insert_pos: usize,
        insert_data: &SplitInsertData<'_, P>,
        guard: &LocalGuard<'_>,
    ) -> SplitInsertResult {
        let split_version = NodeVersion::new_for_split(&self.version);
        // SAFETY: new_leaf_ptr is not yet visible to other threads (caller guarantee).
        // Writing the version field initializes the new leaf before publication.
        unsafe {
            StdPtr::write(
                StdPtr::addr_of!((*new_leaf_ptr).version).cast_mut(),
                split_version,
            );
        }

        let old_perm: Permuter15 = self.permutation();
        let old_size: usize = old_perm.size();
        // SAFETY: new_leaf_ptr is valid and properly aligned (caller guarantee).
        let new_leaf: &Self = unsafe { &*new_leaf_ptr };

        if self.has_external_ksuf() {
            // SAFETY: new_leaf is not yet visible; pre-allocating suffix storage
            // before entries are moved ensures suffix writes won't fail.
            let _ = unsafe { new_leaf.ensure_external_ksuf() };
        }

        let insert_goes_right: bool = insert_pos >= split_pos;
        let entries_to_move: usize = old_size - split_pos;
        let split_ikey: u64;

        if insert_goes_right {
            // =================================================================
            // Case: Insert goes to RIGHT leaf (insert_pos >= split_pos)
            // =================================================================
            let right_insert_pos: usize = insert_pos - split_pos;
            let right_size: usize = entries_to_move + 1;

            let mut src_idx: usize = 0;

            for dst_slot in 0..right_size {
                if dst_slot == right_insert_pos {
                    new_leaf.set_ikey(dst_slot, insert_data.ikey);
                    new_leaf.set_keylenx(dst_slot, insert_data.keylenx);
                    new_leaf.store_value_relaxed(dst_slot, &insert_data.value);

                    if insert_data.keylenx == KSUF_KEYLENX
                        && let Some(suffix) = insert_data.suffix
                    {
                        // SAFETY: new_leaf is not yet visible. Initializing suffix
                        // storage for the newly inserted entry.
                        unsafe {
                            new_leaf.assign_ksuf_init(dst_slot, suffix, guard);
                        };
                    }
                } else {
                    let old_logical_pos: usize = split_pos + src_idx;
                    let old_slot: usize = old_perm.get(old_logical_pos);

                    let ikey: u64 = self.ikey(old_slot);
                    let keylenx_val: u8 = self.keylenx(old_slot);

                    new_leaf.set_ikey(dst_slot, ikey);
                    new_leaf.set_keylenx(dst_slot, keylenx_val);

                    self.values.move_slot(&new_leaf.values, old_slot, dst_slot);

                    if keylenx_val == KSUF_KEYLENX {
                        if let Some(suffix) = self.ksuf(old_slot) {
                            // SAFETY: We hold lock on self; new_leaf is not yet visible.
                            // Suffix data is copied to new_leaf's storage.
                            unsafe {
                                new_leaf.assign_ksuf_init(dst_slot, suffix, guard);
                            };
                        }

                        // SAFETY: We hold lock on self. Old suffix cleared after copy;
                        // old bag pointer retired via guard for deferred reclamation.
                        unsafe { self.clear_ksuf(old_slot, guard) };
                    }

                    self.values.clear(old_slot);
                    src_idx += 1;
                }
            }

            let mut left_perm = old_perm;
            left_perm.set_size(split_pos);
            self.set_permutation(left_perm);

            let right_perm: Permuter15 = Permuter15::make_sorted(right_size);
            new_leaf.set_permutation_relaxed(right_perm);

            split_ikey = new_leaf.ikey(0);

            SplitInsertResult {
                split_ikey,
                insert_target: InsertTarget::Right,
            }
        } else {
            // =================================================================
            // Case: Insert goes to LEFT leaf (insert_pos < split_pos)
            // =================================================================

            // Move right portion first.
            for i in 0..entries_to_move {
                let old_logical: usize = split_pos + i;
                let old_slot: usize = old_perm.get(old_logical);
                let new_slot: usize = i;

                let ikey: u64 = self.ikey(old_slot);
                let keylenx_val: u8 = self.keylenx(old_slot);

                new_leaf.set_ikey(new_slot, ikey);
                new_leaf.set_keylenx(new_slot, keylenx_val);

                self.values.move_slot(&new_leaf.values, old_slot, new_slot);

                if keylenx_val == KSUF_KEYLENX {
                    if let Some(suffix) = self.ksuf(old_slot) {
                        // SAFETY: We hold lock on self; new_leaf is not yet visible.
                        // Suffix data is copied to new_leaf's storage.
                        unsafe {
                            new_leaf.assign_ksuf_init(new_slot, suffix, guard);
                        };
                    }

                    // SAFETY: We hold lock on self. Old suffix cleared after copy;
                    // old bag pointer retired via guard for deferred reclamation.
                    unsafe { self.clear_ksuf(old_slot, guard) };
                }

                self.values.clear(old_slot);
            }

            // Find a free physical slot from the back of the updated permutation.
            let mut left_perm = old_perm;
            left_perm.set_size(split_pos);

            let new_slot: usize = left_perm.back();

            // Check slot-0 reuse rule.
            let actual_slot: usize = if new_slot == 0 && !self.can_reuse_slot0(insert_data.ikey) {
                let free_count: usize = WIDTH_15 - split_pos;
                let mut found_slot: usize = new_slot;

                for offset in 1..free_count {
                    let candidate: usize = left_perm.back_at_offset(offset);

                    if candidate != 0 {
                        left_perm.swap_free_slots(WIDTH_15 - 1, WIDTH_15 - 1 - offset);
                        found_slot = candidate;
                        break;
                    }
                }

                found_slot
            } else {
                new_slot
            };

            self.set_ikey(actual_slot, insert_data.ikey);
            self.set_keylenx(actual_slot, insert_data.keylenx);
            self.store_value(actual_slot, &insert_data.value);

            let suffix_retire_ptr: *mut u8 = if insert_data.keylenx == KSUF_KEYLENX
                && let Some(suffix) = insert_data.suffix
            {
                // SAFETY: We hold lock on self. Assigns suffix to the newly
                // inserted slot; returns old bag pointer for deferred retirement.
                unsafe { self.assign_ksuf(actual_slot, suffix, guard) }
            } else {
                StdPtr::null_mut()
            };

            let allocated: usize = left_perm.insert_from_back(insert_pos);
            debug_assert_eq!(allocated, actual_slot, "allocated unexpected slot");
            self.set_permutation(left_perm);

            let right_perm: Permuter15 = Permuter15::make_sorted(entries_to_move);
            new_leaf.set_permutation_relaxed(right_perm);

            split_ikey = new_leaf.ikey(0);

            // Retire old suffix bag if needed.
            if !suffix_retire_ptr.is_null() {
                // SAFETY: suffix_retire_ptr was returned by assign_ksuf and points
                // to a heap-allocated suffix bag. Guard defers reclamation until safe.
                unsafe { Self::retire_suffix_bag_ptr(suffix_retire_ptr, guard) };
            }

            SplitInsertResult {
                split_ikey,
                insert_target: InsertTarget::Left,
            }
        }
    }

    /// Link a new sibling after this leaf during a split.
    ///
    /// # Safety
    /// - Caller must hold the lock on `self`.
    /// - `new_sibling` must be valid and not yet in the chain.
    pub unsafe fn link_sibling(&self, new_sibling: *mut Self) {
        // Lock our next pointer (mark it to signal split in progress).
        let old_next: *mut Self = self.lock_next();

        // SAFETY: new_sibling is valid.
        let new_sib: &Self = unsafe { &*new_sibling };

        // Set new sibling's prev/next.
        new_sib.set_prev_relaxed(StdPtr::from_ref(self).cast_mut());
        new_sib.set_next_relaxed(old_next);

        // Update old_next's prev to point to new sibling.
        if !old_next.is_null() {
            // SAFETY: old_next is non-null (just checked) and points to a valid
            // leaf in the chain. We hold the lock on self which protects the
            // next pointer, preventing concurrent modification of this link.
            unsafe { (*old_next).set_prev(new_sibling) };
        }

        // Publish: store new_sibling as our next (Release store).
        // This simultaneously unmarks the pointer and publishes the new sibling.
        self.set_next(new_sibling);
    }
}

// ============================================================================
//  Send + Sync
// ============================================================================

// SAFETY: LeafNode15 is safe to send/share between threads.
// The atomic fields handle concurrent access, and the raw pointers are
// protected by the tree's concurrency protocol (version validation, locks).
// P::Values and P::Suffix implement Send+Sync via their own unsafe impls.
unsafe impl<P: LeafPolicy> Send for LeafNode15<P> {}
unsafe impl<P: LeafPolicy> Sync for LeafNode15<P> {}
// ============================================================================
//  TreeLeafNode Implementation
// ============================================================================

impl<P: LeafPolicy> TreeLeafNode<P> for LeafNode15<P> {
    type Perm = Permuter15;
    type Internode = InternodeNode;

    const WIDTH: usize = WIDTH_15;
    const SPLIT_THRESHOLD: usize = WIDTH_15;

    #[cfg(not(feature = "small-suffix-capacity"))]
    const INLINE_KSUF_CAPACITY: usize = 512;

    #[cfg(feature = "small-suffix-capacity")]
    const INLINE_KSUF_CAPACITY: usize = 256;

    // ========================================================================
    //  Construction
    // ========================================================================

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

    // ========================================================================
    //  Version / Lock
    // ========================================================================

    #[inline(always)]
    fn version(&self) -> &NodeVersion {
        self.version()
    }

    // ========================================================================
    //  Permutation
    // ========================================================================

    #[inline(always)]
    fn permutation(&self) -> Permuter15 {
        self.permutation()
    }

    #[inline(always)]
    fn set_permutation(&self, perm: Permuter15) {
        self.set_permutation(perm);
    }

    #[inline(always)]
    fn set_permutation_relaxed(&self, perm: Permuter15) {
        self.set_permutation_relaxed(perm);
    }

    #[inline(always)]
    fn permutation_raw(&self) -> u64 {
        self.permutation_raw()
    }

    // ========================================================================
    //  Key Operations
    // ========================================================================

    #[inline(always)]
    fn ikey(&self, slot: usize) -> u64 {
        self.ikey(slot)
    }

    #[inline(always)]
    fn ikey_relaxed(&self, slot: usize) -> u64 {
        self.ikey_relaxed(slot)
    }

    #[inline(always)]
    fn set_ikey(&self, slot: usize, ikey: u64) {
        self.set_ikey(slot, ikey);
    }

    #[inline(always)]
    fn set_ikey_relaxed(&self, slot: usize, ikey: u64) {
        self.set_ikey_relaxed(slot, ikey);
    }

    #[inline(always)]
    fn ikey_bound(&self) -> u64 {
        self.ikey_bound()
    }

    #[inline(always)]
    fn find_ikey_matches(&self, target_ikey: u64) -> u32 {
        self.find_ikey_matches(target_ikey)
    }

    #[inline(always)]
    fn keylenx(&self, slot: usize) -> u8 {
        self.keylenx(slot)
    }

    #[inline(always)]
    fn set_keylenx(&self, slot: usize, keylenx: u8) {
        self.set_keylenx(slot, keylenx);
    }

    #[inline(always)]
    fn set_keylenx_relaxed(&self, slot: usize, keylenx: u8) {
        self.set_keylenx_relaxed(slot, keylenx);
    }

    #[inline(always)]
    fn is_layer(&self, slot: usize) -> bool {
        self.is_layer(slot)
    }

    #[inline(always)]
    fn has_ksuf(&self, slot: usize) -> bool {
        self.has_ksuf(slot)
    }

    // ========================================================================
    //  Value Operations (typed)
    // ========================================================================

    #[inline(always)]
    fn load_value(&self, slot: usize) -> Option<P::Output> {
        self.load_value(slot)
    }

    #[inline(always)]
    fn store_value(&self, slot: usize, output: &P::Output) {
        self.store_value(slot, output);
    }

    #[inline(always)]
    fn store_value_relaxed(&self, slot: usize, output: &P::Output) {
        self.store_value_relaxed(slot, output);
    }

    #[inline(always)]
    fn update_value_in_place(&self, slot: usize, output: &P::Output) -> RetireHandle {
        self.update_value_in_place(slot, output)
    }

    #[inline(always)]
    fn take_value(&self, slot: usize) -> Option<P::Output> {
        self.take_value(slot)
    }

    #[inline(always)]
    fn load_layer(&self, slot: usize) -> *mut u8 {
        self.load_layer(slot)
    }

    #[inline(always)]
    fn store_layer(&self, slot: usize, ptr: *mut u8) {
        self.store_layer(slot, ptr);
    }

    #[inline(always)]
    fn is_slot_empty(&self, slot: usize) -> bool {
        self.is_slot_empty(slot)
    }

    #[inline(always)]
    fn classify_slot(&self, slot: usize) -> SlotKind<P::Output> {
        self.classify_slot(slot)
    }

    #[inline(always)]
    fn classify_slot_light(&self, slot: usize) -> SlotState {
        self.classify_slot_light(slot)
    }

    #[inline(always)]
    fn move_value_to(&self, dst: &Self, src_slot: usize, dst_slot: usize) {
        self.move_value_to(dst, src_slot, dst_slot);
    }

    #[inline(always)]
    fn clear_value(&self, slot: usize) {
        self.clear_value(slot);
    }

    // ========================================================================
    //  Slot Clearing
    // ========================================================================

    #[inline(always)]
    fn clear_slot(&self, slot: usize) {
        self.clear_slot(slot);
    }

    #[inline(always)]
    fn clear_slot_and_permutation(&self, slot: usize) {
        self.clear_slot_and_permutation(slot);
    }

    // ========================================================================
    //  Navigation
    //
    //  SAFETY for all navigation forwarding methods below: These delegate to
    //  the inherent `unsafe` methods on LeafNode15. The caller (tree traversal
    //  code) ensures pointers are valid and won't be retired during use by
    //  holding an epoch guard or the node's version lock.
    // ========================================================================

    #[inline(always)]
    fn safe_next(&self) -> *mut Self {
        // SAFETY: Caller holds an epoch guard protecting the next pointer target.
        unsafe { self.safe_next_unguarded() }
    }

    #[inline(always)]
    fn next_is_marked(&self) -> bool {
        self.next_is_marked()
    }

    #[inline(always)]
    fn set_next(&self, next: *mut Self) {
        self.set_next(next);
    }

    #[inline(always)]
    fn mark_next(&self) {
        self.mark_next();
    }

    #[inline(always)]
    fn unmark_next(&self) {
        self.unmark_next();
    }

    #[inline(always)]
    fn prev(&self) -> *mut Self {
        // SAFETY: Caller holds an epoch guard protecting the prev pointer target.
        unsafe { self.prev_unguarded() }
    }

    #[inline(always)]
    fn set_prev(&self, prev: *mut Self) {
        self.set_prev(prev);
    }

    #[inline(always)]
    fn parent(&self) -> *mut u8 {
        // SAFETY: Caller holds an epoch guard protecting the parent pointer target.
        unsafe { self.parent_unguarded() }
    }

    #[inline(always)]
    fn set_parent(&self, parent: *mut u8) {
        self.set_parent(parent);
    }

    #[inline(always)]
    unsafe fn unlink_from_chain(&self) {
        // SAFETY: Caller holds the version lock and guarantees prev is non-null
        // and all linked leaves are valid (trait method preconditions).
        unsafe { self.unlink_from_chain() }
    }

    #[inline(always)]
    fn next_raw(&self) -> *mut Self {
        // SAFETY: Caller holds an epoch guard protecting the next pointer target.
        unsafe { self.next_raw_unguarded() }
    }

    #[inline(always)]
    fn wait_for_split(&self) {
        self.wait_for_split();
    }

    // ========================================================================
    //  Slot Assignment
    // ========================================================================

    #[inline(always)]
    fn can_reuse_slot0(&self, new_ikey: u64) -> bool {
        self.can_reuse_slot0(new_ikey)
    }

    // ========================================================================
    //  Split Operations
    // ========================================================================

    fn calculate_split_point(&self, insert_pos: usize, insert_ikey: u64) -> Option<SplitPoint> {
        self.calculate_split_point(insert_pos, insert_ikey)
    }

    unsafe fn split_into_preallocated(
        &self,
        split_pos: usize,
        new_leaf_ptr: *mut Self,
        guard: &LocalGuard<'_>,
    ) -> (u64, InsertTarget) {
        // SAFETY: Caller holds lock on self, new_leaf_ptr is valid and not yet
        // visible to other threads (trait method preconditions).
        unsafe { self.split_into_preallocated(split_pos, new_leaf_ptr, guard) }
    }

    unsafe fn split_all_to_right_preallocated(
        &self,
        new_leaf_ptr: *mut Self,
        guard: &LocalGuard<'_>,
    ) -> (u64, InsertTarget) {
        // SAFETY: Caller holds lock on self, new_leaf_ptr is valid and not yet
        // visible to other threads (trait method preconditions).
        unsafe { self.split_all_to_right_preallocated(new_leaf_ptr, guard) }
    }

    unsafe fn split_and_insert(
        &self,
        split_pos: usize,
        new_leaf_ptr: *mut Self,
        insert_pos: usize,
        insert_data: &SplitInsertData<'_, P>,
        guard: &LocalGuard<'_>,
    ) -> SplitInsertResult {
        // SAFETY: Caller holds lock on self, new_leaf_ptr is valid and not yet
        // visible to other threads (trait method preconditions).
        unsafe { self.split_and_insert(split_pos, new_leaf_ptr, insert_pos, insert_data, guard) }
    }

    unsafe fn link_sibling(&self, new_sibling: *mut Self) {
        // SAFETY: Caller holds lock on self, new_sibling is valid and not yet
        // in the chain (trait method preconditions).
        unsafe { self.link_sibling(new_sibling) }
    }

    // ========================================================================
    //  Suffix Operations
    // ========================================================================

    #[inline(always)]
    fn ksuf(&self, slot: usize) -> Option<&[u8]> {
        self.ksuf(slot)
    }

    unsafe fn assign_ksuf(&self, slot: usize, suffix: &[u8], guard: &LocalGuard<'_>) -> *mut u8 {
        // SAFETY: Caller holds lock on this leaf. Suffix storage is modified
        // under lock; old bag pointer returned for deferred retirement.
        unsafe { self.assign_ksuf(slot, suffix, guard) }
    }

    unsafe fn retire_suffix_bag_ptr(ptr: *mut u8, guard: &LocalGuard<'_>) {
        // SAFETY: ptr points to a heap-allocated suffix bag that is no longer
        // reachable. Guard defers deallocation until no readers hold references.
        unsafe { Self::retire_suffix_bag_ptr(ptr, guard) }
    }

    unsafe fn assign_ksuf_init(&self, slot: usize, suffix: &[u8], guard: &LocalGuard<'_>) {
        // SAFETY: Caller holds lock. Slot has no prior suffix (first assignment).
        unsafe { self.assign_ksuf_init(slot, suffix, guard) }
    }

    unsafe fn clear_ksuf(&self, slot: usize, guard: &LocalGuard<'_>) {
        // SAFETY: Caller holds lock. Old suffix bag pointer retired via guard.
        unsafe { self.clear_ksuf(slot, guard) }
    }

    // ========================================================================
    //  Suffix Comparison
    // ========================================================================

    #[inline(always)]
    fn ksuf_equals(&self, slot: usize, suffix: &[u8]) -> bool {
        self.ksuf_equals(slot, suffix)
    }

    #[inline(always)]
    fn ksuf_compare(&self, slot: usize, suffix: &[u8]) -> Option<Ordering> {
        self.ksuf_compare(slot, suffix)
    }

    #[inline(always)]
    fn ksuf_or_empty(&self, slot: usize) -> &[u8] {
        self.ksuf_or_empty(slot)
    }

    #[inline(always)]
    fn ksuf_matches(&self, slot: usize, ikey: u64, suffix: &[u8]) -> bool {
        self.ksuf_matches(slot, ikey, suffix)
    }

    #[inline(always)]
    fn ksuf_match_result(&self, slot: usize, keylenx: u8, suffix: &[u8]) -> i32 {
        self.ksuf_match_result(slot, keylenx, suffix)
    }

    // ========================================================================
    //  Cache Optimization
    // ========================================================================

    #[inline(always)]
    fn prefetch(&self) {
        self.prefetch();
    }

    #[inline(always)]
    fn prefetch_ikey(&self, slot: usize) {
        self.prefetch_ikey(slot);
    }

    #[inline(always)]
    fn prefetch_for_search(&self) {
        self.prefetch_for_search();
    }

    #[inline(always)]
    fn prefetch_for_search_adaptive(&self, size: usize) {
        self.prefetch_for_search_adaptive(size);
    }

    // ========================================================================
    //  Modification State
    // ========================================================================

    #[inline(always)]
    fn modstate(&self) -> u8 {
        self.modstate()
    }

    #[inline(always)]
    fn set_modstate(&self, state: u8) {
        self.set_modstate(state);
    }

    #[inline(always)]
    fn deleted_layer(&self) -> bool {
        self.deleted_layer()
    }

    #[inline(always)]
    fn mark_deleted_layer(&self) {
        self.mark_deleted_layer();
    }

    #[inline(always)]
    fn mark_remove(&self) {
        self.mark_remove();
    }

    #[inline(always)]
    fn is_removing(&self) -> bool {
        self.is_removing()
    }

    #[inline(always)]
    fn is_empty_state(&self) -> bool {
        self.is_empty_state()
    }

    #[inline(always)]
    fn mark_empty(&self) {
        self.mark_empty();
    }

    #[inline(always)]
    fn clear_empty_state(&self) {
        self.clear_empty_state();
    }
}

// =============================================================================
// Drop Implementation
// =============================================================================

impl<P: LeafPolicy> Drop for LeafNode15<P> {
    fn drop(&mut self) {
        // ====================================================================
        //  Value Cleanup
        // ====================================================================
        //
        // For BoxPolicy: each terminal value slot holds a raw Box<V> pointer.
        // We must call cleanup() to drop it (deallocate).
        //
        // For InlinePolicy: values are Copy bits in AtomicU64. No heap
        // allocation, no cleanup needed. The `if P::NEEDS_RETIREMENT` check
        // compiles away entirely for inline mode.
        if P::NEEDS_RETIREMENT {
            for slot in 0..WIDTH_15 {
                // Skip empty slots first (null pointer = no value stored).
                // NOTE: We check is_empty() before keylenx because keylenx == 0
                // is a valid terminal value state (keys exactly 8 bytes long
                // produce keylenx == 0). Using keylenx == 0 as an empty check
                // would leak those Arc values.
                if self.values.is_empty(slot) {
                    continue;
                }

                let klx: u8 = self.keylenx[slot].load(RELAXED);

                // Skip layer pointer slots — layer pointers are owned by the
                // tree and cleaned up during tree teardown traversal, not here.
                if klx >= LAYER_KEYLENX {
                    continue;
                }

                // Terminal value slot — value array has content and it's not a layer.
                // SAFETY:
                // - We have &mut self (exclusive access via Drop).
                // - Slot contains a terminal value (value is non-empty and
                //   keylenx < LAYER_KEYLENX).
                // - The pointer was stored by us via Arc::into_raw.
                unsafe {
                    self.values.cleanup(slot);
                }
            }
        }
        // For InlinePolicy (NEEDS_RETIREMENT = false):
        //   - inline_values are AtomicU64 bits, dropped automatically
        //   - tags are AtomicPtr with sentinel pointers, no ownership
        //   - No value cleanup needed

        // ====================================================================
        //  Suffix Cleanup
        // ====================================================================
        //
        // Drops the heap-allocated SuffixSidecar (which in turn drops its
        // InlineSuffixBag and external SuffixBag if allocated).
        //
        // SAFETY: We have &mut self (exclusive access via Drop).
        unsafe {
            self.suffix.drop_storage();
        }
    }
}

// =============================================================================
// LayerCapableLeaf Implementation
// =============================================================================

impl<P: LeafPolicy> LayerCapableLeaf<P> for LeafNode15<P> {
    #[inline]
    fn try_clone_output(&self, slot: usize) -> Option<P::Output> {
        debug_assert!(
            slot < WIDTH_15,
            "try_clone_output: slot {slot} >= WIDTH_15 {WIDTH_15}"
        );

        if self.keylenx(slot) >= LAYER_KEYLENX {
            return None;
        }

        self.values.load(slot)
    }

    #[expect(clippy::cast_possible_truncation)]
    unsafe fn assign_from_key(
        &self,
        slot: usize,
        key: &Key<'_>,
        value: Option<P::Output>,
        guard: &LocalGuard<'_>,
    ) {
        debug_assert!(
            slot < WIDTH_15,
            "assign_from_key: slot {slot} >= WIDTH_15 {WIDTH_15}"
        );

        let output: P::Output = value.expect(
            "assign_from_key: value cannot be None (source slot was not a value); \
             this indicates a bug in conflict detection",
        );

        // Store key.
        self.set_ikey(slot, key.ikey());

        // Store value — typed, no raw pointer conversion.
        self.store_value(slot, &output);

        // Store keylenx and suffix.
        if key.has_suffix() {
            self.set_keylenx(slot, KSUF_KEYLENX);
            // SAFETY: Caller holds lock.
            unsafe { self.assign_ksuf(slot, key.suffix(), guard) };
        } else {
            let inline_len: u8 = key.current_len().min(8) as u8;
            self.set_keylenx(slot, inline_len);
        }
    }
}
