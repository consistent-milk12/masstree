//! Filepath: src/leaf24.rs
//!
//! Leaf node for [`crate::MassTree`] with WIDTH=24 (24 slots).
//!
//! This module provides `LeafNode24`, a leaf node variant optimized for reduced
//! split frequency by using 24 slots instead of the standard 15. The key difference
//! is the use of [`AtomicPermuter24`] (u128) instead of `AtomicU64` for permutation.
//!
//! # Design
//!
//! The 24-slot design requires 5 bits per slot (values 0-23) vs 4 bits for WIDTH=15.
//! Total: 5 (size) + 24×5 (slots) = 125 bits, requiring u128 storage.

use static_assertions::const_assert_eq;
use std::array as StdArray;
use std::cell::UnsafeCell;
use std::cmp::Ordering;
use std::fmt as StdFmt;
use std::hint as StdHint;
use std::marker::PhantomData;
use std::ptr as StdPtr;
use std::sync::Arc;
use std::sync::atomic::{self as StdAtomic, Ordering as AtomicOrdering};
use std::sync::atomic::{AtomicPtr, AtomicU8, AtomicU64};

use crate::key::IKEY_SIZE;
use crate::nodeversion::NodeVersion;
use crate::ordering::{CAS_FAILURE, CAS_SUCCESS, READ_ORD, RELAXED, WRITE_ORD};
use crate::permuter24::{AtomicPermuter24, Permuter24};
use crate::prefetch::prefetch_read;
use crate::slot::ValueSlot;
use crate::suffix::{InlineSuffixBag, SuffixBag};
use crate::{is_marked, mark_ptr};
use seize::{Guard, LocalGuard};

/// Default capacity for inline suffix storage (bytes).
/// Matches C++ Masstree's typical iksuf size.
const INLINE_KSUF_CAPACITY: usize = 256;

/// Special keylenx value indicating key has a suffix.
pub const KSUF_KEYLENX: u8 = 64;

/// Base keylenx value indicating a layer pointer (>= this means layer).
pub const LAYER_KEYLENX: u8 = 128;

/// Width constant for [`LeafNode24`].
pub const WIDTH_24: usize = 24;

/// Return value from [`LeafNode24::ksuf_match_result`] indicating an exact match.
pub const MATCH_RESULT_EXACT: i32 = 1;

/// Return value from [`LeafNode24::ksuf_match_result`] indicating same ikey but different key.
pub const MATCH_RESULT_MISMATCH: i32 = 0;

/// Return value from [`LeafNode24::ksuf_match_result`] indicating a layer pointer.
///
/// This is `-IKEY_SIZE` (i.e., `-8`), signaling the caller should descend into
/// the sublayer rather than treating this as a key match or mismatch.
#[expect(
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
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

/// Leaf node with 24 slots using u128 permutation.
///
/// # Concurrency Model
///
/// Uses optimistic concurrency control (OCC) via [`NodeVersion`] for readers,
/// and lock-based writes. The [`AtomicPermuter24`] permutation field enables
/// lock-free slot ordering updates.
///
/// # Memory Layout (896 bytes, 14 cache lines)
///
/// ```text
/// Offset   Size   Field
/// ------   ----   -----
/// 0        4B     version (NodeVersion)
/// 4        1B     modstate
/// 5        55B    _pad0 (cache line isolation)
/// 64       16B    permutation (AtomicPermuter24)
/// 80       48B    _pad1 (cache line isolation)
/// 128      192B   ikey0[24] (24 × 8B)
/// 320      24B    keylenx[24]
/// 344      192B   leaf_values[24] (24 × 8B)
/// 536      318B   inline_ksuf (InlineSuffixBag)
/// 854      2B     implicit padding
/// 856      8B     external_ksuf
/// 864      8B     next
/// 872      8B     prev
/// 880      8B     parent
/// 888      8B     tail padding (align to 64B)
/// ```
#[repr(C, align(64))]
pub struct LeafNode24<S: ValueSlot> {
    // ========================================================================
    // Cache Line 0: Version + metadata (read-heavy, rarely written)
    // ========================================================================
    /// Version for optimistic concurrency control.
    version: NodeVersion,

    /// Modification state for coordinating insert/remove operations.
    /// - 0 = `MODSTATE_INSERT` (default, normal operation)
    /// - 1 = `MODSTATE_REMOVE` (being removed)
    /// - 2 = `MODSTATE_DELETED_LAYER` (sublayer was gc'd)
    /// - 3 = `MODSTATE_EMPTY` (all keys removed, can be reused)
    modstate: AtomicU8,

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
    /// Stores `Arc<V>` raw pointer or layer pointer as `*mut u8`.
    /// Type is determined by keylenx: if < `LAYER_KEYLENX` → `Arc<V>`, else → layer node.
    leaf_values: [AtomicPtr<u8>; WIDTH_24],

    /// Inline suffix storage (embedded, no heap allocation for small suffixes).
    /// Uses `UnsafeCell` for interior mutability under lock.
    inline_ksuf: UnsafeCell<InlineSuffixBag<WIDTH_24, INLINE_KSUF_CAPACITY>>,

    /// External suffix storage (heap-allocated overflow).
    /// Only allocated when inline storage is full.
    external_ksuf: AtomicPtr<SuffixBag<WIDTH_24>>,

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

// Compile-time layout verification.
// LeafNode24 must be cache-line aligned (64 bytes) for optimal performance.
const_assert_eq!(std::mem::align_of::<LeafNode24<crate::LeafValue<u64>>>(), 64);

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
            modstate: AtomicU8::new(MODSTATE_INSERT),
            _pad0: [0; 55],
            permutation: AtomicPermuter24::new(),
            _pad1: [0; 48],
            ikey0: StdArray::from_fn(|_| AtomicU64::new(0)),
            keylenx: StdArray::from_fn(|_| AtomicU8::new(0)),
            leaf_values: StdArray::from_fn(|_| AtomicPtr::new(StdPtr::null_mut())),
            inline_ksuf: UnsafeCell::new(InlineSuffixBag::new()),
            external_ksuf: AtomicPtr::new(StdPtr::null_mut()),
            next: AtomicPtr::new(StdPtr::null_mut()),
            prev: AtomicPtr::new(StdPtr::null_mut()),
            parent: AtomicPtr::new(StdPtr::null_mut()),
            _marker: PhantomData,
        }
    }

    /// Initialize a leaf node directly at the given pointer.
    ///
    /// This avoids stack allocation and copy by writing directly to the destination.
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
            // Zero the entire struct first (most fields are zero-initialized)
            StdPtr::write_bytes(ptr, 0, 1);

            // Now write the non-zero fields
            let node = &mut *ptr;

            // Version: leaf node, optionally root
            StdPtr::write(&raw mut node.version, NodeVersion::new(true));
            if is_root {
                node.version.mark_root();
            }

            // ModState: Insert mode (atomic)
            StdPtr::write(&raw mut node.modstate, AtomicU8::new(MODSTATE_INSERT));

            // Permutation: empty
            StdPtr::write(&raw mut node.permutation, AtomicPermuter24::new());

            // InlineSuffixBag: new (contains non-zero atomics)
            StdPtr::write(
                &raw mut (*ptr).inline_ksuf,
                UnsafeCell::new(InlineSuffixBag::new()),
            );
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
    /// NOTE: This matches `LeafNode::make_layer_root` in `src/leaf/layer.rs`.
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
    /// Uses Acquire ordering to synchronize with writer's Release stores.
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

    /// Get the ikey at the given physical slot using Relaxed ordering.
    ///
    /// # Safety Justification
    ///
    /// Safe to use Relaxed when:
    /// 1. Caller has already loaded permutation with Acquire ordering, which
    ///    synchronizes with the writer's Release fence after modifications
    /// 2. OCC version validation at the end of the read catches any races
    ///
    /// This avoids redundant Acquire fences on each ikey load (up to 24 per search),
    /// improving read throughput by 10-15%.
    ///
    /// # Panics
    /// Panics in debug mode if `slot >= WIDTH_24`.
    #[must_use]
    #[inline(always)]
    #[expect(
        clippy::indexing_slicing,
        reason = "Slot from Permuter24, valid by construction"
    )]
    pub fn ikey_relaxed(&self, slot: usize) -> u64 {
        debug_assert!(slot < WIDTH_24, "ikey_relaxed: slot out of bounds");

        self.ikey0[slot].load(RELAXED)
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

    /// Prefetch leaf node data for range scans.
    ///
    /// Brings the node's key arrays (`ikey0`, `keylenx`) and value pointers
    /// (`leaf_values`) into CPU cache before they're accessed, reducing memory
    /// latency during sequential scanning.
    ///
    /// # Memory Layout (WIDTH=24)
    ///
    /// ```text
    /// Offset   Size    Field
    /// ------   ----    -----
    /// 0        64B     Cache line 0: version + modstate + padding
    /// 64       64B     Cache line 1: permutation (u128) + padding
    /// 128      192B    ikey0 (24 × 8B = 192B, ~3 cache lines)
    /// 320      24B     keylenx (24 × 1B)
    /// 344      192B    leaf_values (24 × 8B = 192B, ~3 cache lines)
    /// ```
    ///
    /// # C++ Reference
    ///
    /// Matches C++ `leaf::prefetch()` pattern from `masstree_scan.hh:195, 299`.
    #[inline(always)]
    pub fn prefetch(&self) {
        let self_ptr: *const u8 = StdPtr::from_ref::<Self>(self).cast::<u8>();

        // Prefetch ikey0 array (starts at offset 128, spans ~3 cache lines)
        // Skip cache lines 0-1 (version/permutation) - already accessed
        // SAFETY: self_ptr is derived from a valid reference, and offsets are within struct bounds.
        unsafe {
            prefetch_read(self_ptr.add(128)); // ikey0[0..8]
            prefetch_read(self_ptr.add(192)); // ikey0[8..16]
            prefetch_read(self_ptr.add(256)); // ikey0[16..24] + keylenx

            // Prefetch leaf_values array (starts at ~344, spans ~3 cache lines)
            prefetch_read(self_ptr.add(320)); // keylenx + leaf_values[0..8]
            prefetch_read(self_ptr.add(384)); // leaf_values[8..16]
            prefetch_read(self_ptr.add(448)); // leaf_values[16..24]
        }
    }

    /// Prefetch the ikey at the given slot into CPU cache.
    ///
    /// This is used during linear search to hide memory latency by
    /// prefetching future ikeys while processing current ones.
    ///
    /// # Arguments
    ///
    /// * `slot` - Physical slot index `(0..WIDTH_24)`
    ///
    /// # Safety
    ///
    /// The slot must be in range `[0, WIDTH_24)`. No bounds check in release mode.
    #[inline(always)]
    #[expect(clippy::indexing_slicing, reason = "Caller ensures slot is valid")]
    pub fn prefetch_ikey(&self, slot: usize) {
        debug_assert!(slot < WIDTH_24, "prefetch_ikey: slot out of bounds");
        prefetch_read(&raw const self.ikey0[slot]);
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

    /// Load external suffix bag pointer (reader).
    #[must_use]
    #[inline(always)]
    pub fn external_ksuf_ptr(&self) -> *mut SuffixBag<WIDTH_24> {
        self.external_ksuf.load(READ_ORD)
    }

    /// Check if this leaf has external suffix storage allocated.
    #[must_use]
    #[inline(always)]
    pub fn has_external_ksuf(&self) -> bool {
        !self.external_ksuf_ptr().is_null()
    }

    /// Get the suffix for a slot (checks inline first, then external).
    ///
    /// # Safety Note
    ///
    /// Caller must ensure suffix storage is stable via version validation or lock.
    #[must_use]
    #[inline]
    pub fn ksuf(&self, slot: usize) -> Option<&[u8]> {
        debug_assert!(slot < WIDTH_24, "ksuf: slot {slot} >= WIDTH_24 {WIDTH_24}");

        if !self.has_ksuf(slot) {
            return None;
        }

        // FAST PATH: Check inline storage first
        // SAFETY: We're reading. Concurrent writes require lock, and readers
        // use version validation to retry on changes.
        let inline: &InlineSuffixBag<WIDTH_24, INLINE_KSUF_CAPACITY> =
            unsafe { &*self.inline_ksuf.get() };
        if let Some(suffix) = inline.get(slot) {
            return Some(suffix);
        }

        // SLOW PATH: Check external storage
        let ext_ptr: *mut SuffixBag<WIDTH_24> = self.external_ksuf_ptr();
        if ext_ptr.is_null() {
            return None;
        }

        // SAFETY: ext_ptr is non-null and came from Box::into_raw.
        unsafe { (*ext_ptr).get(slot) }
    }

    /// Get the suffix for a slot, or an empty slice if none.
    #[must_use]
    #[inline(always)]
    pub fn ksuf_or_empty(&self, slot: usize) -> &[u8] {
        self.ksuf(slot).unwrap_or(&[])
    }

    /// Assign a suffix to a slot (two-tier: inline first, then external).
    ///
    /// This uses the C++ Masstree optimization: try inline storage first,
    /// only allocate external storage when inline is full.
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
        debug_assert!(
            self.version.is_locked() || self.version.is_unpublished(),
            "assign_ksuf: caller must hold lock or node must be unpublished"
        );

        // FAST PATH 1: Try inline storage first (no allocation!)
        // SAFETY: We hold the lock (verified above), so no concurrent writers.
        let inline: &mut InlineSuffixBag<WIDTH_24, INLINE_KSUF_CAPACITY> =
            unsafe { &mut *self.inline_ksuf.get() };

        if inline.try_assign(slot, suffix) {
            self.keylenx[slot].store(KSUF_KEYLENX, WRITE_ORD);
            return;
        }

        // FAST PATH 2: Try external storage in-place (if exists and has room)
        let old_ext: *mut SuffixBag<WIDTH_24> = self.external_ksuf.load(RELAXED);
        if !old_ext.is_null() {
            // SAFETY: old_ext is non-null and came from Box::into_raw.
            let bag: &mut SuffixBag<WIDTH_24> = unsafe { &mut *old_ext };
            if bag.try_assign_in_place(slot, suffix) {
                // Clear from inline if it was there
                inline.clear(slot);
                self.keylenx[slot].store(KSUF_KEYLENX, WRITE_ORD);
                return;
            }
        }

        // SLOW PATH: Drain inline to external and allocate new external bag
        // SAFETY: Same preconditions as this function (caller holds lock, guard is valid).
        unsafe { self.assign_ksuf_slow(slot, suffix, guard) };
    }

    /// Slow path for suffix assignment: allocate/reallocate external bag.
    ///
    /// # Safety
    /// Same as `assign_ksuf`.
    #[cold]
    #[inline(never)]
    #[expect(clippy::indexing_slicing, reason = "Slot bounds checked by caller")]
    unsafe fn assign_ksuf_slow(&self, slot: usize, suffix: &[u8], guard: &LocalGuard<'_>) {
        debug_assert!(
            self.version.is_locked(),
            "assign_ksuf_slow: caller must hold lock"
        );

        let perm = self.permutation();
        // SAFETY: Caller holds lock (verified above), ensuring exclusive access to inline_ksuf.
        let inline: &mut InlineSuffixBag<WIDTH_24, INLINE_KSUF_CAPACITY> =
            unsafe { &mut *self.inline_ksuf.get() };

        // Drain inline to a new external bag with the new suffix
        let mut new_bag: SuffixBag<WIDTH_24> = inline.drain_to_external(&perm, slot, suffix);

        // Merge with existing external suffixes (if any)
        let old_ext: *mut SuffixBag<WIDTH_24> = self.external_ksuf.load(RELAXED);
        if !old_ext.is_null() {
            // SAFETY: old_ext is non-null
            let old_bag: &SuffixBag<WIDTH_24> = unsafe { &*old_ext };

            for i in 0..perm.size() {
                let s: usize = perm.get(i);

                if s != slot
                    && let Some(ext_suffix) = old_bag.get(s)
                {
                    new_bag.assign(s, ext_suffix);
                }
            }
        }

        // Install new external bag
        let new_ptr: *mut SuffixBag<WIDTH_24> = Box::into_raw(Box::new(new_bag));
        self.external_ksuf.store(new_ptr, WRITE_ORD);

        // Retire old external bag
        if !old_ext.is_null() {
            // SAFETY: old_ext is non-null and came from Box::into_raw
            unsafe {
                guard.defer_retire(old_ext, |ptr, _| {
                    drop(Box::from_raw(ptr));
                });
            }
        }

        self.keylenx[slot].store(KSUF_KEYLENX, WRITE_ORD);
    }

    /// Clear the suffix from a slot (no allocation needed!).
    ///
    /// Unlike the old copy-on-write approach, this just marks the slot
    /// as empty in both inline and external storage. No cloning required.
    ///
    /// # Safety
    /// - Caller must hold lock and have called `mark_insert()`
    /// - `guard` must come from this tree's collector (unused but kept for API compat)
    #[expect(
        clippy::indexing_slicing,
        reason = "Slot bounds checked via debug_assert"
    )]
    pub unsafe fn clear_ksuf(&self, slot: usize, _guard: &LocalGuard<'_>) {
        debug_assert!(
            slot < WIDTH_24,
            "clear_ksuf: slot {slot} >= WIDTH_24 {WIDTH_24}"
        );
        debug_assert!(
            self.version.is_locked(),
            "clear_ksuf: caller must hold lock"
        );

        // Clear from inline storage
        // SAFETY: We hold the lock (verified above), so no concurrent writers.
        let inline: &mut InlineSuffixBag<WIDTH_24, INLINE_KSUF_CAPACITY> =
            unsafe { &mut *self.inline_ksuf.get() };
        inline.clear(slot);

        // Clear from external storage (if exists)
        let ext_ptr: *mut SuffixBag<WIDTH_24> = self.external_ksuf.load(RELAXED);
        if !ext_ptr.is_null() {
            // SAFETY: ext_ptr is non-null and came from Box::into_raw.
            // We hold the lock, so we can mutate in place.
            let bag: &mut SuffixBag<WIDTH_24> = unsafe { &mut *ext_ptr };
            bag.clear(slot);
        }

        self.keylenx[slot].store(0, WRITE_ORD);
    }

    /// Check if a slot's suffix equals the given suffix.
    #[must_use]
    #[inline]
    pub fn ksuf_equals(&self, slot: usize, suffix: &[u8]) -> bool {
        debug_assert!(
            slot < WIDTH_24,
            "ksuf_equals: slot {slot} >= WIDTH_24 {WIDTH_24}"
        );

        if !self.has_ksuf(slot) {
            return false;
        }

        // Check inline first
        // SAFETY: Reader access, concurrent writes require lock.
        let inline: &InlineSuffixBag<WIDTH_24, INLINE_KSUF_CAPACITY> =
            unsafe { &*self.inline_ksuf.get() };
        if inline.suffix_equals(slot, suffix) {
            return true;
        }

        // Check external
        let ext_ptr: *mut SuffixBag<WIDTH_24> = self.external_ksuf_ptr();
        if ext_ptr.is_null() {
            return false;
        }

        // SAFETY: ext_ptr is non-null and came from Box::into_raw
        unsafe { (*ext_ptr).suffix_equals(slot, suffix) }
    }

    /// Compare a slot's suffix with the given suffix.
    #[must_use]
    #[inline]
    pub fn ksuf_compare(&self, slot: usize, suffix: &[u8]) -> Option<Ordering> {
        debug_assert!(
            slot < WIDTH_24,
            "ksuf_compare: slot {slot} >= WIDTH_24 {WIDTH_24}"
        );

        if !self.has_ksuf(slot) {
            return None;
        }

        // Check inline first
        // SAFETY: Reader access, concurrent writes require lock.
        let inline: &InlineSuffixBag<WIDTH_24, INLINE_KSUF_CAPACITY> =
            unsafe { &*self.inline_ksuf.get() };
        if let Some(cmp) = inline.suffix_compare(slot, suffix) {
            return Some(cmp);
        }

        // Check external
        let ext_ptr: *mut SuffixBag<WIDTH_24> = self.external_ksuf_ptr();
        if ext_ptr.is_null() {
            return None;
        }

        // SAFETY: ext_ptr is non-null and came from Box::into_raw
        unsafe { (*ext_ptr).suffix_compare(slot, suffix) }
    }

    /// Check if a slot's key matches the given key.
    #[must_use]
    #[inline]
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
    /// * [`MATCH_RESULT_EXACT`] (1) - Exact match
    /// * [`MATCH_RESULT_MISMATCH`] (0) - Same ikey but different key
    /// * [`MATCH_RESULT_LAYER`] (-8) - Slot is a layer pointer
    #[must_use]
    #[inline(always)]
    pub fn ksuf_match_result(&self, slot: usize, keylenx: u8, suffix: &[u8]) -> i32 {
        debug_assert!(
            slot < WIDTH_24,
            "ksuf_match_result: slot {slot} >= WIDTH_24 {WIDTH_24}"
        );

        let stored_keylenx: u8 = self.keylenx(slot);

        if Self::keylenx_is_layer(stored_keylenx) {
            return MATCH_RESULT_LAYER;
        }

        if !self.has_ksuf(slot) {
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

    /// Compact external suffix storage.
    ///
    /// Note: Inline storage doesn't need compaction (fixed size, no fragmentation).
    /// This only compacts the external bag if it exists.
    ///
    /// # Safety
    /// - Caller must hold lock
    /// - The `guard` must be valid and from the same collector as the tree.
    pub unsafe fn compact_ksuf(
        &self,
        exclude_slot: Option<usize>,
        guard: &LocalGuard<'_>,
    ) -> usize {
        debug_assert!(
            self.version.is_locked(),
            "compact_ksuf: caller must hold lock"
        );

        let old_ptr: *mut SuffixBag<WIDTH_24> = self.external_ksuf.load(RELAXED);
        if old_ptr.is_null() {
            return 0;
        }

        let perm = self.permutation();
        // SAFETY: old_ptr is non-null
        let mut new_bag: SuffixBag<WIDTH_24> = unsafe { (*old_ptr).clone() };
        let reclaimed = new_bag.compact_with_permuter(&perm, exclude_slot);
        let new_ptr: *mut SuffixBag<WIDTH_24> = Box::into_raw(Box::new(new_bag));

        self.external_ksuf.store(new_ptr, WRITE_ORD);

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

    /// Store key metadata (`ikey`, `keylenx`) for a CAS insert attempt.
    ///
    /// # Safety
    /// - The caller must have successfully claimed the slot via `cas_slot_value` and ensured
    ///   the slot still belongs to the CAS attempt (i.e. `leaf_values[slot]` still equals the
    ///   claimed pointer).
    ///
    /// Note: writing key metadata *before* claiming the slot is not safe in this design because
    /// multiple concurrent CAS attempts can overwrite each other's metadata before publish.
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
    /// Spins until the next pointer is unmarked, the version is stable,
    /// OR the node is marked as deleted.
    ///
    /// # Note
    ///
    /// A marked next pointer can mean either:
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
                StdHint::spin_loop();
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
                    // Return UNMARKED old next (may be null). We intentionally mark even
                    // `NULL` next pointers to avoid two concurrent splits both "seeing"
                    // `NULL` and racing to publish different siblings, orphaning one.
                    return next;
                }
                Err(_) => {
                    // CAS failed: someone else updated next, retry
                    StdHint::spin_loop();
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
        self.next
            .compare_exchange(current, new, CAS_SUCCESS, CAS_FAILURE)
    }

    /// Unlink this leaf from the B-link doubly-linked chain.
    ///
    /// This is the inverse of `link_sibling`. Used when removing
    /// an empty leaf from the tree.
    ///
    /// # Algorithm (from C++ `btree_leaflink.hh:76-96`)
    ///
    /// 1. Lock our next pointer via CAS marking
    /// 2. CAS prev->next from self to marked(self) to signal unlinking
    /// 3. Update next->prev = prev
    /// 4. Release fence for visibility
    /// 5. Store prev->next = next (unmarked), completing the unlink
    ///
    /// # Preconditions
    ///
    /// - Self is locked (caller holds version lock)
    /// - Self has a predecessor (`prev` is non-null)
    ///
    /// # Safety
    ///
    /// - Caller must hold the version lock on this leaf
    /// - `self.prev()` must be non-null (not the leftmost leaf)
    /// - The prev and next pointers must be valid leaves
    pub unsafe fn unlink_from_chain(&self) {
        // Step 1: Lock our next pointer (mark it)
        // This prevents concurrent splits from interfering
        let next: *mut Self = self.lock_next();

        // Step 2: CAS prev->next from self to marked(self)
        // This signals to prev that we're unlinking
        //
        // IMPORTANT: We must re-read prev on each iteration because if prev splits,
        // a new node becomes our predecessor and we need to CAS on that node instead.
        // (This matches the C++ implementation in btree_leaflink.hh:86-91)
        let self_ptr: *mut Self = StdPtr::from_ref(self).cast_mut();
        let marked_self: *mut Self = mark_ptr(self_ptr);

        let final_prev: *mut Self;
        loop {
            // Re-read prev on each iteration (may change if prev splits)
            let prev: *mut Self = self.prev();
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
                    if is_marked(current) {
                        // SAFETY: prev is valid
                        unsafe { (*prev).wait_for_split() };
                    }
                    // Otherwise, prev->next doesn't point to us - prev may have split
                    // and our new prev is the split sibling. Loop and re-read prev.
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
    ///
    /// This is distinct from `version.is_deleted()`:
    /// - `is_deleted()` means the node itself is removed from the tree
    /// - `deleted_layer()` means the sublayer this node was root of has been gc'd
    ///
    /// When `deleted_layer()` is true, readers should reset their key position
    /// (`unshift_all`) and retry from the main tree root.
    ///
    /// # C++ Reference
    ///
    /// Matches `leaf::deleted_layer()` in `masstree_struct.hh:456-458`.
    #[must_use]
    #[inline(always)]
    pub fn deleted_layer(&self) -> bool {
        self.modstate() == MODSTATE_DELETED_LAYER
    }

    /// Mark this layer as deleted (for `gc_layer`).
    ///
    /// Called when garbage collecting an empty sublayer. The parent's slot
    /// that pointed to this sublayer will be cleared, and this leaf is marked
    /// so concurrent readers know to retry from the tree root.
    ///
    /// # C++ Reference
    ///
    /// Matches setting `modstate_ = modstate_deleted_layer` in C++.
    #[inline(always)]
    pub fn mark_deleted_layer(&self) {
        self.set_modstate(MODSTATE_DELETED_LAYER);
    }

    /// Mark this node as being in remove mode.
    ///
    /// Called at the start of a remove operation to prevent suffix allocation
    /// during the remove process.
    ///
    /// # C++ Reference
    ///
    /// Matches the modstate transition in `finish_remove` (`masstree_remove.hh:162-166`).
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
    ///
    /// Empty state means the leaf had all its keys removed and is available
    /// for reuse by insert or cleanup by the coalescing background task.
    #[must_use]
    #[inline(always)]
    pub fn is_empty_state(&self) -> bool {
        self.modstate() == MODSTATE_EMPTY
    }

    /// Mark this leaf as empty (all keys removed).
    ///
    /// Called when the last key is removed from a leaf. The leaf remains
    /// in the tree structure but is marked for potential reuse or cleanup.
    ///
    /// Empty leaves can be:
    /// - Reused by insert operations (saves allocation)
    /// - Cleaned up by background coalescing task
    #[inline(always)]
    pub fn mark_empty(&self) {
        self.set_modstate(MODSTATE_EMPTY);
    }

    /// Clear empty state, returning to normal insert mode.
    ///
    /// Called when an empty leaf is being reused for a new insert.
    /// This resets the modstate to allow normal operation.
    #[inline(always)]
    pub fn clear_empty_state(&self) {
        self.set_modstate(MODSTATE_INSERT);
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

    // ============================================================================
    //  Slot Clearing (for gc_layer)
    // ============================================================================

    /// Clear a slot completely, removing any value or layer pointer.
    ///
    /// This is used by `gc_layer` when cleaning up an empty sublayer.
    /// The parent leaf's slot that pointed to the sublayer is cleared.
    ///
    /// # Memory Ordering
    ///
    /// Uses Release ordering to ensure the clear is visible to subsequent readers.
    /// The permutation should be updated separately to remove this slot from
    /// the logical ordering.
    ///
    /// # Safety
    ///
    /// The caller must ensure:
    /// - The leaf is locked
    /// - The slot is valid (0..WIDTH)
    /// - Any value/layer at this slot has been or will be properly retired
    #[inline]
    #[expect(clippy::indexing_slicing)]
    pub fn clear_slot(&self, slot: usize) {
        debug_assert!(slot < WIDTH_24, "clear_slot: slot out of bounds");

        // Clear keylenx to 0 (marks slot as empty for searches)
        self.keylenx[slot].store(0, AtomicOrdering::Release);

        // Clear the value pointer
        self.leaf_values[slot].store(StdPtr::null_mut(), AtomicOrdering::Release);

        // Note: ikey is NOT cleared - it's only meaningful when keylenx > 0
        // Note: suffix is NOT cleared - it's only meaningful when keylenx indicates suffix
    }

    /// Clear a slot and update permutation atomically.
    ///
    /// This is a convenience method that:
    /// 1. Clears the slot contents
    /// 2. Removes the slot from the permutation
    ///
    /// # Safety
    ///
    /// The caller must ensure the leaf is locked.
    pub fn clear_slot_and_permutation(&self, slot: usize) {
        // Clear the slot
        self.clear_slot(slot);

        // Remove from permutation
        let mut perm = self.permutation();
        perm.remove_slot(slot);
        self.set_permutation(perm);
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
    // Internodes use fixed WIDTH=15 (4-bit permutation slots)
    type Internode = crate::internode::InternodeNode<S>;
    const WIDTH: usize = WIDTH_24;
    /// 80% of 24 = 19.2, use 19 to trigger splits earlier
    const SPLIT_THRESHOLD: usize = 19;

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

    /// SIMD-accelerated ikey matching for WIDTH=24.
    ///
    /// Uses `load_all_ikeys()` + SIMD comparison instead of
    /// sequential per-slot atomic loads.
    #[inline]
    fn find_ikey_matches(&self, target_ikey: u64) -> u32 {
        crate::ksearch::find_ikey_matches_leaf24(target_ikey, self)
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

    #[inline]
    fn clear_slot(&self, slot: usize) {
        Self::clear_slot(self, slot);
    }

    fn clear_slot_and_permutation(&self, slot: usize) {
        Self::clear_slot_and_permutation(self, slot);
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
    unsafe fn unlink_from_chain(&self) {
        // SAFETY: Caller guarantees preconditions
        unsafe { Self::unlink_from_chain(self) };
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
        // SAFETY: Caller guarantees slot was claimed via cas_slot_value
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

        // Split at midpoint
        let mut split_pos = size / 2;
        if split_pos == 0 {
            return None;
        }

        // Adjust for equal ikeys: if keys at split boundary are equal,
        // move split point to keep equal keys together
        while (split_pos > 0) && (split_pos < size) {
            let left_slot = perm.get(split_pos - 1);
            let right_slot = perm.get(split_pos);
            let left_ikey = self.ikey(left_slot);
            let right_ikey = self.ikey(right_slot);

            if left_ikey == right_ikey {
                // Equal keys - check if insert_ikey matches
                match insert_ikey.cmp(&left_ikey) {
                    Ordering::Equal => {
                        // Insert goes with this group - move split right
                        split_pos += 1;
                    }

                    Ordering::Less => {
                        // Insert goes left - move split left
                        split_pos -= 1;
                    }

                    Ordering::Greater => {
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
        new_leaf_ptr: *mut Self,
        guard: &seize::LocalGuard<'_>,
    ) -> (u64, crate::value::InsertTarget) {
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
        // We're writing to a freshly allocated leaf that is not yet visible.
        // Using ptr::write because NodeVersion doesn't implement Copy.
        unsafe {
            let new_leaf: &Self = &*new_leaf_ptr;
            let split_version = crate::nodeversion::NodeVersion::new_for_split(&self.version);

            StdPtr::write(
                StdPtr::addr_of!((*new_leaf_ptr).version).cast_mut(),
                split_version,
            );

            // Load current permutation (caller holds lock)
            let old_perm: Permuter24 = self.permutation();
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

            // Publish truncated permutation
            self.set_permutation(old_perm_updated);

            // Get split key from new leaf's first entry
            let split_ikey = new_leaf.ikey(new_perm.get(0));

            (split_ikey, crate::value::InsertTarget::Left)
        }
    }

    unsafe fn split_all_to_right_preallocated(
        &self,
        new_leaf_ptr: *mut Self,
        guard: &seize::LocalGuard<'_>,
    ) -> (u64, crate::value::InsertTarget) {
        // CRITICAL (Help-Along Protocol): Initialize new leaf's version for split
        // (same as split_into_preallocated)
        let split_version = crate::nodeversion::NodeVersion::new_for_split(&self.version);
        // SAFETY: new_leaf is not yet visible to other threads.
        unsafe {
            StdPtr::write(
                StdPtr::addr_of!((*new_leaf_ptr).version).cast_mut(),
                split_version,
            );
        }

        let new_leaf: &Self = unsafe { &*new_leaf_ptr };

        // Load current permutation (caller holds lock)
        let old_perm: Permuter24 = self.permutation();
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

        // Old leaf becomes empty
        self.set_permutation(Permuter24::empty());

        // Split key is first key of new leaf
        let split_ikey = new_leaf.ikey(new_perm.get(0));

        (split_ikey, crate::value::InsertTarget::Right)
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

        // Step 1: Lock the next pointer via CAS mark
        // This returns the unmarked old_next pointer
        let old_next: *mut Self = self.lock_next();

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
        StdAtomic::fence(AtomicOrdering::Release);

        // Step 5: Store new_sibling (unmarked) - atomically publishes the link
        // This also "unlocks" the next pointer by clearing the mark bit
        <Self as crate::leaf_trait::TreeLeafNode<S>>::set_next(self, new_sibling);
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
    fn ksuf_compare(&self, slot: usize, suffix: &[u8]) -> Option<Ordering> {
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

    #[inline(always)]
    fn prefetch_ikey(&self, slot: usize) {
        Self::prefetch_ikey(self, slot);
    }

    // ========================================================================
    // Modification State (modstate) Operations
    // ========================================================================

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

        // External suffix bag (inline is embedded, no cleanup needed)
        let ext_ptr: *mut SuffixBag<WIDTH_24> = self.external_ksuf.load(RELAXED);
        if !ext_ptr.is_null() {
            // SAFETY: ext_ptr came from Box::into_raw in assign_ksuf_slow.
            unsafe {
                drop(Box::from_raw(ext_ptr));
            }
        }
    }
}

// =============================================================================
// LayerCapableLeaf Implementation
// =============================================================================

impl<V: Send + Sync + 'static> crate::leaf_trait::LayerCapableLeaf<crate::value::LeafValue<V>>
    for LeafNode24<crate::value::LeafValue<V>>
{
    fn try_clone_output(&self, slot: usize) -> Option<Arc<V>> {
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
            Arc::increment_strong_count(value_ptr);
            Some(Arc::from_raw(value_ptr))
        }
    }

    unsafe fn assign_from_key_arc(
        &self,
        slot: usize,
        key: &crate::key::Key<'_>,
        value: Option<Arc<V>>,
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
        let arc: Arc<V> = value.expect(
            "assign_from_key_arc: value cannot be None (source slot was not a value); \
             this indicates a bug in conflict detection",
        );

        // Store ikey (8 bytes, big-endian encoded)
        self.set_ikey(slot, key.ikey());

        // Store Arc as raw pointer
        // NOTE: Arc ownership transfers to the slot; the slot now owns one strong reference.
        // The caller must NOT drop `value` again - it's been consumed via into_raw.
        let ptr: *mut u8 = Arc::into_raw(arc).cast_mut().cast::<u8>();
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

// =============================================================================
// LayerCapableLeaf Implementation for LeafValueIndex (Inline Mode)
// =============================================================================

impl<V: Copy + Send + Sync + 'static>
    crate::leaf_trait::LayerCapableLeaf<crate::value::LeafValueIndex<V>>
    for LeafNode24<crate::value::LeafValueIndex<V>>
{
    fn try_clone_output(&self, slot: usize) -> Option<V> {
        debug_assert!(
            slot < WIDTH_24,
            "try_clone_output: slot {slot} >= WIDTH_24 {WIDTH_24}"
        );

        // Check for layer pointer - layer pointers are NOT values
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
        // - ptr came from Box::into_raw during insert
        // - V is Copy, so we just read the value
        // - Caller ensures slot is stable (lock or version validation)
        unsafe { Some(*ptr.cast::<V>()) }
    }

    unsafe fn assign_from_key_arc(
        &self,
        slot: usize,
        key: &crate::key::Key<'_>,
        value: Option<V>,
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
        let v: V = value.expect(
            "assign_from_key_arc: value cannot be None (source slot was not a value); \
             this indicates a bug in conflict detection",
        );

        // Store ikey (8 bytes, big-endian encoded)
        self.set_ikey(slot, key.ikey());

        // Store value as boxed raw pointer
        // NOTE: Value ownership transfers to the slot.
        let ptr: *mut u8 = Box::into_raw(Box::new(v)).cast::<u8>();
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
mod unit_tests;
