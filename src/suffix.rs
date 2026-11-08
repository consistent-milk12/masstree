//! Filepath: src/suffix.rs
//!
//! Suffix storage for keys longer than 8 bytes.
//!
//! When a key is longer than 8 bytes, the first 8 bytes are stored as `ikey0`
//! and the remaining bytes are stored in a [`SuffixBag`].

use std::cell::UnsafeCell;
use std::fmt::{self, Debug, Formatter};
use std::ptr as StdPtr;
use std::slice;
use std::sync::atomic::{AtomicPtr, AtomicU8, AtomicU64, AtomicUsize, Ordering};

use crate::TreePermutation;

mod clone;
mod compact;
mod inline;
mod sidecar;

pub use inline::InlineSuffixBag;

pub use sidecar::{SideCarUtils, SuffixSidecar};

/// Number of slots (matches `WIDTH_15` leaf node).
const WIDTH: usize = 15;

/// Initial capacity for suffix storage.
const INITIAL_CAPACITY: usize = 128;

// ============================================================================
//  SlotMeta
// ============================================================================

/// Metadata for a single slot's suffix.
///
/// Stored internally as one `AtomicU64` so a reader can never observe a torn
/// `(offset, len)` pair.
#[derive(Clone, Copy, Debug, Default)]
struct SlotMeta {
    /// Offset into the data buffer (`u32::MAX` if no suffix).
    offset: u32,

    /// Length of the suffix.
    len: u16,

    /// Padding for alignment (unused).
    pad: u16,
}

impl SlotMeta {
    /// Sentinel value indicating no suffix stored.
    const EMPTY: Self = Self {
        offset: u32::MAX,
        len: 0,
        pad: 0,
    };

    /// Packed representation of [`Self::EMPTY`].
    const EMPTY_PACKED: u64 = Self::EMPTY.pack();

    /// Pack into a single 64-bit word.
    #[inline(always)]
    const fn pack(self) -> u64 {
        (self.offset as u64) | ((self.len as u64) << 32) | ((self.pad as u64) << 48)
    }

    /// Unpack from a 64-bit word.
    #[expect(
        clippy::cast_possible_truncation,
        reason = "packed layout is exactly 32/16/16 bits"
    )]
    #[inline(always)]
    const fn unpack(packed: u64) -> Self {
        Self {
            offset: packed as u32,
            len: (packed >> 32) as u16,
            pad: (packed >> 48) as u16,
        }
    }

    /// Check if this slot has a suffix.
    #[inline(always)]
    const fn has_suffix(self) -> bool {
        self.offset != u32::MAX
    }
}

// ============================================================================
//  PermutationProvider Trait
// ============================================================================

/// Trait for types that can provide permutation information.
///
/// This allows [`SuffixBag`] to work with different permutation implementations,
/// primarily [`crate::permuter::Permuter`].
pub trait PermutationProvider {
    /// Return the number of active slots.
    fn size(&self) -> usize;

    /// Return the physical slot index at logical position `i`.
    fn get(&self, i: usize) -> usize;
}

// ============================================================================
//  SuffixBag
// ============================================================================

/// Contiguous storage for key suffixes.
///
/// Reader-visible metadata and the used length are atomic, and the data bytes
/// below `used` are never rewritten by concurrent writers. Readers therefore
/// form slices through a raw pointer and the atomic `used` value instead of
/// touching `Vec::len`.
///
/// Mutation requires exclusive access: a writer either owns the bag (`&mut`) or
/// holds the leaf lock that publishes this bag through a [`SuffixBagCell`].
/// All mutation goes through interior mutability so no `&mut SuffixBag` is ever
/// created while a reader holds `&SuffixBag`.
pub struct SuffixBag {
    /// Per-slot metadata packed as `(offset, len)`.
    slots: [AtomicU64; WIDTH],

    /// Contiguous suffix data buffer. Writers access it through `UnsafeCell`;
    /// readers use `data_ptr` + `used`.
    data: UnsafeCell<Vec<u8>>,

    /// Stable base pointer of `data`, updated only when the buffer reallocates
    /// (which is forbidden for a published bag).
    data_ptr: AtomicPtr<u8>,

    /// Number of bytes published at the front of the buffer.
    used: AtomicUsize,

    /// Cached count of slots with suffixes (avoids O(WIDTH) recount).
    suffix_count: AtomicU8,
}

impl SuffixBag {
    // ========================================================================
    //  Constructor
    // ========================================================================

    /// Create a new suffix bag with initial capacity.
    #[must_use]
    #[inline(always)]
    pub fn new() -> Self {
        Self::from_vec_inner(Vec::with_capacity(INITIAL_CAPACITY))
    }

    /// Create a new suffix bag with specified capacity.
    #[must_use]
    #[inline(always)]
    pub fn with_capacity(capacity: usize) -> Self {
        Self::from_vec_inner(Vec::with_capacity(capacity))
    }

    /// Construct [`SuffixBag`] by reusing an existing [`Vec<u8>`] buffer.
    ///
    /// # Panics
    ///
    /// Panics if `data` is not empty.
    #[must_use]
    #[inline(always)]
    pub fn from_vec(data: Vec<u8>) -> Self {
        assert!(data.is_empty(), "from_vec expects empty Vec");
        Self::from_vec_inner(data)
    }

    /// Build from a data buffer with all slots empty.
    #[inline(always)]
    fn from_vec_inner(data: Vec<u8>) -> Self {
        let ptr: *mut u8 = data.as_ptr().cast_mut();

        Self {
            slots: std::array::from_fn(|_| AtomicU64::new(SlotMeta::EMPTY_PACKED)),
            data: UnsafeCell::new(data),
            data_ptr: AtomicPtr::new(ptr),
            used: AtomicUsize::new(0),
            suffix_count: AtomicU8::new(0),
        }
    }

    /// Build from a data buffer and slot metadata (clone path).
    #[expect(
        clippy::cast_possible_truncation,
        reason = "at most WIDTH (15) suffixes fit in a u8"
    )]
    #[inline]
    fn from_parts(data: Vec<u8>, metas: [SlotMeta; WIDTH]) -> Self {
        let ptr: *mut u8 = data.as_ptr().cast_mut();
        let used: usize = data.len();
        let count: u8 = metas.iter().filter(|m| m.has_suffix()).count() as u8;

        Self {
            slots: std::array::from_fn(|i: usize| AtomicU64::new(metas[i].pack())),
            data: UnsafeCell::new(data),
            data_ptr: AtomicPtr::new(ptr),
            used: AtomicUsize::new(used),
            suffix_count: AtomicU8::new(count),
        }
    }

    // ========================================================================
    //  Internal Access
    // ========================================================================

    /// Get a mutable reference to the backing buffer.
    ///
    /// # Safety
    ///
    /// Callers must have exclusive write access (a `&mut` bag or the leaf lock
    /// that publishes this bag). Readers only use `data_ptr` + `used`.
    #[inline(always)]
    #[expect(clippy::mut_from_ref, reason = "Interior mutability via UnsafeCell")]
    unsafe fn data_mut(&self) -> &mut Vec<u8> {
        // SAFETY: Caller guarantees exclusive write access.
        unsafe { &mut *self.data.get() }
    }

    #[inline(always)]
    fn load_meta(&self, slot: usize) -> SlotMeta {
        SlotMeta::unpack(self.slots[slot].load(Ordering::Acquire))
    }

    #[inline(always)]
    fn store_meta(&self, slot: usize, meta: SlotMeta) {
        self.slots[slot].store(meta.pack(), Ordering::Release);
    }

    /// Publish the new used length after bytes have been written.
    #[inline(always)]
    fn publish_used(&self, used: usize) {
        self.used.store(used, Ordering::Release);
    }

    // ========================================================================
    //  Capacity & Size
    // ========================================================================

    /// Return the current capacity of the data buffer.
    #[must_use]
    #[inline(always)]
    pub fn capacity(&self) -> usize {
        // SAFETY: capacity is stable for a published bag; writers that may
        // reallocate use `reserve` and update `data_ptr` under exclusivity.
        unsafe { (*self.data.get()).capacity() }
    }

    /// Return the number of bytes currently used.
    #[must_use]
    #[inline(always)]
    pub fn used(&self) -> usize {
        self.used.load(Ordering::Acquire)
    }

    /// Return the number of slots that have suffixes.
    #[must_use]
    #[inline(always)]
    pub fn count(&self) -> usize {
        self.suffix_count.load(Ordering::Acquire) as usize
    }

    /// Reserve additional capacity for suffix data.
    ///
    /// Growth is only permitted for an exclusively owned bag; appending to a
    /// published bag must stay within its capacity.
    #[inline(always)]
    pub fn reserve(&mut self, additional: usize) {
        // SAFETY: `&mut self` guarantees no live shared borrow of the bag.
        unsafe { self.reserve_locked(additional) };
    }

    /// Reserve additional capacity while holding the leaf lock.
    ///
    /// # Safety
    ///
    /// Caller must hold the leaf lock and the bag must not be visible to
    /// concurrent readers that hold `&SuffixBag` (its published form never
    /// grows).
    #[inline(always)]
    unsafe fn reserve_locked(&self, additional: usize) {
        // SAFETY: Caller guarantees exclusive logical write access.
        let data: &mut Vec<u8> = unsafe { self.data_mut() };
        data.reserve(additional);
        self.data_ptr
            .store(data.as_ptr().cast_mut(), Ordering::Release);
    }

    // ========================================================================
    //  Smart Assignment with Compaction
    // ========================================================================

    /// Append suffix data and update slot metadata + count.
    ///
    /// # Safety
    ///
    /// Caller must have exclusive ownership of the bag (no concurrent readers
    /// or writers).
    #[expect(clippy::cast_possible_truncation)]
    unsafe fn append_and_record_locked(&self, slot: usize, suffix: &[u8]) {
        // SAFETY: Caller guarantees exclusive access.
        let data: &mut Vec<u8> = unsafe { self.data_mut() };
        let offset: usize = data.len();

        debug_assert!(
            offset + suffix.len() <= data.capacity(),
            "published suffix bag reallocated"
        );

        data.extend_from_slice(suffix);
        self.publish_used(data.len());

        if !self.load_meta(slot).has_suffix() {
            self.suffix_count.fetch_add(1, Ordering::Relaxed);
        }

        self.store_meta(
            slot,
            SlotMeta {
                offset: offset as u32,
                len: suffix.len() as u16,
                pad: 0,
            },
        );
    }

    /// Assign a suffix with smart space management.
    ///
    /// Returns `true` if the suffix was stored within the existing buffer,
    /// `false` if the buffer had to grow.
    ///
    /// # Panics
    ///
    /// Panics if `suffix.len() > u16::MAX` (65535 bytes).
    pub fn try_assign(&mut self, slot: usize, suffix: &[u8]) -> bool {
        // SAFETY: `&mut self` guarantees exclusive ownership.
        unsafe { self.try_assign_locked(slot, suffix) }
    }

    /// Assign a suffix while holding the leaf lock.
    ///
    /// # Safety
    ///
    /// Caller must have exclusive ownership of the bag. A merely locked
    /// *published* leaf is not sufficient: this path may reallocate/compact,
    /// invalidating reader slices.
    #[expect(clippy::indexing_slicing, reason = "Checked access")]
    #[expect(clippy::cast_possible_truncation)]
    pub(crate) unsafe fn try_assign_locked(&self, slot: usize, suffix: &[u8]) -> bool {
        assert!(slot < WIDTH, "slot {slot} >= WIDTH {WIDTH}");

        let suffix_len: usize = suffix.len();

        assert!(
            u16::try_from(suffix_len).is_ok(),
            "suffix too long: {suffix_len} > {}",
            u16::MAX
        );

        let meta: SlotMeta = self.load_meta(slot);

        // Fast path: reuse existing slot space if new suffix fits. In-place
        // overwrite is only for an exclusively owned bag.
        if meta.has_suffix() && (suffix_len <= (meta.len as usize)) {
            let start: usize = meta.offset as usize;

            // SAFETY: Caller has exclusive access.
            let data: &mut Vec<u8> = unsafe { self.data_mut() };
            data[start..(start + suffix_len)].copy_from_slice(suffix);

            self.store_meta(
                slot,
                SlotMeta {
                    offset: meta.offset,
                    len: suffix_len as u16,
                    pad: 0,
                },
            );

            return true;
        }

        // SAFETY: Caller guarantees exclusive access for the rest of this
        // path (compaction/append may move bytes).
        unsafe {
            // Try append to end of buffer.
            if (self.used() + suffix_len) <= self.capacity() {
                self.append_and_record_locked(slot, suffix);
                return true;
            }

            // Compact and retry.
            self.compact_in_place_locked();

            if (self.used() + suffix_len) <= self.capacity() {
                self.append_and_record_locked(slot, suffix);
                return true;
            }

            // Still not enough space, grow (aborts on OOM).
            self.reserve_locked(suffix_len);
            self.append_and_record_locked(slot, suffix);
        }

        false // Buffer grew
    }

    // ========================================================================
    //  Slot Access
    // ========================================================================

    /// Check if a slot has a suffix.
    ///
    /// # Panics
    ///
    /// Panics if `slot >= 15`.
    #[must_use]
    #[inline(always)]
    pub fn has_suffix(&self, slot: usize) -> bool {
        assert!(slot < WIDTH, "slot {slot} >= WIDTH {WIDTH}");
        self.load_meta(slot).has_suffix()
    }

    /// Get the suffix for a slot, or `None` if no suffix.
    ///
    /// # Panics
    ///
    /// Panics if `slot >= 15`.
    #[must_use]
    #[inline(always)]
    pub fn get(&self, slot: usize) -> Option<&[u8]> {
        assert!(slot < WIDTH, "slot {slot} >= WIDTH {WIDTH}");

        let meta: SlotMeta = self.load_meta(slot);

        if !meta.has_suffix() {
            return None;
        }

        let start: usize = meta.offset as usize;
        let len: usize = meta.len as usize;
        let end: usize = start.checked_add(len)?;
        let used: usize = self.used.load(Ordering::Acquire);

        // A stale/torn read under OCC can produce an out-of-range pair.
        // Return None so the caller's OCC retry handles it safely.
        if end > used {
            return None;
        }

        let base: *const u8 = self.data_ptr.load(Ordering::Acquire);

        if base.is_null() {
            return None;
        }

        // SAFETY: `start + len <= used`, bytes below `used` are immutable to
        // concurrent readers (append-only writers write at or after `used`),
        // and `base` is valid for at least `used` bytes.
        Some(unsafe { slice::from_raw_parts(base.add(start), len) })
    }

    /// Get the suffix for a slot, or empty slice if no suffix.
    ///
    /// # Panics
    ///
    /// Panics if `slot >= 15`.
    #[must_use]
    #[inline(always)]
    pub fn get_or_empty(&self, slot: usize) -> &[u8] {
        self.get(slot).unwrap_or(&[])
    }

    // ========================================================================
    //  Suffix Assignment
    // ========================================================================

    /// Append-only suffix assignment for concurrent use.
    ///
    /// Unlike [`try_assign_in_place`](Self::try_assign_in_place), this method
    /// never overwrites existing suffix bytes. New data is always appended to
    /// the end of the buffer, so concurrent OCC readers reading old offsets
    /// see stable data. Old suffix space becomes dead until drain-and-rebuild.
    ///
    /// # Panics
    ///
    /// Panics if `slot >= 15` or if suffix length exceeds `u16::MAX`.
    #[inline]
    pub fn try_assign_append_only(&self, slot: usize, suffix: &[u8]) -> bool {
        assert!(slot < WIDTH, "slot {slot} >= WIDTH {WIDTH}");
        assert!(
            u16::try_from(suffix.len()).is_ok(),
            "suffix too long: {} > {}",
            suffix.len(),
            u16::MAX
        );

        self.try_append_suffix(slot, suffix)
    }

    /// Try to assign a suffix to a slot in-place, without growing the buffer.
    ///
    /// This method may overwrite existing suffix bytes in-place when the new
    /// suffix fits. It is NOT safe for concurrent use (readers may observe
    /// partially-written data). Use [`try_assign_append_only`](Self::try_assign_append_only)
    /// for the concurrent path.
    ///
    /// # Panics
    ///
    /// Panics if `slot >= 15` or if suffix length exceeds `u16::MAX`.
    #[inline]
    pub fn try_assign_in_place(&mut self, slot: usize, suffix: &[u8]) -> bool {
        // SAFETY: `&mut self` guarantees exclusive ownership.
        unsafe { self.try_assign_in_place_locked(slot, suffix) }
    }

    /// In-place assignment for the lock-protected path.
    ///
    /// # Safety
    ///
    /// Caller must have exclusive ownership of the bag.
    #[inline]
    #[expect(clippy::cast_possible_truncation, reason = "len checked above")]
    pub(crate) unsafe fn try_assign_in_place_locked(&self, slot: usize, suffix: &[u8]) -> bool {
        assert!(slot < WIDTH, "slot {slot} >= WIDTH {WIDTH}");
        assert!(
            u16::try_from(suffix.len()).is_ok(),
            "suffix too long: {} > {}",
            suffix.len(),
            u16::MAX
        );

        let meta: SlotMeta = self.load_meta(slot);

        if meta.has_suffix() && (suffix.len() <= (meta.len as usize)) {
            let start: usize = meta.offset as usize;

            // SAFETY: Caller has exclusive access.
            let data: &mut Vec<u8> = unsafe { self.data_mut() };
            data[start..(start + suffix.len())].copy_from_slice(suffix);

            self.store_meta(
                slot,
                SlotMeta {
                    offset: meta.offset,
                    len: suffix.len() as u16,
                    pad: 0,
                },
            );

            return true;
        }

        self.try_append_suffix(slot, suffix)
    }

    /// Append suffix data to the end of the buffer if capacity allows.
    ///
    /// Shared by `try_assign_append_only` and the fallback path of
    /// `try_assign_in_place`.
    #[inline]
    #[expect(clippy::cast_possible_truncation, reason = "offset and len checked")]
    fn try_append_suffix(&self, slot: usize, suffix: &[u8]) -> bool {
        let new_offset: usize = self.used();
        let new_used: usize = new_offset + suffix.len();

        if new_used > self.capacity() {
            return false;
        }

        if !suffix.is_empty() {
            let base: *mut u8 = self.data_ptr.load(Ordering::Relaxed);

            if base.is_null() {
                return false;
            }

            // Write only into the unpublished spare capacity. Forming
            // `&mut [u8]` here would alias lock-free readers' published
            // slices derived from `data_ptr`.
            // SAFETY: `new_used <= capacity`; bytes below `new_offset` are
            // published and untouched, and readers never read at or beyond
            // `used` until `publish_used` runs below.
            unsafe {
                StdPtr::copy_nonoverlapping(suffix.as_ptr(), base.add(new_offset), suffix.len());
            }

            // Keep `Vec::len` in sync. This retags only the Vec header, not
            // the heap buffer that readers borrow.
            // SAFETY: [0, new_used) is initialized and within capacity.
            unsafe {
                (*self.data.get()).set_len(new_used);
            }
        }

        self.publish_used(new_used);

        if !self.load_meta(slot).has_suffix() {
            self.suffix_count.fetch_add(1, Ordering::Relaxed);
        }

        self.store_meta(
            slot,
            SlotMeta {
                offset: new_offset as u32,
                len: suffix.len() as u16,
                pad: 0,
            },
        );

        true
    }

    /// Assign a suffix to a slot.
    ///
    /// # Panics
    ///
    /// Panics if `slot >= 15` or if suffix length exceeds `u16::MAX`.
    #[inline]
    pub fn assign(&mut self, slot: usize, suffix: &[u8]) {
        // SAFETY: `&mut self` guarantees exclusive ownership.
        unsafe { self.assign_locked(slot, suffix) };
    }

    /// Assign a suffix while holding the leaf lock.
    ///
    /// # Safety
    ///
    /// Caller must have exclusive ownership of the bag (an unpublished leaf
    /// or a freshly built drain destination), not merely the leaf lock.
    #[inline]
    #[expect(clippy::cast_possible_truncation, reason = "offset and len checked")]
    pub(crate) unsafe fn assign_locked(&self, slot: usize, suffix: &[u8]) {
        assert!(slot < WIDTH, "slot {slot} >= WIDTH {WIDTH}");
        assert!(
            u16::try_from(suffix.len()).is_ok(),
            "suffix too long: {} > {}",
            suffix.len(),
            u16::MAX
        );

        // SAFETY: Caller has exclusive access.
        let data: &mut Vec<u8> = unsafe { self.data_mut() };
        let offset: usize = data.len();
        data.extend_from_slice(suffix);

        if !StdPtr::eq(data.as_ptr(), self.data_ptr.load(Ordering::Relaxed)) {
            self.data_ptr
                .store(data.as_ptr().cast_mut(), Ordering::Release);
        }

        self.publish_used(data.len());

        if !self.load_meta(slot).has_suffix() {
            self.suffix_count.fetch_add(1, Ordering::Relaxed);
        }

        self.store_meta(
            slot,
            SlotMeta {
                offset: offset as u32,
                len: suffix.len() as u16,
                pad: 0,
            },
        );
    }

    /// Clear the suffix for a slot.
    ///
    /// # Panics
    ///
    /// Panics if `slot >= 15`.
    #[inline(always)]
    pub fn clear(&mut self, slot: usize) {
        // SAFETY: `&mut self` guarantees exclusive ownership.
        unsafe { self.clear_locked(slot) };
    }

    /// Clear a slot's suffix while holding the leaf lock.
    ///
    /// # Safety
    ///
    /// Caller must hold the leaf lock. This only updates atomics, so it is
    /// safe with concurrent readers.
    #[inline(always)]
    pub(crate) unsafe fn clear_locked(&self, slot: usize) {
        assert!(slot < WIDTH, "slot {slot} >= WIDTH {WIDTH}");

        if self.load_meta(slot).has_suffix() {
            self.suffix_count.fetch_sub(1, Ordering::Relaxed);
        }

        self.store_meta(slot, SlotMeta::EMPTY);
    }

    // ========================================================================
    //  Comparison Helpers
    // ========================================================================

    /// Check if a slot's suffix equals the given suffix.
    #[must_use]
    #[inline(always)]
    pub fn suffix_equals(&self, slot: usize, suffix: &[u8]) -> bool {
        self.get(slot).is_some_and(|stored: &[u8]| stored == suffix)
    }

    /// Compare a slot's suffix with the given suffix.
    #[must_use]
    #[inline(always)]
    pub fn suffix_compare(&self, slot: usize, suffix: &[u8]) -> Option<std::cmp::Ordering> {
        self.get(slot).map(|stored: &[u8]| stored.cmp(suffix))
    }
}

impl Debug for SuffixBag {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        f.debug_struct("SuffixBag")
            .field("used", &self.used())
            .field("count", &self.count())
            .finish_non_exhaustive()
    }
}

impl Default for SuffixBag {
    #[inline(always)]
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
//  SuffixBagCell (internal concurrent wrapper)
// ============================================================================

/// Interior-mutable wrapper for [`SuffixBag`] used inside leaf nodes.
///
/// Provides `UnsafeCell`-based access so that writers (under the leaf lock)
/// can mutate through `&self` while readers form `&SuffixBag` for read-only
/// access under the OCC protocol. Writers use [`as_writer`](Self::as_writer);
/// the returned reference is shared, and every mutation goes through the
/// bag's own atomic/interior-mutable fields. A `&mut SuffixBag` is never
/// created while readers may hold `&SuffixBag`.
///
/// This type is NOT publicly exported. It exists solely to eliminate the
/// `&mut SuffixBag` / `&SuffixBag` aliasing UB in the tree's concurrent paths.
pub struct SuffixBagCell {
    inner: UnsafeCell<SuffixBag>,
}

// SAFETY: SuffixBagCell is Send because SuffixBag is Send. We intentionally do
// NOT implement Sync. The tree accesses SuffixBagCell only through raw pointers
// from AtomicPtr, forming short-lived references locally. Writers hold the leaf
// lock; readers validate via OCC version checks.
unsafe impl Send for SuffixBagCell {}

impl SuffixBagCell {
    /// Create a new cell wrapping a default `SuffixBag`.
    #[inline(always)]
    pub(crate) fn new() -> Self {
        Self {
            inner: UnsafeCell::new(SuffixBag::new()),
        }
    }

    /// Create a cell from an existing `SuffixBag`.
    #[inline(always)]
    pub(crate) const fn from_bag(bag: SuffixBag) -> Self {
        Self {
            inner: UnsafeCell::new(bag),
        }
    }

    /// Get a shared reference for read-only access.
    ///
    /// # Safety
    ///
    /// Caller must ensure the bag is not being replaced or freed concurrently.
    /// Readers rely on the leaf's OCC protocol; all bag metadata is atomic.
    #[inline(always)]
    pub(crate) unsafe fn as_ref(&self) -> &SuffixBag {
        // SAFETY: Caller guarantees the cell is live.
        unsafe { &*self.inner.get() }
    }

    /// Get shared access for mutation under the leaf lock.
    ///
    /// # Safety
    ///
    /// Caller must hold the leaf lock (exclusive write access). The returned
    /// shared reference only exposes the bag's interior-mutable writer API.
    #[inline(always)]
    pub(crate) unsafe fn as_writer(&self) -> &SuffixBag {
        // SAFETY: Caller holds the leaf lock; all writes go through atomics and
        // `UnsafeCell` fields inside the bag.
        unsafe { &*self.inner.get() }
    }

    /// Exclusive access for single-threaded tests.
    ///
    /// # Safety
    ///
    /// Caller must have exclusive access and no concurrent readers.
    #[cfg(test)]
    #[expect(
        clippy::mut_from_ref,
        reason = "Test-only exclusive access through UnsafeCell"
    )]
    pub(crate) unsafe fn as_mut_for_test(&self) -> &mut SuffixBag {
        // SAFETY: Caller guarantees exclusive access.
        unsafe { &mut *self.inner.get() }
    }
}

impl Default for SuffixBagCell {
    #[inline(always)]
    fn default() -> Self {
        Self::new()
    }
}

impl Debug for SuffixBagCell {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        // SAFETY: Debug is called from single-threaded contexts (tests, logging).
        let bag: &SuffixBag = unsafe { &*self.inner.get() };

        f.debug_struct("SuffixBagCell").field("inner", bag).finish()
    }
}

// ============================================================================
//  Tests
// ============================================================================

#[cfg(test)]
#[expect(clippy::indexing_slicing)]
mod unit_tests;
