//! Filepath: src/suffix.rs
//!
//! Suffix storage for keys longer than 8 bytes.
//!
//! When a key is longer than 8 bytes, the first 8 bytes are stored as `ikey0`
//! and the remaining bytes are stored in a [`SuffixBag`].

use std::cell::{Cell, UnsafeCell};
use std::fmt::{self, Debug, Formatter};
use std::marker::PhantomData;
use std::mem::ManuallyDrop;
use std::ptr as StdPtr;
use std::slice;
use std::sync::atomic::{AtomicPtr, AtomicU64, AtomicUsize, Ordering};

use crate::TreePermutation;

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
//  SuffixBag
// ============================================================================

/// Append-only storage for key suffixes, owning one raw byte buffer.
///
/// The pointer and capacity never change once the bag is published in a leaf.
/// The single writer appends to spare capacity before publishing `used` and the
/// slot metadata with Release. Readers retain immutable slices of earlier bytes.
/// Growth requires exclusive ownership, not merely the published leaf's lock.
///
/// This type is deliberately not `Sync`: sharing it across threads requires the
/// tree's single-writer protocol even though its metadata is atomic.
pub struct SuffixBag {
    slots: [AtomicU64; WIDTH],
    data_ptr: AtomicPtr<u8>,
    capacity: AtomicUsize,
    used: AtomicUsize,
    _single_writer: PhantomData<Cell<()>>,
}

static_assertions::const_assert_eq!(
    std::mem::size_of::<SuffixBag>(),
    WIDTH * std::mem::size_of::<AtomicU64>() + 3 * std::mem::size_of::<usize>()
);
static_assertions::assert_not_impl_any!(SuffixBag: Sync);

impl SuffixBag {
    /// Create an empty bag with the default buffer capacity.
    #[must_use]
    #[inline]
    pub fn new() -> Self {
        Self::with_capacity(INITIAL_CAPACITY)
    }

    /// Create an empty bag with the given buffer capacity.
    #[must_use]
    #[inline]
    pub fn with_capacity(capacity: usize) -> Self {
        Self::from_vec(Vec::with_capacity(capacity))
    }

    /// Take ownership of an empty vector's buffer without copying it.
    ///
    /// # Panics
    ///
    /// Panics if `data` is not empty.
    #[must_use]
    #[inline]
    pub fn from_vec(data: Vec<u8>) -> Self {
        assert!(data.is_empty(), "from_vec expects empty Vec");
        let mut data = ManuallyDrop::new(data);
        Self {
            slots: [const { AtomicU64::new(SlotMeta::EMPTY_PACKED) }; WIDTH],
            data_ptr: AtomicPtr::new(data.as_mut_ptr()),
            capacity: AtomicUsize::new(data.capacity()),
            used: AtomicUsize::new(0),
            _single_writer: PhantomData,
        }
    }

    #[inline(always)]
    fn load_meta(&self, slot: usize) -> SlotMeta {
        SlotMeta::unpack(self.slots[slot].load(Ordering::Acquire))
    }

    #[inline(always)]
    fn store_meta(&self, slot: usize, meta: SlotMeta) {
        self.slots[slot].store(meta.pack(), Ordering::Release);
    }

    /// Return the capacity of the owned buffer.
    #[must_use]
    #[inline(always)]
    pub fn capacity(&self) -> usize {
        self.capacity.load(Ordering::Relaxed)
    }

    /// Return the published byte count, including obsolete suffixes.
    #[must_use]
    #[inline(always)]
    pub fn used(&self) -> usize {
        self.used.load(Ordering::Acquire)
    }

    /// Reserve additional bytes on an exclusively owned bag.
    #[inline]
    pub fn reserve(&mut self, additional: usize) {
        // SAFETY: The exclusive borrow rules out outstanding reader slices.
        unsafe { self.reserve_locked(additional) };
    }

    /// Grow a buffer belonging to an unpublished, exclusively owned leaf.
    ///
    /// # Safety
    ///
    /// No reader or other writer may access the bag or its slices. Holding a
    /// published leaf's lock does not satisfy this requirement.
    pub(crate) unsafe fn reserve_locked(&self, additional: usize) {
        let used = self.used();
        let required = used
            .checked_add(additional)
            .expect("suffix capacity overflow");
        if required <= self.capacity() {
            return;
        }

        // SAFETY: This bag uniquely owns the Vec allocation and [0, used) is
        // initialized. Exclusivity excludes reader slices. ManuallyDrop leaves
        // ownership with the bag even if reserve panics before reallocating.
        let mut data = ManuallyDrop::new(unsafe {
            Vec::from_raw_parts(self.data_ptr.load(Ordering::Relaxed), used, self.capacity())
        });
        data.reserve(additional);
        self.data_ptr.store(data.as_mut_ptr(), Ordering::Release);
        self.capacity.store(data.capacity(), Ordering::Relaxed);
    }

    /// Check whether a slot has a suffix. Panics if `slot >= 15`.
    #[must_use]
    #[inline(always)]
    pub fn has_suffix(&self, slot: usize) -> bool {
        self.load_meta(slot).has_suffix()
    }

    /// Read a suffix. Panics if `slot >= 15`.
    #[must_use]
    #[inline(always)]
    pub fn get(&self, slot: usize) -> Option<&[u8]> {
        let meta = self.load_meta(slot);
        if !meta.has_suffix() {
            return None;
        }
        let start = meta.offset as usize;
        let len = meta.len as usize;
        if start.checked_add(len)? > self.used() {
            return None;
        }
        let base = self.data_ptr.load(Ordering::Acquire);
        // SAFETY: Metadata covers initialized immutable bytes within `used`.
        // The allocation lives as long as the bag and cannot grow while shared.
        // Even a zero-capacity Vec supplies a non-null aligned dangling pointer
        // valid for the zero-length slice at offset zero.
        Some(unsafe { slice::from_raw_parts(base.add(start), len) })
    }

    /// Read a suffix, or an empty slice for an empty slot.
    #[must_use]
    #[inline(always)]
    pub fn get_or_empty(&self, slot: usize) -> &[u8] {
        self.get(slot).unwrap_or(&[])
    }

    /// Append without changing existing bytes or growing the buffer.
    ///
    /// Returns false without changing the bag if capacity is insufficient.
    /// The bag is not Sync; the tree serializes callers under the leaf lock.
    ///
    /// # Panics
    ///
    /// Panics if `slot >= 15` or the suffix exceeds `u16::MAX` bytes.
    #[inline]
    pub fn try_assign_append_only(&self, slot: usize, suffix: &[u8]) -> bool {
        assert!(slot < WIDTH, "slot {slot} >= WIDTH {WIDTH}");
        assert!(u16::try_from(suffix.len()).is_ok(), "suffix too long");
        self.try_append_suffix(slot, suffix)
    }

    #[inline]
    #[expect(clippy::cast_possible_truncation, reason = "offset and length checked")]
    fn try_append_suffix(&self, slot: usize, suffix: &[u8]) -> bool {
        let offset = self.used();
        let Some(used) = offset.checked_add(suffix.len()) else {
            return false;
        };
        if used > self.capacity() || u32::try_from(offset).is_err() {
            return false;
        }
        // SAFETY: A single writer owns spare capacity [offset, used); it does
        // not overlap any published slice, including `suffix` if borrowed from
        // this bag. Neither the pointer nor capacity changes on this path.
        unsafe {
            StdPtr::copy_nonoverlapping(
                suffix.as_ptr(),
                self.data_ptr.load(Ordering::Relaxed).add(offset),
                suffix.len(),
            );
        }
        self.used.store(used, Ordering::Release);
        self.store_meta(
            slot,
            SlotMeta {
                offset: offset as u32,
                len: suffix.len() as u16,
                pad: 0,
            },
        );
        true
    }

    /// Assign a suffix on an exclusively owned bag, growing geometrically.
    ///
    /// # Panics
    ///
    /// Panics if `slot >= 15`, the suffix exceeds `u16::MAX`, or the offset
    /// exceeds `u32::MAX`.
    #[inline]
    pub fn assign(&mut self, slot: usize, suffix: &[u8]) {
        // SAFETY: The exclusive borrow rules out readers and other writers.
        unsafe { self.assign_locked(slot, suffix) };
    }

    /// Assign during unpublished node initialization.
    ///
    /// # Safety
    ///
    /// Caller exclusively owns the bag and all its bytes. A merely locked
    /// published leaf is insufficient: growth invalidates existing slices.
    #[inline]
    pub(crate) unsafe fn assign_locked(&self, slot: usize, suffix: &[u8]) {
        assert!(slot < WIDTH, "slot {slot} >= WIDTH {WIDTH}");
        assert!(u16::try_from(suffix.len()).is_ok(), "suffix too long");
        assert!(u32::try_from(self.used()).is_ok(), "suffix offset overflow");
        if self.try_append_suffix(slot, suffix) {
            return;
        }
        let used = self.used();
        let required = used
            .checked_add(suffix.len())
            .expect("suffix capacity overflow");
        let target = required.saturating_mul(2).max(INITIAL_CAPACITY);
        // SAFETY: Caller guarantees exclusive ownership, including all slices.
        unsafe { self.reserve_locked(target - used) };
        assert!(self.try_append_suffix(slot, suffix));
    }

    /// Clear metadata without reclaiming bytes. Panics if `slot >= 15`.
    #[inline(always)]
    pub fn clear(&mut self, slot: usize) {
        // SAFETY: The exclusive borrow prevents any concurrent writer.
        unsafe { self.clear_locked(slot) };
    }

    /// Clear a suffix under the leaf lock, leaving reader-visible bytes intact.
    ///
    /// # Safety
    ///
    /// Caller must hold the leaf lock (or exclusively own the bag).
    #[inline(always)]
    pub(crate) unsafe fn clear_locked(&self, slot: usize) {
        self.store_meta(slot, SlotMeta::EMPTY);
    }
}

impl Drop for SuffixBag {
    fn drop(&mut self) {
        // SAFETY: The bag is the sole owner of this Vec allocation, including
        // the original capacity and provenance. Exclusive drop excludes readers.
        // Bytes have no destructors, so a zero length suffices to deallocate.
        unsafe {
            drop(Vec::from_raw_parts(
                *self.data_ptr.get_mut(),
                0,
                *self.capacity.get_mut(),
            ));
        }
    }
}

impl Debug for SuffixBag {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        f.debug_struct("SuffixBag")
            .field("used", &self.used())
            .field("capacity", &self.capacity())
            .finish_non_exhaustive()
    }
}

impl Default for SuffixBag {
    #[inline]
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
        // the uniquely owned raw buffer inside the bag.
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
