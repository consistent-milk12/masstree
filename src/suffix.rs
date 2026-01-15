//! Filepath: src/suffix.rs
//!
//! Suffix storage for keys longer than 8 bytes.
//!
//! When a key is longer than 8 bytes, the first 8 bytes are stored as `ikey0`
//! and the remaining bytes are stored in a [`SuffixBag`].

use crate::{AllocError, AllocResult, TreePermutation};

mod clone;
mod cmp;
mod compact;
mod inline;
mod sidecar;

use cmp::CompareSuffix;
pub use inline::InlineSuffixBag;

pub use sidecar::{SideCarUtils, SuffixSidecar};

/// Initial capacity for suffix storage (matches C++ `INITIAL_KSUF_CAPACITY`).
const INITIAL_CAPACITY: usize = 128;

// ============================================================================
//  SlotMeta
// ============================================================================

/// Metadata for a single slot's suffix.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
struct SlotMeta {
    /// Offset into the data buffer (`u32::MAX` if no suffix).
    offset: u32,

    /// Length of the suffix.
    len: u16,

    /// Padding for alignment (unused).
    _pad: u16,
}

impl SlotMeta {
    /// Sentinel value indicating no suffix stored.
    const EMPTY: Self = Self {
        offset: u32::MAX,
        len: 0,
        _pad: 0,
    };

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
/// Each leaf node can have at most `WIDTH` suffixes (one per slot).
/// Suffixes are stored contiguously in a growable buffer.
///
/// # Memory Layout
///
/// ```text
/// SuffixBag {
///     slots: [(offset, len); WIDTH],  // Per-slot metadata
///     data: [u8],                      // Contiguous suffix bytes
/// }
/// ```
///
/// # Growth Strategy
///
/// When a new suffix doesn't fit:
/// 1. Calculate total size of active suffixes + new suffix
/// 2. Allocate new buffer with 2x capacity (at least needed size)
/// 3. Copy only active suffixes (garbage collection)
/// 4. Assign new suffix
///
/// # Type Parameters
///
/// * `WIDTH` - Number of slots (must match the leaf node's WIDTH)
#[derive(Debug)]
pub struct SuffixBag<const WIDTH: usize> {
    /// Per-slot metadata: (offset, length) pairs.
    slots: [SlotMeta; WIDTH],

    /// Contiguous suffix data buffer.
    data: Vec<u8>,

    /// Cached count of slots with suffixes (avoids O(WIDTH) recount).
    suffix_count: u8,
}

impl<const WIDTH: usize> SuffixBag<WIDTH> {
    /// Compile-time assert that WIDTH fits in u8 for `suffix_count`.
    const ASSERT_WIDTH_FITS_U8: () = assert!(WIDTH <= 255, "WDITH mst be <= 255 to fit in u8");

    // ========================================================================
    //  Constructor
    // ========================================================================

    /// Create a new suffix bag with initial capacity.
    #[must_use]
    #[inline(always)]
    pub fn new() -> Self {
        // Force compile-time evaluation of WIDTH assertion
        let () = Self::ASSERT_WIDTH_FITS_U8;

        Self {
            slots: [SlotMeta::EMPTY; WIDTH],
            data: Vec::with_capacity(INITIAL_CAPACITY),
            suffix_count: 0,
        }
    }

    /// Create a new suffix bag with specified capacity.
    #[must_use]
    #[inline(always)]
    pub fn with_capacity(capacity: usize) -> Self {
        // Force compile-time evaluation of WIDTH assertion
        let () = Self::ASSERT_WIDTH_FITS_U8;

        Self {
            slots: [SlotMeta::EMPTY; WIDTH],
            data: Vec::with_capacity(capacity),
            suffix_count: 0,
        }
    }

    // ========================================================================
    //  Capacity & Size
    // ========================================================================

    /// Return the current capacity of the data buffer.
    #[must_use]
    #[inline(always)]
    pub const fn capacity(&self) -> usize {
        self.data.capacity()
    }

    /// Return the number of bytes currently used.
    #[must_use]
    #[inline(always)]
    pub const fn used(&self) -> usize {
        self.data.len()
    }

    /// Return the number of slots that have suffixes.
    ///
    /// This is now O(1) - the count is cached and maintained incrementally.
    #[must_use]
    #[inline(always)]
    pub const fn count(&self) -> usize {
        self.suffix_count as usize
    }

    // ========================================================================
    //  Fallible Operations
    // ========================================================================

    /// Try to create a new suffix bag with initial capacity.
    ///
    /// # Returns
    ///
    /// * `Ok(bag)` - Successfully allocated bag
    /// * `Err(AllocError)` - Could not allocate capacity
    ///
    /// # Errors
    ///
    /// Upon allocation failure.
    #[inline(always)]
    pub fn try_with_capacity(capacity: usize) -> AllocResult<Self> {
        let mut data: Vec<u8> = Vec::new();
        data.try_reserve(capacity)
            .map_err(|_| AllocError::for_suffix(capacity))?;

        Ok(Self {
            slots: [SlotMeta::EMPTY; WIDTH],
            data,
            suffix_count: 0,
        })
    }

    /// Try to assign a suffix, returning error if allocation fails.
    ///
    /// Attempts to store `suffix` at the given `slot` index. Uses several
    /// optimization strategies in order:
    /// 1. Re-use existing slot if new suffix fits in allocated space
    /// 2. Append to data buffer if capacity allows
    /// 3. Compact in-place to reclaim fragmented space
    /// 4. Grow the buffer if compaction is insufficient
    ///
    /// # Arguments
    ///
    /// * `slot` - Slot index, must be `< WIDTH`
    /// * `suffix` - Suffix bytes to store
    ///
    /// # Errors
    ///
    /// Returns [`AllocError`] if buffer growth fails due to memory exhaustion.
    ///
    /// # Panics
    ///
    /// Panics if `suffix.len() > u16::MAX` (65535 bytes).
    #[expect(clippy::indexing_slicing, reason = "Checked access")]
    pub fn try_assign(&mut self, slot: usize, suffix: &[u8]) -> AllocResult<()> {
        debug_assert!(slot < WIDTH, "slot {slot} >= WIDTH {WIDTH}");

        let suffix_len: usize = suffix.len();

        assert!(
            u16::try_from(suffix_len).is_ok(),
            "suffix too long: {suffix_len} > {}",
            u16::MAX
        );

        let meta: SlotMeta = self.slots[slot];

        // Fast Path 1: Re-use existing slot if new suffix fits
        if meta.has_suffix() && (suffix_len <= (meta.len as usize)) {
            let start: usize = meta.offset as usize;
            self.data[start..(start + suffix_len)].copy_from_slice(suffix);

            #[expect(clippy::cast_possible_truncation)]
            {
                self.slots[slot] = SlotMeta {
                    offset: meta.offset,
                    len: suffix_len as u16,
                    _pad: 0,
                };
            }

            // Count unchanged, slot already had suffix
            return Ok(());
        }

        // Fast Path 2: Append if there's room
        let new_offset: usize = self.data.len();

        if (new_offset + suffix_len) <= self.data.capacity() {
            self.data.extend_from_slice(suffix);

            // Update count if this is a new suffix
            if !meta.has_suffix() {
                self.suffix_count += 1;
            }

            #[expect(clippy::cast_possible_truncation)]
            {
                self.slots[slot] = SlotMeta {
                    offset: new_offset as u32,
                    len: suffix_len as u16,
                    _pad: 0,
                };
            }

            return Ok(());
        }

        // Slow Path: Need more space, try compacting first
        self.compact_in_place();

        let new_offset: usize = self.data.len();

        if (new_offset + suffix_len) <= self.data.capacity() {
            // Compaction freed enough space
            self.data.extend_from_slice(suffix);

            // Note: After compact_in_place, we need to re-check if slot had suffix
            // The slot metadata was updated by compact_in_place
            if !self.slots[slot].has_suffix() {
                self.suffix_count += 1;
            }

            #[expect(clippy::cast_possible_truncation)]
            {
                self.slots[slot] = SlotMeta {
                    offset: new_offset as u32,
                    len: suffix_len as u16,
                    _pad: 0,
                };
            }

            return Ok(());
        }

        // Still not enough space - must grow
        self.data
            .try_reserve(suffix_len)
            .map_err(|_| AllocError::for_suffix(suffix_len))?;

        let new_offset: usize = self.data.len();
        self.data.extend_from_slice(suffix);

        if !self.slots[slot].has_suffix() {
            self.suffix_count += 1;
        }

        #[expect(clippy::cast_possible_truncation)]
        {
            self.slots[slot] = SlotMeta {
                offset: new_offset as u32,
                len: suffix_len as u16,
                _pad: 0,
            };
        }

        Ok(())
    }

    /// Try to grow capacity, returning error on allocation failure.
    ///
    /// # Errors
    ///
    /// Upon allocation failure.
    pub fn try_reserve(&mut self, additional: usize) -> AllocResult<()> {
        self.data
            .try_reserve(additional)
            .map_err(|_| AllocError::for_suffix(additional))
    }

    // ========================================================================
    //  Slot Access
    // ========================================================================

    /// Check if a slot has a suffix.
    ///
    /// # Panics
    ///
    /// Panics if `slot >= WIDTH`.
    #[must_use]
    #[inline(always)]
    #[expect(
        clippy::indexing_slicing,
        reason = "Slot bounds checked via caller contract"
    )]
    pub fn has_suffix(&self, slot: usize) -> bool {
        debug_assert!(slot < WIDTH, "slot {slot} >= WIDTH {WIDTH}");
        self.slots[slot].has_suffix()
    }

    /// Get the suffix for a slot, or `None` if no suffix.
    ///
    /// # Panics
    ///
    /// Panics if `slot >= WIDTH`.
    #[must_use]
    #[inline(always)]
    #[expect(
        clippy::indexing_slicing,
        reason = "Bounds checked via debug_assert and invariant maintenance"
    )]
    pub fn get(&self, slot: usize) -> Option<&[u8]> {
        debug_assert!(slot < WIDTH, "slot {slot} >= WIDTH {WIDTH}");

        let meta: SlotMeta = self.slots[slot];

        if !meta.has_suffix() {
            return None;
        }

        let start: usize = meta.offset as usize;
        let end: usize = start + meta.len as usize;

        // INVARIANT: Valid metadata points to valid data range.
        debug_assert!(
            end <= self.data.len(),
            "suffix metadata points past data end: {end} > {}",
            self.data.len()
        );

        Some(&self.data[start..end])
    }

    /// Get the suffix for a slot, or empty slice if no suffix.
    ///
    /// # Panics
    ///
    /// Panics if `slot >= WIDTH`.
    #[must_use]
    #[inline(always)]
    pub fn get_or_empty(&self, slot: usize) -> &[u8] {
        self.get(slot).unwrap_or(&[])
    }

    // ========================================================================
    //  Suffix Assignment
    // ========================================================================

    /// Try to assign a suffix to a slot in-place, without growing the buffer.
    ///
    /// This is an optimization for the common case where we hold the lock
    /// and can mutate in place. It avoids the clone + box allocation overhead.
    ///
    /// # Returns
    ///
    /// - `true` if the suffix was assigned successfully (fits in existing capacity)
    /// - `false` if the suffix doesn't fit and caller should reallocate
    ///
    /// # Fast Paths (like C++ `stringbag::assign`)
    ///
    /// 1. **Reuse existing slot**: If the new suffix fits in the old suffix's space
    /// 2. **Append to end**: If there's room in the buffer
    ///
    /// # Panics
    ///
    /// Panics if `slot >= WIDTH` or if suffix length exceeds `u16::MAX`.
    #[inline]
    #[expect(
        clippy::indexing_slicing,
        reason = "Slot bounds checked via debug_assert"
    )]
    pub fn try_assign_in_place(&mut self, slot: usize, suffix: &[u8]) -> bool {
        debug_assert!(slot < WIDTH, "slot {slot} >= WIDTH {WIDTH}");
        assert!(
            u16::try_from(suffix.len()).is_ok(),
            "suffix too long: {} > {}",
            suffix.len(),
            u16::MAX
        );

        let meta: SlotMeta = self.slots[slot];

        // Fast Path 1: Reuse existing slot if new suffix fits in old space
        if meta.has_suffix() && (suffix.len() <= (meta.len as usize)) {
            let start: usize = meta.offset as usize;
            self.data[start..(start + suffix.len())].copy_from_slice(suffix);

            #[expect(clippy::cast_possible_truncation, reason = "len checked above")]
            {
                self.slots[slot] = SlotMeta {
                    offset: meta.offset,
                    len: suffix.len() as u16,
                    _pad: 0,
                };
            }

            // Count unchanged, slot already had suffix
            return true;
        }

        // Fast Path 2: Append to end if there's room
        let new_offset: usize = self.data.len();

        if (new_offset + suffix.len()) <= self.data.capacity() {
            // Update count if this is a new suffix
            if !meta.has_suffix() {
                self.suffix_count += 1;
            }

            self.data.extend_from_slice(suffix);

            #[expect(clippy::cast_possible_truncation, reason = "offset and len checked")]
            {
                self.slots[slot] = SlotMeta {
                    offset: new_offset as u32,
                    len: suffix.len() as u16,
                    _pad: 0,
                };
            }

            return true;
        }

        // Slow Path: doesn't fit, caller should realloc
        false
    }

    /// Assign a suffix to a slot.
    ///
    /// This always appends to the data buffer. If the buffer is full,
    /// it will grow automatically. Old suffix data is not reclaimed
    /// until [`compact()`](Self::compact) is called.
    ///
    /// # Panics
    ///
    /// Panics if `slot >= WIDTH` or if suffix length exceeds `u16::MAX`.
    #[inline]
    #[expect(
        clippy::indexing_slicing,
        reason = "Slot bounds checked via debug_assert"
    )]
    pub fn assign(&mut self, slot: usize, suffix: &[u8]) {
        debug_assert!(slot < WIDTH, "slot {slot} >= WIDTH {WIDTH}");
        assert!(
            u16::try_from(suffix.len()).is_ok(),
            "suffix too long: {} > {}",
            suffix.len(),
            u16::MAX
        );

        let meta: SlotMeta = self.slots[slot];

        // Update count if this is a new suffix
        if !meta.has_suffix() {
            self.suffix_count += 1;
        }

        let offset: usize = self.data.len();
        self.data.extend_from_slice(suffix);

        // Safe casts: offset fits in u32 (Vec max is isize::MAX), len checked above
        #[expect(
            clippy::cast_possible_truncation,
            reason = "offset bounded by Vec capacity, len checked above"
        )]
        {
            self.slots[slot] = SlotMeta {
                offset: offset as u32,
                len: suffix.len() as u16,
                _pad: 0,
            };
        }
    }

    /// Clear the suffix for a slot.
    ///
    /// This marks the slot as having no suffix but does NOT reclaim
    /// the data buffer space. Call [`compact()`](Self::compact) to reclaim space.
    ///
    /// # Panics
    ///
    /// Panics if `slot >= WIDTH`.
    #[inline(always)]
    #[expect(
        clippy::indexing_slicing,
        reason = "Slot bounds checked via debug_assert"
    )]
    pub fn clear(&mut self, slot: usize) {
        debug_assert!(slot < WIDTH, "slot {slot} >= WIDTH {WIDTH}");

        if self.slots[slot].has_suffix() {
            self.suffix_count -= 1;
        }

        self.slots[slot] = SlotMeta::EMPTY;
    }

    // ========================================================================
    //  Comparison Helpers
    // ========================================================================

    /// Check if a slot's suffix equals the given suffix.
    ///
    /// Uses word-aligned comparison for suffixes >= 8 bytes.
    ///
    /// # Returns
    ///
    /// - `true` if suffixes match exactly
    /// - `false` if slot has no suffix or suffixes differ
    #[must_use]
    #[inline(always)]
    pub fn suffix_equals(&self, slot: usize, suffix: &[u8]) -> bool {
        self.get(slot)
            .is_some_and(|stored: &[u8]| CompareSuffix::fast_slice_eq(stored, suffix))
    }

    /// Compare a slot's suffix with the given suffix.
    ///
    /// Uses word-aligned comparison for suffixes >= 8 bytes.
    ///
    /// # Returns
    ///
    /// - `Some(Ordering)` if slot has a suffix
    /// - `None` if slot has no suffix
    #[must_use]
    #[inline(always)]
    pub fn suffix_compare(&self, slot: usize, suffix: &[u8]) -> Option<std::cmp::Ordering> {
        self.get(slot)
            .map(|stored: &[u8]| CompareSuffix::fast_slice_cmp(stored, suffix))
    }
}

impl<const WIDTH: usize> Default for SuffixBag<WIDTH> {
    #[inline(always)]
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
//  Tests
// ============================================================================

#[cfg(test)]
#[expect(clippy::indexing_slicing)]
mod unit_tests;
