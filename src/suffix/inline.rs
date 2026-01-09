use std::cmp::Ordering;

use super::{AllocResult, CompareSuffix, SuffixBag, TreePermutation};

const U16_MAX: usize = u16::MAX as usize;

// ============================================================================
//  InlineSlotMeta
// ============================================================================

/// Metadata for a single slot's suffix in inline storage.
///
/// Uses `u16` for offset to keep metadata compact (4 bytes per slot).
/// Maximum inline capacity is 65535 bytes.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
struct InlineSlotMeta {
    /// Offset into the data buffer (`u16::MAX` if no suffix).
    offset: u16,

    /// Length of the suffix.
    len: u16,
}

impl InlineSlotMeta {
    /// Sentinel value indicating no suffix stored.
    const EMPTY: Self = Self {
        offset: u16::MAX,
        len: 0,
    };

    /// Check if this slot has a suffix.
    #[inline(always)]
    const fn has_suffix(self) -> bool {
        self.offset != u16::MAX
    }
}

impl Default for InlineSlotMeta {
    #[inline(always)]
    fn default() -> Self {
        Self::EMPTY
    }
}

// ============================================================================
//  InlineSuffixBag
// ============================================================================

/// Fixed-capacity suffix storage embedded directly in a leaf node.
///
/// This is an optimization to avoid heap allocation for the common case
/// where total suffix data is small. Based on C++ Masstree's `iksuf_`
/// (internal key suffix) design.
///
/// # Design
///
/// - Embedded in the leaf node (no heap allocation)
/// - Fixed capacity determined at compile time
/// - Append-only with slot reuse when new suffix fits in old space
/// - When full, caller must drain to external `SuffixBag`
///
/// # Memory Layout
///
/// ```text
/// InlineSuffixBag<WIDTH=24, CAPACITY=256> (356 bytes total)
/// ├── slots: [InlineSlotMeta; 24]  // 96 bytes (4 bytes each)
/// ├── size: u16                     // 2 bytes
/// ├── suffix_count: u8              // 1 byte
/// ├── data: [u8; 256]               // 256 bytes
/// └── (1 byte padding for u16 alignment)
/// ```
///
/// # Type Parameters
///
/// * `WIDTH` - Number of slots (must match the leaf node's WIDTH)
/// * `CAPACITY` - Fixed capacity in bytes for suffix data
#[repr(C)]
#[derive(Debug)]
pub struct InlineSuffixBag<const WIDTH: usize, const CAPACITY: usize> {
    /// Per-slot metadata: (offset, length) pairs.
    slots: [InlineSlotMeta; WIDTH],

    /// Current write position in data buffer.
    size: u16,

    /// Cached count of slots with suffixes.
    suffix_count: u8,

    /// Fixed-size data buffer.
    data: [u8; CAPACITY],
}

impl<const WIDTH: usize, const CAPACITY: usize> InlineSuffixBag<WIDTH, CAPACITY> {
    /// Compile-time assertion that WIDTH fits in u8 for `suffix_count`.
    const ASSERT_WIDTH_FITS_U8: () = assert!(WIDTH <= 255, "WIDTH must be <= 255 to fit in u8");

    // ========================================================================
    //  Constructor
    // ========================================================================

    /// Create an empty inline suffix bag.
    ///
    /// This is a const fn so it can be used in static/const contexts.
    #[must_use]
    #[inline(always)]
    pub const fn new() -> Self {
        // Force compile-time evaluation of WIDTH assertion
        let () = Self::ASSERT_WIDTH_FITS_U8;

        Self {
            slots: [InlineSlotMeta::EMPTY; WIDTH],
            size: 0,
            suffix_count: 0,
            data: [0u8; CAPACITY],
        }
    }

    // ========================================================================
    //  Capacity & Size
    // ========================================================================

    /// Return the fixed capacity of this inline bag.
    #[must_use]
    #[inline(always)]
    pub const fn capacity(&self) -> usize {
        CAPACITY
    }

    /// Return the number of bytes currently used.
    #[must_use]
    #[inline(always)]
    pub const fn used(&self) -> usize {
        self.size as usize
    }

    /// Return the remaining capacity.
    #[must_use]
    #[inline(always)]
    pub const fn remaining(&self) -> usize {
        CAPACITY - (self.size as usize)
    }

    /// Return the number of slots that have suffixes.
    ///
    /// This is now O(1), the count is cached and maintained incrementally.
    #[must_use]
    #[inline(always)]
    pub const fn count(&self) -> usize {
        self.suffix_count as usize
    }

    // ========================================================================
    //  Fallible Operations
    // ========================================================================

    /// Try to drain inline suffixes to a new external bag.
    ///
    /// Called when inline storage is full and we need to create an external bag.
    /// Uses only 2 iterations over the permutation (down from 3).
    ///
    /// # Argument
    ///
    /// * `perm` - Current permutation (to iterate active slots)
    /// * `new_slot` - Slot for the new suffix being added
    /// * `new_suffix` - The new suffix that triggered this drain
    ///
    /// # Returns
    ///
    /// * `Ok(bag)` - New external bag with drained suffixes plus new one
    ///
    /// # Errors
    ///
    /// Returns `Err(AllocError)` if the external bag allocation fails.
    pub fn drain_to_external(
        &mut self,
        perm: &impl TreePermutation,
        new_slot: usize,
        new_suffix: &[u8],
    ) -> AllocResult<SuffixBag<WIDTH>> {
        // Pass 1: Calculate required capacity and collect slot data
        let mut required_capacity: usize = new_suffix.len();
        let perm_size: usize = perm.size();

        // Stack-allocated storage for slots to copy
        //                       (slot, start, len)
        let mut slots_to_copy: [(usize, usize, usize); WIDTH] = [(0, 0, 0); WIDTH];
        let mut copy_count: usize = 0;

        #[expect(clippy::indexing_slicing)]
        for i in 0..perm_size {
            let slot: usize = perm.get(i);

            if (slot != new_slot) && (slot < WIDTH) {
                let meta: InlineSlotMeta = self.slots[slot];

                if meta.has_suffix() {
                    let start: usize = meta.offset as usize;
                    let len: usize = meta.len as usize;
                    required_capacity += len;

                    if copy_count < WIDTH {
                        slots_to_copy[copy_count] = (slot, start, len);
                        copy_count += 1;
                    }
                }
            }
        }

        // Try to allocate external bag with capacity
        let mut external: SuffixBag<WIDTH> = SuffixBag::try_with_capacity(required_capacity)?;

        // Pass 2: Copy suffixes to external bag using collected data
        for &(slot, start, len) in &slots_to_copy[..copy_count] {
            // SAFETY: start and len come from valid InlineSlotMeta entries
            let suffix: &[u8] = &self.data[start..(start + len)];
            external.assign(slot, suffix);
        }

        // Assign new suffix
        external.assign(new_slot, new_suffix);

        // Reset inline state completely
        self.slots = [InlineSlotMeta::EMPTY; WIDTH];
        self.size = 0;
        self.suffix_count = 0;

        Ok(external)
    }

    // ========================================================================
    //  Slot Access
    // ========================================================================

    /// Check if a slot has a suffix.
    ///
    /// # Panics
    ///
    /// Panics in debug mode if `slot >= WIDTH`.
    #[must_use]
    #[inline(always)]
    #[expect(
        clippy::indexing_slicing,
        reason = "Slot bounds checked via debug_assert"
    )]
    pub fn has_suffix(&self, slot: usize) -> bool {
        debug_assert!(slot < WIDTH, "slot {slot} >= WIDTH {WIDTH}");
        self.slots[slot].has_suffix()
    }

    /// Get the suffix for a slot, or `None` if no suffix.
    ///
    /// # Panics
    ///
    /// Panics in debug mode if `slot >= WIDTH`.
    #[must_use]
    #[inline(always)]
    #[expect(
        clippy::indexing_slicing,
        reason = "Bounds checked via debug_assert and invariant"
    )]
    pub fn get(&self, slot: usize) -> Option<&[u8]> {
        debug_assert!(slot < WIDTH, "slot {slot} >= WIDTH {WIDTH}");

        let meta: InlineSlotMeta = self.slots[slot];

        if !meta.has_suffix() {
            return None;
        }

        let start: usize = meta.offset as usize;
        let end: usize = start + meta.len as usize;

        // INVARIANT: Valid metadata points to valid data range.
        debug_assert!(
            end <= CAPACITY,
            "inline suffix metadata points past capacity: {end} > {CAPACITY}"
        );

        Some(&self.data[start..end])
    }

    /// Get the suffix for a slot, or empty slice if no suffix.
    #[must_use]
    #[inline(always)]
    pub fn get_or_empty(&self, slot: usize) -> &[u8] {
        self.get(slot).unwrap_or(&[])
    }

    // ========================================================================
    //  Suffix Assignment
    // ========================================================================

    /// Try to assign a suffix to a slot in-place.
    ///
    /// This is the fast path matching C++ `stringbag::assign()`:
    /// 1. If new suffix fits in old slot's space, reuse it
    /// 2. Otherwise, append to end if there's room
    /// 3. If no room, return `false` (caller should use external bag)
    ///
    /// # Returns
    ///
    /// - `true` if the suffix was assigned successfully
    /// - `false` if there's not enough capacity (caller should drain to external)
    ///
    /// # Panics
    ///
    /// Panics if `slot >= WIDTH` or if suffix length exceeds `u16::MAX`.
    #[inline]
    #[expect(
        clippy::indexing_slicing,
        reason = "Slot bounds checked via debug_assert"
    )]
    pub fn try_assign(&mut self, slot: usize, suffix: &[u8]) -> bool {
        debug_assert!(slot < WIDTH, "slot {slot} >= WIDTH {WIDTH}");

        let suffix_len: usize = suffix.len();

        // Suffix must fit in u16
        if suffix_len > U16_MAX {
            return false;
        }

        let meta: InlineSlotMeta = self.slots[slot];

        // Fast Path 1: Reuse existing slot if new suffix fits in old space
        if meta.has_suffix() && (suffix_len <= (meta.len as usize)) {
            let start: usize = meta.offset as usize;
            self.data[start..(start + suffix_len)].copy_from_slice(suffix);

            #[expect(clippy::cast_possible_truncation, reason = "len checked above")]
            {
                self.slots[slot] = InlineSlotMeta {
                    offset: meta.offset,
                    len: suffix_len as u16,
                };
            }

            // Count unchanged, slot already had suffix
            return true;
        }

        // Fast Path 2: Append to end if there's room
        let new_offset: usize = self.size as usize;

        if (new_offset + suffix_len) <= CAPACITY {
            self.data[new_offset..(new_offset + suffix_len)].copy_from_slice(suffix);

            // Update count if this is a new suffix
            if !meta.has_suffix() {
                self.suffix_count += 1;
            }

            #[expect(
                clippy::cast_possible_truncation,
                reason = "offset and len checked to fit"
            )]
            {
                self.slots[slot] = InlineSlotMeta {
                    offset: new_offset as u16,
                    len: suffix_len as u16,
                };

                self.size = (new_offset + suffix_len) as u16;
            }

            return true;
        }

        // Out of capacity, caller should drain to external
        false
    }

    /// Clear the suffix for a slot.
    ///
    /// This marks the slot as having no suffix but does not reclaim
    /// the data buffer space. Space is only reclaimed when draining
    /// to an external bag.
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

        self.slots[slot] = InlineSlotMeta::EMPTY;
    }

    /// Clear all slots and reset size to zero.
    ///
    /// Used after draining to an external bag.
    #[inline(always)]
    pub const fn clear_all(&mut self) {
        self.slots = [InlineSlotMeta::EMPTY; WIDTH];
        self.size = 0;
        self.suffix_count = 0;
    }

    // ========================================================================
    //  Comparison Helpers
    // ========================================================================

    /// Check if a slot's suffix equals the given suffix.
    ///
    /// Uses word-aligned comparision for suffixes >= 8 bytes.
    #[must_use]
    #[inline(always)]
    pub fn suffix_equals(&self, slot: usize, suffix: &[u8]) -> bool {
        self.get(slot)
            .is_some_and(|stored: &[u8]| CompareSuffix::fast_slice_eq(stored, suffix))
    }

    /// Compare a slot's suffix with the given suffix.
    ///
    /// Uses word-aligned comparison for suffixes >= 8 bytes.
    #[must_use]
    #[inline(always)]
    pub fn suffix_compare(&self, slot: usize, suffix: &[u8]) -> Option<Ordering> {
        self.get(slot)
            .map(|stored: &[u8]| CompareSuffix::fast_slice_cmp(stored, suffix))
    }
}

impl<const WIDTH: usize, const CAPACITY: usize> Default for InlineSuffixBag<WIDTH, CAPACITY> {
    #[inline(always)]
    fn default() -> Self {
        Self::new()
    }
}

impl<const WIDTH: usize, const CAPACITY: usize> Clone for InlineSuffixBag<WIDTH, CAPACITY> {
    #[inline(always)]
    fn clone(&self) -> Self {
        Self {
            slots: self.slots,
            size: self.size,
            suffix_count: self.suffix_count,
            data: self.data,
        }
    }
}
