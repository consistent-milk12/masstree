//! Filepath: src/suffix.rs
//!
//! Suffix storage for keys longer than 8 bytes.
//!
//! When a key is longer than 8 bytes, the first 8 bytes are stored as `ikey0`
//! and the remaining bytes are stored in a [`SuffixBag`].

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
#[derive(Debug)]
pub struct SuffixBag {
    /// Per-slot metadata: (offset, length) pairs.
    slots: [SlotMeta; WIDTH],

    /// Contiguous suffix data buffer.
    data: Vec<u8>,

    /// Cached count of slots with suffixes (avoids O(WIDTH) recount).
    suffix_count: u8,
}

impl SuffixBag {
    // ========================================================================
    //  Constructor
    // ========================================================================

    /// Create a new suffix bag with initial capacity.
    #[must_use]
    #[inline(always)]
    pub fn new() -> Self {
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
        Self {
            slots: [SlotMeta::EMPTY; WIDTH],
            data: Vec::with_capacity(capacity),
            suffix_count: 0,
        }
    }

    /// Construct [`SuffixBag`] by reusing an existing [`Vec<u8>`] buffer.
    #[must_use]
    #[inline(always)]
    pub fn from_vec(data: Vec<u8>) -> Self {
        debug_assert!(data.is_empty(), "from_vec expects empty Vec");

        Self {
            slots: [SlotMeta::EMPTY; WIDTH],
            data,
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
    #[must_use]
    #[inline(always)]
    pub const fn count(&self) -> usize {
        self.suffix_count as usize
    }

    /// Reserve additional capacity for suffix data.
    #[inline(always)]
    pub fn reserve(&mut self, additional: usize) {
        self.data.reserve(additional);
    }

    // ========================================================================
    //  Smart Assignment with Compaction
    // ========================================================================

    /// Assign a suffix with smart space management.
    ///
    /// # Panics
    ///
    /// Panics if `suffix.len() > u16::MAX` (65535 bytes).
    #[expect(clippy::indexing_slicing, reason = "Checked access")]
    pub fn try_assign(&mut self, slot: usize, suffix: &[u8]) -> bool {
        debug_assert!(slot < WIDTH, "slot {slot} >= WIDTH {WIDTH}");

        let suffix_len: usize = suffix.len();

        assert!(
            u16::try_from(suffix_len).is_ok(),
            "suffix too long: {suffix_len} > {}",
            u16::MAX
        );

        let meta: SlotMeta = self.slots[slot];

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

            return true;
        }

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

            return true;
        }

        self.compact_in_place();

        let new_offset: usize = self.data.len();

        if (new_offset + suffix_len) <= self.data.capacity() {
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

            return true;
        }

        // Still not enough space - must grow (aborts on OOM)
        self.data.reserve(suffix_len);

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
    /// Panics if `slot >= 15`.
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
    /// Panics if `slot >= 15`.
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
    /// # Panics
    ///
    /// Panics if `slot >= 15` or if suffix length exceeds `u16::MAX`.
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

            return true;
        }

        let new_offset: usize = self.data.len();

        if (new_offset + suffix.len()) <= self.data.capacity() {
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
    /// # Panics
    ///
    /// Panics if `slot >= 15` or if suffix length exceeds `u16::MAX`.
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

        if !meta.has_suffix() {
            self.suffix_count += 1;
        }

        let offset: usize = self.data.len();
        self.data.extend_from_slice(suffix);

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
    /// # Panics
    ///
    /// Panics if `slot >= 15`.
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

impl Default for SuffixBag {
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
