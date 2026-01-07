//! Filepath: src/suffix.rs
//!
//! Suffix storage for keys longer than 8 bytes.
//!
//! When a key is longer than 8 bytes, the first 8 bytes are stored as `ikey0`
//! and the remaining bytes are stored in a [`SuffixBag`].

use crate::{AllocError, AllocResult, TreePermutation};

/// Initial capacity for suffix storage (matches C++ `INITIAL_KSUF_CAPACITY`).
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
}

impl SlotMeta {
    /// Sentinel value indicating no suffix stored.
    const EMPTY: Self = Self {
        offset: u32::MAX,
        len: 0,
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
}

impl<const WIDTH: usize> SuffixBag<WIDTH> {
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
        }
    }

    /// Create a new suffix bag with specified capacity.
    #[must_use]
    #[inline(always)]
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            slots: [SlotMeta::EMPTY; WIDTH],
            data: Vec::with_capacity(capacity),
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
    pub fn count(&self) -> usize {
        self.slots.iter().filter(|s| s.has_suffix()).count()
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
        })
    }

    /// Try to assign a suffix, returning error if allocation fails.
    ///
    /// Unlike `assign`, this method returns an error instead of panicking
    /// if the [`Vec`] needs to grow and allocation fails.
    ///
    /// # Returns
    ///
    /// * `Ok(())` - Suffix assigned successfully
    /// * `Err(AllocError)` - Could not grow storage
    ///
    /// # Errors
    ///
    /// Upon allocation failure.
    ///
    /// # Panics
    ///
    /// If suffix is longer than `u16::MAX`
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

        // Fast path 1: Reuse existing slot if new suffix fits
        if meta.has_suffix() && (suffix_len <= (meta.len as usize)) {
            let start: usize = meta.offset as usize;
            self.data[start..(start + suffix_len)].copy_from_slice(suffix);

            #[expect(clippy::cast_possible_truncation)]
            {
                self.slots[slot] = SlotMeta {
                    offset: meta.offset,
                    len: suffix_len as u16,
                };
            }

            return Ok(());
        }

        // Fast path 2: Append if there's room
        let new_offset: usize = self.data.len();

        if (new_offset + suffix_len) <= self.data.capacity() {
            self.data.extend_from_slice(suffix);

            #[expect(clippy::cast_possible_truncation)]
            {
                self.slots[slot] = SlotMeta {
                    offset: new_offset as u32,
                    len: suffix_len as u16,
                };
            }

            return Ok(());
        }

        // Slow path: Need to grow, try to reserve
        self.data
            .try_reserve(suffix_len)
            .map_err(|_| AllocError::for_suffix(suffix_len))?;

        // Now we have capacity
        let new_offset = self.data.len();
        self.data.extend_from_slice(suffix);

        #[expect(clippy::cast_possible_truncation)]
        {
            self.slots[slot] = SlotMeta {
                offset: new_offset as u32,
                len: suffix_len as u16,
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
    #[expect(
        clippy::indexing_slicing,
        reason = "Slot bounds checked via debug_assert"
    )]
    #[inline]
    pub fn try_assign_in_place(&mut self, slot: usize, suffix: &[u8]) -> bool {
        debug_assert!(slot < WIDTH, "slot {slot} >= WIDTH {WIDTH}");
        assert!(
            u16::try_from(suffix.len()).is_ok(),
            "suffix too long: {} > {}",
            suffix.len(),
            u16::MAX
        );

        let meta: SlotMeta = self.slots[slot];

        // Fast path 1: Reuse existing slot if new suffix fits in old space
        if meta.has_suffix() && suffix.len() <= meta.len as usize {
            let start: usize = meta.offset as usize;
            // SAFETY: meta is valid, we're writing within existing bounds
            self.data[start..start + suffix.len()].copy_from_slice(suffix);

            #[expect(clippy::cast_possible_truncation, reason = "len checked above")]
            {
                self.slots[slot] = SlotMeta {
                    offset: meta.offset,
                    len: suffix.len() as u16,
                };
            }
            return true;
        }

        // Fast path 2: Append to end if there's room
        let new_offset: usize = self.data.len();
        if new_offset + suffix.len() <= self.data.capacity() {
            self.data.extend_from_slice(suffix);

            #[expect(clippy::cast_possible_truncation, reason = "offset and len checked")]
            {
                self.slots[slot] = SlotMeta {
                    offset: new_offset as u32,
                    len: suffix.len() as u16,
                };
            }
            return true;
        }

        // Slow path: doesn't fit, caller should reallocate
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
    #[expect(
        clippy::indexing_slicing,
        reason = "Slot bounds checked via debug_assert"
    )]
    #[inline(always)]
    pub fn clear(&mut self, slot: usize) {
        debug_assert!(slot < WIDTH, "slot {slot} >= WIDTH {WIDTH}");
        self.slots[slot] = SlotMeta::EMPTY;
    }

    // ========================================================================
    //  Compaction
    // ========================================================================

    /// Compact the suffix bag, keeping only the specified active slots.
    ///
    /// This creates a new data buffer containing only the suffixes for
    /// slots that are both marked active AND have suffixes stored.
    /// This effectively garbage-collects unused suffix data.
    ///
    /// # Arguments
    ///
    /// * `active_slots` - Iterator yielding physical slot indices that are active
    ///
    /// # Returns
    ///
    /// The number of bytes reclaimed.
    #[expect(
        clippy::indexing_slicing,
        reason = "Slot bounds explicitly checked in the loop"
    )]
    pub fn compact(&mut self, active_slots: impl Iterator<Item = usize>) -> usize {
        let old_used: usize = self.data.len();

        // Collect active slots to avoid borrowing issues
        let active: Vec<usize> = active_slots.collect();

        // Calculate new size needed
        let new_size: usize = active
            .iter()
            .filter_map(|&slot| {
                if slot < WIDTH && self.slots[slot].has_suffix() {
                    Some(self.slots[slot].len as usize)
                } else {
                    None
                }
            })
            .sum();

        // Allocate new buffer with power-of-2 capacity
        let new_capacity: usize = new_size.next_power_of_two().max(INITIAL_CAPACITY);
        let mut new_data: Vec<u8> = Vec::with_capacity(new_capacity);

        // Copy active suffixes and update metadata
        let mut new_slots: [SlotMeta; WIDTH] = [SlotMeta::EMPTY; WIDTH];

        for &slot in &active {
            if slot >= WIDTH {
                continue;
            }

            let meta: SlotMeta = self.slots[slot];
            if !meta.has_suffix() {
                continue;
            }

            // Direct slice access - bounds already validated by SlotMeta invariant
            let start: usize = meta.offset as usize;
            let end: usize = start + meta.len as usize;
            let suffix: &[u8] = &self.data[start..end];

            let new_offset: usize = new_data.len();
            new_data.extend_from_slice(suffix);

            #[expect(
                clippy::cast_possible_truncation,
                reason = "new_offset bounded by new_capacity which fits in u32"
            )]
            {
                new_slots[slot] = SlotMeta {
                    offset: new_offset as u32,
                    len: meta.len, // Reuse existing len instead of recomputing
                };
            }
        }

        self.data = new_data;
        self.slots = new_slots;

        old_used.saturating_sub(self.data.len())
    }

    /// Compact using a permutation to determine active slots.
    ///
    /// This is the typical usage pattern: compact based on which slots
    /// are currently in-use according to the leaf's permutation.
    ///
    /// # Arguments
    ///
    /// * `perm` - Permuter indicating which slots are active
    /// * `exclude_slot` - Optional slot to exclude (e.g., slot being removed)
    ///
    /// # Returns
    ///
    /// The number of bytes reclaimed.
    pub fn compact_with_permuter<P: PermutationProvider>(
        &mut self,
        perm: &P,
        exclude_slot: Option<usize>,
    ) -> usize {
        let active = (0..perm.size())
            .map(|i| perm.get(i))
            .filter(|&s| Some(s) != exclude_slot);

        self.compact(active)
    }

    // ========================================================================
    //  Comparison Helpers
    // ========================================================================

    /// Check if a slot's suffix equals the given suffix.
    ///
    /// # Returns
    ///
    /// - `true` if suffixes match exactly
    /// - `false` if slot has no suffix or suffixes differ
    #[must_use]
    #[inline(always)]
    pub fn suffix_equals(&self, slot: usize, suffix: &[u8]) -> bool {
        self.get(slot).is_some_and(|stored| stored == suffix)
    }

    /// Compare a slot's suffix with the given suffix.
    ///
    /// # Returns
    ///
    /// - `Some(Ordering)` if slot has a suffix
    /// - `None` if slot has no suffix
    #[must_use]
    #[inline(always)]
    pub fn suffix_compare(&self, slot: usize, suffix: &[u8]) -> Option<std::cmp::Ordering> {
        self.get(slot).map(|stored| stored.cmp(suffix))
    }
}

impl<const WIDTH: usize> Default for SuffixBag<WIDTH> {
    #[inline(always)]
    fn default() -> Self {
        Self::new()
    }
}

impl<const WIDTH: usize> Clone for SuffixBag<WIDTH> {
    #[inline(always)]
    fn clone(&self) -> Self {
        Self {
            slots: self.slots,
            data: self.data.clone(),
        }
    }
}

// ============================================================================
//  InlineSlotMeta
// ============================================================================

/// Metadata for a single slot's suffix in inline storage.
///
/// Uses `u16` for offset to keep metadata compact (4 bytes per slot).
/// Maximum inline capacity is 65535 bytes.
#[derive(Clone, Copy, Debug)]
#[repr(C)]
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
/// InlineSuffixBag<WIDTH=24, CAPACITY=256> (354 bytes total)
/// ├── slots: [InlineSlotMeta; 24]  // 96 bytes (4 bytes each)
/// ├── size: u16                     // 2 bytes
/// └── data: [u8; 256]               // 256 bytes
/// ```
///
/// # Type Parameters
///
/// * `WIDTH` - Number of slots (must match the leaf node's WIDTH)
/// * `CAPACITY` - Fixed capacity in bytes for suffix data
#[derive(Debug)]
#[repr(C)]
pub struct InlineSuffixBag<const WIDTH: usize, const CAPACITY: usize> {
    /// Per-slot metadata: (offset, length) pairs.
    slots: [InlineSlotMeta; WIDTH],

    /// Current write position in data buffer.
    size: u16,

    /// Fixed-size data buffer.
    data: [u8; CAPACITY],
}

impl<const WIDTH: usize, const CAPACITY: usize> InlineSuffixBag<WIDTH, CAPACITY> {
    // ========================================================================
    //  Constructor
    // ========================================================================

    /// Create an empty inline suffix bag.
    ///
    /// This is a const fn so it can be used in static/const contexts.
    #[must_use]
    #[inline(always)]
    pub const fn new() -> Self {
        Self {
            slots: [InlineSlotMeta::EMPTY; WIDTH],
            size: 0,
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
        CAPACITY - self.size as usize
    }

    /// Return the number of slots that have suffixes.
    #[must_use]
    #[inline(always)]
    pub fn count(&self) -> usize {
        self.slots.iter().filter(|s| s.has_suffix()).count()
    }

    // ========================================================================
    //  Fallible Operations
    // ========================================================================

    /// Try to drain inline suffixes to a new external bag.
    ///
    /// Called when inline storage is full and we need to create an external bag.
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
        // calculate required cap
        let mut required_capacity: usize = new_suffix.len();

        for i in 0..perm.size() {
            let slot: usize = perm.get(i);

            if slot != new_slot
                && let Some(suffix) = self.get(slot)
            {
                required_capacity += suffix.len();
            }
        }

        // Try to allocate external bag with capacity
        let mut external: SuffixBag<_> = SuffixBag::try_with_capacity(required_capacity)?;

        // Copy existing suffixes from inline storage
        for i in 0..perm.size() {
            let slot: usize = perm.get(i);

            if slot != new_slot
                && let Some(suffix) = self.get(slot)
            {
                // This should not fail since we pre-reserved capcity.
                external.assign(slot, suffix);
            }
        }

        // Assign new suffix (also should fail)
        external.assign(new_slot, new_suffix);

        // Clear inline storage
        for i in 0..perm.size() {
            let slot: usize = perm.get(i);
            self.clear(slot);
        }

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
        if suffix_len > u16::MAX as usize {
            return false;
        }

        let meta: InlineSlotMeta = self.slots[slot];

        // Fast path 1: Reuse existing slot if new suffix fits in old space
        if meta.has_suffix() && suffix_len <= meta.len as usize {
            let start: usize = meta.offset as usize;
            self.data[start..start + suffix_len].copy_from_slice(suffix);

            #[expect(clippy::cast_possible_truncation, reason = "len checked above")]
            {
                self.slots[slot] = InlineSlotMeta {
                    offset: meta.offset,
                    len: suffix_len as u16,
                };
            }
            return true;
        }

        // Fast path 2: Append to end if there's room
        let new_offset: usize = self.size as usize;
        if new_offset + suffix_len <= CAPACITY {
            self.data[new_offset..new_offset + suffix_len].copy_from_slice(suffix);

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

        // Out of capacity - caller should drain to external
        false
    }

    /// Clear the suffix for a slot.
    ///
    /// This marks the slot as having no suffix but does NOT reclaim
    /// the data buffer space. Space is only reclaimed when draining
    /// to an external bag.
    ///
    /// # Panics
    ///
    /// Panics in debug mode if `slot >= WIDTH`.
    #[inline(always)]
    #[expect(
        clippy::indexing_slicing,
        reason = "Slot bounds checked via debug_assert"
    )]
    pub fn clear(&mut self, slot: usize) {
        debug_assert!(slot < WIDTH, "slot {slot} >= WIDTH {WIDTH}");
        self.slots[slot] = InlineSlotMeta::EMPTY;
    }

    /// Clear all slots and reset size to zero.
    ///
    /// Used after draining to an external bag.
    #[inline(always)]
    pub const fn clear_all(&mut self) {
        self.slots = [InlineSlotMeta::EMPTY; WIDTH];
        self.size = 0;
    }

    // ========================================================================
    //  Comparison Helpers
    // ========================================================================

    /// Check if a slot's suffix equals the given suffix.
    #[must_use]
    #[inline(always)]
    pub fn suffix_equals(&self, slot: usize, suffix: &[u8]) -> bool {
        self.get(slot).is_some_and(|stored| stored == suffix)
    }

    /// Compare a slot's suffix with the given suffix.
    #[must_use]
    #[inline(always)]
    pub fn suffix_compare(&self, slot: usize, suffix: &[u8]) -> Option<std::cmp::Ordering> {
        self.get(slot).map(|stored| stored.cmp(suffix))
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
            data: self.data,
        }
    }
}

// ============================================================================
//  Tests
// ============================================================================

#[cfg(test)]
#[expect(clippy::indexing_slicing)]
mod unit_tests;
