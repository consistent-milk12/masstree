//! Filepath: src/suffix.rs
//!
//! Suffix storage for keys longer than 8 bytes.
//!
//! When a key is longer than 8 bytes, the first 8 bytes are stored as `ikey0`
//! and the remaining bytes are stored in a [`SuffixBag`].

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
    /// # Fast Paths (like C++ stringbag::assign)
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
//  Tests
// ============================================================================

#[cfg(test)]
#[expect(clippy::indexing_slicing)]
mod tests {
    use super::*;

    // ========================================================================
    //  Basic Tests
    // ========================================================================

    #[test]
    fn test_new_suffix_bag() {
        let bag: SuffixBag<15> = SuffixBag::new();

        assert_eq!(bag.count(), 0);
        assert!(bag.capacity() >= INITIAL_CAPACITY);
        assert_eq!(bag.used(), 0);
    }

    #[test]
    fn test_with_capacity() {
        let bag: SuffixBag<15> = SuffixBag::with_capacity(256);

        assert!(bag.capacity() >= 256);
        assert_eq!(bag.count(), 0);
    }

    #[test]
    fn test_default() {
        let bag: SuffixBag<15> = SuffixBag::default();

        assert_eq!(bag.count(), 0);
    }

    // ========================================================================
    //  Assign and Get Tests
    // ========================================================================

    #[test]
    fn test_assign_and_get() {
        let mut bag: SuffixBag<15> = SuffixBag::new();

        bag.assign(0, b"hello");
        bag.assign(5, b"world");
        bag.assign(10, b"!");

        assert_eq!(bag.get(0), Some(b"hello".as_slice()));
        assert_eq!(bag.get(5), Some(b"world".as_slice()));
        assert_eq!(bag.get(10), Some(b"!".as_slice()));
        assert_eq!(bag.get(1), None);
        assert_eq!(bag.count(), 3);
    }

    #[test]
    fn test_empty_suffix() {
        let mut bag: SuffixBag<15> = SuffixBag::new();

        bag.assign(0, b"");

        assert_eq!(bag.get(0), Some(b"".as_slice()));
        assert!(bag.has_suffix(0));
    }

    #[test]
    fn test_get_or_empty() {
        let mut bag: SuffixBag<15> = SuffixBag::new();

        bag.assign(0, b"hello");

        assert_eq!(bag.get_or_empty(0), b"hello".as_slice());
        assert_eq!(bag.get_or_empty(1), b"".as_slice());
    }

    #[test]
    fn test_overwrite_suffix() {
        let mut bag: SuffixBag<15> = SuffixBag::new();

        bag.assign(0, b"hello");
        assert_eq!(bag.get(0), Some(b"hello".as_slice()));

        bag.assign(0, b"goodbye");
        assert_eq!(bag.get(0), Some(b"goodbye".as_slice()));

        // Old data still in buffer (not compacted)
        assert!(bag.used() >= "hello".len() + "goodbye".len());
    }

    // ========================================================================
    //  Clear Tests
    // ========================================================================

    #[test]
    fn test_clear_suffix() {
        let mut bag: SuffixBag<15> = SuffixBag::new();

        bag.assign(0, b"hello");
        assert!(bag.has_suffix(0));

        bag.clear(0);

        assert!(!bag.has_suffix(0));
        assert_eq!(bag.get(0), None);
    }

    #[test]
    fn test_clear_already_empty() {
        let mut bag: SuffixBag<15> = SuffixBag::new();

        // Clearing an empty slot should not panic
        bag.clear(0);

        assert!(!bag.has_suffix(0));
    }

    // ========================================================================
    //  Compact Tests
    // ========================================================================

    #[test]
    fn test_compact() {
        let mut bag: SuffixBag<15> = SuffixBag::new();

        // Add several suffixes
        bag.assign(0, b"aaaa");
        bag.assign(1, b"bbbb");
        bag.assign(2, b"cccc");
        bag.assign(3, b"dddd");

        let before: usize = bag.used();
        assert_eq!(before, 16);

        // Compact keeping only slots 0 and 2
        let reclaimed: usize = bag.compact([0, 2].into_iter());

        assert!(reclaimed > 0);
        assert_eq!(bag.get(0), Some(b"aaaa".as_slice()));
        assert_eq!(bag.get(2), Some(b"cccc".as_slice()));
        assert_eq!(bag.get(1), None);
        assert_eq!(bag.get(3), None);
        assert_eq!(bag.used(), 8);
    }

    #[test]
    fn test_compact_empty() {
        let mut bag: SuffixBag<15> = SuffixBag::new();

        // Compact with no active slots should work
        let reclaimed: usize = bag.compact(std::iter::empty());

        assert_eq!(reclaimed, 0);
        assert_eq!(bag.count(), 0);
    }

    #[test]
    fn test_compact_all() {
        let mut bag: SuffixBag<15> = SuffixBag::new();

        bag.assign(0, b"test");
        bag.assign(1, b"data");

        // Compact keeping all
        let reclaimed: usize = bag.compact([0, 1].into_iter());

        // No garbage to collect
        assert_eq!(reclaimed, 0);
        assert_eq!(bag.get(0), Some(b"test".as_slice()));
        assert_eq!(bag.get(1), Some(b"data".as_slice()));
    }

    #[test]
    fn test_compact_with_permuter() {
        // Create a mock permuter
        struct MockPerm {
            slots: Vec<usize>,
        }

        impl PermutationProvider for MockPerm {
            fn size(&self) -> usize {
                self.slots.len()
            }

            fn get(&self, i: usize) -> usize {
                self.slots[i]
            }
        }

        let mut bag: SuffixBag<15> = SuffixBag::new();
        bag.assign(0, b"keep0");
        bag.assign(1, b"drop1");
        bag.assign(2, b"keep2");

        let perm = MockPerm {
            slots: vec![0, 2], // Only slots 0 and 2 are active
        };

        bag.compact_with_permuter(&perm, None);

        assert_eq!(bag.get(0), Some(b"keep0".as_slice()));
        assert_eq!(bag.get(1), None);
        assert_eq!(bag.get(2), Some(b"keep2".as_slice()));
    }

    #[test]
    fn test_compact_with_exclude() {
        struct MockPerm {
            slots: Vec<usize>,
        }

        impl PermutationProvider for MockPerm {
            fn size(&self) -> usize {
                self.slots.len()
            }

            fn get(&self, i: usize) -> usize {
                self.slots[i]
            }
        }

        let mut bag: SuffixBag<15> = SuffixBag::new();
        bag.assign(0, b"keep");
        bag.assign(1, b"exclude");
        bag.assign(2, b"keep2");

        let perm = MockPerm {
            slots: vec![0, 1, 2],
        };

        // Exclude slot 1 from compaction
        bag.compact_with_permuter(&perm, Some(1));

        assert_eq!(bag.get(0), Some(b"keep".as_slice()));
        assert_eq!(bag.get(1), None); // Excluded
        assert_eq!(bag.get(2), Some(b"keep2".as_slice()));
    }

    // ========================================================================
    //  Growth Tests
    // ========================================================================

    #[test]
    fn test_growth() {
        let mut bag: SuffixBag<15> = SuffixBag::with_capacity(16);

        // Fill past capacity
        for i in 0..15 {
            bag.assign(i, b"12345678"); // 8 bytes each = 120 bytes total
        }

        assert!(bag.capacity() > 16);
        assert_eq!(bag.used(), 120);

        // All suffixes should still be accessible
        for i in 0..15 {
            assert_eq!(bag.get(i), Some(b"12345678".as_slice()));
        }
    }

    #[test]
    fn test_long_suffix() {
        let mut bag: SuffixBag<15> = SuffixBag::new();

        let long_suffix: Vec<u8> = vec![b'x'; 1000];
        bag.assign(0, &long_suffix);

        assert_eq!(bag.get(0), Some(long_suffix.as_slice()));
    }

    // ========================================================================
    //  Comparison Tests
    // ========================================================================

    #[test]
    fn test_suffix_equals() {
        let mut bag: SuffixBag<15> = SuffixBag::new();

        bag.assign(0, b"hello");

        assert!(bag.suffix_equals(0, b"hello"));
        assert!(!bag.suffix_equals(0, b"world"));
        assert!(!bag.suffix_equals(0, b"hell"));
        assert!(!bag.suffix_equals(1, b"hello")); // No suffix at slot 1
    }

    #[test]
    fn test_suffix_compare() {
        let mut bag: SuffixBag<15> = SuffixBag::new();

        bag.assign(0, b"hello");

        assert_eq!(
            bag.suffix_compare(0, b"hello"),
            Some(std::cmp::Ordering::Equal)
        );
        assert_eq!(
            bag.suffix_compare(0, b"hella"),
            Some(std::cmp::Ordering::Greater)
        );
        assert_eq!(
            bag.suffix_compare(0, b"hellz"),
            Some(std::cmp::Ordering::Less)
        );
        assert_eq!(bag.suffix_compare(1, b"hello"), None);
    }

    // ========================================================================
    //  Clone Tests
    // ========================================================================

    #[test]
    fn test_clone() {
        let mut bag: SuffixBag<15> = SuffixBag::new();
        bag.assign(0, b"hello");
        bag.assign(5, b"world");

        let cloned: SuffixBag<15> = bag.clone();

        assert_eq!(cloned.get(0), Some(b"hello".as_slice()));
        assert_eq!(cloned.get(5), Some(b"world".as_slice()));
        assert_eq!(cloned.count(), 2);
    }

    // ========================================================================
    //  Width Variants Tests
    // ========================================================================

    #[test]
    fn test_width_7() {
        let mut bag: SuffixBag<7> = SuffixBag::new();

        bag.assign(0, b"test0");
        bag.assign(6, b"test6");

        assert_eq!(bag.get(0), Some(b"test0".as_slice()));
        assert_eq!(bag.get(6), Some(b"test6".as_slice()));
        assert_eq!(bag.count(), 2);
    }

    #[test]
    fn test_width_3() {
        let mut bag: SuffixBag<3> = SuffixBag::new();

        bag.assign(0, b"a");
        bag.assign(1, b"b");
        bag.assign(2, b"c");

        assert_eq!(bag.count(), 3);
    }

    // ========================================================================
    //  In-Place Assignment Tests
    // ========================================================================

    #[test]
    fn test_try_assign_in_place_fresh_bag() {
        let mut bag: SuffixBag<15> = SuffixBag::new();

        // Fresh bag has capacity, should succeed
        assert!(bag.try_assign_in_place(0, b"hello"));
        assert_eq!(bag.get(0), Some(b"hello".as_slice()));
    }

    #[test]
    fn test_try_assign_in_place_reuse_slot() {
        let mut bag: SuffixBag<15> = SuffixBag::new();

        // Assign a longer suffix first
        bag.assign(0, b"hello world");
        let used_before: usize = bag.used();

        // Assign a shorter suffix - should reuse the slot
        assert!(bag.try_assign_in_place(0, b"hi"));
        assert_eq!(bag.get(0), Some(b"hi".as_slice()));

        // Used bytes should not increase (reused existing space)
        assert_eq!(bag.used(), used_before);
    }

    #[test]
    fn test_try_assign_in_place_append() {
        let mut bag: SuffixBag<15> = SuffixBag::new();

        // Assign to slot 0
        assert!(bag.try_assign_in_place(0, b"first"));

        // Assign to slot 1 - should append
        assert!(bag.try_assign_in_place(1, b"second"));

        assert_eq!(bag.get(0), Some(b"first".as_slice()));
        assert_eq!(bag.get(1), Some(b"second".as_slice()));
    }

    #[test]
    fn test_try_assign_in_place_fails_when_full() {
        // Create a bag with very small capacity
        let mut bag: SuffixBag<15> = SuffixBag::with_capacity(10);

        // First assignment should succeed
        assert!(bag.try_assign_in_place(0, b"12345"));

        // Second assignment that exceeds capacity should fail
        assert!(!bag.try_assign_in_place(1, b"678901234567890"));

        // First slot should still be valid
        assert_eq!(bag.get(0), Some(b"12345".as_slice()));
        // Second slot should not exist
        assert_eq!(bag.get(1), None);
    }

    #[test]
    fn test_try_assign_in_place_same_length() {
        let mut bag: SuffixBag<15> = SuffixBag::new();

        bag.assign(0, b"hello");
        let used_before: usize = bag.used();

        // Same length should reuse slot
        assert!(bag.try_assign_in_place(0, b"world"));
        assert_eq!(bag.get(0), Some(b"world".as_slice()));
        assert_eq!(bag.used(), used_before);
    }

    #[test]
    fn test_try_assign_in_place_longer_suffix_needs_append() {
        let mut bag: SuffixBag<15> = SuffixBag::new();

        bag.assign(0, b"hi");
        let used_before: usize = bag.used();

        // Longer suffix can't reuse slot, needs append
        assert!(bag.try_assign_in_place(0, b"hello world"));
        assert_eq!(bag.get(0), Some(b"hello world".as_slice()));

        // Used bytes should increase
        assert!(bag.used() > used_before);
    }

    #[test]
    fn test_try_assign_in_place_mixed_usage() {
        let mut bag: SuffixBag<15> = SuffixBag::new();

        // Fill several slots
        for i in 0..5 {
            assert!(bag.try_assign_in_place(i, b"test"));
        }

        // Reuse slot 2 with shorter suffix
        assert!(bag.try_assign_in_place(2, b"ab"));
        assert_eq!(bag.get(2), Some(b"ab".as_slice()));

        // Other slots unchanged
        assert_eq!(bag.get(0), Some(b"test".as_slice()));
        assert_eq!(bag.get(1), Some(b"test".as_slice()));
        assert_eq!(bag.get(3), Some(b"test".as_slice()));
        assert_eq!(bag.get(4), Some(b"test".as_slice()));
    }
}
