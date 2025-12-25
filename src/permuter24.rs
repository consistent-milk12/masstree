//! 24-slot permutation using u128 storage.
//!
//! This module provides `Permuter24`, a permutation type supporting 24 slots
//! using 5-bit encoding in a u128. This enables 60% more capacity per leaf node
//! compared to the standard 15-slot `Permuter`.
//!
//! # Bit Layout
//!
//! ```text
//! u128 (128 bits):
//! [3 unused] [slot23] [slot22] ... [slot1] [slot0] [size]
//!  127-125   124-120  119-115      14-10    9-5     4-0
//!
//! Slot i: bits (i*5 + 5) to (i*5 + 9)
//! Size:   bits 0-4 (5 bits, values 0-24)
//! ```

use portable_atomic::{AtomicU128, Ordering};

use crate::{leaf_trait::TreePermutation, suffix::PermutationProvider};

// Re-export Freeze24Utils from freeze24 module
pub use crate::freeze24::Freeze24Utils;

// =============================================================================
// Constants
// =============================================================================

/// Number of slots supported by Permuter24.
pub const WIDTH_24: usize = 24;

/// Bits per slot in u128 encoding.
const SLOT_BITS: usize = 5;

/// Bits for size field.
const SIZE_BITS: usize = 5;

/// Mask for size (lower 5 bits).
const SIZE_MASK: u128 = 0x1F;

/// Mask for single slot (5 bits).
const SLOT_MASK: u128 = 0x1F;

/// Invalid slot value used as freeze sentinel.
/// Value 31 is invalid because valid slots are 0-23.
pub const FREEZE_SENTINEL: u128 = 0x1F;

/// Bit position of slot 23 (freeze position).
pub const FREEZE_SHIFT: usize = 23 * SLOT_BITS + SIZE_BITS; // = 120

/// Mask for slot 23 (freeze slot).
#[allow(dead_code)]
const FREEZE_SLOT_MASK: u128 = SLOT_MASK << FREEZE_SHIFT;

// =============================================================================
// Permuter24
// =============================================================================

/// A 24-slot permutation using u128 storage.
///
/// Encodes which physical slot holds the key at each logical position.
/// Uses 5-bit slots allowing values 0-23 for slot indices, 0-24 for size.
///
/// # Invariants
///
/// - `size() <= 24`
/// - All slot indices 0-23 appear exactly once in the encoding
/// - Positions `0..size()` are "in use", positions `size()..24` are "free"
/// - Value 31 (0x1F) never appears as a valid slot (reserved for freeze)
///
/// # Memory
///
/// - Size: 16 bytes (u128)
/// - Alignment: 16 bytes
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[repr(transparent)]
pub struct Permuter24 {
    value: u128,
}

// =============================================================================
// Construction
// =============================================================================

impl Permuter24 {
    /// Initial value with slots in reverse order, size = 0.
    ///
    /// This constant is used by both `empty()` and `AtomicPermuter24::new()`.
    /// Position i holds slot (23 - i), so `back()` returns 0 initially.
    ///
    /// Bit layout: size=0, slot[0]=23, slot[1]=22, ..., slot[23]=0
    pub const INITIAL: u128 = {
        let mut value: u128 = 0;
        let mut i: usize = 0;
        while i < 24 {
            let slot: u128 = (23 - i) as u128;
            // Position i is at bits (i * 5 + 5) to (i * 5 + 9)
            value |= slot << (i * SLOT_BITS + SIZE_BITS);
            i += 1;
        }
        value
    };

    /// Sorted value: position i -> slot i.
    const SORTED: u128 = {
        let mut value: u128 = 0;
        let mut i: usize = 0;
        while i < 24 {
            value |= (i as u128) << (i * SLOT_BITS + SIZE_BITS);
            i += 1;
        }
        value
    };

    /// Create an empty permuter with size 0.
    ///
    /// Slots are stored in reverse order so `back()` returns 0 initially.
    /// Free slots will be allocated in order: 0, 1, 2, ...
    #[inline(always)]
    #[must_use]
    pub const fn empty() -> Self {
        Self {
            value: Self::INITIAL,
        }
    }

    /// Create a sorted permuter with `n` elements.
    ///
    /// Position i maps to slot i for i in `0..n` (sorted order).
    /// Remaining slots are free and will be allocated from `back()`.
    ///
    /// # Panics
    ///
    /// Debug-panics if `n > 24`.
    #[must_use]
    pub fn make_sorted(n: usize) -> Self {
        debug_assert!(n <= 24, "make_sorted: n ({n}) > 24");

        if n == 24 {
            return Self {
                value: Self::SORTED | 24,
            };
        }

        // Build: positions 0..n sorted, positions n..24 hold remaining in reverse
        let mut value: u128 = n as u128; // size = n

        // Positions 0..n: slot i at position i
        for i in 0..n {
            value |= (i as u128) << (i * SLOT_BITS + SIZE_BITS);
        }

        // Positions n..24: remaining slots in reverse order
        // So back() = get(23) returns n (next to allocate)
        for pos in n..24 {
            let slot = 23 - (pos - n); // 23, 22, ..., n
            value |= (slot as u128) << (pos * SLOT_BITS + SIZE_BITS);
        }

        Self { value }
    }

    /// Create from raw u128 value.
    #[must_use]
    #[inline(always)]
    pub const fn from_value(value: u128) -> Self {
        Self { value }
    }
}

impl Default for Permuter24 {
    #[inline(always)]
    fn default() -> Self {
        Self::empty()
    }
}

// =============================================================================
// Query Methods
// =============================================================================

impl Permuter24 {
    /// Return the number of slots in use.
    #[must_use]
    #[inline(always)]
    pub const fn size(&self) -> usize {
        (self.value & SIZE_MASK) as usize
    }

    /// Return the slot index at logical position `i`.
    ///
    /// # Panics
    ///
    /// Debug-panics if `i >= 24`.
    #[must_use]
    #[inline(always)]
    pub const fn get(&self, i: usize) -> usize {
        debug_assert!(i < 24, "get: index out of bounds");
        ((self.value >> (i * SLOT_BITS + SIZE_BITS)) & SLOT_MASK) as usize
    }

    /// Return the slot at the back (position 23).
    ///
    /// This is the next slot to be allocated on `insert_from_back`.
    #[must_use]
    #[inline(always)]
    pub const fn back(&self) -> usize {
        self.get(23)
    }

    /// Get the slot at `back()` with an offset into the free region.
    ///
    /// `back_at_offset(0)` == `back()`.
    ///
    /// # Panics
    ///
    /// Debug-panics if `size() + offset >= 24`.
    #[must_use]
    #[inline(always)]
    pub const fn back_at_offset(&self, offset: usize) -> usize {
        debug_assert!(
            self.size() + offset < 24,
            "back_at_offset: offset exceeds free slots"
        );
        self.get(23 - offset)
    }

    /// Return the raw u128 value.
    #[must_use]
    #[inline(always)]
    pub const fn value(&self) -> u128 {
        self.value
    }
}

// =============================================================================
// Basic Setters
// =============================================================================

impl Permuter24 {
    /// Set the size without changing slot positions.
    #[inline(always)]
    pub fn set_size(&mut self, n: usize) {
        debug_assert!(n <= 24, "set_size: n ({n}) > 24");
        self.value = (self.value & !SIZE_MASK) | (n as u128);
    }

    /// Set the slot at a given position.
    #[inline(always)]
    pub fn set(&mut self, i: usize, slot: usize) {
        debug_assert!(i < 24, "set: position {i} >= 24");
        debug_assert!(slot < 24, "set: slot {slot} >= 24");

        let shift: usize = i * SLOT_BITS + SIZE_BITS;
        self.value = (self.value & !(SLOT_MASK << shift)) | ((slot as u128) << shift);
    }

    /// Swap two slots in the free region (positions >= size).
    #[inline(always)]
    pub fn swap_free_slots(&mut self, pos_i: usize, pos_j: usize) {
        let size = self.size();
        debug_assert!(pos_i >= size, "swap_free_slots: pos_i in use region");
        debug_assert!(pos_j >= size, "swap_free_slots: pos_j in use region");
        debug_assert!(pos_i < 24 && pos_j < 24, "swap_free_slots: out of range");

        if pos_i == pos_j {
            return;
        }

        // XOR swap
        let i_shift = pos_i * SLOT_BITS + SIZE_BITS;
        let j_shift = pos_j * SLOT_BITS + SIZE_BITS;
        let diff = ((self.value >> i_shift) ^ (self.value >> j_shift)) & SLOT_MASK;
        self.value ^= (diff << i_shift) | (diff << j_shift);
    }
}

// =============================================================================
// Insert Operations
// =============================================================================

impl Permuter24 {
    /// Allocate a slot from the back and insert at position `i`.
    ///
    /// Returns the allocated slot index.
    ///
    /// # Algorithm
    ///
    /// 1. Take slot from `back()` (position 23)
    /// 2. Shift positions `i..size()` up by one
    /// 3. Insert the slot at position i
    /// 4. Increment size
    ///
    /// # Panics
    ///
    /// Debug-panics if `i > size()` or `size() >= 24`.
    #[inline]
    #[must_use]
    pub fn insert_from_back(&mut self, i: usize) -> usize {
        debug_assert!(i <= self.size(), "insert_from_back: i > size");
        debug_assert!(self.size() < 24, "insert_from_back: permuter full");

        let slot = self.back();
        let i_shift = i * SLOT_BITS + SIZE_BITS;

        // Mask for size + positions 0..(i-1)
        let low_mask: u128 = (1u128 << i_shift) - 1;

        // Increment size, keep low positions, insert slot, shift high positions
        self.value = ((self.value + 1) & low_mask)
            | ((slot as u128) << i_shift)
            | ((self.value << SLOT_BITS) & !(low_mask | (SLOT_MASK << i_shift)));

        #[cfg(debug_assertions)]
        self.debug_assert_valid();

        slot
    }

    /// Compute insert result without mutation (for CAS).
    ///
    /// Returns `(new_permuter, allocated_slot)`.
    #[inline]
    #[must_use]
    pub fn insert_from_back_immutable(&self, i: usize) -> (Self, usize) {
        debug_assert!(i <= self.size(), "insert_from_back_immutable: i > size");
        debug_assert!(self.size() < 24, "insert_from_back_immutable: full");

        let slot = self.back();
        let i_shift = i * SLOT_BITS + SIZE_BITS;
        let low_mask: u128 = (1u128 << i_shift) - 1;

        let new_value = ((self.value + 1) & low_mask)
            | ((slot as u128) << i_shift)
            | ((self.value << SLOT_BITS) & !(low_mask | (SLOT_MASK << i_shift)));

        let new_perm = Self { value: new_value };

        #[cfg(debug_assertions)]
        new_perm.debug_assert_valid();

        (new_perm, slot)
    }
}

// =============================================================================
// Remove Operations
// =============================================================================

impl Permuter24 {
    /// Remove element at position `i` and move to back.
    ///
    /// # Panics
    ///
    /// Debug-panics if `i >= size()`.
    #[inline(always)]
    pub fn remove_to_back(&mut self, i: usize) {
        debug_assert!(i < self.size(), "remove_to_back: i >= size");

        // Mask covers positions >= i (bits from position i onwards)
        // Position i starts at bit (i * 5 + 5), so we mask from there
        let i_shift = i * SLOT_BITS + SIZE_BITS;
        let mask: u128 = !((1u128 << i_shift) - 1);

        // Width mask: 25 * 5 = 125 bits used
        let width_mask: u128 = (1u128 << 125) - 1;
        let x = self.value & width_mask;

        let shift_to_back = (23 - i) * SLOT_BITS;

        self.value = ((x - 1) & !mask)       // decrement size, keep < i
            | ((x >> SLOT_BITS) & mask)       // shift >= i down
            | ((x & mask) << shift_to_back); // move removed to back

        #[cfg(debug_assertions)]
        self.debug_assert_valid();
    }

    /// Remove element at position `i`.
    ///
    /// Removed slot goes to position `size()` (first free slot).
    ///
    /// # Panics
    ///
    /// Debug-panics if `i >= size()`.
    #[inline(always)]
    pub fn remove(&mut self, i: usize) {
        let size = self.size();
        debug_assert!(i < size, "remove: i >= size");

        // Fast path: removing last element
        if i + 1 == size {
            self.value -= 1;

            #[cfg(debug_assertions)]
            self.debug_assert_valid();

            return;
        }

        let rot_amount: usize = (size - i - 1) * SLOT_BITS;
        // rot_mask covers positions i through size-1
        let rot_mask: u128 =
            (((1u128 << rot_amount) << SLOT_BITS) - 1) << (i * SLOT_BITS + SIZE_BITS);

        self.value = ((self.value - 1) & !rot_mask)
            | (((self.value & rot_mask) >> SLOT_BITS) & rot_mask)
            | (((self.value & rot_mask) << rot_amount) & rot_mask);

        #[cfg(debug_assertions)]
        self.debug_assert_valid();
    }
}

// =============================================================================
// Reorder Operations
// =============================================================================

impl Permuter24 {
    /// Exchange (swap) elements at positions `i` and `j`.
    #[inline(always)]
    pub fn exchange(&mut self, i: usize, j: usize) {
        debug_assert!(i < 24 && j < 24, "exchange: out of range");

        if i == j {
            return;
        }

        // Position i is at bits (i * 5 + 5), position j at (j * 5 + 5)
        let i_shift: usize = i * SLOT_BITS + SIZE_BITS;
        let j_shift: usize = j * SLOT_BITS + SIZE_BITS;
        let diff: u128 = ((self.value >> i_shift) ^ (self.value >> j_shift)) & SLOT_MASK;
        self.value ^= (diff << i_shift) | (diff << j_shift);

        #[cfg(debug_assertions)]
        self.debug_assert_valid();
    }

    /// Rotate elements between positions `i` and `j`.
    #[inline(always)]
    pub fn rotate(&mut self, i: usize, j: usize) {
        debug_assert!(i <= j && j <= 24, "rotate: invalid range");

        if i == j || i == 24 {
            return;
        }

        // Mask covers size + positions 0..(i-1), i.e., bits 0 to (i*5+5)-1
        let i_shift: usize = i * SLOT_BITS + SIZE_BITS;
        let mask: u128 = (1u128 << i_shift) - 1;
        let width_mask: u128 = (1u128 << 125) - 1;
        let x: u128 = self.value & width_mask;

        let rotate_amount: usize = (j - i) * SLOT_BITS;
        let rotate_back: usize = (24 - j) * SLOT_BITS;

        self.value = (x & mask) | ((x >> rotate_amount) & !mask) | ((x & !mask) << rotate_back);

        #[cfg(debug_assertions)]
        self.debug_assert_valid();
    }
}

// =============================================================================
// Validation
// =============================================================================

impl Permuter24 {
    /// Verify permuter invariants (debug builds only).
    ///
    /// # Note
    ///
    /// This method must NOT be called on frozen permuters. A frozen permuter
    /// has slot 23 set to 0x1F (31), which is invalid during normal operation.
    /// Frozen state is only valid during split operations.
    ///
    /// If called on a frozen permuter, this method silently returns without
    /// checking invariants to avoid false positives.
    ///
    /// # Panics
    ///
    /// Panics in debug builds if the permuter is invalid (not frozen) and:
    /// - The size exceeds 24
    /// - Any slot index is >= 24
    /// - Any slot index appears more than once
    /// - Not all slot indices 0-23 are present
    #[cfg(debug_assertions)]
    pub fn debug_assert_valid(&self) {
        // Skip validation for frozen permuters (slot 23 = 0x1F)
        // Frozen state is valid during splits but would fail slot < 24 check
        if Freeze24Utils::is_frozen(self.value) {
            return;
        }

        let size: usize = self.size();
        assert!(size <= 24, "invalid size: {size} > 24");

        // Check all slots 0-23 appear exactly once
        let mut seen: u32 = 0;
        for i in 0..24 {
            let slot: usize = self.get(i);
            assert!(slot < 24, "invalid slot index: {slot}");

            let bit: u32 = 1u32 << slot;
            assert!(seen & bit == 0, "duplicate slot: {slot}");
            seen |= bit;
        }

        assert_eq!(seen, (1u32 << 24) - 1, "missing slot indices");
    }

    /// No-op in release builds.
    #[cfg(not(debug_assertions))]
    #[inline]
    pub fn debug_assert_valid(&self) {}
}

// =============================================================================
// Trait Implementations
// =============================================================================

impl PermutationProvider for Permuter24 {
    #[inline(always)]
    fn size(&self) -> usize {
        self.size()
    }

    #[inline(always)]
    fn get(&self, i: usize) -> usize {
        self.get(i)
    }
}

// =============================================================================
// AtomicPermuter24
// =============================================================================

/// Atomic wrapper for Permuter24.
///
/// Uses `portable-atomic` crate for cross-platform 128-bit atomics.
/// On x86-64 with CMPXCHG16B, uses native lock-free operations.
#[derive(Debug)]
#[repr(transparent)]
pub struct AtomicPermuter24 {
    inner: AtomicU128,
}

impl AtomicPermuter24 {
    /// Create with empty permutation.
    #[must_use]
    #[inline(always)]
    pub const fn new() -> Self {
        Self {
            inner: AtomicU128::new(Permuter24::INITIAL),
        }
    }

    /// Create from existing permuter.
    #[must_use]
    #[inline(always)]
    pub const fn from_permuter(perm: Permuter24) -> Self {
        Self {
            inner: AtomicU128::new(perm.value),
        }
    }

    /// Load with ordering.
    #[inline(always)]
    pub fn load(&self, order: Ordering) -> Permuter24 {
        Permuter24::from_value(self.inner.load(order))
    }

    /// Store with ordering.
    #[inline(always)]
    pub fn store(&self, val: Permuter24, order: Ordering) {
        self.inner.store(val.value, order);
    }

    /// Compare-and-exchange.
    ///
    /// # Errors
    ///
    /// Returns `Err` with the current value if the comparison failed
    /// (i.e., the current value did not match `expected`).
    #[inline(always)]
    pub fn compare_exchange(
        &self,
        expected: Permuter24,
        new: Permuter24,
        success: Ordering,
        failure: Ordering,
    ) -> Result<Permuter24, Permuter24> {
        self.inner
            .compare_exchange(expected.value, new.value, success, failure)
            .map(Permuter24::from_value)
            .map_err(Permuter24::from_value)
    }

    /// Compare-and-exchange weak.
    ///
    /// # Errors
    ///
    /// Returns `Err` with the current value if the comparison failed
    /// (i.e., the current value did not match `expected`), or spuriously.
    #[inline(always)]
    pub fn compare_exchange_weak(
        &self,
        expected: Permuter24,
        new: Permuter24,
        success: Ordering,
        failure: Ordering,
    ) -> Result<Permuter24, Permuter24> {
        self.inner
            .compare_exchange_weak(expected.value, new.value, success, failure)
            .map(Permuter24::from_value)
            .map_err(Permuter24::from_value)
    }

    // === Raw operations for freeze mechanism ===

    /// Load raw u128 value.
    #[inline(always)]
    pub fn load_raw(&self, order: Ordering) -> u128 {
        self.inner.load(order)
    }

    /// Store raw u128 value.
    #[inline(always)]
    pub fn store_raw(&self, val: u128, order: Ordering) {
        self.inner.store(val, order);
    }

    /// CAS on raw value.
    ///
    /// # Errors
    ///
    /// Returns `Err` with the current value if the comparison failed
    /// (i.e., the current value did not match `expected`).
    #[inline(always)]
    pub fn compare_exchange_raw(
        &self,
        expected: u128,
        new: u128,
        success: Ordering,
        failure: Ordering,
    ) -> Result<u128, u128> {
        self.inner.compare_exchange(expected, new, success, failure)
    }
}

impl Default for AtomicPermuter24 {
    fn default() -> Self {
        Self::new()
    }
}

// =============================================================================
// TreePermutation Implementation
// =============================================================================

impl TreePermutation for Permuter24 {
    type Raw = u128;
    const WIDTH: usize = WIDTH_24;

    #[inline(always)]
    fn empty() -> Self {
        Self::empty()
    }

    #[inline(always)]
    fn make_sorted(n: usize) -> Self {
        Self::make_sorted(n)
    }

    #[inline(always)]
    fn from_value(raw: u128) -> Self {
        Self::from_value(raw)
    }

    #[inline(always)]
    fn value(&self) -> u128 {
        Self::value(self)
    }

    #[inline(always)]
    fn size(&self) -> usize {
        Self::size(self)
    }

    #[inline(always)]
    fn get(&self, i: usize) -> usize {
        Self::get(self, i)
    }

    #[inline(always)]
    fn back(&self) -> usize {
        Self::back(self)
    }

    #[inline(always)]
    fn back_at_offset(&self, offset: usize) -> usize {
        Self::back_at_offset(self, offset)
    }

    #[inline(always)]
    fn insert_from_back(&mut self, i: usize) -> usize {
        Self::insert_from_back(self, i)
    }

    #[inline(always)]
    fn insert_from_back_immutable(&self, i: usize) -> (Self, usize) {
        Self::insert_from_back_immutable(self, i)
    }

    #[inline(always)]
    fn swap_free_slots(&mut self, pos_i: usize, pos_j: usize) {
        Self::swap_free_slots(self, pos_i, pos_j);
    }

    #[inline(always)]
    fn set_size(&mut self, n: usize) {
        Self::set_size(self, n);
    }

    #[inline(always)]
    fn is_frozen_raw(raw: u128) -> bool {
        crate::freeze24::Freeze24Utils::is_frozen(raw)
    }

    #[inline(always)]
    fn freeze_raw(raw: u128) -> u128 {
        crate::freeze24::Freeze24Utils::freeze_raw(raw)
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_empty() {
        let p = Permuter24::empty();
        assert_eq!(p.size(), 0);
        assert_eq!(p.back(), 0);

        // Verify initial slot ordering: position i holds slot (23 - i)
        for i in 0..24 {
            assert_eq!(p.get(i), 23 - i, "position {i}");
        }
    }

    #[test]
    fn test_make_sorted_zero() {
        let p = Permuter24::make_sorted(0);
        assert_eq!(p.size(), 0);
        assert_eq!(p.back(), 0);
    }

    #[test]
    fn test_make_sorted_partial() {
        let p = Permuter24::make_sorted(10);
        assert_eq!(p.size(), 10);

        // Positions 0..10 should be sorted
        for i in 0..10 {
            assert_eq!(p.get(i), i, "sorted position {i}");
        }

        // back() should return next to allocate (10)
        assert_eq!(p.back(), 10);
    }

    #[test]
    fn test_make_sorted_full() {
        let p = Permuter24::make_sorted(24);
        assert_eq!(p.size(), 24);
        for i in 0..24 {
            assert_eq!(p.get(i), i);
        }
    }

    #[test]
    fn test_insert_from_back_sequential() {
        let mut p = Permuter24::empty();

        for i in 0..24 {
            let slot = p.insert_from_back(i);
            assert_eq!(slot, i, "allocated slot for position {i}");
            assert_eq!(p.size(), i + 1);
            assert_eq!(p.get(i), i);
        }

        assert_eq!(p.size(), 24);
    }

    #[test]
    fn test_insert_from_back_at_front() {
        let mut p = Permuter24::empty();

        // Insert at position 0 repeatedly
        let s0 = p.insert_from_back(0);
        assert_eq!(s0, 0);
        assert_eq!(p.size(), 1);
        assert_eq!(p.get(0), 0);

        let s1 = p.insert_from_back(0);
        assert_eq!(s1, 1);
        assert_eq!(p.size(), 2);
        assert_eq!(p.get(0), 1);
        assert_eq!(p.get(1), 0);

        let s2 = p.insert_from_back(0);
        assert_eq!(s2, 2);
        assert_eq!(p.size(), 3);
        assert_eq!(p.get(0), 2);
        assert_eq!(p.get(1), 1);
        assert_eq!(p.get(2), 0);
    }

    #[test]
    fn test_insert_from_back_immutable() {
        let p = Permuter24::make_sorted(5);
        let (new_p, slot) = p.insert_from_back_immutable(2);

        assert_eq!(slot, 5); // Next free slot
        assert_eq!(new_p.size(), 6);
        assert_eq!(new_p.get(2), 5); // Inserted at position 2

        // Original unchanged
        assert_eq!(p.size(), 5);
    }

    #[test]
    fn test_remove_last() {
        let mut p = Permuter24::make_sorted(5);
        p.remove(4);
        assert_eq!(p.size(), 4);

        // Positions 0..4 unchanged
        for i in 0..4 {
            assert_eq!(p.get(i), i);
        }
    }

    #[test]
    fn test_remove_middle() {
        let mut p = Permuter24::make_sorted(5);
        // Remove position 2 (slot 2)
        p.remove(2);
        assert_eq!(p.size(), 4);

        // Positions shifted
        assert_eq!(p.get(0), 0);
        assert_eq!(p.get(1), 1);
        assert_eq!(p.get(2), 3);
        assert_eq!(p.get(3), 4);
    }

    #[test]
    fn test_exchange() {
        let mut p = Permuter24::make_sorted(5);
        p.exchange(1, 3);

        assert_eq!(p.get(0), 0);
        assert_eq!(p.get(1), 3);
        assert_eq!(p.get(2), 2);
        assert_eq!(p.get(3), 1);
        assert_eq!(p.get(4), 4);
    }

    #[test]
    fn test_freeze() {
        let p = Permuter24::make_sorted(10);
        let frozen = Freeze24Utils::freeze_raw(p.value());

        assert!(Freeze24Utils::is_frozen(frozen));
        assert!(!Freeze24Utils::is_frozen(p.value()));
    }

    #[test]
    fn test_atomic_basic() {
        let ap = AtomicPermuter24::new();
        let p = ap.load(Ordering::Relaxed);
        assert_eq!(p.size(), 0);
        assert_eq!(p.back(), 0);
    }

    #[test]
    fn test_atomic_cas() {
        let ap = AtomicPermuter24::new();
        let old = ap.load(Ordering::Relaxed);
        let (new, _slot) = old.insert_from_back_immutable(0);

        let result = ap.compare_exchange(old, new, Ordering::SeqCst, Ordering::Relaxed);

        assert!(result.is_ok());
        assert_eq!(ap.load(Ordering::Relaxed).size(), 1);
    }

    #[test]
    fn test_fill_to_capacity() {
        let mut p = Permuter24::empty();
        for i in 0..24 {
            let slot = p.insert_from_back(i);
            assert_eq!(slot, i);
        }
        assert_eq!(p.size(), 24);
    }

    #[test]
    fn test_back_at_offset() {
        let p = Permuter24::make_sorted(10);

        // Free slots are at positions 10..24
        // back() = get(23) = 10 (next to allocate)
        assert_eq!(p.back(), 10);
        assert_eq!(p.back_at_offset(0), 10);
        assert_eq!(p.back_at_offset(1), 11);
    }

    #[test]
    fn test_make_sorted_via_trait() {
        fn check_trait<P: TreePermutation>() {
            let p = P::make_sorted(2);

            assert_eq!(p.size(), 2);
            assert_eq!(p.get(0), 0);
            assert_eq!(p.get(1), 1);
        }

        check_trait::<Permuter24>();
    }
}
