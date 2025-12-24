//! Filepath: src/permuter.rs
//!
//! Permutation array for leaf node slot ordering.
//!
//! The [`Permuter`] encodes a permutation of slot indices in a single `u64`,
//! enabling O(1) logical reordering without moving key/value data.
//!
//! # Const Generic WIDTH
//!
//! The permuter supports configurable WIDTH via const generics:
//! - `Permuter<15>` (default): Standard 15-slot nodes
//! - `Permuter<7>`: Compact nodes for memory-constrained scenarios
//!
//! **Maximum WIDTH is 15** due to u64 permuter encoding.
//! 4 bits * 15 slots + 4 bits size = 64 bits exactly.

/// Maximum allowed WIDTH for u64 permuter encoding.
pub const MAX_WIDTH: usize = 15;

/// Number of bits used to store the size.
#[allow(dead_code)]
const SIZE_BITS: usize = 4;

/// Mask for extracting size (lower 4 bits).
const SIZE_MASK: u64 = 0xF;

use crate::suffix::PermutationProvider;

/// Utility functions for [`Permuter`].
struct PermuterUtils;

impl PermuterUtils {
    /// Compute initial permuter value for a given WIDTH.
    ///
    /// Slots are in REVERSE order: position i holds slot (WIDTH - 1 - i).
    /// This ensures `back()` returns slot 0, so slots are allocated 0, 1, 2, ... in order.
    ///
    /// For WIDTH = 15: produces `0x0123_4567_89AB_CDE0`
    ///     - position 0 -> slot 14 (0xE)
    ///     - position 1 -> slot 13 (0xD)
    ///     - ...
    ///     - position 14 -> slot 0
    ///     - size = 0
    const fn compute_initial_value<const WIDTH: usize>() -> u64 {
        // Build value: position i contains slot (WIDTH - 1 - i), size = 0
        // This is REVERSE order to match C++ reference: x0123456789ABCDE0
        let mut value: u64 = 0;
        let mut i: usize = 0;

        while i < WIDTH {
            let slot: u64 = (WIDTH - 1 - i) as u64;

            value |= slot << ((i * 4) + 4);
            i += 1;
        }

        value
    }

    /// Compute sorted permuter value for a given WIDTH.
    ///
    /// Position i maps to slot i for all i in 0..WIDTH (natural/sorted order).
    /// Used by `make_sorted()` to create a permuter where positions match slots.
    ///
    /// For WIDTH=15: produces `0xEDCB_A987_6543_2100`
    ///   - position 0 → slot 0
    ///   - position 1 → slot 1
    ///   - ...
    ///   - position 14 → slot 14 (0xE)
    ///   - size = 0
    const fn compute_sorted_value<const WIDTH: usize>() -> u64 {
        // Build value: position i contains slot i (sorted order)
        let mut value: u64 = 0;
        let mut i: usize = 0;

        while i < WIDTH {
            value |= (i as u64) << ((i * 4) + 4);
            i += 1;
        }

        // size bits are 0
        value
    }
}

/// A permutation of slot indices for a leaf node.
///
/// Encodes which physical slot holds the key at each logical position.
/// The logical positions `0..size()` are in sorted key order.
///
/// # Type Parameters
///
/// * `WIDTH` - Number of slots (default: 15, max: 15)
///
/// # Invariants
///
/// - `size() <= WIDTH`
/// - All slot indices 0 to WIDTH-1 appear exactly once in the encoding
/// - Positions `0..size()` are "in use", positions `size()..WIDTH` are "free"
///
/// # Compile-Time Constraints
///
/// WIDTH must be in range 1..=15. WIDTH > 15 is a compile error because
/// the u64 encoding cannot fit more than 15 slots.
///
/// # Example
///
/// ```rust,ignore
/// use masstree::permuter::Permuter;
///
/// // Default WIDTH=15
/// let mut p = Permuter::empty();
/// assert_eq!(p.size(), 0);
///
/// // Compact WIDTH=7
/// let mut compact: Permuter<7> = Permuter::empty();
/// assert_eq!(compact.size(), 0);
///
/// // Insert at position 0, allocates slot from back
/// let slot = p.insert_from_back(0);
/// assert_eq!(p.size(), 1);
/// assert_eq!(p.get(0), slot);
/// ```
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Permuter<const WIDTH: usize = 15> {
    value: u64,
}

/// Compile-time assertion: WIDTH must be in range 1..=15
impl<const WIDTH: usize> Permuter<WIDTH> {
    /// Compile-time check that WIDTH is valid.
    const WIDTH_CHECK: () = {
        assert!(WIDTH > 0, "WIDTH must be at least 1");

        assert!(
            WIDTH <= MAX_WIDTH,
            "WIDTH must be at most 15 (u64 encoding limit)"
        );
    };
}

impl<const WIDTH: usize> Default for Permuter<WIDTH> {
    fn default() -> Self {
        Self::empty()
    }
}

/// Type alias for standard 15-slot permuter.
pub type Permuter15 = Permuter<15>;

/// Type alias for compact 7-slot permuter (2 cache lines).
pub type PermuterCompact = Permuter<7>;

impl<const WIDTH: usize> Permuter<WIDTH> {
    /// Create an empty permuter with size 0.
    ///
    /// Slots are stored in reverse order so that `back()` return 0 initially.
    /// Free slots will allocated in order: 0, 1, 2, ...
    ///
    /// For WIDTH = 15: value = `0x0123_4567_89AB_CDE0` (matches C++ reference)
    #[inline]
    #[must_use]
    pub const fn empty() -> Self {
        // Trigger compile-time WIDTH check
        let _: () = Self::WIDTH_CHECK;

        Self {
            value: PermuterUtils::compute_initial_value::<WIDTH>(),
        }
    }

    /// Create a sorted permuter with `n` elements.
    ///
    /// Position i maps to slot i for i in `0..n` (sorted/natural order).
    /// Remaining slots are free and will be allocated from `back()`.
    ///
    /// # Panics
    /// Panics if `n > WIDTH`.
    #[must_use]
    pub fn make_sorted(n: usize) -> Self {
        debug_assert!(n <= WIDTH, "make_sorted: n ({n}) > WIDTH ({WIDTH})");

        if n == WIDTH {
            // Fully sorted: positions 0..WIDTH map to slots 0..WIDTH, size = WIDTH
            // Uses sorted_value (position i → slot i), NOT initial_value (reverse order)
            return Self {
                value: PermuterUtils::compute_sorted_value::<WIDTH>() | (WIDTH as u64),
            };
        }

        // For partially sorted permuters:
        // - Positions 0..n are in use and map to slots 0..n (sorted)
        // - Positions n..WIDTH are free and hold remaining slots in reverse order
        //   so that back() returns next slot to allocate

        // Start with sorted value (positions map to their slot indices)
        let sorted: u64 = PermuterUtils::compute_sorted_value::<WIDTH>();

        // We need positions 0..n to be sorted (slot i at position i)
        // and positions n..WIDTH to hold remaining slots in reverse order
        // so back() = get(WIDTH - 1) return slot n (next to allocate)

        // Build the free slot porting: positions n..WIDTH hold slots n..WIDTH - 1
        // in reverse order so back() = slot n
        let mut value: u64 = n as u64;

        // Copy sorted positions 0..n
        let sorted_mask: u64 = ((1u64 << (n * 4)) - 1) << 4;
        value |= sorted & sorted_mask;

        // Fill positions n..WIDTH with remaining slots in reverse order
        // Position n gets slot WIDTH - 1, position n + 1 gets slot WIDTH - 2, etc.
        // Position WIDTH - 1 (back) gets slot n
        let mut pos: usize = n;

        while pos < WIDTH {
            let slot: u64 = (WIDTH - 1 - (pos - n)) as u64;
            value |= slot << ((pos * 4) + 4);
            pos += 1;
        }

        Self { value }
    }

    /// Return the number of slots in use.
    #[must_use]
    #[inline(always)]
    pub const fn size(&self) -> usize {
        (self.value & SIZE_MASK) as usize
    }

    /// Return the slot index at logical position `i`.
    ///
    /// # Panics
    /// Panics in debug mode if `i >= WIDTH`.
    #[must_use]
    #[inline(always)]
    pub const fn get(&self, i: usize) -> usize {
        debug_assert!(i < WIDTH, "get: index out of bounds");

        ((self.value >> ((i * 4) + 4)) & 0xF) as usize
    }

    /// Return the slot at the back (position WIDTH - 1)
    ///
    /// This is the next slot to be allocated on `insert_from_back`.
    #[must_use]
    #[inline(always)]
    pub const fn back(&self) -> usize {
        self.get(WIDTH - 1)
    }

    /// Get the slot at `back()` with an offset into the free region.
    ///
    /// `back_at_offset(0)` == `back()`, `back_at_offset(1)` is the next free slot, etc.
    /// This is used when a slot is claimed but not yet published in the permutation,
    /// allowing the locked path to try the next available slot.
    ///
    /// # Panics
    /// Debug-panics if `size() + offset >= WIDTH` (no more free slots at that offset).
    #[must_use]
    #[inline(always)]
    pub const fn back_at_offset(&self, offset: usize) -> usize {
        debug_assert!(
            self.size() + offset < WIDTH,
            "back_at_offset: offset exceeds free slots"
        );
        self.get(WIDTH - 1 - offset)
    }

    /// Return the raw u64 value.
    #[must_use]
    #[inline(always)]
    pub const fn value(&self) -> u64 {
        self.value
    }

    /// Create a Permuter from a raw u64 value.
    ///
    /// Used when loading from atomic storage.
    ///
    /// # Arguments
    /// * `value` - The raw u64 value loaded from atomic
    #[must_use]
    #[inline(always)]
    pub const fn from_value(value: u64) -> Self {
        Self { value }
    }

    /// Set the size without changing slot positions.
    ///
    /// # Safety
    /// Caller must ensure the new size is valid (0..=WIDTH).
    #[inline(always)]
    pub fn set_size(&mut self, n: usize) {
        debug_assert!(n <= WIDTH, "set_size: n ({n}) > WIDTH ({WIDTH})");

        self.value = (self.value & !SIZE_MASK) | (n as u64);
    }

    /// Set the slot at a given position.
    ///
    /// # Arguments
    ///
    /// * `i` - Position index (0..WIDTH)
    /// * `slot` - Slot value to store (0..WIDTH)
    ///
    /// # Panics
    ///
    /// Panics in debug mode if position or slot is out of range.
    #[inline(always)]
    pub fn set(&mut self, i: usize, slot: usize) {
        debug_assert!(i < WIDTH, "set: position {i} >= WIDTH {WIDTH}");
        debug_assert!(slot < WIDTH, "set: slot {slot} >= WIDTH {WIDTH}");

        let shift: usize = (i + 1) * 4;
        let mask: u64 = 0xFu64 << shift;
        self.value = (self.value & !mask) | ((slot as u64) << shift);
    }

    /// Swap two slots in the free region (positions >= size).
    ///
    /// Used to skip slot 0 when it can't be reused due to `ikey_bound` constraints.
    /// The free region starts at position `size()` and extends to `WIDTH - 1`.
    ///
    /// # Arguments
    ///
    /// * `pos_i` - First position to swap (must be >= size)
    /// * `pos_j` - Second position to swap (must be >= size)
    ///
    /// # Panics
    ///
    /// Panics in debug mode if positions are not in the free region.
    pub fn swap_free_slots(&mut self, pos_i: usize, pos_j: usize) {
        let size: usize = self.size();

        debug_assert!(
            pos_i >= size,
            "swap_free_slots: pos_i ({pos_i}) must be >= size ({size})"
        );
        debug_assert!(
            pos_j >= size,
            "swap_free_slots: pos_j ({pos_j}) must be >= size ({size})"
        );
        debug_assert!(pos_i < WIDTH, "swap_free_slots: pos_i out of range");
        debug_assert!(pos_j < WIDTH, "swap_free_slots: pos_j out of range");

        if pos_i == pos_j {
            return; // Nothing to swap
        }

        // XOR swap trick (same as exchange()) - single operation instead of 4
        let i_shift: usize = (pos_i + 1) * 4;
        let j_shift: usize = (pos_j + 1) * 4;
        let diff: u64 = ((self.value >> i_shift) ^ (self.value >> j_shift)) & 0xF;
        self.value ^= (diff << i_shift) | (diff << j_shift);
    }

    /// Allocate a slot from the back and insert it at position `i`.
    ///
    /// Returns the allocated slot index.
    ///
    /// # Panics
    /// Panics in debug mode if `i > size()` or `size() >= WIDTH`.
    ///
    /// # Algorithm
    /// 1. Take slot from `back()` (position WIDTH - 1)
    /// 2. Shift positions `i..size()` up by one
    /// 3. Insert the slot at position i
    /// 4. Increment size
    #[must_use]
    pub fn insert_from_back(&mut self, i: usize) -> usize {
        debug_assert!(i <= self.size(), "insert_from_back: i > size");
        debug_assert!(self.size() < WIDTH, "insert_from_back: permuter full");

        let slot: usize = self.back();

        // Bit manipulation matching C++ reference:
        // - Increment size, keep positions < i unchanged
        // - Insert slot at position i
        // - Shift positions >= i up by one (they move to i + 1, i + 2, ...)
        let i_shift: usize = (i * 4) + 4;

        // Bits for size + positions 0..(i - 1)
        let low_mask: u64 = (1u64 << i_shift) - 1;

        // Algorithm:
        // 1. Increment size, keep lower positions unchanged
        // 2. Insert slot at position i
        // 3. Shift higher positions up (including free slots)
        self.value = ((self.value + 1) & low_mask)
            | ((slot as u64) << i_shift)
            | ((self.value << 4) & !(low_mask | (0xF << i_shift)));

        #[cfg(debug_assertions)]
        self.debug_assert_valid();

        slot
    }

    /// Compute the result of `insert_from_back` without mutating self.
    ///
    /// Returns `(new_permuter, allocated_slot)` for use in CAS operations.
    /// The returned permuter represents the state after insertion.
    ///
    /// This is the immutable version of [`Self::insert_from_back`] for lock-free
    /// CAS-based insertion where we need to compute the new permutation value
    /// before attempting an atomic compare-and-swap.
    ///
    /// # Panics
    /// Panics in debug mode if `i > size()` or `size() >= WIDTH`.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let perm = Permuter::<15>::empty();
    /// let (new_perm, slot) = perm.insert_from_back_immutable(0);
    /// assert_eq!(new_perm.size(), 1);
    /// assert_eq!(new_perm.get(0), slot);
    /// // Original unchanged
    /// assert_eq!(perm.size(), 0);
    /// ```
    #[must_use]
    #[inline]
    pub fn insert_from_back_immutable(&self, i: usize) -> (Self, usize) {
        debug_assert!(i <= self.size(), "insert_from_back_immutable: i > size");
        debug_assert!(
            self.size() < WIDTH,
            "insert_from_back_immutable: permuter full"
        );

        let slot: usize = self.back();
        let i_shift: usize = (i * 4) + 4;
        let low_mask: u64 = (1u64 << i_shift) - 1;

        let new_value: u64 = ((self.value + 1) & low_mask)
            | ((slot as u64) << i_shift)
            | ((self.value << 4) & !(low_mask | (0xF << i_shift)));

        let new_perm = Self { value: new_value };

        #[cfg(debug_assertions)]
        new_perm.debug_assert_valid();

        (new_perm, slot)
    }

    /// Remove the element at position `i` and move it to the back.
    ///
    /// After this operation:
    /// - `size()` is decremented by 1
    /// - Positions `0..i` are unchanged
    /// - Positions `i..size()-1` shift down (position j gets old position j+1)
    /// - `back()` returns the slot that was at position `i`
    ///
    /// # Panics
    /// Panics in debug mode if `i >= size()`.
    ///
    /// # Algorithm (matches C++ `kpermuter::remove_to_back`)
    /// 1. Decrement size, keep positions `0..i` unchanged
    /// 2. Shift positions `i+1..WIDTH` down by one
    /// 3. Move the removed slot to position `WIDTH-1` (back)
    pub fn remove_to_back(&mut self, i: usize) {
        debug_assert!(i < self.size(), "remove_to_back: i >= size");

        // mask covers bits for positions >= i (not including size bits for positions < i)
        // i_shift = bit offset of position i = (i + 1) * 4
        let i_shift: usize = (i + 1) * 4;
        let mask: u64 = !((1u64 << i_shift) - 1);

        // Clear unused bits above WIDTH positions (for 64-bit safety)
        // For WIDTH=15, width_shift=64 which would overflow, so use saturating logic
        let width_shift: usize = (WIDTH + 1) * 4;

        let width_mask: u64 = if width_shift >= 64 {
            u64::MAX
        } else {
            (1u64 << width_shift) - 1
        };

        let x: u64 = self.value & width_mask;

        // Bit manipulation matching C++ reference:
        // - Decrement size, keep positions < i unchanged
        // - Shift positions >= i down by one (position i gets position i+1's slot, etc.)
        // - Move the removed element (original position i) to the back (position WIDTH-1)
        let shift_to_back: usize = (WIDTH - i - 1) * 4;

        self.value = ((x - 1) & !mask)           // decrement size, keep positions < i
            | ((x >> 4) & mask)                  // shift positions >= i down
            | ((x & mask) << shift_to_back); // move removed element to back

        #[cfg(debug_assertions)]
        self.debug_assert_valid();
    }

    /// Remove the element at position `i`.
    ///
    /// After this operation:
    /// - `size()` is decremented by 1
    /// - Positions `0..i` are unchanged
    /// - Positions `i..size()-1` shift down (position j gets old position j+1)
    /// - Position `size()` (the new first free slot) gets the removed element
    ///
    /// This differs from `remove_to_back` in where the removed element goes:
    /// - `remove`: puts removed element at position `size()` (just past in-use)
    /// - `remove_to_back`: puts removed element at position `WIDTH-1` (back)
    ///
    /// # Panics
    /// Panics in debug mode if `i >= size()`.
    ///
    /// # Algorithm (matches C++ `kpermuter::remove`)
    pub fn remove(&mut self, i: usize) {
        let size: usize = self.size();
        debug_assert!(i < size, "remove: i >= size");

        // Fast path: removing the last element
        if size == i + 1 {
            self.value -= 1;

            #[cfg(debug_assertions)]
            self.debug_assert_valid();

            return;
        }

        // rot_amount = (size - i - 1) * 4 = number of bits to rotate
        let rot_amount: usize = (size - i - 1) * 4;

        // rot_mask covers positions i through size-1
        let rot_mask: u64 = (((1u64 << rot_amount) << 4) - 1) << ((i + 1) * 4);

        // Bit manipulation matching C++ reference:
        // - Decrement size, keep positions < i unchanged
        // - Shift positions i+1..size down (into positions i..size-1)
        // - Rotate the removed element up to position size-1 (new free slot)
        self.value = ((self.value - 1) & !rot_mask)
            | (((self.value & rot_mask) >> 4) & rot_mask)
            | (((self.value & rot_mask) << rot_amount) & rot_mask);

        #[cfg(debug_assertions)]
        self.debug_assert_valid();
    }

    /// Exchange (swap) the elements at positions `i` and `j`.
    ///
    /// After this operation:
    /// - `size()` is unchanged
    /// - `get(i)` returns what was at position `j`
    /// - `get(j)` returns what was at position `i`
    /// - All other positions are unchanged
    ///
    /// # Panics
    /// Panics in debug mode if `i >= WIDTH` or `j >= WIDTH`.
    ///
    /// # Algorithm (matches C++ `kpermuter::exchange`)
    /// Uses XOR swap trick on the 4-bit slot values.
    pub fn exchange(&mut self, i: usize, j: usize) {
        debug_assert!(i < WIDTH, "exchange: i >= WIDTH");
        debug_assert!(j < WIDTH, "exchange: j >= WIDTH");

        if i == j {
            return;
        }

        // Shift amounts for positions i and j (add 4 to skip size bits)
        let i_shift: usize = (i + 1) * 4;
        let j_shift: usize = (j + 1) * 4;

        // XOR the two 4-bit values, then XOR back into both positions
        // This swaps the values without a temporary variable
        let diff: u64 = ((self.value >> i_shift) ^ (self.value >> j_shift)) & 0xF;
        self.value ^= (diff << i_shift) | (diff << j_shift);

        #[cfg(debug_assertions)]
        self.debug_assert_valid();
    }

    /// Rotate elements between positions `i` and `j`.
    ///
    /// After this operation:
    /// - `size()` is unchanged
    /// - Positions `0..i` are unchanged
    /// - For positions `k` in `i..WIDTH`:
    ///   `new[k] = old[i + (k - i + j - i) mod (WIDTH - i)]`
    ///
    /// This effectively rotates the elements from position `i` to the end
    /// by `(j - i)` positions.
    ///
    /// # Panics
    /// Panics in debug mode if `i > j` or `j > WIDTH`.
    ///
    /// # Algorithm (matches C++ `kpermuter::rotate`)
    pub fn rotate(&mut self, i: usize, j: usize) {
        debug_assert!(i <= j, "rotate: i > j");
        debug_assert!(j <= WIDTH, "rotate: j > WIDTH");

        if i == j || i == WIDTH {
            return;
        }

        // mask covers positions 0..i-1 (bits to keep unchanged)
        let i_shift: usize = (i + 1) * 4;
        let mask: u64 = (1u64 << i_shift) - 1;

        // Clear unused bits above WIDTH positions (for 64-bit safety)
        let width_shift: usize = (WIDTH + 1) * 4;

        let width_mask: u64 = if width_shift >= 64 {
            u64::MAX
        } else {
            (1u64 << width_shift) - 1
        };

        let x: u64 = self.value & width_mask;

        let rotate_amount: usize = (j - i) * 4;
        let rotate_back: usize = (WIDTH - j) * 4;

        // Bit manipulation matching C++ reference:
        // - Keep positions < i unchanged (including size)
        // - Shift positions >= i right by (j-i)*4 bits
        // - Wrap positions that fall off the right back to the left
        self.value = (x & mask) | ((x >> rotate_amount) & !mask) | ((x & !mask) << rotate_back);

        #[cfg(debug_assertions)]
        self.debug_assert_valid();
    }

    /// Verify permuter invariants (debug builds only).
    ///
    /// Checks:
    /// - Size is in range [0, WIDTH]
    /// - All slot indices 0-14 appear exactly once
    ///
    /// # Panics
    /// If any of the invariants are not satisfied.
    #[cfg(debug_assertions)]
    pub fn debug_assert_valid(&self) {
        let size: usize = self.size();
        assert!(size <= WIDTH, "invalid size: {size} > {WIDTH}");

        // Check all slots 0-14 appear exactly once
        let mut seen: u16 = 0;

        for i in 0..WIDTH {
            let slot: usize = self.get(i);
            assert!(slot < WIDTH, "invalid slot index: {slot}");

            let bit: u16 = 1u16 << slot;
            assert!(seen & bit == 0, "duplicate slot index: {slot}");

            seen |= bit;
        }

        assert_eq!(seen, (1u16 << WIDTH) - 1, "missing slot indices");
    }

    /// Verify permuter invariants (no-op in release builds).
    #[inline]
    #[cfg(not(debug_assertions))]
    pub fn debug_assert_valid(&self) {}
}

// ============================================================================
//  PermutationProvider Implementation
// ============================================================================

impl<const WIDTH: usize> PermutationProvider for Permuter<WIDTH> {
    #[inline]
    fn size(&self) -> usize {
        self.size()
    }

    #[inline]
    fn get(&self, i: usize) -> usize {
        self.get(i)
    }
}

// ============================================================================
//  TreePermutation Implementation
// ============================================================================

impl<const WIDTH: usize> crate::leaf_trait::TreePermutation for Permuter<WIDTH> {
    type Raw = u64;
    const WIDTH: usize = WIDTH;

    #[inline]
    fn empty() -> Self {
        Permuter::empty()
    }

    #[inline]
    fn from_value(raw: u64) -> Self {
        Permuter::from_value(raw)
    }

    #[inline]
    fn value(&self) -> u64 {
        Permuter::value(self)
    }

    #[inline]
    fn size(&self) -> usize {
        Permuter::size(self)
    }

    #[inline]
    fn get(&self, i: usize) -> usize {
        Permuter::get(self, i)
    }

    #[inline]
    fn back(&self) -> usize {
        Permuter::back(self)
    }

    #[inline]
    fn back_at_offset(&self, offset: usize) -> usize {
        Permuter::back_at_offset(self, offset)
    }

    #[inline]
    fn insert_from_back(&mut self, i: usize) -> usize {
        Permuter::insert_from_back(self, i)
    }

    #[inline]
    fn insert_from_back_immutable(&self, i: usize) -> (Self, usize) {
        Permuter::insert_from_back_immutable(self, i)
    }

    #[inline]
    fn swap_free_slots(&mut self, pos_i: usize, pos_j: usize) {
        Permuter::swap_free_slots(self, pos_i, pos_j)
    }

    #[inline]
    fn set_size(&mut self, n: usize) {
        Permuter::set_size(self, n)
    }

    #[inline]
    fn is_frozen_raw(raw: u64) -> bool {
        crate::freeze::LeafFreezeUtils::is_frozen::<WIDTH>(raw)
    }

    #[inline]
    fn freeze_raw(raw: u64) -> u64 {
        crate::freeze::LeafFreezeUtils::freeze_raw::<WIDTH>(raw)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ==================== Basic Tests ====================

    #[test]
    fn test_empty_permuter() {
        let p: Permuter<15> = Permuter::empty();
        assert_eq!(p.size(), 0);

        // Initial value: positions in reverse order, back() returns 0
        assert_eq!(p.back(), 0);
        assert_eq!(p.value(), 0x0123_4567_89AB_CDE0);
    }

    #[test]
    fn test_default_is_empty() {
        let p: Permuter<15> = Permuter::default();

        assert_eq!(p.size(), 0);
        assert_eq!(p.value(), Permuter::<15>::empty().value());
    }

    // ==================== make_sorted Tests ====================

    #[test]
    fn test_make_sorted_full() {
        // make_sorted(WIDTH) should give position i → slot i (sorted order)
        let p: Permuter<15> = Permuter::make_sorted(15);
        assert_eq!(p.size(), 15);

        // Verify position i maps to slot i
        for i in 0..15 {
            assert_eq!(p.get(i), i, "position {i} should map to slot {i}");
        }

        // Expected value: 0xEDCBA9876543210F (sorted_value | 15)
        assert_eq!(p.value(), 0xEDCB_A987_6543_210F);
    }

    #[test]
    fn test_make_sorted_partial() {
        let p: Permuter<15> = Permuter::make_sorted(3);
        assert_eq!(p.size(), 3);

        // Positions 0..3 are sorted (slot i at position i)
        assert_eq!(p.get(0), 0);
        assert_eq!(p.get(1), 1);
        assert_eq!(p.get(2), 2);

        // back() should return 3 (next slot to allocate)
        assert_eq!(p.back(), 3);
    }

    #[test]
    fn test_make_sorted_zero() {
        let p: Permuter<15> = Permuter::make_sorted(0);
        assert_eq!(p.size(), 0);

        // back() should return 0 (first slot to allocate)
        assert_eq!(p.back(), 0);
    }

    #[test]
    fn test_make_sorted_one() {
        let p: Permuter<15> = Permuter::make_sorted(1);
        assert_eq!(p.size(), 1);
        assert_eq!(p.get(0), 0);

        // back() should return 1 (next slot to allocate)
        assert_eq!(p.back(), 1);
    }

    // ==================== insert_from_back Tests ====================

    #[test]
    fn test_insert_from_back() {
        let mut p: Permuter<15> = Permuter::empty();
        assert_eq!(p.size(), 0);

        // Insert at position 0
        let slot0: usize = p.insert_from_back(0);
        assert_eq!(slot0, 0); // First slot allocated is 0
        assert_eq!(p.size(), 1);
        assert_eq!(p.get(0), 0);

        // Insert at position 0 again (shifts previous to position 1)
        let slot1: usize = p.insert_from_back(0);
        assert_eq!(slot1, 1); // Second slot allocated is 1
        assert_eq!(p.size(), 2);
        assert_eq!(p.get(0), 1); // New slot at position 0
        assert_eq!(p.get(1), 0); // Old slot shifted to position 1

        // Insert at position 1 (between the two)
        let slot2: usize = p.insert_from_back(1);
        assert_eq!(slot2, 2);
        assert_eq!(p.size(), 3);
        assert_eq!(p.get(0), 1);
        assert_eq!(p.get(1), 2); // New slot at position 1
        assert_eq!(p.get(2), 0); // Old position 1 shifted to position 2
    }

    #[test]
    fn test_insert_from_back_at_end() {
        let mut p: Permuter<15> = Permuter::empty();

        // Insert at end positions
        let slot0: usize = p.insert_from_back(0);
        let slot1: usize = p.insert_from_back(1); // Insert at end
        let slot2: usize = p.insert_from_back(2); // Insert at end

        assert_eq!(p.size(), 3);
        assert_eq!(p.get(0), slot0);
        assert_eq!(p.get(1), slot1);
        assert_eq!(p.get(2), slot2);
    }

    #[test]
    fn test_insert_fill_to_capacity() {
        let mut p: Permuter<15> = Permuter::empty();

        // Fill the permuter completely
        for i in 0..15 {
            let slot = p.insert_from_back(i);
            assert_eq!(slot, i);
        }

        assert_eq!(p.size(), 15);

        // Verify all positions
        for i in 0..15 {
            assert_eq!(p.get(i), i);
        }
    }

    // ==================== remove_to_back Tests ====================

    #[test]
    fn test_remove_to_back() {
        // Start with a sorted permuter of size 5
        let mut p: Permuter<15> = Permuter::make_sorted(5);

        // Positions: 0→0, 1→1, 2→2, 3→3, 4→4
        assert_eq!(p.size(), 5);

        // Remove position 2 (slot 2)
        p.remove_to_back(2);
        assert_eq!(p.size(), 4);

        // Positions 0, 1 unchanged
        assert_eq!(p.get(0), 0);
        assert_eq!(p.get(1), 1);

        // Position 2 now has what was at position 3
        assert_eq!(p.get(2), 3);

        // Position 3 now has what was at position 4
        assert_eq!(p.get(3), 4);

        // back() should return the removed slot (2)
        assert_eq!(p.back(), 2);
    }

    #[test]
    fn test_remove_to_back_first() {
        let mut p: Permuter<15> = Permuter::make_sorted(3);
        // Positions: 0→0, 1→1, 2→2

        // Remove position 0 (slot 0)
        p.remove_to_back(0);
        assert_eq!(p.size(), 2);

        // Position 0 now has slot 1
        assert_eq!(p.get(0), 1);

        // Position 1 now has slot 2
        assert_eq!(p.get(1), 2);

        // back() should return the removed slot (0)
        assert_eq!(p.back(), 0);
    }

    #[test]
    fn test_remove_to_back_last() {
        let mut p: Permuter<15> = Permuter::make_sorted(3);
        // Positions: 0→0, 1→1, 2→2

        // Remove position 2 (the last in-use position)
        p.remove_to_back(2);
        assert_eq!(p.size(), 2);

        // Positions 0, 1 unchanged
        assert_eq!(p.get(0), 0);
        assert_eq!(p.get(1), 1);

        // back() should return the removed slot (2)
        assert_eq!(p.back(), 2);
    }

    // ==================== remove Tests ====================

    #[test]
    fn test_remove_middle() {
        let mut p: Permuter<15> = Permuter::make_sorted(5);
        // Positions: 0→0, 1→1, 2→2, 3→3, 4→4

        // Remove position 2
        p.remove(2);
        assert_eq!(p.size(), 4);

        // Positions 0, 1 unchanged
        assert_eq!(p.get(0), 0);
        assert_eq!(p.get(1), 1);

        // Positions 2, 3 shifted down
        assert_eq!(p.get(2), 3);
        assert_eq!(p.get(3), 4);

        // Position 4 (first free) should have the removed slot
        assert_eq!(p.get(4), 2);
    }

    #[test]
    fn test_remove_first() {
        let mut p: Permuter<15> = Permuter::make_sorted(3);

        p.remove(0);
        assert_eq!(p.size(), 2);

        assert_eq!(p.get(0), 1);
        assert_eq!(p.get(1), 2);
        assert_eq!(p.get(2), 0); // Removed slot at position 2 (size)
    }

    #[test]
    fn test_remove_last() {
        let mut p: Permuter<15> = Permuter::make_sorted(3);

        // Remove the last element (fast path)
        p.remove(2);
        assert_eq!(p.size(), 2);

        assert_eq!(p.get(0), 0);
        assert_eq!(p.get(1), 1);

        // Position 2 still has slot 2 (unchanged by fast path)
        assert_eq!(p.get(2), 2);
    }

    // ==================== exchange Tests ====================

    #[test]
    fn test_exchange_basic() {
        let mut p: Permuter<15> = Permuter::make_sorted(5);
        // Positions: 0→0, 1→1, 2→2, 3→3, 4→4

        // Exchange positions 1 and 3
        p.exchange(1, 3);

        assert_eq!(p.size(), 5); // Size unchanged
        assert_eq!(p.get(0), 0);
        assert_eq!(p.get(1), 3); // Swapped
        assert_eq!(p.get(2), 2);
        assert_eq!(p.get(3), 1); // Swapped
        assert_eq!(p.get(4), 4);
    }

    #[test]
    fn test_exchange_same() {
        let mut p: Permuter<15> = Permuter::make_sorted(5);
        let original: u64 = p.value();

        // Exchange same position (no-op)
        p.exchange(2, 2);

        assert_eq!(p.value(), original);
    }

    #[test]
    fn test_exchange_adjacent() {
        let mut p: Permuter<15> = Permuter::make_sorted(5);

        // Exchange adjacent positions
        p.exchange(2, 3);

        assert_eq!(p.get(2), 3);
        assert_eq!(p.get(3), 2);
    }

    #[test]
    fn test_exchange_first_last() {
        let mut p: Permuter<15> = Permuter::make_sorted(15);

        // Exchange first and last positions
        p.exchange(0, 14);

        assert_eq!(p.get(0), 14);
        assert_eq!(p.get(14), 0);
    }

    // ==================== rotate Tests ====================

    #[test]
    fn test_rotate_basic() {
        let mut p: Permuter<7> = Permuter::make_sorted(7);
        // Positions: 0→0, 1→1, 2→2, 3→3, 4→4, 5→5, 6→6

        // Rotate starting at position 2 by 2 positions
        p.rotate(2, 4);

        // Positions 0, 1 unchanged
        assert_eq!(p.get(0), 0);
        assert_eq!(p.get(1), 1);

        // Positions 2+ are rotated
        // The rotation shifts by (j-i) = 2 positions
    }

    #[test]
    fn test_rotate_no_op() {
        let mut p: Permuter<15> = Permuter::make_sorted(5);
        let original: u64 = p.value();

        // rotate(i, i) is a no-op
        p.rotate(2, 2);
        assert_eq!(p.value(), original);

        // rotate(WIDTH, j) is a no-op
        p.rotate(15, 15);
        assert_eq!(p.value(), original);
    }

    // ==================== Roundtrip Tests ====================

    #[test]
    fn test_insert_remove_roundtrip() {
        let mut p: Permuter<15> = Permuter::empty();

        // Insert 5 elements
        for i in 0..5 {
            let _ = p.insert_from_back(i);
        }
        assert_eq!(p.size(), 5);

        // Remove them one by one using remove_to_back
        for _ in 0..5 {
            p.remove_to_back(0);
        }
        assert_eq!(p.size(), 0);

        // Should still have valid invariants
        p.debug_assert_valid();
    }

    #[test]
    fn test_insert_remove_roundtrip_alt() {
        let mut p: Permuter<15> = Permuter::empty();

        // Insert 5 elements
        for i in 0..5 {
            let _ = p.insert_from_back(i);
        }

        // Remove them using remove() (different from remove_to_back)
        for _ in 0..5 {
            p.remove(0);
        }
        assert_eq!(p.size(), 0);

        p.debug_assert_valid();
    }

    // ==================== WIDTH Variant Tests ====================

    #[test]
    fn test_compact_permuter() {
        // Test WIDTH=7 variant
        let mut p: Permuter<7> = Permuter::empty();
        assert_eq!(p.size(), 0);
        assert_eq!(p.back(), 0);

        let slot: usize = p.insert_from_back(0);
        assert_eq!(slot, 0);
        assert_eq!(p.size(), 1);

        let sorted: Permuter<7> = Permuter::make_sorted(7);
        assert_eq!(sorted.size(), 7);

        for i in 0..7 {
            assert_eq!(sorted.get(i), i);
        }
    }

    #[test]
    fn test_width_3_permuter() {
        let mut p: Permuter<3> = Permuter::empty();
        assert_eq!(p.size(), 0);

        let _ = p.insert_from_back(0);
        let _ = p.insert_from_back(0);
        let _ = p.insert_from_back(0);

        assert_eq!(p.size(), 3);
        p.debug_assert_valid();
    }

    // ==================== Edge Case Tests ====================

    #[test]
    fn test_set_size() {
        let mut p: Permuter<15> = Permuter::make_sorted(10);
        assert_eq!(p.size(), 10);

        p.set_size(5);
        assert_eq!(p.size(), 5);

        p.set_size(0);
        assert_eq!(p.size(), 0);
    }

    #[test]
    fn test_value_accessor() {
        let p: Permuter<15> = Permuter::empty();
        let v: u64 = p.value();

        // Create another permuter with same value
        let p2: Permuter<15> = Permuter::empty();
        assert_eq!(p.value(), p2.value());
        assert_eq!(v, 0x0123_4567_89AB_CDE0);
    }

    #[test]
    fn test_clone_and_eq() {
        let p1: Permuter<15> = Permuter::make_sorted(5);
        let p2: Permuter = p1; // Copy

        assert_eq!(p1, p2);
        assert_eq!(p1.value(), p2.value());
    }

    // ==================== insert_from_back_immutable Tests ====================

    #[test]
    fn test_insert_from_back_immutable_basic() {
        let p: Permuter<15> = Permuter::empty();
        assert_eq!(p.size(), 0);

        // Insert at position 0 (immutable)
        let (new_p, slot) = p.insert_from_back_immutable(0);

        // Original unchanged
        assert_eq!(p.size(), 0);

        // New permuter has the insert
        assert_eq!(new_p.size(), 1);
        assert_eq!(slot, 0);
        assert_eq!(new_p.get(0), 0);
    }

    #[test]
    fn test_insert_from_back_immutable_matches_mutable() {
        // Verify immutable version produces same result as mutable version
        let original: Permuter<15> = Permuter::make_sorted(5);

        // Mutable insert
        let mut mutable = original;
        let mutable_slot = mutable.insert_from_back(2);

        // Immutable insert
        let (immutable, immutable_slot) = original.insert_from_back_immutable(2);

        // Should produce identical results
        assert_eq!(mutable_slot, immutable_slot);
        assert_eq!(mutable.value(), immutable.value());
        assert_eq!(mutable.size(), immutable.size());

        for i in 0..mutable.size() {
            assert_eq!(mutable.get(i), immutable.get(i));
        }
    }

    #[test]
    fn test_insert_from_back_immutable_chain() {
        // Test chaining multiple immutable inserts
        let p0: Permuter<15> = Permuter::empty();

        let (p1, slot0) = p0.insert_from_back_immutable(0);
        let (p2, slot1) = p1.insert_from_back_immutable(0);
        let (p3, slot2) = p2.insert_from_back_immutable(1);

        // Original unchanged
        assert_eq!(p0.size(), 0);

        // Each step added one
        assert_eq!(p1.size(), 1);
        assert_eq!(p2.size(), 2);
        assert_eq!(p3.size(), 3);

        // Slots allocated in order
        assert_eq!(slot0, 0);
        assert_eq!(slot1, 1);
        assert_eq!(slot2, 2);

        // Final permuter has correct structure
        assert_eq!(p3.get(0), 1); // slot1 at position 0
        assert_eq!(p3.get(1), 2); // slot2 at position 1
        assert_eq!(p3.get(2), 0); // slot0 at position 2 (shifted)
    }

    #[test]
    fn test_insert_from_back_immutable_for_cas() {
        // Simulate CAS usage: compute new value, compare old
        let current: Permuter<15> = Permuter::make_sorted(3);
        let current_value = current.value();

        let (new_perm, slot) = current.insert_from_back_immutable(1);

        // Simulate CAS: old value should match current
        assert_eq!(current.value(), current_value);

        // New value is different
        assert_ne!(new_perm.value(), current_value);

        // Could now do: compare_exchange(current_value, new_perm.value())
        assert_eq!(slot, 3); // Next slot after sorted(3)
        assert_eq!(new_perm.size(), 4);
    }
}
