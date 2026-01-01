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

/// Mask for extracting size (lower 4 bits).
const SIZE_MASK: u64 = 0xF;

use crate::{leaf_trait::TreePermutation, suffix::PermutationProvider};

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
    #[inline(always)]
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
    #[must_use]
    #[inline(always)]
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
    /// # Free Slot Ordering
    ///
    /// Free slots (positions `n..WIDTH`) are stored in **reverse order** so that
    /// `back()` returns slot `n` (the next slot to allocate). This maintains the
    /// invariant that slots are allocated in order: 0, 1, 2, ...
    ///
    /// ## Example (WIDTH=15, n=5)
    ///
    /// ```text
    /// Positions 0-4 (in use):  slot 0, slot 1, slot 2, slot 3, slot 4
    /// Positions 5-14 (free):   slot 14, slot 13, ..., slot 6, slot 5
    /// back() = get(14) = slot 5  ← next to allocate
    /// ```
    ///
    /// # Panics
    ///
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
        // so back() = get(WIDTH - 1) returns slot n (next to allocate)

        // Build the free slot portion: positions n..WIDTH hold slots n..WIDTH-1
        // in reverse order so back() = slot n
        let mut value: u64 = n as u64;

        // Copy sorted positions 0..n
        let sorted_mask: u64 = ((1u64 << (n * 4)) - 1) << 4;
        value |= sorted & sorted_mask;

        // Fill positions n..WIDTH with remaining slots in REVERSE order.
        // This ensures back() returns the lowest free slot (slot n).
        //
        // Example for n=5, WIDTH=15:
        //   Position 5:  slot = 15 - 1 - (5-5) = 14
        //   Position 6:  slot = 15 - 1 - (6-5) = 13
        //   ...
        //   Position 14: slot = 15 - 1 - (14-5) = 5  ← back() returns this
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
    ///
    /// # Implementation Note
    ///
    /// Uses the same XOR swap trick as [`Self::exchange`]. The XOR swap works on
    /// 4-bit slot values without requiring a temporary variable.
    #[inline(always)]
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

        // XOR swap trick: swap two 4-bit values without a temporary.
        // 1. diff = XOR of the two 4-bit values
        // 2. XOR diff into both positions → values are swapped
        // Same algorithm as exchange(), kept inline for performance.
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
    #[inline(always)]
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
    #[inline(always)]
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
    #[inline(always)]
    pub fn remove_to_back(&mut self, i: usize) {
        debug_assert!(i < self.size(), "remove_to_back: i >= size");

        // mask covers bits for positions >= i (not including size bits for positions < i)
        // i_shift = bit offset of position i = (i + 1) * 4
        let i_shift: usize = (i + 1) * 4;
        let mask: u64 = !((1u64 << i_shift) - 1);

        // Clear unused bits above WIDTH positions (for 64-bit safety).
        //
        // For WIDTH=15: width_shift = 16 * 4 = 64, which would cause `1u64 << 64`
        // to overflow (in Rust, this wraps to 1, not 0). We use u64::MAX instead.
        // For WIDTH < 15: we compute the proper mask to clear upper bits.
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
    #[inline(always)]
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
    #[inline(always)]
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
    /// # Example
    ///
    /// ```text
    /// rotate(2, 4) with WIDTH=7:
    ///   Before: [A, B, C, D, E, F, G]
    ///                ^     ^
    ///                i     j
    ///   After:  [A, B, E, F, G, C, D]
    ///
    /// Positions 2..7 rotated left by 2 places (j - i = 2).
    /// Elements at positions 2,3 wrap around to positions 5,6.
    /// ```
    ///
    /// # Panics
    /// Panics in debug mode if `i > j` or `j > WIDTH`.
    ///
    /// # Algorithm (matches C++ `kpermuter::rotate`)
    #[inline(always)]
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
    #[inline(always)]
    fn size(&self) -> usize {
        self.size()
    }

    #[inline(always)]
    fn get(&self, i: usize) -> usize {
        self.get(i)
    }
}

// ============================================================================
//  TreePermutation Implementation
// ============================================================================

// ============================================================================
//  AtomicPermuter (u64-based atomic wrapper)
// ============================================================================

use std::sync::atomic::{AtomicU64, Ordering};

/// Atomic wrapper for [`Permuter<WIDTH>`] where WIDTH <= 15.
///
/// Uses standard library `AtomicU64` since `Permuter<WIDTH>` fits in 64 bits.
/// This is simpler than `AtomicPermuter24` which requires `portable-atomic` for u128.
#[derive(Debug)]
#[repr(transparent)]
pub struct AtomicPermuter<const WIDTH: usize = 15> {
    inner: AtomicU64,
}

impl<const WIDTH: usize> AtomicPermuter<WIDTH> {
    /// Create with empty permutation.
    #[must_use]
    #[inline(always)]
    pub const fn new() -> Self {
        Self {
            inner: AtomicU64::new(Permuter::<WIDTH>::empty().value()),
        }
    }

    /// Create from existing permuter.
    #[must_use]
    #[inline(always)]
    pub const fn from_permuter(perm: Permuter<WIDTH>) -> Self {
        Self {
            inner: AtomicU64::new(perm.value),
        }
    }

    /// Load with ordering.
    #[inline(always)]
    pub fn load(&self, order: Ordering) -> Permuter<WIDTH> {
        Permuter::from_value(self.inner.load(order))
    }

    /// Store with ordering.
    #[inline(always)]
    pub fn store(&self, val: Permuter<WIDTH>, order: Ordering) {
        self.inner.store(val.value, order);
    }

    /// Compare-and-exchange.
    ///
    /// # Errors
    ///
    /// Returns `Err` with the current value if the comparison failed.
    #[inline(always)]
    pub fn compare_exchange(
        &self,
        expected: Permuter<WIDTH>,
        new: Permuter<WIDTH>,
        success: Ordering,
        failure: Ordering,
    ) -> Result<Permuter<WIDTH>, Permuter<WIDTH>> {
        self.inner
            .compare_exchange(expected.value, new.value, success, failure)
            .map(Permuter::from_value)
            .map_err(Permuter::from_value)
    }

    /// Compare-and-exchange weak.
    ///
    /// # Errors
    ///
    /// Returns `Err` with the current value if the comparison failed, or spuriously.
    #[inline(always)]
    pub fn compare_exchange_weak(
        &self,
        expected: Permuter<WIDTH>,
        new: Permuter<WIDTH>,
        success: Ordering,
        failure: Ordering,
    ) -> Result<Permuter<WIDTH>, Permuter<WIDTH>> {
        self.inner
            .compare_exchange_weak(expected.value, new.value, success, failure)
            .map(Permuter::from_value)
            .map_err(Permuter::from_value)
    }

    /// Load raw u64 value.
    #[inline(always)]
    pub fn load_raw(&self, order: Ordering) -> u64 {
        self.inner.load(order)
    }

    /// Store raw u64 value.
    #[inline(always)]
    pub fn store_raw(&self, val: u64, order: Ordering) {
        self.inner.store(val, order);
    }

    /// CAS on raw value.
    ///
    /// # Errors
    ///
    /// Returns `Err` with the current value if the comparison failed.
    #[inline(always)]
    pub fn compare_exchange_raw(
        &self,
        expected: u64,
        new: u64,
        success: Ordering,
        failure: Ordering,
    ) -> Result<u64, u64> {
        self.inner.compare_exchange(expected, new, success, failure)
    }
}

impl<const WIDTH: usize> Default for AtomicPermuter<WIDTH> {
    fn default() -> Self {
        Self::new()
    }
}

/// Type alias for the standard 15-slot atomic permuter.
pub type AtomicPermuter15 = AtomicPermuter<15>;

// ============================================================================
//  TreePermutation Implementation
// ============================================================================

impl<const WIDTH: usize> TreePermutation for Permuter<WIDTH> {
    type Raw = u64;
    const WIDTH: usize = WIDTH;

    #[inline(always)]
    fn empty() -> Self {
        Self::empty()
    }

    #[inline(always)]
    fn make_sorted(n: usize) -> Self {
        Self::make_sorted(n)
    }

    #[inline(always)]
    fn from_value(raw: u64) -> Self {
        Self::from_value(raw)
    }

    #[inline(always)]
    fn value(&self) -> u64 {
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
    fn remove(&mut self, i: usize) {
        Self::remove(self, i);
    }
}

#[cfg(test)]
mod unit_tests;
