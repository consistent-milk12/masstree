use std::cell::UnsafeCell;
use std::cmp::Ordering;
use std::fmt::{self as StdFmt, Debug, Formatter};
use std::ptr as StdPtr;
use std::sync::atomic::{AtomicU16, AtomicU32, AtomicU8, Ordering as AtomicOrdering};

use super::{SuffixBag, TreePermutation};

/// Number of slots (matches `WIDTH_15` leaf node).
const WIDTH: usize = 15;

/// Inline suffix data capacity.
///
/// Default: 512 bytes (fewer heap allocations, better insert throughput).
/// With `small-suffix-capacity` feature: 256 bytes (smaller sidecar heap allocation).
#[cfg(not(feature = "small-suffix-capacity"))]
const CAPACITY: usize = 512;

#[cfg(feature = "small-suffix-capacity")]
const CAPACITY: usize = 256;

const U16_MAX: usize = u16::MAX as usize;

/// Atomic ordering for reading slot metadata (pairs with Release stores).
const READ_ORD: AtomicOrdering = AtomicOrdering::Acquire;

/// Atomic ordering for writing slot metadata (pairs with Acquire loads).
const WRITE_ORD: AtomicOrdering = AtomicOrdering::Release;

/// Relaxed ordering for non-synchronizing accesses (under lock).
const RELAXED: AtomicOrdering = AtomicOrdering::Relaxed;

// ============================================================================
//  InlineSlotMeta
// ============================================================================

/// Metadata for a single slot's suffix in inline storage.
///
/// Uses `u16` for offset to keep metadata compact (4 bytes per slot).
/// Maximum inline capacity is 65535 bytes.
///
/// Packed into `AtomicU32` for concurrent access:
/// - High 16 bits: offset
/// - Low 16 bits: length
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

    /// Packed representation of EMPTY for atomic initialization.
    const EMPTY_PACKED: u32 = Self::EMPTY.pack();

    /// Check if this slot has a suffix.
    #[inline(always)]
    const fn has_suffix(self) -> bool {
        self.offset != u16::MAX
    }

    /// Pack into u32 for atomic storage.
    #[inline(always)]
    const fn pack(self) -> u32 {
        ((self.offset as u32) << 16) | (self.len as u32)
    }

    /// Unpack from u32 loaded atomically.
    #[inline(always)]
    #[expect(
        clippy::cast_possible_truncation,
        reason = "Intentional: extracting u16 fields from packed u32"
    )]
    const fn unpack(packed: u32) -> Self {
        Self {
            offset: (packed >> 16) as u16,
            len: packed as u16,
        }
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
/// # Concurrency
///
/// This structure uses interior mutability for safe concurrent access:
/// - Slot metadata uses `AtomicU32` (Acquire/Release ordering)
/// - Size and count use atomics for writer updates under lock
/// - Data buffer uses `UnsafeCell` (writers hold lock, readers use version validation)
///
/// **Writers** must hold the leaf lock before calling mutation methods.
/// **Readers** may call read methods without the lock, using OCC validation.
///
/// # Design
///
/// - Embedded in the leaf node (no heap allocation)
/// - Fixed capacity: 256 bytes for data, 15 slots
/// - Append-only with slot reuse when new suffix fits in old space
/// - When full, caller must drain to external `SuffixBag`
///
/// # Memory Layout
///
/// ```text
/// InlineSuffixBag (320 bytes total)
/// ├── slots: [AtomicU32; 15]        // 60 bytes (4 bytes each)
/// ├── size: AtomicU16               // 2 bytes
/// ├── suffix_count: AtomicU8        // 1 byte
/// ├── _pad: u8                      // 1 byte padding
/// └── data: UnsafeCell<[u8; 256]>   // 256 bytes
/// ```
#[repr(C)]
pub struct InlineSuffixBag {
    /// Per-slot metadata: packed (offset, length) pairs.
    /// Accessed atomically for concurrent read/write safety.
    slots: [AtomicU32; WIDTH],

    /// Current write position in data buffer.
    /// Writers update under lock, readers don't need this.
    size: AtomicU16,

    /// Cached count of slots with suffixes.
    /// Writers update under lock, readers don't need this.
    suffix_count: AtomicU8,

    /// Padding for alignment.
    _pad: u8,

    /// Fixed-size data buffer.
    /// Writers update under lock, readers access with OCC validation.
    data: UnsafeCell<[u8; CAPACITY]>,
}

// SAFETY: InlineSuffixBag is Sync because:
// - slots use AtomicU32 for thread-safe access
// - size/suffix_count use atomics (writers hold lock)
// - data is protected by OCC protocol (readers validate version after read)
unsafe impl Sync for InlineSuffixBag {}

impl Debug for InlineSuffixBag {
    fn fmt(&self, f: &mut Formatter<'_>) -> StdFmt::Result {
        f.debug_struct("InlineSuffixBag")
            .field("size", &self.size.load(RELAXED))
            .field("suffix_count", &self.suffix_count.load(RELAXED))
            .field("capacity", &CAPACITY)
            .finish_non_exhaustive()
    }
}

impl InlineSuffixBag {
    // ========================================================================
    //  Constructor
    // ========================================================================

    /// Create an empty inline suffix bag.
    #[must_use]
    #[inline(always)]
    pub const fn new() -> Self {
        Self {
            slots: [
                AtomicU32::new(InlineSlotMeta::EMPTY_PACKED),
                AtomicU32::new(InlineSlotMeta::EMPTY_PACKED),
                AtomicU32::new(InlineSlotMeta::EMPTY_PACKED),
                AtomicU32::new(InlineSlotMeta::EMPTY_PACKED),
                AtomicU32::new(InlineSlotMeta::EMPTY_PACKED),
                AtomicU32::new(InlineSlotMeta::EMPTY_PACKED),
                AtomicU32::new(InlineSlotMeta::EMPTY_PACKED),
                AtomicU32::new(InlineSlotMeta::EMPTY_PACKED),
                AtomicU32::new(InlineSlotMeta::EMPTY_PACKED),
                AtomicU32::new(InlineSlotMeta::EMPTY_PACKED),
                AtomicU32::new(InlineSlotMeta::EMPTY_PACKED),
                AtomicU32::new(InlineSlotMeta::EMPTY_PACKED),
                AtomicU32::new(InlineSlotMeta::EMPTY_PACKED),
                AtomicU32::new(InlineSlotMeta::EMPTY_PACKED),
                AtomicU32::new(InlineSlotMeta::EMPTY_PACKED),
            ],
            size: AtomicU16::new(0),
            suffix_count: AtomicU8::new(0),
            _pad: 0,
            data: UnsafeCell::new([0u8; CAPACITY]),
        }
    }

    // ========================================================================
    //  Slot Access (Atomic)
    // ========================================================================

    /// Load slot metadata atomically.
    #[inline(always)]
    #[expect(clippy::indexing_slicing, reason = "Bounds checked via debug_assert")]
    fn load_meta(&self, slot: usize) -> InlineSlotMeta {
        debug_assert!(slot < WIDTH, "slot {slot} >= WIDTH {WIDTH}");
        // SAFETY: slot bounds checked above
        let packed: u32 = self.slots[slot].load(READ_ORD);
        InlineSlotMeta::unpack(packed)
    }

    /// Store slot metadata atomically.
    #[inline(always)]
    #[expect(clippy::indexing_slicing, reason = "Bounds checked via debug_assert")]
    fn store_meta(&self, slot: usize, meta: InlineSlotMeta) {
        debug_assert!(slot < WIDTH, "slot {slot} >= WIDTH {WIDTH}");
        // SAFETY: slot bounds checked above
        self.slots[slot].store(meta.pack(), WRITE_ORD);
    }

    // ========================================================================
    //  Capacity & Size
    // ========================================================================

    /// Return the fixed capacity of this inline bag (256 bytes).
    #[must_use]
    #[inline(always)]
    pub const fn capacity(&self) -> usize {
        CAPACITY
    }

    /// Return the number of bytes currently used.
    #[must_use]
    #[inline(always)]
    pub fn used(&self) -> usize {
        self.size.load(RELAXED) as usize
    }

    /// Return the remaining capacity.
    #[must_use]
    #[inline(always)]
    pub fn remaining(&self) -> usize {
        CAPACITY - self.used()
    }

    /// Return the number of slots that have suffixes.
    #[must_use]
    #[inline(always)]
    pub fn count(&self) -> usize {
        self.suffix_count.load(RELAXED) as usize
    }

    // ========================================================================
    //  Fallible Operations
    // ========================================================================

    /// Drain inline suffixes to external bag (normal operation).
    ///
    /// Uses the permutation to find active slots. This is the common case
    /// for suffix overflow during normal inserts.
    ///
    /// # Atomicity
    ///
    /// This function does NOT clear inline state. The caller must:
    /// 1. Store the returned external bag pointer (Release)
    /// 2. Only then is it safe to clear inline state (if desired)
    ///
    /// This ensures readers always see either:
    /// - Valid inline data, OR
    /// - Valid external data (after Acquire load of external pointer)
    ///
    /// The inline state becomes "orphaned" metadata after drain, but this is
    /// safe - readers will find the suffix in external storage.
    ///
    /// Aborts on allocation failure (standard Rust OOM behavior).
    pub fn drain_to_external(
        &self,
        perm: &impl TreePermutation,
        new_slot: usize,
        new_suffix: &[u8],
    ) -> SuffixBag {
        // Pass 1: Calculate required capacity and collect slot data
        let mut required_capacity: usize = new_suffix.len();
        let perm_size: usize = perm.size();

        // Stack-allocated storage for slots to copy
        let mut slots_to_copy: [(usize, usize, usize); WIDTH] = [(0, 0, 0); WIDTH];
        let mut copy_count: usize = 0;

        // SAFETY: We hold the lock, so data buffer is stable
        let data: &[u8; CAPACITY] = unsafe { &*self.data.get() };

        #[expect(clippy::indexing_slicing)]
        for i in 0..perm_size {
            let slot: usize = perm.get(i);

            if (slot != new_slot) && (slot < WIDTH) {
                let meta: InlineSlotMeta = self.load_meta(slot);

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

        // Allocate external bag with capacity (aborts on OOM)
        let mut external: SuffixBag = SuffixBag::with_capacity(required_capacity);

        // Pass 2: Copy suffixes to external bag using collected data
        for &(slot, start, len) in &slots_to_copy[..copy_count] {
            // SAFETY: start and len come from valid InlineSlotMeta entries
            let suffix: &[u8] = &data[start..(start + len)];
            external.assign(slot, suffix);
        }

        // Assign new suffix
        external.assign(new_slot, new_suffix);

        // NOTE: We deliberately do NOT clear inline state here.
        // The caller must publish the external pointer first, then
        // may optionally clear inline state.

        external
    }

    /// Drain inline suffixes to external bag, reusing a pre-allocated buffer.
    ///
    /// # Panics
    ///
    /// On OOM, while trying to allocate for [`SuffixBag`].
    pub fn drain_to_external_with_vec(
        &self,
        perm: &impl TreePermutation,
        new_slot: usize,
        new_suffix: &[u8],
        buffer: Vec<u8>,
    ) -> SuffixBag {
        // Pass 1: Calculate required capacity and collect slot data.
        let mut required_capacity: usize = new_suffix.len();
        let perm_size: usize = perm.size();
        let mut slots_to_copy: [(usize, usize, usize); WIDTH] = [(0, 0, 0); WIDTH];
        let mut copy_count: usize = 0;

        // SAFETY: We hold the lock, so data buffer is stable
        let data: &[u8; CAPACITY] = unsafe { &*self.data.get() };

        #[expect(clippy::indexing_slicing)]
        for i in 0..perm_size {
            let slot: usize = perm.get(i);

            if (slot != new_slot) && (slot < WIDTH) {
                let meta: InlineSlotMeta = self.load_meta(slot);

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

        let mut external: SuffixBag = SuffixBag::from_vec(buffer);

        // Reserve capacity if needed (aborts on OOM)
        if external.capacity() < required_capacity {
            external.reserve(required_capacity - external.capacity());
        }

        for &(slot, start, len) in &slots_to_copy[..copy_count] {
            let suffix: &[u8] = &data[start..(start + len)];
            external.assign(slot, suffix);
        }

        external.assign(new_slot, new_suffix);
        external
    }

    /// Drain inline suffixes to external bag during node initialization.
    ///
    /// Unlike `drain_to_external`, this assumes slots `0..new_slot` are
    /// already filled sequentially and doesn't rely on the permutation.
    /// Used during split operations when the new node's permutation hasn't
    /// been set up yet.
    ///
    /// # Atomicity
    ///
    /// Same as `drain_to_external` - does NOT clear inline state.
    ///
    /// Aborts on allocation failure (standard Rust OOM behavior).
    #[cold]
    pub fn drain_to_external_init(&self, new_slot: usize, new_suffix: &[u8]) -> SuffixBag {
        // Calculate required capacity
        let mut required_capacity: usize = new_suffix.len();

        // Collect existing suffixes (slots 0..new_slot filled sequentially)
        let mut slots_to_copy: [(usize, usize, usize); WIDTH] = [(0, 0, 0); WIDTH];
        let mut copy_count: usize = 0;

        // SAFETY: We hold the lock, so data buffer is stable
        let data: &[u8; CAPACITY] = unsafe { &*self.data.get() };

        #[expect(clippy::indexing_slicing)]
        for slot in 0..new_slot {
            if slot >= WIDTH {
                break;
            }

            let meta: InlineSlotMeta = self.load_meta(slot);

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

        // Allocate external bag (aborts on OOM)
        let mut external: SuffixBag = SuffixBag::with_capacity(required_capacity);

        // Copy existing suffixes
        for &(slot, start, len) in &slots_to_copy[..copy_count] {
            let suffix: &[u8] = &data[start..(start + len)];
            external.assign(slot, suffix);
        }

        // Assign new suffix
        external.assign(new_slot, new_suffix);

        external
    }

    // ========================================================================
    //  Slot Access (Read)
    // ========================================================================

    /// Check if a slot has a suffix.
    ///
    /// # Panics
    ///
    /// Panics in debug mode if `slot >= 15`.
    #[must_use]
    #[inline(always)]
    pub fn has_suffix(&self, slot: usize) -> bool {
        debug_assert!(slot < WIDTH, "slot {slot} >= WIDTH {WIDTH}");
        self.load_meta(slot).has_suffix()
    }

    /// Get the suffix for a slot, or `None` if no suffix.
    ///
    /// # Safety Contract
    ///
    /// Readers must use OCC validation after calling this method.
    /// The returned slice is only valid if the version check passes.
    ///
    /// # Panics
    ///
    /// Panics in debug mode if `slot >= 15`.
    #[must_use]
    #[inline(always)]
    pub fn get(&self, slot: usize) -> Option<&[u8]> {
        debug_assert!(slot < WIDTH, "slot {slot} >= WIDTH {WIDTH}");

        let meta: InlineSlotMeta = self.load_meta(slot);

        if !meta.has_suffix() {
            return None;
        }

        let start: usize = meta.offset as usize;
        let len: usize = meta.len as usize;

        // INVARIANT: Valid metadata points to valid data range.
        debug_assert!(
            start + len <= CAPACITY,
            "inline suffix metadata points past capacity: {} > {CAPACITY}",
            start + len
        );

        // Use raw pointer to create slice directly, avoiding a retag of the entire array.
        // This allows concurrent readers while a writer (holding the lock) modifies
        // a different region. Readers use OCC validation to detect modifications.
        //
        // SAFETY:
        // - Metadata bounds are validated by invariant (debug_assert above)
        // - Readers retry if version changed (OCC protocol)
        // - Writers hold the lock, ensuring no concurrent writes
        let data_ptr: *const u8 = self.data.get().cast::<u8>();
        let suffix: &[u8] = unsafe { std::slice::from_raw_parts(data_ptr.add(start), len) };

        Some(suffix)
    }

    /// Get the suffix for a slot, or empty slice if no suffix.
    #[must_use]
    #[inline(always)]
    pub fn get_or_empty(&self, slot: usize) -> &[u8] {
        self.get(slot).unwrap_or(&[])
    }

    // ========================================================================
    //  Suffix Assignment (Write - requires lock)
    // ========================================================================

    /// Try to assign a suffix to a slot in-place.
    ///
    /// This is the fast path matching C++ `stringbag::assign()`:
    /// 1. If new suffix fits in old slot's space, reuse it
    /// 2. Otherwise, append to end if there's room
    /// 3. If no room, return `false` (caller should use external bag)
    ///
    /// # Safety
    ///
    /// Caller must hold the leaf lock.
    ///
    /// # Returns
    ///
    /// - `true` if the suffix was assigned successfully
    /// - `false` if there's not enough capacity (caller should drain to external)
    ///
    /// # Panics
    ///
    /// Panics if `slot >= 15` or if suffix length exceeds `u16::MAX`.
    #[inline]
    pub fn try_assign(&self, slot: usize, suffix: &[u8]) -> bool {
        debug_assert!(slot < WIDTH, "slot {slot} >= WIDTH {WIDTH}");

        let suffix_len: usize = suffix.len();

        // Suffix must fit in u16
        if suffix_len > U16_MAX {
            return false;
        }

        let meta: InlineSlotMeta = self.load_meta(slot);

        // Use raw pointer to avoid creating &mut that conflicts with concurrent readers.
        // SAFETY: We hold the lock, exclusive write access to data buffer.
        // Readers use OCC validation to detect our writes.
        let data_ptr: *mut u8 = self.data.get().cast::<u8>();

        // Fast Path 1: Reuse existing slot if new suffix fits in old space
        if meta.has_suffix() && (suffix_len <= (meta.len as usize)) {
            let start: usize = meta.offset as usize;
            // SAFETY: start + suffix_len <= meta.len <= CAPACITY (invariant)
            unsafe {
                StdPtr::copy_nonoverlapping(suffix.as_ptr(), data_ptr.add(start), suffix_len);
            }

            #[expect(clippy::cast_possible_truncation, reason = "len checked above")]
            self.store_meta(
                slot,
                InlineSlotMeta {
                    offset: meta.offset,
                    len: suffix_len as u16,
                },
            );

            // Count unchanged, slot already had suffix
            return true;
        }

        // Fast Path 2: Append to end if there's room
        let current_size: usize = self.size.load(RELAXED) as usize;

        if (current_size + suffix_len) <= CAPACITY {
            // SAFETY: current_size + suffix_len <= CAPACITY (checked above)
            unsafe {
                StdPtr::copy_nonoverlapping(
                    suffix.as_ptr(),
                    data_ptr.add(current_size),
                    suffix_len,
                );
            }

            // Update count if this is a new suffix
            if !meta.has_suffix() {
                self.suffix_count.fetch_add(1, RELAXED);
            }

            #[expect(
                clippy::cast_possible_truncation,
                reason = "offset and len checked to fit"
            )]
            {
                self.store_meta(
                    slot,
                    InlineSlotMeta {
                        offset: current_size as u16,
                        len: suffix_len as u16,
                    },
                );

                self.size.store((current_size + suffix_len) as u16, RELAXED);
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
    ///
    /// # Safety
    ///
    /// Caller must hold the leaf lock.
    #[inline(always)]
    pub fn clear(&self, slot: usize) {
        debug_assert!(slot < WIDTH, "slot {slot} >= WIDTH {WIDTH}");

        if self.load_meta(slot).has_suffix() {
            self.suffix_count.fetch_sub(1, RELAXED);
        }

        self.store_meta(slot, InlineSlotMeta::EMPTY);
    }

    /// Clear all slots and reset size to zero.
    ///
    /// Used after draining to an external bag.
    ///
    /// # Safety
    ///
    /// Caller must hold the leaf lock.
    #[inline(always)]
    pub fn clear_all(&self) {
        for slot in 0..WIDTH {
            self.store_meta(slot, InlineSlotMeta::EMPTY);
        }
        self.size.store(0, RELAXED);
        self.suffix_count.store(0, RELAXED);
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
    pub fn suffix_compare(&self, slot: usize, suffix: &[u8]) -> Option<Ordering> {
        self.get(slot).map(|stored: &[u8]| stored.cmp(suffix))
    }
}

impl Default for InlineSuffixBag {
    #[inline(always)]
    fn default() -> Self {
        Self::new()
    }
}

impl Clone for InlineSuffixBag {
    #[inline(always)]
    fn clone(&self) -> Self {
        // SAFETY: Clone is typically called under exclusive access or during init
        let data: [u8; CAPACITY] = unsafe { *self.data.get() };
        Self {
            slots: [
                AtomicU32::new(self.slots[0].load(RELAXED)),
                AtomicU32::new(self.slots[1].load(RELAXED)),
                AtomicU32::new(self.slots[2].load(RELAXED)),
                AtomicU32::new(self.slots[3].load(RELAXED)),
                AtomicU32::new(self.slots[4].load(RELAXED)),
                AtomicU32::new(self.slots[5].load(RELAXED)),
                AtomicU32::new(self.slots[6].load(RELAXED)),
                AtomicU32::new(self.slots[7].load(RELAXED)),
                AtomicU32::new(self.slots[8].load(RELAXED)),
                AtomicU32::new(self.slots[9].load(RELAXED)),
                AtomicU32::new(self.slots[10].load(RELAXED)),
                AtomicU32::new(self.slots[11].load(RELAXED)),
                AtomicU32::new(self.slots[12].load(RELAXED)),
                AtomicU32::new(self.slots[13].load(RELAXED)),
                AtomicU32::new(self.slots[14].load(RELAXED)),
            ],
            size: AtomicU16::new(self.size.load(RELAXED)),
            suffix_count: AtomicU8::new(self.suffix_count.load(RELAXED)),
            _pad: 0,
            data: UnsafeCell::new(data),
        }
    }
}
