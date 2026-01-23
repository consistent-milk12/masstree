//! Filepath: `src/tree/range/cursor_key.rs`
//!
//! Mutable key buffer for range scan operations.
//!
//! [`CursorKey`] tracks the current scan position and stores the "last emitted"
//! key for duplicate filtering. Unlike [`Key<'a>`] which is
//! an immutable borrowed view, `CursorKey` owns its data in a fixed-size buffer
//! and supports in-place modifications.
//!
//! # Design Rationale
//!
//! The C++ Masstree scan uses `Masstree::key` which is a mutable key type that
//! can be modified during scanning. In Rust, `Key<'a>` holds a borrowed reference
//! that cannot be modified. Rather than making `Key<'a>` self-referential or
//! dual-mode (which is complex and error-prone), we introduce `CursorKey` as
//! a separate type specifically for scan operations.
//!
//! # Layer Navigation
//!
//! - [`CursorKey::shift`]: Descend using the user's start key bytes
//! - [`CursorKey::shift_clear`]: Descend into a scan-discovered
//!   layer (sets ikey=0 to scan from sublayer minimum)
//! - [`CursorKey::unshift`]: Return to parent layer after exhausting
//!   a sublayer (sets len=9 as sentinel for duplicate filtering)
//!
//! # Constants
//!
//! Layer pointers are identified by `keylenx >= 128` (the `LAYER_KEYLENX` constant
//! from `leaf24.rs`). This affects comparison logic in [`CursorKey::compare`].
//!
//! # Duplicate Filtering
//!
//! After emitting a key, call `assign_store_*` methods to record it. The
//! `compare()` method can then detect duplicates that might occur after
//! version-triggered retries.

#![expect(clippy::indexing_slicing, reason = "Safety notes are provided inline")]

#[cfg(debug_assertions)]
use std::fmt::{self as StdFmt, Display, Formatter};
use std::{cmp::Ordering, fmt::Debug};

use crate::key::{IKEY_SIZE, MAX_KEY_LENGTH};

/// Sentinel length value set by [`CursorKey::unshift`] after ascending from a sublayer.
///
/// This value (9 = `IKEY_SIZE + 1`) ensures that when comparing against the layer
/// pointer slot in the parent, `compare()` returns `Ordering::Equal` or `Greater`,
/// causing the scan to skip the layer pointer (since we've already scanned its contents).
///
/// See [`CursorKey::unshift`] for the full algorithm explanation.
pub const UNSHIFT_SENTINEL_LEN: usize = IKEY_SIZE + 1;

/// Mutable key buffer for scan operations.
///
/// Tracks the current scan position within the trie and stores the "last emitted"
/// key for duplicate filtering during concurrent modifications.
///
/// # Memory Layout
///
/// ```text
/// buf: [u8; MAX_KEY_LENGTH]
///      ├───────────────────┬─────────────────┬──────────────────┤
///      │  prefix layers    │   current ikey  │     suffix       │
///      │  (offset bytes)   │   (8 bytes)     │   (remaining)    │
///      └───────────────────┴─────────────────┴──────────────────┘
///      0                offset          offset+8            offset+len
///
/// - `offset`: Byte position of current layer (always multiple of 8)
/// - `len`: Length of key from `offset` to end
/// - `ikey`: Cached current layer's ikey for fast access
#[derive(Clone)]
pub struct CursorKey {
    /// Fixed-size buffer for key bytes.
    buf: [u8; MAX_KEY_LENGTH],

    /// Offset to current layer's ikey (always a multiple of [`IKEY_SIZE`]).
    ///
    /// For layer 0: offset = 0
    /// For layer 1: offset = 8
    /// For layer N: offset = N * 8
    offset: usize,

    /// Length of key from `offset` to end.
    ///
    /// This is the "current layer length" - how many bytes remain at this layer.
    /// - `len <= 8`: Inline key (no suffix)
    /// - `len > 8`: Has suffix
    /// - `len == 9`: Special sentinel after `unshift()` to compare >= layer pointers
    len: usize,

    /// Current layer's ikey in host order (cached for fast access).
    ///
    /// This is `u64::from_be_bytes(buf[offset..offset+8])` but cached to avoid
    /// repeated computation during comparisons.
    ikey: u64,
}

#[expect(clippy::missing_fields_in_debug, reason = "don't print buffer")]
impl Debug for CursorKey {
    fn fmt(&self, f: &mut Formatter<'_>) -> StdFmt::Result {
        f.debug_struct("CursorKey")
            .field("offset", &self.offset)
            .field("len", &self.len)
            .field("ikey", &format_args!("{:#018x}", self.ikey))
            .field("full_key", &self.full_key())
            .finish()
    }
}

/// Creates a minimum-key cursor (ikey=0, len=0).
///
/// Equivalent to [`CursorKey::empty()`]. This positions the cursor at the
/// smallest possible key, suitable for unbounded forward scans.
impl Default for CursorKey {
    fn default() -> Self {
        Self::empty()
    }
}

impl CursorKey {
    // ========================================================================
    //  Constructors
    // ========================================================================

    /// Create a `CursorKey` from an initial key slice.
    ///
    /// Used at scan initialization to position at the start bound.
    ///
    /// # Panics
    ///
    /// Panics if `data.len() > MAX_KEY_LENGTH`.
    ///
    /// # Example
    ///
    /// ```ignore
    /// let cursor = CursorKey::from_slice(b"hello world");
    /// assert_eq!(cursor.current_ikey(), u64::from_be_bytes(*b"hello wo"));
    /// ```
    #[must_use]
    #[inline]
    pub fn from_slice(data: &[u8]) -> Self {
        assert!(
            data.len() <= MAX_KEY_LENGTH,
            "key length {} exceeds maximum {}",
            data.len(),
            MAX_KEY_LENGTH
        );

        let mut buf: [u8; MAX_KEY_LENGTH] = [0u8; MAX_KEY_LENGTH];

        // SAFETY: data.len() <= MAX_KEY_LENGTH checked above
        buf[..data.len()].copy_from_slice(data);

        let ikey: u64 = Self::read_ikey_from_buf(&buf, 0, data.len());

        Self {
            buf,
            offset: 0,
            len: data.len(),
            ikey,
        }
    }

    /// Create an empty cursor for unbounded scans.
    ///
    /// The cursor starts at position 0 with ikey = 0, which compares less than
    /// all other keys (minimum key).
    #[must_use]
    #[inline(always)]
    pub const fn empty() -> Self {
        Self {
            buf: [0u8; MAX_KEY_LENGTH],
            offset: 0,
            len: 0,
            ikey: 0,
        }
    }

    /// Create a cursor for reverse scan from an end bound.
    ///
    /// For reverse iteration:
    /// - `Unbounded`: Start at maximum (ikey = MAX, len = 9 sentinel)
    /// - `Included(k)`: Start at key k
    /// - `Excluded(k)`: Start at key k (filtering handled by `emit_equal`)
    ///
    /// # Note
    ///
    /// For unbounded end, we use `len = UNSHIFT_SENTINEL_LEN` (9) as a sentinel
    /// that signals "start from maximum" to `lower_reverse`.
    #[must_use]
    #[inline]
    pub fn for_reverse_scan(end: &super::iterator::RangeBound<'_>) -> Self {
        use super::iterator::RangeBound;

        match end {
            RangeBound::Unbounded => {
                // Maximum key cursor: ikey = MAX, len = 9 (sentinel)
                // This makes lower_reverse return size - 1 (last slot)
                Self {
                    buf: [0xFF; MAX_KEY_LENGTH],
                    offset: 0,
                    len: UNSHIFT_SENTINEL_LEN,
                    ikey: u64::MAX,
                }
            }
            RangeBound::Included(k) | RangeBound::Excluded(k) => {
                // Start at the specified key
                Self::from_slice(k)
            }
        }
    }

    // ========================================================================
    //  Accessors
    // ========================================================================

    /// Returns the current layer's ikey.
    ///
    /// This is a big-endian u64 that can be directly compared with stored ikeys.
    #[must_use]
    #[inline(always)]
    pub const fn current_ikey(&self) -> u64 {
        self.ikey
    }

    /// Returns the full key bytes from start to current position + len.
    ///
    /// This is the complete key that would be emitted to the visitor.
    #[must_use]
    #[inline(always)]
    pub fn full_key(&self) -> &[u8] {
        let end: usize = self.offset + self.len;
        // SAFETY: offset + len <= MAX_KEY_LENGTH by construction
        &self.buf[..end]
    }

    /// Returns the suffix bytes (after current ikey).
    ///
    /// Returns an empty slice if there is no suffix (len <= 8).
    #[must_use]
    #[inline]
    pub fn suffix(&self) -> &[u8] {
        if self.len > IKEY_SIZE {
            let suffix_start: usize = self.offset + IKEY_SIZE;
            let suffix_end: usize = self.offset + self.len;
            // SAFETY: bounds checked by construction
            &self.buf[suffix_start..suffix_end]
        } else {
            &[]
        }
    }

    /// Returns true if the current position has suffix bytes (len > 8).
    #[must_use]
    #[inline(always)]
    pub const fn has_suffix(&self) -> bool {
        self.len > IKEY_SIZE
    }

    /// Returns the current key length at this layer.
    #[must_use]
    #[inline(always)]
    pub const fn current_len(&self) -> usize {
        self.len
    }

    /// Returns the layer offset (number of bytes consumed by shifts).
    #[must_use]
    #[inline(always)]
    pub const fn offset(&self) -> usize {
        self.offset
    }

    /// Returns true if the cursor is at layer 0 (no shifts performed).
    #[must_use]
    #[inline(always)]
    pub const fn is_at_root_layer(&self) -> bool {
        self.offset == 0
    }

    /// Returns the number of layers deep (offset / 8).
    #[must_use]
    #[inline(always)]
    pub const fn layer_depth(&self) -> usize {
        self.offset / IKEY_SIZE
    }

    // ========================================================================
    //  Layer Navigation
    // ========================================================================

    /// Shift to next layer following the start key bytes.
    ///
    /// Called during initial descent when following the user's start bound key.
    /// The next layer's ikey is computed from the existing buffer contents.
    ///
    /// # Preconditions
    ///
    /// - `has_suffix()` must be true (there must be bytes remaining after current ikey)
    ///
    /// # Panics
    ///
    /// Debug-panics if `!has_suffix()`.
    ///
    /// # Example
    ///
    /// ```ignore
    /// let mut cursor = CursorKey::from_slice(b"hello world!!!!");
    /// assert_eq!(cursor.current_ikey(), u64::from_be_bytes(*b"hello wo"));
    ///
    /// cursor.shift();
    /// assert_eq!(cursor.current_ikey(), u64::from_be_bytes(*b"rld!!!!!"));
    /// ```
    #[inline]
    pub fn shift(&mut self) {
        debug_assert!(self.has_suffix(), "shift() called without suffix");

        self.offset += IKEY_SIZE;
        self.len = self.len.saturating_sub(IKEY_SIZE);

        // Recompute ikey from buffer at new offset
        self.ikey = Self::read_ikey_from_buf(&self.buf, self.offset, self.len);
    }

    /// Shift to sublayer for reverse scan (clear and set to max).
    ///
    /// Unlike `shift_clear()` which sets `ikey = 0, len = 0` for forward scan min,
    /// this sets `ikey = MAX, len = 9` for reverse scan max.
    ///
    /// # C++ Ref
    ///
    /// ```cpp
    /// void shift_clear_reverse() {
    ///     ikey0_ = ~ikey_type(0);
    ///     len_ = ikey_size + 1;
    ///     s_ += ikey_size;
    /// }
    /// ```
    ///
    /// The `len = 9` is critical: it makes the cursor behave like it has a suffix,
    /// which affects comparisons with layer pointers and duplicate filtering.
    #[inline]
    pub fn shift_clear_reverse(&mut self) {
        debug_assert!(
            self.offset + IKEY_SIZE <= MAX_KEY_LENGTH,
            "shift_clear_reverse: would exceed MAX_KEY_LENGTH"
        );

        // Move offset to next layer
        self.offset += IKEY_SIZE;

        // Set to maximum for this layer
        self.ikey = u64::MAX;

        // Write MAX ikey to buffer
        self.buf[self.offset..self.offset + IKEY_SIZE].copy_from_slice(&u64::MAX.to_be_bytes());

        // CRITICAL: len = 9, not 8. This matches C++ and affects comparisons.
        self.len = UNSHIFT_SENTINEL_LEN;
    }

    /// Shift to next layer and clear (for layer pointer descent during scan).
    ///
    /// Unlike `shift()`, this does NOT use the existing buffer bytes. Instead,
    /// it sets ikey = 0 and len = 0, which represents "scan from the minimum key
    /// in this sublayer".
    ///
    /// This is called when the scan encounters a layer pointer and needs to
    /// enumerate all keys in that sublayer, regardless of the original start bound.
    ///
    /// # C++ Reference
    ///
    /// Matches `key::shift_clear()` in `masstree_key.hh`.
    ///
    /// # Example
    ///
    /// ```ignore
    /// let mut cursor = CursorKey::from_slice(b"hello");
    /// cursor.shift_clear();
    ///
    /// // Now at layer 1 with minimum key
    /// assert_eq!(cursor.current_ikey(), 0);
    /// assert_eq!(cursor.current_len(), 0);
    /// ```
    #[inline]
    pub fn shift_clear(&mut self) {
        self.offset += IKEY_SIZE;
        self.len = 0;
        self.ikey = 0;

        // Debug-only: clear buffer bytes for deterministic full_key() output in tests.
        // Not needed for correctness since full_key() only reads up to offset+len.
        #[cfg(debug_assertions)]
        {
            let clear_start: usize = self.offset;
            let clear_end: usize = (self.offset + IKEY_SIZE).min(MAX_KEY_LENGTH);
            self.buf[clear_start..clear_end].fill(0);
        }
    }

    /// Return to parent layer after exhausting a sublayer.
    ///
    /// Sets `len = 9` as a sentinel value. This ensures that when comparing
    /// against the layer pointer slot in the parent, `compare()` returns
    /// `Ordering::Equal` or `Ordering::Greater`, causing the scan to skip
    /// the layer pointer (since we've already scanned its contents).
    ///
    /// # Why len = 9?
    ///
    /// The duplicate filter uses `compare(ikey, keylenx)`. For a layer pointer:
    /// - `keylenx >= 128` ([`LAYER_KEYLENX`])
    /// - A key with `len > 8` (has suffix) compares Equal to any keylenx > 8
    ///
    /// By setting `len = 9`, we ensure `has_suffix() == true`, which makes
    /// `compare()` return Equal/Greater for the layer pointer slot.
    ///
    /// # Panics
    ///
    /// Debug-panics if `offset == 0` (cannot unshift from root layer).
    #[inline(always)]
    pub fn unshift(&mut self) {
        debug_assert!(self.offset >= IKEY_SIZE, "unshift() called at root layer");

        self.offset -= IKEY_SIZE;

        // Recompute ikey from buffer at parent offset
        self.ikey = Self::read_ikey_from_buf(&self.buf, self.offset, IKEY_SIZE);

        // Set sentinel length to skip layer pointer (see UNSHIFT_SENTINEL_LEN docs)
        self.len = UNSHIFT_SENTINEL_LEN;
    }

    /// Check if cursor represents an empty key at the root layer.
    ///
    /// Returns `true` when both:
    /// - `offset == 0` (at root layer, no shifts performed)
    /// - `len == 0` (no key content)
    ///
    /// This condition occurs after `unshift()` ascends past the original key's
    /// layer, indicating the scan has exhausted all sublayers and should continue
    /// ascending. Used by `handle_up_back` for multi-level ascent detection.
    ///
    /// # C++ Reference
    ///
    /// Equivalent to C++ `ka.empty()` check in `masstree_scan.hh:359-372`:
    /// ```cpp
    /// do {
    ///     ka.unshift();
    /// } while (unlikely(ka.empty()));
    /// ```
    #[inline(always)]
    pub const fn is_at_empty_root(&self) -> bool {
        self.offset == 0 && self.len == 0
    }

    /// Reset to root layer (undo all shifts).
    ///
    /// This is a full reset - the cursor will point to the original key
    /// from the buffer. Uses an O(n) scan to find the key end, so this
    /// should only be used for error recovery, not in hot paths.
    ///
    /// # Performance
    ///
    /// Calls [`find_key_end`] which scans backward through the buffer.
    /// Marked `#[cold]` as this is only used for recovery/reset scenarios.
    #[cold]
    pub fn unshift_all(&mut self) {
        if self.offset > 0 {
            // Find total key length by scanning for last non-zero byte.
            // This is O(n) but acceptable since unshift_all is rare.
            let total_len: usize = self.find_key_end();

            self.offset = 0;
            self.len = total_len;
            self.ikey = Self::read_ikey_from_buf(&self.buf, 0, self.len);
        }
    }

    // ========================================================================
    //  Key Assignment (for duplicate filtering)
    // ========================================================================

    /// Store an ikey from a leaf slot into the buffer at current offset.
    ///
    /// Called after reading a slot's ikey to record it as the "current candidate"
    /// for duplicate filtering.
    ///
    /// # Arguments
    ///
    /// - `ikey`: The ikey value to store (big-endian u64)
    #[inline(always)]
    pub fn assign_store_ikey(&mut self, ikey: u64) {
        self.ikey = ikey;

        // Write ikey bytes to buffer
        let bytes: [u8; 8] = ikey.to_be_bytes();
        let start: usize = self.offset;
        let end: usize = start + IKEY_SIZE;

        // SAFETY: offset is always < MAX_KEY_LENGTH - IKEY_SIZE by construction
        self.buf[start..end].copy_from_slice(&bytes);
    }

    /// Store suffix bytes and return total key length (8 + suffix.len).
    ///
    /// Called after reading a slot's suffix to complete the key in the buffer.
    ///
    /// # Arguments
    ///
    /// - `suffix`: The suffix bytes to store
    ///
    /// # Returns
    ///
    /// The total key length at this layer (`IKEY_SIZE` + suffix.len).
    ///
    /// # Safety Invariant
    ///
    /// Callers must ensure `offset + IKEY_SIZE + suffix.len() <= MAX_KEY_LENGTH`.
    /// This is guaranteed when suffix comes from a valid leaf slot, as keys are
    /// validated on insertion. Debug builds verify this with an assertion.
    #[inline]
    pub fn assign_store_suffix(&mut self, suffix: &[u8]) -> usize {
        let suffix_start: usize = self.offset + IKEY_SIZE;
        let suffix_end: usize = suffix_start + suffix.len();

        debug_assert!(
            suffix_end <= MAX_KEY_LENGTH,
            "suffix would overflow buffer: offset={}, suffix.len={}",
            self.offset,
            suffix.len()
        );

        self.buf[suffix_start..suffix_end].copy_from_slice(suffix);

        IKEY_SIZE + suffix.len()
    }

    /// Set the key length at current layer (for inline keys without suffix).
    ///
    /// # Arguments
    ///
    /// - `len`: The key length (0-8 for inline, or computed from `assign_store_suffix`)
    #[inline(always)]
    pub fn assign_store_length(&mut self, len: usize) {
        debug_assert!(
            len <= MAX_KEY_LENGTH - self.offset,
            "len {} would overflow at offset {}",
            len,
            self.offset
        );
        self.len = len;
    }

    /// Mark the current key as complete (ready for emission).
    ///
    /// # Implementation Note
    ///
    /// This is a **permanent no-op** in Rust. The C++ version uses this to set
    /// a "key complete" flag for certain suffix operations, but our design
    /// ensures the key is always complete after `assign_store_*` calls.
    ///
    /// Kept for API parity with C++ `key::mark_key_complete()` to ease porting.
    #[inline(always)]
    #[expect(clippy::unused_self, reason = "C++ API compatibility")]
    pub const fn mark_key_complete(&self) {
        // Permanent no-op: key is complete after assign_store_* calls
    }

    // ========================================================================
    //  Comparison
    // ========================================================================

    /// Compare this cursor's position against a (ikey, keylenx) pair.
    ///
    /// This matches the C++ `key::compare()` semantics from `masstree_key.hh`:
    ///
    /// 1. Compare ikeys first (big-endian u64 comparison = lexicographic)
    /// 2. If equal, compare by length:
    ///    - If we have suffix (len > 8) and they don't (keylenx <= 8): Greater
    ///    - If we have suffix and they do (keylenx > 8): Equal (need suffix compare)
    ///    - Otherwise: compare lengths directly
    ///
    /// # Arguments
    ///
    /// - `other_ikey`: The stored ikey to compare against
    /// - `keylenx`: The stored key length/type indicator
    ///
    /// # Returns
    ///
    /// - `Ordering::Less`: This key < stored key
    /// - `Ordering::Equal`: Keys match at this layer (may need suffix comparison)
    /// - `Ordering::Greater`: This key > stored key
    ///
    /// # Note on Layer Pointers
    ///
    /// Layer pointers have `keylenx >= 128`. This comparison treats them as
    /// "very long keys", so a cursor without suffix will compare Less.
    /// After `unshift()` (which sets len=9), the cursor has suffix and will
    /// compare Equal/Greater to skip the already-processed layer pointer.
    #[must_use]
    #[inline(always)]
    pub fn compare(&self, other_ikey: u64, keylenx: usize) -> Ordering {
        // First compare ikeys
        match self.ikey.cmp(&other_ikey) {
            Ordering::Equal => {}
            ord => return ord,
        }

        // ikeys are equal - compare lengths (matching C++ reference)
        if self.len > IKEY_SIZE {
            // We have a suffix
            if keylenx <= IKEY_SIZE {
                // They don't have suffix -> we're greater
                Ordering::Greater
            } else {
                // They also have suffix -> equal (need suffix comparison at leaf)
                Ordering::Equal
            }
        } else {
            // We don't have a suffix -> compare lengths directly
            self.len.cmp(&keylenx)
        }
    }

    /// Compare suffix bytes against stored suffix.
    ///
    /// Used after `compare()` returns `Equal` to resolve keys that match
    /// at the ikey level but differ in suffix.
    ///
    /// # Arguments
    ///
    /// - `stored_suffix`: The suffix bytes from the leaf slot
    ///
    /// # Returns
    ///
    /// Lexicographic comparison of suffix bytes.
    #[must_use]
    #[inline(always)]
    pub fn compare_suffix(&self, stored_suffix: &[u8]) -> Ordering {
        self.suffix().cmp(stored_suffix)
    }

    // ========================================================================
    //  Debug Helpers
    // ========================================================================

    /// Returns a debug representation of the cursor state for tracing.
    ///
    /// This is useful for debugging ordering violations where the cursor state
    /// needs to be captured at various points during iteration.
    #[must_use]
    #[cfg(debug_assertions)]
    pub fn debug_state(&self) -> CursorDebugState {
        CursorDebugState {
            offset: self.offset,
            len: self.len,
            ikey: self.ikey,
            full_key: self.full_key().to_vec(),
            has_suffix: self.has_suffix(),
            layer_depth: self.layer_depth(),
        }
    }

    // ========================================================================
    //  Internal Helpers
    // ========================================================================

    /// Read an ikey from the buffer at the given offset.
    ///
    /// Zero-overhead for the 8-byte fast path (direct array conversion).
    /// Pads with zeros if fewer than 8 bytes remain.
    #[inline]
    fn read_ikey_from_buf(buf: &[u8; MAX_KEY_LENGTH], offset: usize, len: usize) -> u64 {
        if len == 0 {
            return 0;
        }

        let available: usize = len.min(IKEY_SIZE);
        let start: usize = offset;
        let end: usize = offset + available;

        // Fast path: full 8 bytes available - zero-overhead conversion
        if available == IKEY_SIZE {
            // TryInto for [u8; 8] is infallible here since we sliced exactly 8 bytes.
            // The expect is optimized away by the compiler.
            #[expect(clippy::expect_used, reason = "infallible: guarded by available == 8")]
            let arr: [u8; 8] = buf[start..end]
                .try_into()
                .expect("slice is exactly 8 bytes");
            return u64::from_be_bytes(arr);
        }

        // Slow path: partial read, need zero-padding
        let mut bytes: [u8; 8] = [0u8; 8];
        bytes[..available].copy_from_slice(&buf[start..end]);
        u64::from_be_bytes(bytes)
    }

    /// Find the end of the key in the buffer (for `unshift_all`).
    ///
    /// Scans backward from the current position to find the last non-zero byte.
    /// This is O(n) and only used by [`unshift_all`] for recovery scenarios.
    #[cold]
    fn find_key_end(&self) -> usize {
        // Start from the furthest point we've written
        let max_end: usize = self.offset + self.len;

        for i in (0..max_end).rev() {
            if self.buf[i] != 0 {
                return i + 1;
            }
        }

        0
    }
}

// ============================================================================
//  Debug State Struct
// ============================================================================

/// Debug snapshot of cursor state for tracing ordering violations.
///
/// Captures the complete cursor state at a point in time, useful for
/// debugging concurrent scan ordering issues.
#[cfg(debug_assertions)]
#[derive(Clone, Debug)]
pub struct CursorDebugState {
    /// Current layer offset
    pub offset: usize,

    /// Current key length at this layer
    pub len: usize,

    /// Current ikey (cached)
    pub ikey: u64,

    /// Complete key bytes (owned copy)
    pub full_key: Vec<u8>,

    /// Whether cursor has suffix (len > 8)
    pub has_suffix: bool,

    /// Layer depth (offset / 8)
    pub layer_depth: usize,
}

#[cfg(debug_assertions)]
impl Display for CursorDebugState {
    fn fmt(&self, f: &mut Formatter<'_>) -> StdFmt::Result {
        write!(
            f,
            "CursorState {{ offset: {}, len: {}, ikey: {:016x}, layer: {}, key: {:?} }}",
            self.offset,
            self.len,
            self.ikey,
            self.layer_depth,
            String::from_utf8_lossy(&self.full_key)
        )
    }
}

// ============================================================================
//  Tests
// ============================================================================

#[cfg(test)]
mod unit_tests;
