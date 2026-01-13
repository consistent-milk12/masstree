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
//! # Duplicate Filtering
//!
//! After emitting a key, call `assign_store_*` methods to record it. The
//! `compare()` method can then detect duplicates that might occur after
//! version-triggered retries.

#![expect(clippy::indexing_slicing, reason = "Safety notes are provided inline")]

use std::cmp::Ordering;

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
impl std::fmt::Debug for CursorKey {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CursorKey")
            .field("offset", &self.offset)
            .field("len", &self.len)
            .field("ikey", &format_args!("{:#018x}", self.ikey))
            .field("full_key", &self.full_key())
            .finish()
    }
}

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
    ///
    #[must_use]
    #[inline(always)]
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
    ///
    #[inline(always)]
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
    #[inline(always)]
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
    ///
    #[inline(always)]
    pub fn shift_clear(&mut self) {
        self.offset += IKEY_SIZE;
        self.len = 0;
        self.ikey = 0;

        // Clear the buffer bytes at the new layer (for clean full_key output)
        let clear_start: usize = self.offset;
        let clear_end: usize = (self.offset + IKEY_SIZE).min(MAX_KEY_LENGTH);
        self.buf[clear_start..clear_end].fill(0);
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

    /// Check if cursor is effectively empty after unshift.
    ///
    /// This occurs when:
    /// - offset is 0 (at root layer)
    /// - len is 0 (no key content)
    ///
    /// Used by `handle_up_back` to detect multi-level ascent needs.
    /// This is the Rust equivalent of C++ `ka.empty()` check.
    ///
    /// # C++ Reference
    ///
    /// ```cpp
    /// // masstree_scan.hh:359-372
    /// do {
    ///     ka.unshift();
    /// } while (unlikely(ka.empty()));
    /// ```
    #[inline(always)]
    pub const fn is_empty_after_unshift(&self) -> bool {
        self.offset == 0 && self.len == 0
    }

    /// Reset to root layer (undo all shifts).
    ///
    /// This is a full reset - the cursor will point to the original key
    /// from the buffer.
    #[inline(always)]
    pub fn unshift_all(&mut self) {
        if self.offset > 0 {
            // Find total key length by scanning for last non-zero byte
            // (This is an approximation - in practice, we track this separately)
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
    /// # Panics
    ///
    /// Panics if the suffix would overflow the buffer.
    #[inline(always)]
    pub fn assign_store_suffix(&mut self, suffix: &[u8]) -> usize {
        let suffix_start: usize = self.offset + IKEY_SIZE;
        let suffix_end: usize = suffix_start + suffix.len();

        assert!(
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
    /// This is a no-op in the current implementation but exists for API
    /// compatibility with C++ `key::mark_key_complete()`.
    #[inline(always)]
    #[expect(clippy::unused_self, reason = "API Consistency")]
    pub const fn mark_key_complete(&self) {
        // No-op: the key is already complete after assign_store_* calls
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
    //  Internal Helpers
    // ========================================================================

    /// Read an ikey from the buffer at the given offset.
    ///
    /// Pads with zeros if fewer than 8 bytes remain.
    #[inline(always)]
    fn read_ikey_from_buf(buf: &[u8; MAX_KEY_LENGTH], offset: usize, len: usize) -> u64 {
        if len == 0 {
            return 0;
        }

        let available: usize = len.min(IKEY_SIZE);
        let start: usize = offset;
        let end: usize = offset + available;

        let mut bytes: [u8; 8] = [0u8; 8];
        bytes[..available].copy_from_slice(&buf[start..end]);

        u64::from_be_bytes(bytes)
    }

    /// Find the end of the key in the buffer (for `unshift_all`).
    ///
    /// This scans backward from [`MAX_KEY_LENGTH`] to find the last non-zero byte.
    /// In practice, we should track this separately, but this works as a fallback.
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
//  Tests
// ============================================================================

#[cfg(test)]
mod unit_tests;
