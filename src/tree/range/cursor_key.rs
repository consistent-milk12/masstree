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
    pub const fn empty() -> Self {
        Self {
            buf: [0u8; MAX_KEY_LENGTH],
            offset: 0,
            len: 0,
            ikey: 0,
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
    #[inline]
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
    #[inline]
    pub fn shift(&mut self) {
        debug_assert!(self.has_suffix(), "shift() called without suffix");

        self.offset += IKEY_SIZE;
        self.len = self.len.saturating_sub(IKEY_SIZE);

        // Recompute ikey from buffer at new offset
        self.ikey = Self::read_ikey_from_buf(&self.buf, self.offset, self.len);
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
    #[inline]
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
    #[inline]
    pub fn unshift(&mut self) {
        debug_assert!(self.offset >= IKEY_SIZE, "unshift() called at root layer");

        self.offset -= IKEY_SIZE;

        // Recompute ikey from buffer at parent offset
        self.ikey = Self::read_ikey_from_buf(&self.buf, self.offset, IKEY_SIZE);

        // Set len = 9 as sentinel (see docstring)
        self.len = IKEY_SIZE + 1;
    }

    /// Reset to root layer (undo all shifts).
    ///
    /// This is a full reset - the cursor will point to the original key
    /// from the buffer.
    #[inline]
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
    #[inline]
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
    #[inline]
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
    #[inline]
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
    #[inline]
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
    #[inline]
    pub fn compare_suffix(&self, stored_suffix: &[u8]) -> Ordering {
        self.suffix().cmp(stored_suffix)
    }

    // ========================================================================
    //  Internal Helpers
    // ========================================================================

    /// Read an ikey from the buffer at the given offset.
    ///
    /// Pads with zeros if fewer than 8 bytes remain.
    #[inline]
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
mod tests {
    use super::*;

    #[test]
    fn test_from_slice_basic() {
        let cursor: CursorKey = CursorKey::from_slice(b"hello");

        assert_eq!(cursor.current_len(), 5);
        assert_eq!(cursor.offset(), 0);
        assert!(!cursor.has_suffix());
        assert!(cursor.is_at_root_layer());
    }

    #[test]
    fn test_from_slice_with_suffix() {
        let cursor: CursorKey = CursorKey::from_slice(b"hello world!");

        assert_eq!(cursor.current_len(), 12);
        assert!(cursor.has_suffix());
        assert_eq!(cursor.suffix(), b"rld!");
    }

    #[test]
    fn test_empty_cursor() {
        let cursor: CursorKey = CursorKey::empty();

        assert_eq!(cursor.current_ikey(), 0);
        assert_eq!(cursor.current_len(), 0);
        assert_eq!(cursor.full_key(), b"");
        assert!(!cursor.has_suffix());
    }

    #[test]
    fn test_ikey_extraction() {
        let cursor: CursorKey = CursorKey::from_slice(b"hello world!");
        let expected: u64 = u64::from_be_bytes(*b"hello wo");

        assert_eq!(cursor.current_ikey(), expected);
    }

    #[test]
    fn test_shift() {
        // "hello world!!!!!" is 16 bytes: "hello wo" (8) + "rld!!!!!" (8)
        let mut cursor: CursorKey = CursorKey::from_slice(b"hello world!!!!!");

        assert_eq!(cursor.current_ikey(), u64::from_be_bytes(*b"hello wo"));
        assert_eq!(cursor.layer_depth(), 0);

        cursor.shift();

        assert_eq!(cursor.current_ikey(), u64::from_be_bytes(*b"rld!!!!!"));
        assert_eq!(cursor.layer_depth(), 1);
        assert_eq!(cursor.offset(), 8);
    }

    #[test]
    fn test_shift_clear() {
        let mut cursor: CursorKey = CursorKey::from_slice(b"hello");

        cursor.shift_clear();

        assert_eq!(cursor.current_ikey(), 0);
        assert_eq!(cursor.current_len(), 0);
        assert_eq!(cursor.offset(), 8);
        assert!(!cursor.has_suffix());
    }

    #[test]
    fn test_unshift() {
        let mut cursor: CursorKey = CursorKey::from_slice(b"hello world!!!!");
        let original_ikey: u64 = cursor.current_ikey();

        cursor.shift();
        assert_ne!(cursor.current_ikey(), original_ikey);

        cursor.unshift();

        assert_eq!(cursor.current_ikey(), original_ikey);
        assert_eq!(cursor.offset(), 0);
        // len is set to 9 (sentinel)
        assert_eq!(cursor.current_len(), 9);
        assert!(cursor.has_suffix());
    }

    #[test]
    fn test_unshift_sentinel() {
        // After unshift, len=9 ensures we compare >= layer pointers
        let mut cursor: CursorKey = CursorKey::from_slice(b"hello world!!!!");

        cursor.shift();
        cursor.unshift();

        // len = 9, so has_suffix() is true
        assert!(cursor.has_suffix());
        assert_eq!(cursor.current_len(), 9);

        // Compare against a layer pointer (keylenx >= 128)
        // With len > 8, we should get Equal (both have "suffix")
        let ikey: u64 = cursor.current_ikey();
        assert_eq!(cursor.compare(ikey, 128), Ordering::Equal);
        assert_eq!(cursor.compare(ikey, 200), Ordering::Equal);
    }

    #[test]
    fn test_assign_store_ikey() {
        let mut cursor: CursorKey = CursorKey::empty();
        let ikey: u64 = u64::from_be_bytes(*b"testkey\0");

        cursor.assign_store_ikey(ikey);

        assert_eq!(cursor.current_ikey(), ikey);
        assert_eq!(&cursor.buf[0..8], b"testkey\0");
    }

    #[test]
    fn test_assign_store_suffix() {
        let mut cursor: CursorKey = CursorKey::empty();

        cursor.assign_store_ikey(u64::from_be_bytes(*b"hello wo"));
        let key_len: usize = cursor.assign_store_suffix(b"rld!");
        cursor.assign_store_length(key_len);

        assert_eq!(key_len, 12);
        assert_eq!(cursor.current_len(), 12);
        assert_eq!(cursor.suffix(), b"rld!");
        assert_eq!(cursor.full_key(), b"hello world!");
    }

    #[test]
    fn test_compare_equal() {
        let cursor: CursorKey = CursorKey::from_slice(b"hello");
        let stored_ikey: u64 = u64::from_be_bytes([b'h', b'e', b'l', b'l', b'o', 0, 0, 0]);

        assert_eq!(cursor.compare(stored_ikey, 5), Ordering::Equal);
    }

    #[test]
    fn test_compare_less_by_ikey() {
        let cursor: CursorKey = CursorKey::from_slice(b"apple");
        let stored_ikey: u64 = u64::from_be_bytes([b'b', b'a', b'n', b'a', b'n', b'a', 0, 0]);

        assert_eq!(cursor.compare(stored_ikey, 6), Ordering::Less);
    }

    #[test]
    fn test_compare_greater_by_ikey() {
        let cursor: CursorKey = CursorKey::from_slice(b"zebra");
        let stored_ikey: u64 = u64::from_be_bytes([b'a', b'p', b'p', b'l', b'e', 0, 0, 0]);

        assert_eq!(cursor.compare(stored_ikey, 5), Ordering::Greater);
    }

    #[test]
    fn test_compare_by_length() {
        let cursor: CursorKey = CursorKey::from_slice(b"hello");
        let stored_ikey: u64 = u64::from_be_bytes([b'h', b'e', b'l', b'l', b'o', 0, 0, 0]);

        // Our key (5 bytes) vs stored key (3 bytes) -> Greater
        assert_eq!(cursor.compare(stored_ikey, 3), Ordering::Greater);

        // Our key (5 bytes) vs stored key (7 bytes) -> Less
        assert_eq!(cursor.compare(stored_ikey, 7), Ordering::Less);
    }

    #[test]
    fn test_compare_with_suffix() {
        let cursor: CursorKey = CursorKey::from_slice(b"hello world!"); // 12 bytes
        let stored_ikey: u64 = u64::from_be_bytes(*b"hello wo");

        // Our key has suffix, stored key has no suffix (length 8) -> Greater
        assert_eq!(cursor.compare(stored_ikey, 8), Ordering::Greater);

        // Our key has suffix, stored key also has suffix -> Equal
        assert_eq!(cursor.compare(stored_ikey, 12), Ordering::Equal);
    }

    #[test]
    fn test_compare_suffix_bytes() {
        let cursor: CursorKey = CursorKey::from_slice(b"hello world!");

        assert_eq!(cursor.compare_suffix(b"rld!"), Ordering::Equal);
        assert_eq!(cursor.compare_suffix(b"aaa!"), Ordering::Greater);
        assert_eq!(cursor.compare_suffix(b"zzz!"), Ordering::Less);
    }

    #[test]
    fn test_full_key() {
        let mut cursor: CursorKey = CursorKey::from_slice(b"hello world!!!!");

        assert_eq!(cursor.full_key(), b"hello world!!!!");

        cursor.shift();

        // After shift, full_key still includes all bytes up to offset + len
        assert_eq!(cursor.full_key(), b"hello world!!!!");
    }

    #[test]
    fn test_layer_depth() {
        let mut cursor: CursorKey = CursorKey::from_slice(b"0123456789ABCDEF01234567");

        assert_eq!(cursor.layer_depth(), 0);

        cursor.shift();
        assert_eq!(cursor.layer_depth(), 1);

        cursor.shift();
        assert_eq!(cursor.layer_depth(), 2);
    }

    #[test]
    fn test_multiple_shift_unshift() {
        let mut cursor: CursorKey = CursorKey::from_slice(b"0123456789ABCDEF01234567"); // 24 bytes
        let layer0_ikey: u64 = cursor.current_ikey();

        cursor.shift();
        let layer1_ikey: u64 = cursor.current_ikey();

        cursor.shift();

        // Now unshift back
        cursor.unshift();
        assert_eq!(cursor.current_ikey(), layer1_ikey);

        cursor.unshift();
        assert_eq!(cursor.current_ikey(), layer0_ikey);
    }

    #[test]
    fn test_shift_clear_then_assign() {
        let mut cursor: CursorKey = CursorKey::from_slice(b"prefix!!");

        cursor.shift_clear();
        assert_eq!(cursor.current_ikey(), 0);

        // Now assign a new ikey at layer 1
        let new_ikey: u64 = u64::from_be_bytes(*b"sublayer");
        cursor.assign_store_ikey(new_ikey);
        cursor.assign_store_length(8);

        assert_eq!(cursor.current_ikey(), new_ikey);
        // full_key now includes the original prefix + new layer
        assert_eq!(&cursor.full_key()[8..16], b"sublayer");
    }

    #[test]
    #[should_panic(expected = "key length")]
    fn test_from_slice_overflow() {
        let oversized: Vec<u8> = vec![b'x'; MAX_KEY_LENGTH + 1];
        let _ = CursorKey::from_slice(&oversized);
    }

    #[test]
    fn test_clone() {
        let cursor1: CursorKey = CursorKey::from_slice(b"hello world!");
        let cursor2: CursorKey = cursor1.clone();

        assert_eq!(cursor1.current_ikey(), cursor2.current_ikey());
        assert_eq!(cursor1.current_len(), cursor2.current_len());
        assert_eq!(cursor1.full_key(), cursor2.full_key());
    }

    #[test]
    fn test_debug_format() {
        let cursor: CursorKey = CursorKey::from_slice(b"test");
        let debug_str: String = format!("{cursor:?}");

        assert!(debug_str.contains("CursorKey"));
        assert!(debug_str.contains("offset"));
        assert!(debug_str.contains("len"));
    }

    #[test]
    fn test_exact_8_bytes() {
        let cursor: CursorKey = CursorKey::from_slice(b"12345678");

        assert_eq!(cursor.current_len(), 8);
        assert!(!cursor.has_suffix());
        assert_eq!(cursor.suffix(), b"");
    }

    #[test]
    fn test_9_bytes() {
        let cursor: CursorKey = CursorKey::from_slice(b"123456789");

        assert_eq!(cursor.current_len(), 9);
        assert!(cursor.has_suffix());
        assert_eq!(cursor.suffix(), b"9");
    }
}
