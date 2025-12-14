//! Filepath: src/key.rs
//!
//! Key representation for [`MassTree`]
//!
//! Keys are divided into 8-byte "ikeys" for efficient comparison.
//! The [`Key`] struct tracks the current position during tree traversal
//! and supports shifting to descend into trie layers.

use std::cmp::Ordering;

/// Size of an ikey in bytes.
pub const IKEY_SIZE: usize = 8;

/// Maximum supported key length in bytes (32 layers * 8 bytes).
///
/// The C++ implementation defaults to 255 (`MASSTREE_MAXKEYLEN` in `configure.ac`).
/// We use 256 for cleaner alignment with the 8-byte ikey size.
///
/// Keys longer than this are rejected to prevent unbounded trie layer recursion.
pub const MAX_KEY_LENGTH: usize = 256;

/// A key for [`MassTree`] operations.
///
/// Holds a borrowed byte slice and tracks the current position during
/// tree traversal. Keys can be "shifted" to descend into trie layers.
///
/// # Example
///
/// ```rust
/// use madtree::key::Key;
///
/// let data = b"hello world!";
/// let mut key = Key::new(data);
///
/// // First 8 bytes as big-endian u64
/// assert_eq!(key.ikey(), u64::from_be_bytes(*b"hello wo"));
///
/// // Shift to next layer
/// key.shift();
/// assert_eq!(key.ikey(), u64::from_be_bytes(*b"rld!\0\0\0\0"));
/// ```
#[derive(Clone, Copy, Debug)]
pub struct Key<'a> {
    /// The full key data (never modified).
    data: &'a [u8],

    /// Current 8-byte slice as big-endian u64.
    ///
    /// This is precomputed for efficient comparison during tree traversal.
    /// Updated on each `shift()` call.
    ikey: u64,

    /// Number of 8-byte chunks consumed by `shift()`.
    ///
    /// - `shift_count == 0`: At the start, ikey covers bytes [0..8)
    /// - `shift_count == 1`: After one shift, ikey covers bytes [8..16]
    /// - continued...
    shift_count: usize,

    /// Offset where the suffix begins (relative to original key start).
    ///
    /// For the current layer, bytes `[shift_count * 8 .. suffix_start]` are
    /// part of the ikey, and bytes `[suffix_start..]` are the suffix.
    ///
    /// This is always equal to `(shift_count + 1) * IKEY_SIZE` clamped to
    /// `data.len()`, but stored for fast access.
    suffix_start: usize,
}

impl<'a> Key<'a> {
    /// Create a new key from a byte slice.
    ///
    /// The key starts at position 0 (no shifts).
    ///
    /// # Panics
    ///
    /// Panics if `data.len() > MAX_KEY_LENGTH` (256 bytes). Keys longer than
    /// this would require more than 32 trie layers.
    ///
    /// # Example
    ///
    /// ```rust
    /// use madtree::key::Key;
    ///
    /// let key = Key::new(b"test");
    /// assert_eq!(key.len(), 4);
    /// ```
    #[must_use]
    pub fn new(data: &'a [u8]) -> Self {
        assert!(
            data.len() <= MAX_KEY_LENGTH,
            "key length {} exceeds maximum {}",
            data.len(),
            MAX_KEY_LENGTH
        );

        let ikey: u64 = Self::read_ikey(data, 0);
        let suffix_start: usize = std::cmp::min(IKEY_SIZE, data.len());

        Self {
            data,
            ikey,
            shift_count: 0,
            suffix_start,
        }
    }

    /// Create a key from a raw ikey value (for testing/internal use).
    ///
    /// The key will have length equal to the significant bytes in the ikey
    /// (trailing zeros are not counted).
    #[must_use]
    pub const fn from_ikey(ikey: u64) -> Self {
        // Calculate length: 8 minus trailing zero bytes
        let len: usize = if ikey == 0 {
            0
        } else {
            IKEY_SIZE - ((ikey.trailing_zeros() / 8) as usize)
        };

        Self {
            data: &[], // No backing data
            ikey,
            shift_count: 0,
            suffix_start: len,
        }
    }

    /// Return the current 8-byte slice as a big-endian u64.
    ///
    /// This value can be directly compared with other ikeys using standard
    /// integer comparison, which is equivalent to lexicographic byte comparison.
    #[inline]
    #[must_use]
    pub const fn len(&self) -> usize {
        self.data.len()
    }

    /// Return the current 8-byte slice as a big-endian u64.
    ///
    /// This value can be directly compared with other ikeys using standard
    /// integer comparison, which is equivalent to lexicographic byte comparison.
    #[inline]
    #[must_use]
    pub const fn ikey(&self) -> u64 {
        self.ikey
    }

    /// Return the number of shifts performed.
    #[inline]
    #[must_use]
    pub const fn shift_count(&self) -> usize {
        self.shift_count
    }

    /// Return the suffix start offset (relative to original key).
    #[inline]
    #[must_use]
    pub const fn suffix_start(&self) -> usize {
        self.suffix_start
    }

    /// Return the length of the key at the current layer.
    ///
    /// This is the number of bytes remaining from the current position.
    #[inline]
    #[must_use]
    pub const fn current_len(&self) -> usize {
        self.data.len().saturating_sub(self.shift_count * IKEY_SIZE)
    }

    /// Check if the key is empty.
    #[inline]
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.ikey == 0 && self.data.is_empty()
    }

    /// Check if the key has been shifted.
    #[inline]
    #[must_use]
    pub const fn is_shifted(&self) -> bool {
        self.shift_count > 0
    }

    /// Check if the key has been shifted.
    #[inline]
    #[must_use]
    pub const fn has_suffix(&self) -> bool {
        self.current_len() > IKEY_SIZE
    }

    /// Return the suffix (bytes after the current ikey).
    ///
    /// Returns an empty slice if there is no suffix.
    #[must_use]
    #[expect(clippy::indexing_slicing)]
    pub fn suffix(&self) -> &'a [u8] {
        //  INVARIANT: suffix_start <= data.len()
        if self.suffix_start < self.data.len() {
            &self.data[self.suffix_start..]
        } else {
            &[]
        }
    }

    /// Return the length of the suffix.
    #[inline]
    #[must_use]
    pub const fn suffix_len(&self) -> usize {
        self.data.len().saturating_sub(self.suffix_start)
    }

    /// Shift the key forward by 8 bytes to the next layer.
    /// After shifting, `ikey()` returns the next 8-byte slice.
    ///
    /// # Panics
    ///
    /// Panics in debug mode if `!has_suffix()`.
    pub fn shift(&mut self) {
        debug_assert!(self.has_suffix(), "shift() called without suffix");

        self.shift_count += 1;

        let offset: usize = self.shift_count * IKEY_SIZE;
        let suffix_start: usize = offset + IKEY_SIZE;

        self.ikey = Self::read_ikey(self.data, offset);
        self.suffix_start = std::cmp::min(suffix_start, self.data.len());
    }

    /// Shift the key backward by 8 bytes (undo one shift).
    ///
    /// # Panics
    /// Panics in debug mode if `is_shifted()`.
    pub fn unshift(&mut self) {
        debug_assert!(self.is_shifted(), "unshift() called at position 0");

        self.shift_count -= 1;

        let offset: usize = self.shift_count * IKEY_SIZE;
        let suffix_start: usize = offset + IKEY_SIZE;

        self.ikey = Self::read_ikey(self.data, offset);
        self.suffix_start = std::cmp::min(suffix_start, self.data.len());
    }

    /// Reset to the original position (undo all shifts).
    pub fn unshift_all(&mut self) {
        if self.shift_count > 0 {
            self.shift_count = 0;
            self.ikey = Self::read_ikey(self.data, 0);
            self.suffix_start = std::cmp::min(IKEY_SIZE, self.data.len());
        }
    }

    /// Compare this key's length against a stored keylenx value.
    ///
    /// This is the "pure" key comparison that only considers ikey and length,
    /// matching the C++ `key::compare()` in `masstree_key.hh`.
    ///
    /// # Arguments
    /// - `other_ikey` - The stored ikey to compare against
    /// - `keylenx` - The stored key length `(0..=IKEY_SIZE for inline, or raw length)`
    ///
    /// # Note on the C++ reference
    /// The C++ implementation in `masstree_key.hh` uses this algorithm:
    ///
    /// ```cpp
    /// int compare(ikey_type ikey, int keylenx) const {
    ///     int cmp = ::compare(this->ikey(), ikey);
    ///
    ///     if (cmp == 0) {
    ///         int al = this->length();
    ///
    ///         if (al > ikey_size) {
    ///             cmp = keylenx <= ikey_size;
    ///         } else {
    ///             cmp = al - keylenx;
    ///         }
    ///     }
    ///
    ///     return cmp;
    /// }
    /// ```
    /// Special keylenx values (like layer pointers) are handled at the leaf
    /// level, not in this comparison function. See leaf node implementation
    /// for `ksuf_keylenx` and `layer_keylenx` handling.
    ///
    /// # Returns
    /// - `Ordering::Less` if this key is less than the stored key
    /// - `Ordering::Equal` if the keys match at this layer
    /// - `Ordering::Greater` if this key is greater than the store key
    #[must_use]
    pub fn compare(&self, other_ikey: u64, keylenx: usize) -> Ordering {
        // First compare the ikeys
        match self.ikey.cmp(&other_ikey) {
            Ordering::Equal => {}

            ord => return ord,
        }

        // ikeys are equal, compare lengths (matching C++ reference)
        let self_len = self.current_len();

        if self_len > IKEY_SIZE {
            // We have a suffix: cmp = (keylenx <= IKEY_SIZE)
            // If stored key has no suffix (keylenx <=8), we're greater (1)
            // If stored key also has suffix (keylenx > 8), equal (0) - need suffix comppare
            if keylenx <= IKEY_SIZE {
                Ordering::Greater
            } else {
                Ordering::Equal
            }
        } else {
            // We don't have a suffix: cmp = self_len - keylenx
            self_len.cmp(&keylenx)
        }
    }

    /// Get the full key data.
    #[inline]
    #[must_use]
    pub const fn full_data(&self) -> &'a [u8] {
        self.data
    }

    /// Compare two ikeys.
    ///
    /// Since ikeys are stored in big-endian order, standard u64 comparison
    /// is equivalent to lexicographic byte comparison.
    #[inline]
    #[must_use]
    pub const fn compare_ikey(a: u64, b: u64) -> Ordering {
        // Use if-else for const fn compatibility
        if a < b {
            Ordering::Less
        } else if a > b {
            Ordering::Greater
        } else {
            Ordering::Equal
        }
    }

    /// Read an 8-byte ikey from data at the given offset.
    ///
    /// Pads with zeros if fewer than 8 bytes remain.
    /// Returns 0 if offset is past the end of data.
    #[inline]
    #[must_use]
    pub fn read_ikey(data: &[u8], offset: usize) -> u64 {
        if let Some(remaining) = data.get(offset..) {
            if let Some(bytes) = remaining.first_chunk::<IKEY_SIZE>() {
                return u64::from_be_bytes(*bytes);
            }

            // Only call slow path if there's actual data (1-7 bytes).
            // Empty remaining means offset == data.len(), return 0.
            if !remaining.is_empty() {
                return Self::read_ikey_slow(remaining);
            }
        }

        0
    }

    /// Slow path for reading partial ikeys (1-7 bytes).
    ///
    /// Separated from `read_ikey` to keep the hot path small for I-cache.
    #[cold]
    #[inline(never)]
    #[must_use]
    pub fn read_ikey_slow(remaining: &[u8]) -> u64 {
        let mut bytes: [u8; 8] = [0u8; 8];

        //  INVARIANT: `read_ikey` only calls this when `remaining.len() < 8`.
        #[expect(clippy::indexing_slicing)]
        bytes[..remaining.len()].copy_from_slice(remaining);

        u64::from_be_bytes(bytes)
    }
}

#[cfg(test)]
mod tests {
    use std::cmp::Ordering;

    use crate::key::MAX_KEY_LENGTH;

    use super::Key;

    #[test]
    fn test_new_key() {
        let key: Key<'_> = Key::new(b"hello");

        assert_eq!(key.len(), 5);
        assert_eq!(key.shift_count(), 0);
        assert!(!key.is_shifted());
        assert!(!key.is_empty());
    }

    #[test]
    fn test_empty_key() {
        let key: Key<'_> = Key::new(b"");

        assert!(key.is_empty());
        assert_eq!(key.ikey(), 0);
        assert_eq!(key.len(), 0);
    }

    #[test]
    fn test_ikey_extraction() {
        let key: Key<'_> = Key::new(b"hello world!");
        let expected: u64 = u64::from_be_bytes(*b"hello wo");

        assert_eq!(key.ikey(), expected);
    }

    #[test]
    fn test_short_key_padding() {
        let key: Key<'_> = Key::new(b"hi");
        let expected: u64 = u64::from_be_bytes([b'h', b'i', 0, 0, 0, 0, 0, 0]);

        assert_eq!(key.ikey(), expected);
    }

    #[test]
    fn test_exact_8_bytes() {
        let key: Key<'_> = Key::new(b"12345678");

        assert_eq!(key.ikey(), u64::from_be_bytes(*b"12345678"));
        assert!(!key.has_suffix());
        assert_eq!(key.suffix_len(), 0);
    }

    #[test]
    fn test_has_suffix() {
        let key: Key<'_> = Key::new(b"123456789"); // 9 bytes

        assert!(key.has_suffix());
        assert_eq!(key.suffix(), b"9");
        assert_eq!(key.suffix_len(), 1);
    }

    #[test]
    fn test_shift() {
        let mut key: Key<'_> = Key::new(b"hello world!!!!!");
        assert_eq!(key.ikey(), u64::from_be_bytes(*b"hello wo"));
        assert_eq!(key.shift_count(), 0);

        key.shift();
        assert_eq!(key.ikey(), u64::from_be_bytes(*b"rld!!!!!"));
        assert_eq!(key.shift_count(), 1);
        assert!(key.is_shifted());
    }

    #[test]
    fn test_shift_with_short_suffix() {
        let mut key: Key<'_> = Key::new(b"hello world!"); // 12 bytes
        key.shift();

        // "rld!" padded with zeros
        let expected: u64 = u64::from_be_bytes([b'r', b'l', b'd', b'!', 0, 0, 0, 0]);
        assert_eq!(key.ikey(), expected);
    }

    #[test]
    fn test_unshift() {
        let mut key: Key<'_> = Key::new(b"hello world!!!!!");
        let original_ikey: u64 = key.ikey();

        key.shift();
        assert_ne!(key.ikey(), original_ikey);

        key.unshift();
        assert_eq!(key.ikey(), original_ikey);
        assert_eq!(key.shift_count(), 0);
    }

    #[test]
    fn test_unshift_all() {
        let mut key: Key<'_> = Key::new(b"hello world!!!!!!!!!!!!!!");
        let original_ikey: u64 = key.ikey();

        key.shift();
        key.shift();
        assert_eq!(key.shift_count(), 2);

        key.unshift_all();
        assert_eq!(key.ikey(), original_ikey);
        assert_eq!(key.shift_count(), 0);
    }

    #[test]
    fn test_current_len() {
        let mut key: Key<'_> = Key::new(b"hello world!"); // 12 bytes
        assert_eq!(key.current_len(), 12);

        key.shift();
        assert_eq!(key.current_len(), 4); // "rld!"
    }

    #[test]
    fn test_compare_equal() {
        let key: Key<'_> = Key::new(b"hello");
        let stored_ikey: u64 = u64::from_be_bytes([b'h', b'e', b'l', b'l', b'o', 0, 0, 0]);

        // Same ikey, same length -> Equal
        assert_eq!(key.compare(stored_ikey, 5), Ordering::Equal);
    }

    #[test]
    fn test_compare_less_by_ikey() {
        let key: Key<'_> = Key::new(b"apple");
        let stored_ikey: u64 = u64::from_be_bytes([b'b', b'a', b'n', b'a', b'n', b'a', 0, 0]);

        // ikey comparison: "apple" < "banana"
        assert_eq!(key.compare(stored_ikey, 6), Ordering::Less);
    }

    #[test]
    fn test_compare_greater_by_ikey() {
        let key: Key<'_> = Key::new(b"zebra");
        let stored_ikey: u64 = u64::from_be_bytes([b'a', b'p', b'p', b'l', b'e', 0, 0, 0]);

        // ikey comparison: "zebra" > "apple"
        assert_eq!(key.compare(stored_ikey, 5), Ordering::Greater);
    }

    #[test]
    fn test_compare_by_length() {
        // Same ikey, different lengths
        let key: Key<'_> = Key::new(b"hello");
        let stored_ikey: u64 = u64::from_be_bytes([b'h', b'e', b'l', b'l', b'o', 0, 0, 0]);

        // Our key (5 bytes) vs stored key (3 bytes) -> Greater
        assert_eq!(key.compare(stored_ikey, 3), Ordering::Greater);

        // Our key (5 bytes) vs stored key (7 bytes) -> Less
        assert_eq!(key.compare(stored_ikey, 7), Ordering::Less);
    }

    #[test]
    fn test_compare_with_suffix() {
        // Key with suffix (> 8 bytes)
        let key: Key<'_> = Key::new(b"hello world!"); // 12 bytes
        let stored_ikey = u64::from_be_bytes(*b"hello wo");

        // Our key has suffix, stored key has no suffix (length 8) -> Greater
        assert_eq!(key.compare(stored_ikey, 8), Ordering::Greater);

        // Our key has suffix, stored key also has suffix (length > 8) -> Equal
        // (actual suffix comparison happens at leaf level)
        assert_eq!(key.compare(stored_ikey, 12), Ordering::Equal);
    }

    #[test]
    fn test_from_ikey() {
        let ikey = u64::from_be_bytes(*b"test\0\0\0\0");
        let key: Key<'_> = Key::from_ikey(ikey);

        assert_eq!(key.ikey(), ikey);
        assert_eq!(key.suffix_start(), 4);
    }

    #[test]
    fn test_lexicographic_ordering() {
        // Verify that u64 comparison equals lexicographic comparison
        let key_a: Key<'_> = Key::new(b"aaa");
        let key_b: Key<'_> = Key::new(b"aab");
        let key_c: Key<'_> = Key::new(b"baa");

        assert!(key_a.ikey() < key_b.ikey());
        assert!(key_b.ikey() < key_c.ikey());
    }

    #[test]
    fn test_suffix_after_multiple_shifts() {
        let mut key: Key<'_> = Key::new(b"0123456789ABCDEF01234567"); // 24 bytes

        assert!(key.has_suffix());
        assert_eq!(key.suffix_len(), 16);

        key.shift();
        assert!(key.has_suffix());
        assert_eq!(key.suffix_len(), 8);

        key.shift();
        assert!(!key.has_suffix());
        assert_eq!(key.suffix_len(), 0);
    }

    #[test]
    fn test_max_key_length() {
        // Exactly at the limit should succeed
        let max_key: Vec<u8> = vec![b'x'; MAX_KEY_LENGTH];
        let key: Key<'_> = Key::new(&max_key);

        assert_eq!(key.len(), MAX_KEY_LENGTH);
    }

    #[test]
    #[should_panic(expected = "key length 257 exceeds maximum 256")]
    fn test_key_length_overflow() {
        let oversized: Vec<u8> = vec![b'x'; MAX_KEY_LENGTH + 1];

        let _ = Key::new(&oversized);
    }

    #[test]
    fn test_compare_ikey_fn() {
        assert_eq!(Key::compare_ikey(100, 200), Ordering::Less);
        assert_eq!(Key::compare_ikey(200, 100), Ordering::Greater);
        assert_eq!(Key::compare_ikey(100, 100), Ordering::Equal);
    }

    // ==================== Additional Edge Case Tests ====================

    #[test]
    fn test_full_data() {
        let key: Key<'_> = Key::new(b"hello world!");
        assert_eq!(key.full_data(), b"hello world!");
    }

    #[test]
    fn test_full_data_after_shift() {
        let mut key: Key<'_> = Key::new(b"hello world!");
        key.shift();
        // full_data should still return the complete original data
        assert_eq!(key.full_data(), b"hello world!");
    }

    #[test]
    fn test_compare_layer_keylenx() {
        // keylenx >= 128 indicates a layer pointer, should return Less
        let key: Key<'_> = Key::new(b"hello");
        let stored_ikey: u64 = u64::from_be_bytes([b'h', b'e', b'l', b'l', b'o', 0, 0, 0]);

        // Layer indicator (keylenx = 128)
        assert_eq!(key.compare(stored_ikey, 128), Ordering::Less);

        // Higher layer values
        assert_eq!(key.compare(stored_ikey, 200), Ordering::Less);
        assert_eq!(key.compare(stored_ikey, 255), Ordering::Less);
    }

    #[test]
    fn test_compare_ksuf_keylenx() {
        // keylenx == 64 indicates stored key has suffix
        let key: Key<'_> = Key::new(b"hello world!"); // 12 bytes, has suffix
        let stored_ikey = u64::from_be_bytes(*b"hello wo");

        // KSUF_KEYLENX = 64, both have suffixes -> Equal (needs suffix comparison)
        assert_eq!(key.compare(stored_ikey, 64), Ordering::Equal);

        // Key without suffix vs stored with suffix
        let short_key: Key<'_> = Key::new(b"hello"); // 5 bytes, no suffix
        assert_eq!(short_key.compare(stored_ikey, 64), Ordering::Less);
    }

    #[test]
    fn test_ikey_byte_order() {
        // Verify big-endian byte order for lexicographic comparison
        let key: Key<'_> = Key::new(&[0x00, 0xFF]);
        let expected: u64 = 0x00FF_0000_0000_0000;
        assert_eq!(key.ikey(), expected);
    }

    #[test]
    fn test_multi_shift_boundary() {
        // Test shifting exactly at 8-byte boundaries
        let mut key: Key<'_> = Key::new(b"12345678ABCDEFGH"); // 16 bytes exactly

        assert_eq!(key.ikey(), u64::from_be_bytes(*b"12345678"));
        assert!(key.has_suffix());
        assert_eq!(key.suffix_len(), 8);

        key.shift();
        assert_eq!(key.ikey(), u64::from_be_bytes(*b"ABCDEFGH"));
        assert!(!key.has_suffix());
        assert_eq!(key.suffix_len(), 0);
    }

    #[test]
    fn test_from_ikey_full_bytes() {
        // ikey with all 8 bytes significant (no trailing zeros)
        let ikey: u64 = u64::from_be_bytes(*b"ABCDEFGH");
        let key: Key<'_> = Key::from_ikey(ikey);

        assert_eq!(key.ikey(), ikey);
        assert_eq!(key.suffix_start(), 8); // All 8 bytes are significant
    }

    #[test]
    fn test_from_ikey_single_byte() {
        // ikey with only 1 significant byte
        let ikey: u64 = u64::from_be_bytes([b'A', 0, 0, 0, 0, 0, 0, 0]);
        let key: Key<'_> = Key::from_ikey(ikey);

        assert_eq!(key.ikey(), ikey);
        assert_eq!(key.suffix_start(), 1);
    }

    #[test]
    fn test_from_ikey_zero() {
        // All-zero ikey
        let key: Key<'_> = Key::from_ikey(0);

        assert_eq!(key.ikey(), 0);
        assert_eq!(key.suffix_start(), 0);
    }

    #[test]
    fn test_no_shift_when_no_suffix() {
        // Keys shorter than 8 bytes have no suffix, so shift() would panic
        let key: Key<'_> = Key::new(b"hi"); // 2 bytes
        assert!(!key.has_suffix());

        // Verify that attempting to shift would be invalid
        // (We don't actually call shift() because it would panic)
        assert_eq!(key.current_len(), 2);
    }

    #[test]
    fn test_suffix_exact_boundary() {
        // Test suffix behavior at exact 8-byte boundary
        let key8: Key<'_> = Key::new(b"12345678");
        assert!(!key8.has_suffix());
        assert_eq!(key8.suffix(), b"");

        let key9: Key<'_> = Key::new(b"123456789");
        assert!(key9.has_suffix());
        assert_eq!(key9.suffix(), b"9");
    }

    #[test]
    fn test_compare_empty_key() {
        let key: Key<'_> = Key::new(b"");

        // Empty key vs non-empty stored key
        let stored_ikey: u64 = u64::from_be_bytes([b'a', 0, 0, 0, 0, 0, 0, 0]);
        assert_eq!(key.compare(stored_ikey, 1), Ordering::Less);

        // Empty key vs empty stored key (keylenx = 0)
        assert_eq!(key.compare(0, 0), Ordering::Equal);
    }

    #[test]
    fn test_binary_keys() {
        // Test with binary data containing null bytes
        let key: Key<'_> = Key::new(&[0x00, 0x01, 0x02, 0x00, 0xFF]);
        let expected: u64 = u64::from_be_bytes([0x00, 0x01, 0x02, 0x00, 0xFF, 0, 0, 0]);

        assert_eq!(key.ikey(), expected);
        assert_eq!(key.len(), 5);
    }
}
