//! Filepath: src/key.rs
//!
//! Key representation for [`crate::MassTree`]
//!
//! Keys are divided into 8-byte "ikeys" for efficient comparison.
//! The [`Key`] struct tracks the current position during tree traversal
//! and supports shifting to descend into trie layers.

use std::cmp::{self as StdCmp, Ordering};

/// Size of an ikey in bytes.
pub const IKEY_SIZE: usize = 8;

/// Maximum supported key length in bytes (32 layers * 8 bytes).
///
/// The C++ implementation defaults to 255 (`MASSTREE_MAXKEYLEN` in `configure.ac`).
/// We use 256 for cleaner alignment with the 8-byte ikey size.
///
/// Keys longer than this are rejected to prevent unbounded trie layer recursion.
pub const MAX_KEY_LENGTH: usize = 256;

/// A key for [`crate::MassTree`] operations.
///
/// Holds a borrowed byte slice and tracks the current position during
/// tree traversal. Keys can be "shifted" to descend into trie layers.
///
/// # Example
///
/// ```rust
/// use masstree::Key;
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
    /// use masstree::Key;
    ///
    /// let key = Key::new(b"test");
    /// assert_eq!(key.len(), 4);
    /// ```
    #[must_use]
    #[inline]
    pub fn new(data: &'a [u8]) -> Self {
        assert!(
            data.len() <= MAX_KEY_LENGTH,
            "key length {} exceeds maximum {}",
            data.len(),
            MAX_KEY_LENGTH
        );

        let ikey: u64 = Self::read_ikey(data, 0);
        let suffix_start: usize = StdCmp::min(IKEY_SIZE, data.len());

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
    #[inline]
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

    /// Create a [`Key`] from an existing suffix.
    ///
    /// Used when extracting the "remaining key" from a slot's suffix
    /// to compare with the new key layer creation.
    #[must_use]
    #[inline(always)]
    pub fn from_suffix(suffix: &'a [u8]) -> Self {
        Self::new(suffix)
    }

    /// Return the current 8-byte slice as a big-endian u64.
    ///
    /// This value can be directly compared with other ikeys using standard
    /// integer comparison, which is equivalent to lexicographic byte comparison.
    #[must_use]
    #[inline(always)]
    pub const fn len(&self) -> usize {
        self.data.len()
    }

    /// Return the current 8-byte slice as a big-endian u64.
    ///
    /// This value can be directly compared with other ikeys using standard
    /// integer comparison, which is equivalent to lexicographic byte comparison.
    #[must_use]
    #[inline(always)]
    pub const fn ikey(&self) -> u64 {
        self.ikey
    }

    /// Return the number of shifts performed.
    #[must_use]
    #[inline(always)]
    pub const fn shift_count(&self) -> usize {
        self.shift_count
    }

    /// Return the suffix start offset (relative to original key).
    #[must_use]
    #[inline(always)]
    pub const fn suffix_start(&self) -> usize {
        self.suffix_start
    }

    /// Return the length of the key at the current layer.
    ///
    /// This is the number of bytes remaining from the current position.
    #[must_use]
    #[inline(always)]
    pub const fn current_len(&self) -> usize {
        self.data.len().saturating_sub(self.shift_count * IKEY_SIZE)
    }

    /// Check if the key is empty.
    #[must_use]
    #[inline(always)]
    pub const fn is_empty(&self) -> bool {
        self.ikey == 0 && self.data.is_empty()
    }

    /// Check if the key has been shifted.
    #[must_use]
    #[inline(always)]
    pub const fn is_shifted(&self) -> bool {
        self.shift_count > 0
    }

    /// Check if the key has been shifted.
    #[must_use]
    #[inline(always)]
    pub const fn has_suffix(&self) -> bool {
        self.current_len() > IKEY_SIZE
    }

    /// Return the suffix (bytes after the current ikey).
    ///
    /// Returns an empty slice if there is no suffix.
    #[must_use]
    #[inline(always)]
    #[expect(
        clippy::indexing_slicing,
        reason = "Guarded by length check, suffix_start <= data.len()"
    )]
    pub fn suffix(&self) -> &'a [u8] {
        //  INVARIANT: suffix_start <= data.len()
        if self.suffix_start < self.data.len() {
            &self.data[self.suffix_start..]
        } else {
            &[]
        }
    }

    /// Return the length of the suffix.
    #[must_use]
    #[inline(always)]
    pub const fn suffix_len(&self) -> usize {
        self.data.len().saturating_sub(self.suffix_start)
    }

    /// Shift the key forward by 8 bytes to the next layer.
    /// After shifting, `ikey()` returns the next 8-byte slice.
    ///
    /// # Panics
    ///
    /// Panics in debug mode if `!has_suffix()`.
    #[inline(always)]
    pub fn shift(&mut self) {
        debug_assert!(self.has_suffix(), "shift() called without suffix");

        self.shift_count += 1;

        let offset: usize = self.shift_count * IKEY_SIZE;
        let suffix_start: usize = offset + IKEY_SIZE;

        self.ikey = Self::read_ikey(self.data, offset);
        self.suffix_start = StdCmp::min(suffix_start, self.data.len());
    }

    /// Shift by a specific number of bytes (must be a multiple of 8).
    ///
    /// Used when descending into layers. The `bytes` parameter comes from
    /// the negative return value of `ksuf_matches` (always 8 in current impl)
    ///
    /// # Panics
    /// Panics if `bytes` is not a multiple of `IKEY_SIZE`
    #[inline(always)]
    pub fn shift_by(&mut self, bytes: usize) {
        debug_assert!(
            bytes.is_multiple_of(IKEY_SIZE),
            "shift_by must be multiple of 8"
        );

        let shifts: usize = bytes / IKEY_SIZE;

        for _ in 0..shifts {
            self.shift();
        }
    }

    /// Shift the key backward by 8 bytes (undo one shift).
    ///
    /// # Panics
    /// Panics in debug mode if `is_shifted()`.
    #[inline(always)]
    pub fn unshift(&mut self) {
        debug_assert!(self.is_shifted(), "unshift() called at position 0");

        self.shift_count -= 1;

        let offset: usize = self.shift_count * IKEY_SIZE;
        let suffix_start: usize = offset + IKEY_SIZE;

        self.ikey = Self::read_ikey(self.data, offset);
        self.suffix_start = StdCmp::min(suffix_start, self.data.len());
    }

    /// Reset to the original position (undo all shifts).
    #[inline(always)]
    pub fn unshift_all(&mut self) {
        if self.shift_count > 0 {
            self.shift_count = 0;
            self.ikey = Self::read_ikey(self.data, 0);
            self.suffix_start = StdCmp::min(IKEY_SIZE, self.data.len());
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
    #[inline(always)]
    pub fn compare(&self, other_ikey: u64, keylenx: usize) -> Ordering {
        // First compare the ikeys
        match self.ikey.cmp(&other_ikey) {
            Ordering::Equal => {}

            ord => return ord,
        }

        // ikeys are equal, compare lengths (matching C++ reference)
        let self_len: usize = self.current_len();

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
    #[must_use]
    #[inline(always)]
    pub const fn full_data(&self) -> &'a [u8] {
        self.data
    }

    /// Compare two ikeys.
    ///
    /// Since ikeys are stored in big-endian order, standard u64 comparison
    /// is equivalent to lexicographic byte comparison.
    #[must_use]
    #[inline(always)]
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
    #[must_use]
    #[inline(always)]
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

    /// Read partial ikeys (1-7 bytes).
    ///
    /// This handles keys shorter than 8 bytes, which are common for small
    /// identifiers. The function is small `(pad + copy + from_be_bytes)`,
    /// so inlining is beneficial when short keys are frequent.
    #[inline]
    #[must_use]
    pub fn read_ikey_slow(remaining: &[u8]) -> u64 {
        let mut bytes: [u8; 8] = [0u8; 8];

        //  INVARIANT: `read_ikey` only calls this when `remaining.len() < 8`.
        #[expect(
            clippy::indexing_slicing,
            reason = "remaining.len() < 8 by caller, bytes is [u8; 8]"
        )]
        bytes[..remaining.len()].copy_from_slice(remaining);

        u64::from_be_bytes(bytes)
    }
}

#[cfg(test)]
mod unit_tests;
