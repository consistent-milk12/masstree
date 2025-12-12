//! Filepath: src/key.rs
//!
//! Key representation for [`MassTree`]
//!
//! Keys are divided into 8-byte "ikeys" for efficient comparison.
//! The [`Key`] struct tracks the current position during tree traversal
//! and supports shifting to descend into trie layers.

use std::cmp::Ordering;
use std::ptr as StdPtr;

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
/// use masstree::key::Key;
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
    /// use masstree::key::Key;
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
        let len = if ikey == 0 {
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
    #[expect(
        clippy::indexing_slicing,
        reason = "INVARIANT: bounds checked before each slice"
    )]
    fn read_ikey(data: &[u8], offset: usize) -> u64 {
        // INVARIANT: Check offset is within bounds.
        if offset >= data.len() {
            return 0;
        }

        // INVARIANT: Bounds have been checked to be valid.
        let remaining: &[u8] = &data[offset..];

        if remaining.len() >= IKEY_SIZE {
            // Fast path: full 8 bytes available

            // SAFETY: We just verified remaining.len() >= IKEY_SIZE (8)
            // The pointer is valid and aligned for u8, and we're reading 8 bytes
            // which we reinterpret as [u8; 8].
            let ptr: *const u8 = remaining.as_ptr();
            let bytes: [u8; 8] = unsafe { StdPtr::read(ptr.cast::<[u8; 8]>()) };

            u64::from_be_bytes(bytes)
        } else {
            // Slow path: pad with zeros
            let mut bytes: [u8; 8] = [0u8; IKEY_SIZE];

            // INVARIANT:
            bytes[..remaining.len()].copy_from_slice(remaining);

            u64::from_be_bytes(bytes)
        }
    }
}
