// ============================================================================
//  Fast Suffix Comparison
// ============================================================================

use std::cmp::Ordering;

pub(super) struct CompareSuffix;

impl CompareSuffix {
    /// Compare two byte slices with word-aligned fast path.
    ///
    /// For slices >= 8 bytes, compares 8 bytes at a time using native word loads.
    /// Falls back to byte-to-byte for remainder and short slices.
    #[inline]
    pub(super) fn fast_slice_eq(a: &[u8], b: &[u8]) -> bool {
        if a.len() != b.len() {
            return false;
        }

        let len: usize = a.len();

        // Fast path: compare 8 bytes at a time.
        let chunks: usize = len / 8;
        let a_ptr: *const u8 = a.as_ptr();
        let b_ptr: *const u8 = b.as_ptr();

        for i in 0..chunks {
            // SAFETY: (i * 8) + 8 <= chunks * 8 <= len, so we within bounds.
            // Unaligned reads are safe on all modern architectures.
            let a_word: u64 = unsafe { a_ptr.add(i * 8).cast::<u64>().read_unaligned() };
            let b_word: u64 = unsafe { b_ptr.add(i * 8).cast::<u64>().read_unaligned() };

            if a_word != b_word {
                return false;
            }
        }

        // Compare remainder byte-by-byte
        let remainder_start: usize = chunks * 8;
        a[remainder_start..] == b[remainder_start..]
    }

    /// Lexicographic comparison with word-aligned fast path.
    #[inline]
    pub(super) fn fast_slice_cmp(a: &[u8], b: &[u8]) -> Ordering {
        let min_len: usize = a.len().min(b.len());

        // Fast path: compare 8 bytes at a time
        let chunks: usize = min_len / 8;
        let a_ptr: *const u8 = a.as_ptr();
        let b_ptr: *const u8 = b.as_ptr();

        for i in 0..chunks {
            // SAFETY: (i * 8) + 8 <= chunks * 8 <= min_lin <= both lengths
            let a_word: u64 = unsafe { a_ptr.add(i * 8).cast::<u64>().read_unaligned() };
            let b_word: u64 = unsafe { b_ptr.add(i * 8).cast::<u64>().read_unaligned() };

            if a_word != b_word {
                // Found diff, read byte-by-byte to get ordering
                // Convert to big-endian for correct lexicographic comparison
                let a_be: u64 = a_word.to_be();
                let b_be: u64 = b_word.to_be();

                return a_be.cmp(&b_be);
            }
        }

        // Compare remainder byte-by-byte, then by length
        let remainder_start: usize = 8 * chunks;

        match a[remainder_start..min_len].cmp(&b[remainder_start..min_len]) {
            Ordering::Equal => a.len().cmp(&b.len()),

            other => other,
        }
    }
}
