//! Software prefetching utilities for cache optimization.
//!
//! Provides hardware-specific prefetch hints to reduce memory latency
//! during tree traversal. When the CPU knows we're about to access a
//! memory location, it can begin fetching it into cache while we
//! continue processing the current node.
//!
//! # Architecture Support
//!
//! - **`x86_64`**: Uses `_mm_prefetch` with `_MM_HINT_T0` (all cache levels)
//! - **`aarch64`**: Uses `_prefetch` with locality hint 3 (keep in cache)
//! - **Other**: No-op (safe fallback)
//!
//! # Usage
//!
//! ```ignore
//! use crate::prefetch::prefetch_read;
//!
//! // During tree traversal, after determining the next child
//! let child_ptr = internode.child(child_idx);
//! prefetch_read(child_ptr);  // Start fetching while we validate version
//!
//! // By the time we follow the pointer, it's likely in cache
//! node = child_ptr;
//! ```

/// Prefetch data for reading into all cache levels.
///
/// This is a hint to the CPU that we're about to read from the given
/// pointer. The CPU may begin fetching the cache line(s) containing
/// this address into L1/L2/L3 cache.
///
/// # Safety
///
/// This function is safe to call:
/// - With null pointers (becomes a no-op)
/// - With invalid pointers (prefetch is a hint, not a load)
/// - The pointer doesn't need to be aligned
///
/// # Performance Notes
///
/// - Prefetching is most effective when there's work to do between
///   the prefetch and the actual access (e.g., version validation)
/// - Over-prefetching can pollute the cache; use judiciously
/// - The prefetch distance should match your access pattern
#[inline(always)]
pub fn prefetch_read<T>(ptr: *const T) {
    if ptr.is_null() {
        return;
    }

    #[cfg(target_arch = "x86_64")]
    {
        // SAFETY: _mm_prefetch is always safe to call.
        // It's a hint that may be ignored by the CPU.
        // Invalid addresses cause no fault (unlike actual loads).
        unsafe {
            std::arch::x86_64::_mm_prefetch(ptr.cast::<i8>(), std::arch::x86_64::_MM_HINT_T0);
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        // SAFETY: _prefetch is always safe to call on aarch64.
        // Parameters: pointer, read (0) vs write (1), locality (0-3)
        // Locality 3 = keep in cache as long as possible
        unsafe {
            std::arch::aarch64::_prefetch(
                ptr.cast::<i8>(),
                std::arch::aarch64::_PREFETCH_READ,
                std::arch::aarch64::_PREFETCH_LOCALITY3,
            );
        }
    }

    // No-op on unsupported architectures
    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    {
        let _ = ptr;
    }
}

/// Prefetch data for writing into all cache levels.
///
/// Similar to [`prefetch_read`], but hints that we intend to write
/// to this address. The CPU may fetch the cache line in exclusive
/// state, avoiding a later upgrade.
///
/// # When to Use
///
/// Use this when you're about to modify data at the pointer location.
/// For read-only traversal, prefer [`prefetch_read`].
#[inline(always)]
pub fn prefetch_write<T>(ptr: *mut T) {
    if ptr.is_null() {
        return;
    }

    #[cfg(target_arch = "x86_64")]
    {
        // SAFETY: _mm_prefetch is always safe to call.
        // _MM_HINT_T0 works for both read and write prefetch on x86.
        // For exclusive access, _MM_HINT_ET0 could be used but T0 is portable.
        unsafe {
            std::arch::x86_64::_mm_prefetch(ptr.cast::<i8>(), std::arch::x86_64::_MM_HINT_T0);
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        // SAFETY: _prefetch is always safe to call on aarch64.
        // _PREFETCH_WRITE hints that we'll modify the data.
        unsafe {
            std::arch::aarch64::_prefetch(
                ptr.cast::<i8>(),
                std::arch::aarch64::_PREFETCH_WRITE,
                std::arch::aarch64::_PREFETCH_LOCALITY3,
            );
        }
    }

    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    {
        let _ = ptr;
    }
}

#[cfg(test)]
#[expect(clippy::indexing_slicing)]
mod tests {
    use super::*;

    #[test]
    fn test_prefetch_null_is_safe() {
        // Should not panic or crash
        prefetch_read::<u64>(std::ptr::null());
        prefetch_write::<u64>(std::ptr::null_mut());
    }

    #[test]
    fn test_prefetch_valid_pointer() {
        let value: u64 = 42;
        let ptr = &raw const value;

        // Should not panic
        prefetch_read(ptr);
    }

    #[test]
    fn test_prefetch_write_valid_pointer() {
        let mut value: u64 = 42;
        let ptr = &raw mut value;

        // Should not panic
        prefetch_write(ptr);
    }

    #[test]
    fn test_prefetch_array() {
        let array: [u64; 16] = [0; 16];

        // Prefetch multiple cache lines
        for i in (0..16).step_by(8) {
            prefetch_read(&raw const array[i]);
        }
    }
}
