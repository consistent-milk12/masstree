//! Software prefetching utilities for cache optimization.
//!
//! Provides hardware-specific prefetch hints to reduce memory latency
//! during tree traversal. When the CPU knows we're about to access a
//! memory location, it can begin fetching it into cache while we
//! continue processing the current node.
//!
//! # Architecture Support
//!
//! - **`x86_64`**: Uses `_mm_prefetch` with `_MM_HINT_T0` (read), `_MM_HINT_ET0` (write), or `_MM_HINT_NTA` (non-temporal)
//! - **`aarch64`**: Uses inline `PRFM` instruction (stable, no feature gates)
//! - **Other**: No-op (safe fallback)
//!
//! # Feature Flags
//!
//! - **`no-prefetch`**: Disables all software prefetching, making all functions no-ops.
//!   Use this for A/B benchmarking to measure the actual benefit of prefetching.
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

/// Prefetch data for reading into all cache levels (temporal).
///
/// This is a hint to the CPU that we're about to read from the given
/// pointer. The CPU may begin fetching the cache line(s) containing
/// this address into L1/L2/L3 cache.
///
/// # Safety
///
/// This function is safe to call with any pointer, including null or invalid.
/// Prefetch instructions are hints that never fault on `x86_64`/`aarch64`.
///
/// # Performance Notes
///
/// - Prefetching is most effective when there's work to do between
///   the prefetch and the actual access (e.g., version validation)
/// - Over-prefetching can pollute the cache; use judiciously
/// - The prefetch distance should match your access pattern
/// - No null check is performed to avoid branch overhead in hot paths
///
/// # Feature Flags
///
/// When `no-prefetch` feature is enabled, this function is a no-op.
#[inline(always)]
pub fn prefetch_read<T>(ptr: *const T) {
    // When no-prefetch is enabled, all prefetch calls become no-ops for A/B benchmarking
    #[cfg(feature = "no-prefetch")]
    {
        let _ = ptr;
        return;
    }

    // NOTE: No null check. Prefetch instructions are no-ops for null/invalid
    // addresses on x86_64 and aarch64. Removing the branch improves performance
    // in tight loops where prefetch is called frequently.

    #[cfg(all(not(feature = "no-prefetch"), target_arch = "x86_64"))]
    {
        // SAFETY: _mm_prefetch is always safe to call.
        // It's a hint that may be ignored by the CPU.
        // Invalid/null addresses cause no fault (unlike actual loads).
        unsafe {
            std::arch::x86_64::_mm_prefetch(ptr.cast::<i8>(), std::arch::x86_64::_MM_HINT_T0);
        }
    }

    #[cfg(all(not(feature = "no-prefetch"), target_arch = "aarch64"))]
    {
        // Use inline asm instead of unstable std::arch::aarch64::_prefetch.
        // PRFM PLDL1KEEP, [ptr] - Prefetch for load, L1 cache, keep in cache.
        // SAFETY: PRFM is always safe - it's a hint that doesn't fault on invalid addresses.
        unsafe {
            std::arch::asm!(
                "prfm pldl1keep, [{ptr}]",
                ptr = in(reg) ptr,
                options(nostack, preserves_flags),
            );
        }
    }

    // No-op on unsupported architectures (or when no-prefetch is enabled)
    #[cfg(not(any(
        feature = "no-prefetch",
        target_arch = "x86_64",
        target_arch = "aarch64"
    )))]
    {
        let _ = ptr;
    }
}

/// Prefetch data for reading with non-temporal hint (streaming access).
///
/// Unlike [`prefetch_read`], this hints that the data will only be used once
/// and shouldn't pollute the cache hierarchy. Use for streaming scans where
/// data won't be revisited.
///
/// # When to Use
///
/// - Large sequential scans that won't revisit data
/// - Bulk operations where cache pollution is a concern
/// - For repeated access patterns, prefer [`prefetch_read`]
///
/// # Feature Flags
///
/// When `no-prefetch` feature is enabled, this function is a no-op.
#[inline(always)]
pub fn prefetch_read_nta<T>(ptr: *const T) {
    #[cfg(feature = "no-prefetch")]
    {
        let _ = ptr;
        return;
    }

    #[cfg(all(not(feature = "no-prefetch"), target_arch = "x86_64"))]
    {
        // SAFETY: _mm_prefetch is always safe to call.
        // _MM_HINT_NTA = non-temporal access, minimizes cache pollution.
        unsafe {
            std::arch::x86_64::_mm_prefetch(ptr.cast::<i8>(), std::arch::x86_64::_MM_HINT_NTA);
        }
    }

    #[cfg(all(not(feature = "no-prefetch"), target_arch = "aarch64"))]
    {
        // PRFM PLDL1STRM - Prefetch for load, L1 cache, streaming (non-temporal).
        // SAFETY: PRFM is always safe - it's a hint that doesn't fault.
        unsafe {
            std::arch::asm!(
                "prfm pldl1strm, [{ptr}]",
                ptr = in(reg) ptr,
                options(nostack, preserves_flags),
            );
        }
    }

    #[cfg(not(any(
        feature = "no-prefetch",
        target_arch = "x86_64",
        target_arch = "aarch64"
    )))]
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
///
/// # Feature Flags
///
/// When `no-prefetch` feature is enabled, this function is a no-op.
#[inline(always)]
pub fn prefetch_write<T>(ptr: *mut T) {
    #[cfg(feature = "no-prefetch")]
    {
        let _ = ptr;
        return;
    }

    // NOTE: No null check. Prefetch instructions are no-ops for null/invalid
    // addresses on x86_64 and aarch64.

    #[cfg(all(not(feature = "no-prefetch"), target_arch = "x86_64"))]
    {
        // SAFETY: _mm_prefetch is always safe to call.
        // _MM_HINT_ET0 prefetches into exclusive state, avoiding a later
        // shared→exclusive upgrade when we write. Supported on all modern
        // x86_64 CPUs (Intel Broadwell+, AMD Bulldozer+).
        unsafe {
            std::arch::x86_64::_mm_prefetch(ptr.cast::<i8>(), std::arch::x86_64::_MM_HINT_ET0);
        }
    }

    #[cfg(all(not(feature = "no-prefetch"), target_arch = "aarch64"))]
    {
        // Use inline asm instead of unstable std::arch::aarch64::_prefetch.
        // PRFM PSTL1KEEP, [ptr] - Prefetch for store, L1 cache, keep in cache.
        // SAFETY: PRFM is always safe - it's a hint that doesn't fault on invalid addresses.
        unsafe {
            std::arch::asm!(
                "prfm pstl1keep, [{ptr}]",
                ptr = in(reg) ptr,
                options(nostack, preserves_flags),
            );
        }
    }

    #[cfg(not(any(
        feature = "no-prefetch",
        target_arch = "x86_64",
        target_arch = "aarch64"
    )))]
    {
        let _ = ptr;
    }
}

#[cfg(test)]
#[expect(clippy::indexing_slicing)]
mod tests {
    use super::*;
    use std::ptr as StdPtr;

    #[test]
    fn test_prefetch_null_is_safe() {
        // Prefetch instructions are no-ops for null pointers on x86_64/aarch64.
        // Should not panic or crash.
        prefetch_read::<u64>(StdPtr::null());
        prefetch_read_nta::<u64>(StdPtr::null());
        prefetch_write::<u64>(StdPtr::null_mut());
    }

    #[test]
    fn test_prefetch_valid_pointer() {
        let value: u64 = 42;
        let ptr = &raw const value;

        // Should not panic
        prefetch_read(ptr);
    }

    #[test]
    fn test_prefetch_read_nta_valid_pointer() {
        let value: u64 = 42;
        let ptr = &raw const value;

        // Should not panic
        prefetch_read_nta(ptr);
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

    #[test]
    fn test_prefetch_nta_streaming() {
        let array: [u64; 64] = [0; 64];

        // Simulate streaming access pattern with non-temporal prefetch
        for i in (0..64).step_by(8) {
            prefetch_read_nta(&raw const array[i]);
        }
    }
}
