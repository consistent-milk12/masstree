//! Software prefetching utilities for cache optimization.
//!
//! Provides hardware-specific prefetch hints to reduce memory latency
//! during tree traversal. When the CPU knows we're about to access a
//! memory location, it can begin fetching it into cache while we
//! continue processing the current node.

/// Prefetch data for reading into all cache levels (temporal).
#[inline(always)]
#[allow(clippy::missing_const_for_fn)]
pub fn prefetch_read<T>(ptr: *const T) {
    // When no-prefetch is enabled, all prefetch calls become no-ops for A/B benchmarking
    #[cfg(feature = "no-prefetch")]
    {
        let _ = ptr;
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

/// Prefetch data for writing into all cache levels.
#[inline(always)]
#[allow(dead_code)]
#[allow(clippy::missing_const_for_fn)]
pub fn prefetch_write<T>(ptr: *mut T) {
    #[cfg(feature = "no-prefetch")]
    #[expect(clippy::needless_return, reason = "Feature gate compatibility")]
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
