//! Traits and utilities for true-inline value storage.
//!
//! This module provides the [`InlineBits`] trait that enables small [`Copy`] values
//! to be stored directly in leaf nodes without heap allocation.

use std::fmt as StdFmt;
use std::mem as StdMem;

// ============================================================================
//  InlineBits Trait
// ============================================================================

/// Trait for values that can be stored inline as 64 bits.
///
/// This trait enables true-inline value storage in leaf nodes, where values
/// are stored in [`std::sync::atomic::AtomicU64`] slots instead of behind heap pointers.
///
/// # Safety
/// Implementations must ensure:
/// - `to_bits` and `from_bits` are inverses: `from_bits(to_bits(v)) == v`
/// - The bit representation is deterministic (same value -> same bits)
/// - The type is `Copy + Send + Sync + 'static`
///
/// # Example
///
/// ```ignore
/// use masstree::InlineBits;
///
/// #[derive(Clone, Copy)]
/// struct MyId(u32);
///
/// impl InlineBits for MyId {
///     fn to_bits(self) -> u64 {
///         self.0 as u64
///     }
///
///     fn from_bits(bits: u64) -> Self {
///         Self(bits as u32)
///     }
/// }
/// ```
pub trait InlineBits: Copy + Send + Sync + 'static {
    /// Convert this value to its 64-bit representation.
    fn to_bits(self) -> u64;

    /// Reconstruct the value from its 64-bit representation.
    fn from_bits(bits: u64) -> Self;
}

// ============================================================================
//  Integer Implementations
// ============================================================================

impl InlineBits for u8 {
    #[inline(always)]
    fn to_bits(self) -> u64 {
        u64::from(self)
    }

    #[inline(always)]
    fn from_bits(bits: u64) -> Self {
        #[expect(clippy::cast_possible_truncation, reason = "Intentional truncation")]
        {
            bits as Self
        }
    }
}

impl InlineBits for u16 {
    #[inline(always)]
    fn to_bits(self) -> u64 {
        u64::from(self)
    }

    #[inline(always)]
    fn from_bits(bits: u64) -> Self {
        #[expect(clippy::cast_possible_truncation, reason = "Intentional truncation")]
        {
            bits as Self
        }
    }
}

impl InlineBits for u32 {
    #[inline(always)]
    fn to_bits(self) -> u64 {
        u64::from(self)
    }
    #[inline(always)]
    fn from_bits(bits: u64) -> Self {
        #[expect(clippy::cast_possible_truncation, reason = "Intentional truncation")]
        {
            bits as Self
        }
    }
}

impl InlineBits for u64 {
    #[inline(always)]
    fn to_bits(self) -> u64 {
        self
    }
    #[inline(always)]
    fn from_bits(bits: u64) -> Self {
        bits
    }
}

impl InlineBits for i8 {
    #[inline(always)]
    #[expect(clippy::cast_sign_loss)]
    fn to_bits(self) -> u64 {
        self as u64
    }

    #[inline(always)]
    fn from_bits(bits: u64) -> Self {
        #[expect(clippy::cast_possible_truncation, reason = "Intentional truncation")]
        {
            bits as Self
        }
    }
}

impl InlineBits for i16 {
    #[inline(always)]
    #[expect(clippy::cast_sign_loss)]
    fn to_bits(self) -> u64 {
        self as u64
    }

    #[inline(always)]
    fn from_bits(bits: u64) -> Self {
        #[expect(clippy::cast_possible_truncation, reason = "Intentional truncation")]
        {
            bits as Self
        }
    }
}

impl InlineBits for i32 {
    #[inline(always)]
    #[expect(clippy::cast_sign_loss)]
    fn to_bits(self) -> u64 {
        self as u64
    }

    #[inline(always)]
    fn from_bits(bits: u64) -> Self {
        #[expect(clippy::cast_possible_truncation, reason = "Intentional truncation")]
        {
            bits as Self
        }
    }
}

impl InlineBits for i64 {
    #[inline(always)]
    #[expect(clippy::cast_sign_loss)]
    fn to_bits(self) -> u64 {
        self as u64
    }

    #[inline(always)]
    fn from_bits(bits: u64) -> Self {
        #[expect(clippy::cast_possible_wrap, reason = "Intentional wrap")]
        {
            bits as Self
        }
    }
}

impl InlineBits for f32 {
    #[inline(always)]
    fn to_bits(self) -> u64 {
        u64::from(self.to_bits())
    }

    #[inline(always)]
    fn from_bits(bits: u64) -> Self {
        #[expect(clippy::cast_possible_truncation, reason = "Intentional truncation")]
        {
            Self::from_bits(bits as u32)
        }
    }
}

impl InlineBits for f64 {
    #[inline(always)]
    fn to_bits(self) -> u64 {
        self.to_bits()
    }

    #[inline(always)]
    fn from_bits(bits: u64) -> Self {
        Self::from_bits(bits)
    }
}

#[cfg(target_pointer_width = "64")]
impl InlineBits for usize {
    #[inline(always)]
    fn to_bits(self) -> u64 {
        self as u64
    }

    #[inline(always)]
    fn from_bits(bits: u64) -> Self {
        #[expect(clippy::cast_possible_truncation, reason = "64-bit platform")]
        {
            bits as Self
        }
    }
}

#[cfg(target_pointer_width = "64")]
impl InlineBits for isize {
    #[inline(always)]
    #[expect(clippy::cast_sign_loss)]
    fn to_bits(self) -> u64 {
        self as u64
    }

    #[inline(always)]
    fn from_bits(bits: u64) -> Self {
        #[expect(clippy::cast_possible_wrap, reason = "Intentional wrap")]
        #[expect(clippy::cast_possible_truncation, reason = "64-bit platform")]
        {
            bits as Self
        }
    }
}

impl InlineBits for bool {
    #[inline(always)]
    fn to_bits(self) -> u64 {
        u64::from(self)
    }

    #[inline(always)]
    fn from_bits(bits: u64) -> Self {
        bits != 0
    }
}

impl InlineBits for () {
    #[inline(always)]
    fn to_bits(self) -> u64 {
        0
    }

    #[inline(always)]
    fn from_bits(_bits: u64) -> Self {}
}

// ============================================================================
//  Tuple Implementations (for small pairs)
// ============================================================================

impl InlineBits for (u32, u32) {
    #[inline(always)]
    fn to_bits(self) -> u64 {
        (u64::from(self.0) << 32) | u64::from(self.1)
    }

    #[inline(always)]
    fn from_bits(bits: u64) -> Self {
        #[expect(clippy::cast_possible_truncation, reason = "Intentional truncation")]
        ((bits >> 32) as u32, bits as u32)
    }
}

impl InlineBits for (u16, u16, u16, u16) {
    #[inline(always)]
    fn to_bits(self) -> u64 {
        (u64::from(self.0) << 48)
            | (u64::from(self.1) << 32)
            | (u64::from(self.2) << 16)
            | u64::from(self.3)
    }

    #[inline(always)]
    fn from_bits(bits: u64) -> Self {
        #[expect(clippy::cast_possible_truncation, reason = "Intentional truncation")]
        (
            (bits >> 48) as u16,
            (bits >> 32) as u16,
            (bits >> 16) as u16,
            bits as u16,
        )
    }
}

// ============================================================================
//  Debug Formatting Helper
// ============================================================================

/// Wrapper for debug formatting inline values.
pub struct InlineBitsDebug<V: InlineBits>(pub V);

impl<V: InlineBits + StdFmt::Debug> StdFmt::Debug for InlineBitsDebug<V> {
    fn fmt(&self, f: &mut StdFmt::Formatter<'_>) -> StdFmt::Result {
        write!(f, "Inline({:?})", self.0)
    }
}

// ============================================================================
//  Compile-time Assertions
// ============================================================================

/// Assert that a type fits in 8 bytes (compile-time check).
#[doc(hidden)]
pub const fn assert_fits_in_u64<V: InlineBits>() {
    assert!(
        StdMem::size_of::<V>() <= 8,
        "InlineBits value must fit in 8 bytes"
    );
}
