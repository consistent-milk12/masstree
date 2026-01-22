//! Filepath: `src/tree/range/iterator/iter_flags.rs`
//!
//! Packed boolean flags for iterator state.

/// Packed boolean flags for iterator state.
///
/// Uses a u16 bitfield to store 9 boolean flags efficiently.
#[derive(Clone, Copy, Debug, Default)]
pub struct IterFlags(u16);

#[allow(dead_code, reason = "API Completeness")]
impl IterFlags {
    // Bit Positions (forward iteration)
    const EXHAUSTED: u16 = 1 << 0;
    const INITIALIZED: u16 = 1 << 1;
    const EMIT_EQUAL: u16 = 1 << 2;
    const NEEDS_DUPLICATE_CHECK: u16 = 1 << 3;
    const SINGLE_LAYER_MODE: u16 = 1 << 4;
    // Bit Positions (reverse iteration for DoubleEndedIterator)
    const BACK_EXHAUSTED: u16 = 1 << 5;
    const BACK_INITIALIZED: u16 = 1 << 6;
    const BACK_EMIT_EQUAL: u16 = 1 << 7;
    const BACK_NEEDS_DUPLICATE_CHECK: u16 = 1 << 8;

    /// Create new flags with all bits cleared.
    #[inline(always)]
    pub const fn new() -> Self {
        Self(0)
    }

    /// Create flags with initial values for forward iteration.
    #[inline(always)]
    pub const fn with_values(emit_equal: bool, single_layer_mode: bool) -> Self {
        let mut bits: u16 = 0;

        if emit_equal {
            bits |= Self::EMIT_EQUAL;
        }

        if single_layer_mode {
            bits |= Self::SINGLE_LAYER_MODE;
        }

        Self(bits)
    }

    /// Create flags with initial values for both forward and backward iteration.
    #[inline(always)]
    pub const fn with_both_bounds(
        emit_equal: bool,
        single_layer_mode: bool,
        back_emit_equal: bool,
    ) -> Self {
        let mut bits: u16 = 0;

        if emit_equal {
            bits |= Self::EMIT_EQUAL;
        }

        if single_layer_mode {
            bits |= Self::SINGLE_LAYER_MODE;
        }

        if back_emit_equal {
            bits |= Self::BACK_EMIT_EQUAL;
        }

        Self(bits)
    }

    // ========================================================================
    //  Getters
    // ========================================================================

    #[inline(always)]
    pub const fn exhausted(self) -> bool {
        self.0 & Self::EXHAUSTED != 0
    }

    #[inline(always)]
    pub const fn initialized(self) -> bool {
        self.0 & Self::INITIALIZED != 0
    }

    #[inline(always)]
    pub const fn emit_equal(self) -> bool {
        self.0 & Self::EMIT_EQUAL != 0
    }

    #[inline(always)]
    pub const fn needs_duplicate_check(self) -> bool {
        self.0 & Self::NEEDS_DUPLICATE_CHECK != 0
    }

    #[inline(always)]
    pub const fn single_layer_mode(self) -> bool {
        self.0 & Self::SINGLE_LAYER_MODE != 0
    }

    // Back iteration getters
    #[inline(always)]
    pub const fn back_exhausted(self) -> bool {
        self.0 & Self::BACK_EXHAUSTED != 0
    }

    #[inline(always)]
    pub const fn back_initialized(self) -> bool {
        self.0 & Self::BACK_INITIALIZED != 0
    }

    #[inline(always)]
    pub const fn back_emit_equal(self) -> bool {
        self.0 & Self::BACK_EMIT_EQUAL != 0
    }

    #[inline(always)]
    pub const fn back_needs_duplicate_check(self) -> bool {
        self.0 & Self::BACK_NEEDS_DUPLICATE_CHECK != 0
    }

    // ========================================================================
    //  Setters
    // ========================================================================

    #[inline(always)]
    pub const fn set_exhausted(&mut self, value: bool) {
        if value {
            self.0 |= Self::EXHAUSTED;
        } else {
            self.0 &= !Self::EXHAUSTED;
        }
    }

    #[inline(always)]
    pub const fn set_initialized(&mut self, value: bool) {
        if value {
            self.0 |= Self::INITIALIZED;
        } else {
            self.0 &= !Self::INITIALIZED;
        }
    }

    #[inline(always)]
    pub const fn set_emit_equal(&mut self, value: bool) {
        if value {
            self.0 |= Self::EMIT_EQUAL;
        } else {
            self.0 &= !Self::EMIT_EQUAL;
        }
    }

    #[inline(always)]
    pub const fn set_needs_duplicate_check(&mut self, value: bool) {
        if value {
            self.0 |= Self::NEEDS_DUPLICATE_CHECK;
        } else {
            self.0 &= !Self::NEEDS_DUPLICATE_CHECK;
        }
    }

    #[inline(always)]
    pub const fn set_single_layer_mode(&mut self, value: bool) {
        if value {
            self.0 |= Self::SINGLE_LAYER_MODE;
        } else {
            self.0 &= !Self::SINGLE_LAYER_MODE;
        }
    }

    // ========================================================================
    //  Convenience methods
    // ========================================================================

    /// Mark as exhausted.
    #[inline(always)]
    pub const fn mark_exhausted(&mut self) {
        self.0 |= Self::EXHAUSTED;
    }

    /// Mark as initialized.
    #[inline(always)]
    pub const fn mark_initialized(&mut self) {
        self.0 |= Self::INITIALIZED;
    }

    /// Clear `needs_duplicate_check` flag.
    #[inline(always)]
    pub const fn clear_duplicate_check(&mut self) {
        self.0 &= !Self::NEEDS_DUPLICATE_CHECK;
    }

    /// Set `needs_duplicate_check` flag.
    #[inline(always)]
    pub const fn require_duplicate_check(&mut self) {
        self.0 |= Self::NEEDS_DUPLICATE_CHECK;
    }

    /// Disable single-layer mode (fall back to multi-layer).
    #[inline(always)]
    pub const fn disable_single_layer_mode(&mut self) {
        self.0 &= !Self::SINGLE_LAYER_MODE;
    }

    // ========================================================================
    //  Back iteration convenience methods
    // ========================================================================

    /// Mark back iterator as exhausted.
    #[inline(always)]
    pub const fn mark_back_exhausted(&mut self) {
        self.0 |= Self::BACK_EXHAUSTED;
    }

    /// Mark back iterator as initialized.
    #[inline(always)]
    pub const fn mark_back_initialized(&mut self) {
        self.0 |= Self::BACK_INITIALIZED;
    }

    /// Check if both front and back are exhausted (completely consumed).
    #[inline(always)]
    pub const fn fully_exhausted(self) -> bool {
        (self.0 & Self::EXHAUSTED != 0) && (self.0 & Self::BACK_EXHAUSTED != 0)
    }

    /// Clear `back_needs_duplicate_check` flag.
    #[inline(always)]
    pub const fn clear_back_duplicate_check(&mut self) {
        self.0 &= !Self::BACK_NEEDS_DUPLICATE_CHECK;
    }

    /// Set `back_needs_duplicate_check` flag.
    #[inline(always)]
    pub const fn require_back_duplicate_check(&mut self) {
        self.0 |= Self::BACK_NEEDS_DUPLICATE_CHECK;
    }
}
