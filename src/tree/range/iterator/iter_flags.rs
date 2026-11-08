//! Filepath: `src/tree/range/iterator/iter_flags.rs`
//!
//! Packed boolean flags for forward and reverse scan contexts.

// ============================================================================
//  ForwardFlags — flags for ForwardScanCtx
// ============================================================================

/// Packed boolean flags for forward scan context state.
///
/// Uses a u8 bitfield to store 5 boolean flags efficiently.
/// Lives inside `ForwardScanCtx`. The `SINGLE_LAYER_MODE` flag is the
/// canonical source for both forward and reverse paths.
#[derive(Clone, Copy, Debug, Default)]
pub struct ForwardFlags(u8);

#[allow(dead_code, reason = "API Completeness")]
impl ForwardFlags {
    const EXHAUSTED: u8 = 1 << 0;
    const INITIALIZED: u8 = 1 << 1;
    const EMIT_EQUAL: u8 = 1 << 2;
    const NEEDS_DUPLICATE_CHECK: u8 = 1 << 3;
    const SINGLE_LAYER_MODE: u8 = 1 << 4;
    const CURSOR_EMITTED: u8 = 1 << 5;
    const DOWN_FOLLOW_SUFFIX: u8 = 1 << 6;
    const DOWN_FRESH_SUBTREE: u8 = 1 << 7;

    /// Create new flags with all bits cleared.
    #[inline(always)]
    pub const fn new() -> Self {
        Self(0)
    }

    /// Create flags with initial values for forward iteration.
    #[inline(always)]
    pub const fn with_values(emit_equal: bool, single_layer_mode: bool) -> Self {
        let mut bits: u8 = 0;

        if emit_equal {
            bits |= Self::EMIT_EQUAL;
        }

        if single_layer_mode {
            bits |= Self::SINGLE_LAYER_MODE;
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

    /// Whether the cursor key has already been emitted (or descended past).
    ///
    /// Before the first emission, advancing to a later leaf must use an
    /// inclusive lower bound; after it, the lower bound must skip the cursor.
    #[inline(always)]
    pub const fn cursor_emitted(self) -> bool {
        self.0 & Self::CURSOR_EMITTED != 0
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

    /// Record that the cursor key is now the last emitted key.
    #[inline(always)]
    pub const fn mark_cursor_emitted(&mut self) {
        self.0 |= Self::CURSOR_EMITTED;
    }

    /// Record that the cursor was reset to a fresh sublayer minimum.
    #[inline(always)]
    pub const fn clear_cursor_emitted(&mut self) {
        self.0 &= !Self::CURSOR_EMITTED;
    }

    /// Record how the next sublayer descent should initialize the cursor.
    ///
    /// `follow_suffix` means the layer shares the cursor's 8-byte prefix and
    /// the real key suffix must be carried into the sublayer. `fresh_subtree`
    /// means the layer is a different prefix, so every key in it is greater
    /// than the last emitted key and the dedup state must be cleared.
    #[inline(always)]
    pub const fn set_down_mode(&mut self, follow_suffix: bool, fresh_subtree: bool) {
        if follow_suffix {
            self.0 |= Self::DOWN_FOLLOW_SUFFIX;
        } else {
            self.0 &= !Self::DOWN_FOLLOW_SUFFIX;
        }

        if fresh_subtree {
            self.0 |= Self::DOWN_FRESH_SUBTREE;
        } else {
            self.0 &= !Self::DOWN_FRESH_SUBTREE;
        }
    }

    /// Take the pending `follow_suffix` decision, clearing it.
    #[inline(always)]
    pub const fn take_down_follow_suffix(&mut self) -> bool {
        let value: bool = self.0 & Self::DOWN_FOLLOW_SUFFIX != 0;
        self.0 &= !Self::DOWN_FOLLOW_SUFFIX;
        value
    }

    /// Take the pending `fresh_subtree` decision, clearing it.
    #[inline(always)]
    pub const fn take_down_fresh_subtree(&mut self) -> bool {
        let value: bool = self.0 & Self::DOWN_FRESH_SUBTREE != 0;
        self.0 &= !Self::DOWN_FRESH_SUBTREE;
        value
    }
}

// ============================================================================
//  ReverseFlags — extracted reverse-only flags for ReverseScanCtx
// ============================================================================

/// Packed boolean flags for reverse scan context state.
///
/// Uses a u8 bitfield to store 4 boolean flags efficiently.
/// Lives inside `ReverseScanCtx`.
#[derive(Clone, Copy, Debug, Default)]
pub struct ReverseFlags(u8);

#[allow(dead_code, reason = "API Completeness")]
impl ReverseFlags {
    const EXHAUSTED: u8 = 1 << 0;
    const INITIALIZED: u8 = 1 << 1;
    const EMIT_EQUAL: u8 = 1 << 2;
    const NEEDS_DUPLICATE_CHECK: u8 = 1 << 3;
    const UPPER_BOUND: u8 = 1 << 4;
    const CURSOR_EMITTED: u8 = 1 << 5;
    const DOWN_FOLLOW_SUFFIX: u8 = 1 << 6;
    const DOWN_FRESH_SUBTREE: u8 = 1 << 7;

    /// Create new flags with all bits cleared.
    #[inline(always)]
    pub const fn new() -> Self {
        Self(0)
    }

    /// Create flags with initial values for reverse iteration.
    #[inline(always)]
    pub const fn with_values(emit_equal: bool) -> Self {
        let mut bits: u8 = 0;

        if emit_equal {
            bits |= Self::EMIT_EQUAL;
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

    /// Whether we're at an upper bound position (scanning from maximum).
    ///
    /// When true:
    /// - `lower_reverse()` returns `size - 1` (start from last slot)
    /// - `is_duplicate_reverse()` returns false (no filtering needed)
    ///
    /// Set true on sublayer descent via `shift_clear_reverse()`,
    /// cleared by `clear_upper_bound()` after successful emission.
    #[inline(always)]
    pub const fn upper_bound(self) -> bool {
        self.0 & Self::UPPER_BOUND != 0
    }

    /// Whether the cursor currently holds a key this scan already emitted.
    #[inline(always)]
    pub const fn cursor_emitted(self) -> bool {
        self.0 & Self::CURSOR_EMITTED != 0
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

    /// Set `upper_bound` flag (scanning from maximum position).
    #[inline(always)]
    pub const fn set_upper_bound(&mut self) {
        self.0 |= Self::UPPER_BOUND;
    }

    /// Clear `upper_bound` flag after successful key emission.
    #[inline(always)]
    pub const fn clear_upper_bound(&mut self) {
        self.0 &= !Self::UPPER_BOUND;
    }

    /// Record that the cursor now holds an emitted key.
    #[inline(always)]
    pub const fn mark_cursor_emitted(&mut self) {
        self.0 |= Self::CURSOR_EMITTED;
    }

    /// Record that the cursor was reset to a positional boundary that has not
    /// been emitted yet (a fresh sublayer or a previous leaf's lower bound).
    #[inline(always)]
    pub const fn clear_cursor_emitted(&mut self) {
        self.0 &= !Self::CURSOR_EMITTED;
    }

    /// Record how the next sublayer descent should initialize the cursor.
    #[inline(always)]
    pub const fn set_down_mode(&mut self, follow_suffix: bool, fresh_subtree: bool) {
        if follow_suffix {
            self.0 |= Self::DOWN_FOLLOW_SUFFIX;
        } else {
            self.0 &= !Self::DOWN_FOLLOW_SUFFIX;
        }

        if fresh_subtree {
            self.0 |= Self::DOWN_FRESH_SUBTREE;
        } else {
            self.0 &= !Self::DOWN_FRESH_SUBTREE;
        }
    }

    /// Take the pending `follow_suffix` decision, clearing it.
    #[inline(always)]
    pub const fn take_down_follow_suffix(&mut self) -> bool {
        let value: bool = self.0 & Self::DOWN_FOLLOW_SUFFIX != 0;
        self.0 &= !Self::DOWN_FOLLOW_SUFFIX;
        value
    }

    /// Take the pending `fresh_subtree` decision, clearing it.
    #[inline(always)]
    pub const fn take_down_fresh_subtree(&mut self) -> bool {
        let value: bool = self.0 & Self::DOWN_FRESH_SUBTREE != 0;
        self.0 &= !Self::DOWN_FRESH_SUBTREE;
        value
    }
}
