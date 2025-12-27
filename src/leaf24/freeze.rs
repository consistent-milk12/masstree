//! Permutation Freezing API for `LeafNode24`.

#![allow(dead_code)]

use std::fmt as StdFmt;
use std::hint as StdHint;

use crate::freeze24::Freeze24Utils;
use crate::ordering::{CAS_FAILURE, CAS_SUCCESS, READ_ORD};
use crate::permuter24::Permuter24;
use crate::slot::ValueSlot;

use super::LeafNode24;

/// Maximum spin iterations before falling back to `stable()`.
pub const MAX_SPIN_ITERS: u32 = 16;

/// Error returned when attempting to read a frozen permutation.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Frozen24;

impl StdFmt::Display for Frozen24 {
    fn fmt(&self, f: &mut StdFmt::Formatter<'_>) -> StdFmt::Result {
        write!(f, "permutation is frozen (split in progress)")
    }
}

impl std::error::Error for Frozen24 {}

/// Error returned when attempting to freeze an already-frozen permutation.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct AlreadyFrozen24 {
    /// The raw permutation value that was already frozen.
    pub raw: u128,
}

impl StdFmt::Display for AlreadyFrozen24 {
    fn fmt(&self, f: &mut StdFmt::Formatter<'_>) -> StdFmt::Result {
        write!(
            f,
            "permutation already frozen (raw={:#034x}); possible invariant violation",
            self.raw
        )
    }
}

impl std::error::Error for AlreadyFrozen24 {}

/// RAII guard for a frozen permutation (WIDTH=24).
///
/// Created by [`LeafNode24::freeze_permutation()`]. The guard:
/// - Holds a snapshot of the pre-freeze permutation value
/// - Automatically restores a valid permutation on drop (panic safety)
/// - Must be consumed via [`unfreeze_set_permutation()`] on the success path
#[must_use = "FreezeGuard24 must be consumed via unfreeze_set_permutation()"]
pub struct FreezeGuard24<'a, S: ValueSlot> {
    leaf: &'a LeafNode24<S>,
    snapshot_raw: u128,
    active: bool,
}

impl<'a, S: ValueSlot> FreezeGuard24<'a, S> {
    /// Construct a new [`FreezeGuard24`]
    #[inline(always)]
    pub const fn new(leaf: &'a LeafNode24<S>, snapshot_raw: u128, active: bool) -> Self {
        Self {
            leaf,
            snapshot_raw,
            active,
        }
    }

    /// Get the permutation snapshot captured at freeze time.
    #[must_use]
    #[inline(always)]
    pub const fn snapshot(&self) -> Permuter24 {
        Permuter24::from_value(self.snapshot_raw)
    }

    /// Get the raw snapshot value (for debugging/logging).
    #[must_use]
    #[inline(always)]
    pub const fn snapshot_raw(&self) -> u128 {
        self.snapshot_raw
    }

    /// Set whether the guard is active.
    #[inline(always)]
    pub const fn set_active(&mut self, active: bool) {
        self.active = active;
    }
}

impl<S: ValueSlot> Drop for FreezeGuard24<'_, S> {
    fn drop(&mut self) {
        if !self.active {
            return;
        }

        // Fail-stop recovery: restore a valid permutation so readers don't spin forever.
        self.leaf.permutation_store_raw_release(self.snapshot_raw);
    }
}

impl<S: ValueSlot> StdFmt::Debug for FreezeGuard24<'_, S> {
    fn fmt(&self, f: &mut StdFmt::Formatter<'_>) -> StdFmt::Result {
        f.debug_struct("FreezeGuard24")
            .field("snapshot_raw", &format_args!("{:#034x}", self.snapshot_raw))
            .field("active", &self.active)
            .finish()
    }
}

// ============================================================================
// FreezeGuardOps Implementation
// ============================================================================

impl<S: ValueSlot> crate::leaf_trait::FreezeGuardOps<Permuter24> for FreezeGuard24<'_, S> {
    #[inline(always)]
    fn snapshot(&self) -> Permuter24 {
        FreezeGuard24::snapshot(self)
    }

    #[inline(always)]
    fn snapshot_raw(&self) -> u128 {
        FreezeGuard24::snapshot_raw(self)
    }

    #[inline(always)]
    fn set_active(&mut self, active: bool) {
        FreezeGuard24::set_active(self, active);
    }
}

impl<S: ValueSlot> LeafNode24<S> {
    /// Try to load permutation, returning error if frozen.
    pub(crate) fn permutation_try(&self) -> Result<Permuter24, Frozen24> {
        let raw: u128 = self.permutation.load_raw(READ_ORD);

        if Freeze24Utils::is_frozen(raw) {
            #[cfg(feature = "tracing")]
            tracing::trace!(
                leaf_ptr = ?std::ptr::from_ref(self),
                raw = format_args!("{raw:#034x}"),
                "permutation_try: FROZEN detected"
            );
            Err(Frozen24)
        } else {
            Ok(Permuter24::from_value(raw))
        }
    }

    /// Wait for permutation to be unfrozen.
    ///
    /// This spins with progressive backoff until the permutation is no longer frozen.
    /// A frozen permutation indicates a split is in progress.
    pub(crate) fn permutation_wait(&self) -> Permuter24 {
        #[cfg(feature = "tracing")]
        tracing::debug!(
            leaf_ptr = ?std::ptr::from_ref(self),
            "permutation_wait: ENTER - waiting for unfrozen permutation"
        );

        loop {
            let raw: u128 = self.permutation.load_raw(READ_ORD);

            if !Freeze24Utils::is_frozen(raw) {
                #[cfg(feature = "tracing")]
                tracing::trace!(
                    leaf_ptr = ?std::ptr::from_ref(self),
                    "permutation_wait: EXIT - permutation unfrozen"
                );
                return Permuter24::from_value(raw);
            }

            // Progressive backoff: spin briefly before blocking on stable()
            for _ in 0..MAX_SPIN_ITERS {
                StdHint::spin_loop();

                let raw: u128 = self.permutation.load_raw(READ_ORD);

                if !Freeze24Utils::is_frozen(raw) {
                    #[cfg(feature = "tracing")]
                    tracing::trace!(
                        leaf_ptr = ?std::ptr::from_ref(self),
                        "permutation_wait: EXIT - unfrozen during spin"
                    );
                    return Permuter24::from_value(raw);
                }
            }

            // A frozen permutation implies a split is in progress.
            // Wait for the version to stabilize (split to complete).
            #[cfg(feature = "tracing")]
            tracing::trace!(
                leaf_ptr = ?std::ptr::from_ref(self),
                version = self.version().value(),
                "permutation_wait: calling stable() - waiting for split"
            );
            let _ = self.version().stable();
        }
    }

    /// Freeze the permutation for split operations.
    ///
    /// # Preconditions
    /// - Caller must hold the leaf lock
    /// - Caller must have called `mark_split()`
    pub(crate) fn freeze_permutation(&self) -> FreezeGuard24<'_, S> {
        // Preconditions (fail-fast in release builds)
        assert!(
            self.version().is_locked(),
            "freeze_permutation: must hold leaf lock"
        );
        assert!(
            self.version().is_splitting(),
            "freeze_permutation: must have called mark_split()"
        );

        #[cfg(feature = "tracing")]
        tracing::debug!(
            leaf_ptr = ?std::ptr::from_ref(self),
            version = self.version().value(),
            "freeze_permutation: ENTER - installing freeze sentinel"
        );

        // CAS loop to install freeze sentinel
        #[cfg(feature = "tracing")]
        let mut cas_retries: u32 = 0;

        loop {
            let raw: u128 = self.permutation.load_raw(READ_ORD);

            // Check for already-frozen (logic error)
            assert!(
                !Freeze24Utils::is_frozen(raw),
                "freeze_permutation: permutation already frozen (raw={raw:#034x}). \
                 This indicates a logic error: either the lock wasn't held, \
                 or a previous split didn't unfreeze",
            );

            let frozen_raw: u128 = Freeze24Utils::freeze_raw(raw);

            // Attempt to install freeze sentinel
            if self
                .permutation
                .compare_exchange_raw(raw, frozen_raw, CAS_SUCCESS, CAS_FAILURE)
                .is_ok()
            {
                #[cfg(feature = "tracing")]
                tracing::debug!(
                    leaf_ptr = ?std::ptr::from_ref(self),
                    cas_retries,
                    old_raw = format_args!("{raw:#034x}"),
                    frozen_raw = format_args!("{frozen_raw:#034x}"),
                    "freeze_permutation: SUCCESS - permutation frozen"
                );

                // Success: return guard with unfrozen snapshot
                return FreezeGuard24::new(self, raw, true);
            }

            #[cfg(feature = "tracing")]
            {
                cas_retries += 1;
                tracing::trace!(
                    leaf_ptr = ?std::ptr::from_ref(self),
                    cas_retries,
                    "freeze_permutation: CAS failed, retrying"
                );
            }

            // A concurrent CAS insert published just before us.
            StdHint::spin_loop();
        }
    }

    /// Try to freeze the permutation, returning an error if already frozen.
    pub(crate) fn try_freeze_permutation(&self) -> Result<FreezeGuard24<'_, S>, AlreadyFrozen24> {
        assert!(
            self.version().is_locked(),
            "try_freeze_permutation: must hold leaf lock"
        );
        assert!(
            self.version().is_splitting(),
            "try_freeze_permutation: must have called mark_split()"
        );

        loop {
            let raw: u128 = self.permutation.load_raw(READ_ORD);

            if Freeze24Utils::is_frozen(raw) {
                return Err(AlreadyFrozen24 { raw });
            }

            let frozen_raw: u128 = Freeze24Utils::freeze_raw(raw);

            match self
                .permutation
                .compare_exchange_raw(raw, frozen_raw, CAS_SUCCESS, CAS_FAILURE)
            {
                Ok(_) => {
                    return Ok(FreezeGuard24::new(self, raw, true));
                }
                Err(_) => {
                    StdHint::spin_loop();
                }
            }
        }
    }

    /// Unfreeze the permutation and publish the final split result.
    #[inline(always)]
    pub(crate) fn unfreeze_set_permutation(
        &self,
        mut guard: FreezeGuard24<'_, S>,
        perm: Permuter24,
    ) {
        #[cfg(feature = "tracing")]
        tracing::debug!(
            leaf_ptr = ?std::ptr::from_ref(self),
            new_perm_size = perm.size(),
            new_perm_raw = format_args!("{:#034x}", perm.value()),
            "unfreeze_set_permutation: UNFREEZING permutation"
        );

        // Disable rollback on drop
        guard.set_active(false);

        // Store the new valid permutation (this is the unfreeze)
        self.set_permutation(perm);
    }
}

#[cfg(test)]
#[expect(clippy::unwrap_used)]
mod tests {
    use super::*;
    use crate::value::LeafValue;

    #[test]
    fn test_permutation_try_unfrozen() {
        let leaf: Box<LeafNode24<LeafValue<u64>>> = LeafNode24::new();
        let result = leaf.permutation_try();
        assert!(result.is_ok());
        assert_eq!(result.unwrap().size(), 0);
    }

    #[test]
    fn test_permutation_try_frozen() {
        let leaf: Box<LeafNode24<LeafValue<u64>>> = LeafNode24::new();

        // Freeze the permutation manually
        let raw = leaf.permutation_raw();
        let frozen = Freeze24Utils::freeze_raw(raw);
        leaf.permutation_store_raw_release(frozen);

        let result = leaf.permutation_try();
        assert!(result.is_err());
    }

    #[test]
    fn test_freeze_guard_restores_on_drop() {
        let leaf: Box<LeafNode24<LeafValue<u64>>> = LeafNode24::new_root();

        let original_raw = leaf.permutation_raw();
        let frozen = Freeze24Utils::freeze_raw(original_raw);
        leaf.permutation_store_raw_release(frozen);

        // Create guard and drop it (simulating panic recovery)
        {
            let guard = FreezeGuard24::new(&leaf, original_raw, true);
            assert!(Freeze24Utils::is_frozen(leaf.permutation_raw()));
            drop(guard);
        }

        // Should be restored
        assert!(!Freeze24Utils::is_frozen(leaf.permutation_raw()));
        assert_eq!(leaf.permutation_raw(), original_raw);
    }

    #[test]
    fn test_freeze_guard_no_restore_when_inactive() {
        let leaf: Box<LeafNode24<LeafValue<u64>>> = LeafNode24::new_root();

        let original_raw = leaf.permutation_raw();
        let frozen = Freeze24Utils::freeze_raw(original_raw);
        leaf.permutation_store_raw_release(frozen);

        // Create guard and deactivate it before drop
        {
            let mut guard = FreezeGuard24::new(&leaf, original_raw, true);
            guard.set_active(false);
            drop(guard);
        }

        // Should still be frozen
        assert!(Freeze24Utils::is_frozen(leaf.permutation_raw()));
    }
}
