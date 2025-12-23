//! ==================================================================
//!  Permutation Freezing API
//! ==================================================================
#![allow(dead_code)]

use std::hint as StdHint;

use crate::{
    AlreadyFrozen, FreezeGuard, Frozen, LeafFreezeUtils, ValueSlot,
    ordering::{CAS_FAILURE, CAS_SUCCESS, READ_ORD, WRITE_ORD},
    permuter::Permuter,
};

use super::LeafNode;

/// Maximum spin iterations before falling back to `stable()`.
pub const MAX_SPIN_ITERS: u32 = 16;

impl<S: ValueSlot, const WIDTH: usize> LeafNode<S, WIDTH> {
    pub(crate) fn permutation_try(&self) -> Result<Permuter<WIDTH>, Frozen> {
        let raw: u64 = self.permutation.load(READ_ORD);

        if LeafFreezeUtils::is_frozen::<WIDTH>(raw) {
            #[cfg(feature = "tracing")]
            tracing::trace!(
                leaf_ptr = ?std::ptr::from_ref(self),
                raw = format_args!("{raw:#018x}"),
                "permutation_try: FROZEN detected"
            );
            Err(Frozen)
        } else {
            Ok(Permuter::from_value(raw))
        }
    }

    pub(crate) fn permutation_wait(&self) -> Permuter<WIDTH> {
        const MAX_STABLE_RETRIES: u32 = 100;
        let mut stable_retries: u32 = 0;

        #[cfg(feature = "tracing")]
        tracing::debug!(
            leaf_ptr = ?std::ptr::from_ref(self),
            "permutation_wait: ENTER - waiting for unfrozen permutation"
        );

        loop {
            let raw: u64 = self.permutation.load(READ_ORD);

            if !LeafFreezeUtils::is_frozen::<WIDTH>(raw) {
                #[cfg(feature = "tracing")]
                tracing::trace!(
                    leaf_ptr = ?std::ptr::from_ref(self),
                    stable_retries,
                    "permutation_wait: EXIT - permutation unfrozen"
                );
                return Permuter::from_value(raw);
            }

            // Progressive backoff: spin briefly before blocking on stable()
            let mut spun: u32 = 0;

            while spun < MAX_SPIN_ITERS {
                StdHint::spin_loop();
                spun += 1;

                let raw: u64 = self.permutation.load(READ_ORD);

                if !LeafFreezeUtils::is_frozen::<WIDTH>(raw) {
                    #[cfg(feature = "tracing")]
                    tracing::trace!(
                        leaf_ptr = ?std::ptr::from_ref(self),
                        stable_retries,
                        spun,
                        "permutation_wait: EXIT - unfrozen during spin"
                    );
                    return Permuter::from_value(raw);
                }
            }

            // A frozen permutation implies a split is in progress.
            // The splitter must have set `SPLITTING_BIT` before freezing, so
            // `stable()` will wait for the split critical section to complete.
            #[cfg(feature = "tracing")]
            tracing::trace!(
                leaf_ptr = ?std::ptr::from_ref(self),
                stable_retries,
                version = self.version().value(),
                "permutation_wait: calling stable() - waiting for split"
            );
            let _ = self.version().stable();

            stable_retries += 1;

            // Bounded wait: if we've waited too long, return an empty permutation.
            // The caller should revalidate version and retry if this happens.
            // This prevents infinite hangs while maintaining correctness via OCC.
            if stable_retries >= MAX_STABLE_RETRIES {
                #[cfg(feature = "tracing")]
                tracing::warn!(
                    stable_retries,
                    leaf_ptr = ?std::ptr::from_ref(self),
                    version = self.version().value(),
                    "permutation_wait: TIMEOUT - exceeded max retries, returning empty"
                );
                // Return empty permutation - caller will detect stale state via version check
                return Permuter::empty();
            }
        }
    }

    pub(crate) fn freeze_permutation(&self) -> FreezeGuard<'_, S, WIDTH> {
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
            let raw: u64 = self.permutation.load(READ_ORD);

            // Check for already-frozen (logic error)
            // FATAL: Under our preconditions (we hold the lock, split intent set),
            // this should never happen. Treat as a fatal invariant violation.
            assert!(
                !LeafFreezeUtils::is_frozen::<WIDTH>(raw),
                "freeze_permutation: permutation already frozen (raw={raw:#018x}). \
                                This indicates a logic error: either the lock wasn't held, \
                                or a previous split didn't unfreeze",
            );

            let frozen_raw: u64 = LeafFreezeUtils::freeze_raw::<WIDTH>(raw);

            // Attempt to install from sentinel
            match self
                .permutation
                .compare_exchange(raw, frozen_raw, CAS_SUCCESS, CAS_FAILURE)
            {
                Ok(_) => {
                    // TEST HOOK: Allow tests to inject barriers after freeze
                    #[cfg(test)]
                    crate::tree::test_hooks::call_after_freeze_hook();

                    #[cfg(feature = "tracing")]
                    tracing::debug!(
                        leaf_ptr = ?std::ptr::from_ref(self),
                        cas_retries,
                        old_raw = format_args!("{raw:#018x}"),
                        frozen_raw = format_args!("{frozen_raw:#018x}"),
                        "freeze_permutation: SUCCESS - permutation frozen"
                    );

                    // Success: return guard with unfrozen snapshot
                    return FreezeGuard::new(self, raw, true);
                }

                Err(_) => {
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
                    // Retry with the new permutation value.
                    StdHint::spin_loop();
                }
            }
        }
    }

    /// Try to freeze the permutation, returning an error if already frozen.
    ///
    /// This is a diagnostic variant of [`Self::freeze_permutation()`] that
    /// returns `Err(AlreadyFrozen)` instead of panicking if the permutation
    /// is already frozen.
    ///
    /// # Use Case
    ///
    /// Primarily for testing and diagnostics. Normal split code should use
    /// [`Self::freeze_permutation()`] which panics on invariant violations.
    ///
    /// # Preconditions
    ///
    /// - Caller must hold the leaf lock
    /// - Caller must have called `mark_split()`
    ///
    /// # Returns
    ///
    /// - `Ok(FreezeGuard)` on success
    /// - `Err(AlreadyFrozen)` if already frozen
    pub(crate) fn try_freeze_permutation(
        &self,
    ) -> Result<FreezeGuard<'_, S, WIDTH>, AlreadyFrozen> {
        // Preconditions (still assert, even for try_ variant)
        assert!(
            self.version().is_locked(),
            "try_freeze_permutation: must hold leaf lock"
        );
        assert!(
            self.version().is_splitting(),
            "try_freeze_permutation: must have called mark_split()"
        );

        // CAS loop to install freeze sentinel
        loop {
            let raw: u64 = self.permutation.load(READ_ORD);

            // Check for already-frozen - return error instead of panic
            if LeafFreezeUtils::is_frozen::<WIDTH>(raw) {
                return Err(AlreadyFrozen { raw });
            }

            let frozen_raw: u64 = LeafFreezeUtils::freeze_raw::<WIDTH>(raw);

            match self
                .permutation
                .compare_exchange(raw, frozen_raw, CAS_SUCCESS, CAS_FAILURE)
            {
                Ok(_) => {
                    return Ok(FreezeGuard::new(self, raw, true));
                }
                Err(_) => {
                    StdHint::spin_loop();
                }
            }
        }
    }

    /// Store a raw permutation value with Release ordering.
    ///
    /// Thisd is a low-level helper for [`FreezeGuard::drop()`]. It stores
    /// `raw` directly without any validation.
    ///
    /// # Use case
    /// Used by [`FreezeGuard`] to restore the original permutation on panic.
    /// Normal code should use [`LeafNode::set_permutation()`] instead.
    #[inline(always)]
    pub(crate) fn permutation_store_raw_release(&self, raw: u64) {
        self.permutation.store(raw, WRITE_ORD);
    }

    /// Unfreeze the permutation and publish the final split result.
    ///
    /// Consumes the freeze guard, stores the new permutation, and disables
    /// the guard's rollback behavior.
    ///
    /// # Arguments
    /// * `guard` - The freeze guard from `freeze_permutation()`
    /// * `perm` - The new permutation to publish (post-split)
    ///
    /// # Memory Ordering
    /// Uses Release store to synchronize with readers' Acquire loads
    #[inline]
    pub(crate) fn unfreeze_set_permutation(
        &self,
        mut guard: FreezeGuard<'_, S, WIDTH>,
        perm: Permuter<WIDTH>,
    ) {
        #[cfg(feature = "tracing")]
        tracing::debug!(
            leaf_ptr = ?std::ptr::from_ref(self),
            new_perm_size = perm.size(),
            new_perm_raw = format_args!("{:#018x}", perm.value()),
            "unfreeze_set_permutation: UNFREEZING permutation"
        );

        // Disable rollback on drop
        guard.set_active(false);

        // Store the new valid permutation (this is the unfreeze)
        self.set_permutation(perm);
    }
}
