//! ==================================================================
//!  Permutation Freezing API
//! ==================================================================
#![allow(dead_code)]

use std::hint as StdHint;

use crate::{
    FreezeGuard, Frozen, LeafFreezeUtils, ValueSlot,
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
            Err(Frozen)
        } else {
            Ok(Permuter::from_value(raw))
        }
    }

    pub(crate) fn permutation_wait(&self) -> Permuter<WIDTH> {
        loop {
            let raw: u64 = self.permutation.load(READ_ORD);

            if !LeafFreezeUtils::is_frozen::<WIDTH>(raw) {
                return Permuter::from_value(raw);
            }

            // Progressive backoff: spin briefly before blocking on stable()
            let mut spun: u32 = 0;

            while spun < MAX_SPIN_ITERS {
                // processor instruction hint
                //
                // A common use case for `spin_loop` is implementing bounded optimistic
                // spinning in a CAS loop in synchronization primitives. To avoid problems
                // like priority inversion, it is strongly recommended that the spin loop is
                // terminated after a finite amount of iterations and an appropriate blocking
                // syscall is made.
                StdHint::spin_loop();
                spun += 1;

                let raw: u64 = self.permutation.load(READ_ORD);

                if !LeafFreezeUtils::is_frozen::<WIDTH>(raw) {
                    return Permuter::from_value(raw);
                }
            }

            // A frozen permutation implies a split is in progress.
            // The splitter must have set `SPLITTING_BIT` before freezing, so
            // `stable()` will wait for the split critical section to complete.
            let _ = self.version().stable();
        }
    }

    pub(crate) fn freeze_permutation(&self) -> FreezeGuard<'_, S, WIDTH> {
        // Preconditions (fail-fast in release buiilds)
        assert!(
            self.version().is_locked(),
            "freeze_permutation: must hold leaf lock"
        );
        assert!(
            self.version().is_splitting(),
            "freeze_permutation: must have called mark_split()"
        );

        // CAS loop to install freeze sentinel
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
                    // Success: return guard with unfrozen snapshot
                    return FreezeGuard::new(self, raw, true);
                }

                Err(_) => {
                    // A concurrent CAS insert published just before us.
                    // Retry with the new permutation value.
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
        // Disable rollback on drop
        guard.set_active(false);

        // Store the new valid permutation (this is the unfreeze)
        self.set_permutation(perm);
    }
}
