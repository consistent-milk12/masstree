//! ==================================================================
//!  Permutation Freezing API
//! ==================================================================

use crate::{Frozen, LeafFreezeUtils, ValueSlot, ordering::READ_ORD, permuter::Permuter};

use super::LeafNode;

/// Maximum spin iterations before falling back to `stable()`.
const MAX_SPIN_ITERS: u32 = 16;

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

            todo!()
        }
    }
}
