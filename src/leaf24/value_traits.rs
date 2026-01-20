use super::{LeafNode24, LocalGuard, StdPtr, ValueSlot};

use crate::{
    BatchedRetire,
    value::traits::{
        LeafValueClear, LeafValueLoad, LeafValueStore, LeafValueTake, LeafValueUpdate,
    },
};

// ============================================================================
//  Leaf Value Trait Implementations for LeafNode24<S> (Pointer-Backed)
// ============================================================================

impl<S: ValueSlot> LeafValueLoad<S> for LeafNode24<S>
where
    S::Output: Clone,
{
    #[inline(always)]
    fn try_load_output(&self, slot: usize) -> Option<<S as ValueSlot>::Output> {
        let ptr: *const u8 = self.leaf_value_ptr(slot);

        if ptr.is_null() {
            return None;
        }

        // SAFETY: ptr is non-null and came from output_to_raw
        Some(unsafe { S::output_from_raw(ptr) })
    }
}

impl<S: ValueSlot> LeafValueStore<S> for LeafNode24<S>
where
    S::Value: Send + Sync + 'static,
    S::Output: Send + Sync,
{
    #[inline(always)]
    fn store_value_output(
        &self,
        slot: usize,
        output: &<S as ValueSlot>::Output,
        _guard: &LocalGuard<'_>,
    ) {
        let ptr: *mut u8 = S::output_to_raw(output);

        self.set_leaf_value_ptr(slot, ptr);
    }
}

impl<S: ValueSlot> LeafValueUpdate<S> for LeafNode24<S>
where
    S::Value: Send + Sync + 'static,
    S::Output: Send + Sync + Clone,
{
    #[inline(always)]
    fn replace_value_output(
        &self,
        slot: usize,
        new_output: <S as ValueSlot>::Output,
        guard: &LocalGuard<'_>,
    ) -> <S as ValueSlot>::Output {
        let old_ptr: *mut u8 = self.leaf_value_ptr(slot);
        let old_output: S::Output = unsafe { S::output_from_raw(old_ptr) };
        let new_ptr: *mut u8 = S::output_consume_to_raw(new_output);
        self.set_leaf_value_ptr(slot, new_ptr);

        if S::NEEDS_RETIREMENT {
            // Use batched retirement to amortize coordination overhead
            unsafe {
                BatchedRetire::defer_value::<S>(old_ptr, guard);
            }
        }

        old_output
    }
}

impl<S: ValueSlot> LeafValueClear<S> for LeafNode24<S>
where
    S::Value: Send + Sync + 'static,
    S::Output: Send + Sync,
{
    #[inline(always)]
    fn clear_value_output(&self, slot: usize, guard: &LocalGuard<'_>) {
        let old_ptr: *mut u8 = self.leaf_value_ptr(slot);
        self.set_leaf_value_ptr(slot, StdPtr::null_mut());

        if !old_ptr.is_null() && S::NEEDS_RETIREMENT {
            // Use batched retirement to amortize coordination overhead
            unsafe {
                BatchedRetire::defer_value::<S>(old_ptr, guard);
            }
        }
    }
}

impl<S: ValueSlot> LeafValueTake<S> for LeafNode24<S>
where
    S::Value: Send + Sync + 'static,
    S::Output: Send + Sync + Clone,
{
    #[inline(always)]
    fn take_value_output(
        &self,
        slot: usize,
        guard: &LocalGuard<'_>,
    ) -> Option<<S as ValueSlot>::Output> {
        let old_ptr: *mut u8 = self.leaf_value_ptr(slot);

        if old_ptr.is_null() {
            return None;
        }

        let output: S::Output = unsafe { S::output_from_raw(old_ptr) };
        self.set_leaf_value_ptr(slot, StdPtr::null_mut());

        if S::NEEDS_RETIREMENT {
            // Use batched retirement to amortize coordination overhead
            unsafe {
                BatchedRetire::defer_value::<S>(old_ptr, guard);
            }
        }

        Some(output)
    }
}
