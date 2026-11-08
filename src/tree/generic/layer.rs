//! Prepare unpublished twig chains outside the parent lock, then publish them.

use std::ptr as StdPtr;

use crate::TreeAllocator;
use crate::leaf15::{KSUF_KEYLENX, LAYER_KEYLENX, LeafNode15};
use crate::nodeversion::LockGuard;
use crate::permuter::Permuter;
use crate::policy::{LeafPolicy, PreparedValue};
use seize::LocalGuard;

use super::{Key, MassTreeGeneric, Ordering};

/// Owns only unpublished nodes and key/suffix data until commit. Terminal
/// values stay in their original owners while preparation can be abandoned.
pub(super) struct PreparedLayer<'a, 'g, P: LeafPolicy, A: TreeAllocator<P>> {
    head: *mut LeafNode15<P>,
    tail: *mut LeafNode15<P>,
    existing_slot: usize,
    allocator: &'a A,
    guard: &'a LocalGuard<'g>,
}

impl<P: LeafPolicy, A: TreeAllocator<P>> PreparedLayer<'_, '_, P, A> {
    fn push_leaf(&mut self) -> &LeafNode15<P> {
        let node = self.allocator.alloc_leaf_direct(false, true);
        if self.head.is_null() {
            self.head = node;
        } else {
            // SAFETY: The tail is an exclusively owned unpublished twig. It
            // has no terminal value; the new node transfers into its layer slot.
            let tail = unsafe { &*self.tail };
            tail.set_keylenx(0, LAYER_KEYLENX);
            // SAFETY: node is initialized and uniquely owned by this builder.
            unsafe { tail.store_layer(0, node.cast()) };
            tail.set_permutation_relaxed(Permuter::make_sorted(1));
        }
        self.tail = node;
        // SAFETY: The builder owns this initialized, unpublished allocation.
        unsafe { &*node }
    }

    /// Transfer values and publish with no allocations or user callbacks.
    ///
    /// # Safety
    /// The parent is locked and its active slot was revalidated as the same
    /// full suffix key. `existing` represents that slot's current value, and
    /// `new` represents the uniquely owned input. Both were encoded before
    /// entering this infallible mutation sequence.
    pub(super) unsafe fn commit(
        mut self,
        parent: &LeafNode15<P>,
        lock: &mut LockGuard<'_>,
        slot: usize,
        existing: &PreparedValue<P>,
        new: &PreparedValue<P>,
    ) {
        lock.mark_insert();
        // SAFETY: The source is the revalidated terminal slot under this lock.
        // Both sealed policies clear the source without retiring its moved value.
        let retire = unsafe { parent.take_value_for_layer(slot) };
        // SAFETY: This handle came from the same policy's source-slot clear.
        unsafe { P::retire_handle(retire, self.guard) };
        // SAFETY: The parent lock is held and marked dirty; the old bytes remain
        // guard-protected while the slot's suffix metadata is cleared.
        unsafe { parent.clear_ksuf(slot, self.guard) };
        parent.set_keylenx(slot, LAYER_KEYLENX);

        // SAFETY: The final leaf is exclusively owned and has two initialized
        // key slots, both without values. The source was cleared above.
        let final_leaf = unsafe { &*self.tail };
        final_leaf.store_prepared_value_relaxed(self.existing_slot, existing);
        final_leaf.store_prepared_value_relaxed(self.existing_slot ^ 1, new);
        // SAFETY: The complete chain and its two values transfer to the parent
        // slot under its dirty lock. This Release store publishes all nodes.
        unsafe { parent.store_layer(slot, self.head.cast()) };
        self.head = StdPtr::null_mut();
    }
}

impl<P: LeafPolicy, A: TreeAllocator<P>> Drop for PreparedLayer<'_, '_, P, A> {
    fn drop(&mut self) {
        let mut node = self.head;
        while !node.is_null() {
            // SAFETY: Uncommitted chains are exclusively owned by this builder.
            // They contain only layer links and suffix data, no terminal values.
            let leaf = unsafe { &*node };
            let next = if leaf.is_layer(0) {
                leaf.load_layer_raw(0).cast()
            } else {
                StdPtr::null_mut()
            };
            // SAFETY: No node in this unpublished chain is reachable from the
            // live tree. Read the next pointer before transferring retirement.
            unsafe { self.allocator.retire_leaf(node, self.guard) };
            node = next;
        }
    }
}

impl<P: LeafPolicy, A: TreeAllocator<P>> MassTreeGeneric<P, A> {
    /// Build all node/key/suffix storage without holding a parent node lock.
    /// Values are deliberately absent until the caller revalidates and commits.
    #[cold]
    pub(super) fn prepare_layer<'a, 'g>(
        &'a self,
        existing_suffix: &[u8],
        new_suffix: &[u8],
        guard: &'a LocalGuard<'g>,
    ) -> PreparedLayer<'a, 'g, P, A> {
        let mut layer = PreparedLayer {
            head: StdPtr::null_mut(),
            tail: StdPtr::null_mut(),
            existing_slot: 0,
            allocator: &self.allocator,
            guard,
        };
        let mut existing_key = Key::from_suffix(existing_suffix);
        let mut new_key = Key::from_suffix(new_suffix);
        let mut cmp = existing_key.compare(new_key.ikey(), new_key.current_len());
        while cmp == Ordering::Equal && existing_key.has_suffix() && new_key.has_suffix() {
            layer.push_leaf().set_ikey(0, existing_key.ikey());
            existing_key.shift();
            new_key.shift();
            cmp = existing_key.compare(new_key.ikey(), new_key.current_len());
        }

        let existing_first = cmp == Ordering::Less
            || (cmp == Ordering::Equal && existing_key.current_len() <= new_key.current_len());
        layer.existing_slot = usize::from(!existing_first);
        let final_leaf = layer.push_leaf();
        let ordered = if existing_first {
            [&existing_key, &new_key]
        } else {
            [&new_key, &existing_key]
        };
        for (slot, key) in ordered.into_iter().enumerate() {
            final_leaf.set_ikey(slot, key.ikey());
            if key.has_suffix() {
                final_leaf.set_keylenx(slot, KSUF_KEYLENX);
                // SAFETY: The builder owns this unpublished leaf; slots are
                // initialized sequentially so suffix growth preserves slot 0.
                unsafe { final_leaf.assign_ksuf_init(slot, key.suffix(), guard) };
            } else {
                #[expect(clippy::cast_possible_truncation, reason = "unsuffixed key <= 8 bytes")]
                final_leaf.set_keylenx(slot, key.current_len() as u8);
            }
        }
        final_leaf.set_permutation_relaxed(Permuter::make_sorted(2));
        layer
    }
}

#[cfg(test)]
mod unit_tests;
