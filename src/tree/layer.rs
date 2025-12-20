use std::sync::Arc;
use std::{cmp::Ordering, ptr as StdPtr};

use crate::alloc::NodeAllocator;
use crate::key::Key;
use crate::leaf::{LeafNode, LeafValue};
use crate::permuter::Permuter;
use seize::LocalGuard;

use super::MassTree;

impl<V, const WIDTH: usize, A: NodeAllocator<LeafValue<V>, WIDTH>> MassTree<V, WIDTH, A> {
    /// Create a new layer when two keys share the same ikey.
    ///
    /// This is called when inserting a key that has the same 8-byte prefix
    /// as an existing entry but a different suffix.
    ///
    /// # Arguments
    ///
    /// * `leaf` - The leaf containing the conflicting slot
    /// * `slot` - Physical slot with the existing key
    /// * `new_key` - The key being inserted (already shifted to current layer)
    /// * `new_value` - The value to insert
    ///
    /// # Returns
    ///
    /// A tuple of (`layer_root`, `insert_slot`) where:
    /// - `layer_root` is the new sublayer to continue insertion
    /// - `insert_slot` is the slot in that layer for the new key
    ///
    /// # Safety
    ///
    /// Caller must ensure no concurrent access to the leaf.
    #[expect(clippy::too_many_lines, reason = "Complex layer creation logic")]
    pub(super) unsafe fn make_new_layer(
        &mut self,
        leaf: &mut LeafNode<LeafValue<V>, WIDTH>,
        slot: usize,
        new_key: &mut Key<'_>,
        new_value: Arc<V>,
    ) -> (*mut LeafNode<LeafValue<V>, WIDTH>, usize) {
        // 1. Extract existing key's suffix and create a Key from it
        // Note: from_suffix creates a Key where the suffix bytes ARE the first ikey
        let existing_suffix: &[u8] = leaf.ksuf(slot).unwrap_or(&[]);
        let mut existing_key: Key<'_> = Key::from_suffix(existing_suffix);

        // 2. Get existing value before we modify the slot
        //
        // Slot holds Arc pointer; cloning is safe under the leaf lock.
        let existing_value: Option<Arc<V>> = unsafe { leaf.try_clone_arc(slot) };

        // 3. Shift new_key to get to the suffix portion
        // new_key has the full original key - shift advances past the ikey we matched on
        // existing_key is already at the "next slice" since it was created from the suffix
        if new_key.has_suffix() {
            new_key.shift();
        }
        // Note: existing_key is created from suffix, its ikey IS the next slice - no shift needed

        // FIXED: Use length-aware comparison instead of ikey-only comparison.
        // This correctly handles prefix-of-other cases where one key is exhausted
        // before the other.
        //
        // C++ reference: masstree_insert.hh:64-83, masstree_key.hh:133-146
        let mut cmp: Ordering = existing_key.compare(new_key.ikey(), new_key.current_len());

        // 4. Create twig chain while ikeys match AND both have more bytes
        let mut twig_head: Option<*mut LeafNode<LeafValue<V>, WIDTH>> = None;
        let mut twig_tail: *mut LeafNode<LeafValue<V>, WIDTH> = StdPtr::null_mut();

        while cmp == Ordering::Equal && existing_key.has_suffix() && new_key.has_suffix() {
            // Create intermediate layer node (single entry)
            let twig: Box<LeafNode<LeafValue<V>, WIDTH>> =
                LeafNode::<LeafValue<V>, WIDTH>::new_layer_root();
            // Note: Can't call methods on twig before alloc since it moves into alloc
            let twig_ptr: *mut LeafNode<LeafValue<V>, WIDTH> = self.alloc_leaf(twig);

            // Initialize the twig node
            // SAFETY: twig_ptr is valid, just allocated
            unsafe {
                (*twig_ptr).assign_initialize_for_layer(0, existing_key.ikey());
                (*twig_ptr).set_permutation(Permuter::make_sorted(1));
            }

            // Link to previous twig
            if twig_head.is_some() {
                // Previous twig points to this one
                //  SAFETY: twig_tail is valid, we just created it.
                unsafe {
                    (*twig_tail).set_layer(0, twig_ptr.cast::<u8>());
                }
            } else {
                twig_head = Some(twig_ptr);
            }

            twig_tail = twig_ptr;

            // Shift both keys again
            existing_key.shift();
            new_key.shift();
            cmp = existing_key.compare(new_key.ikey(), new_key.current_len());
        }

        // 5. Create final leaf with both keys (they now have different ikeys)
        let final_leaf = LeafNode::<LeafValue<V>, WIDTH>::new_layer_root();

        // Allocate final_leaf first, before creating guard
        let final_ptr: *mut LeafNode<LeafValue<V>, WIDTH> = self.alloc_leaf(final_leaf);

        // Determine ordering: existing vs new
        //
        // FIXED: Handle Equal case for prefix-of-other scenarios.
        // When cmp is Equal here, one key is a prefix of the other (same ikey,
        // different remaining length). The shorter key (exhausted) is "less than".
        let (first_key, first_val, second_key, second_val, new_slot) = match cmp {
            Ordering::Less => {
                // existing < new : existing at slot 0, new at slot 1
                (existing_key, existing_value, *new_key, Some(new_value), 1)
            }

            Ordering::Greater => {
                // existing > new: new at slot 0, existing at slot 1
                (*new_key, Some(new_value), existing_key, existing_value, 0)
            }

            Ordering::Equal => {
                // One key is prefix of other (same ikey, different remaining length)
                // Shorter key (exhausted or shorter suffix) gets lower slot
                if existing_key.current_len() <= new_key.current_len() {
                    // existing is prefix of new (or equal length)
                    (existing_key, existing_value, *new_key, Some(new_value), 1)
                } else {
                    // new is prefix of existing
                    (*new_key, Some(new_value), existing_key, existing_value, 0)
                }
            }
        };

        // Create guard only for assign_from_key operations (after all allocations)
        // SAFETY: final_ptr is valid from alloc_leaf
        {
            let guard: LocalGuard<'_> = self.collector.enter();
            let final_leaf_ref: &LeafNode<LeafValue<V>, WIDTH> = unsafe { &*final_ptr };
            // Assign both slots using the unified helper
            // SAFETY: guard comes from tree's collector
            unsafe {
                final_leaf_ref.assign_from_key(0, &first_key, first_val, &guard);
                final_leaf_ref.assign_from_key(1, &second_key, second_val, &guard);
            }
        } // guard dropped here

        // Set up permutation (size 2, but slot for new key not yet "inserted")
        // The caller will call finish_insert to add the new slot to permutation
        // SAFETY: final_ptr is valid
        unsafe {
            let final_leaf_ref: &LeafNode<LeafValue<V>, WIDTH> = &*final_ptr;
            let mut perm: Permuter<WIDTH> = Permuter::make_sorted(2);
            perm.remove_to_back(new_slot); // Remove new slot from "visible" permutation
            final_leaf_ref.set_permutation(perm);
        }

        // 6. Link twig chain to final leaf
        if twig_tail.is_null() {
            // No twig chain, final leaf is directly linked
            twig_head = Some(final_ptr);
        } else {
            // SAFETY: twig_tail is valid
            unsafe {
                (*twig_tail).set_layer(0, final_ptr.cast::<u8>());
            }
        }

        // 7. Update original slot to point to layer
        // INVARIANT: twig_head is always set because:
        // - Either the while loop ran at least once (setting twig_head), OR
        // - The loop didn't run and we set twig_head = Some(final_ptr) above
        #[expect(
            clippy::option_if_let_else,
            reason = "match is more readable for invariant"
        )]
        let layer_root: *mut LeafNode<LeafValue<V>, WIDTH> = match twig_head {
            Some(ptr) => ptr,
            None => unreachable!("twig_head is always set before this point"),
        };

        // CRITICAL: Drop the original Arc before overwriting with layer pointer.
        // try_clone_arc() above incremented the refcount; we must drop the original
        // to avoid leaking it when set_layer overwrites the slot.
        // SAFETY: slot contains a valid Arc pointer (we just cloned it above)
        let old_ptr: *mut u8 = leaf.take_leaf_value_ptr(slot);
        if !old_ptr.is_null() {
            // SAFETY: old_ptr came from Arc::into_raw in a previous insert
            let _old_arc: Arc<V> = unsafe { Arc::from_raw(old_ptr.cast::<V>()) };
            // _old_arc is dropped here, decrementing refcount
        }

        leaf.set_layer(slot, layer_root.cast::<u8>());

        (final_ptr, new_slot)
    }
}
