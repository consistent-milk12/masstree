use seize::LocalGuard;
use std::ptr as StdPtr;

use crate::{
    key::Key,
    leaf_trait::LayerCapableLeaf,
    tree::generic::optimistic_reads::{search_leaf_multi_layer, LookupResult},
    value::traits::LeafValueLoad,
    MassTreeGeneric, NodeAllocatorGeneric, TreePermutation, ValueSlot,
};

impl<S, L, A> MassTreeGeneric<S, L, A>
where
    S: ValueSlot,
    S::Value: Send + Sync + 'static,
    S::Output: Send + Sync + Clone,
    L: LayerCapableLeaf<S> + LeafValueLoad<S>,
    A: NodeAllocatorGeneric<S, L>,
{
    /// Get a value by key, returning a clone of the output.
    ///
    /// This is the main read path for all storage modes, including true-inline.
    /// Uses optimistic concurrency control with version validation.
    ///
    /// # Optimizations
    ///
    /// - **Single-layer fast path**: Keys ≤8 bytes skip suffix/layer checks
    /// - **Contention escape**: Uses `try_stable()` with B-link fallback
    /// - **Deleted leaf recovery**: Handles concurrent coalesce operations
    ///
    /// # Arguments
    ///
    /// * `key` - The key to look up (byte slice)
    /// * `guard` - A guard from [`MassTreeGeneric::guard()`]
    ///
    /// # Returns
    ///
    /// * `Some(output)` - The value if found
    /// * `None` - If the key was not found
    #[inline(always)]
    pub fn get_with_guard(&self, key: &[u8], guard: &LocalGuard<'_>) -> Option<S::Output> {
        let mut key: Key<'_> = Key::new(key);

        // Find root
        let layer_root: *const u8 = self.load_root_ptr_generic(guard);

        if layer_root.is_null() {
            return None;
        }

        // Dispatch to single-layer or multi-layer path
        if key.has_suffix() {
            self.get_with_guard_multi_layer(&mut key, layer_root, guard)
        } else {
            self.get_with_guard_single_layer(&key, layer_root, guard)
        }
    }

    /// Single-layer fast path for keys <=8 bytes.
    ///
    /// Optimized for the common case of short keys:
    /// - No suffix comparison needed
    /// - No layer pointer detection needed
    /// - Simpler search loop
    /// - Stores pointer directly (matches `get_ref` pattern)
    #[inline(always)]
    fn get_with_guard_single_layer(
        &self,
        key: &Key<'_>,
        layer_root: *const u8,
        guard: &LocalGuard<'_>,
    ) -> Option<S::Output> {
        let target_ikey: u64 = key.ikey();

        #[expect(clippy::cast_possible_truncation, reason = "current_len() <= 8")]
        let search_keylenx: u8 = key.current_len() as u8;

        // Navigate from root to leaf
        let mut leaf_ptr: *mut L =
            self.reach_leaf_concurrent_generic(layer_root, key, false, guard);

        'leaf_loop: loop {
            // SAFETY: leaf_ptr protected by guard
            let leaf: &L = unsafe { &*leaf_ptr };

            // Handle deleted leaf (concurrent coalesce)
            if leaf.version().is_deleted() {
                leaf_ptr = self.handle_deleted_leaf(leaf, layer_root, key, false, guard);

                continue 'leaf_loop;
            }

            // Prefetch ikey cache lines while checking version
            leaf.prefetch_for_search();

            // OPTIM: Use try_stable() to avoid spinning on locked leaf.
            // If locked, check B-link chain, key may have moved to sibling.
            let mut version: u32 = if let Some(v) = leaf.version().try_stable() {
                v
            } else {
                // Leaf is locked, try B-link escape
                if let Some(next_ptr) = self.check_blink_chain(leaf, target_ikey) {
                    leaf_ptr = next_ptr;

                    continue 'leaf_loop;
                }

                // No escape route, must wait
                leaf.version().stable()
            };

            'search_loop: loop {
                let perm = leaf.permutation();
                let size: usize = perm.size();
                let mut found_ptr: *mut u8 = StdPtr::null_mut();

                // Simple linear search - store pointer directly (no redundant read)
                for i in 0..size {
                    let slot: usize = perm.get(i);

                    if (leaf.ikey(slot) == target_ikey) && (leaf.keylenx(slot) == search_keylenx) {
                        let ptr: *mut u8 = leaf.leaf_value_ptr(slot);

                        if !ptr.is_null() {
                            found_ptr = ptr;

                            break;
                        }
                    }
                }

                // Read output before version validation (OCC pattern)
                // SAFETY: ptr came from valid slot, guard protects from deallocation
                let output: Option<S::Output> = if found_ptr.is_null() {
                    None
                } else {
                    Some(unsafe { S::output_from_raw(found_ptr) })
                };

                // Version validation after all reads (common case: unchanged)
                if leaf.version().has_changed(version) {
                    if leaf.version().has_split_no_compiler_fence(version) {
                        let (advanced, new_version) =
                            self.advance_to_key_generic(leaf, key, version, guard);

                        if !StdPtr::eq(advanced, leaf) {
                            leaf_ptr = StdPtr::from_ref(advanced).cast_mut();

                            continue 'leaf_loop;
                        }

                        version = new_version;
                    } else {
                        // Update only, re-stabilize without B-link check
                        version = leaf.version().stable();
                    }

                    continue 'search_loop;
                }

                // Version validated, safe to return
                if output.is_some() {
                    return output;
                }

                // Not found, check dirty or B-link
                if leaf.version().is_dirty() {
                    version = leaf.version().stable();

                    continue 'search_loop;
                }

                if let Some(next_ptr) = self.check_blink_chain(leaf, target_ikey) {
                    leaf_ptr = next_ptr;

                    continue 'leaf_loop;
                }

                return None;
            }
        }
    }

    /// Multi-layer path for keys >8 bytes.
    ///
    /// Handles:
    /// - Suffix comparison for keys with same 8-byte prefix
    /// - Layer pointer detection and descent
    /// - `deleted_layer()` recovery (GC'd sublayer)
    /// - `check_sublayer_valid` before descent
    /// - Stores pointer directly (matches `get_ref` pattern)
    #[inline(always)]
    #[expect(clippy::too_many_lines, reason = "Complex multi-layer logic")]
    fn get_with_guard_multi_layer(
        &self,
        key: &mut Key<'_>,
        initial_root: *const u8,
        guard: &LocalGuard<'_>,
    ) -> Option<S::Output> {
        let mut layer_root: *const u8 = initial_root;
        let mut in_sublayer: bool = false;

        'layer_loop: loop {
            layer_root = self.maybe_parent_generic(layer_root);

            let mut leaf_ptr: *mut L =
                self.reach_leaf_concurrent_generic(layer_root, key, in_sublayer, guard);

            'leaf_loop: loop {
                // SAFETY: leaf_ptr protected by guard
                let leaf: &L = unsafe { &*leaf_ptr };

                // Handle deleted leaf (concurrent coalesce)
                if leaf.version().is_deleted() {
                    leaf_ptr = self.handle_deleted_leaf(leaf, layer_root, key, in_sublayer, guard);

                    continue 'leaf_loop;
                }

                // Prefetch ikey cache lines while checking version
                leaf.prefetch_for_search();

                // OPTIMIZATION: Use try_stable() to avoid spinning
                let mut version: u32 = if let Some(v) = leaf.version().try_stable() {
                    v
                } else {
                    let target_ikey: u64 = key.ikey();

                    if let Some(next_ptr) = self.check_blink_chain(leaf, target_ikey) {
                        leaf_ptr = next_ptr;

                        continue 'leaf_loop;
                    }

                    leaf.version().stable()
                };

                'search_loop: loop {
                    // Check for GC'd sublayer - must restart from root
                    if leaf.deleted_layer() {
                        key.unshift_all();
                        layer_root = self.load_root_ptr_generic(guard);
                        in_sublayer = false;

                        continue 'layer_loop;
                    }

                    let target_ikey: u64 = key.ikey();
                    let result: LookupResult = search_leaf_multi_layer::<S, L>(leaf, key);

                    match result {
                        LookupResult::ValueSlot(slot) => {
                            // Read pointer and extract output BEFORE version validation
                            // Store pointer directly - no redundant read via try_load_output
                            let ptr: *mut u8 = leaf.leaf_value_ptr(slot);

                            // SAFETY: ptr came from valid slot, guard protects from deallocation
                            let output: Option<S::Output> = if ptr.is_null() {
                                None
                            } else {
                                Some(unsafe { S::output_from_raw(ptr) })
                            };

                            if leaf.version().has_changed(version) {
                                if leaf.version().has_split_no_compiler_fence(version) {
                                    let (new_ptr, new_version, changed_leaf) =
                                        self.handle_version_change(leaf, key, version, guard);

                                    if changed_leaf {
                                        leaf_ptr = new_ptr;

                                        continue 'leaf_loop;
                                    }

                                    version = new_version;
                                } else {
                                    version = leaf.version().stable();
                                }

                                continue 'search_loop;
                            }

                            return output;
                        }

                        LookupResult::Layer(layer_ptr) => {
                            // Validate version BEFORE descending
                            if leaf.version().has_changed(version) {
                                if leaf.version().has_split_no_compiler_fence(version) {
                                    let (new_ptr, new_version, changed_leaf) =
                                        self.handle_version_change(leaf, key, version, guard);

                                    if changed_leaf {
                                        leaf_ptr = new_ptr;

                                        continue 'leaf_loop;
                                    }

                                    version = new_version;
                                } else {
                                    version = leaf.version().stable();
                                }

                                continue 'search_loop;
                            }

                            // Check sublayer is not deleted before descent
                            if !self.check_sublayer_valid(layer_ptr) {
                                return None;
                            }

                            key.shift();
                            layer_root = layer_ptr;
                            in_sublayer = true;

                            continue 'layer_loop;
                        }

                        LookupResult::NotFound => {
                            // Validate before returning None
                            if leaf.version().has_changed(version) {
                                if leaf.version().has_split_no_compiler_fence(version) {
                                    let (new_ptr, new_version, changed_leaf) =
                                        self.handle_version_change(leaf, key, version, guard);

                                    if changed_leaf {
                                        leaf_ptr = new_ptr;

                                        continue 'leaf_loop;
                                    }

                                    version = new_version;
                                } else {
                                    version = leaf.version().stable();
                                }

                                continue 'search_loop;
                            }

                            if leaf.version().is_dirty() {
                                version = leaf.version().stable();

                                continue 'search_loop;
                            }

                            if let Some(next_ptr) = self.check_blink_chain(leaf, target_ikey) {
                                leaf_ptr = next_ptr;

                                continue 'leaf_loop;
                            }

                            return None;
                        }
                    }
                }
            }
        }
    }
}
