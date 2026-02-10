use seize::LocalGuard;
use std::ptr as StdPtr;

use crate::leaf15::LeafNode15;
use crate::{
    LeafPolicy, MassTreeGeneric, NodeVersion, TreeAllocator,
    hints::unlikely,
    key::Key,
    tree::generic::optimistic_reads::{LookupResult, search_leaf_multi_layer},
};

impl<P, A> MassTreeGeneric<P, A>
where
    P: LeafPolicy,
    A: TreeAllocator<P>,
{
    /// Get a value by key, returning a clone of the output.
    ///
    /// This is the main read path for all storage modes, including true-inline.
    /// Uses optimistic concurrency control with version validation.
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
    pub fn get_with_guard(&self, key: &[u8], guard: &LocalGuard<'_>) -> Option<P::Output> {
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
    #[inline]
    fn get_with_guard_single_layer(
        &self,
        key: &Key<'_>,
        layer_root: *const u8,
        guard: &LocalGuard<'_>,
    ) -> Option<P::Output> {
        let target_ikey: u64 = key.ikey();

        #[expect(clippy::cast_possible_truncation, reason = "current_len() <= 8")]
        let search_keylenx: u8 = key.current_len() as u8;

        // Navigate from root to leaf
        let mut leaf_ptr: *mut LeafNode15<P> =
            self.reach_leaf_concurrent_generic(layer_root, key, false, guard);

        'leaf_loop: loop {
            // SAFETY: leaf_ptr protected by guard
            let leaf: &LeafNode15<P> = unsafe { &*leaf_ptr };

            // Handle deleted leaf (concurrent coalesce)
            if leaf.version().is_deleted() {
                leaf_ptr = self.handle_deleted_leaf(leaf, layer_root, key, false, guard);

                continue 'leaf_loop;
            }

            // OPTIM: Use try_stable() to avoid spinning on locked leaf.
            //
            // If locked, check B-link chain, key may have moved to sibling.
            let mut version: u32 = if let Some(v) = leaf.version().try_stable() {
                leaf.prefetch_for_search();
                v
            } else {
                // Leaf is locked, try B-link escape
                if let Some(next_ptr) = self.check_blink_chain(leaf, target_ikey, guard) {
                    leaf_ptr = next_ptr;

                    continue 'leaf_loop;
                }

                // No escape route, must wait - prefetch while spinning
                leaf.prefetch_for_search();
                leaf.version().stable()
            };

            // Early too-right check
            if !leaf.prev(guard).is_null() && target_ikey < leaf.ikey_bound() {
                // Reload root to get latest pointer after concurrent modifications
                leaf_ptr = self.reach_leaf_concurrent_generic(layer_root, key, false, guard);

                continue 'leaf_loop;
            }

            'search_loop: loop {
                let perm = leaf.permutation();
                let size: usize = perm.size();

                // Simple linear search - store slot index directly
                let mut found_slot: Option<usize> = None;
                for i in 0..size {
                    let slot: usize = perm.get(i);

                    // Use Relaxed ordering - permutation() Acquire already synchronizes
                    if (leaf.ikey_relaxed(slot) == target_ikey)
                        && (leaf.keylenx(slot) == search_keylenx)
                        && !leaf.is_value_empty(slot)
                    {
                        found_slot = Some(slot);
                        break;
                    }
                }

                // Read output before version validation (OCC pattern)
                let output: Option<P::Output> = found_slot.and_then(|s| leaf.load_value(s));

                // Version validation after all reads (common case: unchanged)
                // Store version reference once to avoid repeated method calls
                let ver: &NodeVersion = leaf.version();
                if unlikely(ver.has_changed(version)) {
                    if unlikely(ver.has_split_no_compiler_fence(version)) {
                        let (advanced, new_version) =
                            self.advance_to_key_generic(leaf, key, version, guard);

                        if !StdPtr::eq(advanced, leaf) {
                            leaf_ptr = StdPtr::from_ref(advanced).cast_mut();

                            continue 'leaf_loop;
                        }

                        version = new_version;
                    } else {
                        // Update only, re-stabilize without B-link check
                        version = ver.stable();
                    }

                    continue 'search_loop;
                }

                // Version validated, safe to return
                if output.is_some() {
                    return output;
                }

                // Not found, check dirty or B-link
                if unlikely(ver.is_dirty()) {
                    version = ver.stable();

                    continue 'search_loop;
                }

                if let Some(next_ptr) = self.check_blink_chain(leaf, target_ikey, guard) {
                    leaf_ptr = next_ptr;

                    continue 'leaf_loop;
                }

                // Fallback too-right check
                //
                // NOTE: This is defense-in-depth; the early check above catches most cases.
                if unlikely(!leaf.prev(guard).is_null() && target_ikey < leaf.ikey_bound()) {
                    leaf_ptr = self.reach_leaf_concurrent_generic(layer_root, key, false, guard);

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
    #[inline]
    #[expect(clippy::too_many_lines, reason = "Complex multi-layer logic")]
    fn get_with_guard_multi_layer(
        &self,
        key: &mut Key<'_>,
        initial_root: *const u8,
        guard: &LocalGuard<'_>,
    ) -> Option<P::Output> {
        let mut layer_root: *const u8 = initial_root;
        let mut in_sublayer: bool = false;

        // DEBUG: detect infinite loops during concurrent gc_layer debugging
        #[cfg(debug_assertions)]
        let mut layer_iters: u32 = 0;

        'layer_loop: loop {
            #[cfg(debug_assertions)]
            {
                layer_iters += 1;
                if layer_iters > 500 {
                    eprintln!(
                        "[DEBUG] layer_loop iter={layer_iters}, \
                         in_sublayer={in_sublayer}, layer_root={layer_root:?}"
                    );
                    if layer_iters > 1000 {
                        eprintln!("[DEBUG] ABORTING: infinite layer_loop detected");
                        return None;
                    }
                }
            }

            layer_root = self.maybe_parent_generic(layer_root);

            let mut leaf_ptr: *mut LeafNode15<P> =
                self.reach_leaf_concurrent_generic(layer_root, key, in_sublayer, guard);

            // DEBUG: detect infinite loops
            #[cfg(debug_assertions)]
            let mut leaf_iters: u32 = 0;

            'leaf_loop: loop {
                #[cfg(debug_assertions)]
                {
                    leaf_iters += 1;
                    if leaf_iters > 500 {
                        eprintln!(
                            "[DEBUG] leaf_loop iter={leaf_iters}, leaf_ptr={leaf_ptr:?}, \
                             in_sublayer={in_sublayer}"
                        );
                        if leaf_iters > 1000 {
                            eprintln!("[DEBUG] ABORTING: infinite leaf_loop");
                            return None;
                        }
                    }
                }

                // OPTIM: Compute ikey once per leaf iteration.
                //
                // key.shift() mutates on layer descent, so this must be per-iteration.
                let target_ikey: u64 = key.ikey();

                // SAFETY: leaf_ptr protected by guard
                let leaf: &LeafNode15<P> = unsafe { &*leaf_ptr };

                // Handle deleted leaf (concurrent coalesce) - rare condition
                if unlikely(leaf.version().is_deleted()) {
                    // If entire sublayer was GC'd (deleted_layer modstate), restart
                    // from tree root. handle_deleted_leaf cannot recover from this:
                    // the sublayer root has no B-link successor and
                    // reach_leaf_concurrent_generic returns the same deleted node,
                    // causing an infinite leaf_loop.
                    if in_sublayer && leaf.deleted_layer() {
                        key.unshift_all();
                        layer_root = self.load_root_ptr_generic(guard);
                        in_sublayer = false;

                        continue 'layer_loop;
                    }

                    leaf_ptr = self.handle_deleted_leaf(leaf, layer_root, key, in_sublayer, guard);

                    continue 'leaf_loop;
                }

                // OPTIM: Use try_stable() to avoid spinning
                let mut version: u32 = if let Some(v) = leaf.version().try_stable() {
                    leaf.prefetch_for_search();
                    v
                } else {
                    if let Some(next_ptr) = self.check_blink_chain(leaf, target_ikey, guard) {
                        leaf_ptr = next_ptr;

                        continue 'leaf_loop;
                    }

                    // Prefetch while spinning
                    leaf.prefetch_for_search();
                    leaf.version().stable()
                };

                // Early too-right check
                if !leaf.prev(guard).is_null() && target_ikey < leaf.ikey_bound() {
                    // Reload root to get latest pointer after concurrent modifications
                    leaf_ptr =
                        self.reach_leaf_concurrent_generic(layer_root, key, in_sublayer, guard);

                    continue 'leaf_loop;
                }

                // DEBUG: detect search_loop infinite loops
                #[cfg(debug_assertions)]
                let mut search_iters: u32 = 0;

                'search_loop: loop {
                    #[cfg(debug_assertions)]
                    {
                        search_iters += 1;
                        if search_iters > 500 {
                            let dl = leaf.deleted_layer();
                            let ver = leaf.version();
                            eprintln!(
                                "[DEBUG] search_loop iter={search_iters}, \
                                 deleted_layer={dl}, version_val={}, \
                                 is_deleted={}, is_dirty={}",
                                ver.value(),
                                ver.is_deleted(),
                                ver.is_dirty()
                            );
                            if search_iters > 1000 {
                                eprintln!("[DEBUG] ABORTING: infinite search_loop");
                                return None;
                            }
                        }
                    }

                    // Check for GC'd sublayer - must restart from root
                    if leaf.deleted_layer() {
                        key.unshift_all();
                        layer_root = self.load_root_ptr_generic(guard);
                        in_sublayer = false;

                        continue 'layer_loop;
                    }

                    // target_ikey already computed at start of 'leaf_loop
                    let result: LookupResult = search_leaf_multi_layer::<P>(leaf, key);

                    // Store version reference once for all validation checks below
                    let ver: &NodeVersion = leaf.version();

                    match result {
                        LookupResult::ValueSlot(slot) => {
                            // Read pointer and extract output BEFORE version validation
                            // Store pointer directly - no redundant read via try_load_output
                            let output: Option<P::Output> = if leaf.is_value_empty(slot) {
                                None
                            } else {
                                leaf.load_value(slot)
                            };

                            if unlikely(ver.has_changed(version)) {
                                if unlikely(ver.has_split_no_compiler_fence(version)) {
                                    let (new_ptr, new_version, changed_leaf) =
                                        self.handle_version_change(leaf, key, version, guard);

                                    if changed_leaf {
                                        leaf_ptr = new_ptr;

                                        continue 'leaf_loop;
                                    }

                                    version = new_version;
                                } else {
                                    version = ver.stable();
                                }

                                continue 'search_loop;
                            }

                            return output;
                        }

                        LookupResult::Layer(layer_ptr) => {
                            // Validate version BEFORE descending
                            if unlikely(ver.has_changed(version)) {
                                if unlikely(ver.has_split_no_compiler_fence(version)) {
                                    let (new_ptr, new_version, changed_leaf) =
                                        self.handle_version_change(leaf, key, version, guard);

                                    if changed_leaf {
                                        leaf_ptr = new_ptr;

                                        continue 'leaf_loop;
                                    }

                                    version = new_version;
                                } else {
                                    version = ver.stable();
                                }

                                continue 'search_loop;
                            }

                            // Check sublayer is not deleted before descent
                            if unlikely(!self.check_sublayer_valid(layer_ptr)) {
                                return None;
                            }

                            key.shift();
                            layer_root = layer_ptr;
                            in_sublayer = true;

                            continue 'layer_loop;
                        }

                        LookupResult::NotFound => {
                            // Validate before returning None
                            if unlikely(ver.has_changed(version)) {
                                if unlikely(ver.has_split_no_compiler_fence(version)) {
                                    let (new_ptr, new_version, changed_leaf) =
                                        self.handle_version_change(leaf, key, version, guard);

                                    if changed_leaf {
                                        leaf_ptr = new_ptr;

                                        continue 'leaf_loop;
                                    }

                                    version = new_version;
                                } else {
                                    version = ver.stable();
                                }

                                continue 'search_loop;
                            }

                            if unlikely(ver.is_dirty()) {
                                version = ver.stable();

                                continue 'search_loop;
                            }

                            if let Some(next_ptr) = self.check_blink_chain(leaf, target_ikey, guard)
                            {
                                leaf_ptr = next_ptr;

                                continue 'leaf_loop;
                            }

                            // Fallback too-right check: If key < ikey_bound and prev != null,
                            // we descended to a leaf that's to the right of where the key should be.
                            // Recovery requires restart from layer root (can't safely walk left).
                            // NOTE: This is defense-in-depth; the early check above catches most cases.
                            if unlikely(
                                !leaf.prev(guard).is_null() && target_ikey < leaf.ikey_bound(),
                            ) {
                                leaf_ptr = self.reach_leaf_concurrent_generic(
                                    layer_root,
                                    key,
                                    in_sublayer,
                                    guard,
                                );

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
