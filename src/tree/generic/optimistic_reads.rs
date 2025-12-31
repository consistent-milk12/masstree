//! ========================================================================
//!  Generic Optimistic Read Path
//! ========================================================================
//!
//! Refactored for performance with:
//! - `#[inline(always)]` on hot path helpers
//! - `#[cold]` on retry/error paths
//! - Unified implementation via closure for value extraction

use super::{
    Key, LayerCapableLeaf, LocalGuard, MassTreeGeneric, NodeAllocatorGeneric, NodeVersion,
    ValueSlot,
};

use crate::leaf_trait::TreePermutation;
use crate::leaf24::KSUF_KEYLENX;
use crate::leaf24::LAYER_KEYLENX;
use crate::link::{is_marked, unmark_ptr};

// ============================================================================
//  LookupResult - Search outcome enum
// ============================================================================

/// Result of searching a leaf node for a key.
///
/// This enum captures the three possible outcomes without interpreting
/// the pointer until after version validation.
enum LookupResult {
    /// Found a value pointer. The `keylenx` confirms it's a value (< LAYER_KEYLENX).
    Value(*mut u8),

    /// Found a layer pointer. Need to descend into sublayer.
    Layer(*mut u8),

    /// Key not found in this leaf.
    NotFound,
}

// ============================================================================
//  Search Helpers (Hot Path)
// ============================================================================

/// Search a leaf for a key in single-layer mode (keys <= 8 bytes).
///
/// Optimized path that skips:
/// - Suffix comparison logic
/// - Layer descent handling (short keys can't descend)
///
/// # Safety
///
/// Caller must ensure `leaf` is valid and protected by a guard.
#[inline(always)]
fn search_leaf_single_layer<S, L>(leaf: &L, target_ikey: u64, search_keylenx: u8) -> LookupResult
where
    S: ValueSlot,
    S::Value: Send + Sync + 'static,
    S::Output: Send + Sync,
    L: LayerCapableLeaf<S>,
{
    let perm = leaf.permutation();

    for i in 0..perm.size() {
        let slot: usize = perm.get(i);
        let slot_ikey: u64 = leaf.ikey(slot);

        if slot_ikey != target_ikey {
            continue;
        }

        let slot_keylenx: u8 = leaf.keylenx(slot);
        let slot_ptr: *mut u8 = leaf.leaf_value_ptr(slot);

        if slot_ptr.is_null() {
            continue;
        }

        // Exact keylenx match = found
        if slot_keylenx == search_keylenx {
            return LookupResult::Value(slot_ptr);
        }

        // Layer pointer with matching ikey means a longer key exists,
        // but our short key (<=8 bytes) is NOT a match - continue searching
        // (Layer pointers have keylenx >= 128, our key has keylenx <= 8)
    }

    LookupResult::NotFound
}

/// Search a leaf for a key in multi-layer mode (keys > 8 bytes).
///
/// Handles:
/// - Suffix comparison for keys with same 8-byte prefix
/// - Layer pointer detection for descent
///
/// # Safety
///
/// Caller must ensure `leaf` is valid and protected by a guard.
#[inline(always)]
fn search_leaf_multi_layer<S, L>(leaf: &L, key: &Key<'_>) -> LookupResult
where
    S: ValueSlot,
    S::Value: Send + Sync + 'static,
    S::Output: Send + Sync,
    L: LayerCapableLeaf<S>,
{
    let perm = leaf.permutation();
    let target_ikey: u64 = key.ikey();

    #[expect(clippy::cast_possible_truncation, reason = "current_len() <= 8")]
    let search_keylenx: u8 = if key.has_suffix() {
        KSUF_KEYLENX
    } else {
        key.current_len() as u8
    };

    for i in 0..perm.size() {
        let slot: usize = perm.get(i);
        let slot_ikey: u64 = leaf.ikey(slot);

        if slot_ikey != target_ikey {
            continue;
        }

        let slot_keylenx: u8 = leaf.keylenx(slot);
        let slot_ptr: *mut u8 = leaf.leaf_value_ptr(slot);

        if slot_ptr.is_null() {
            continue;
        }

        if slot_keylenx == search_keylenx {
            // Potential exact match - verify suffix if present
            let suffix_match: bool = if slot_keylenx == KSUF_KEYLENX {
                leaf.ksuf_equals(slot, key.suffix())
            } else {
                true
            };

            if suffix_match {
                return LookupResult::Value(slot_ptr);
            }
        } else if slot_keylenx >= LAYER_KEYLENX && key.has_suffix() {
            // Layer pointer - record for descent after validation
            return LookupResult::Layer(slot_ptr);
        }
    }

    LookupResult::NotFound
}

// ============================================================================
//  Cold Path Helpers (Retry / Error Paths)
// ============================================================================

impl<S, L, A> MassTreeGeneric<S, L, A>
where
    S: ValueSlot,
    S::Value: Send + Sync + 'static,
    S::Output: Send + Sync,
    L: LayerCapableLeaf<S>,
    A: NodeAllocatorGeneric<S, L>,
{
    /// Handle version change during optimistic read.
    ///
    /// Called when version validation fails. Follows B-link chain if split
    /// occurred, otherwise returns new version for retry.
    ///
    /// Returns `(new_leaf_ptr, should_restart_leaf_loop)`:
    /// - If leaf changed: `(new_ptr, true)`
    /// - If same leaf, new version: `(same_ptr, false)` with updated version
    #[cold]
    #[inline(never)]
    fn handle_version_change<'a>(
        &self,
        leaf: &'a L,
        key: &Key<'_>,
        version: u32,
        guard: &LocalGuard<'_>,
    ) -> (*mut L, u32, bool) {
        let (advanced, new_version) = self.advance_to_key_generic(leaf, key, version, guard);

        if !std::ptr::eq(advanced, leaf) {
            // Different leaf - search there
            (std::ptr::from_ref(advanced).cast_mut(), new_version, true)
        } else {
            // Same leaf, new version - retry search
            (std::ptr::from_ref(leaf).cast_mut(), new_version, false)
        }
    }

    /// Check if key should be in a sibling leaf via B-link.
    ///
    /// Returns `Some(next_leaf_ptr)` if we should follow the B-link,
    /// `None` if key is definitively not found.
    #[cold]
    #[inline(never)]
    fn check_blink_chain(&self, leaf: &L, target_ikey: u64) -> Option<*mut L> {
        let next_raw: *mut L = leaf.next_raw();
        let next_ptr: *mut L = unmark_ptr(next_raw);

        if !next_ptr.is_null() && !is_marked(next_raw) {
            // SAFETY: next_ptr is valid (protected by guard in caller)
            let next_bound: u64 = unsafe { (*next_ptr).ikey_bound() };
            if target_ikey >= next_bound {
                return Some(next_ptr);
            }
        }

        None
    }

    /// Check if sublayer is deleted before descending.
    ///
    /// Returns `true` if sublayer is valid, `false` if deleted (key not found).
    #[cold]
    #[inline(never)]
    fn check_sublayer_valid(&self, layer_ptr: *mut u8) -> bool {
        // SAFETY: ptr is non-null (came from valid slot) and protected by guard.
        #[expect(clippy::cast_ptr_alignment, reason = "Checked")]
        let sublayer_version: &NodeVersion = unsafe { &*layer_ptr.cast::<NodeVersion>() };

        !sublayer_version.is_deleted()
    }
}

// ============================================================================
//  Public API
// ============================================================================

impl<S, L, A> MassTreeGeneric<S, L, A>
where
    S: ValueSlot,
    S::Value: Send + Sync + 'static,
    S::Output: Send + Sync,
    L: LayerCapableLeaf<S>,
    A: NodeAllocatorGeneric<S, L>,
{
    /// Get a value by key.
    ///
    /// Creates a guard internally. For bulk operations, prefer
    /// [`get_with_guard`](Self::get_with_guard) to amortize guard creation cost.
    ///
    /// # Returns
    ///
    /// * `Some(Arc<V>)` - If the key was found
    /// * `None` - If the key was not found
    #[must_use]
    #[inline]
    pub fn get(&self, key: &[u8]) -> Option<S::Output> {
        let guard = self.guard();
        self.get_with_guard(key, &guard)
    }

    /// Get a value by key using an explicit guard.
    ///
    /// Use this when performing multiple operations to amortize guard overhead.
    ///
    /// # Arguments
    ///
    /// * `key` - The key to look up (byte slice)
    /// * `guard` - A guard from [`MassTreeGeneric::guard()`]
    ///
    /// # Returns
    ///
    /// * `Some(Arc<V>)` - If the key was found
    /// * `None` - If the key was not found
    #[must_use]
    #[inline(always)]
    pub fn get_with_guard(&self, key: &[u8], guard: &LocalGuard<'_>) -> Option<S::Output> {
        let mut search_key: Key<'_> = Key::new(key);
        self.get_impl(&mut search_key, guard, |ptr| {
            // SAFETY: version validated, ptr points to valid value
            unsafe { S::output_from_raw(ptr) }
        })
    }

    /// Get a borrowed reference to a value by key.
    ///
    /// This is significantly faster than [`get_with_guard`] for read-heavy workloads
    /// because it avoids atomic reference count operations (Arc clone/drop).
    ///
    /// # Performance
    ///
    /// Under high concurrency, `get_ref` can be **2-5x faster** than `get_with_guard`
    /// because it eliminates cache line bouncing on shared Arc reference counts.
    ///
    /// # Arguments
    ///
    /// * `key` - The key to look up (byte slice)
    /// * `guard` - A guard from [`MassTreeGeneric::guard()`]
    ///
    /// # Returns
    ///
    /// * `Some(&V)` - A reference to the value, valid for the guard's lifetime
    /// * `None` - If the key was not found
    #[must_use]
    #[inline(always)]
    pub fn get_ref<'g>(&self, key: &[u8], guard: &'g LocalGuard<'_>) -> Option<&'g S::Value> {
        let mut search_key: Key<'_> = Key::new(key);
        self.get_impl(&mut search_key, guard, |ptr| {
            // SAFETY: version validated, guard protects from deallocation
            unsafe { &*(ptr.cast::<S::Value>()) }
        })
    }

    /// Unified get implementation.
    ///
    /// Both `get_with_guard` and `get_ref` delegate to this function.
    /// The `extract` closure handles the difference in return type.
    ///
    /// # Type Parameters
    ///
    /// * `R` - Return type (`S::Output` or `&'g S::Value`)
    /// * `F` - Closure that extracts the value from a raw pointer
    #[inline(always)]
    fn get_impl<R, F>(&self, key: &mut Key<'_>, guard: &LocalGuard<'_>, extract: F) -> Option<R>
    where
        F: Fn(*mut u8) -> R,
    {
        // Detect single-layer mode: key <= 8 bytes means no suffix, no layer descent
        let single_layer_mode: bool = !key.has_suffix();

        // Start at tree root
        let mut layer_root: *const u8 = self.load_root_ptr_generic(guard);
        let mut in_sublayer: bool = false;

        'layer_loop: loop {
            // Find the actual layer root (handles layer root promotion)
            layer_root = self.maybe_parent_generic(layer_root);

            // Traverse to leaf for current layer
            let mut leaf_ptr: *mut L =
                self.reach_leaf_concurrent_generic(layer_root, key, in_sublayer, guard);

            // Inner loop for searching within a leaf (may follow B-links)
            'leaf_loop: loop {
                // SAFETY: leaf_ptr protected by guard
                let leaf: &L = unsafe { &*leaf_ptr };

                // Take version snapshot (spins if dirty)
                let mut version: u32 = leaf.version().stable();

                'search_loop: loop {
                    // Check for deleted node
                    if leaf.version().is_deleted() {
                        continue 'layer_loop;
                    }

                    let target_ikey: u64 = key.ikey();

                    // ============================================================
                    //  Search Phase (hot path)
                    // ============================================================
                    let result: LookupResult = if single_layer_mode {
                        #[expect(clippy::cast_possible_truncation, reason = "current_len() <= 8")]
                        let search_keylenx: u8 = key.current_len() as u8;
                        search_leaf_single_layer::<S, L>(leaf, target_ikey, search_keylenx)
                    } else {
                        search_leaf_multi_layer::<S, L>(leaf, key)
                    };

                    // ============================================================
                    //  Version Validation (must happen AFTER all reads)
                    // ============================================================
                    if leaf.version().has_changed(version) {
                        let (new_ptr, new_version, changed_leaf) =
                            self.handle_version_change(leaf, key, version, guard);

                        if changed_leaf {
                            leaf_ptr = new_ptr;
                            continue 'leaf_loop;
                        }

                        version = new_version;
                        continue 'search_loop;
                    }

                    // ============================================================
                    //  Interpret Result (version validated)
                    // ============================================================
                    match result {
                        LookupResult::Value(ptr) => {
                            return Some(extract(ptr));
                        }

                        LookupResult::Layer(ptr) => {
                            // Verify sublayer not deleted before descending
                            if !self.check_sublayer_valid(ptr) {
                                #[cfg(feature = "tracing")]
                                tracing::debug!(
                                    layer_ptr = ?ptr,
                                    "get: sublayer deleted, key not found"
                                );
                                return None;
                            }

                            // Descend into sublayer
                            key.shift();
                            layer_root = ptr;
                            in_sublayer = true;
                            continue 'layer_loop;
                        }

                        LookupResult::NotFound => {
                            // Check for dirty (concurrent modification)
                            if leaf.version().is_dirty() {
                                version = leaf.version().stable();
                                continue 'search_loop;
                            }

                            // Check B-link chain
                            if let Some(next_ptr) = self.check_blink_chain(leaf, target_ikey) {
                                leaf_ptr = next_ptr;
                                continue 'leaf_loop;
                            }

                            // Truly not found
                            return None;
                        }
                    }
                }
            }
        }
    }
}
