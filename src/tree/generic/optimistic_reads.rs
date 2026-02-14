//! ========================================================================
//!  Generic Optimistic Read Path
//! ========================================================================
//!
//! Refactored for performance with:
//! - `#[inline(always)]` on hot path helpers
//! - Linear search (predictable branches, cache-friendly)
//! - Unified implementation via closure for value extraction

use std::ptr as StdPtr;

use super::{Key, LeafPolicy, LocalGuard, MassTreeGeneric, NodeVersion, TreeAllocator};

use crate::hints::unlikely;
use crate::leaf15::KSUF_KEYLENX;
use crate::leaf15::LAYER_KEYLENX;
use crate::leaf15::LeafNode15;
use crate::link::Linker;
use crate::ref_value_slot::RefLeafPolicy;

mod get_guarded;

// ============================================================================
//  LookupResult - Search outcome enum
// ============================================================================

/// Result of searching a leaf node for a key.
///
/// This enum captures the three possible outcomes without interpreting
/// the pointer until after version validation.
enum LookupResult {
    /// Found a terminal value at the given slot index.
    /// The `keylenx` confirms it's a value (< [`LAYER_KEYLENX`])
    ValueSlot(usize),

    /// Found a layer pointer. Need to descend into sublayer.
    /// Still returns the raw pointer since layer pointers are always real pointers.
    Layer(*mut u8),

    /// Key not found in this leaf.
    NotFound,
}

/// Result of twig-chain fast descent.
///
/// Explicitly encodes the three possible outcomes to avoid sentinel values
/// and null-pointer pitfalls. The `Leaf` parameter is the leaf node type.
enum TwigDescentResult<Leaf> {
    /// Continue to `'leaf_loop` with the given leaf pointer.
    /// The leaf is valid and ready for normal search.
    ContinueLeafLoop {
        layer_root: *const u8,
        leaf_ptr: *mut Leaf,
    },

    /// Restart `'layer_loop` with the given layer root.
    /// Used when: `deleted_layer` detected, or fast path reached a non-leaf/non-root node.
    RestartLayerLoop {
        layer_root: *const u8,
        in_sublayer: bool,
    },

    /// Return `None` from `get_impl_multi_layer`.
    /// Used when: sublayer is deleted (`is_deleted()` check failed).
    ReturnNone,
}

// ============================================================================
//  Search Helpers (Hot Path)
// ============================================================================

/// Search a leaf for a key in multi-layer mode (keys > 8 bytes).
///
/// Handles:
/// - Suffix comparison for keys with same 8-byte prefix
/// - Layer pointer detection for descent
///
/// Optimized with loop unrolling (3 at a time).
///
/// Uses Relaxed ordering for ikey loads after the initial Acquire on permutation.
/// This is safe because:
/// 1. `permutation()` uses Acquire ordering, synchronizing with writer's Release
/// 2. OCC version validation at the end catches any races
///
/// Uses `#[inline]` - medium-sized function with loop unrolling; let compiler
/// decide based on call-site context to avoid I-cache pressure.
#[inline]
#[expect(clippy::collapsible_if, reason = "Leads to unusual regressions?!")]
fn search_leaf_multi_layer<P>(leaf: &LeafNode15<P>, key: &Key<'_>) -> LookupResult
where
    P: LeafPolicy,
{
    // Acquire ordering on permutation synchronizes with writer's Release fence
    let perm = leaf.permutation();
    let size = perm.size();
    let target_ikey: u64 = key.ikey();

    #[expect(clippy::cast_possible_truncation, reason = "current_len() <= 8")]
    let search_keylenx: u8 = if key.has_suffix() {
        KSUF_KEYLENX
    } else {
        key.current_len() as u8
    };

    // OPTIM: Fast path flag - if search key has no suffix, skip suffix/layer checks
    // For inline keys (≤8 bytes), we don't need to compare suffixes or check for layer pointers
    let needs_suffix_check: bool = key.has_suffix();

    let mut i: usize = 0;

    // Unrolled loop: process 3 slots per iteration
    // Speculative batch load: load all slots and ikeys upfront for better ILP
    // Use Relaxed ordering - synchronization already established by permutation load
    while i + 3 <= size {
        // Batch load slots (bit extraction only, no memory access)
        let s0: usize = perm.get(i);
        let s1: usize = perm.get(i + 1);
        let s2: usize = perm.get(i + 2);

        // Batch load ikeys with Relaxed ordering (safe after permutation Acquire)
        let ikey0: u64 = leaf.ikey_relaxed(s0);
        let ikey1: u64 = leaf.ikey_relaxed(s1);
        let ikey2: u64 = leaf.ikey_relaxed(s2);

        // Now check sequentially with early exit
        if ikey0 == target_ikey {
            if let Some(result) =
                check_slot_match(leaf, s0, search_keylenx, key, needs_suffix_check)
            {
                return result;
            }
        }

        if ikey1 == target_ikey {
            if let Some(result) =
                check_slot_match(leaf, s1, search_keylenx, key, needs_suffix_check)
            {
                return result;
            }
        }

        if ikey2 == target_ikey {
            if let Some(result) =
                check_slot_match(leaf, s2, search_keylenx, key, needs_suffix_check)
            {
                return result;
            }
        }

        i += 3;
    }

    // Handle remainder (0-2 elements)
    while i < size {
        let slot: usize = perm.get(i);
        let slot_ikey: u64 = leaf.ikey_relaxed(slot);

        if slot_ikey == target_ikey {
            if let Some(result) =
                check_slot_match(leaf, slot, search_keylenx, key, needs_suffix_check)
            {
                return result;
            }
        }

        i += 1;
    }

    LookupResult::NotFound
}

/// Optimized slot match check with suffix-check bypass.
///
/// When `needs_suffix_check` is false (inline keys ≤8 bytes), skips:
/// - Suffix comparison (no suffix exists)
/// - Layer pointer detection (inline keys can't be layer pointers)
///
/// # Prefetch Strategy
///
/// Prefetches the value/layer pointer target immediately after loading it.
/// This hides memory latency while suffix comparison runs.
#[inline(always)]
fn check_slot_match<P>(
    leaf: &LeafNode15<P>,
    slot: usize,
    search_keylenx: u8,
    key: &Key<'_>,
    needs_suffix_check: bool,
) -> Option<LookupResult>
where
    P: LeafPolicy,
{
    let slot_keylenx: u8 = leaf.keylenx(slot);

    // Empty slot = concurrent modification — skip.
    if leaf.is_value_empty(slot) {
        return None;
    }

    // Prefetch value target to hide memory latency during suffix check.
    // For Arc: prefetches the heap allocation behind the pointer.
    // For inline: no-op (value is already in the leaf's cache lines).
    leaf.prefetch_value(slot);

    if slot_keylenx == search_keylenx {
        // Potential exact match
        // OPTIM: Only check suffix if the search key has one
        if needs_suffix_check
            && slot_keylenx == KSUF_KEYLENX
            && !leaf.ksuf_equals(slot, key.suffix())
        {
            return None;
        }

        // Return slot index, not pointer
        return Some(LookupResult::ValueSlot(slot));
    }

    // OPTIM: Layer pointer check only relevant if search key has suffix
    if needs_suffix_check && slot_keylenx >= LAYER_KEYLENX {
        let layer_ptr: *mut u8 = leaf.load_layer_raw(slot);

        // Null check: concurrent gc_layer can null the value between is_value_empty
        // and load_layer_raw (TOCTOU). has_changed() misses in-progress modifications
        // (lock bits only), so null can slip past OCC validation. Skip slot and let
        // the caller's version check detect the change after the writer unlocks.
        if layer_ptr.is_null() {
            return None;
        }

        return Some(LookupResult::Layer(layer_ptr));
    }

    None
}

// ============================================================================
//  Helper Functions
// ============================================================================

impl<P, A> MassTreeGeneric<P, A>
where
    P: LeafPolicy,
    A: TreeAllocator<P>,
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
    fn handle_version_change(
        &self,
        leaf: &LeafNode15<P>,
        key: &Key<'_>,
        version: u32,
        guard: &LocalGuard<'_>,
    ) -> (*mut LeafNode15<P>, u32, bool) {
        let (advanced, new_version) = self.advance_to_key_generic(leaf, key, version, guard);

        if StdPtr::eq(advanced, leaf) {
            // Same leaf, new version - retry search
            (StdPtr::from_ref(leaf).cast_mut(), new_version, false)
        } else {
            // Different leaf - search there
            (StdPtr::from_ref(advanced).cast_mut(), new_version, true)
        }
    }

    #[expect(
        clippy::unused_self,
        reason = "method signature kept for API consistency"
    )]
    fn check_blink_chain(
        &self,
        leaf: &LeafNode15<P>,
        target_ikey: u64,
        guard: &LocalGuard<'_>,
    ) -> Option<*mut LeafNode15<P>> {
        let next_raw: *mut LeafNode15<P> = leaf.next_raw(guard);

        // If leaf is deleted, we gotta follow the next ptr (unmarked) to find our key.
        // C++ ref: masstree_struct.hh:704 - "while (likely(!v.deleted()) && ..."
        // When deleted, the next ptr is marked but still valid.
        if leaf.version().is_deleted() {
            let next_ptr: *mut LeafNode15<P> = Linker::unmark_ptr(next_raw);

            if !next_ptr.is_null() {
                return Some(next_ptr);
            }

            // Deleted leaf with no successor, key can not exist
            return None;
        }

        // Normal case: follow B-link if key >= next leaf's bound
        // only follow unmarked ptr's (marked = being unlinked)
        let next_ptr: *mut LeafNode15<P> = Linker::unmark_ptr(next_raw);

        if !next_ptr.is_null() && !Linker::is_marked(next_raw) {
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
    ///
    /// This check runs on every layer descent (hot path). Only finding a deleted
    /// sublayer is rare. The function is tiny (pointer cast + load), so inlining
    /// is always beneficial.
    #[inline(always)]
    #[expect(clippy::unused_self, reason = "API consistency with other methods")]
    fn check_sublayer_valid(&self, layer_ptr: *mut u8) -> bool {
        // Defense-in-depth: concurrent gc_layer can produce null layer pointers
        // via TOCTOU race in check_slot_match (is_value_empty vs load_layer_raw).
        // Callers should check for null before calling, but guard here too.
        if layer_ptr.is_null() {
            return false;
        }

        // SAFETY: ptr is non-null (checked above) and protected by guard.
        #[expect(clippy::cast_ptr_alignment, reason = "Checked")]
        let sublayer_version: &NodeVersion = unsafe { &*layer_ptr.cast::<NodeVersion>() };

        !sublayer_version.is_deleted()
    }

    /// Descend through a chain of trivial twigs (single-entry layer leaves).
    ///
    /// A "trivial twig" is a leaf node with exactly one entry that is a layer pointer,
    /// still marked as a layer root. These form when keys share multiple 8-byte prefix chunks.
    ///
    /// This method loops through consecutive twigs without restarting `'layer_loop`,
    /// avoiding repeated overhead. It maintains all OCC safety invariants and exits
    /// to the normal path on any concurrent modification or unexpected structure.
    ///
    /// # Preconditions
    ///
    /// - `initial_ptr` is non-null and points to a valid node (caller must verify)
    /// - The node at `initial_ptr` has already passed `is_deleted()` check
    ///
    /// # Safety
    ///
    /// This function performs unsafe pointer casts. The caller must ensure `initial_ptr`
    /// is a valid pointer to a node protected by the guard's epoch.
    #[inline]
    #[expect(
        clippy::too_many_lines,
        reason = "complex twig-chain traversal with safety checks"
    )]
    fn descend_twig_chain(
        &self,
        initial_ptr: *mut u8,
        key: &mut Key<'_>,
        guard: &LocalGuard<'_>,
    ) -> TwigDescentResult<LeafNode15<P>> {
        let mut ptr: *mut u8 = initial_ptr;

        loop {
            key.shift();

            // ------------------------------------------------------------------
            // STEP 1: Read version and validate node type BEFORE casting to leaf
            // ------------------------------------------------------------------
            // Layer pointers can point to internodes after splits. We must check
            // is_leaf() before assuming we can treat this as a leaf node.

            // SAFETY: ptr is non-null (checked by caller or previous iteration)
            #[expect(clippy::cast_ptr_alignment, reason = "NodeVersion is first field")]
            let node_version: &NodeVersion = unsafe { &*ptr.cast::<NodeVersion>() };

            // Check if sublayer was deleted (return None, matching line 698-701 behavior)
            if node_version.is_deleted() {
                return TwigDescentResult::ReturnNone;
            }

            // Check if node is still a leaf (layer roots can become internodes after splits)
            if !node_version.is_leaf() {
                // Not a leaf - must use normal traversal via reach_leaf_concurrent_generic
                return TwigDescentResult::RestartLayerLoop {
                    layer_root: ptr.cast_const(),
                    in_sublayer: true,
                };
            }

            // Check if node is still a layer root (skipping reach_leaf is only safe for roots)
            // Twigs are allocated with ROOT_BIT set (try_alloc_leaf(false, true))
            if !node_version.is_root() {
                // No longer a root - must use normal traversal
                return TwigDescentResult::RestartLayerLoop {
                    layer_root: ptr.cast_const(),
                    in_sublayer: true,
                };
            }

            // ------------------------------------------------------------------
            // STEP 2: Now safe to cast to leaf and read leaf-specific fields
            // ------------------------------------------------------------------

            // SAFETY: Verified is_leaf() above, ptr is valid sublayer pointer
            let twig: &LeafNode15<P> = unsafe { &*ptr.cast::<LeafNode15<P>>() };

            // Check for deleted_layer (sublayer was GC'd - must restart from main root)
            if twig.deleted_layer() {
                key.unshift_all();
                let root = self.load_root_ptr_generic(guard);

                return TwigDescentResult::RestartLayerLoop {
                    layer_root: root,
                    in_sublayer: false,
                };
            }

            // ------------------------------------------------------------------
            // STEP 3: Version stabilization
            // ------------------------------------------------------------------
            // Use stable() which spins until dirty bits clear. This is acceptable
            // in this niche path because:
            // 1. Twigs are rarely contended (single-entry nodes with no values)
            // 2. If contention is high, we'll exit fast path via has_changed() anyway
            // 3. Avoiding try_stable() simplifies the code (no B-link fallback needed)
            let twig_version: u32 = twig.version().stable();

            // Prefetch for next iteration (hides memory latency)
            twig.prefetch_for_search();

            // ------------------------------------------------------------------
            // STEP 4: "Too-right" detection (concurrent split moved our key)
            // ------------------------------------------------------------------
            let target_ikey: u64 = key.ikey();

            if !twig.prev(guard).is_null() && target_ikey < twig.ikey_bound() {
                // We're on a leaf that's to the right of where key should be.
                // Exit to normal leaf_loop which will handle this (either via
                // the early too-right check or NotFound → too-right recovery).
                return TwigDescentResult::ContinueLeafLoop {
                    layer_root: ptr.cast_const(),
                    leaf_ptr: ptr.cast::<LeafNode15<P>>(),
                };
            }

            // ------------------------------------------------------------------
            // STEP 5: Check if this is a trivial twig we can fast-path through
            // ------------------------------------------------------------------
            let twig_perm = twig.permutation();

            // Exit fast path if not single-entry
            if twig_perm.size() != 1 {
                return TwigDescentResult::ContinueLeafLoop {
                    layer_root: ptr.cast_const(),
                    leaf_ptr: ptr.cast::<LeafNode15<P>>(),
                };
            }

            let slot: usize = twig_perm.get(0);
            let twig_ikey: u64 = twig.ikey_relaxed(slot);

            // Verify ikey matches (detects concurrent modification)
            if twig_ikey != target_ikey {
                return TwigDescentResult::ContinueLeafLoop {
                    layer_root: ptr.cast_const(),
                    leaf_ptr: ptr.cast::<LeafNode15<P>>(),
                };
            }

            let twig_keylenx: u8 = twig.keylenx(slot);

            // Exit fast path if not a layer pointer (it's a value)
            if twig_keylenx < LAYER_KEYLENX {
                return TwigDescentResult::ContinueLeafLoop {
                    layer_root: ptr.cast_const(),
                    leaf_ptr: ptr.cast::<LeafNode15<P>>(),
                };
            }

            // ------------------------------------------------------------------
            // STEP 6: Extract next pointer and validate
            // ------------------------------------------------------------------
            // Empty slot means slot is being modified (matches check_slot_match)
            if twig.is_value_empty(slot) {
                return TwigDescentResult::ContinueLeafLoop {
                    layer_root: ptr.cast_const(),
                    leaf_ptr: ptr.cast::<LeafNode15<P>>(),
                };
            }
            let next_ptr: *mut u8 = twig.load_layer_raw(slot);

            // Null check: concurrent gc_layer can null the value between
            // is_value_empty and load_layer_raw (TOCTOU race).
            if next_ptr.is_null() {
                return TwigDescentResult::ContinueLeafLoop {
                    layer_root: ptr.cast_const(),
                    leaf_ptr: ptr.cast::<LeafNode15<P>>(),
                };
            }

            // ------------------------------------------------------------------
            // STEP 7: Version validation before following pointer
            // ------------------------------------------------------------------
            if twig.version().has_changed(twig_version) {
                // Concurrent modification detected - fall back to safe path
                return TwigDescentResult::ContinueLeafLoop {
                    layer_root: ptr.cast_const(),
                    leaf_ptr: ptr.cast::<LeafNode15<P>>(),
                };
            }

            // ------------------------------------------------------------------
            // STEP 8: Check if search key has more layers to descend
            // ------------------------------------------------------------------
            // CRITICAL: This mirrors the check in search_leaf_multi_layer/check_slot_match
            // (line 210-211) that only returns LookupResult::Layer when needs_suffix_check
            // is true. Without this, we'd call key.shift() on a key with no suffix,
            // triggering a debug assertion panic (src/key.rs:238).
            //
            // Scenario: search key is a strict prefix of longer keys that created
            // a deeper twig chain. Normal lookup would stop here (search returns
            // NotFound), but we must also stop the fast path.
            if !key.has_suffix() {
                // Search key has no more layers - it's a prefix of stored keys.
                // Exit to normal search which will return NotFound.
                return TwigDescentResult::ContinueLeafLoop {
                    layer_root: ptr.cast_const(),
                    leaf_ptr: ptr.cast::<LeafNode15<P>>(),
                };
            }

            // ------------------------------------------------------------------
            // STEP 9: Follow the layer pointer to next node
            // ------------------------------------------------------------------
            ptr = next_ptr;

            // Loop continues: check if next node is also a trivial twig
        }
    }
}

// ============================================================================
//  Public API
// ============================================================================

impl<P, A> MassTreeGeneric<P, A>
where
    P: LeafPolicy,
    A: TreeAllocator<P>,
{
    /// Get a value by key.
    ///
    /// Creates a guard internally. For bulk operations, prefer
    /// [`get_with_guard`](Self::get_with_guard) to amortize guard creation cost.
    ///
    /// Returns an owned clone of the value. The internal EBR guard is
    /// released before returning, so the value is cloned to ensure
    /// independent ownership.
    ///
    /// # Returns
    ///
    /// * `Some(V)` - If the key was found (owned clone)
    /// * `None` - If the key was not found
    #[must_use]
    #[inline]
    pub fn get(&self, key: &[u8]) -> Option<P::Value>
    where
        P::Value: Clone,
    {
        let guard = self.guard();
        self.get_with_guard(key, &guard)
            .map(|output| P::clone_value_from_output(&output))
    }

    /// Check if a key exists in the tree.
    ///
    /// Creates a guard internally. For bulk operations, prefer
    /// [`contains_key_with_guard`](Self::contains_key_with_guard) to amortize
    /// guard creation cost.
    ///
    /// # Example
    ///
    /// ```ignore
    /// let tree = MassTree::<u64>::new();
    /// tree.insert(b"hello", 42);
    ///
    /// assert!(tree.contains_key(b"hello"));
    /// assert!(!tree.contains_key(b"world"));
    /// ```
    #[must_use]
    #[inline]
    pub fn contains_key(&self, key: &[u8]) -> bool {
        let guard = self.guard();
        self.contains_key_with_guard(key, &guard)
    }

    /// Check if a key exists using an existing guard.
    ///
    /// Use this when performing multiple lookups under the same guard
    /// to amortize guard creation overhead.
    ///
    /// # Example
    ///
    /// ```ignore
    /// let tree = MassTree::<u64>::new();
    /// let guard = tree.guard();
    ///
    /// tree.insert_with_guard(b"a", 1, &guard).unwrap();
    /// tree.insert_with_guard(b"b", 2, &guard).unwrap();
    ///
    /// assert!(tree.contains_key_with_guard(b"a", &guard));
    /// assert!(tree.contains_key_with_guard(b"b", &guard));
    /// assert!(!tree.contains_key_with_guard(b"c", &guard));
    /// ```
    #[must_use]
    #[inline]
    pub fn contains_key_with_guard(&self, key: &[u8], guard: &LocalGuard<'_>) -> bool {
        self.get_with_guard(key, guard).is_some()
    }

    /// Unified get implementation.
    ///
    /// Both `get_with_guard` and `get_ref` delegate to this function.
    /// The `extract` closure handles the difference in return type.
    ///
    /// # Type Parameters
    ///
    /// * `R` - Return type (`P::Output` or `&'g P::Value`)
    /// * `F` - Closure that extracts the value from a raw pointer
    #[inline(always)]
    fn get_impl<R, F>(&self, key: &mut Key<'_>, guard: &LocalGuard<'_>, extract: F) -> Option<R>
    where
        F: Fn(*mut u8) -> R,
    {
        // Detect single-layer mode: key <= 8 bytes means no suffix, no layer descent
        // This enables a completely inline fast path without enum overhead
        if !key.has_suffix() {
            return self.get_impl_single_layer(key, guard, extract);
        }

        // Multi-layer path for keys > 8 bytes
        self.get_impl_multi_layer(key, guard, extract)
    }

    /// Handle landing on a deleted leaf during point read.
    ///
    /// Called when we detect `is_deleted()` at the start of leaf processing.
    /// Follows B-link chain to find the correct successor leaf.
    ///
    /// Returns the next valid leaf pointer, or restarts from root if no successor.
    #[cold]
    #[inline(never)]
    fn handle_deleted_leaf(
        &self,
        leaf: &LeafNode15<P>,
        layer_root: *const u8,
        key: &Key<'_>,
        is_sublayer: bool,
        guard: &LocalGuard<'_>,
    ) -> *mut LeafNode15<P> {
        // Try to follow B-link to successor
        let next_raw: *mut LeafNode15<P> = leaf.next_raw(guard);
        let next_ptr: *mut LeafNode15<P> = Linker::unmark_ptr(next_raw);

        if !next_ptr.is_null() {
            // Follow to successor leaf
            return next_ptr;
        }

        // No successor mean we must restart from root
        // This can happen if the deleted leaf was the last in its layer
        self.reach_leaf_concurrent_generic(layer_root, key, is_sublayer, guard)
    }

    #[inline(always)]
    fn get_impl_single_layer<R, F>(
        &self,
        key: &Key<'_>,
        guard: &LocalGuard<'_>,
        extract: F,
    ) -> Option<R>
    where
        F: Fn(*mut u8) -> R,
    {
        let mut layer_root: *const u8 = self.load_root_ptr_generic(guard);
        let target_ikey: u64 = key.ikey();

        #[expect(clippy::cast_possible_truncation, reason = "current_len() <= 8")]
        let search_keylenx: u8 = key.current_len() as u8;

        // Traverse to leaf
        let mut leaf_ptr: *mut LeafNode15<P> =
            self.reach_leaf_concurrent_generic(layer_root, key, false, guard);

        'leaf_loop: loop {
            // SAFETY: leaf_ptr protected by guard
            let leaf: &LeafNode15<P> = unsafe { &*leaf_ptr };

            // If we landed on a deleted leaf, follow B-link to successor.
            // This can happen during concurrent coalesce operations.
            if leaf.version().is_deleted() {
                leaf_ptr = self.handle_deleted_leaf(leaf, layer_root, key, false, guard);

                continue 'leaf_loop;
            }

            // OPTIMIZATION: Use try_stable() to avoid spinning on locked leaf.
            // If leaf is locked, opportunistically check B-link chain - under
            // high contention, our key may have moved to a sibling leaf.
            let mut version: u32 = if let Some(v) = leaf.version().try_stable() {
                // Prefetch AFTER successful try_stable to avoid cache pollution
                leaf.prefetch_for_search();
                v
            } else {
                // Leaf is locked - check if key might be in sibling
                if let Some(next_ptr) = self.check_blink_chain(leaf, target_ikey, guard) {
                    leaf_ptr = next_ptr;

                    continue 'leaf_loop;
                }

                // No B-link escape route, prefetch while waiting
                leaf.prefetch_for_search();
                leaf.version().stable()
            };

            // EARLY too-right check: detect if we descended to a leaf that's
            // to the right of where the key should be. This can happen during
            // concurrent splits when internode routing is momentarily inconsistent.
            // Check BEFORE searching to avoid wasting cycles on the wrong leaf.
            if !leaf.prev(guard).is_null() && target_ikey < leaf.ikey_bound() {
                // Reload root to get latest pointer after concurrent modifications
                layer_root = self.load_root_ptr_generic(guard);
                leaf_ptr = self.reach_leaf_concurrent_generic(layer_root, key, false, guard);

                continue 'leaf_loop;
            }

            'search_loop: loop {
                let perm = leaf.permutation();
                let size: usize = perm.size();
                let mut found_ptr: *mut u8 = StdPtr::null_mut();

                // Simple linear search - let LLVM decide optimal unrolling.
                // LLVM SHOULD auto unroll with #[inline(always)], and speculative batch
                // loads waste work on early matches.
                for i in 0..size {
                    let slot: usize = perm.get(i);

                    // Use Relaxed ordering - permutation() Acquire already synchronizes
                    if (leaf.ikey_relaxed(slot) == target_ikey)
                        && (leaf.keylenx(slot) == search_keylenx)
                        && !leaf.is_value_empty(slot)
                    {
                        // Prefetch value target while version validation runs.
                        leaf.prefetch_value(slot);
                        found_ptr = leaf.load_value_raw(slot);

                        break;
                    }
                }

                // Version validation after all reads (common case: unchanged)
                if unlikely(leaf.version().has_changed(version)) {
                    // Only do full B-link handling if split occurred
                    // For update-only, simple retry is faster
                    if unlikely(leaf.version().has_split_no_compiler_fence(version)) {
                        let (advanced, new_version) =
                            self.advance_to_key_generic(leaf, key, version, guard);

                        if !StdPtr::eq(advanced, leaf) {
                            leaf_ptr = StdPtr::from_ref(advanced).cast_mut();

                            continue 'leaf_loop;
                        }

                        version = new_version;
                    } else {
                        // Update only - re-stabilize without B-link check
                        version = leaf.version().stable();
                    }

                    continue 'search_loop;
                }

                if !found_ptr.is_null() {
                    return Some(extract(found_ptr));
                }

                // Not found, check dirty or B-link (also handles deleted via check_blink_chain)
                if unlikely(leaf.version().is_dirty()) {
                    version = leaf.version().stable();

                    continue 'search_loop;
                }

                // Check lower bound for non-leftmost leaves ("too-right" detection).
                // If key < ikey_bound and prev != null, we descended to a leaf that's
                // to the right of where the key should be. Recovery requires restart
                // from layer root (can't safely walk left in a B-link tree).
                // NOTE: This is a fallback; the early check above should usually catch this.
                if unlikely(!leaf.prev(guard).is_null() && target_ikey < leaf.ikey_bound()) {
                    // Reload root to get latest pointer after concurrent modifications
                    layer_root = self.load_root_ptr_generic(guard);
                    leaf_ptr = self.reach_leaf_concurrent_generic(layer_root, key, false, guard);

                    continue 'leaf_loop;
                }

                if let Some(next_ptr) = self.check_blink_chain(leaf, target_ikey, guard) {
                    leaf_ptr = next_ptr;

                    continue 'leaf_loop;
                }

                return None;
            }
        }
    }

    /// Multi-layer path for keys > 8 bytes.
    ///
    /// Handles layer descent, suffix matching, and complex key structures.
    #[expect(clippy::too_many_lines, reason = "complex multi-layer traversal logic")]
    #[inline]
    fn get_impl_multi_layer<R, F>(
        &self,
        key: &mut Key<'_>,
        guard: &LocalGuard<'_>,
        extract: F,
    ) -> Option<R>
    where
        F: Fn(*mut u8) -> R,
    {
        let mut layer_root: *const u8 = self.load_root_ptr_generic(guard);
        let mut in_sublayer: bool = false;

        'layer_loop: loop {
            layer_root = self.maybe_parent_generic(layer_root);

            let mut leaf_ptr: *mut LeafNode15<P> =
                self.reach_leaf_concurrent_generic(layer_root, key, in_sublayer, guard);

            'leaf_loop: loop {
                // OPTIM: Compute ikey once per leaf iteration.
                // key.shift() mutates on layer descent, so this must be per-iteration.
                let target_ikey: u64 = key.ikey();

                let leaf: &LeafNode15<P> = unsafe { &*leaf_ptr };

                // If we landed on a deleted leaf, follow B-link to successor.
                // This can happen during concurrent coalesce operations.
                if leaf.version().is_deleted() {
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

                // OPTIMIZATION: Use try_stable() to avoid spinning on locked leaf.
                let mut version: u32 = if let Some(v) = leaf.version().try_stable() {
                    // Prefetch AFTER successful try_stable to avoid cache pollution
                    leaf.prefetch_for_search();
                    v
                } else {
                    if let Some(next_ptr) = self.check_blink_chain(leaf, target_ikey, guard) {
                        leaf_ptr = next_ptr;

                        continue 'leaf_loop;
                    }

                    // Prefetch while waiting
                    leaf.prefetch_for_search();
                    leaf.version().stable()
                };

                // EARLY too-right check: detect if we descended to a leaf that's
                // to the right of where the key should be. Check BEFORE searching.
                if !leaf.prev(guard).is_null() && target_ikey < leaf.ikey_bound() {
                    // Reload root to get latest pointer after concurrent modifications
                    layer_root = self.load_root_ptr_generic(guard);
                    leaf_ptr =
                        self.reach_leaf_concurrent_generic(layer_root, key, in_sublayer, guard);

                    continue 'leaf_loop;
                }

                'search_loop: loop {
                    // Check for gc'd sublayer
                    if leaf.deleted_layer() {
                        key.unshift_all();
                        layer_root = self.load_root_ptr_generic(guard);
                        in_sublayer = false;

                        continue 'layer_loop;
                    }

                    // target_ikey already computed at start of 'leaf_loop
                    let result: LookupResult = search_leaf_multi_layer::<P>(leaf, key);

                    if unlikely(leaf.version().has_changed(version)) {
                        // Only do full B-link handling if split occurred
                        if unlikely(leaf.version().has_split_no_compiler_fence(version)) {
                            let (new_ptr, new_version, changed_leaf) =
                                self.handle_version_change(leaf, key, version, guard);

                            if changed_leaf {
                                leaf_ptr = new_ptr;

                                continue 'leaf_loop;
                            }

                            version = new_version;
                        } else {
                            // Update only - re-stabilize without B-link check
                            version = leaf.version().stable();
                        }

                        continue 'search_loop;
                    }

                    match result {
                        LookupResult::ValueSlot(slot) => {
                            let ptr: *mut u8 = leaf.load_value_raw(slot);
                            return Some(extract(ptr));
                        }

                        LookupResult::Layer(ptr) => {
                            // Defense-in-depth null check: concurrent gc_layer can null
                            // the value between is_value_empty and load_layer_raw in
                            // check_slot_match (TOCTOU race). check_slot_match has its
                            // own null guard, but we keep this as a safety net.
                            if ptr.is_null() {
                                continue 'search_loop;
                            }

                            // SAFETY: ptr is non-null, cast to read version
                            #[expect(
                                clippy::cast_ptr_alignment,
                                reason = "NodeVersion is first field"
                            )]
                            let sublayer_version: &NodeVersion =
                                unsafe { &*ptr.cast::<NodeVersion>() };

                            if sublayer_version.is_deleted() {
                                return None;
                            }

                            // === TWIG-CHAIN FAST PATH ===
                            // Collapse consecutive single-entry layer pointers without
                            // full outer-loop restart. Falls back to normal path on any
                            // concurrent modification or non-trivial structure.
                            match self.descend_twig_chain(ptr, key, guard) {
                                TwigDescentResult::ContinueLeafLoop {
                                    layer_root: new_root,
                                    leaf_ptr: new_leaf,
                                } => {
                                    layer_root = new_root;
                                    leaf_ptr = new_leaf;
                                    in_sublayer = true;

                                    continue 'leaf_loop;
                                }

                                TwigDescentResult::RestartLayerLoop {
                                    layer_root: new_root,
                                    in_sublayer: new_in_sublayer,
                                } => {
                                    layer_root = new_root;
                                    in_sublayer = new_in_sublayer;

                                    continue 'layer_loop;
                                }

                                TwigDescentResult::ReturnNone => {
                                    return None;
                                }
                            }
                        }

                        LookupResult::NotFound => {
                            if leaf.version().is_dirty() {
                                version = leaf.version().stable();

                                continue 'search_loop;
                            }

                            // Check lower bound for non-leftmost leaves ("too-right" detection).
                            // If key < ikey_bound and prev != null, we descended to a leaf that's
                            // to the right of where the key should be. Recovery requires restart
                            // from layer root (can't safely walk left in a B-link tree).
                            // NOTE: This is a fallback; the early check above should usually catch this.
                            if !leaf.prev(guard).is_null() && target_ikey < leaf.ikey_bound() {
                                // Reload root to get latest pointer after concurrent modifications
                                layer_root = self.load_root_ptr_generic(guard);
                                leaf_ptr = self.reach_leaf_concurrent_generic(
                                    layer_root,
                                    key,
                                    in_sublayer,
                                    guard,
                                );

                                continue 'leaf_loop;
                            }

                            // check_blink_chain now also handles deleted leaves
                            if let Some(next_ptr) = self.check_blink_chain(leaf, target_ikey, guard)
                            {
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

// ============================================================================
//  Reference-Returning API (Pointer-Backed Storage Only)
// ============================================================================

impl<P, A> MassTreeGeneric<P, A>
where
    P: LeafPolicy + RefLeafPolicy,
    A: TreeAllocator<P>,
{
    /// Get a borrowed reference to a value by key.
    ///
    /// This is significantly faster than [`Self::get_with_guard`] for read-heavy workloads
    /// because it avoids atomic reference count operations (Arc clone/drop).
    ///
    /// # Note
    ///
    /// This method is only available for pointer-backed storage modes (Currently only MassTree15,
    /// which uses Box)
    ///
    /// It is not available for true-inline storage (`MassTree15Inline`) because
    /// values are stored as atomic bits, not at stable addresses.
    ///
    /// For true-inline trees, use `get`/`get_with_guard` instead (returns by copy).
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
    pub fn get_ref<'g>(&self, key: &[u8], guard: &'g LocalGuard<'_>) -> Option<&'g P::Value> {
        let mut search_key: Key<'_> = Key::new(key);

        self.get_impl(&mut search_key, guard, |ptr: *mut u8| {
            // SAFETY: version validated, guard protects from deallocation
            unsafe { &*(ptr.cast::<P::Value>()) }
        })
    }
}
