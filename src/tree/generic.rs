#[cfg(feature = "tracing")]
use std::time::Instant;
use std::{
    cmp::Ordering,
    marker::PhantomData,
    ptr as StdPtr,
    sync::atomic::{AtomicPtr, AtomicUsize, Ordering as AtomicOrdering},
};

use parking_lot::{Condvar, Mutex};
use seize::{Collector, Guard, LocalGuard};

use crate::{
    MassTreeGeneric, NodeAllocatorGeneric, TreeInternode, TreePermutation,
    key::Key,
    leaf_trait::LayerCapableLeaf,
    leaf24::{KSUF_KEYLENX, LAYER_KEYLENX},
    nodeversion::NodeVersion,
    slot::ValueSlot,
    tree::split::Propagation,
    tree::{CasInsertResultGeneric, InsertError, InsertSearchResultGeneric},
};

/// Sentinel for the CLAIMING state in the Option A (Safe) CAS insert protocol.
///
/// When a slot's value pointer equals this sentinel, the slot is reserved by an in-progress
/// CAS insert attempt. The inserter has exclusive right to write key metadata, but hasn't
/// yet installed the real value pointer.
///
/// State machine: `NULL -> CLAIMING -> arc_ptr -> (permutation publish) -> visible`
///
/// This sentinel:
/// - Must be non-null (to distinguish from "free")
/// - Must never be dereferenced
/// - Must be stable for the program lifetime
/// - Must be easy to recognize (`== claiming_ptr()`)
static CLAIMING_SENTINEL: u8 = 0;

/// Returns the CLAIMING sentinel pointer (provenance-sound).
#[inline(always)]
fn claiming_ptr() -> *mut u8 {
    StdPtr::from_ref(&CLAIMING_SENTINEL).cast_mut()
}

/// Returns true if `ptr` is the CLAIMING sentinel.
#[inline(always)]
#[expect(dead_code, reason = "CAS path disabled")]
fn is_claiming_ptr(ptr: *mut u8) -> bool {
    StdPtr::eq(ptr, claiming_ptr())
}

impl<S, L, A> MassTreeGeneric<S, L, A>
where
    S: ValueSlot,
    S::Value: Send + Sync + 'static,
    S::Output: Send + Sync,
    L: LayerCapableLeaf<S>,
    A: NodeAllocatorGeneric<S, L>,
{
    #[inline]
    #[expect(dead_code, reason = "CAS path disabled")]
    fn cas_insert_enabled() -> bool {
        use std::sync::OnceLock;

        // CAS insert is currently correctness-sensitive under high contention.
        // Default to disabled unless explicitly enabled for benchmarking/experiments.
        //
        // - Set `MASSTREE_ENABLE_CAS=1` to enable the CAS fast path.
        // - Set `MASSTREE_ENABLE_CAS=0` or unset to disable.
        static ENABLE_CAS: OnceLock<bool> = OnceLock::new();
        *ENABLE_CAS.get_or_init(|| {
            std::env::var("MASSTREE_ENABLE_CAS")
                .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
                .unwrap_or(false)
        })
    }

    /// Pick a free physical slot from the permutation's free region, skipping reserved slots.
    ///
    /// The Option A (Safe) CAS insert protocol uses a 3-phase state machine:
    /// `NULL -> CLAIMING -> arc_ptr -> (permutation publish) -> visible`
    ///
    /// A CAS inserter can temporarily set `leaf_values[slot]` to `CLAIMING` or `arc_ptr` while
    /// the slot is still in the permutation's free region (not yet published by permutation CAS).
    ///
    /// The locked insert path must treat such slots as **reserved** and avoid reusing them,
    /// otherwise it can overwrite a CAS-reserved slot and later publish it, creating an
    /// inconsistent (ikey/keylenx/ptr) tuple visible to readers.
    ///
    /// Returns `(slot, back_offset)` where `slot == perm.back_at_offset(back_offset)`.
    #[inline(always)]
    #[expect(dead_code, reason = "CAS path disabled")]
    fn pick_free_slot_avoiding_reserved(
        leaf: &L,
        perm: &L::Perm,
        ikey: u64,
    ) -> Option<(usize, usize)> {
        use crate::leaf_trait::TreePermutation;

        let size: usize = perm.size();
        debug_assert!(
            size < L::WIDTH,
            "pick_free_slot_avoiding_reserved: no free slots"
        );

        let free_count: usize = L::WIDTH - size;
        for offset in 0..free_count {
            let slot: usize = perm.back_at_offset(offset);

            // Slot-0 / ikey_bound invariant: skip slot 0 if it can't be reused.
            if slot == 0 && !leaf.can_reuse_slot0(ikey) {
                continue;
            }

            // Option A (Safe): treat non-null in free region as reserved.
            // This includes both CLAIMING (reservation sentinel) and arc_ptr (value installed
            // but not yet published via permutation CAS).
            if !leaf.leaf_value_ptr(slot).is_null() {
                continue;
            }

            return Some((slot, offset));
        }

        None
    }

    /// Create a new empty `MassTreeGeneric` with the given allocator.
    ///
    /// The tree starts with a single empty leaf as root.
    #[must_use]
    pub fn with_allocator(allocator: A) -> Self {
        // Create root leaf and register with allocator.
        let root_leaf: Box<L> = L::new_root_boxed();
        let root_ptr: *mut L = allocator.alloc_leaf(root_leaf);

        Self {
            collector: Collector::new(),
            allocator,
            root_ptr: AtomicPtr::new(root_ptr.cast::<u8>()),
            count: AtomicUsize::new(0),
            parent_set_condvar: Condvar::new(),
            parent_set_mutex: Mutex::new(()),
            _marker: PhantomData,
        }
    }

    /// Enter a protected region and return a guard.
    ///
    /// The guard protects any pointers loaded during its lifetime from being
    /// reclaimed. Call this before reading tree nodes or values.
    #[must_use]
    #[inline(always)]
    pub fn guard(&self) -> LocalGuard<'_> {
        self.collector.enter()
    }

    /// Get the number of keys in the tree.
    ///
    /// This is O(1) as we track the count incrementally.
    #[must_use]
    #[inline(always)]
    pub fn len(&self) -> usize {
        self.count.load(AtomicOrdering::Relaxed)
    }

    /// Check if the tree is empty.
    #[must_use]
    #[inline(always)]
    pub fn is_empty(&self) -> bool {
        if self.root_is_leaf_generic() {
            // SAFETY: root_is_leaf_generic confirmed this is a leaf
            let leaf_ptr: *const L = self.root_ptr.load(AtomicOrdering::Acquire).cast();
            unsafe { (*leaf_ptr).is_empty() }
        } else {
            // Internode implies at least one key
            false
        }
    }

    // ========================================================================
    //  Internal Helpers
    // ========================================================================

    /// Notify all threads waiting for a parent pointer to be set.
    ///
    /// Called after setting a node's parent pointer during split propagation.
    /// This wakes up any threads waiting in the condvar.
    #[inline(always)]
    pub(crate) fn notify_parent_set(&self) {
        self.parent_set_condvar.notify_all();
    }

    /// Wait for a parent pointer to be set, with timeout.
    ///
    /// Returns `true` if notified (should recheck condition), `false` on timeout.
    #[inline(always)]
    fn wait_for_parent_set(&self, timeout: std::time::Duration) -> bool {
        !self
            .parent_set_condvar
            .wait_for(&mut self.parent_set_mutex.lock(), timeout)
            .timed_out()
    }

    /// Load the root pointer atomically.
    #[inline(always)]
    pub(crate) fn load_root_ptr_generic(&self, _guard: &LocalGuard<'_>) -> *const u8 {
        self.root_ptr.load(AtomicOrdering::Acquire)
    }

    /// Compare-and-swap the root pointer atomically.
    #[inline(always)]
    pub(crate) fn cas_root_ptr_generic(
        &self,
        expected: *mut u8,
        new: *mut u8,
    ) -> Result<(), *mut u8> {
        self.root_ptr
            .compare_exchange(
                expected,
                new,
                AtomicOrdering::AcqRel,
                AtomicOrdering::Acquire,
            )
            .map(|_| ())
    }

    /// Check if the current root is a leaf node.
    ///
    /// # Safety
    /// Reads the version field through a raw pointer. The `root_ptr` must
    /// point to a valid node (guaranteed by construction).
    #[inline(always)]
    #[expect(
        clippy::cast_ptr_alignment,
        reason = "root_ptr points to L or L::Internode, both have NodeVersion \
                  as first field with proper alignment"
    )]
    fn root_is_leaf_generic(&self) -> bool {
        let root: *const u8 = self.root_ptr.load(AtomicOrdering::Acquire);

        // SAFETY: `root_ptr` always points to a valid node.
        // `NodeVersion` is the first field of both leaf and internode types.
        let version_ptr: *const NodeVersion = root.cast::<NodeVersion>();

        unsafe { (*version_ptr).is_leaf() }
    }

    /// Get a mutable reference to the allocator.
    #[inline(always)]
    pub(crate) const fn allocator_mut(&mut self) -> &mut A {
        &mut self.allocator
    }

    /// Get an immutable reference to the allocator.
    #[inline(always)]
    pub(crate) const fn allocator(&self) -> &A {
        &self.allocator
    }

    /// Get a reference to the collector.
    #[inline(always)]
    pub(crate) const fn collector(&self) -> &Collector {
        &self.collector
    }

    /// Increment the entry count.
    #[inline(always)]
    pub(crate) fn inc_count(&self) {
        self.count.fetch_add(1, AtomicOrdering::Relaxed);
    }

    // ========================================================================
    //  Generic Tree Traversal
    // ========================================================================

    /// Reach the leaf node that should contain the given key.
    ///
    /// Traverses from root through internodes to find the target leaf.
    ///
    /// # Arguments
    ///
    /// * `key` - The key to search for
    ///
    /// # Returns
    ///
    /// Reference to the leaf node that contains or should contain the key.
    #[inline(always)]
    pub(crate) fn reach_leaf_generic(&self, key: &Key<'_>) -> &L {
        let root: *const u8 = self.root_ptr.load(AtomicOrdering::Acquire);

        // SAFETY: root_ptr always points to a valid node.
        // NodeVersion is the first field of both L and L::Internode.
        #[expect(
            clippy::cast_ptr_alignment,
            reason = "root points to L or L::Internode, both properly aligned"
        )]
        let version_ptr: *const NodeVersion = root.cast::<NodeVersion>();
        let is_leaf: bool = unsafe { (*version_ptr).is_leaf() };

        if is_leaf {
            // SAFETY: is_leaf() confirmed this is a leaf node
            unsafe { &*(root.cast::<L>()) }
        } else {
            // SAFETY: !is_leaf() confirmed this is an internode
            let internode: &L::Internode = unsafe { &*(root.cast::<L::Internode>()) };
            self.reach_leaf_via_internode_generic(internode, key)
        }
    }

    /// Traverse from an internode down to the target leaf.
    ///
    /// Uses generic internode search to find the correct child at each level.
    #[expect(
        clippy::unused_self,
        reason = "Method signature matches reach_leaf pattern"
    )]
    fn reach_leaf_via_internode_generic(&self, mut inode: &L::Internode, key: &Key<'_>) -> &L {
        use crate::ksearch::upper_bound_internode_generic;
        use crate::prefetch::prefetch_read;

        let target_ikey: u64 = key.ikey();

        loop {
            // Find child index using generic search
            let child_idx: usize =
                upper_bound_internode_generic::<S, L::Internode>(target_ikey, inode);
            let child_ptr: *mut u8 = inode.child(child_idx);

            // Prefetch child node
            prefetch_read(child_ptr);

            // Check child type via NodeVersion
            // SAFETY: All children have NodeVersion as first field, properly aligned
            #[expect(
                clippy::cast_ptr_alignment,
                reason = "child_ptr points to L or L::Internode, both properly aligned"
            )]
            let child_version: &NodeVersion = unsafe { &*(child_ptr.cast::<NodeVersion>()) };

            if child_version.is_leaf() {
                // SAFETY: is_leaf() confirms this is a leaf
                return unsafe { &*(child_ptr.cast::<L>()) };
            }

            // Descend to child internode
            // SAFETY: !is_leaf() confirms InternodeNode
            inode = unsafe { &*(child_ptr.cast::<L::Internode>()) };
        }
    }

    /// Reach the leaf node that should contain the given key (mutable).
    #[inline(always)]
    #[expect(
        clippy::needless_pass_by_ref_mut,
        reason = "Returns &mut L which requires &mut self for lifetime"
    )]
    pub(crate) fn reach_leaf_mut_generic(&mut self, key: &Key<'_>) -> &mut L {
        use crate::ksearch::upper_bound_internode_generic;
        use crate::prefetch::prefetch_read;

        let root: *mut u8 = self.root_ptr.load(AtomicOrdering::Acquire);

        // SAFETY: root_ptr always points to a valid node.
        #[expect(
            clippy::cast_ptr_alignment,
            reason = "root points to L or L::Internode, both properly aligned"
        )]
        let version_ptr: *const NodeVersion = root.cast::<NodeVersion>();
        let is_leaf: bool = unsafe { (*version_ptr).is_leaf() };

        if is_leaf {
            // SAFETY: is_leaf() confirmed this is a leaf
            unsafe { &mut *(root.cast::<L>()) }
        } else {
            // SAFETY: !is_leaf() confirmed this is an internode
            let internode: &L::Internode = unsafe { &*(root.cast::<L::Internode>()) };

            let ikey: u64 = key.ikey();
            let child_idx: usize =
                upper_bound_internode_generic::<S, L::Internode>(ikey, internode);
            let start_ptr: *mut u8 = internode.child(child_idx);

            // Prefetch child node
            prefetch_read(start_ptr);

            let children_are_leaves: bool = internode.children_are_leaves();

            if children_are_leaves {
                // SAFETY: children_are_leaves() guarantees child is a leaf
                unsafe { &mut *start_ptr.cast::<L>() }
            } else {
                // Iterative traversal for deeper trees
                // SAFETY: The returned pointer is valid for the tree's lifetime
                unsafe { &mut *Self::reach_leaf_mut_iterative_generic(start_ptr, ikey) }
            }
        }
    }

    /// Iterative leaf reach for deeply nested trees (generic version).
    ///
    /// # Safety
    ///
    /// The returned pointer is valid for as long as the tree's allocations remain valid.
    fn reach_leaf_mut_iterative_generic(mut current: *mut u8, ikey: u64) -> *mut L {
        use crate::ksearch::upper_bound_internode_generic;
        use crate::prefetch::prefetch_read;

        loop {
            // SAFETY: current is a valid internode pointer from traversal
            let internode: &L::Internode = unsafe { &*(current.cast::<L::Internode>()) };
            let child_idx: usize =
                upper_bound_internode_generic::<S, L::Internode>(ikey, internode);
            let child_ptr: *mut u8 = internode.child(child_idx);

            // Prefetch child node
            prefetch_read(child_ptr);

            if internode.children_are_leaves() {
                // SAFETY: children_are_leaves() guarantees child is a leaf
                return child_ptr.cast::<L>();
            }

            current = child_ptr;
        }
    }

    // ========================================================================
    //  Generic Optimistic Read Path
    // ========================================================================

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
        self.get_concurrent_generic(&mut search_key, guard)
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
        self.get_ref_generic(&mut search_key, guard)
    }

    /// Internal concurrent get implementation returning a reference.
    ///
    /// Same protocol as [`get_concurrent_generic`] but returns `&V` instead of `Arc<V>`.
    /// Eliminates Arc clone overhead for maximum read performance.
    #[expect(clippy::too_many_lines, reason = "Complex Concurrency Logic")]
    fn get_ref_generic<'g>(
        &self,
        key: &mut Key<'_>,
        guard: &'g LocalGuard<'_>,
    ) -> Option<&'g S::Value> {
        use crate::leaf_trait::TreePermutation;
        use crate::leaf24::KSUF_KEYLENX;
        use crate::leaf24::LAYER_KEYLENX;
        use crate::link::{is_marked, unmark_ptr};

        // Start at tree root
        let mut layer_root: *const u8 = self.load_root_ptr_generic(guard);
        let mut in_sublayer: bool = false;

        'layer_loop: loop {
            // Find the actual layer root (handles layer root promotion for sublayers)
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

                    // Load permutation - if frozen, a split is in progress
                    let Ok(perm) = leaf.permutation_try() else {
                        version = leaf.version().stable();
                        continue 'search_loop;
                    };

                    let target_ikey: u64 = key.ikey();

                    #[expect(clippy::cast_possible_truncation, reason = "current_len() <= 8")]
                    let search_keylenx: u8 = if key.has_suffix() {
                        KSUF_KEYLENX
                    } else {
                        key.current_len() as u8
                    };

                    // Search for matching key using SIMD-accelerated ikey comparison.
                    // Record snapshot only - do NOT interpret until version validated.
                    let mut match_snapshot: Option<(u8, *mut u8)> = None;

                    // SIMD: Find all physical slots with matching ikey in parallel.
                    let ikey_matches: u32 = leaf.find_ikey_matches(target_ikey);

                    for i in 0..perm.size() {
                        let slot: usize = perm.get(i);

                        // O(1) bit test instead of atomic load + compare
                        if (ikey_matches & (1 << slot)) == 0 {
                            continue;
                        }

                        // ikey matches - now load keylenx and ptr
                        let slot_keylenx: u8 = leaf.keylenx(slot);
                        let slot_ptr: *mut u8 = leaf.leaf_value_ptr(slot);

                        if slot_ptr.is_null() {
                            continue;
                        }

                        if slot_keylenx == search_keylenx {
                            let suffix_match: bool = if slot_keylenx == KSUF_KEYLENX {
                                leaf.ksuf_equals(slot, key.suffix())
                            } else {
                                true
                            };

                            if suffix_match {
                                match_snapshot = Some((slot_keylenx, slot_ptr));
                                break;
                            }
                        } else if slot_keylenx >= LAYER_KEYLENX && key.has_suffix() {
                            match_snapshot = Some((slot_keylenx, slot_ptr));
                            break;
                        }
                    }

                    // Validate version AFTER all reads
                    if leaf.version().has_changed(version) {
                        let (advanced, new_version) =
                            self.advance_to_key_generic(leaf, key, version, guard);

                        if !std::ptr::eq(advanced, leaf) {
                            leaf_ptr = std::ptr::from_ref(advanced).cast_mut();
                            continue 'leaf_loop;
                        }

                        version = new_version;
                        continue 'search_loop;
                    }

                    // VERSION VALIDATED - NOW SAFE TO INTERPRET SNAPSHOT
                    if let Some((keylenx, ptr)) = match_snapshot {
                        if keylenx >= LAYER_KEYLENX {
                            // Layer pointer - descend into sublayer
                            key.shift();
                            layer_root = ptr;
                            in_sublayer = true;
                            continue 'layer_loop;
                        }

                        // Value - return reference WITHOUT cloning Arc
                        // SAFETY: version validated, guard protects from deallocation,
                        // ptr points to valid Arc<V> data
                        let value_ref: &'g S::Value = unsafe { &*(ptr.cast::<S::Value>()) };
                        return Some(value_ref);
                    }

                    // Not found - check for dirty or B-link
                    if leaf.version().is_dirty() {
                        version = leaf.version().stable();
                        continue 'search_loop;
                    }

                    let next_raw: *mut L = leaf.next_raw();
                    let next_ptr: *mut L = unmark_ptr(next_raw);
                    if !next_ptr.is_null() && !is_marked(next_raw) {
                        let next_bound: u64 = unsafe { (*next_ptr).ikey_bound() };
                        if target_ikey >= next_bound {
                            leaf_ptr = next_ptr;
                            continue 'leaf_loop;
                        }
                    }

                    return None;
                }
            }
        }
    }

    /// Internal concurrent get implementation with layer descent support.
    #[expect(clippy::too_many_lines, reason = "Complex Concurrency Logic")]
    fn get_concurrent_generic(
        &self,
        key: &mut Key<'_>,
        guard: &LocalGuard<'_>,
    ) -> Option<S::Output> {
        use crate::leaf_trait::TreePermutation;
        use crate::leaf24::KSUF_KEYLENX;
        use crate::leaf24::LAYER_KEYLENX;
        use crate::link::{is_marked, unmark_ptr};

        #[cfg(feature = "tracing")]
        let target_ikey_for_trace: u64 = key.ikey();

        #[cfg(feature = "tracing")]
        tracing::trace!(
            ikey = format_args!("{:016x}", target_ikey_for_trace),
            "get: START"
        );

        // Start at tree root
        let mut layer_root: *const u8 = self.load_root_ptr_generic(guard);
        let mut in_sublayer: bool = false;

        'layer_loop: loop {
            // Find the actual layer root (handles layer root promotion for sublayers)
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
                        continue 'layer_loop; // Retry from layer root
                    }

                    // Load permutation - if frozen, a split is in progress
                    let Ok(perm) = leaf.permutation_try() else {
                        version = leaf.version().stable();
                        continue 'search_loop;
                    };

                    let target_ikey: u64 = key.ikey();

                    #[expect(clippy::cast_possible_truncation, reason = "current_len() <= 8")]
                    let search_keylenx: u8 = if key.has_suffix() {
                        KSUF_KEYLENX
                    } else {
                        key.current_len() as u8
                    };

                    // Search for matching key using SIMD-accelerated ikey comparison.
                    // CRITICAL: Only RECORD the snapshot (keylenx, ptr) here.
                    // Do NOT interpret the pointer until AFTER version validation.
                    let mut match_snapshot: Option<(u8, *mut u8)> = None;

                    // SIMD: Find all physical slots with matching ikey in parallel.
                    // Returns bitmask where bit i is set if ikey[i] == target_ikey.
                    let ikey_matches: u32 = leaf.find_ikey_matches(target_ikey);

                    for i in 0..perm.size() {
                        let slot: usize = perm.get(i);

                        // O(1) bit test instead of atomic load + compare
                        if (ikey_matches & (1 << slot)) == 0 {
                            continue;
                        }

                        // ikey matches - now load keylenx and ptr
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
                                match_snapshot = Some((slot_keylenx, slot_ptr));
                                break;
                            }
                        } else if slot_keylenx >= LAYER_KEYLENX && key.has_suffix() {
                            // Layer pointer - record for descent after validation
                            match_snapshot = Some((slot_keylenx, slot_ptr));
                            break;
                        }
                    }

                    // Validate version AFTER all reads
                    if leaf.version().has_changed(version) {
                        // Version changed - follow B-link chain if split occurred
                        let (advanced, new_version) =
                            self.advance_to_key_generic(leaf, key, version, guard);

                        if !std::ptr::eq(advanced, leaf) {
                            // Different leaf - search there
                            leaf_ptr = std::ptr::from_ref(advanced).cast_mut();
                            continue 'leaf_loop;
                        }

                        // Same leaf, new version - retry search with returned version
                        version = new_version;
                        continue 'search_loop;
                    }

                    // ================================================================
                    //  VERSION VALIDATED - NOW SAFE TO INTERPRET SNAPSHOT
                    // ================================================================

                    if let Some((keylenx, ptr)) = match_snapshot {
                        if keylenx >= LAYER_KEYLENX {
                            // Layer pointer - descend into sublayer
                            key.shift();
                            layer_root = ptr;
                            in_sublayer = true;
                            continue 'layer_loop;
                        }

                        // Value - NOW safe to clone
                        // SAFETY: version validated, so keylenx correctly identifies ptr as value
                        let output: S::Output = unsafe { S::output_from_raw(ptr) };
                        return Some(output);
                    }

                    // Not found - but might be in wrong leaf due to split!
                    // If version is dirty (split/insert in progress), retry
                    if leaf.version().is_dirty() {
                        version = leaf.version().stable();
                        continue 'search_loop;
                    }

                    // Check if key belongs to a right sibling via B-link
                    let next_raw: *mut L = leaf.next_raw();
                    let next_ptr: *mut L = unmark_ptr(next_raw);
                    if !next_ptr.is_null() && !is_marked(next_raw) {
                        // SAFETY: next_ptr is valid
                        let next_bound: u64 = unsafe { (*next_ptr).ikey_bound() };
                        if target_ikey >= next_bound {
                            // Key should be in the next leaf - follow B-link
                            #[cfg(feature = "tracing")]
                            tracing::debug!(
                                ikey = target_ikey,
                                leaf_ptr = ?std::ptr::from_ref(leaf),
                                next_ptr = ?next_ptr,
                                next_bound = next_bound,
                                "get: NotFound but ikey >= next_bound; following B-link"
                            );
                            #[cfg(feature = "tracing")]
                            crate::tree::optimistic::BLINK_SHOULD_FOLLOW_COUNT
                                .fetch_add(1, AtomicOrdering::Relaxed);
                            leaf_ptr = next_ptr;
                            continue 'leaf_loop;
                        }
                    }

                    // Truly not found
                    #[cfg(feature = "tracing")]
                    tracing::debug!(
                        ikey = format_args!("{:016x}", target_ikey),
                        leaf_ptr = ?std::ptr::from_ref(leaf),
                        perm_size = perm.size(),
                        next_ptr = ?next_ptr,
                        is_marked = is_marked(next_raw),
                        "get: NOT_FOUND"
                    );
                    #[cfg(feature = "tracing")]
                    crate::tree::optimistic::SEARCH_NOT_FOUND_COUNT
                        .fetch_add(1, AtomicOrdering::Relaxed);
                    return None;
                }
            }
        }
    }

    /// Follow parent pointers to find the actual layer root.
    #[expect(clippy::unused_self, reason = "Method signature pattern")]
    fn maybe_parent_generic(&self, mut node: *const u8) -> *const u8 {
        loop {
            // SAFETY: node is valid, both types have NodeVersion as first field
            #[expect(clippy::cast_ptr_alignment, reason = "proper alignment")]
            let version: &NodeVersion = unsafe { &*(node.cast::<NodeVersion>()) };

            let parent = if version.is_leaf() {
                // SAFETY: version.is_leaf() confirmed
                let leaf: &L = unsafe { &*(node.cast::<L>()) };
                leaf.parent()
            } else {
                // SAFETY: !version.is_leaf() confirmed
                let inode: &L::Internode = unsafe { &*(node.cast::<L::Internode>()) };
                inode.parent()
            };

            if parent.is_null() {
                return node;
            }

            node = parent;
        }
    }

    /// Traverse from layer root to target leaf with version validation.
    ///
    /// Simple loop that descends through internodes to find the target leaf.
    /// B-link walking in `advance_to_key` handles any splits that occur.
    #[expect(clippy::unused_self, reason = "API Consistency")]
    #[expect(clippy::used_underscore_binding, reason = "Lock guard")]
    #[cfg_attr(
        feature = "tracing",
        tracing::instrument(level = "trace", skip(self, _guard), fields(ikey = %format_args!("{:016x}", key.ikey())))
    )]
    fn reach_leaf_concurrent_generic(
        &self,
        start: *const u8,
        key: &Key<'_>,
        _is_sublayer: bool,
        _guard: &LocalGuard<'_>,
    ) -> *mut L {
        use crate::ksearch::upper_bound_internode_generic;
        use crate::leaf_trait::TreeInternode;
        use crate::prefetch::prefetch_read;

        let target_ikey: u64 = key.ikey();
        let mut node: *const u8 = start;

        loop {
            // SAFETY: node is valid, both node types have NodeVersion as first field
            #[expect(clippy::cast_ptr_alignment, reason = "proper alignment")]
            let version: &NodeVersion = unsafe { &*(node.cast::<NodeVersion>()) };

            // Get stable version (spins if dirty)
            let v: u32 = version.stable();

            if version.is_leaf() {
                // Reached a leaf
                return node.cast_mut().cast::<L>();
            }

            // It's an internode - traverse down
            // SAFETY: !is_leaf() confirmed above
            let inode: &L::Internode = unsafe { &*(node.cast::<L::Internode>()) };

            // Binary search for child
            let child_idx: usize =
                upper_bound_internode_generic::<S, L::Internode>(target_ikey, inode);
            let child: *mut u8 = inode.child(child_idx);

            // Prefetch child node while we validate version (hides memory latency)
            prefetch_read(child);

            if child.is_null() {
                // Concurrent split in progress - retry from start
                node = start;
                continue;
            }

            // Check if internode changed during our read
            if inode.version().has_changed(v) {
                // Version changed - check for split
                if inode.version().has_split(v) {
                    // Key might have escaped to sibling - retry from start
                    node = start;
                    continue;
                }
                // Just retry this internode
                continue;
            }

            // Descend to child
            node = child;
        }
    }

    // ========================================================================
    //  Generic CAS Insert Path (disabled - kept for future reference)
    // ========================================================================

    /// Maximum CAS retry attempts before falling back to locked path.
    const MAX_CAS_RETRIES_GENERIC: usize = 3;

    /// Try CAS-based lock-free insert.
    ///
    /// Attempts to insert a new key-value pair using optimistic concurrency.
    /// Returns result indicating success or reason for fallback.
    #[expect(dead_code, reason = "CAS path disabled")]
    #[expect(clippy::too_many_lines, reason = "Complex concurrency logic")]
    pub(crate) fn try_cas_insert_generic(
        &self,
        key: &Key<'_>,
        value: &S::Output,
        guard: &LocalGuard<'_>,
    ) -> CasInsertResultGeneric<S::Output> {
        use crate::leaf_trait::TreePermutation;
        use std::ptr as StdPtr;

        let ikey: u64 = key.ikey();

        #[expect(clippy::cast_possible_truncation, reason = "current_len() <= 8")]
        let keylenx: u8 = key.current_len() as u8;

        // Suffix keys require locked path
        if key.has_suffix() {
            return CasInsertResultGeneric::ContentionFallback;
        }

        let mut retries: usize = 0;
        let mut leaf_ptr: *mut L = StdPtr::null_mut();
        let mut use_reach: bool = true;

        loop {
            // 1. Optimistic traversal to find target leaf
            // CAS path only operates on layer 0 (no suffix keys), so is_sublayer=false
            if use_reach {
                let layer_root: *const u8 = self.load_root_ptr_generic(guard);
                // Note: get_fresh_root inside reach_leaf_concurrent_generic handles parent traversal
                leaf_ptr = self.reach_leaf_concurrent_generic(layer_root, key, false, guard);
            } else {
                use_reach = true;
            }

            let leaf: &L = unsafe { &*leaf_ptr };

            // B-link advance if needed
            let (advanced, exceeded_hop_limit) =
                self.advance_to_key_by_bound_generic(leaf, key, guard);

            // If we exceeded the hop limit, fall back to locked path which will re-traverse
            if exceeded_hop_limit {
                return CasInsertResultGeneric::ContentionFallback;
            }

            if !StdPtr::eq(advanced, leaf) {
                leaf_ptr = StdPtr::from_ref(advanced).cast_mut();
                use_reach = false;
                continue;
            }

            // 2. Get version (fail-fast if dirty)
            let version: u32 = leaf.version().value();
            if leaf.version().is_dirty() {
                return CasInsertResultGeneric::ContentionFallback;
            }

            // Check for frozen permutation
            // If frozen, wait briefly for version to stabilize before falling back.
            // This prevents spinning on a transient frozen state (Fix B: freeze-wait protocol).
            let Ok(perm) = leaf.permutation_try() else {
                let _ = leaf.version().stable();
                return CasInsertResultGeneric::ContentionFallback;
            };

            // 3. Search for key position
            let search_result = self.search_for_insert_generic(leaf, key, &perm);

            match search_result {
                InsertSearchResultGeneric::Found { slot } => {
                    return CasInsertResultGeneric::ExistsNeedLock { slot };
                }

                InsertSearchResultGeneric::Layer { slot, .. }
                | InsertSearchResultGeneric::Conflict { slot } => {
                    return CasInsertResultGeneric::LayerNeedLock { slot };
                }

                InsertSearchResultGeneric::NotFound { logical_pos } => {
                    // 4. Check if leaf has space
                    if perm.size() >= L::WIDTH {
                        return CasInsertResultGeneric::FullNeedLock;
                    }

                    // 5. Pick a free slot from the free region, scanning for usable slots.
                    //
                    // This mirrors the locked path's `pick_free_slot_avoiding_reserved`:
                    // - Skip slot 0 if it can't be reused (ikey_bound invariant)
                    // - Skip slots that are already reserved (non-null in free region)
                    //
                    // This is a key optimization: instead of only trying perm.back(),
                    // we scan all free slots to find one that's actually available.
                    let size: usize = perm.size();
                    let free_count: usize = L::WIDTH - size;

                    let mut found_slot: Option<(usize, usize)> = None; // (slot, offset)
                    for offset in 0..free_count {
                        let slot: usize = perm.back_at_offset(offset);

                        // Slot-0 / ikey_bound invariant: skip slot 0 if it can't be reused
                        if slot == 0 && !leaf.can_reuse_slot0(ikey) {
                            continue;
                        }

                        // Skip reserved slots (CLAIMING or arc_ptr from another CAS in progress)
                        if !leaf.load_slot_value(slot).is_null() {
                            continue;
                        }

                        found_slot = Some((slot, offset));
                        break;
                    }

                    let Some((slot, back_offset)) = found_slot else {
                        // No usable slot found - all free slots are either:
                        // - slot 0 that can't be reused, or
                        // - reserved by another CAS in progress
                        // Fall back to locked path which can wait for reservations to clear
                        return CasInsertResultGeneric::ContentionFallback;
                    };

                    // 6. Compute new permutation with the chosen slot.
                    //
                    // If the slot is not at offset 0 (the natural back position),
                    // we need to swap it to the back before inserting.
                    let (new_perm, allocated_slot) = if back_offset == 0 {
                        // Slot is already at back, use directly
                        perm.insert_from_back_immutable(logical_pos)
                    } else {
                        // Swap the chosen slot to back position, then insert
                        let mut perm_copy = perm;
                        let back_pos: usize = L::WIDTH - 1;
                        let chosen_pos: usize = back_pos - back_offset;
                        perm_copy.swap_free_slots(back_pos, chosen_pos);
                        let allocated = perm_copy.insert_from_back(logical_pos);
                        (perm_copy, allocated)
                    };
                    debug_assert_eq!(allocated_slot, slot, "slot mismatch after insert");

                    // ============================================================
                    // Option A (Safe) Protocol: 3-phase CAS insert
                    //
                    // State machine: NULL -> CLAIMING -> arc_ptr -> (perm publish)
                    //
                    // Phase 1: Reserve slot (NULL -> CLAIMING)
                    // Phase 2: Write key metadata (exclusive access via CLAIMING)
                    // Phase 3: Install value (CLAIMING -> arc_ptr)
                    // Phase 4: Publish (permutation CAS)
                    // ============================================================

                    // 8. Phase 1: Reserve the slot (NULL -> CLAIMING).
                    //
                    // This gives us exclusive right to write key metadata into this slot.
                    // The CLAIMING sentinel is non-null, so other CAS attempts and the
                    // locked path will see it as "reserved".
                    let claiming: *mut u8 = claiming_ptr();
                    if leaf
                        .cas_slot_value(slot, StdPtr::null_mut(), claiming)
                        .is_err()
                    {
                        // Contention: another thread claimed this slot first.
                        #[cfg(feature = "tracing")]
                        crate::tree::optimistic::CAS_INSERT_RETRY_COUNT
                            .fetch_add(1, AtomicOrdering::Relaxed);
                        retries += 1;
                        if retries > Self::MAX_CAS_RETRIES_GENERIC {
                            return CasInsertResultGeneric::ContentionFallback;
                        }
                        Self::backoff_generic(retries);
                        continue;
                    }

                    // 9. Version check before writing metadata.
                    //
                    // If version changed (split, etc.), release the reservation and retry.
                    if leaf.version().has_changed_or_locked(version) {
                        // Release reservation: CLAIMING -> NULL
                        let _ = leaf.cas_slot_value(slot, claiming, StdPtr::null_mut());
                        #[cfg(feature = "tracing")]
                        crate::tree::optimistic::CAS_INSERT_RETRY_COUNT
                            .fetch_add(1, AtomicOrdering::Relaxed);
                        retries += 1;
                        if retries > Self::MAX_CAS_RETRIES_GENERIC {
                            return CasInsertResultGeneric::ContentionFallback;
                        }
                        Self::backoff_generic(retries);
                        continue;
                    }

                    // 10. Phase 2: Write key metadata.
                    //
                    // We have exclusive access to this slot via CLAIMING. No other CAS
                    // attempt can write metadata here until we release the reservation.
                    // The slot is not visible to readers until permutation publishes.
                    unsafe {
                        leaf.store_key_data_for_cas(slot, ikey, keylenx);
                    }

                    // 11. Version check after metadata.
                    if leaf.version().has_changed_or_locked(version) {
                        // Release reservation: CLAIMING -> NULL
                        let _ = leaf.cas_slot_value(slot, claiming, StdPtr::null_mut());
                        #[cfg(feature = "tracing")]
                        crate::tree::optimistic::CAS_INSERT_RETRY_COUNT
                            .fetch_add(1, AtomicOrdering::Relaxed);
                        retries += 1;
                        if retries > Self::MAX_CAS_RETRIES_GENERIC {
                            return CasInsertResultGeneric::ContentionFallback;
                        }
                        Self::backoff_generic(retries);
                        continue;
                    }

                    // 12. Phase 3: Install the value pointer (CLAIMING -> arc_ptr).
                    //
                    // Prepare the value pointer and transition from CLAIMING to the real value.
                    let value_ptr: *mut u8 = S::output_to_raw(value);
                    match leaf.cas_slot_value(slot, claiming, value_ptr) {
                        Ok(()) => {
                            // Successfully installed value pointer.
                        }
                        Err(actual) => {
                            // Invariant violation: nobody else should touch a CLAIMING slot
                            // in the free region. If this fires, prefer leaking over double-free.
                            debug_assert!(
                                false,
                                "CLAIMING->value_ptr CAS failed; expected CLAIMING, actual={actual:p}"
                            );
                            // Drop the value we just created (it was never installed).
                            // SAFETY: value_ptr was just created by output_to_raw
                            unsafe { S::cleanup_value_ptr(value_ptr) };
                            // Best-effort release: try to reset to NULL.
                            let _ = leaf.cas_slot_value(slot, claiming, StdPtr::null_mut());
                            return CasInsertResultGeneric::ContentionFallback;
                        }
                    }

                    // 13. Verify slot ownership (should still be value_ptr).
                    if leaf.load_slot_value(slot) != value_ptr {
                        // Slot was stolen after we installed value_ptr. This is unexpected
                        // but we handle it by cleaning up our value and falling back.
                        // SAFETY: value_ptr was just created by output_to_raw
                        unsafe { S::cleanup_value_ptr(value_ptr) };
                        #[cfg(feature = "tracing")]
                        crate::tree::optimistic::CAS_INSERT_RETRY_COUNT
                            .fetch_add(1, AtomicOrdering::Relaxed);
                        retries += 1;
                        if retries > Self::MAX_CAS_RETRIES_GENERIC {
                            return CasInsertResultGeneric::ContentionFallback;
                        }
                        Self::backoff_generic(retries);
                        continue;
                    }

                    // 14. Final version check before permutation publish.
                    if leaf.version().has_changed_or_locked(version) {
                        match leaf.cas_slot_value(slot, value_ptr, StdPtr::null_mut()) {
                            Ok(()) | Err(_) => {
                                // SAFETY: value_ptr was just created by output_to_raw
                                unsafe { S::cleanup_value_ptr(value_ptr) };
                            }
                        }
                        #[cfg(feature = "tracing")]
                        crate::tree::optimistic::CAS_INSERT_RETRY_COUNT
                            .fetch_add(1, AtomicOrdering::Relaxed);
                        retries += 1;
                        if retries > Self::MAX_CAS_RETRIES_GENERIC {
                            return CasInsertResultGeneric::ContentionFallback;
                        }
                        Self::backoff_generic(retries);
                        continue;
                    }

                    // 15. Phase 4: CAS permutation to publish.
                    //
                    // The permutation CAS is the linearization point. Once it succeeds,
                    // the insert is logically complete and visible to other threads.
                    // No post-publish waits or checks are needed because:
                    //
                    // 1. Permutation freezing prevents CAS racing with splits
                    // 2. If a split happens immediately after, entry migrates correctly
                    // 3. Post-publish waits (stable(), wait_for_split(), permutation_wait())
                    //    defeat the purpose of a fast path and can take milliseconds
                    match leaf.cas_permutation_raw(perm, new_perm) {
                        Ok(()) => {
                            // Success! Increment count and return.
                            self.count.fetch_add(1, AtomicOrdering::Relaxed);
                            return CasInsertResultGeneric::Success(None);
                        }

                        Err(failure) => {
                            match leaf.cas_slot_value(slot, value_ptr, StdPtr::null_mut()) {
                                Ok(()) | Err(_) => {
                                    // SAFETY: value_ptr was just created by output_to_raw
                                    unsafe { S::cleanup_value_ptr(value_ptr) };
                                }
                            }

                            if failure.is_frozen() {
                                return CasInsertResultGeneric::ContentionFallback;
                            }

                            #[cfg(feature = "tracing")]
                            crate::tree::optimistic::CAS_INSERT_RETRY_COUNT
                                .fetch_add(1, AtomicOrdering::Relaxed);
                            retries += 1;
                            if retries > Self::MAX_CAS_RETRIES_GENERIC {
                                return CasInsertResultGeneric::ContentionFallback;
                            }
                            Self::backoff_generic(retries);
                        }
                    }
                }
            }
        }
    }

    /// Exponential backoff for CAS retries.
    #[inline(always)]
    fn backoff_generic(retries: usize) {
        let spins = 1usize << retries.min(6);
        for _ in 0..spins {
            std::hint::spin_loop();
        }
    }

    /// Search for insert position in a leaf (generic version).
    #[expect(clippy::unused_self, reason = "API Consistency")]
    fn search_for_insert_generic(
        &self,
        leaf: &L,
        key: &Key<'_>,
        perm: &L::Perm,
    ) -> InsertSearchResultGeneric {
        use crate::leaf_trait::TreePermutation;
        use crate::leaf24::KSUF_KEYLENX;
        use crate::leaf24::LAYER_KEYLENX;

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

            if slot_ikey == target_ikey {
                let slot_keylenx: u8 = leaf.keylenx(slot);
                let slot_ptr: *mut u8 = leaf.leaf_value_ptr(slot);

                if slot_ptr.is_null() {
                    continue;
                }

                // Layer pointer - only descend if the new key has more bytes
                if slot_keylenx >= LAYER_KEYLENX {
                    if key.has_suffix() {
                        // Key has more bytes - descend into the layer
                        return InsertSearchResultGeneric::Layer {
                            slot,
                            shift_amount: 8,
                        };
                    }
                    // Key terminates here - it's distinct from layer contents
                    // Continue searching for an exact match or insert position
                    continue;
                }

                // Exact match check
                if slot_keylenx == search_keylenx {
                    if slot_keylenx == KSUF_KEYLENX {
                        // Both have suffixes - compare them
                        let key_suffix: &[u8] = key.suffix();
                        if let Some(slot_suffix) = leaf.ksuf(slot) {
                            if key_suffix == slot_suffix {
                                // Same suffix = same key
                                return InsertSearchResultGeneric::Found { slot };
                            }
                            // Different suffixes = conflict, need layer
                            return InsertSearchResultGeneric::Conflict { slot };
                        }
                        // No stored suffix (shouldn't happen for KSUF_KEYLENX)
                        // but treat as conflict to be safe
                        return InsertSearchResultGeneric::Conflict { slot };
                    }
                    // Inline keys (no suffix) with matching keylenx = same key
                    return InsertSearchResultGeneric::Found { slot };
                }

                // Same ikey, different keylenx - check if conflict is needed
                let slot_has_suffix: bool = slot_keylenx == KSUF_KEYLENX;
                let key_has_suffix: bool = key.has_suffix();

                if slot_has_suffix && key_has_suffix {
                    // Both have suffixes with same 8-byte prefix - need layer
                    return InsertSearchResultGeneric::Conflict { slot };
                }
                // One inline, one suffix - distinct keys, continue searching
            }

            // Sorted order - found insert position
            if slot_ikey > target_ikey {
                return InsertSearchResultGeneric::NotFound { logical_pos: i };
            }
        }

        // Insert at end
        InsertSearchResultGeneric::NotFound {
            logical_pos: perm.size(),
        }
    }

    /// Advance to correct leaf via B-link after version change detected.
    ///
    /// This is called when `has_changed(old_version)` returns true, indicating
    /// a split may have occurred. It follows B-links to find the correct leaf.
    #[expect(clippy::unused_self, reason = "API Consistency")]
    fn advance_to_key_generic<'a>(
        &'a self,
        mut leaf: &'a L,
        key: &Key<'_>,
        old_version: u32,
        _guard: &LocalGuard<'_>,
    ) -> (&'a L, u32) {
        use crate::link::{is_marked, unmark_ptr};

        let key_ikey: u64 = key.ikey();
        let mut version: u32 = leaf.version().stable();

        // Only follow chain if split occurred or is in progress
        if !leaf.version().has_split(old_version) {
            // Double-check: split could have started after has_split check
            if !leaf.version().is_splitting() {
                return (leaf, version);
            }
        }

        // Wait for any in-progress split to complete
        version = leaf.version().stable();

        while !leaf.version().is_deleted() {
            let next_raw: *mut L = leaf.next_raw();

            // Check for marked pointer (split in progress)
            if is_marked(next_raw) {
                leaf.wait_for_split();
                continue;
            }

            let next_ptr: *mut L = unmark_ptr(next_raw);
            if next_ptr.is_null() {
                break;
            }

            // SAFETY: next_ptr protected by guard
            let next: &L = unsafe { &*next_ptr };
            let next_bound: u64 = next.ikey_bound();

            if key_ikey >= next_bound {
                // Key belongs in next leaf or further
                leaf = next;
                version = leaf.version().stable();
                continue;
            }

            // Key belongs in current leaf
            break;
        }

        (leaf, version)
    }

    /// Maximum B-link hops before giving up and signaling re-descent.
    ///
    /// This prevents unbounded sibling walks when routing is inconsistent.
    /// After this many hops, we return the current leaf and let the caller
    /// detect the mismatch and retry from the root.
    const MAX_BLINK_HOPS: usize = 128;

    /// Advance to correct leaf via B-link (generic version).
    /// Used by insert path before locking.
    ///
    /// Returns `(leaf, exceeded_hop_limit)`. If `exceeded_hop_limit` is true,
    /// the caller should retry from the root instead of trusting this leaf.
    #[cfg_attr(
        feature = "tracing",
        tracing::instrument(level = "trace", skip_all, fields(ikey = %format_args!("{:016x}", key.ikey())))
    )]
    #[expect(clippy::unused_self, reason = "API Consistency")]
    fn advance_to_key_by_bound_generic<'a>(
        &'a self,
        mut leaf: &'a L,
        key: &Key<'_>,
        _guard: &LocalGuard<'_>,
    ) -> (&'a L, bool) {
        use crate::link::{is_marked, unmark_ptr};

        let key_ikey: u64 = key.ikey();
        let mut hops: usize = 0;

        // Wait for any in-progress split to complete
        if leaf.version().is_splitting() {
            let _ = leaf.version().stable();
        }

        loop {
            // Check hop limit to prevent unbounded B-link walks
            if hops >= Self::MAX_BLINK_HOPS {
                #[cfg(feature = "tracing")]
                tracing::warn!(
                    ikey = format_args!("{:016x}", key_ikey),
                    hops = hops,
                    "BLINK_HOP_LIMIT_EXCEEDED: too many sibling hops, signaling re-descent"
                );
                return (leaf, true);
            }

            let next_raw: *mut L = leaf.next_raw();
            if is_marked(next_raw) {
                leaf.wait_for_split();
                continue;
            }

            let next_ptr: *mut L = unmark_ptr(next_raw);
            if next_ptr.is_null() {
                break;
            }

            // SAFETY: next_ptr is valid
            let next: &L = unsafe { &*next_ptr };
            let next_bound: u64 = next.ikey_bound();

            if key_ikey >= next_bound {
                // Key belongs in next leaf or further
                leaf = next;
                hops += 1;
                continue;
            }

            break;
        }

        (leaf, false)
    }

    // ========================================================================
    //  Generic Locked Insert Path
    // ========================================================================

    /// Insert a key-value pair.
    ///
    /// Creates a guard internally. For bulk operations, prefer
    /// [`insert_with_guard`](Self::insert_with_guard) to amortize guard creation cost.
    ///
    /// # Returns
    ///
    /// * `Ok(None)` - New key inserted
    /// * `Ok(Some(old))` - Key existed, old value returned
    /// * `Err(InsertError)` - Insert failed (key too long)
    ///
    /// # Errors
    /// If insert fails.
    #[inline]
    pub fn insert(&self, key: &[u8], value: S::Value) -> Result<Option<S::Output>, InsertError> {
        let guard = self.guard();
        self.insert_with_guard(key, value, &guard)
    }

    /// Insert a key-value pair using an explicit guard.
    ///
    /// This is the main public insert API for `MassTreeGeneric`.
    ///
    /// # Arguments
    ///
    /// * `key` - The key to insert (byte slice)
    /// * `value` - The value to insert
    /// * `guard` - A guard from [`MassTreeGeneric::guard()`]
    ///
    /// # Returns
    ///
    /// * `Ok(None)` - New key inserted
    /// * `Ok(Some(old))` - Key existed, old value returned
    /// * `Err(InsertError)` - Insert failed (key too long)
    ///
    /// # Errors
    /// If insert fails.
    pub fn insert_with_guard(
        &self,
        key: &[u8],
        value: S::Value,
        guard: &LocalGuard<'_>,
    ) -> Result<Option<S::Output>, InsertError> {
        let mut key = Key::new(key);
        let output = S::into_output(value);
        self.insert_concurrent_generic(&mut key, output, guard)
    }

    /// Internal concurrent insert with CAS fast path and locked fallback.
    #[cfg_attr(
        feature = "tracing",
        tracing::instrument(level = "debug", skip_all, fields(ikey = %format_args!("{:016x}", key.ikey())))
    )]
    #[expect(clippy::too_many_lines, reason = "Complex concurrency logic")]
    fn insert_concurrent_generic(
        &self,
        key: &mut Key<'_>,
        value: S::Output,
        guard: &LocalGuard<'_>,
    ) -> Result<Option<S::Output>, InsertError> {
        #[cfg(feature = "tracing")]
        let ikey_for_trace: u64 = key.ikey();

        #[cfg(feature = "tracing")]
        let _insert_start = Instant::now();

        #[cfg(feature = "tracing")]
        let mut retry_count: u32 = 0;

        #[cfg(feature = "tracing")]
        tracing::trace!(
            ikey = format_args!("{:016x}", ikey_for_trace),
            "INSERT_START"
        );

        // Track current layer root
        let mut layer_root: *const u8 = self.load_root_ptr_generic(guard);

        // Track whether we're in a sublayer (for layer traversal)
        let mut in_sublayer: bool = false;

        loop {
            // Find the actual layer root (handles layer root promotion for sublayers)
            layer_root = self.maybe_parent_generic(layer_root);

            // Locked path - traverse to leaf
            let leaf_ptr: *mut L =
                self.reach_leaf_concurrent_generic(layer_root, key, in_sublayer, guard);

            let leaf: &L = unsafe { &*leaf_ptr };

            // B-link advance if needed
            let (leaf, exceeded_hop_limit) = self.advance_to_key_by_bound_generic(leaf, key, guard);

            // If we exceeded the hop limit, re-traverse from root
            if exceeded_hop_limit {
                layer_root = self.load_root_ptr_generic(guard);
                continue;
            }

            // Lock the leaf
            #[cfg(feature = "tracing")]
            let lock_start = Instant::now();

            let mut lock = leaf.version().lock();

            #[cfg(feature = "tracing")]
            #[expect(clippy::cast_possible_truncation)]
            {
                let lock_elapsed = lock_start.elapsed();
                if lock_elapsed > std::time::Duration::from_millis(1) {
                    tracing::warn!(
                        ikey = format_args!("{:016x}", ikey_for_trace),
                        leaf_ptr = ?std::ptr::from_ref(leaf),
                        lock_elapsed_us = lock_elapsed.as_micros() as u64,
                        retry_count = retry_count,
                        "SLOW_LEAF_LOCK: acquiring leaf lock took >1ms"
                    );
                }
            }

            // Post-lock membership check (C++ masstree_insert/split pattern):
            // The key may have moved to a newly-linked right sibling between:
            // 1) `advance_to_key_by_bound_generic` and
            // 2) acquiring the lock.
            //
            // If we insert into the wrong (left) leaf, the key becomes unreachable via the
            // normal get path (which only follows B-links to the right).
            {
                use crate::link::{is_marked, unmark_ptr};

                let next_raw: *mut L = leaf.next_raw();
                if is_marked(next_raw) {
                    #[cfg(feature = "tracing")]
                    tracing::debug!(
                        ikey = format_args!("{:016x}", ikey_for_trace),
                        leaf_ptr = ?std::ptr::from_ref(leaf),
                        "INSERT_RETRY: leaf marked for split, waiting"
                    );
                    leaf.wait_for_split();
                    drop(lock);
                    #[cfg(feature = "tracing")]
                    {
                        retry_count += 1;
                    }
                    continue;
                }

                let next_ptr: *mut L = unmark_ptr(next_raw);
                if !next_ptr.is_null() {
                    // SAFETY: next_ptr is a valid leaf pointer (protected by the guard).
                    let next_bound: u64 = unsafe { (*next_ptr).ikey_bound() };
                    if key.ikey() >= next_bound {
                        #[cfg(feature = "tracing")]
                        tracing::debug!(
                            ikey = format_args!("{:016x}", ikey_for_trace),
                            leaf_ptr = ?std::ptr::from_ref(leaf),
                            next_bound = format_args!("{:016x}", next_bound),
                            "INSERT_RETRY: key moved to next sibling (post-lock check)"
                        );
                        #[cfg(feature = "tracing")]
                        crate::tree::optimistic::WRONG_LEAF_INSERT_COUNT
                            .fetch_add(1, AtomicOrdering::Relaxed);
                        drop(lock);
                        #[cfg(feature = "tracing")]
                        {
                            retry_count += 1;
                        }
                        continue;
                    }
                }
            }

            // Get permutation (must not be frozen since we hold lock)
            let perm = leaf.permutation();

            // Search for insert position
            let search_result = self.search_for_insert_generic(leaf, key, &perm);

            match search_result {
                InsertSearchResultGeneric::Found { slot } => {
                    // Key exists - update value
                    let old_ptr: *mut u8 = leaf.leaf_value_ptr(slot);
                    if !old_ptr.is_null() {
                        // Clone old value for return BEFORE we store new pointer.
                        // SAFETY: old_ptr is non-null and came from output_to_raw
                        let old_output: S::Output = unsafe { S::output_from_raw(old_ptr) };

                        let new_ptr: *mut u8 = S::output_consume_to_raw(value);

                        // Mark insert, store value, unlock happens on drop
                        lock.mark_insert();
                        leaf.set_leaf_value_ptr(slot, new_ptr);
                        drop(lock);

                        // Defer retirement of the old value.
                        // This ensures readers who captured old_ptr before our store
                        // can safely complete their validation and retry.
                        // SAFETY: old_ptr came from output_to_raw
                        unsafe {
                            guard.defer_retire(old_ptr, |ptr, _| {
                                S::cleanup_value_ptr(ptr);
                            });
                        }

                        #[cfg(feature = "tracing")]
                        crate::tree::optimistic::LOCKED_INSERT_COUNT
                            .fetch_add(1, AtomicOrdering::Relaxed);
                        return Ok(Some(old_output));
                    }
                    drop(lock);
                }

                InsertSearchResultGeneric::NotFound { logical_pos } => {
                    let ikey: u64 = key.ikey();

                    // New key - check if leaf has space
                    if perm.size() >= L::WIDTH {
                        // Leaf is full - perform split using SPLIT-THEN-RETRY pattern
                        let leaf_ptr_current: *mut L = std::ptr::from_ref(leaf).cast_mut();

                        // Split the leaf (takes lock ownership, releases before returning)
                        self.handle_leaf_split_generic(
                            leaf_ptr_current,
                            lock, // Move lock ownership
                            logical_pos,
                            ikey,
                            guard,
                        )?;

                        // Lock was released by handle_leaf_split_generic.
                        // Retry the insert - next iteration will find correct leaf with space.
                        continue;
                    }

                    // Pick a free slot, handling slot-0 / ikey_bound invariant.
                    // Since we hold the lock, we can use perm.back() directly.
                    let slot: usize = perm.back();

                    // Check slot-0 rule: slot 0 stores ikey_bound and can only be
                    // reused if the new key has the same ikey as the current bound.
                    if slot == 0 && !leaf.can_reuse_slot0(ikey) {
                        // Need to find another slot or split.
                        // Scan free region for a non-zero slot.
                        let free_count: usize = L::WIDTH - perm.size();
                        let mut found_slot: Option<(usize, usize)> = None;

                        for offset in 1..free_count {
                            let candidate: usize = perm.back_at_offset(offset);
                            if candidate != 0 {
                                found_slot = Some((candidate, offset));
                                break;
                            }
                        }

                        if let Some((alt_slot, back_offset)) = found_slot {
                            // Use the alternative slot
                            self.assign_slot_generic(leaf, &mut lock, alt_slot, key, &value, guard);

                            let mut new_perm = perm;
                            let back_pos: usize = L::WIDTH - 1;
                            let chosen_pos: usize = back_pos - back_offset;
                            new_perm.swap_free_slots(back_pos, chosen_pos);
                            let allocated: usize = new_perm.insert_from_back(logical_pos);
                            debug_assert_eq!(allocated, alt_slot, "allocated unexpected slot");
                            leaf.set_permutation(new_perm);
                            drop(lock);

                            self.count.fetch_add(1, AtomicOrdering::Relaxed);
                            #[cfg(feature = "tracing")]
                            crate::tree::optimistic::LOCKED_INSERT_COUNT
                                .fetch_add(1, AtomicOrdering::Relaxed);
                            return Ok(None);
                        }

                        // Only slot 0 is free and can't be reused - must split
                        let leaf_ptr_current: *mut L = std::ptr::from_ref(leaf).cast_mut();
                        self.handle_leaf_split_generic(
                            leaf_ptr_current,
                            lock,
                            logical_pos,
                            ikey,
                            guard,
                        )?;
                        continue;
                    }

                    // Use perm.back() directly - simple case
                    self.assign_slot_generic(leaf, &mut lock, slot, key, &value, guard);

                    let mut new_perm = perm;
                    let allocated: usize = new_perm.insert_from_back(logical_pos);
                    debug_assert_eq!(allocated, slot, "allocated unexpected slot");
                    leaf.set_permutation(new_perm);
                    drop(lock);

                    self.count.fetch_add(1, AtomicOrdering::Relaxed);
                    #[cfg(feature = "tracing")]
                    crate::tree::optimistic::LOCKED_INSERT_COUNT
                        .fetch_add(1, AtomicOrdering::Relaxed);
                    return Ok(None);
                }

                InsertSearchResultGeneric::Layer { slot, .. } => {
                    // Descend into sublayer
                    let layer_ptr: *mut u8 = leaf.leaf_value_ptr(slot);
                    drop(lock);

                    key.shift();

                    layer_root = layer_ptr;
                    in_sublayer = true; // We're now in a sublayer
                }

                InsertSearchResultGeneric::Conflict { slot } => {
                    // =================================================================
                    // Suffix Conflict: Same ikey, different suffix
                    // Create a new layer to distinguish the keys
                    // =================================================================

                    // Mark insert before modifying the node
                    lock.mark_insert();

                    // Create new layer for the conflicting keys
                    //
                    // SAFETY:
                    // - We hold the lock on `leaf`
                    // - `guard` is from this tree's collector
                    let layer_ptr: *mut u8 = unsafe {
                        self.create_layer_concurrent_generic(leaf, slot, key, value.clone(), guard)
                    };

                    // CRITICAL: Defer retirement of the existing value in the conflict slot.
                    //
                    // The create_layer_concurrent_generic function cloned it via try_clone_output(),
                    // so the slot's reference is now redundant. We must retire it to avoid
                    // leaking memory when we overwrite with the layer pointer.
                    //
                    // IMPORTANT: We use defer_retire instead of immediate cleanup because
                    // concurrent get_ref() readers may still hold references to this value.
                    // The guard ensures the value isn't freed until all readers have completed.
                    //
                    // SAFETY:
                    // - We hold the lock, so no concurrent modification
                    // - old_ptr came from output_to_raw during the original insert
                    // - defer_retire ensures readers complete before cleanup
                    let old_ptr: *mut u8 = leaf.take_leaf_value_ptr(slot);
                    if !old_ptr.is_null() {
                        unsafe {
                            guard.defer_retire(old_ptr, |ptr, _| {
                                S::cleanup_value_ptr(ptr);
                            });
                        }
                    }

                    // Clear any existing suffix for this slot
                    // SAFETY: We hold the lock
                    unsafe { leaf.clear_ksuf(slot, guard) };

                    // Install the layer pointer in the conflict slot
                    //
                    // NOTE: The original ikey remains unchanged (it's the shared prefix).
                    // We only change keylenx to indicate this is now a layer pointer,
                    // and set the pointer to the new layer chain.
                    leaf.set_keylenx(slot, LAYER_KEYLENX);
                    leaf.set_leaf_value_ptr(slot, layer_ptr);

                    // Release lock and increment entry count
                    // (new key was added to the layer, so count increases by 1)
                    drop(lock);
                    self.count.fetch_add(1, AtomicOrdering::Relaxed);

                    #[cfg(feature = "tracing")]
                    crate::tree::optimistic::LOCKED_INSERT_COUNT
                        .fetch_add(1, AtomicOrdering::Relaxed);
                    return Ok(None);
                }
            }
        }
    }

    /// Assign a value to a slot in a locked leaf.
    ///
    /// Handles both inline keys (0-8 bytes) and suffix keys (>8 bytes).
    /// For suffix keys, stores `keylenx = KSUF_KEYLENX` and allocates suffix storage.
    #[inline(always)]
    #[expect(clippy::unused_self, reason = "API Consistency")]
    #[expect(clippy::too_many_arguments, reason = "Internals")]
    fn assign_slot_generic(
        &self,
        leaf: &L,
        lock: &mut crate::nodeversion::LockGuard<'_>,
        slot: usize,
        key: &Key<'_>,
        value: &S::Output,
        guard: &LocalGuard<'_>,
    ) {
        let ikey: u64 = key.ikey();
        let value_ptr: *mut u8 = S::output_to_raw(value);

        // Mark insert dirty
        lock.mark_insert();

        // Store key data and value
        leaf.set_ikey(slot, ikey);
        leaf.set_leaf_value_ptr(slot, value_ptr);

        // Handle suffix keys correctly
        if key.has_suffix() {
            // Key has suffix bytes beyond the 8-byte ikey
            leaf.set_keylenx(slot, KSUF_KEYLENX);
            // SAFETY: We hold the lock, guard is from this tree's collector
            unsafe { leaf.assign_ksuf(slot, key.suffix(), guard) };
        } else {
            // Inline key (0-8 bytes total, no suffix)
            #[expect(clippy::cast_possible_truncation, reason = "current_len() <= 8")]
            let keylenx: u8 = key.current_len() as u8;
            leaf.set_keylenx(slot, keylenx);
        }
    }

    // ========================================================================
    // Generic Split Methods
    // ========================================================================

    /// Handle a leaf split when the leaf is full.
    ///
    /// This function implements the SPLIT-THEN-RETRY pattern:
    /// 1. Calculate split point
    /// 2. Allocate new leaf (pre-allocation before marking split)
    /// 3. Mark split in progress
    /// 4. Perform split (creates split-locked right sibling)
    /// 5. Link leaves (B-link)
    /// 6. Propagate to parent using TRUE hand-over-hand
    /// 7. Return Ok - caller retries insert
    ///
    /// # Arguments
    ///
    /// - `left_leaf_ptr`: Pointer to the leaf being split
    /// - `lock`: Lock guard (ownership transferred to propagation)
    /// - `logical_pos`: Insert position for split point calculation
    /// - `ikey`: Key being inserted
    /// - `guard`: Memory reclamation guard
    ///
    /// # Lock Protocol
    ///
    /// The left leaf's lock is maintained throughout propagation. This is the
    /// key difference from the previous (broken) implementation that dropped
    /// the lock before propagation.
    ///
    /// # Split-Locked Right Sibling
    ///
    /// The right sibling is created with a split-locked version in
    /// `split_into_preallocated()`. This is NOT done by the allocator.
    /// The split-locked version prevents other threads from operating on
    /// the right sibling until its parent pointer is set.
    ///
    /// # C++ Reference
    ///
    /// Matches `tcursor::make_split()` in `reference/masstree_split.hh:179-297`.
    #[cfg_attr(
    feature = "tracing",
    tracing::instrument(
        level = "debug",
        skip(self, lock, guard),
        fields(
            left_leaf = ?left_leaf_ptr,
            ikey = %format_args!("{:016x}", ikey)
        )
    )
)]
    #[inline]
    fn handle_leaf_split_generic(
        &self,
        left_leaf_ptr: *mut L,
        lock: crate::nodeversion::LockGuard<'_>,
        logical_pos: usize,
        ikey: u64,
        guard: &LocalGuard<'_>,
    ) -> Result<(), InsertError> {
        #[cfg(feature = "tracing")]
        let split_start = Instant::now();

        #[cfg(feature = "tracing")]
        tracing::info!(
            left_leaf_ptr = ?left_leaf_ptr,
            ikey = format_args!("{:016x}", ikey),
            logical_pos,
            "SPLIT_START: beginning leaf split"
        );

        let left_leaf: &L = unsafe { &*left_leaf_ptr };
        #[cfg(feature = "tracing")]
        crate::tree::optimistic::SPLIT_COUNT.fetch_add(1, AtomicOrdering::Relaxed);

        // Calculate split point
        let split_point = left_leaf
            .calculate_split_point(logical_pos, ikey)
            .ok_or(InsertError::SplitFailed)?;

        // =========================================================================
        //  CRITICAL: Capture root status BEFORE mark_split
        // =========================================================================
        //
        // SPLIT_UNLOCK_MASK clears ROOT_BIT on unlock. We must capture both
        // booleans separately BEFORE marking:
        //
        // - is_main_root: This leaf is THE main tree root (root_ptr points here)
        // - is_layer_root: This leaf is a layer root (null parent, root flag, NOT main)
        //
        // These are MUTUALLY EXCLUSIVE for handling:
        // - Main root: CAS on root_ptr to install new internode
        // - Layer root: NO CAS, just parent pointer updates

        let root_flag_set: bool = left_leaf.version().is_root();
        let parent_is_null: bool = left_leaf.parent().is_null();

        let is_main_root: bool = root_flag_set && {
            let current_root: *const L = self.root_ptr.load(AtomicOrdering::Acquire).cast();
            std::ptr::eq(current_root, left_leaf_ptr)
        };

        // Layer root: has root flag, null parent, but is NOT the main tree root
        let is_layer_root: bool = root_flag_set && parent_is_null && !is_main_root;

        #[cfg(feature = "tracing")]
        tracing::debug!(
            root_flag_set,
            parent_is_null,
            is_main_root,
            is_layer_root,
            "SPLIT_START: captured root status BEFORE mark_split"
        );

        // Allocate new leaf BEFORE marking split
        // This ensures allocation doesn't happen while we hold the split lock
        let new_leaf: Box<L> = L::new_boxed();

        // Mark split in progress (sets SPLITTING_BIT)
        let mut lock = lock;
        lock.mark_split();

        // Perform the split
        // NOTE: The right sibling receives a split-locked version from
        // split_into_preallocated() - this is NOT done by the allocator!
        // insert_target is ignored - we use SPLIT-THEN-RETRY pattern
        let (new_leaf_box, split_ikey, _insert_target) =
            unsafe { left_leaf.split_into_preallocated(split_point.pos, new_leaf, guard) };

        // Allocate new leaf in allocator (just tracks it, doesn't set version)
        let right_leaf_ptr: *mut L = self.allocator.alloc_leaf(new_leaf_box);

        #[cfg(feature = "tracing")]
        {
            let right_leaf: &L = unsafe { &*right_leaf_ptr };
            tracing::debug!(
                right_leaf_ptr = ?right_leaf_ptr,
                split_ikey = format_args!("{:016x}", split_ikey),
                right_is_split_locked = right_leaf.version().is_split_locked(),
                "SPLIT_CREATED: right sibling allocated (split-locked by split_into_preallocated)"
            );
        }

        // Link leaves in B-link order (while left is still locked)
        unsafe { left_leaf.link_sibling(right_leaf_ptr) };

        #[cfg(feature = "tracing")]
        tracing::debug!(
            left_leaf_ptr = ?left_leaf_ptr,
            right_leaf_ptr = ?right_leaf_ptr,
            "SPLIT_LINKED: B-link established"
        );

        // =========================================================================
        // TRUE HAND-OVER-HAND PROPAGATION
        // =========================================================================
        //
        // Pass ownership of the lock to Propagation::make_split_leaf.
        // The lock is maintained throughout propagation - this is the key
        // difference from the previous (broken) implementation.

        let result: Result<(), InsertError> = Propagation::make_split_leaf::<S, L, A>(
            &self.root_ptr,
            &self.allocator,
            left_leaf_ptr,
            lock, // Ownership transferred - lock maintained during propagation
            right_leaf_ptr,
            split_ikey,
            is_main_root,
            is_layer_root,
            guard,
        );

        #[cfg(feature = "tracing")]
        #[expect(clippy::cast_possible_truncation, reason = "logs")]
        {
            let total_elapsed = split_start.elapsed();
            if total_elapsed > std::time::Duration::from_millis(1) {
                tracing::warn!(
                    left_leaf_ptr = ?left_leaf_ptr,
                    right_leaf_ptr = ?right_leaf_ptr,
                    split_ikey = format_args!("{:016x}", split_ikey),
                    total_elapsed_us = total_elapsed.as_micros() as u64,
                    "SLOW_SPLIT: leaf split took >1ms"
                );
            }
        }

        result
    }

    /// Propagate a leaf split to the parent.
    ///
    /// # Arguments
    /// * `is_layer_root` - True if the left leaf was a layer root BEFORE the lock was dropped.
    ///   This must be captured before `drop(lock)` because `SPLIT_UNLOCK_MASK` clears `ROOT_BIT`.
    ///
    /// # Help-Along Protocol
    ///
    /// The right sibling (`right_leaf_ptr`) is created with a split-locked version
    /// ([`LOCK_BIT`] | [`SPLITTING_BIT`] set). This function unlocks it after setting its
    /// parent pointer. This prevents other threads from trying to split the right
    /// sibling while its parent is NULL.
    ///
    /// All exit paths must call `(*right_leaf_ptr).version().unlock_for_split()`.
    /// Try to find the child index for a given child pointer in an internode.
    ///
    /// Returns `Some(index)` if found, `None` if not found. Use this in retry loops
    /// where not finding the child is a valid transient state during concurrent splits.
    #[expect(clippy::unused_self, reason = "API Consistency")]
    fn try_find_child_index_generic(&self, parent: &L::Internode, child: *mut u8) -> Option<usize> {
        use crate::leaf_trait::TreeInternode;

        let nkeys = parent.nkeys();
        (0..=nkeys).find(|&i| parent.child(i) == child)
    }

    /// Find the child index for a given child pointer in an internode.
    /// Panics if not found.
    #[expect(clippy::expect_used, reason = "FATAL: Fail Fast")]
    fn find_child_index_generic(&self, parent: &L::Internode, child: *mut u8) -> usize {
        self.try_find_child_index_generic(parent, child)
            .expect("Child not found in parent internode")
    }
}

// =============================================================================
// Generic Layer Creation
// =============================================================================

impl<S, L, A> MassTreeGeneric<S, L, A>
where
    S: ValueSlot,
    S::Value: Send + Sync + 'static,
    S::Output: Send + Sync,
    L: LayerCapableLeaf<S>,
    A: NodeAllocatorGeneric<S, L>,
{
    /// Create a new layer for suffix conflict (generic version).
    ///
    /// Called when two keys share the same 8-byte ikey but have different suffixes.
    /// Creates a twig chain if needed, ending in a leaf with both keys.
    ///
    /// # Algorithm
    ///
    /// 1. Extract existing key's suffix and Arc value from conflict slot
    /// 2. Shift `new_key` past the matching ikey
    /// 3. While both keys have matching ikeys AND both have more bytes:
    ///    - Create intermediate "twig" layer node with just the matching ikey
    ///    - Chain twig nodes together via layer pointers
    /// 4. Create final leaf with both keys (now diverged)
    /// 5. Link twig chain to final leaf
    /// 6. Return head of chain (or final leaf if no chain)
    ///
    /// # Arguments
    ///
    /// * `parent_leaf` - The leaf containing the conflict slot
    /// * `conflict_slot` - Physical slot index with the existing key
    /// * `new_key` - The new key being inserted (will be mutated via shift)
    /// * `new_value` - Arc value for the new key
    /// * `guard` - Seize guard for memory reclamation
    ///
    /// # Returns
    ///
    /// Raw pointer to the head of the layer chain (either a twig or the final leaf).
    /// This pointer should be stored in the conflict slot with `LAYER_KEYLENX`.
    ///
    /// # Safety
    ///
    /// - Caller must hold the lock on `parent_leaf`
    /// - Caller must have called `lock.mark_insert()` before calling this
    /// - `guard` must come from this tree's collector
    unsafe fn create_layer_concurrent_generic(
        &self,
        parent_leaf: &L,
        conflict_slot: usize,
        new_key: &mut Key<'_>,
        new_value: S::Output,
        guard: &LocalGuard<'_>,
    ) -> *mut u8 {
        // =====================================================================
        // Step 1: Extract existing key's suffix and Arc value
        // =====================================================================

        // Get existing suffix (empty slice if no suffix stored)
        let existing_suffix: &[u8] = parent_leaf.ksuf(conflict_slot).unwrap_or(&[]);

        // Create a Key iterator from the existing suffix for comparison
        let mut existing_key: Key<'_> = Key::from_suffix(existing_suffix);

        // Clone the existing value from the conflict slot
        // INVARIANT: Conflict case means the slot contains a value, not a layer pointer.
        let existing_output: Option<S::Output> = parent_leaf.try_clone_output(conflict_slot);
        debug_assert!(
            existing_output.is_some(),
            "create_layer_concurrent_generic: conflict slot {} should contain a value, \
             not a layer pointer. keylenx={}",
            conflict_slot,
            parent_leaf.keylenx(conflict_slot)
        );

        // =====================================================================
        // Step 2: Shift new_key past the matching ikey
        // =====================================================================

        // The new_key's current ikey matched the conflict slot's ikey.
        // If new_key has more bytes (suffix), shift to the next 8-byte chunk.
        if new_key.has_suffix() {
            new_key.shift();
        }

        // =====================================================================
        // Step 3: Compare keys to determine twig chain depth
        // =====================================================================

        // Compare the next ikeys of both keys
        let mut cmp: Ordering = existing_key.compare(new_key.ikey(), new_key.current_len());

        // =====================================================================
        // Step 4: Create twig chain while ikeys match AND both have more bytes
        // =====================================================================

        // Twig chain head (first twig node, returned to caller)
        let mut twig_head: Option<*mut L> = None;
        // Twig chain tail (last twig node, where we link the next node)
        let mut twig_tail: *mut L = std::ptr::null_mut();

        while cmp == Ordering::Equal && existing_key.has_suffix() && new_key.has_suffix() {
            // Both keys have the same ikey at this level AND both have more bytes.
            // Create an intermediate twig node that just holds this matching ikey.

            // Allocate new twig node configured as layer root
            let twig: Box<L> = L::new_layer_root_boxed();
            let twig_ptr: *mut L = self.allocator.alloc_leaf(twig);

            // Initialize twig with the matching ikey in slot 0
            // SAFETY: twig_ptr is valid, we just allocated it
            unsafe {
                (*twig_ptr).set_ikey(0, existing_key.ikey());
                // Twig has exactly 1 entry (the matching ikey, will point to next layer)
                (*twig_ptr).set_permutation(<L::Perm as TreePermutation>::make_sorted(1));
            }

            // Link to previous twig in chain (if any)
            if twig_head.is_some() {
                // Previous twig's slot 0 now points to this twig as a layer
                // SAFETY: twig_tail is valid from previous iteration
                unsafe {
                    (*twig_tail).set_keylenx(0, LAYER_KEYLENX);
                    (*twig_tail).set_leaf_value_ptr(0, twig_ptr.cast::<u8>());
                }
            } else {
                // First twig becomes the head of the chain
                twig_head = Some(twig_ptr);
            }
            twig_tail = twig_ptr;

            // Shift both keys to compare the next 8-byte chunk
            existing_key.shift();
            new_key.shift();
            cmp = existing_key.compare(new_key.ikey(), new_key.current_len());
        }

        // =====================================================================
        // Step 5: Create final leaf with both keys (now diverged or one is prefix)
        // =====================================================================

        let final_leaf: Box<L> = L::new_layer_root_boxed();
        let final_ptr: *mut L = self.allocator.alloc_leaf(final_leaf);

        // Assign both entries to the final leaf in sorted order
        // SAFETY: final_ptr is valid (just allocated), guard is from caller
        unsafe {
            self.assign_final_layer_entries(
                final_ptr,
                &existing_key,
                existing_output,
                new_key,
                Some(new_value),
                cmp,
                guard,
            );
        }

        // =====================================================================
        // Step 6: Link twig chain to final leaf
        // =====================================================================

        twig_head.map_or_else(
            || final_ptr.cast::<u8>(),
            |head| {
                // Link last twig to the final leaf
                // SAFETY: twig_tail is valid (we have at least one twig since head is Some)
                unsafe {
                    (*twig_tail).set_keylenx(0, LAYER_KEYLENX);
                    (*twig_tail).set_leaf_value_ptr(0, final_ptr.cast::<u8>());
                }
                // Return head of twig chain
                head.cast::<u8>()
            },
        )
    }

    /// Assign two entries to the final layer leaf in sorted order.
    ///
    /// The entries are ordered by:
    /// 1. ikey comparison (lexicographic via u64 big-endian)
    /// 2. If ikeys equal: shorter key first (prefix before extension)
    ///
    /// # Safety
    ///
    /// - `final_ptr` must be valid and point to an empty leaf
    /// - `guard` must come from this tree's collector
    /// - Caller must ensure no concurrent access to `final_ptr`
    #[expect(clippy::too_many_arguments, reason = "Internal helper")]
    #[expect(clippy::unused_self, reason = "API Consistency")]
    unsafe fn assign_final_layer_entries(
        &self,
        final_ptr: *mut L,
        existing_key: &Key<'_>,
        existing_output: Option<S::Output>,
        new_key: &Key<'_>,
        new_arc: Option<S::Output>,
        cmp: Ordering,
        guard: &LocalGuard<'_>,
    ) {
        // SAFETY: final_ptr is valid per caller contract
        let final_leaf: &L = unsafe { &*final_ptr };

        match cmp {
            Ordering::Less => {
                // existing_key.ikey() < new_key.ikey()
                // existing goes in slot 0, new goes in slot 1
                // SAFETY: guard requirement passed through from caller
                unsafe {
                    final_leaf.assign_from_key_arc(0, existing_key, existing_output, guard);
                    final_leaf.assign_from_key_arc(1, new_key, new_arc, guard);
                }
            }
            Ordering::Greater => {
                // new_key.ikey() < existing_key.ikey()
                // new goes in slot 0, existing goes in slot 1
                // SAFETY: guard requirement passed through from caller
                unsafe {
                    final_leaf.assign_from_key_arc(0, new_key, new_arc, guard);
                    final_leaf.assign_from_key_arc(1, existing_key, existing_output, guard);
                }
            }
            Ordering::Equal => {
                // Keys have same ikey at this level.
                // This happens when one key is a prefix of the other.
                // Convention: shorter key first (prefix before extension).
                if existing_key.current_len() <= new_key.current_len() {
                    // existing is shorter or equal length -> existing first
                    // SAFETY: guard requirement passed through from caller
                    unsafe {
                        final_leaf.assign_from_key_arc(0, existing_key, existing_output, guard);
                        final_leaf.assign_from_key_arc(1, new_key, new_arc, guard);
                    }
                } else {
                    // new is shorter -> new first
                    // SAFETY: guard requirement passed through from caller
                    unsafe {
                        final_leaf.assign_from_key_arc(0, new_key, new_arc, guard);
                        final_leaf.assign_from_key_arc(1, existing_key, existing_output, guard);
                    }
                }
            }
        }

        // Set permutation: final leaf now has exactly 2 entries in slots 0 and 1
        final_leaf.set_permutation(<L::Perm as TreePermutation>::make_sorted(2));
    }
}
