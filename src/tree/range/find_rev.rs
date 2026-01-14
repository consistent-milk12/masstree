// ============================================================================
//  Reverse Scan Helper Types
// ============================================================================

use crate::{TreeInternode, ksearch::upper_bound_internode_generic, leaf15::KSUF_KEYLENX};
use std::cmp::Ordering;

use seize::LocalGuard;

use crate::{
    TreeLeafNode, TreePermutation, ValueSlot,
    key::IKEY_SIZE,
    leaf15::LAYER_KEYLENX,
    nodeversion::NodeVersion,
    prefetch::prefetch_read,
    tree::range::{
        cursor_key::CursorKey,
        helper::ReverseScanHelper,
        scan_state::{BackStackElement, LayerContext, LayerStack, ScanSnapshot, ScanStateBack},
    },
};

/// Result of attempting to find initial position in a single layer.
#[expect(dead_code, reason = "Reserved for future use")]
enum InitialPositionResult<S: ValueSlot> {
    /// Found a value to emit.
    Emit(ScanSnapshot<S>),

    /// Need to descend into sublayer at the given root.
    LayerDescent(*const u8),

    /// No match at current position, retreat to find previous.
    FindPrev,

    /// Layer is empty or exhausted, ascend.
    Up,

    /// Version conflict, retry from root.
    Retry,
}

enum EmitResult<S: ValueSlot> {
    /// Successfully prepared snapshot for emission.
    Emit(ScanSnapshot<S>),

    /// Entry doesn't match criteria (wrong suffix, null value, etc.).
    /// Caller should continue searching.
    NoMatch,

    /// Version changed during read, need retry from root.
    VersionChanged,
}

/// Reverse scan operations for Masstree range iteration.
///
/// This struct provides static methods for reverse-direction traversal
/// during range scans. It implements the `DoubleEndedIterator` support
/// functions for `RangeIter`.
///
/// # Key Functions
///
/// - `find_initial_reverse`: Position scan at end bound
/// - `find_prev`: Main workhorse for reverse iteration
/// - `advance_to_prev_leaf`: O(1) leaf-to-leaf retreat
/// - `reposition_back`: Version conflict recovery
#[derive(Debug)]
pub struct ReverseScan;

impl ReverseScan {
    /// Find the initial position for a reverse range scan.
    ///
    /// Positions the scan at the correct leaf and slot for the end bound.
    /// Uses an iterative loop instead of recursion for layer descent.
    ///
    /// # Algorithm
    ///
    /// 1. Loop through layers starting from root
    /// 2. For each layer:
    ///    - Reach target leaf via `reach_leaf_for_scan`
    ///    - Handle concurrent inserts via `stable_reverse`
    ///    - Find position via `lower_reverse`
    ///    - If layer pointer: setup descent and continue loop
    ///    - If value: try to emit
    ///    - Otherwise: return `FindPrev` to retreat
    ///
    /// # Performance
    ///
    /// - Iterative (no stack frames for layer descent)
    /// - Early null check before `stable_reverse`
    /// - Single version acquisition per layer
    ///
    /// # C++ Reference
    ///
    /// Corresponds to `scanstackelt::find_initial` in `masstree_scan.hh:130-188`
    /// with `reverse_scan_helper` semantics.
    #[expect(clippy::too_many_arguments)]
    pub fn find_initial_reverse<L, S>(
        root: *const u8,
        stack: &mut BackStackElement<L, S>,
        cursor_key: &mut CursorKey,
        layer_stack: &mut LayerStack<L>,
        emit_equal: bool,
        helper: &mut ReverseScanHelper,
        guard: &LocalGuard<'_>,
    ) -> (ScanStateBack, Option<ScanSnapshot<S>>)
    where
        L: TreeLeafNode<S>,
        S: ValueSlot,
        S::Output: Clone,
    {
        let mut current_root: *const u8 = root;

        // Iterative layer descent loop
        loop {
            stack.set_root(current_root);

            // Reach target leaf
            let mut leaf_ptr: *mut L =
                Self::reach_leaf_for_scan::<L, S>(current_root, cursor_key, guard);

            // CRITICAL: Check null BEFORE calling stable_reverse
            if leaf_ptr.is_null() {
                return (ScanStateBack::Up, None);
            }

            // Handle concurrent inserts - may follow forward chain
            let version: u32 = ReverseScanHelper::stable_reverse(&mut leaf_ptr, cursor_key, guard);
            stack.set_leaf(leaf_ptr);

            // Fast path: check deleted version
            if NodeVersion::is_deleted_version(version) {
                return (ScanStateBack::Retry, None);
            }

            // SAFETY: leaf_ptr is valid (null checked above, stable_reverse ensures validity)
            let leaf: &L = unsafe { &*leaf_ptr };
            let perm: L::Perm = leaf.permutation();
            let size: usize = perm.size();

            // Fast path: empty leaf
            if size == 0 {
                return (ScanStateBack::Up, None);
            }

            // Try to find initial position in this layer
            match Self::try_initial_position_reverse(
                leaf,
                perm,
                size,
                version,
                current_root,
                leaf_ptr,
                stack,
                cursor_key,
                layer_stack,
                emit_equal,
                helper,
            ) {
                InitialPositionResult::Emit(snapshot) => {
                    return (ScanStateBack::Emit, Some(snapshot));
                }

                InitialPositionResult::LayerDescent(layer_ptr) => {
                    // Continue loop with new layer root
                    current_root = layer_ptr;
                }

                InitialPositionResult::FindPrev => {
                    // Version check before returning
                    if leaf.version().has_changed(version) {
                        return (ScanStateBack::Retry, None);
                    }
                    return (ScanStateBack::FindPrev, None);
                }

                InitialPositionResult::Up => {
                    return (ScanStateBack::Up, None);
                }

                InitialPositionResult::Retry => {
                    return (ScanStateBack::Retry, None);
                }
            }
        }
    }

    /// Attempt to find initial position within a single layer.
    ///
    /// Handles the core logic of position finding, slot classification,
    /// and deciding whether to emit, descend, or retreat.
    ///
    /// # Returns
    ///
    /// - `Emit(snapshot)`: Found a value to emit
    /// - `LayerDescent(ptr)`: Need to descend into sublayer
    /// - `FindPrev`: No match, caller should retreat
    /// - `Up`: Layer exhausted
    /// - `Retry`: Version conflict
    #[expect(clippy::too_many_arguments, reason = "Internal helper")]
    fn try_initial_position_reverse<L, S>(
        leaf: &L,
        perm: L::Perm,
        size: usize,
        version: u32,
        current_root: *const u8,
        leaf_ptr: *mut L,
        stack: &mut BackStackElement<L, S>,
        cursor_key: &mut CursorKey,
        layer_stack: &mut LayerStack<L>,
        emit_equal: bool,
        helper: &mut ReverseScanHelper,
    ) -> InitialPositionResult<S>
    where
        L: TreeLeafNode<S>,
        S: ValueSlot,
        S::Output: Clone,
    {
        // Find position using reverse helper
        let ki: isize = helper.lower_reverse(cursor_key, leaf, &perm);

        // Check if position is valid: ki must be in [0, size)
        if !(ki >= 0 && ki.cast_unsigned() < size) {
            // Position is -1 or beyond size, need to retreat
            stack.update_state(version, perm, ki);
            return InitialPositionResult::FindPrev;
        }

        let slot: usize = perm.get(ki.cast_unsigned());
        let keylenx: u8 = leaf.keylenx(slot);
        let slot_ikey: u64 = leaf.ikey_relaxed(slot);

        // Handle layer pointer - must descend
        if keylenx >= LAYER_KEYLENX {
            return Self::handle_layer_descent_reverse(
                current_root,
                leaf_ptr,
                slot,
                slot_ikey,
                version,
                perm,
                ki,
                stack,
                cursor_key,
                layer_stack,
                helper,
                leaf,
            );
        }

        // Try to emit this slot
        match Self::try_emit_slot_reverse(
            leaf, slot, slot_ikey, keylenx, version, &perm, ki, cursor_key, stack, emit_equal,
            helper,
        ) {
            EmitResult::Emit(snapshot) => InitialPositionResult::Emit(snapshot),

            EmitResult::NoMatch => {
                // Entry doesn't match, update position and retreat
                stack.update_state(version, perm, ki - 1);
                InitialPositionResult::FindPrev
            }

            EmitResult::VersionChanged => {
                // Version conflict - must retry from root
                InitialPositionResult::Retry
            }
        }
    }

    /// Set up layer descent for reverse scan.
    ///
    /// Pushes parent context to layer stack, updates cursor, and returns
    /// the sublayer root for the caller to continue the loop.
    ///
    /// # Critical: Initial Descent Distinction
    ///
    /// - If cursor has suffix: use `shift()` to follow user's end bound
    /// - If no suffix: use `shift_clear_reverse()` to scan entire sublayer from max
    #[expect(clippy::too_many_arguments, reason = "Internal helper")]
    fn handle_layer_descent_reverse<L, S>(
        current_root: *const u8,
        leaf_ptr: *mut L,
        slot: usize,
        slot_ikey: u64,
        version: u32,
        perm: L::Perm,
        ki: isize,
        stack: &mut BackStackElement<L, S>,
        cursor_key: &mut CursorKey,
        layer_stack: &mut LayerStack<L>,
        helper: &mut ReverseScanHelper,
        leaf: &L,
    ) -> InitialPositionResult<S>
    where
        L: TreeLeafNode<S>,
        S: ValueSlot,
        S::Output: Clone,
    {
        // Push parent context for return
        layer_stack.push(LayerContext::new(current_root, leaf_ptr));

        cursor_key.assign_store_ikey(slot_ikey);
        let layer_ptr: *mut u8 = leaf.leaf_value_ptr(slot);

        // Prefetch layer root before descending (hide memory latency)
        prefetch_read(layer_ptr);

        // Update position to before layer pointer for when we return
        stack.update_state(version, perm, ki - 1);

        // CRITICAL: Initial descent distinction (from C++ reference)
        // - If cursor has suffix: use shift() to follow user's end bound
        // - If no suffix: use shift_clear_reverse() to scan entire sublayer from max
        if cursor_key.has_suffix() {
            cursor_key.shift();
        } else {
            cursor_key.shift_clear_reverse();
            helper.upper_bound = true;
        }

        // Return layer pointer for iterative descent
        InitialPositionResult::LayerDescent(layer_ptr.cast_const())
    }

    /// Try to emit a slot value for reverse scan.
    ///
    /// Handles both suffix keys (`KSUF_KEYLENX`) and inline keys (0-8).
    /// Returns a three-way result distinguishing success, no-match, and version conflict.
    ///
    /// # Arguments
    ///
    /// - `leaf`: The leaf node
    /// - `slot`: Physical slot index
    /// - `slot_ikey`: The slot's ikey (already loaded)
    /// - `keylenx`: The slot's keylenx (already loaded)
    /// - `version`: Version snapshot for validation
    /// - `perm`: Permutation snapshot (borrowed for stack update)
    /// - `ki`: Logical position
    /// - `cursor_key`: Cursor to update on emit
    /// - `stack`: Stack to update on emit
    /// - `emit_equal`: Whether to emit on exact match
    /// - `helper`: Reverse scan helper (for `upper_bound` check)
    #[inline]
    #[expect(clippy::too_many_arguments, reason = "Internal helper")]
    fn try_emit_slot_reverse<L, S>(
        leaf: &L,
        slot: usize,
        slot_ikey: u64,
        keylenx: u8,
        version: u32,
        perm: &L::Perm,
        ki: isize,
        cursor_key: &mut CursorKey,
        stack: &mut BackStackElement<L, S>,
        emit_equal: bool,
        helper: &ReverseScanHelper,
    ) -> EmitResult<S>
    where
        L: TreeLeafNode<S>,
        S: ValueSlot,
        S::Output: Clone,
    {
        // Handle suffix keys
        if keylenx == KSUF_KEYLENX {
            return Self::try_emit_suffix_slot_reverse(
                leaf, slot, slot_ikey, version, perm, ki, cursor_key, stack, emit_equal,
            );
        }

        // Handle inline keys - only emit if emit_equal or at upper_bound
        if !emit_equal && !helper.upper_bound {
            return EmitResult::NoMatch;
        }

        let value_ptr: *mut u8 = leaf.leaf_value_ptr(slot);
        if value_ptr.is_null() {
            return EmitResult::NoMatch;
        }

        // Version check before read
        if leaf.version().has_changed(version) {
            return EmitResult::VersionChanged;
        }

        // SAFETY: Version validated, pointer is valid
        let output: S::Output = unsafe { S::output_from_raw(value_ptr) };

        #[expect(clippy::cast_possible_truncation, reason = "Known const")]
        let key_len: usize = std::cmp::min(keylenx, IKEY_SIZE as u8) as usize;

        cursor_key.assign_store_ikey(slot_ikey);
        cursor_key.assign_store_length(key_len);

        stack.update_state(version, *perm, ki - 1);

        EmitResult::Emit(ScanSnapshot {
            value: output,
            key_len,
        })
    }

    /// Try to emit a suffix slot for reverse scan.
    ///
    /// Separated from `try_emit_slot_reverse` for clarity and to avoid
    /// nested conditionals.
    #[inline]
    #[expect(clippy::too_many_arguments, reason = "Internal helper")]
    fn try_emit_suffix_slot_reverse<L, S>(
        leaf: &L,
        slot: usize,
        slot_ikey: u64,
        version: u32,
        perm: &L::Perm,
        ki: isize,
        cursor_key: &mut CursorKey,
        stack: &mut BackStackElement<L, S>,
        emit_equal: bool,
    ) -> EmitResult<S>
    where
        L: TreeLeafNode<S>,
        S: ValueSlot,
        S::Output: Clone,
    {
        let Some(stored_suffix) = leaf.ksuf(slot) else {
            return EmitResult::NoMatch;
        };

        let cmp: Ordering = stored_suffix.cmp(cursor_key.suffix());

        if !ReverseScanHelper::initial_ksuf_match_reverse(cmp, emit_equal) {
            return EmitResult::NoMatch;
        }

        let value_ptr: *mut u8 = leaf.leaf_value_ptr(slot);
        if value_ptr.is_null() {
            return EmitResult::NoMatch;
        }

        // Version check before read
        if leaf.version().has_changed(version) {
            return EmitResult::VersionChanged;
        }

        // SAFETY: Version validated, pointer is valid
        let output: S::Output = unsafe { S::output_from_raw(value_ptr) };
        let key_len: usize = IKEY_SIZE + stored_suffix.len();

        cursor_key.assign_store_ikey(slot_ikey);
        cursor_key.assign_store_suffix(stored_suffix);
        cursor_key.assign_store_length(key_len);

        stack.update_state(version, *perm, ki - 1);

        EmitResult::Emit(ScanSnapshot {
            value: output,
            key_len,
        })
    }

    /// Traverse from layer root to target leaf.
    ///
    /// Similar to `reach_leaf_concurrent_generic` but uses cursor key's ikey.
    ///
    /// # Note
    ///
    /// The `guard` parameter ensures pointer validity through lifetime binding.
    #[inline]
    fn reach_leaf_for_scan<L, S>(
        start: *const u8,
        cursor_key: &CursorKey,
        _guard: &LocalGuard<'_>,
    ) -> *mut L
    where
        L: TreeLeafNode<S>,
        S: ValueSlot,
    {
        if start.is_null() {
            return std::ptr::null_mut();
        }

        let target_ikey: u64 = cursor_key.current_ikey();
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
                upper_bound_internode_generic::<L::Internode>(target_ikey, inode);
            let child: *mut u8 = inode.child(child_idx);

            // Prefetch child node
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
    //  find_prev - Main Reverse Scan Workhorse
    // ========================================================================

    /// Find the previous entry for reverse iteration.
    ///
    /// This is the main workhorse for reverse scanning, called repeatedly
    /// after `find_initial_reverse` positions the scan.
    ///
    /// # Algorithm
    ///
    /// 1. Fast path: check if leaf is exhausted (`ki < 0`)
    /// 2. Validate version hasn't changed
    /// 3. Process current slot (layer pointer, value, or skip)
    /// 4. Update position for next iteration
    ///
    /// # Performance
    ///
    /// - O(1) per slot processed
    /// - Early exits ordered by cost (cheapest first)
    /// - Prefetch optimization for sequential access
    /// - Duplicate check skipped in normal iteration (only after Retry)
    ///
    /// # C++ Reference
    ///
    /// Corresponds to `scanstackelt::find_next` with `reverse_scan_helper`
    /// in `masstree_scan.hh:230-280`.
    #[inline]
    pub fn find_prev<L, S>(
        stack: &mut BackStackElement<L, S>,
        cursor_key: &mut CursorKey,
        layer_stack: &mut LayerStack<L>,
        helper: &mut ReverseScanHelper,
        guard: &LocalGuard<'_>,
    ) -> (ScanStateBack, Option<ScanSnapshot<S>>)
    where
        L: TreeLeafNode<S>,
        S: ValueSlot,
        S::Output: Clone,
    {
        // OPTIMIZATION: Skip duplicate check in normal reverse iteration.
        // Duplicates can only occur after Retry states (version conflict).
        Self::find_prev_inner(stack, cursor_key, layer_stack, helper, guard, false)
    }

    /// Find the previous entry with duplicate checking enabled.
    ///
    /// Called after a Retry state to skip already-emitted entries.
    #[inline]
    pub fn find_prev_with_duplicate_check<L, S>(
        stack: &mut BackStackElement<L, S>,
        cursor_key: &mut CursorKey,
        layer_stack: &mut LayerStack<L>,
        helper: &mut ReverseScanHelper,
        guard: &LocalGuard<'_>,
    ) -> (ScanStateBack, Option<ScanSnapshot<S>>)
    where
        L: TreeLeafNode<S>,
        S: ValueSlot,
        S::Output: Clone,
    {
        Self::find_prev_inner(stack, cursor_key, layer_stack, helper, guard, true)
    }

    /// Inner implementation of `find_prev` with configurable duplicate checking.
    #[inline]
    fn find_prev_inner<L, S>(
        stack: &mut BackStackElement<L, S>,
        cursor_key: &mut CursorKey,
        layer_stack: &mut LayerStack<L>,
        helper: &mut ReverseScanHelper,
        guard: &LocalGuard<'_>,
        needs_duplicate_check: bool,
    ) -> (ScanStateBack, Option<ScanSnapshot<S>>)
    where
        L: TreeLeafNode<S>,
        S: ValueSlot,
        S::Output: Clone,
    {
        // Fast path: null leaf means we need to go up
        let leaf_ptr: *mut L = stack.get_leaf_ptr();
        if leaf_ptr.is_null() {
            return (ScanStateBack::Up, None);
        }

        let ki: isize = stack.get_ki();

        // Fast path: leaf exhausted (ki went negative)
        // Check BEFORE version to avoid unnecessary atomic load
        if ki < 0 {
            return Self::advance_to_prev_leaf(stack, cursor_key, helper, guard);
        }

        // SAFETY: leaf_ptr is valid (null checked above)
        let leaf: &L = unsafe { &*leaf_ptr };
        let version: u32 = stack.get_version();

        // Version check - if changed, reposition
        if leaf.version().has_changed(version) {
            return Self::reposition_back(stack, cursor_key, helper, guard);
        }

        let perm: L::Perm = *stack.get_perm_ref();
        let size: usize = perm.size();

        // Defensive check: ki might be >= size due to concurrent deletion
        if ki.unsigned_abs() >= size {
            return Self::advance_to_prev_leaf(stack, cursor_key, helper, guard);
        }

        // Process the current slot
        Self::process_slot_reverse(
            leaf,
            leaf_ptr,
            perm,
            size,
            version,
            ki,
            stack,
            cursor_key,
            layer_stack,
            helper,
            needs_duplicate_check,
        )
    }

    /// Process a single slot during reverse scan.
    ///
    /// Handles slot classification (layer pointer vs value) and emission logic.
    /// This is the hot inner loop of reverse scanning.
    #[inline]
    #[expect(clippy::too_many_arguments, reason = "Internal helper")]
    fn process_slot_reverse<L, S>(
        leaf: &L,
        leaf_ptr: *mut L,
        perm: L::Perm,
        _size: usize,
        version: u32,
        ki: isize,
        stack: &mut BackStackElement<L, S>,
        cursor_key: &mut CursorKey,
        layer_stack: &mut LayerStack<L>,
        helper: &mut ReverseScanHelper,
        needs_duplicate_check: bool,
    ) -> (ScanStateBack, Option<ScanSnapshot<S>>)
    where
        L: TreeLeafNode<S>,
        S: ValueSlot,
        S::Output: Clone,
    {
        let slot: usize = perm.get(ki.unsigned_abs());
        let slot_ikey: u64 = leaf.ikey_relaxed(slot);
        let keylenx: u8 = leaf.keylenx(slot);

        // Prefetch next slot's data to hide memory latency
        // For reverse scan, "next" is ki-1
        if ki > 0 {
            let next_slot: usize = perm.get((ki - 1).unsigned_abs());
            // Prefetch the value slot area
            prefetch_read(leaf.leaf_value_ptr(next_slot));
        }

        // Check for duplicate only when needed (after Retry)
        // OPTIMIZATION: In normal reverse iteration, stack.prev() already advances
        // past the previous entry, so duplicates can't occur
        if needs_duplicate_check
            && ReverseScanHelper::is_duplicate_reverse(
                cursor_key,
                slot_ikey,
                keylenx,
                helper.upper_bound,
            )
        {
            stack.set_ki(ReverseScanHelper::prev(ki));
            return (ScanStateBack::FindPrev, None);
        }

        // Handle layer pointer
        if keylenx >= LAYER_KEYLENX {
            return Self::handle_layer_pointer_reverse(
                leaf,
                leaf_ptr,
                slot,
                slot_ikey,
                version,
                perm,
                ki,
                stack,
                cursor_key,
                layer_stack,
            );
        }

        // Try to emit this slot's value (returns tuple directly, no intermediate enum)
        Self::try_emit_value_reverse(
            leaf, slot, slot_ikey, keylenx, version, &perm, ki, cursor_key, stack, helper,
        )
    }

    /// Handle a layer pointer during reverse scan (`find_prev` path).
    ///
    /// Pushes parent context and prepares for sublayer descent.
    /// Unlike `handle_layer_descent_reverse` (used by `find_initial`), this
    /// always uses `shift_clear_reverse` since we're doing scan-discovered descent.
    #[inline]
    #[expect(clippy::too_many_arguments, reason = "Internal helper")]
    fn handle_layer_pointer_reverse<L, S>(
        leaf: &L,
        leaf_ptr: *mut L,
        slot: usize,
        slot_ikey: u64,
        version: u32,
        perm: L::Perm,
        ki: isize,
        stack: &mut BackStackElement<L, S>,
        cursor_key: &mut CursorKey,
        layer_stack: &mut LayerStack<L>,
    ) -> (ScanStateBack, Option<ScanSnapshot<S>>)
    where
        L: TreeLeafNode<S>,
        S: ValueSlot,
        S::Output: Clone,
    {
        // Push current context to layer stack for return
        layer_stack.push(LayerContext::new(stack.get_root(), leaf_ptr));

        // Update cursor with layer pointer's ikey
        cursor_key.assign_store_ikey(slot_ikey);

        // Get layer root and prefetch
        let layer_ptr: *mut u8 = leaf.leaf_value_ptr(slot);
        prefetch_read(layer_ptr);

        // Update stack for sublayer (ki-1 for when we return)
        stack.set_root(layer_ptr.cast_const());
        stack.update_state(version, perm, ReverseScanHelper::prev(ki));

        // Return Down state - iterator will call handle_down_back then find_initial_reverse
        (ScanStateBack::Down, None)
    }

    /// Try to emit a value slot during reverse scan (`find_prev` path).
    ///
    /// Handles both suffix keys and inline keys with proper version validation.
    /// Returns the state tuple directly to avoid intermediate enum allocation.
    ///
    /// # Critical: Calls `mark_key_complete()`
    ///
    /// This function calls `helper.mark_key_complete()` on successful emission
    /// to clear the `upper_bound` flag. This is required for correct duplicate
    /// filtering after version-triggered repositioning.
    #[inline]
    #[expect(clippy::too_many_arguments, reason = "Internal helper")]
    fn try_emit_value_reverse<L, S>(
        leaf: &L,
        slot: usize,
        slot_ikey: u64,
        keylenx: u8,
        version: u32,
        perm: &L::Perm,
        ki: isize,
        cursor_key: &mut CursorKey,
        stack: &mut BackStackElement<L, S>,
        helper: &mut ReverseScanHelper,
    ) -> (ScanStateBack, Option<ScanSnapshot<S>>)
    where
        L: TreeLeafNode<S>,
        S: ValueSlot,
        S::Output: Clone,
    {
        // Get value pointer first (before version check to pipeline loads)
        let value_ptr: *mut u8 = leaf.leaf_value_ptr(slot);

        if value_ptr.is_null() {
            // Null value pointer - skip to previous slot
            stack.set_ki(ReverseScanHelper::prev(ki));
            return (ScanStateBack::FindPrev, None);
        }

        // Version check BEFORE reading value
        if leaf.version().has_changed(version) {
            return (ScanStateBack::Retry, None);
        }

        // Build key from slot data
        cursor_key.assign_store_ikey(slot_ikey);

        // CRITICAL: Clear upper_bound after successful emission
        helper.mark_key_complete();

        // Calculate key length and handle suffix
        let key_len: usize = if keylenx == KSUF_KEYLENX {
            leaf.ksuf(slot).map_or(IKEY_SIZE, |suffix| {
                cursor_key.assign_store_suffix(suffix);
                IKEY_SIZE + suffix.len()
            })
        } else {
            #[expect(clippy::cast_possible_truncation, reason = "Known const")]
            let len: usize = std::cmp::min(keylenx, IKEY_SIZE as u8) as usize;
            len
        };

        cursor_key.assign_store_length(key_len);

        // SAFETY: value_ptr is valid, version validated
        let output: S::Output = unsafe { S::output_from_raw(value_ptr) };

        // Update position for next iteration
        stack.update_state(version, *perm, ReverseScanHelper::prev(ki));

        (
            ScanStateBack::Emit,
            Some(ScanSnapshot {
                value: output,
                key_len,
            }),
        )
    }

    // ========================================================================
    //  advance_to_prev_leaf - O(1) Local Leaf Retreat
    // ========================================================================

    /// Advance to the previous leaf in the B-link chain.
    ///
    /// Called when the current leaf is exhausted (`ki < 0`).
    ///
    /// # Algorithm
    ///
    /// 1. Get `ikey_bound()` from current leaf for cursor repositioning
    /// 2. Follow `prev_` pointer to previous leaf
    /// 3. If null: layer exhausted, return `Up`
    /// 4. Otherwise: prefetch, get stable version, find position
    ///
    /// # Performance
    ///
    /// - **O(1)** per leaf (direct pointer follow, no root traversal)
    /// - Prefetch hides memory latency (~300 cycles → ~50 cycles)
    /// - Single version acquisition combines stable + position
    ///
    /// This is the difference between:
    /// - O(height) per leaf (3000 internode accesses for 1000 leaves, height 3)
    /// - O(1) per leaf (1000 prev_ pointer follows)
    ///
    /// # C++ Reference
    ///
    /// ```cpp
    /// // masstree_scan.hh:240-255 with reverse_scan_helper
    /// n_ = helper.advance(n_, ka);  // Sets ka to ikey_bound, returns prev_
    /// if (!n_) return scan_up;
    /// n_->prefetch();
    /// v_ = helper.stable(n_, ka);
    /// perm_ = n_->permutation();
    /// ki_ = helper.lower(ka, this);
    /// ```
    #[inline]
    fn advance_to_prev_leaf<L, S>(
        stack: &mut BackStackElement<L, S>,
        cursor_key: &mut CursorKey,
        helper: &ReverseScanHelper,
        guard: &LocalGuard<'_>,
    ) -> (ScanStateBack, Option<ScanSnapshot<S>>)
    where
        L: TreeLeafNode<S>,
        S: ValueSlot,
        S::Output: Clone,
    {
        let current_ptr: *mut L = stack.get_leaf_ptr();

        // SAFETY: current_ptr was validated before calling this function
        // (find_prev checks null and version before reaching here via ki < 0)
        let current_leaf: &L = unsafe { &*current_ptr };

        // Step 1: Update cursor key with current leaf's ikey_bound
        // This is critical for stable_reverse to find the correct position
        // C++: ka.assign_store_ikey(n->ikey_bound()); ka.assign_store_length(0);
        let ikey_bound: u64 = current_leaf.ikey_bound();
        cursor_key.assign_store_ikey(ikey_bound);
        cursor_key.assign_store_length(0);

        // Step 2: Get previous leaf pointer (O(1) - just follows prev_ pointer)
        let prev_ptr: *mut L = ReverseScanHelper::retreat(current_leaf);

        // Step 3: Check for layer exhaustion
        if prev_ptr.is_null() {
            // No previous leaf - layer is exhausted
            // Clear leaf to signal exhaustion
            stack.set_leaf(std::ptr::null_mut());
            return (ScanStateBack::Up, None);
        }

        // Step 4: Prefetch previous leaf before accessing
        // This hides the memory latency of following the pointer
        prefetch_read(prev_ptr.cast::<u8>());

        // Step 5: Set up stack with new leaf
        stack.set_leaf(prev_ptr);

        // Step 6: Get stable version (may follow forward chain for concurrent inserts)
        let mut leaf_ptr: *mut L = prev_ptr;
        let version: u32 = ReverseScanHelper::stable_reverse(&mut leaf_ptr, cursor_key, guard);

        // Step 7: Update stack if stable_reverse followed forward chain
        if leaf_ptr != prev_ptr {
            stack.set_leaf(leaf_ptr);
        }

        // Step 8: Get permutation and set position to LAST slot
        // SAFETY: leaf_ptr is valid (stable_reverse ensures it)
        let leaf: &L = unsafe { &*leaf_ptr };
        let perm: L::Perm = leaf.permutation();

        // CRITICAL FIX: When moving to previous leaf via prev_ pointer,
        // we always start from the LAST slot (size - 1), not from lower_reverse.
        // Using lower_reverse with cursor_key = ikey_bound would return `size`
        // (past the last slot) because ikey_bound > all keys in the prev leaf.
        // This caused us to skip entire leaves.
        let size: usize = perm.size();
        let ki: isize = if size > 0 {
            (size - 1).cast_signed()
        } else {
            -1 // Empty leaf, will trigger another advance_to_prev_leaf
        };
        let _ = helper; // Silence unused warning - we intentionally don't use lower_reverse here

        // Step 9: Update stack state
        stack.update_state(version, perm, ki);

        // Continue finding previous entry
        (ScanStateBack::FindPrev, None)
    }

    // ========================================================================
    //  reposition_back - Version Conflict Recovery
    // ========================================================================

    /// Reposition after version conflict during reverse scan.
    ///
    /// When a version conflict is detected, we must traverse from the layer
    /// root to find the correct leaf for the current cursor key.
    ///
    /// # Algorithm
    ///
    /// 1. Get layer root from stack
    /// 2. Traverse from root to find target leaf (`reach_leaf_for_scan`)
    /// 3. Handle concurrent inserts via `stable_reverse`
    /// 4. Find position via `lower_reverse`
    /// 5. Update stack state
    ///
    /// # Retry Behavior
    ///
    /// Uses bounded iteration instead of recursion:
    /// - Maximum 16 retries to avoid livelock
    /// - Each retry handles deleted version scenario
    /// - Debug assertions detect excessive retries
    ///
    /// # Performance
    ///
    /// - O(height) per call (full root-to-leaf traversal)
    /// - Should be called rarely (< 1% of iterations)
    /// - Uses prefetch during traversal
    ///
    /// # C++ Reference
    ///
    /// Corresponds to `find_retry` logic in `masstree_scan.hh:305-330`.
    pub fn reposition_back<L, S>(
        stack: &mut BackStackElement<L, S>,
        cursor_key: &mut CursorKey,
        helper: &mut ReverseScanHelper,
        guard: &LocalGuard<'_>,
    ) -> (ScanStateBack, Option<ScanSnapshot<S>>)
    where
        L: TreeLeafNode<S>,
        S: ValueSlot,
        S::Output: Clone,
    {
        const MAX_REPOSITION_RETRIES: u32 = 16;

        let root: *const u8 = stack.get_root();

        // Handle null root (shouldn't happen, but be defensive)
        if root.is_null() {
            return (ScanStateBack::Up, None);
        }

        // Iterative retry loop (avoids recursion)
        for _retry in 0..MAX_REPOSITION_RETRIES {
            // Traverse from root to leaf
            let mut leaf_ptr: *mut L = Self::reach_leaf_for_scan::<L, S>(root, cursor_key, guard);

            // Check for empty tree / layer
            if leaf_ptr.is_null() {
                stack.set_leaf(std::ptr::null_mut());
                return (ScanStateBack::Up, None);
            }

            // Handle concurrent inserts (may follow forward chain)
            let version: u32 = ReverseScanHelper::stable_reverse(&mut leaf_ptr, cursor_key, guard);

            // Check for deleted version
            if NodeVersion::is_deleted_version(version) {
                // Leaf was deleted - retry from root
                // This should be extremely rare in practice
                continue;
            }

            // SAFETY: leaf_ptr is valid (null checked, not deleted)
            let leaf: &L = unsafe { &*leaf_ptr };
            let perm: L::Perm = leaf.permutation();
            let ki: isize = helper.lower_reverse(cursor_key, leaf, &perm);

            // Update stack with new position
            stack.set_leaf(leaf_ptr);
            stack.update_state(version, perm, ki);

            // Success - continue finding
            return (ScanStateBack::FindPrev, None);
        }

        // Exceeded max retries - this indicates a serious problem
        // (extreme contention or a bug). Return Up to terminate safely.
        debug_assert!(
            false,
            "reposition_back exceeded MAX_REPOSITION_RETRIES ({MAX_REPOSITION_RETRIES})"
        );

        stack.set_leaf(std::ptr::null_mut());
        (ScanStateBack::Up, None)
    }

    // ========================================================================
    //  handle_down_back - Sublayer Descent
    // ========================================================================

    /// Handle descent into sublayer for reverse scan.
    ///
    /// Called when state machine is in `Down` state after encountering
    /// a layer pointer during reverse iteration.
    ///
    /// # Algorithm
    ///
    /// 1. Shift cursor to sublayer with `shift_clear_reverse()` (ikey=MAX, len=9)
    /// 2. Set `helper.upper_bound = true` to start from last slot
    /// 3. Return - caller will call `find_initial_reverse` with new layer root
    ///
    /// # Critical: Why `shift_clear_reverse()` Not `shift()`
    ///
    /// For scan-discovered descent (not initial positioning), we always want
    /// to start at the MAXIMUM key in the sublayer. This ensures we scan
    /// the entire sublayer from end to beginning.
    ///
    /// # Performance
    ///
    /// O(1) - only modifies cursor and helper state.
    ///
    /// # C++ Reference
    ///
    /// ```cpp
    /// // masstree_scan.hh with reverse_scan_helper
    /// void shift_clear(K& ka) const {
    ///     ka.shift_clear_reverse();  // ikey=MAX, len=9
    ///     upper_bound_ = true;
    /// }
    /// ```
    #[inline(always)]
    pub fn handle_down_back(cursor_key: &mut CursorKey, helper: &mut ReverseScanHelper) {
        // Shift cursor to sublayer maximum
        // This sets ikey = MAX, len = 9, offset += 8
        cursor_key.shift_clear_reverse();

        // Signal to lower_reverse that we start from upper bound
        // This makes it return size - 1 instead of searching
        helper.upper_bound = true;
    }

    // ========================================================================
    //  handle_up_back - Layer Ascent
    // ========================================================================

    /// Handle ascent to parent layer for reverse scan.
    ///
    /// Called when state machine is in `Up` state after exhausting a sublayer
    /// (reached the beginning of the sublayer's key space).
    ///
    /// # Algorithm
    ///
    /// 1. Pop parent context from `layer_stack`
    /// 2. Restore stack with parent root/leaf
    /// 3. Unshift cursor (sets len=9 sentinel)
    /// 4. Get stable version (may follow forward chain)
    /// 5. Find position using `lower_reverse`
    /// 6. Update stack state
    ///
    /// # Returns
    ///
    /// - `true`: Successfully restored to parent layer, continue scanning
    /// - `false`: Layer stack empty, scan is complete
    ///
    /// # Key Insight: len=9 Sentinel
    ///
    /// After unshift, `len = 9` (`IKEY_SIZE + 1`) acts as a sentinel:
    /// - Signals "we just finished processing a sublayer"
    /// - Makes `lower_reverse` find position AFTER the layer pointer slot
    /// - Prevents re-descending into the same layer pointer
    ///
    /// # C++ Reference
    ///
    /// ```cpp
    /// // masstree_scan.hh:359-372
    /// do {
    ///     if (stack.node_stack_.empty()) goto done;
    ///     stack.n_ = pop(); stack.root_ = pop();
    ///     ka.unshift();
    /// } while (unlikely(ka.empty()));
    /// ```
    pub fn handle_up_back<L, S>(
        stack: &mut BackStackElement<L, S>,
        cursor_key: &mut CursorKey,
        layer_stack: &mut LayerStack<L>,
        helper: &mut ReverseScanHelper,
        guard: &LocalGuard<'_>,
    ) -> bool
    where
        L: TreeLeafNode<S>,
        S: ValueSlot,
    {
        // Step 1: Pop parent context from layer stack
        let Some(parent_ctx) = layer_stack.pop() else {
            // Layer stack is empty - scan is complete
            return false;
        };

        // Step 2: Restore stack with parent context
        stack.set_root(parent_ctx.root);
        stack.set_leaf(parent_ctx.leaf.as_ptr());

        // Step 3: Unshift cursor to parent layer
        // This sets len = 9 (IKEY_SIZE + 1) as sentinel
        cursor_key.unshift();

        // Step 4: Handle edge case - cursor became empty after unshift
        // This happens if we ascend from a layer that was at offset 0
        // C++: while (unlikely(ka.empty())) { pop + unshift }
        if cursor_key.is_empty_after_unshift() {
            // Need to continue ascending (recursive, but bounded by layer depth)
            return Self::handle_up_back(stack, cursor_key, layer_stack, helper, guard);
        }

        // Step 5: Get stable version from parent leaf
        // May follow forward chain if concurrent insert moved our key
        let mut leaf_ptr: *mut L = parent_ctx.leaf.as_ptr();
        let version: u32 = ReverseScanHelper::stable_reverse(&mut leaf_ptr, cursor_key, guard);

        // Update leaf if stable_reverse followed forward chain
        if leaf_ptr != parent_ctx.leaf.as_ptr() {
            stack.set_leaf(leaf_ptr);
        }

        // Step 6: Find position in parent leaf
        // SAFETY: leaf_ptr is valid (from NonNull in LayerContext, stable_reverse ensures it)
        let leaf: &L = unsafe { &*leaf_ptr };
        let perm: L::Perm = leaf.permutation();

        // lower_reverse with len=9 will return position AFTER the layer pointer
        // we just finished, which is correct (we want the previous slot)
        let ki: isize = helper.lower_reverse(cursor_key, leaf, &perm);

        // Step 7: Update stack state
        stack.update_state(version, perm, ki);

        // Successfully restored to parent layer
        true
    }
}

// ============================================================================
//  Intra-Leaf Batch Processing for Reverse Scan
// ============================================================================

/// Result of processing entries within a single leaf (reverse direction).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum LeafBatchResultBack {
    /// All entries in leaf processed, need to retreat to previous leaf
    LeafExhausted = 0,

    /// Encountered a layer pointer, need to descend
    LayerEncountered = 1,

    /// Version changed during processing, need retry
    VersionChanged = 2,

    /// Visitor returned false, stop iteration
    Stopped = 3,

    /// Start bound exceeded, stop iteration
    StartBoundExceeded = 4,
}

use super::iterator::RangeBound;

/// Process remaining entries in current leaf in reverse order (tight loop).
///
/// This is the core intra-leaf batch optimization for reverse scans. Instead of
/// returning after each entry, we process all remaining entries in the current
/// leaf before returning control to the caller.
///
/// # Algorithm
///
/// For each remaining slot in permutation (from ki down to 0):
/// 1. Read slot data `(ikey, keylenx, value_ptr)`
/// 2. If layer pointer -> return [`LeafBatchResultBack::LayerEncountered`]
/// 3. If null value -> skip
/// 4. Build key and call visitor
/// 5. Check start bound
/// 6. Validate version (OCC) after batch
///
/// # Performance
///
/// This eliminates:
/// - Function call overhead per entry
/// - State machine dispatch per entry
/// - Redundant leaf/version checks
///
/// Expected 2-3x improvement for reverse scans touching many entries per leaf.
#[inline]
#[expect(clippy::too_many_arguments)]
pub fn process_prev_leaf_batch_ptr<L, S, F>(
    stack: &mut BackStackElement<L, S>,
    cursor_key: &mut CursorKey,
    layer_stack: &mut LayerStack<L>,
    start_bound: &RangeBound<'_>,
    helper: &mut ReverseScanHelper,
    visitor: &mut F,
    count: &mut usize,
) -> LeafBatchResultBack
where
    L: TreeLeafNode<S>,
    S: ValueSlot,
    F: FnMut(&[u8], &S::Value) -> bool,
{
    // Cache leaf pointer to avoid borrow conflicts
    let leaf_ptr: *const L = stack.get_leaf_ptr();
    let leaf: &L = unsafe { &*leaf_ptr };
    let perm = *stack.get_perm_ref();
    let perm_size = perm.size();
    let cached_version = stack.get_version();

    // Check if leaf was deleted since we cached the version
    if leaf.version().is_deleted() {
        return LeafBatchResultBack::VersionChanged;
    }

    // Get current position (signed for reverse iteration)
    let mut ki: isize = stack.get_ki();

    // Process remaining entries in reverse order
    while ki >= 0 && ki.cast_unsigned() < perm_size {
        let slot = perm.get(ki.cast_unsigned());

        // Read slot data with relaxed ordering (permutation provides synchronization)
        let slot_ikey: u64 = leaf.ikey_relaxed(slot);
        let slot_keylenx: u8 = leaf.keylenx(slot);

        // Prefetch previous slot's value to hide memory latency
        if ki > 0 {
            let prev_slot: usize = perm.get((ki - 1).cast_unsigned());
            prefetch_read(leaf.leaf_value_ptr(prev_slot));
        }

        // Check for layer pointer - must handle via state machine
        if slot_keylenx >= LAYER_KEYLENX {
            // Set up for layer descent
            let slot_ptr: *mut u8 = leaf.leaf_value_ptr(slot);
            layer_stack.push(LayerContext::new(stack.get_root(), stack.get_leaf_ptr()));
            cursor_key.assign_store_ikey(slot_ikey);
            prefetch_read(slot_ptr);
            stack.set_root(slot_ptr.cast_const());
            // Update ki for when we return from sublayer
            stack.set_ki(ki - 1);
            return LeafBatchResultBack::LayerEncountered;
        }

        // Get value pointer
        let slot_ptr: *mut u8 = leaf.leaf_value_ptr(slot);
        if slot_ptr.is_null() {
            ki -= 1;
            continue;
        }

        // Build key
        cursor_key.assign_store_ikey(slot_ikey);

        let _key_len: usize = if slot_keylenx == KSUF_KEYLENX {
            if let Some(suffix) = leaf.ksuf(slot) {
                let suffix_len = suffix.len();
                let _ = cursor_key.assign_store_suffix(suffix);
                cursor_key.assign_store_length(IKEY_SIZE + suffix_len);
                IKEY_SIZE + suffix_len
            } else {
                cursor_key.assign_store_length(IKEY_SIZE);
                IKEY_SIZE
            }
        } else {
            let len = slot_keylenx as usize;
            cursor_key.assign_store_length(len);
            len
        };

        cursor_key.mark_key_complete();

        // Clear upper_bound after first successful key emission
        helper.mark_key_complete();

        // Check start bound (for reverse iteration)
        let key: &[u8] = cursor_key.full_key();
        if !start_bound.contains_reverse(key) {
            return LeafBatchResultBack::StartBoundExceeded;
        }

        // SAFETY: Guard protects value pointer, slot is valid
        let value_ref: &S::Value = unsafe { &*slot_ptr.cast::<S::Value>() };

        *count += 1;
        ki -= 1;

        if !visitor(key, value_ref) {
            // Update stack position before returning
            stack.set_ki(ki);
            return LeafBatchResultBack::Stopped;
        }
    }

    // Update stack with final position
    stack.set_ki(ki);

    // Validate version after processing batch (OCC)
    if leaf.version().has_changed(cached_version) {
        return LeafBatchResultBack::VersionChanged;
    }

    LeafBatchResultBack::LeafExhausted
}

/// Advance to previous leaf in the B-link chain (batch processing variant).
///
/// This is a simplified version of `advance_to_prev_leaf` for use in batch
/// processing. It doesn't return state machine states, just updates the stack.
///
/// # Returns
///
/// - `true`: Successfully advanced to previous leaf
/// - `false`: No previous leaf (layer exhausted)
#[inline]
pub fn advance_prev_leaf_ptr<L, S>(
    stack: &mut BackStackElement<L, S>,
    cursor_key: &mut CursorKey,
    guard: &LocalGuard<'_>,
) -> bool
where
    L: TreeLeafNode<S>,
    S: ValueSlot,
{
    let current_ptr: *mut L = stack.get_leaf_ptr();
    if current_ptr.is_null() {
        return false;
    }

    // SAFETY: current_ptr was validated
    let current_leaf: &L = unsafe { &*current_ptr };

    // Update cursor key with current leaf's ikey_bound
    let ikey_bound: u64 = current_leaf.ikey_bound();
    cursor_key.assign_store_ikey(ikey_bound);
    cursor_key.assign_store_length(0);

    // Get previous leaf pointer
    let prev_ptr: *mut L = ReverseScanHelper::retreat(current_leaf);

    if prev_ptr.is_null() {
        stack.set_leaf(std::ptr::null_mut());
        return false;
    }

    // Prefetch previous leaf and its prev pointer for 2-way pipelining
    prefetch_read(prev_ptr.cast::<u8>());

    // Set up stack with new leaf
    stack.set_leaf(prev_ptr);

    // Get stable version (may follow forward chain for concurrent inserts)
    let mut leaf_ptr: *mut L = prev_ptr;
    let version: u32 = ReverseScanHelper::stable_reverse(&mut leaf_ptr, cursor_key, guard);

    // Update stack if stable_reverse followed forward chain
    if leaf_ptr != prev_ptr {
        stack.set_leaf(leaf_ptr);
    }

    // Prefetch the prev-prev leaf for pipelining
    let leaf: &L = unsafe { &*leaf_ptr };
    let prev_prev: *mut L = leaf.prev();
    if !prev_prev.is_null() {
        prefetch_read(prev_prev.cast::<u8>());
    }

    let perm: L::Perm = leaf.permutation();
    let size: usize = perm.size();
    let ki: isize = if size > 0 {
        (size - 1).cast_signed()
    } else {
        -1
    };

    stack.update_state(version, perm, ki);
    true
}

// ============================================================================
//  Single-Layer Fast Path for Reverse Scan
// ============================================================================

use super::scan_state::ScanSnapshotPtr;

/// Fast path for reverse iteration on single-layer trees (keys ≤ 8 bytes).
///
/// This is a specialized variant of `find_prev` that skips:
/// - Layer pointer handling (assumes no layer pointers exist)
/// - Complex suffix comparisons (keys are always ≤ 8 bytes)
/// - Layer stack operations
///
/// # When to Use
///
/// Use this when `single_layer_mode` is enabled, which happens when:
/// - Both start and end bounds have keys ≤ 8 bytes, OR
/// - The tree is known to contain only short keys
///
/// # Returns
///
/// - `ScanStateBack::Emit` with snapshot: Found a value to emit
/// - `ScanStateBack::FindPrev`: Continue searching (advanced position)
/// - `ScanStateBack::Retry`: Version conflict, need repositioning
/// - `ScanStateBack::Down`: Unexpectedly encountered layer pointer (fallback)
///
/// # Performance
///
/// ~20% faster than standard `find_prev` for single-layer data by eliminating
/// branch mispredictions from layer pointer checks.
#[inline]
pub fn find_prev_single_layer_ptr<L, S>(
    stack: &mut BackStackElement<L, S>,
    cursor_key: &mut CursorKey,
    helper: &mut ReverseScanHelper,
    guard: &LocalGuard<'_>,
    needs_duplicate_check: bool,
) -> (ScanStateBack, Option<ScanSnapshotPtr<S::Value>>)
where
    L: TreeLeafNode<S>,
    S: ValueSlot,
{
    // Check for null leaf (layer exhausted in single-layer mode)
    let leaf_ptr: *mut L = stack.get_leaf_ptr();
    if leaf_ptr.is_null() {
        return (ScanStateBack::FindPrev, None);
    }

    let ki: isize = stack.get_ki();

    // Fast path: leaf exhausted (ki went negative)
    if ki < 0 {
        return advance_prev_leaf_single_layer(stack, cursor_key, guard);
    }

    // SAFETY: leaf_ptr is valid (null checked above)
    let leaf: &L = unsafe { &*leaf_ptr };

    // Check if leaf is deleted (cheap - single atomic load)
    if leaf.version().is_deleted() {
        return (ScanStateBack::Retry, None);
    }

    let perm: L::Perm = *stack.get_perm_ref();
    let perm_size: usize = perm.size();

    // Defensive: ki might be >= size due to concurrent deletion
    if ki.unsigned_abs() >= perm_size {
        return advance_prev_leaf_single_layer(stack, cursor_key, guard);
    }

    // Get current slot
    let slot: usize = perm.get(ki.unsigned_abs());
    let slot_ikey: u64 = leaf.ikey_relaxed(slot);
    let slot_keylenx: u8 = leaf.keylenx(slot);

    // Check for duplicate only when needed (after Retry)
    if needs_duplicate_check
        && ReverseScanHelper::is_duplicate_reverse(
            cursor_key,
            slot_ikey,
            slot_keylenx,
            helper.upper_bound,
        )
    {
        stack.set_ki(ki - 1);
        return (ScanStateBack::FindPrev, None);
    }

    // DEFENSIVE: If we encounter a layer pointer, signal fallback
    if slot_keylenx >= LAYER_KEYLENX {
        cursor_key.assign_store_ikey(slot_ikey);
        let layer_ptr: *mut u8 = leaf.leaf_value_ptr(slot);
        prefetch_read(layer_ptr);
        return (ScanStateBack::Down, None);
    }

    // Value slot - prepare for emit
    let slot_ptr: *mut u8 = leaf.leaf_value_ptr(slot);
    if slot_ptr.is_null() {
        stack.set_ki(ki - 1);
        return (ScanStateBack::FindPrev, None);
    }

    // Single-layer keys are always ≤ 8 bytes, no suffix handling
    let key_len: usize = slot_keylenx as usize;
    cursor_key.assign_store_ikey(slot_ikey);
    cursor_key.assign_store_length(key_len);
    cursor_key.mark_key_complete();

    // Clear upper_bound after successful emit
    helper.mark_key_complete();

    // Advance position for next call (go backwards)
    stack.set_ki(ki - 1);

    // Return raw pointer
    (
        ScanStateBack::Emit,
        Some(ScanSnapshotPtr::from_raw(slot_ptr.cast_const(), key_len)),
    )
}

/// Advance to previous leaf in single-layer mode.
///
/// Simplified version that doesn't handle Up transitions (no layer stack).
#[inline(always)]
fn advance_prev_leaf_single_layer<L, S>(
    stack: &mut BackStackElement<L, S>,
    cursor_key: &mut CursorKey,
    _guard: &LocalGuard<'_>,
) -> (ScanStateBack, Option<ScanSnapshotPtr<S::Value>>)
where
    L: TreeLeafNode<S>,
    S: ValueSlot,
{
    let leaf_ptr: *mut L = stack.get_leaf_ptr();
    if leaf_ptr.is_null() {
        return (ScanStateBack::FindPrev, None);
    }

    let leaf: &L = unsafe { &*leaf_ptr };
    let version: u32 = stack.get_version();

    // Check if version changed (concurrent modification)
    if leaf.version().has_changed(version) {
        return (ScanStateBack::Retry, None);
    }

    // Update cursor with current leaf's bound before moving
    let ikey_bound: u64 = leaf.ikey_bound();
    cursor_key.assign_store_ikey(ikey_bound);
    cursor_key.assign_store_length(0);

    // Get previous leaf
    let prev: *mut L = leaf.prev();

    if prev.is_null() {
        // No more leaves - scan exhausted (no Up in single-layer)
        stack.set_leaf(std::ptr::null_mut());
        return (ScanStateBack::FindPrev, None);
    }

    // Move to previous leaf
    stack.set_leaf(prev);

    // SAFETY: prev is non-null
    let prev_leaf: &L = unsafe { &*prev };

    // Prefetch the previous leaf's data
    prev_leaf.prefetch();

    // Prefetch prev-prev leaf for 2-way pipelining
    let prev_prev: *mut L = prev_leaf.prev();
    if !prev_prev.is_null() {
        let prev_prev_leaf: &L = unsafe { &*prev_prev };
        prev_prev_leaf.prefetch();
    }

    // Get stable version
    let prev_version: u32 = prev_leaf.version().stable();

    if NodeVersion::is_deleted_version(prev_version) {
        return (ScanStateBack::Retry, None);
    }

    // Load permutation and start from last slot
    let perm: L::Perm = prev_leaf.permutation();
    let size: usize = perm.size();
    let ki: isize = if size > 0 {
        (size - 1).cast_signed()
    } else {
        -1
    };

    stack.update_state(prev_version, perm, ki);

    (ScanStateBack::FindPrev, None)
}

// ============================================================================
//  Tests
// ============================================================================

#[cfg(test)]
mod unit_tests;
