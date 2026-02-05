//! Shared traversal utilities for range scans.
//!
//! This module contains traversal functions shared between forward
//! and reverse range iteration.

use std::ptr as StdPtr;

use seize::LocalGuard;

use crate::ksearch::upper_bound_internode_generic;
use crate::leaf_trait::{TreeInternode, TreeLeafNode};
use crate::nodeversion::NodeVersion;
use crate::prefetch::prefetch_read;
use crate::slot::ValueSlot;

use super::cursor_key::CursorKey;

/// Traverse from layer root to target leaf for range scans.
///
/// Similar to `reach_leaf_concurrent_generic` but optimized for
/// range scan access patterns:
/// - No root fix-up (scans always start from known root)
/// - Single-pointer traversal (simpler for sequential access)
/// - Minimal prefetch (scan will access sequentially anyway)
///
/// # Arguments
///
/// - `start`: Layer root pointer
/// - `cursor_key`: Cursor containing target ikey
/// - `_guard`: Memory reclamation guard (lifetime binding)
///
/// # Returns
///
/// Pointer to the leaf containing (or that should contain) the key.
/// Returns null if `start` is null.
#[inline]
pub fn reach_leaf_for_scan<L, S>(
    start: *const u8,
    cursor_key: &CursorKey,
    _guard: &LocalGuard<'_>,
) -> *mut L
where
    L: TreeLeafNode<S>,
    S: ValueSlot,
{
    if start.is_null() {
        return StdPtr::null_mut();
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
        let child_idx: usize = upper_bound_internode_generic::<L::Internode>(target_ikey, inode);
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
