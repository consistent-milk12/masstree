//! Shared traversal utilities for range scans.
//!
//! This module contains traversal functions shared between forward
//! and reverse range iteration.

use std::ptr as StdPtr;

use seize::LocalGuard;

use crate::internode::InternodeNode;
use crate::ksearch::upper_bound_internode_generic;
use crate::leaf15::LeafNode15;
use crate::nodeversion::NodeVersion;
use crate::policy::LeafPolicy;
use crate::prefetch::prefetch_read;

use super::cursor_key::CursorKey;

/// Traverse from layer root to target leaf for range scans.
#[inline]
pub fn reach_leaf_for_scan<P>(
    start: *const u8,
    cursor_key: &CursorKey,
    _guard: &LocalGuard<'_>,
) -> *mut LeafNode15<P>
where
    P: LeafPolicy,
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
            return node.cast_mut().cast::<LeafNode15<P>>();
        }

        // It's an internode - traverse down
        // SAFETY: !is_leaf() confirmed above
        let inode: &InternodeNode = unsafe { &*(node.cast::<InternodeNode>()) };

        // Binary search for child
        let child_idx: usize = upper_bound_internode_generic::<InternodeNode>(target_ikey, inode);
        // Use the guard-protected load: the child must remain allocated for
        // the guard's lifetime, not merely be an Acquire snapshot.
        let child: *mut u8 = inode.child(child_idx, _guard);

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
