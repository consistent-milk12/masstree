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

/// Upper bound on parent climbs / B-link hops before giving up.
const MAX_REACH_HOPS: usize = 64;

/// Climb parent pointers from `start` to the true layer root.
///
/// After layer-root promotion the parent slot still points at the original
/// left leaf; point reads climb `parent` pointers to find the internode root.
/// Scans must do the same or they will only ever search that one leaf.
#[inline]
fn climb_layer_root<P>(mut node: *const u8, guard: &LocalGuard<'_>) -> *const u8
where
    P: LeafPolicy,
{
    for _ in 0..MAX_REACH_HOPS {
        if node.is_null() {
            return node;
        }

        // SAFETY: node is a live node protected by `guard`; both node kinds
        // have NodeVersion as their first field.
        #[expect(clippy::cast_ptr_alignment, reason = "NodeVersion is first field")]
        let version: &NodeVersion = unsafe { &*node.cast::<NodeVersion>() };

        let parent: *mut u8 = if version.is_leaf() {
            // SAFETY: is_leaf() confirmed above.
            let leaf: &LeafNode15<P> = unsafe { &*node.cast::<LeafNode15<P>>() };
            leaf.parent(guard)
        } else {
            // SAFETY: !is_leaf() confirmed above.
            let inode: &InternodeNode = unsafe { &*node.cast::<InternodeNode>() };
            inode.parent(guard)
        };

        if parent.is_null() {
            return node;
        }

        node = parent.cast_const();
    }

    node
}

/// Descend from a layer root to the leaf that should contain `cursor_key`.
///
/// Uses the same stability rules as point reads: climb to the true root,
/// restart on a stale/deleted node or when the leaf is too far right, and
/// follow `next` when a split moved the target right. A hop cap keeps a
/// corrupted chain from spinning forever.
#[inline]
pub fn reach_leaf_for_scan<P>(
    start: *const u8,
    cursor_key: &CursorKey,
    guard: &LocalGuard<'_>,
) -> *mut LeafNode15<P>
where
    P: LeafPolicy,
{
    if start.is_null() {
        return StdPtr::null_mut();
    }

    let mut root: *const u8 = climb_layer_root::<P>(start, guard);
    let target_ikey: u64 = cursor_key.current_ikey();
    let mut node: *const u8 = root;

    for _ in 0..MAX_REACH_HOPS {
        if node.is_null() {
            return StdPtr::null_mut();
        }

        // SAFETY: node is a live node protected by `guard`.
        #[expect(clippy::cast_ptr_alignment, reason = "NodeVersion is first field")]
        let version: &NodeVersion = unsafe { &*node.cast::<NodeVersion>() };
        let v: u32 = version.stable();

        if version.is_leaf() {
            // SAFETY: is_leaf() confirmed above.
            let leaf: &LeafNode15<P> = unsafe { &*node.cast::<LeafNode15<P>>() };

            if leaf.version().is_deleted() {
                root = climb_layer_root::<P>(start, guard);
                node = root;
                continue;
            }

            // Too far right: the target belongs at or before this leaf.
            if !leaf.prev(guard).is_null() && target_ikey < leaf.ikey_bound() {
                root = climb_layer_root::<P>(start, guard);
                node = root;
                continue;
            }

            // A concurrent split may have moved the target right.
            let next: *mut LeafNode15<P> = leaf.safe_next(guard);
            if !next.is_null() {
                // SAFETY: next is protected by the guard.
                let next_bound: u64 = unsafe { &*next }.ikey_bound();
                if target_ikey >= next_bound {
                    node = next.cast::<u8>().cast_const();
                    continue;
                }
            }

            return node.cast_mut().cast::<LeafNode15<P>>();
        }

        // SAFETY: !is_leaf() confirmed above.
        let inode: &InternodeNode = unsafe { &*node.cast::<InternodeNode>() };

        let child_idx: usize = upper_bound_internode_generic::<InternodeNode>(target_ikey, inode);
        // Guard-protected load: the child must remain allocated for the guard.
        let child: *mut u8 = inode.child(child_idx, guard);

        prefetch_read(child);

        if child.is_null() {
            root = climb_layer_root::<P>(start, guard);
            node = root;
            continue;
        }

        if inode.version().has_changed(v) {
            // A split (or a layer-root promotion) may have moved the key;
            // re-climb from the original start before retrying.
            root = climb_layer_root::<P>(start, guard);
            node = root;
            continue;
        }

        node = child;
    }

    StdPtr::null_mut()
}
