#![expect(clippy::unwrap_used, clippy::indexing_slicing)]

use crate::{BoxPolicy, MassTreeGeneric, NodeVersion};
use crate::{
    internode::InternodeNode,
    leaf15::LeafNode15,
    policy::{InlinePolicy, LeafPolicy},
};

use super::{ROOT_CLIMB_COUNT, resolve_layer_root};

fn check_slot_caching<P: LeafPolicy<Value = u64>>() {
    for count in [40, 300] {
        let tree = MassTreeGeneric::with_allocator(crate::SeizeAllocator::<P>::new());
        let keys: Vec<_> = (0..count)
            .map(|i: u64| [b"PREFIX00".as_slice(), &i.to_be_bytes()].concat())
            .collect();
        for (i, key) in keys.iter().enumerate() {
            tree.insert(key, i as u64);
        }
        let guard = tree.guard();
        // SAFETY: Only one top-level ikey was inserted, so the top root remains
        // a leaf. The tree guard protects it throughout the test.
        let parent = unsafe { &*tree.load_root_ptr_generic(&guard).cast::<LeafNode15<P>>() };
        let slot = parent.permutation().get(0);
        let raw = parent.load_layer_raw(slot);
        // SAFETY: raw came from the guarded parent slot and is a live layer node.
        let root = unsafe { resolve_layer_root::<P>(raw, &guard) }.unwrap();
        let mut left = root;
        loop {
            // SAFETY: Every node was reached through this guarded tree; both
            // layouts begin with NodeVersion and have immutable node kinds.
            #[expect(clippy::cast_ptr_alignment, reason = "NodeVersion is first field")]
            let version = unsafe { &*left.cast::<NodeVersion>() };
            if version.is_leaf() {
                break;
            }
            // SAFETY: The kind check established an internode. No writer runs
            // during this setup, and child(0) is protected by `guard`.
            left = unsafe { &*left.cast::<InternodeNode>() }.child(0, &guard);
        }
        assert_ne!(left, root);
        // Simulate a skipped opportunistic refresh during promotion. Pointing
        // at the original left leaf remains a valid representation of a layer.
        let mut lock = parent.version().lock();
        lock.mark_insert();
        // SAFETY: We hold the parent leaf lock, the slot is still this layer,
        // and `left` remains a published child of the same live root.
        unsafe { parent.store_layer(slot, left.cast_mut()) };
        drop(lock);

        // A reader must make progress when the optional refresh cannot acquire
        // the parent lock. Release before asserting the watchdog so a failure
        // can join the worker and report rather than hanging the test process.
        let lock = parent.version().lock();
        let (send, recv) = std::sync::mpsc::channel();
        std::thread::scope(|scope| {
            scope.spawn(|| send.send(tree.get(&keys[0])).unwrap());
            let result = recv.recv_timeout(std::time::Duration::from_secs(30));
            drop(lock);
            assert_eq!(result.unwrap(), Some(0));
        });
        assert_eq!(parent.load_layer_raw(slot).cast_const(), left);
        assert_eq!(tree.get(&keys[0]), Some(0));
        assert_eq!(parent.load_layer_raw(slot).cast_const(), root);
        ROOT_CLIMB_COUNT.with(|count| count.set(0));
        assert_eq!(tree.get(&keys[39]), Some(39));
        assert_eq!(ROOT_CLIMB_COUNT.with(std::cell::Cell::get), 0);

        for key in &keys {
            tree.remove_with_guard(key, &guard).unwrap();
        }
        tree.process_coalesce(&guard);
        assert_eq!(tree.len(), 0);
        assert!(tree.get(&keys[39]).is_none());
    }
}

#[test]
fn point_reads_refresh_promoted_parent_slots() {
    check_slot_caching::<BoxPolicy<u64>>();
    check_slot_caching::<InlinePolicy<u64>>();
}

#[test]
fn reach_rejects_retired_promoted_root() {
    let tree = crate::MassTree15::<u64>::new();
    let keys: Vec<_> = (0..40u64)
        .map(|i| [b"PREFIX00".as_slice(), &i.to_be_bytes()].concat())
        .collect();
    for (i, key) in keys.iter().enumerate() {
        tree.insert(key, i as u64);
    }
    let guard = tree.guard();
    // SAFETY: The one-prefix top root is a leaf, protected by guard.
    let parent = unsafe {
        &*tree
            .load_root_ptr_generic(&guard)
            .cast::<LeafNode15<BoxPolicy<u64>>>()
    };
    let raw = parent.load_layer_raw(parent.permutation().get(0));
    // SAFETY: The guarded parent slot points into the live sublayer.
    let root = unsafe { resolve_layer_root::<BoxPolicy<u64>>(raw, &guard) }.unwrap();
    for key in &keys {
        tree.remove_with_guard(key, &guard).unwrap();
    }
    tree.process_coalesce(&guard);
    // SAFETY: GC retires this node, but guard prevents its reclamation. Both
    // layouts begin with NodeVersion, and this root is an internode.
    #[expect(clippy::cast_ptr_alignment, reason = "NodeVersion is first field")]
    let version = unsafe { &*root.cast::<NodeVersion>() };
    assert!(version.is_deleted());
    println!("entering traversal with a retired, guard-protected root");
    let reached =
        tree.reach_leaf_concurrent_generic(root, &crate::key::Key::from_ikey(0), true, &guard);
    assert!(
        reached.is_null(),
        "retired root must request a fresh descent"
    );
}
