use masstree::{BoxPolicy, InternodeNode, LeafNode15, MassTree15, NodeVersion};
use seize::LocalGuard;

fn key(i: usize) -> [u8; 8] {
    format!("a{:07}", i).into_bytes().try_into().unwrap()
}

fn collect_leaves(
    tree: &MassTree15<u64>,
    guard: &LocalGuard<'_>,
) -> Vec<*mut LeafNode15<BoxPolicy<u64>>> {
    // SAFETY: quiescent inspection under the tree guard.
    let root = unsafe { tree.debug_root_ptr(guard) };
    let ver = unsafe { &*(root.cast::<NodeVersion>()) };
    let mut out = Vec::new();
    if ver.is_leaf() {
        out.push(root.cast::<LeafNode15<BoxPolicy<u64>>>());
    } else {
        let inode = unsafe { &*(root.cast::<InternodeNode>()) };
        let n = inode.nkeys();
        for i in 0..=n {
            let c = unsafe { inode.child_unguarded(i) };
            if !c.is_null() {
                let cver = unsafe { &*(c.cast::<NodeVersion>()) };
                if cver.is_leaf() {
                    out.push(c.cast::<LeafNode15<BoxPolicy<u64>>>());
                }
            }
        }
    }
    out
}

fn main() {
    let tree: MassTree15<u64> = MassTree15::new();
    let guard = tree.guard();
    for i in 0..32 {
        tree.insert_with_guard(&key(i), i as u64, &guard);
    }
    let leaves = collect_leaves(&tree, &guard);
    eprintln!("leaves: {}", leaves.len());
    for (i, l) in leaves.iter().enumerate() {
        let leaf = unsafe { &**l };
        let perm = leaf.permutation();
        let n = perm.size();
        eprintln!(
            "leaf {} ptr {:p} size {} first {:016x} last_slot {} bound {:016x} prev_null {}",
            i,
            *l,
            n,
            leaf.ikey(perm.get(0)),
            perm.get(n - 1),
            leaf.ikey_bound(),
            unsafe { leaf.prev_unguarded() }.is_null()
        );
    }
    if leaves.len() < 2 {
        eprintln!("need >=2 leaves");
        return;
    }
    // Pick a non-leftmost leaf and remove all its keys.
    let victim = leaves[1];
    let leaf = unsafe { &*victim };
    let perm = leaf.permutation();
    let n = perm.size();
    let mut keys = Vec::new();
    for i in 0..n {
        keys.push(leaf.ikey(perm.get(i)));
    }
    eprintln!("victim keys: {:016x?}", keys);
    for k in &keys {
        let kb = k.to_be_bytes();
        let r = tree.remove_with_guard(&kb, &guard).unwrap();
        assert!(r.is_some(), "remove missing");
    }
    eprintln!("pending after empty: {}", tree.pending_coalesce());
    // Reuse the empty leaf with the key equal to its ikey_bound.
    let reuse = keys[0].to_be_bytes();
    tree.insert_with_guard(&reuse, 999, &guard);
    eprintln!("reinserted, pending={}", tree.pending_coalesce());
    let r = tree.remove_with_guard(&reuse, &guard).unwrap();
    eprintln!(
        "removed reuse -> {:?}, pending now {}",
        r,
        tree.pending_coalesce()
    );
    eprintln!("calling process_coalesce_batch(1) ...");
    let processed = tree.process_coalesce_batch(&guard, 1);
    eprintln!("process_coalesce_batch(1) returned {processed}, pending={}", tree.pending_coalesce());
    eprintln!("dropping without second pass (would reprocess retired leaf)");
    drop(guard);
    drop(tree);
    eprintln!("done");
}
