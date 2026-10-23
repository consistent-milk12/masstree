//! Repro: a layer root that has been promoted to an internode can never be
//! GC'd once every key in the sublayer is deleted.
//!
//! `try_remove_sublayer` always casts the parent-slot layer pointer to a leaf.
//! After layer-root promotion that pointer is the original left leaf, and the
//! "isolated" test requires `prev == null && next == null`. A promoted
//! sublayer has >= 2 linked leaves forever (there is no merge, and empty
//! leaves inside a sublayer are never chain-coalesced), so isolation never
//! holds: entries are requeued 10 times, counted as abandoned, and the whole
//! sublayer (leaves + internode root) is leaked.

use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use masstree::{
    BoxPolicy, LeafNode15, MassTreeGeneric, NodeVersion, SeizeAllocator, TreeAllocator,
};
use seize::LocalGuard;

#[derive(Default)]
struct Counters {
    leaf_allocs: AtomicUsize,
    leaf_retires: AtomicUsize,
    inode_allocs: AtomicUsize,
    inode_retires: AtomicUsize,
}

struct CountingAlloc {
    inner: SeizeAllocator<BoxPolicy<u64>>,
    c: Arc<Counters>,
}

impl CountingAlloc {
    fn new(c: Arc<Counters>) -> Self {
        Self {
            inner: SeizeAllocator::new(),
            c,
        }
    }
}

fn live_leaves(c: &Counters) -> usize {
    c.leaf_allocs.load(Ordering::Relaxed) - c.leaf_retires.load(Ordering::Relaxed)
}
fn live_inodes(c: &Counters) -> usize {
    c.inode_allocs.load(Ordering::Relaxed) - c.inode_retires.load(Ordering::Relaxed)
}

impl TreeAllocator<BoxPolicy<u64>> for CountingAlloc {
    fn alloc_leaf_direct(&self, is_root: bool, is_layer_root: bool) -> *mut LeafNode15<BoxPolicy<u64>> {
        self.c.leaf_allocs.fetch_add(1, Ordering::Relaxed);
        self.inner.alloc_leaf_direct(is_root, is_layer_root)
    }
    unsafe fn retire_leaf(&self, ptr: *mut LeafNode15<BoxPolicy<u64>>, guard: &LocalGuard<'_>) {
        self.c.leaf_retires.fetch_add(1, Ordering::Relaxed);
        unsafe { self.inner.retire_leaf(ptr, guard) }
    }
    fn alloc_internode_direct(&self, height: u32) -> *mut u8 {
        self.c.inode_allocs.fetch_add(1, Ordering::Relaxed);
        self.inner.alloc_internode_direct(height)
    }
    fn alloc_internode_direct_root(&self, height: u32) -> *mut u8 {
        self.c.inode_allocs.fetch_add(1, Ordering::Relaxed);
        self.inner.alloc_internode_direct_root(height)
    }
    fn alloc_internode_direct_for_split(
        &self,
        parent_version: &NodeVersion,
        height: u32,
    ) -> *mut u8 {
        self.c.inode_allocs.fetch_add(1, Ordering::Relaxed);
        self.inner
            .alloc_internode_direct_for_split(parent_version, height)
    }
    unsafe fn retire_internode_erased(&self, ptr: *mut u8, guard: &LocalGuard<'_>) {
        self.c.inode_retires.fetch_add(1, Ordering::Relaxed);
        unsafe { self.inner.retire_internode_erased(ptr, guard) }
    }
    fn teardown_tree(&self, root_ptr: *mut u8) {
        self.inner.teardown_tree(root_ptr)
    }
}

fn key(i: u32) -> Vec<u8> {
    let mut k = b"ABCDEFGH".to_vec(); // exactly one ikey
    k.extend_from_slice(&i.to_be_bytes()); // 4-byte suffix
    k
}

#[test]
fn promoted_sublayer_is_leaked() {
    let counters = Arc::new(Counters::default());
    let tree: MassTreeGeneric<BoxPolicy<u64>, CountingAlloc> =
        MassTreeGeneric::with_allocator(CountingAlloc::new(Arc::clone(&counters)));

    // 40 keys share the first 8 bytes -> one sublayer; >15 keys force the
    // sublayer leaf to split, promoting its root to an internode.
    for i in 0..40u32 {
        tree.insert(&key(i), u64::from(i) + 1);
    }
    for i in 0..40u32 {
        assert_eq!(tree.get(&key(i)), Some(u64::from(i) + 1), "setup {i}");
    }

    let before_leaves = live_leaves(&counters);
    let before_inodes = live_inodes(&counters);
    println!("before: live leaves={before_leaves}, live inodes={before_inodes}");

    for i in 0..40u32 {
        let r = tree.remove(&key(i)).unwrap();
        assert_eq!(r, Some(u64::from(i) + 1), "remove {i}");
    }
    assert_eq!(tree.len(), 0, "all keys removed");
    for i in 0..40u32 {
        assert_eq!(tree.get(&key(i)), None, "key {i} still visible");
    }

    // Plenty of maintenance cycles: requeues are capped at 10.
    for _ in 0..200 {
        let g = tree.guard();
        tree.process_coalesce(&g);
    }

    let after_leaves = live_leaves(&counters);
    let after_inodes = live_inodes(&counters);
    println!(
        "after: live leaves={after_leaves}, live inodes={after_inodes}, \
         pending={}, deferred={}, abandoned={}",
        tree.pending_coalesce(),
        tree.coalesce_deferred(),
        tree.coalesce_abandoned()
    );

    // The main tree is a single root leaf (all keys have one ikey). A working
    // sublayer GC would leave exactly that one leaf and zero internodes.
    assert_eq!(
        after_leaves, 1,
        "leaked {} sublayer leaf/leaves (abandoned={})",
        after_leaves - 1,
        tree.coalesce_abandoned()
    );
    assert_eq!(after_inodes, 0, "leaked promoted layer-root internode(s)");
}
