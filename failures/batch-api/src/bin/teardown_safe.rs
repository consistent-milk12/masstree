use masstree::{BoxPolicy, MassTree15, SeizeAllocator, TreeAllocator};

fn main() {
    let tree: MassTree15<u64> = MassTree15::new();
    tree.insert(b"k", 1);
    let g = tree.guard();
    // SAFETY: Quiescent inspection under the tree guard.
    let root = unsafe { tree.debug_root_ptr(&g) };
    let alloc: SeizeAllocator<BoxPolicy<u64>> = SeizeAllocator::new();
    // Entirely safe calls: frees every node of the still-live tree.
    // SAFETY: The call intentionally frees every node of the still-live tree;
    // this is the documented unsafe teardown contract.
    unsafe { alloc.teardown_tree(root) };
    let _ = tree.get(b"k"); // use-after-free, and tree Drop will teardown again
}
