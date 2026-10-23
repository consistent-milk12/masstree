use masstree::{BoxPolicy, MassTree15, SeizeAllocator, TreeAllocator};

fn main() {
    let tree: MassTree15<u64> = MassTree15::new();
    tree.insert(b"k", 1);
    let g = tree.guard();
    let root = tree.debug_root_ptr(&g);
    let alloc: SeizeAllocator<BoxPolicy<u64>> = SeizeAllocator::new();
    // Entirely safe calls: frees every node of the still-live tree.
    alloc.teardown_tree(root);
    let _ = tree.get(b"k"); // use-after-free, and tree Drop will teardown again
}
