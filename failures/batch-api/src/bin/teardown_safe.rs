use masstree::{BoxPolicy, MassTree15, SeizeAllocator, TreeAllocator};

fn main() {
    let tree: MassTree15<u64> = MassTree15::new();
    tree.insert(b"k", 1);

    let root = {
        let g = tree.guard();
        // SAFETY: quiescent inspection under the tree guard.
        unsafe { tree.debug_root_ptr(&g) }
    };

    let alloc: SeizeAllocator<BoxPolicy<u64>> = SeizeAllocator::new();
    // SAFETY: the tree is being destroyed. No guard or reader remains, and
    // `mem::forget` prevents Drop from tearing it down a second time.
    unsafe { alloc.teardown_tree(root) };
    std::mem::forget(tree);

    println!("teardown contract honored");
}
