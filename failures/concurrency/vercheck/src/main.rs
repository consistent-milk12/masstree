use masstree::{BoxPolicy, LeafNode15, MassTree15};
use seize::LocalGuard;

fn leaf(tree: &MassTree15<u64>, guard: &LocalGuard<'_>) -> *mut LeafNode15<BoxPolicy<u64>> {
    tree.debug_root_ptr(guard).cast()
}
fn main() {
    let tree: MassTree15<u64> = MassTree15::new();
    let guard = tree.guard();
    let l = leaf(&tree, &guard);
    let v0 = unsafe { &*l }.version().value();
    let p0 = unsafe { &*l }.permutation_raw();
    tree.insert_with_guard(b"a", 1, &guard);
    let v1 = unsafe { &*l }.version().value();
    let p1 = unsafe { &*l }.permutation_raw();
    tree.insert_with_guard(b"b", 2, &guard);
    let v2 = unsafe { &*l }.version().value();
    let p2 = unsafe { &*l }.permutation_raw();
    tree.insert_with_guard(b"a", 3, &guard);
    let v3 = unsafe { &*l }.version().value();
    let p3 = unsafe { &*l }.permutation_raw();
    eprintln!("v0={v0:#010x} p0={p0:#018x}");
    eprintln!("v1={v1:#010x} p1={p1:#018x}");
    eprintln!("v2={v2:#010x} p2={p2:#018x}");
    eprintln!("v3={v3:#010x} p3={p3:#018x}");
}
