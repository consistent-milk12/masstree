use std::alloc::System;

#[global_allocator]
static MY_ALLOC: System = System;

fn main() {
    let tree: masstree::MassTree15<u64> = masstree::MassTree15::new();
    tree.insert(b"k", 1);
    println!("{:?}", tree.get(b"k"));
}
