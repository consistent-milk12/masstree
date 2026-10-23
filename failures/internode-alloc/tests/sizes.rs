use masstree::{BoxPolicy, LeafNode15, InternodeNode};
#[test]
fn sizes() {
    println!("internode={} align={}", std::mem::size_of::<InternodeNode>(), std::mem::align_of::<InternodeNode>());
    println!("box_leaf={} align={}", std::mem::size_of::<LeafNode15<BoxPolicy<u64>>>(), std::mem::align_of::<LeafNode15<BoxPolicy<u64>>>());
}
