use masstree::MassTree15;
#[derive(Clone)]
struct Boom;
impl Drop for Boom { fn drop(&mut self) { panic!("boom value drop"); } }
#[test]
fn panicking_value_drop() {
    let tree: MassTree15<Boom> = MassTree15::new();
    tree.insert(b"k", Boom);
    tree.insert(b"k", Boom);
    println!("dropping tree");
    drop(tree);
    println!("tree dropped");
}
