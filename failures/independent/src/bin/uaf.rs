use masstree::MassTree15;

fn main() {
    let tree: MassTree15<String> = MassTree15::new();
    {
        let guard = tree.guard();
        tree.insert_with_guard(b"k", "hello".to_string(), &guard);
    }
    let ptr = {
        let guard = tree.guard();
        tree.get_with_guard(b"k", &guard).unwrap()
    };
    drop(tree);
    println!("{}", *ptr);
}
