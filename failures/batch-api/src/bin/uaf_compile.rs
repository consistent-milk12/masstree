use masstree::MassTree15;

fn main() {
    let tree: MassTree15<String> = MassTree15::new();
    tree.insert(b"k", "hello".to_string());
    // No lifetime ties this pointer to the guard or tree.
    let escaped = {
        let g = tree.guard();
        tree.get_with_guard(b"k", &g).unwrap()
    };
    drop(tree);
    // Safe `Deref` after the tree (and its EBR collector) is gone.
    println!("{}", escaped.as_str());
}
