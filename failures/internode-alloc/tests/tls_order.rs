use std::cell::RefCell;
use masstree::MassTree15;

thread_local! {
    static TREE: RefCell<Option<MassTree15<u64>>> = const { RefCell::new(None) };
}

#[test]
fn tls_destructor_order() {
    let tree: MassTree15<u64> = MassTree15::new();
    for i in 0..64u64 { tree.insert(&i.to_be_bytes(), i); }
    let h = std::thread::spawn(move || {
        // First access registers TREE dtor; tree creation happened on main thread,
        // so this thread has not initialized POOL yet.
        TREE.with(|c| *c.borrow_mut() = Some(tree));
        TREE.with(|c| {
            let t = c.borrow();
            let t = t.as_ref().unwrap();
            for i in 0..64u64 { t.remove(&i.to_be_bytes()).unwrap(); }
            let g = t.guard();
            t.process_coalesce(&g);
        });
        println!("worker exiting");
    });
    let _ = h.join();
    println!("joined");
}
