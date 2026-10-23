use std::cell::RefCell;
use masstree::MassTree15Inline;
thread_local! { static TREE: RefCell<Option<MassTree15Inline<u64>>> = const { RefCell::new(None) }; }
#[test]
fn tls_tree_abort() {
    let tree: MassTree15Inline<u64> = MassTree15Inline::new();
    let h = std::thread::spawn(move || {
        TREE.with(|c| *c.borrow_mut() = Some(tree));
        TREE.with(|c| {
            let t = c.borrow();
            let t = t.as_ref().unwrap();
            for i in 0..64u64 { t.insert(&i.to_be_bytes(), i); }
            for i in 0..64u64 { t.remove(&i.to_be_bytes()).unwrap(); }
            let g = t.guard();
            t.process_coalesce(&g);
            println!("retired, pending={} abandoned={}", t.pending_coalesce(), t.coalesce_abandoned());
        });
        println!("worker exiting");
    });
    let r = h.join();
    println!("joined ok={}", r.is_ok());
}
