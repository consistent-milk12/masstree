use std::cell::RefCell;
use masstree::MassTree15Inline;

struct Probe;
impl Drop for Probe {
    fn drop(&mut self) {
        println!("probe drop: allocating tree");
        let t: MassTree15Inline<u64> = MassTree15Inline::new();
        t.insert(&0u64.to_be_bytes(), 1);
        println!("probe drop: ok");
        std::mem::forget(t);
    }
}
thread_local! { static TREE: RefCell<Option<MassTree15Inline<u64>>> = const { RefCell::new(None) }; }
thread_local! { static PROBE: RefCell<Option<Probe>> = const { RefCell::new(None) }; }
#[test]
fn probe_order() {
    std::thread::spawn(|| {
        TREE.with(|c| *c.borrow_mut() = Some(MassTree15Inline::new()));
        TREE.with(|c| { let t=c.borrow(); let t=t.as_ref().unwrap(); for i in 0..64u64 { t.insert(&i.to_be_bytes(), i); } for i in 0..64u64 { t.remove(&i.to_be_bytes()).unwrap(); } let g=t.guard(); t.process_coalesce(&g); });
        PROBE.with(|c| *c.borrow_mut() = Some(Probe));
        println!("worker done");
    }).join().unwrap();
}
