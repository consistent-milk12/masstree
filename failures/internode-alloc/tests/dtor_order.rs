use std::cell::RefCell;
struct A(&'static str);
impl Drop for A { fn drop(&mut self) { println!("drop {}", self.0); } }
thread_local! { static X: RefCell<Option<A>> = const { RefCell::new(None) }; }
thread_local! { static Y: RefCell<Option<A>> = const { RefCell::new(None) }; }
#[test]
fn dtor_order() {
    std::thread::spawn(|| {
        X.with(|c| *c.borrow_mut() = Some(A("X-first")));
        Y.with(|c| *c.borrow_mut() = Some(A("Y-second")));
        println!("worker done");
    }).join().unwrap();
}
