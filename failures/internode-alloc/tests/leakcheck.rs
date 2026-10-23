#[test]
fn intentional_leak() {
    let v = Box::new([0u8; 64]);
    std::mem::forget(v);
}
