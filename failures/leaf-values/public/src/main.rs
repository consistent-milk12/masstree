fn main() {
    eprintln!("Run one of the failure binaries, e.g.:");
    eprintln!("  cargo run --bin padded_write_through");
    eprintln!("  cargo +nightly miri run --bin valueptr_escape");
}
