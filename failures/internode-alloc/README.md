# Internode/allocation/reclamation reproducers

`cold_pool` is a passing PERF-15 regression. A counting allocator observes 16
leaf allocations at first insert on each fresh thread (previously 128). Each
thread then grows to 5,000 keys and checks every value. The local seven-thread
median was 6.87 µs before and 5.35 µs after; timing is reported, not asserted.

```bash
CARGO_TARGET_DIR=/tmp/masstree-target cargo run --release --bin cold_pool
```

The table below records historical audit behavior.

| Target | Finding | Expected |
|---|---|---|
| `tests/tls_abort.rs` | ALLOC-01 | tree in `thread_local!`; worker exit panics on `POOL.with` and then SIGSEGV/SIGABRT |
| `tests/panic_value.rs` | ALLOC-02 | panicking value `Drop` during reclamation causes a second panic/UB and abort |
| `tests/stuck.rs`, `tests/stuck2.rs` | GC-03/OCC-04 | deleted/incomplete internode traversal does not terminate |
| `tests/dup_chain.rs`, `tests/dup_chain_maint.rs` | GC-01 | duplicate chain entries wedge maintenance or target retired nodes |
| `tests/dtor_order.rs`, `tests/tls_order.rs` | ALLOC-01 | TLS destructor-order variants |
| `tests/probe_order.rs` | GC-03 | retire-before-unlink ordering probe |
| `tests/promote.rs` | GC-02 | promoted-layer object graph / reclamation |
| `tests/sizes.rs`, `tests/inspect.rs`, `tests/leakcheck.rs` | — | layout/pool/teardown controls |

```bash
cargo test --release --test tls_abort -- --nocapture
cargo test --release --test panic_value -- --nocapture
timeout 30s cargo test --release --test double_unlink -- --nocapture
```

Aborts and hangs are the expected observations. Use `timeout` and a disposable
process.
