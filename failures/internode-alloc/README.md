# Internode/allocation/reclamation reproducers

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
