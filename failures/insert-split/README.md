# Insert/split reproducers

| Target | Finding | Expected |
|---|---|---|
| `tests/prepend.rs` | INS-01 | `get("AAAAAAAA")` returns `None` after the prepend split, although a scan lists it |
| `tests/prepend_impact.rs` | INS-01 | reinsert accepts the orphan, returns `None`, and `len()` becomes 17 for 16 distinct keys |
| `tests/equal_run.rs` | INS-02 | `insert` panics with `SplitFailed` for an ikey run at position 0 |
| `tests/equal_run2.rs` | INS-02 | second split-point path panics with `SplitFailed` |
| `src/bin/livelock2.rs` | INS-03 | insert/coalesce stress eventually stops making progress at 100% CPU |
| `src/bin/fuzz.rs` | control | single-threaded insert/get/remove/len oracle over short NUL keys passes |

```bash
cargo test --test prepend -- --nocapture
cargo test --test equal_run -- --nocapture
timeout 120s cargo run --release --bin livelock2
```

`livelock2` needs many rounds; a timeout or a watchdog stop at a round count is
the reproduction, not a normal completion.
