# Independent differential/Miri harness

This is the independent harness used to cross-check the subsystem reports. It
contains both broad differential tests and minimal repros.

| Binary | Finding | Expected |
|---|---|---|
| `repros` | RNG-01/02/04, RNG-08 | prints incorrect forward/reverse bounds, promoted-layer start skip, and empty-rightmost reverse |
| `insbugs` | INS-01/02 | prints a lost key after prepend split; two `SplitFailed` panics |
| `gc` | GC-02 | `coalesce_abandoned()>0` and an empty `SUBLAYER` remains after maintenance |
| `uaf` | SAFE-01 | Miri reports `ValuePtr` use-after-free |
| `race` | SAFE-02 | Miri reports non-atomic read/atomic store race |
| `main` | RNG-01/05, INS-01, batch | broad randomized differential + batch bookkeeping |

```bash
cargo run --quiet --bin repros
cargo run --quiet --bin insbugs
cargo run --quiet --bin gc
cargo +nightly miri run --bin uaf
cargo +nightly miri run --bin race
```

`insbugs` prints expected panics; continue after each. `gc` is deterministic
and prints counters.
