# Reproducible failure corpus

This directory contains the reproducers that drove `SystemAudit.md`. Every
crate is standalone and path-depends on the repository root with
`path = "../.."` or `path = "../../.."`. Build artifacts are intentionally not
checked in.

> **Post-remediation status.** Every P0/P1 finding in `SystemAudit.md` has been
> fixed (see §2.5 there). The corpus is retained as regression coverage: the
> programs that used to fail, hang, abort, or report Miri UB now pass, and the
> `failures/tooling` scripts assert that the tooling fixes stay in place. A few
> reproducers are intentionally compile-fail regressions because the unsound
> API they exercised no longer exists (for example, escaping a guard-scoped
> `ValueRef`).

## Layout

| Directory | Findings covered | What it contains |
|---|---|---|
| `range/` | RNG-01 … RNG-10 | forward/reverse/prefix/values differential repros, promoted-layer and empty-rightmost cases |
| `insert-split/` | INS-01, INS-02, INS-03, INS-04 | prepend split, split-point panic, insert/coalesce livelock, encoder panic |
| `remove-coalesce/` | REM-01, REM-02, REM-03, GC-01, GC-02, GC-03, GC-04, GC-05 | wrong-key removal, lost remove during split, duplicate chain entries, promoted-layer leak, cascade ordering |
| `batch-api/` | SAFE-03, BATCH-01 … BATCH-06 | batch ownership/aliasing, duplicate order, stale key cache, leaks, teardown |
| `suffix/` | SUF-01 … SUF-05, SAFE-06 | `SuffixBag` public workload, suffix boundary/compaction probes |
| `primitives/nodeversion/` | OCC-01, OCC-02, OCC-03, PRIM-02 | version-word probes, layer-read race, permuter raw bits |
| `primitives/shard/` | PRIM-01 | concurrent `ShardedCounter::load` snapshot |
| `internode-alloc/` | GC-03, ALLOC-01, ALLOC-02, SAFE-04 | cascade ordering, TLS teardown, panic-in-reclaimer, allocator probes |
| `concurrency/` | OCC-01, OCC-02, REM-01, GC-01, SAFE-01, SAFE-02 | version/type-confusion, remove race, lifetime/guard checks |
| `leaf-values/` | SAFE-01 … SAFE-06 | public Miri bins plus patch-ready crate-internal tests |
| `independent/` | RNG-01/02/03/04/08, INS-01/02, GC-02, SAFE-01/02 | independent differential/Miri/repro harness |
| `tooling/` | TOOL-01 … TOOL-14 | scripts that assert the build/lint/test/benchmark/tracing fixes stay in place |

## Quick start

For all crates, prefer a target directory outside the repository:

```bash
export CARGO_TARGET_DIR=/tmp/masstree-failures-target
```

### Range (expected: pass)

```bash
cd failures/range
cargo run --quiet --bin nul          # forward bounds within one ikey
cargo run --quiet --bin nulpfx       # scan_prefix over NUL keys
cargo run --quiet --bin layerend     # reverse bound on a layer chunk
cargo run --quiet --bin promoted     # promoted-layer start key
cargo run --quiet --bin vals         # values-only empty/range bounds
cargo run --quiet --bin valfar       # values-only layer past end
cargo run --quiet --bin maxprefix    # values-only max-length prefix
cargo run --quiet --bin doubleended  # DoubleEnded exhaustion
cargo run --quiet --bin empty_right  # empty rightmost leaf reverse/last
cargo run --quiet --bin apis         # API variant matrix
cargo run --quiet --bin race         # remove+reinsert scan ordering
```

All outputs should match the exact-key expectations printed by the binaries.
`nul` no longer emits the excluded bound, `promoted` includes the requested
start key and prefix, `vals`/`valfar`/`maxprefix` do not emit out-of-range
values, `doubleended` returns an empty opposite direction, and `empty_right`
returns the 15 remaining keys in reverse.

### Insert/split (expected: pass)

```bash
cd failures/insert-split
cargo test --test prepend -- --nocapture
cargo test --test prepend_impact -- --nocapture
cargo test --test equal_run -- --nocapture
cargo test --test equal_run2 -- --nocapture
timeout 120s cargo run --release --bin livelock2
cargo run --release --bin fuzz
```

The prepend/equal-run tests now complete with every key reachable; `livelock2`
makes progress; `fuzz` remains a single-threaded oracle.

### Remove/coalesce (expected: pass)

```bash
cd failures/remove-coalesce
cargo test --release --test wrong_key -- --nocapture
cargo test --release --test lost_remove -- --nocapture
cargo test --release --test double_unlink_deterministic -- --nocapture
cargo test --release --test sublayer_leak -- --nocapture
timeout 30s cargo test --release --test double_unlink -- --nocapture
```

`wrong_key`/`lost_remove` find the correct key, `double_unlink*` terminate, and
`sublayer_leak` returns to one live leaf and zero live internodes with no
abandoned entries.

### Batch/API (expected: pass)

```bash
cd failures/batch-api
cargo run --quiet --bin alias
cargo run --quiet --bin doublefree
cargo run --quiet --bin mut
cargo run --quiet --bin leak2
cargo run --quiet --bin leak3
cargo run --quiet --bin teardown_safe
```

`alias`/`doublefree` demonstrate distinct ownership and exactly-once drops;
`mut` uses private/owned entries; `leak3` reports no drop leak; `teardown_safe`
uses the documented unsafe teardown contract.

### Suffix (expected: pass)

```bash
cd failures/suffix
cargo run --quiet
```

The public `SuffixBag`/tree workload no longer panics or corrupts metadata.

### Primitives (expected: pass)

```bash
cd failures/primitives/nodeversion
cargo run --quiet
cargo run --release --bin layer_race
cargo run --quiet --bin perm_raw
cd ../shard
cargo run --quiet
```

The version-word probes observe the fixed insert/validation protocol, the
layer race no longer returns false misses, and the permuter/shard-counter
probes exit cleanly.

### Internode/allocation (expected: pass)

```bash
cd failures/internode-alloc
cargo test --release --test tls_abort -- --nocapture
cargo test --release --test panic_value -- --nocapture
cargo test --release --test stuck -- --nocapture
cargo test --release --test dup_chain -- --nocapture
```

TLS destruction falls back safely, a panicking value `Drop` is contained, and
maintenance terminates without stale pointers.

### Concurrency (expected: pass or compile-fail)

```bash
cd failures/concurrency
(cd layer_race && cargo run --release)
(cd typeconf && cargo run --release)
(cd remove_race && cargo run --release)
(cd vercheck && cargo run --quiet)
(cd lifetime && cargo run --quiet)   # compile-fail: the escape is rejected
```

`lifetime` is intentionally a compile-fail regression: the old program returned
a guard-scoped value from its guard. The other binaries verify correct reads
and removes under contention.

### Leaf/values Miri binaries (expected: clean)

```bash
cd failures/leaf-values/public
cargo +nightly miri run --bin valueptr_escape
cargo +nightly miri run --bin valueptr_race
cargo +nightly miri run --bin padded_write_through
cargo +nightly miri run --bin move_value_same_leaf
cargo +nightly miri run --bin batch_write_through_leak
```

Each command should complete without `Undefined Behavior`, `memory leaked`, or
a data race. `private-tests/` contains crate-internal tests that require a
temporary checkout; see its README.

### Tooling (expected: pass)

```bash
cd failures/tooling
./run-all.sh
```

Each script exits 0 when the fixed behavior is present and non-zero when a
defect has regressed.

## Evidence labels used by `SystemAudit.md`

- **Confirmed** — reproduced by one of these binaries/tests, by Miri, or by a
  deterministic test requiring crate-private access.
- **Likely** — the full interleaving is mechanically constructed but no
  scheduler/hardware reproduction was obtained.
- **Speculative** — contract-level or defensive finding that needs a caller
  outside the current tree.

## Relationship to the reports

The detailed subsystem reports are retained under
`/tmp/opencode/audit/*.md` for this session. The merged, canonical findings,
severity/confidence labels, fixes, and test gaps are in the repository-root
`SystemAudit.md`; §2.5 records the remediation status and commit for each.
