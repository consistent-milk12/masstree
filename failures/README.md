# Reproducible failure corpus

This directory contains the runnable reproducers used for `SystemAudit.md`.
Every crate is standalone and path-depends on the repository root with
`path = "../.."` or `path = "../../.."`. Build artifacts are intentionally not
checked in.

> **Safety warning.** These programs intentionally trigger ordinary test
> failures, hangs, aborts, data races, use-after-free, and double-free. Run them
> in a disposable process and use `timeout` for the stress/hang cases. Do not
> run the whole directory in one process.

Most crates are deliberately small tests, not a stable library. The tests that
end in `FAILED`, exit 42/101, or abort are *successful reproductions*.

## Layout

| Directory | Canonical findings | What it contains |
|---|---|---|
| `range/` | RNG-01 … RNG-08 | forward/reverse/prefix/values differential repros, promoted-layer and empty-rightmost cases |
| `insert-split/` | INS-01, INS-02, INS-03 | prepend split, split-point panic, two-thread insert/coalesce livelock |
| `remove-coalesce/` | REM-01, REM-02, REM-03, GC-01, GC-02 | wrong-key removal, lost remove during split, duplicate chain entries, promoted-layer leak |
| `batch-api/` | SAFE-03, BATCH-01 … BATCH-05 (BATCH-06 is docs-only) | batch aliasing/double-free, stale `BatchEntry.key`, duplicate order, leaks, teardown |
| `suffix/` | SUF-01 … SUF-05, SAFE-06, PERF-SUF | `SuffixBag` public fuzz, allocation prose, suffix boundary probes |
| `primitives/nodeversion/` | OCC-01, OCC-02, OCC-03, PRIM-02 | version-word probes, layer-read race, permuter raw bits |
| `primitives/shard/` | PRIM-01 | concurrent `ShardedCounter::load` debug assertion |
| `internode-alloc/` | GC-03, ALLOC-01, ALLOC-02, SAFE-04 | cascade ordering, TLS teardown abort, panic-in-reclaimer, allocator probes |
| `concurrency/` | OCC-01, OCC-02, REM-01, GC-01, SAFE-01, SAFE-02 | version/type-confusion, remove race, lifetime escape, guard/pointer checks |
| `leaf-values/` | SAFE-01, SAFE-02, SAFE-03, SAFE-04, SAFE-05; private OCC/REM tests | public Miri bins plus patch-ready crate-internal tests |
| `independent/` | RNG-01/02/03/04/08, INS-01/02, GC-02, SAFE-01/02 | the independent differential/Miri/repro harness |
| `tooling/` | TOOL-01 … TOOL-10 | scripts that assert build/lint/test/benchmark workflow defects |

## Quick start

For all crates, prefer a target directory outside the repository:

```bash
export CARGO_TARGET_DIR=/tmp/masstree-failures-target
```

### Range

```bash
cd failures/range
cargo run --quiet --bin nul          # forward bounds within one ikey
cargo run --quiet --bin nulpfx       # scan_prefix over NUL keys
cargo run --quiet --bin layerend     # reverse bound on a layer chunk
cargo run --quiet --bin promoted     # promoted-layer start skip
cargo run --quiet --bin vals         # values-only empty/range bounds
cargo run --quiet --bin valfar       # values-only layer past end
cargo run --quiet --bin maxprefix    # values-only prefix leaks adjacent key
cargo run --quiet --bin doubleended  # DoubleEnded re-emits after exhaustion
cargo run --quiet --bin empty_right  # empty rightmost leaf breaks reverse/last
cargo run --quiet --bin apis         # API variant matrix
cargo run --quiet --bin race         # remove+reinsert scan ordering (may take a few rounds)
```

`promoted` prints `inc_ok=false` and `pfx full: got=0 exp=1` for indexes at
or after 15. `vals` prints non-empty values for empty/inverted ranges.
`maxprefix` asserts that `scan_prefix_values` returns one value but receives
two (including an adjacent, non-matching key).

### Insert/split

```bash
cd failures/insert-split
cargo test --test prepend -- --nocapture          # FAILED: key becomes unreachable
cargo test --test prepend_impact -- --nocapture   # FAILED: count drift, duplicate
cargo test --test equal_run -- --nocapture        # FAILED: SplitFailed panic
cargo test --test equal_run2 -- --nocapture       # FAILED: SplitFailed panic
timeout 120s cargo run --release --bin livelock2   # may wedge in insert/coalesce
cargo run --release --bin fuzz                    # single-thread oracle (passes)
```

`livelock2` can take hundreds of thousands of rounds. Use `timeout` and treat
either the watchdog/hang or a 100% CPU non-progress state as the reproduction.

### Remove/coalesce

```bash
cd failures/remove-coalesce
cargo test --release --test wrong_key -- --nocapture
cargo test --release --test lost_remove -- --nocapture
cargo test --release --test double_unlink_deterministic -- --nocapture
cargo test --release --test sublayer_leak -- --nocapture
timeout 30s cargo test --release --test double_unlink -- --nocapture
```

`wrong_key` and `lost_remove` fail through their own assertions.
`double_unlink_deterministic` exits 42 after its watchdog fires.
`sublayer_leak` fails because live leaf/internode counts do not return to
baseline.

### Batch/API

```bash
cd failures/batch-api
cargo run --quiet --bin alias
cargo run --quiet --bin doublefree
cargo run --quiet --bin mut
cargo run --quiet --bin leak2
cargo run --quiet --bin leak3
cargo run --quiet --bin teardown_safe
```

`alias`, `doublefree`, and `teardown_safe` intentionally abort or corrupt
memory. Run them one at a time.

### Suffix

```bash
cd failures/suffix
cargo run --quiet
```

The probe is a public-API `SuffixBag`/tree workload. See the audit for the
exact `compact` panic sequence and the allocation measurements.

### Primitives

```bash
cd failures/primitives/nodeversion
cargo run --quiet                         # version-word/layer-read micro probes
cargo run --release --bin layer_race      # transient misses; may take ~200k pairs
cargo run --quiet --bin perm_raw          # Permuter<WIDTH<15> raw bits
cd ../shard
cargo run --quiet                         # debug-assertion panic under contention
```

### Internode/allocation

```bash
cd failures/internode-alloc
cargo test --release --test tls_abort -- --nocapture
cargo test --release --test panic_value -- --nocapture
cargo test --release --test stuck -- --nocapture
cargo test --release --test dup_chain -- --nocapture
```

`tls_abort` ends with a TLS-destruction panic followed by SIGSEGV/SIGABRT;
`panic_value` intentionally panics in a value `Drop`; `stuck`/`dup_chain`
exercise maintenance hangs and stale pointers.

### Concurrency

```bash
cd failures/concurrency
(cd layer_race && cargo run --release)
(cd typeconf && cargo run --release)
(cd remove_race && cargo run --release)
(cd vercheck && cargo run --quiet)
(cd lifetime && cargo +nightly miri run)
```

`lifetime` is the minimal Miri use-after-free. `layer_race` and `typeconf`
need a few hundred thousand operations to hit a miss/type-confused read.

### Leaf/values Miri binaries

```bash
cd failures/leaf-values/public
cargo +nightly miri run --bin valueptr_escape
cargo +nightly miri run --bin valueptr_race
cargo +nightly miri run --bin padded_write_through
cargo +nightly miri run --bin move_value_same_leaf
cargo +nightly miri run --bin batch_write_through_leak
```

Each command should report `Undefined Behavior`, `memory leaked`, or a data
race. `private-tests/` contains the crate-internal tests that require a
temporary checkout; see its README.

### Tooling

```bash
cd failures/tooling
./run-all.sh
```

Each script exits 0 when the defect is reproduced and non-zero when the
repository has changed enough that it no longer reproduces.

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
`SystemAudit.md`.
