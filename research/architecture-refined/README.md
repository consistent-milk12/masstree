# Architecture research evidence

Supporting experiments for [ARCHITECTURERefined.md](../../ARCHITECTURERefined.md).
This standalone Cargo workspace does not change the production crate.

## Metadata model

From the repository root:

```sh
cargo test --offline --locked --manifest-path research/architecture-refined/Cargo.toml --lib
```

The model checks one writer changing two atomic metadata fields. The fenced
protocol must not accept a torn pair. Three negative controls remove one or
both fences and must panic with the specific torn-pair assertion. Thus four
passing tests mean one positive model passes and three defective models are
detected. This is not a model of the real tree, pointer lifetimes, suffixes,
structural transactions, or the complete Rust memory model.

## Current-tree layout experiment

```sh
python3 research/architecture-refined/run_probe.py --cpu 2 --out /tmp/masstree-layout-results
```

Choose an available CPU; omit `--cpu` to use the lowest allowed CPU. Linux
`taskset`, Python 3, Rust and cached Cargo dependencies are required. The
runner builds all configurations before timing, copies their binaries to a
temporary directory, and runs nine shuffled process blocks of five trials
each. `Cargo.lock` pins this workspace's dependencies. The runner explicitly
uses this workspace's ignored `target` directory.

The existing inline tree uses 100,000 shuffled 8-byte keys and a deterministic
two-million-operation trace: roughly 60% reads, 20% existing-key replacements,
20% forward scans of 32 entries. A sequential value-array model supplies the
expected checksum, scan item count and final values. Setup, warmup and final
point-read validation are untimed; checksum accumulation is timed.

The experiment does not measure concurrent updates, insertion/removal churn,
long keys, reverse scans, total memory, latency distributions or a replacement
implementation. Its checksum/count checks are useful diagnostics, not a
substitute for exact per-scan differential tests. Report the complete sample
distribution, not only the best run. No Rust/C++ throughput comparison was
performed.

## Archived observations

`evidence/current-rust/` contains logs and JSONL command manifests from the
current-worktree investigation. `evidence/layout/` contains all 45 measured
trials, schedule, environment and build logs. `evidence/occ-model.log` records
the four-test model run. `evidence/cpp/` records C++ build and smoke results.

The command manifests preserve the original working directories and environment
overrides. Adapt their repository root to reproduce elsewhere; their `log`
fields point to archived relative files. A zero exit from a diagnostic failure
binary does **not** imply correctness: several print a mismatch and exit zero.
Miri errors and watchdog exit 42 are expected evidence of current bugs. Never
run the dangerous corpus in one process; use a timeout per command and a
target directory outside the repository.

The C++ source was cloned to `/tmp/masstree-reference`, from
`https://github.com/kohler/masstree-beta.git`, commit
`11198427a1170654ca646dd20d96c8f349bca2bd`. To build on a host with autoconf,
automake, m4 and a C++ compiler:

```sh
git clone https://github.com/kohler/masstree-beta.git /tmp/masstree-reference
git -C /tmp/masstree-reference checkout 11198427a1170654ca646dd20d96c8f349bca2bd
cd /tmp/masstree-reference
./bootstrap.sh
./configure --enable-assertions --with-malloc=malloc
make -j4 mttest
timeout 30s ./mttest -j2 -d1 -l50000 --no-notebook rw1 rw3 scan1 rscan1
```

This reference's separate `unit-mt` and `scantest` targets have stale epoch
declarations and fail compilation. Their failure log is retained. The initial
smoke run used disabled assertions; the checked rebuild and successful repeat
of all four workloads are recorded separately.
