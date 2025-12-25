# Justfile for masstree development
# Install just: cargo install just
# Run `just` to see available commands

# Default recipe - show help
default:
    @just --list

# Build the project
build:
    cargo build

# Build in release mode
build-release:
    cargo build --release

# Run all tests with nextest (saves failures to file if any)
test:
    #!/usr/bin/env bash
    set -o pipefail
    cargo nextest run --no-fail-fast 2>&1 | tee .test-output.tmp
    exit_code=$?
    if [ $exit_code -ne 0 ]; then
        # Ensure failures directory exists
        mkdir -p failures
        # Extract only FAIL sections with their details (stdout/stderr blocks)
        outfile="failures/$(date +%Y%m%d-%H%M%S).txt"
        awk '/^[[:space:]]*FAIL/{found=1} found{print} /^────────────$/{if(found) found=0} /^[[:space:]]*Summary/{found=1}' .test-output.tmp > "$outfile"
        echo "Failures saved to $outfile"
    fi
    rm -f .test-output.tmp
    exit $exit_code

# Run all tests with nextest, tracing, and mimalloc (prioritized mode)
# This is the recommended way to run tests for development
next:
    cargo nextest run --no-fail-fast --features "tracing,mimalloc"

# Run all tests with trace-level logging, then pretty-print logs to JSON
next-trace:
    #!/usr/bin/env bash
    rm -f logs/masstree.jsonl logs/masstree.json
    RUST_LOG=trace MASSTREE_LOG_CONSOLE=0 cargo nextest run --no-fail-fast --features "mimalloc,tracing"
    if [ -f logs/masstree.jsonl ]; then
        jaq -s '.' logs/masstree.jsonl > logs/masstree.json
        rm -f logs/masstree.jsonl
        echo "Logs written to logs/masstree.json"
    fi

# Run a specific test with nextest, tracing, and mimalloc
next-one TEST:
    cargo nextest run --no-fail-fast --features "tracing,mimalloc" {{TEST}}

# Run stress test until failure (for debugging race conditions)
# Uses nextest with mimalloc - runs tests in separate processes for more parallel stress
stress-until-fail:
    #!/usr/bin/env bash
    set -uo pipefail
    i=0
    while true; do
        i=$((i + 1))
        echo -n "Run $i: "
        output=$(cargo nextest run --features mimalloc --test concurrent_regression stress_concurrent_insert_many_keys 2>&1)
        if echo "$output" | grep -qE "FAIL|CRITICAL: Key not found"; then
            echo "FOUND BUG!"
            echo "$output" | grep -E "FAIL|CRITICAL|ERROR|immediate_verify_failures|Missing" | head -10
            echo ""
            echo "To debug: RUST_LOG=masstree=debug cargo test --features tracing --test concurrent_regression stress_concurrent_insert_many_keys -- --nocapture"
            break
        else
            echo "ok"
        fi
    done

# Run nextest N times to catch intermittent failures
# Usage: just next-repeat 20
next-repeat N="10":
    #!/usr/bin/env bash
    set -uo pipefail
    passed=0
    failed=0
    for i in $(seq 1 {{N}}); do
        echo "=== Run $i/{{N}} ==="
        output=$(cargo nextest run --no-fail-fast --features "tracing,mimalloc" 2>&1)
        status=$?
        if [ $status -eq 0 ]; then
            echo "PASS"
            passed=$((passed + 1))
        else
            echo "FAIL"
            echo "$output" | rg -n "(FAIL|Summary|error:|panicked at)" | head -n 20 || true
            mkdir -p logs/next-repeat
            if [ -f logs/masstree.jsonl ]; then
                cp -f logs/masstree.jsonl "logs/next-repeat/run-${i}.jsonl"
                echo "Saved logs to logs/next-repeat/run-${i}.jsonl"
            fi
            printf '%s\n' "$output" > "logs/next-repeat/run-${i}.out"
            echo "Saved output to logs/next-repeat/run-${i}.out"
            failed=$((failed + 1))
        fi
    done
    echo ""
    echo "Results: $passed passed, $failed failed out of {{N}} runs"
    if [ $failed -gt 0 ]; then
        exit 1
    fi

# Run all tests with short output format
test-short:
    cargo test --message-format=short

# Run tests with output
test-verbose:
    cargo test -- --nocapture

# Run only unit tests (lib tests)
test-unit:
    cargo test --lib

# Run only doc tests
test-doc:
    cargo test --doc

# Run only integration tests (tests/ folder)
test-integration:
    cargo test --test '*'

# Run a specific test by name
test-one TEST:
    cargo test {{TEST}} -- --nocapture

# Run ALL possible tests (unit, doc, integration, loom, shuttle, miri)
# This is the most comprehensive test command
test-all: test test-loom test-shuttle miri-strict
    @echo "All tests passed!"

# Run loom tests for deterministic concurrency verification
# Loom explores all possible thread interleavings
test-loom:
    RUSTFLAGS="--cfg loom" cargo test --lib nodeversion::loom_tests
    RUSTFLAGS="--cfg loom" cargo test --lib tree::loom_tests

# Run shuttle linearizability tests
# Shuttle tests concurrent operations for correctness
test-shuttle:
    cargo test --lib tree::shuttle_tests

# Run clippy lints
lint:
    cargo clippy --all-targets --all-features

# Run clippy and fail on warnings
lint-strict:
    cargo clippy --all-targets --all-features -- -D warnings

# Format code
fmt:
    cargo fmt

# Check formatting
fmt-check:
    cargo fmt --all -- --check

# Run all checks (format, lint, test)
check-all: fmt-check lint-strict test

# Generate documentation
doc:
    cargo doc --no-deps --open

# Generate documentation including private items
doc-private:
    cargo doc --no-deps --document-private-items --open

# Clean build artifacts
clean:
    cargo clean

# === Miri commands (require nightly) ===

# Install nightly toolchain with miri (run once)
# Commented out - nightly and miri already installed
# miri-setup:
#     rustup toolchain install nightly --component miri
#     rustup run nightly cargo miri setup

# Run unit tests under Miri to detect undefined behavior
# Note: Only runs --lib tests; proptest is too slow under Miri (~100x overhead)
miri:
    cargo +nightly miri test --lib

# Run a specific test under Miri
miri-test TEST:
    cargo +nightly miri test {{TEST}}

# Run Miri with stricter checks (Stacked Borrows)
miri-strict:
    MIRIFLAGS="-Zmiri-strict-provenance" cargo +nightly miri test --lib

# Run Miri checking for memory leaks
miri-leaks:
    MIRIFLAGS="-Zmiri-strict-provenance -Zmiri-symbolic-alignment-check" cargo +nightly miri test --lib

# Run Miri with Tree Borrows (experimental, more permissive than Stacked Borrows)
miri-tree-borrows:
    MIRIFLAGS="-Zmiri-tree-borrows" cargo +nightly miri test --lib

# Run ALL tests under Miri (slow! includes proptest)
# Note: -Zmiri-disable-isolation required for proptest (uses getcwd)
miri-all:
    MIRIFLAGS="-Zmiri-disable-isolation" cargo +nightly miri test

# === Benchmarks ===

# Run benchmarks (requires uncommenting [[bench]] in Cargo.toml)
bench:
    cargo bench

# Run benchmarks with native CPU optimizations (AVX2, etc.)
bench-native:
    RUSTFLAGS="-C target-cpu=native" cargo bench --bench concurrent_maps --features mimalloc

# Run specific benchmark with native optimizations
# Usage: just bench-native-one 08a_read_scaling
bench-native-one FILTER:
    RUSTFLAGS="-C target-cpu=native" cargo bench --bench concurrent_maps --features mimalloc -- {{FILTER}}

# Run C++ Masstree-compatible read scaling benchmark (apples-to-apples comparison)
# Pre-populates 10M keys, then measures pure read throughput with shuffled access
# Compare with: cd reference && ./mttest -j32 -l 10000000 rw1
apples:
    cargo bench --bench concurrent_maps --features mimalloc -- rw1_reads_only

# === Profiling ===

# Build with debug symbols for profiling
build-profile:
    cargo build --profile release-with-debug

# === Flamegraph (requires cargo-flamegraph) ===

# Run flamegraph on the profile example
# Usage: just flamegraph [workload]
# Workloads: key, permuter, tree, all (default: tree)
flamegraph workload="tree":
    CARGO_PROFILE_PROFILING_DEBUG=2 \
    RUSTFLAGS="-C force-frame-pointers=yes" \
    cargo flamegraph --profile profiling --example profile -- {{workload}}
    @echo "Output: flamegraph.svg"

# Run flamegraph with mimalloc (allocator symbols may be incomplete)
flamegraph-mimalloc workload="tree":
    CARGO_PROFILE_PROFILING_DEBUG=2 \
    RUSTFLAGS="-C force-frame-pointers=yes" \
    CFLAGS="-fno-omit-frame-pointer" \
    cargo flamegraph --profile profiling --example profile --features mimalloc -- {{workload}}
    @echo "Output: flamegraph.svg"

# Run flamegraph on a specific benchmark
# Usage: just flamegraph-bench concurrent_maps "08a_read_scaling"
flamegraph-bench bench filter="":
    CARGO_PROFILE_PROFILING_DEBUG=2 \
    RUSTFLAGS="-C force-frame-pointers=yes" \
    cargo flamegraph --profile profiling --bench {{bench}} -- {{filter}}
    @echo "Output: flamegraph.svg"

# Run perf record manually (for more control)
# Usage: just perf-record [workload]
perf-record workload="tree":
    RUSTFLAGS="-C force-frame-pointers=yes -C debuginfo=2" \
    cargo build --profile profiling --example profile
    perf record -g --call-graph dwarf ./target/profiling/examples/profile {{workload}}
    @echo "Output: perf.data"
    @echo "View:   perf report -g"

# Generate flamegraph from existing perf.data
perf-flamegraph:
    perf script | stackcollapse-perf.pl | flamegraph.pl > perf-flamegraph.svg
    @echo "Output: perf-flamegraph.svg"

# Build the callgrind profiling example (with debug symbols)
build-callgrind:
    cargo build --profile release-with-debug --example profile

# Run callgrind profiler (default: key workload)
# Usage: just callgrind [key|permuter|all]
callgrind workload="key":
    @cargo build --profile release-with-debug --example profile
    valgrind --tool=callgrind ./target/release-with-debug/examples/profile {{workload}}
    @echo "Output: callgrind.out.*"
    @echo "Analyze: just callgrind-annotate"
    @echo "GUI:     just callgrind-view"

# Run callgrind with cache simulation
callgrind-cache workload="key":
    @cargo build --profile release-with-debug --example profile
    valgrind --tool=callgrind --cache-sim=yes --branch-sim=yes ./target/release-with-debug/examples/profile {{workload}}

# Annotate the most recent callgrind output
callgrind-annotate:
    @callgrind_annotate $(ls -t callgrind.out.* | head -1) --auto=yes

# Open most recent callgrind output in kcachegrind
callgrind-view:
    @kcachegrind $(ls -t callgrind.out.* | head -1)

# Clean callgrind output files
callgrind-clean:
    rm -f callgrind.out.*

# === Safety ===

# Run address sanitizer (requires nightly)
# Note: --lib --tests excludes doc tests which don't support sanitizers
asan:
    RUSTFLAGS="-Z sanitizer=address" cargo +nightly test --lib --tests --target x86_64-unknown-linux-gnu

# Run address sanitizer then clean (avoids polluting normal builds)
asan-clean:
    RUSTFLAGS="-Z sanitizer=address" cargo +nightly test --lib --tests --target x86_64-unknown-linux-gnu
    cargo clean

# Run thread sanitizer (requires nightly)
# Note: --lib --tests excludes doc tests which don't support sanitizers
tsan:
    RUSTFLAGS="-Z sanitizer=thread" cargo +nightly test --lib --tests --target x86_64-unknown-linux-gnu

# Run thread sanitizer then clean (avoids polluting normal builds)
tsan-clean:
    RUSTFLAGS="-Z sanitizer=thread" cargo +nightly test --lib --tests --target x86_64-unknown-linux-gnu
    cargo clean

# === Assembly Inspection (requires cargo-show-asm) ===

# List all inspectable symbols in the crate
asm:
    @cargo asm --lib || true

# View assembly for a function (use index from `just asm`)
# Usage: just asm-view 0
asm-view index:
    cargo asm --lib --rust {{index}}

# Side-by-side diff of two functions by index
# Usage: just asm-diff 0 1
asm-diff idx1 idx2:
    @mkdir -p /tmp/asm-diff
    @cargo asm --lib {{idx1}} > /tmp/asm-diff/a.asm 2>&1
    @cargo asm --lib {{idx2}} > /tmp/asm-diff/b.asm 2>&1
    @if command -v delta >/dev/null 2>&1; then \
        delta /tmp/asm-diff/a.asm /tmp/asm-diff/b.asm; \
    else \
        diff -y --width=140 /tmp/asm-diff/a.asm /tmp/asm-diff/b.asm || true; \
    fi

# Analyze function throughput with llvm-mca
# Usage: just asm-mca 0
asm-mca index:
    @if ! command -v llvm-mca >/dev/null 2>&1; then \
        echo "Error: llvm-mca not found. Install: sudo apt install llvm"; \
        exit 1; \
    fi
    cargo asm --lib {{index}} | llvm-mca -mcpu=native
