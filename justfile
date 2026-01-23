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
buildr:
    cargo build --release

# Run benchmarks
bench:
    cargo bench --bench mttest --features mimalloc

# Run benchmarks with specific args (e.g., just bench-args "all -j12 -d10")
bench-args ARGS:
    cargo bench --bench mttest --features mimalloc -- {{ARGS}}

# === Benchmark comparison commands ===

# Default benchmark parameters
BENCH_THREADS := "12"
BENCH_DURATION := "10"

# Run all C++ benchmarks and save to cpp_benches/
cpp-bench:
    #!/usr/bin/env bash
    set -e
    mkdir -p cpp_benches
    tests="rw1 rw2g98 rw3 rw4 same uscale wscale"
    echo "Running C++ benchmarks ({{BENCH_THREADS}} threads, {{BENCH_DURATION}}s)..."
    for test in $tests; do
        echo "  Running $test..."
        ./reference/mttest -j{{BENCH_THREADS}} -d{{BENCH_DURATION}} $test 2>&1 > cpp_benches/$test.txt
    done
    echo "Done. Results saved to cpp_benches/"

# Parse C++ benchmark results
cpp-parse:
    python3 cpp_benches/parse_results.py

# Run all Rust benchmarks and save to rust_benches/
rust-bench:
    #!/usr/bin/env bash
    set -e
    mkdir -p rust_benches
    tests="rw1 rw2g98 rw3 rw4 same uscale wscale"
    echo "Running Rust benchmarks ({{BENCH_THREADS}} threads, {{BENCH_DURATION}}s)..."
    for test in $tests; do
        echo "  Running $test..."
        cargo bench --bench mttest --features mimalloc -- $test -j{{BENCH_THREADS}} -d{{BENCH_DURATION}} 2>&1 > rust_benches/$test.txt
    done
    echo "Done. Results saved to rust_benches/"

# Parse Rust benchmark results
rust-parse:
    python3 rust_benches/parse_results.py

# Compare C++ and Rust benchmark results
bench-compare:
    python3 scripts/compare_benches.py

# Run all benchmarks and compare (full benchmark suite)
bench-all: cpp-bench rust-bench bench-compare

# Run all tests with nextest (saves failures to file if any)
test:
    #!/usr/bin/env bash
    set -o pipefail
    cargo nextest run --no-fail-fast --status-level=all 2>&1 | tee .test-output.tmp
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

# Run all tests with nextest and mimalloc
next:
    cargo nextest run --no-fail-fast --features mimalloc

# Run a specific test with nextest and mimalloc
next-one TEST:
    cargo nextest run --no-fail-fast --features mimalloc {{TEST}}

# Run nextest N times to catch intermittent failures
# Usage: just next-repeat 20
next-repeat N="10":
    #!/usr/bin/env bash
    set -uo pipefail
    passed=0
    failed=0
    for i in $(seq 1 {{N}}); do
        echo "=== Run $i/{{N}} ==="
        if cargo nextest run --no-fail-fast --features mimalloc; then
            echo "PASS"
            passed=$((passed + 1))
        else
            echo "FAIL"
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
test-all: test test-loom test-shuttle miri-strict
    @echo "All tests passed!"

# Run loom tests for deterministic concurrency verification
test-loom:
    RUSTFLAGS="--cfg loom" cargo test --lib loom_tests

# Run shuttle linearizability tests
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

# Format and lint (quick pre-commit)
pre: fmt lint

# Quick pre-commit check (faster than check-all)
check: fmt lint test-unit

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

# Run unit tests under Miri to detect undefined behavior
miri:
    cargo +nightly miri test --lib

# Run a specific test under Miri
miri-test TEST:
    cargo +nightly miri test {{TEST}}

# Run Miri with stricter checks (Stacked Borrows)
miri-strict:
    MIRIFLAGS="-Zmiri-strict-provenance" cargo +nightly miri test --lib

# Run Miri with strict provenance on integration tests
miri-strict-int:
    MIRIFLAGS="-Zmiri-strict-provenance" cargo +nightly miri test --test '*'

# Run Miri checking for memory leaks
miri-leaks:
    MIRIFLAGS="-Zmiri-strict-provenance -Zmiri-symbolic-alignment-check" cargo +nightly miri test --lib

# Run Miri with Tree Borrows (experimental, more permissive than Stacked Borrows)
miri-tree-borrows:
    MIRIFLAGS="-Zmiri-tree-borrows" cargo +nightly miri test --lib

# Bump patch version (0.X.Y -> 0.X.(Y+1)) in Cargo.toml and README.md
bump:
    python3 scripts/bump_version.py && cargo build

# Bump version (dry run - show what would change)
bump-dry:
    python3 scripts/bump_version.py --dry-run
