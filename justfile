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
        if cargo nextest run --no-fail-fast --features mimalloc 2>&1 | tail -5; then
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
    RUSTFLAGS="--cfg loom" cargo test --lib nodeversion::loom_tests
    RUSTFLAGS="--cfg loom" cargo test --lib tree::loom_tests

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

# Run Miri checking for memory leaks
miri-leaks:
    MIRIFLAGS="-Zmiri-strict-provenance -Zmiri-symbolic-alignment-check" cargo +nightly miri test --lib

# Run Miri with Tree Borrows (experimental, more permissive than Stacked Borrows)
miri-tree-borrows:
    MIRIFLAGS="-Zmiri-tree-borrows" cargo +nightly miri test --lib

# === Benchmarks ===

# Run benchmarks
bench:
    cargo bench

# Run benchmarks with native CPU optimizations (AVX2, etc.)
bench-native:
    RUSTFLAGS="-C target-cpu=native" cargo bench --bench concurrent_maps --features mimalloc

# Run specific benchmark with native optimizations
# Usage: just bench-native-one 08a_read_scaling
bench-native-one FILTER:
    RUSTFLAGS="-C target-cpu=native" cargo bench --bench concurrent_maps --features mimalloc -- {{FILTER}}

# === Profiling ===

# Build with debug symbols for profiling
build-profile:
    cargo build --profile release-with-debug

# Run flamegraph on the profile example
# Usage: just flamegraph [workload]
flamegraph workload="tree":
    CARGO_PROFILE_PROFILING_DEBUG=2 \
    RUSTFLAGS="-C force-frame-pointers=yes" \
    cargo flamegraph --profile profiling --example profile -- {{workload}}
    @echo "Output: flamegraph.svg"

# Run flamegraph on a specific benchmark
flamegraph-bench bench filter="":
    CARGO_PROFILE_PROFILING_DEBUG=2 \
    RUSTFLAGS="-C force-frame-pointers=yes" \
    cargo flamegraph --profile profiling --bench {{bench}} -- {{filter}}
    @echo "Output: flamegraph.svg"

# === Safety ===

# Run address sanitizer (requires nightly)
asan:
    RUSTFLAGS="-Z sanitizer=address" cargo +nightly test --lib --tests --target x86_64-unknown-linux-gnu

# Run thread sanitizer (requires nightly)
tsan:
    RUSTFLAGS="-Z sanitizer=thread" cargo +nightly test --lib --tests --target x86_64-unknown-linux-gnu

# === Assembly Inspection (requires cargo-show-asm) ===

# List all inspectable symbols in the crate
asm:
    @cargo asm --lib || true

# View assembly for a function (use index from `just asm`)
asm-view index:
    cargo asm --lib --rust {{index}}
