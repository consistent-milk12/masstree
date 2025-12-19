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

# Run all tests (unit + doc + integration)
test:
    cargo test

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

# Run ALL possible tests (unit, doc, integration, loom, miri)
# This is the most comprehensive test command
test-all: test test-loom miri-strict
    @echo "All tests passed!"

# Run loom tests for deterministic concurrency verification
# Loom explores all possible thread interleavings
test-loom:
    RUSTFLAGS="--cfg loom" cargo test --lib nodeversion::loom_tests

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

# === Diagrams ===

# Generate all SVG diagrams from mermaid sources
diagrams:
    #!/usr/bin/env bash
    set -euo pipefail
    cd docs/diagrams
    for f in *.mmd; do
        svg="${f%.mmd}.svg"
        echo "Generating $svg..."
        mmdc -i "$f" -o "$svg" -b transparent -p puppeteer-config.json
    done
    echo "Done! Generated $(ls -1 *.svg | wc -l) diagrams."

# Generate a single diagram by name (without extension)
diagram name:
    mmdc -i "docs/diagrams/{{name}}.mmd" -o "docs/diagrams/{{name}}.svg" -b transparent -p docs/diagrams/puppeteer-config.json

# Watch and regenerate diagrams on change (requires entr)
diagrams-watch:
    ls docs/diagrams/*.mmd | entr -c just diagrams

# Clean generated diagram SVGs
diagrams-clean:
    rm -f docs/diagrams/*.svg

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

# === Profiling ===

# Build with debug symbols for profiling
build-profile:
    cargo build --profile release-with-debug

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
