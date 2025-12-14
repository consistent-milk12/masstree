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

# Run all tests
test:
    cargo test --message-format=short

# Run tests with output
test-verbose:
    cargo test -- --nocapture

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

# Run tests under Miri to detect undefined behavior
miri:
    cargo +nightly miri test

# Run a specific test under Miri
miri-test TEST:
    cargo +nightly miri test {{TEST}}

# Run Miri with stricter checks (Stacked Borrows)
miri-strict:
    MIRIFLAGS="-Zmiri-strict-provenance" cargo +nightly miri test

# Run Miri checking for memory leaks
miri-leaks:
    MIRIFLAGS="-Zmiri-strict-provenance -Zmiri-symbolic-alignment-check" cargo +nightly miri test

# Run Miri with Tree Borrows (experimental, more permissive than Stacked Borrows)
miri-tree-borrows:
    MIRIFLAGS="-Zmiri-tree-borrows" cargo +nightly miri test

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
asan:
    RUSTFLAGS="-Z sanitizer=address" cargo +nightly test --target x86_64-unknown-linux-gnu

# Run thread sanitizer (requires nightly)
tsan:
    RUSTFLAGS="-Z sanitizer=thread" cargo +nightly test --target x86_64-unknown-linux-gnu

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
