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
    cargo test

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

# === Safety ===

# Run address sanitizer (requires nightly)
asan:
    RUSTFLAGS="-Z sanitizer=address" cargo +nightly test --target x86_64-unknown-linux-gnu

# Run thread sanitizer (requires nightly)
tsan:
    RUSTFLAGS="-Z sanitizer=thread" cargo +nightly test --target x86_64-unknown-linux-gnu
