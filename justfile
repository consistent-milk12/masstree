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

# Run all tests with nextest, tracing, and mimalloc (prioritized mode)
# This is the recommended way to run tests for development
next:
    cargo nextest run --no-fail-fast --features "tracing,mimalloc"

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
            if [ -f logs/masstree.json ]; then
                # Copy and finalize the JSON array (add closing bracket if missing)
                cp -f logs/masstree.json "logs/next-repeat/run-${i}.json"
                # Check if file ends with ] and add it if not
                if ! tail -c 2 "logs/next-repeat/run-${i}.json" | grep -q ']'; then
                    echo ']' >> "logs/next-repeat/run-${i}.json"
                fi
                echo "Saved logs to logs/next-repeat/run-${i}.json"
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

# === Tracing & Log Analysis ===

# Log file location
log_file := "logs/masstree.json"

# Run test with tracing enabled (JSON logs to logs/masstree.json)
test-trace TEST:
    rm -f {{log_file}}
    RUST_LOG=masstree=debug cargo test --features tracing {{TEST}} -- --nocapture

# Run test with trace-level logging (very verbose)
test-trace-verbose TEST:
    rm -f {{log_file}}
    RUST_LOG=masstree=trace cargo test --features tracing {{TEST}} -- --nocapture

# Run test with file-only logging (no console spam)
test-trace-quiet TEST:
    rm -f {{log_file}}
    RUST_LOG=masstree=debug MASSTREE_LOG_CONSOLE=0 cargo test --features tracing {{TEST}}

# Finalize log file by adding closing bracket if missing
# (The tracing writer uses Box::leak so Drop never runs)
log-finalize:
    @if [ -f {{log_file}} ] && ! tail -c 2 {{log_file}} | grep -q ']'; then \
        echo ']' >> {{log_file}}; \
    fi

# Pretty print the entire log file (logs are already formatted JSON arrays)
log-pretty:
    @just log-finalize
    @cat {{log_file}} | jq '.'

# Show log summary (count by level)
log-summary:
    @just log-finalize
    @cat {{log_file}} | jq 'group_by(.level) | map({level: .[0].level, count: length})'

# Find all events for a specific ikey
log-key KEY:
    @just log-finalize
    @cat {{log_file}} | jq '[.[] | select(.fields.ikey == {{KEY}})]'

# Find all events for a specific ikey (compact, one per line)
log-key-compact KEY:
    @just log-finalize
    @cat {{log_file}} | jq -c '.[] | select(.fields.ikey == {{KEY}})'

# Show only errors
log-errors:
    @just log-finalize
    @cat {{log_file}} | jq '[.[] | select(.level == "ERROR")]'

# Show only warnings and errors
log-warnings:
    @just log-finalize
    @cat {{log_file}} | jq '[.[] | select(.level == "WARN" or .level == "ERROR")]'

# Show split events
log-splits:
    @just log-finalize
    @cat {{log_file}} | jq '[.[] | select(.fields.message | contains("split"))]'

# Show events for a specific leaf pointer
log-leaf PTR:
    @just log-finalize
    @cat {{log_file}} | jq '[.[] | select(.fields.leaf_ptr == "{{PTR}}" or .fields.left_ptr == "{{PTR}}")]'

# Show events by thread
log-thread THREAD:
    @just log-finalize
    @cat {{log_file}} | jq '[.[] | select(.threadId == "ThreadId({{THREAD}})")]'

# Sort by timestamp and show key fields
log-timeline:
    @just log-finalize
    @cat {{log_file}} | jq 'sort_by(.timestamp) | [.[] | {time: .timestamp, level, thread: .threadId, msg: .fields.message, ikey: .fields.ikey}]'

# Show CAS insert successes
log-cas:
    @just log-finalize
    @cat {{log_file}} | jq '[.[] | select(.fields.message | contains("CAS insert"))]'

# Show retry events
log-retries:
    @just log-finalize
    @cat {{log_file}} | jq '[.[] | select(.fields.message | contains("retry"))]'

# Clear log file
log-clear:
    rm -f {{log_file}}

# === Advanced Log Analysis (Python) ===

# Python analyzer script
analyzer := "python3 scripts/masstree_log_analyze.py"

# Show log analyzer help
analyze-help:
    {{analyzer}} --help

# High-level log summary with
analyze-summary:
    @just log-finalize 2>/dev/null || true
    {{analyzer}} {{log_file}} summary

# Count events by field (e.g., just analyze-stats fields.message)
analyze-stats FIELD:
    @just log-finalize 2>/dev/null || true
    {{analyzer}} {{log_file}} stats --by {{FIELD}}

# Trace a specific key's lifecycle (insert → split → loss)
analyze-correlate KEY:
    @just log-finalize 2>/dev/null || true
    {{analyzer}} {{log_file}} correlate --ikey {{KEY}}

# Show complete lifecycle of a specific leaf
analyze-leaf PTR:
    @just log-finalize 2>/dev/null || true
    {{analyzer}} {{log_file}} leaf-timeline {{PTR}}

# Find missing keys from expected set (e.g., just analyze-missing "0-100,20000")
analyze-missing EXPECTED:
    @just log-finalize 2>/dev/null || true
    {{analyzer}} {{log_file}} missing-keys --expected "{{EXPECTED}}"

# Show thread interleaving on contested leaves
analyze-interleave:
    @just log-finalize 2>/dev/null || true
    {{analyzer}} {{log_file}} interleave --min-threads 2

# Analyze CAS failure patterns
analyze-cas-failures:
    @just log-finalize 2>/dev/null || true
    {{analyzer}} {{log_file}} cas-failures --limit 10

# Show splits with slot indices (distinguishes perm corruption vs overwrite)
analyze-splits:
    @just log-finalize 2>/dev/null || true
    {{analyzer}} {{log_file}} splits --show-slots

# Decode and validate permutation values
analyze-perms:
    @just log-finalize 2>/dev/null || true
    {{analyzer}} {{log_file}} perms

# Run all anomaly detectors (slot-reuse, slot-steal, missing-after-success)
analyze-anomalies:
    @just log-finalize 2>/dev/null || true
    {{analyzer}} {{log_file}} anomalies --all

# Detect slot reuse within epoch
analyze-slot-reuse:
    @just log-finalize 2>/dev/null || true
    {{analyzer}} {{log_file}} anomalies --slot-reuse

# Detect slot stealing
analyze-slot-steal:
    @just log-finalize 2>/dev/null || true
    {{analyzer}} {{log_file}} anomalies --slot-steal

# Detect keys missing after successful insert
analyze-missing-after-success:
    @just log-finalize 2>/dev/null || true
    {{analyzer}} {{log_file}} anomalies --missing

# Export events to CSV for external analysis
analyze-export OUT:
    @just log-finalize 2>/dev/null || true
    {{analyzer}} {{log_file}} export-csv {{OUT}}

# debugging workflow: run test, finalize logs, run all anomaly detectors
debug-p06 TEST:
    @echo "=== Running test with tracing ==="
    rm -f {{log_file}}
    RUST_LOG=masstree=debug cargo test --features tracing {{TEST}} -- --nocapture || true
    @echo ""
    @echo "=== Finalizing logs ==="
    just log-finalize
    @echo ""
    @echo "=== Running anomaly detectors ==="
    {{analyzer}} {{log_file}} anomalies --all
    @echo ""
    @echo "=== Summary ==="
    {{analyzer}} {{log_file}} summary

# Run a test multiple times to catch race conditions (default: 10 times)
test-repeat TEST TIMES="10":
    #!/usr/bin/env bash
    set -euo pipefail
    passed=0
    failed=0
    for i in $(seq 1 {{TIMES}}); do
        echo -n "Run $i/{{TIMES}}: "
        if cargo test {{TEST}} 2>&1 | grep -q "test result: ok"; then
            echo "PASS"
            ((passed++))
        else
            echo "FAIL"
            ((failed++))
        fi
    done
    echo ""
    echo "Results: $passed passed, $failed failed out of {{TIMES}} runs"
    if [ $failed -gt 0 ]; then
        exit 1
    fi

# Run concurrent tests multiple times with tracing on failure
test-concurrent-stress TIMES="10":
    #!/usr/bin/env bash
    set -uo pipefail
    passed=0
    failed=0
    for i in $(seq 1 {{TIMES}}); do
        echo -n "Run $i/{{TIMES}}: "
        rm -f {{log_file}}
        output=$(RUST_LOG=masstree=debug cargo test --features tracing concurrent_insert 2>&1)
        if echo "$output" | grep -q "test result: ok. 13 passed"; then
            echo "PASS (13/13)"
            ((passed++)) || true
        else
            # Extract which tests failed
            failing=$(echo "$output" | grep "FAILED" | head -5)
            echo "FAIL"
            echo "$failing"
            echo "--- Analyzing logs ---"
            just log-finalize 2>/dev/null || true
            {{analyzer}} {{log_file}} anomalies --all 2>/dev/null || true
            ((failed++)) || true
        fi
    done
    echo ""
    echo "Results: $passed passed, $failed failed out of {{TIMES}} runs"

# === Rigorous Stress Tests ===

# Run all stress tests with nextest + mimalloc (recommended)
stress:
    cargo nextest run --features mimalloc --test stress_tests --release

# Run all stress tests in debug mode (slower but catches more issues)
stress-debug:
    cargo nextest run --features mimalloc --test stress_tests

# Run multilayer stress tests only
stress-multilayer:
    cargo nextest run --features mimalloc --test stress_tests multilayer --release

# Run high thread count tests only
stress-threads:
    cargo nextest run --features mimalloc --test stress_tests high_thread --release

# Run large volume tests only
stress-volume:
    cargo nextest run --features mimalloc --test stress_tests large_volume --release

# Run key pattern tests only
stress-patterns:
    cargo nextest run --features mimalloc --test stress_tests pattern --release

# Run mixed read/write tests only
stress-mixed:
    cargo nextest run --features mimalloc --test stress_tests mixed --release

# Run repeated run tests (for catching intermittent bugs)
stress-repeated:
    cargo nextest run --features mimalloc --test stress_tests repeated --release

# Run extreme stress tests (longer duration, use for extended testing)
stress-extreme:
    cargo nextest run --features mimalloc --test stress_tests --release -- --ignored

# Run stress tests until failure (rigorous version)
stress-until-fail-rigorous:
    #!/usr/bin/env bash
    set -uo pipefail
    i=0
    while true; do
        i=$((i + 1))
        echo -n "Run $i: "
        output=$(cargo nextest run --features mimalloc --test stress_tests --release 2>&1)
        if echo "$output" | grep -qE "FAIL|CRITICAL|verification failures"; then
            echo "FOUND BUG!"
            echo "$output" | grep -E "FAIL|CRITICAL|verification|Missing|panic" | head -15
            echo ""
            echo "Failed test output saved. To debug:"
            echo "  RUST_LOG=masstree=debug cargo test --features tracing --test stress_tests <TEST_NAME> -- --nocapture"
            break
        else
            echo "ok"
        fi
    done

# Run stress tests N times to measure stability
stress-repeat N="10":
    #!/usr/bin/env bash
    set -uo pipefail
    passed=0
    failed=0
    for i in $(seq 1 {{N}}); do
        echo "=== Run $i/{{N}} ==="
        output=$(cargo nextest run --features mimalloc --test stress_tests --release 2>&1)
        if echo "$output" | grep -qE "FAIL|verification failures"; then
            echo "FAIL"
            echo "$output" | grep -E "FAIL|verification|Missing" | head -10
            failed=$((failed + 1))
        else
            echo "PASS"
            passed=$((passed + 1))
        fi
    done
    echo ""
    echo "Stress test results: $passed passed, $failed failed out of {{N}} runs"
    if [ $failed -gt 0 ]; then
        echo "Failure rate: $(echo "scale=1; $failed * 100 / {{N}}" | bc)%"
        exit 1
    fi

# Run ALL tests (including stress) with nextest + mimalloc
test-full:
    cargo nextest run --features mimalloc --release

# Run ALL tests repeatedly to measure overall stability
test-stability N="20":
    #!/usr/bin/env bash
    set -uo pipefail
    passed=0
    failed=0
    for i in $(seq 1 {{N}}); do
        echo "=== Stability run $i/{{N}} ==="
        output=$(cargo nextest run --features mimalloc --release 2>&1)
        if echo "$output" | grep -qE "FAIL"; then
            failed_tests=$(echo "$output" | grep "FAIL" | wc -l)
            echo "FAIL ($failed_tests tests failed)"
            failed=$((failed + 1))
        else
            passed_count=$(echo "$output" | grep -oP '\d+ passed' | head -1 || echo "? passed")
            echo "PASS ($passed_count)"
            passed=$((passed + 1))
        fi
    done
    echo ""
    echo "========================================"
    echo "Stability results: $passed passed, $failed failed out of {{N}} runs"
    if [ $failed -gt 0 ]; then
        echo "Failure rate: $(echo "scale=1; $failed * 100 / {{N}}" | bc)%"
        exit 1
    else
        echo "SUCCESS: 100% stability across {{N}} runs"
    fi
