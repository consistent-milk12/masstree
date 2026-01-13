#!/usr/bin/env python3
"""
Run C++ and Rust mttest benchmarks with comparison support.

Usage:
    ./scripts/run_cpp_bench.py                      # C++ only, 12 threads, 10s
    ./scripts/run_cpp_bench.py --compare            # Run both C++ and Rust, show comparison
    ./scripts/run_cpp_bench.py --compare --markdown # Output as markdown table
    ./scripts/run_cpp_bench.py -j 6 -d 5            # Custom threads/duration
    ./scripts/run_cpp_bench.py rw1 rw3              # Specific tests only
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

# ANSI color codes
GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
BOLD = "\033[1m"
RESET = "\033[0m"

DEFAULT_TESTS = ["rw1", "rw2g98", "rw3", "rw4", "same", "uscale", "wscale"]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run C++ and Rust mttest benchmarks",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                      Run C++ benchmarks (12 threads, 10s)
  %(prog)s --compare            Compare C++ vs Rust side-by-side
  %(prog)s --compare --markdown Output comparison as markdown table
  %(prog)s -j6 -d5 rw1 rw3      Run specific tests with custom config
  %(prog)s --rust-only          Run only Rust benchmarks
        """,
    )
    parser.add_argument("-j", "--threads", type=int, default=12,
                        help="Number of threads (default: 12)")
    parser.add_argument("-d", "--duration", type=int, default=10,
                        help="Duration in seconds (default: 10)")
    parser.add_argument("-p", "--pin", action="store_true",
                        help="Pin threads to cores")
    parser.add_argument("--compare", action="store_true",
                        help="Run both C++ and Rust, show comparison")
    parser.add_argument("--rust-only", action="store_true",
                        help="Run only Rust benchmarks")
    parser.add_argument("--markdown", action="store_true",
                        help="Output as markdown table (useful with --compare)")
    parser.add_argument("--no-color", action="store_true",
                        help="Disable colored output")
    parser.add_argument("-q", "--quiet", action="store_true",
                        help="Suppress progress messages")
    parser.add_argument("tests", nargs="*", default=DEFAULT_TESTS,
                        help="Tests to run (default: all standard tests)")
    return parser.parse_args()


def run_cpp_benchmark(mttest_path: Path, threads: int, duration: int, test: str,
                      pin: bool = False, quiet: bool = False) -> float | None:
    """Run a single C++ benchmark and return total ops/sec."""
    cmd = [str(mttest_path), f"-j{threads}", f"-d{duration}"]
    if pin:
        cmd.append("-p")
    cmd.append(test)

    if not quiet:
        print(f"  Running C++ {test}...", end="", flush=True, file=sys.stderr)

    try:
        # Timeout: duration * 2 for two-phase tests (put + get) + buffer
        timeout = duration * 3 + 30
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        output = result.stderr
    except subprocess.TimeoutExpired:
        if not quiet:
            print(" timeout", file=sys.stderr)
        return None
    except Exception as e:
        if not quiet:
            print(f" error: {e}", file=sys.stderr)
        return None

    # Parse JSON lines from stderr
    total_ops = 0.0
    lines_parsed = 0
    for line in output.splitlines():
        brace_idx = line.find("{")
        if brace_idx == -1:
            continue
        try:
            data = json.loads(line[brace_idx:])
            if data.get("test") != test:
                continue
            # Prefer ops_per_sec, fall back to puts_per_sec for write-only tests
            if "ops_per_sec" in data:
                total_ops += float(data["ops_per_sec"])
                lines_parsed += 1
            elif "puts_per_sec" in data:
                total_ops += float(data["puts_per_sec"])
                lines_parsed += 1
        except json.JSONDecodeError:
            continue

    if not quiet:
        if lines_parsed > 0:
            print(f" {total_ops/1e6:.2f} Mops/s", file=sys.stderr)
        else:
            print(" parse error", file=sys.stderr)

    return total_ops if lines_parsed > 0 else None


def run_rust_benchmark(project_dir: Path, threads: int, duration: int, test: str,
                       pin: bool = False, quiet: bool = False) -> float | None:
    """Run a single Rust benchmark and return total ops/sec."""
    cmd = [
        "cargo", "bench", "--bench", "mttest", "--features", "mimalloc",
        "--", test, f"-j{threads}", f"-d{duration}", "-t"
    ]
    if pin:
        cmd.append("-p")

    if not quiet:
        print(f"  Running Rust {test}...", end="", flush=True, file=sys.stderr)

    try:
        timeout = duration * 3 + 60  # Extra time for compilation
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout, cwd=project_dir
        )
        output = result.stdout
    except subprocess.TimeoutExpired:
        if not quiet:
            print(" timeout", file=sys.stderr)
        return None
    except Exception as e:
        if not quiet:
            print(f" error: {e}", file=sys.stderr)
        return None

    # Parse output like "rw1:       9.13 Mops/s"
    for line in output.splitlines():
        line = line.strip()
        if line.startswith(f"{test}:"):
            parts = line.split()
            for i, part in enumerate(parts):
                if part == "Mops/s" and i > 0:
                    try:
                        mops = float(parts[i - 1])
                        if not quiet:
                            print(f" {mops:.2f} Mops/s", file=sys.stderr)
                        return mops * 1_000_000
                    except ValueError:
                        pass

    if not quiet:
        print(" parse error", file=sys.stderr)
    return None


def format_comparison(test: str, cpp_mops: float | None, rust_mops: float | None,
                      use_color: bool = True) -> tuple[str, str, str, str]:
    """Format a comparison row, returning (test, cpp, rust, ratio)."""
    cpp_str = f"{cpp_mops:.2f}" if cpp_mops else "N/A"
    rust_str = f"{rust_mops:.2f}" if rust_mops else "N/A"

    if cpp_mops and rust_mops:
        ratio = rust_mops / cpp_mops * 100
        ratio_str = f"{ratio:.0f}%"

        if use_color:
            if ratio >= 100:
                ratio_str = f"{GREEN}{BOLD}{ratio_str}{RESET}"
            elif ratio >= 80:
                ratio_str = f"{YELLOW}{ratio_str}{RESET}"
            else:
                ratio_str = f"{RED}{ratio_str}{RESET}"
    else:
        ratio_str = "N/A"

    return test, cpp_str, rust_str, ratio_str


def print_comparison_table(results: list[tuple[str, float | None, float | None]],
                           markdown: bool = False, use_color: bool = True):
    """Print a comparison table."""
    if markdown:
        print("\n| Benchmark | C++ (Mops/s) | Rust (Mops/s) | Rust/C++ |")
        print("|-----------|--------------|---------------|----------|")
        for test, cpp_ops, rust_ops in results:
            cpp_mops = cpp_ops / 1e6 if cpp_ops else None
            rust_mops = rust_ops / 1e6 if rust_ops else None
            test, cpp, rust, ratio = format_comparison(test, cpp_mops, rust_mops, use_color=False)
            print(f"| {test:<9} | {cpp:>12} | {rust:>13} | {ratio:>8} |")
    else:
        print(f"\n{'Benchmark':<10} {'C++ (Mops/s)':>12} {'Rust (Mops/s)':>14} {'Rust/C++':>10}")
        print("-" * 48)
        for test, cpp_ops, rust_ops in results:
            cpp_mops = cpp_ops / 1e6 if cpp_ops else None
            rust_mops = rust_ops / 1e6 if rust_ops else None
            test, cpp, rust, ratio = format_comparison(test, cpp_mops, rust_mops, use_color)
            print(f"{test:<10} {cpp:>12} {rust:>14} {ratio:>10}")


def main():
    args = parse_args()
    use_color = not args.no_color and sys.stdout.isatty()

    script_dir = Path(__file__).parent
    project_dir = script_dir.parent
    mttest_path = project_dir / "reference" / "mttest"

    run_cpp = not args.rust_only
    run_rust = args.compare or args.rust_only

    # Validate C++ binary exists
    if run_cpp and not mttest_path.exists():
        print(f"Error: C++ mttest binary not found at {mttest_path}", file=sys.stderr)
        print(f"Build it with: cd {project_dir / 'reference'} && make", file=sys.stderr)
        sys.exit(1)

    # Header
    mode = "C++ vs Rust" if args.compare else ("Rust" if args.rust_only else "C++")
    pin_str = ", pinned" if args.pin else ""
    print(f"{mode} Masstree mttest - {args.threads} threads, {args.duration}s{pin_str}")

    if args.compare:
        # Comparison mode: run both and show table
        results = []
        for test in args.tests:
            cpp_ops = run_cpp_benchmark(
                mttest_path, args.threads, args.duration, test, args.pin, args.quiet
            )
            rust_ops = run_rust_benchmark(
                project_dir, args.threads, args.duration, test, args.pin, args.quiet
            )
            results.append((test, cpp_ops, rust_ops))

        print_comparison_table(results, args.markdown, use_color)

    elif args.rust_only:
        # Rust only
        if not args.quiet:
            print(file=sys.stderr)
        for test in args.tests:
            rust_ops = run_rust_benchmark(
                project_dir, args.threads, args.duration, test, args.pin, quiet=True
            )
            if rust_ops:
                print(f"{test + ':':<8} {rust_ops/1e6:>6.2f} Mops/s")
            else:
                print(f"{test + ':':<8} error")

    else:
        # C++ only (original behavior)
        if not args.quiet:
            print(file=sys.stderr)
        for test in args.tests:
            cpp_ops = run_cpp_benchmark(
                mttest_path, args.threads, args.duration, test, args.pin, quiet=True
            )
            if cpp_ops:
                print(f"{test + ':':<8} {cpp_ops/1e6:>6.2f} Mops/s")
            else:
                print(f"{test + ':':<8} error")


if __name__ == "__main__":
    main()
