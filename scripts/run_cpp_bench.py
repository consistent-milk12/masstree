#!/usr/bin/env python3
"""
Run C++ and Rust mttest benchmarks with comparison support.

Usage:
    ./scripts/run_cpp_bench.py                      # C++ only, 12 threads, 10s
    ./scripts/run_cpp_bench.py --compare            # Run both C++ and Rust, show comparison
    ./scripts/run_cpp_bench.py --compare --markdown # Output as markdown table
    ./scripts/run_cpp_bench.py -j 6 -d 5            # Custom threads/duration
    ./scripts/run_cpp_bench.py rw1 rw3              # Specific tests only

Requires Python 3.9+
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional, Tuple, List

# ANSI color codes
GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
BOLD = "\033[1m"
RESET = "\033[0m"

DEFAULT_TESTS = ["rw1", "rw2g98", "rw3", "rw4", "same", "uscale", "wscale"]

# Regex for parsing Rust benchmark output (more robust than split)
# Matches: "test_name:   123.45 Mops/s" with flexible whitespace
RUST_OUTPUT_RE = re.compile(r"^(\w+):\s+([\d.]+)\s*Mops/s", re.IGNORECASE)


def parse_args() -> argparse.Namespace:
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
  %(prog)s --no-mimalloc        Run Rust without mimalloc allocator
        """,
    )
    parser.add_argument(
        "-j", "--threads", type=int, default=12, help="Number of threads (default: 12)"
    )
    parser.add_argument(
        "-d",
        "--duration",
        type=int,
        default=10,
        help="Duration in seconds (default: 10)",
    )
    parser.add_argument("-p", "--pin", action="store_true", help="Pin threads to cores")
    parser.add_argument(
        "--compare", action="store_true", help="Run both C++ and Rust, show comparison"
    )
    parser.add_argument(
        "--rust-only", action="store_true", help="Run only Rust benchmarks"
    )
    parser.add_argument(
        "--markdown",
        action="store_true",
        help="Output as markdown table (useful with --compare)",
    )
    parser.add_argument(
        "--no-color", action="store_true", help="Disable colored output"
    )
    parser.add_argument(
        "-q", "--quiet", action="store_true", help="Suppress progress messages"
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Show raw benchmark output on parse errors",
    )
    parser.add_argument(
        "--no-mimalloc",
        action="store_true",
        help="Run Rust benchmarks without mimalloc feature",
    )
    parser.add_argument(
        "tests",
        nargs="*",
        default=DEFAULT_TESTS,
        help="Tests to run (default: all standard tests)",
    )
    return parser.parse_args()


def run_cpp_benchmark(
    mttest_path: Path,
    threads: int,
    duration: int,
    test: str,
    pin: bool = False,
    quiet: bool = False,
    verbose: bool = False,
) -> Optional[float]:
    """Run a single C++ benchmark and return total ops/sec.

    For multi-phase tests (e.g., rw1 with put+get phases), returns the sum
    of ops/sec across all phases, matching how C++ mttest reports totals.
    """
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
    except subprocess.TimeoutExpired:
        if not quiet:
            print(" timeout", file=sys.stderr)
        return None
    except Exception as e:
        if not quiet:
            print(f" error: {e}", file=sys.stderr)
        return None

    # Check for non-zero exit code
    if result.returncode != 0:
        if not quiet:
            print(f" exit code {result.returncode}", file=sys.stderr)
        if verbose:
            print(f"    stdout: {result.stdout[:500]}", file=sys.stderr)
            print(f"    stderr: {result.stderr[:500]}", file=sys.stderr)
        return None

    output = result.stderr

    # Parse JSON lines from stderr
    ops_per_sec_values: List[float] = []
    puts_per_sec_values: List[float] = []
    gets_per_sec_values: List[float] = []
    json_lines_found = 0

    for line in output.splitlines():
        brace_idx = line.find("{")
        if brace_idx == -1:
            continue
        json_lines_found += 1
        try:
            data = json.loads(line[brace_idx:])
            if data.get("test") != test:
                continue
            # Collect each metric type separately
            if "ops_per_sec" in data:
                ops_per_sec_values.append(float(data["ops_per_sec"]))
            if "puts_per_sec" in data:
                puts_per_sec_values.append(float(data["puts_per_sec"]))
            if "gets_per_sec" in data:
                gets_per_sec_values.append(float(data["gets_per_sec"]))
        except (json.JSONDecodeError, ValueError, KeyError):
            continue

    # Determine total ops: prefer ops_per_sec (combined metric), fall back to
    # puts_per_sec for write-only tests, or gets_per_sec for read-only tests
    if ops_per_sec_values:
        total_ops = sum(ops_per_sec_values)
        lines_parsed = len(ops_per_sec_values)
    elif puts_per_sec_values:
        total_ops = sum(puts_per_sec_values)
        lines_parsed = len(puts_per_sec_values)
    elif gets_per_sec_values:
        total_ops = sum(gets_per_sec_values)
        lines_parsed = len(gets_per_sec_values)
    else:
        total_ops = 0.0
        lines_parsed = 0

    if not quiet:
        if lines_parsed > 0:
            print(f" {total_ops / 1e6:.2f} Mops/s", file=sys.stderr)
        else:
            if json_lines_found == 0:
                print(" no JSON output found", file=sys.stderr)
            else:
                print(
                    f" parse error (found {json_lines_found} JSON lines, no ops/puts/gets metrics for '{test}')",
                    file=sys.stderr,
                )
            if verbose:
                print(f"    stderr: {output[:1000]}", file=sys.stderr)

    return total_ops if lines_parsed > 0 else None


def run_rust_benchmark(
    project_dir: Path,
    threads: int,
    duration: int,
    test: str,
    pin: bool = False,
    quiet: bool = False,
    verbose: bool = False,
    use_mimalloc: bool = True,
) -> Optional[float]:
    """Run a single Rust benchmark and return total ops/sec."""
    cmd = ["cargo", "bench", "--bench", "mttest"]
    if use_mimalloc:
        cmd.extend(["--features", "mimalloc"])
    cmd.extend(["--", test, f"-j{threads}", f"-d{duration}", "-t"])
    if pin:
        cmd.append("-p")

    if not quiet:
        print(f"  Running Rust {test}...", end="", flush=True, file=sys.stderr)

    try:
        timeout = duration * 3 + 60  # Extra time for compilation
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout, cwd=project_dir
        )
    except subprocess.TimeoutExpired:
        if not quiet:
            print(" timeout", file=sys.stderr)
        return None
    except Exception as e:
        if not quiet:
            print(f" error: {e}", file=sys.stderr)
        return None

    # Check for non-zero exit code (compilation failure, test crash, etc.)
    if result.returncode != 0:
        if not quiet:
            print(f" exit code {result.returncode}", file=sys.stderr)
        if verbose:
            # Show stderr which contains compilation errors
            print(f"    stderr: {result.stderr[:1000]}", file=sys.stderr)
        return None

    output = result.stdout

    # Parse output using regex for robustness
    # Format: "test_name:   123.45 Mops/s"
    for line in output.splitlines():
        line = line.strip()
        match = RUST_OUTPUT_RE.match(line)
        if match and match.group(1).lower() == test.lower():
            try:
                mops = float(match.group(2))
                if not quiet:
                    print(f" {mops:.2f} Mops/s", file=sys.stderr)
                # Store as integer ops/sec to avoid float precision issues
                return int(mops * 1_000_000)
            except ValueError:
                pass

    # Fallback: try the original split-based parsing
    for line in output.splitlines():
        line = line.strip()
        if line.lower().startswith(f"{test.lower()}:"):
            parts = line.split()
            for i, part in enumerate(parts):
                if part.lower() == "mops/s" and i > 0:
                    try:
                        mops = float(parts[i - 1])
                        if not quiet:
                            print(f" {mops:.2f} Mops/s", file=sys.stderr)
                        return int(mops * 1_000_000)
                    except ValueError:
                        pass

    if not quiet:
        print(" parse error", file=sys.stderr)
        if verbose:
            print(f"    stdout: {output[:1000]}", file=sys.stderr)

    return None


def format_comparison(
    test: str,
    cpp_mops: Optional[float],
    rust_mops: Optional[float],
    use_color: bool = True,
) -> Tuple[str, str, str, str]:
    """Format a comparison row, returning (test, cpp, rust, ratio)."""
    cpp_str = f"{cpp_mops:.2f}" if cpp_mops is not None else "N/A"
    rust_str = f"{rust_mops:.2f}" if rust_mops is not None else "N/A"

    if cpp_mops is not None and rust_mops is not None and cpp_mops > 0:
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


def print_comparison_table(
    results: List[Tuple[str, Optional[float], Optional[float]]],
    markdown: bool = False,
    use_color: bool = True,
) -> None:
    """Print a comparison table."""
    if markdown:
        print("\n| Benchmark | C++ (Mops/s) | Rust (Mops/s) | Rust/C++ |")
        print("|-----------|--------------|---------------|----------|")
        for test, cpp_ops, rust_ops in results:
            cpp_mops = cpp_ops / 1e6 if cpp_ops is not None else None
            rust_mops = rust_ops / 1e6 if rust_ops is not None else None
            test, cpp, rust, ratio = format_comparison(
                test, cpp_mops, rust_mops, use_color=False
            )
            print(f"| {test:<9} | {cpp:>12} | {rust:>13} | {ratio:>8} |")
    else:
        print(
            f"\n{'Benchmark':<10} {'C++ (Mops/s)':>12} {'Rust (Mops/s)':>14} {'Rust/C++':>10}"
        )
        print("-" * 48)
        for test, cpp_ops, rust_ops in results:
            cpp_mops = cpp_ops / 1e6 if cpp_ops is not None else None
            rust_mops = rust_ops / 1e6 if rust_ops is not None else None
            test, cpp, rust, ratio = format_comparison(
                test, cpp_mops, rust_mops, use_color
            )
            print(f"{test:<10} {cpp:>12} {rust:>14} {ratio:>10}")


def main() -> None:
    args = parse_args()
    use_color = not args.no_color and sys.stdout.isatty()
    use_mimalloc = not args.no_mimalloc

    script_dir = Path(__file__).parent
    project_dir = script_dir.parent
    mttest_path = project_dir / "reference" / "mttest"

    run_cpp = not args.rust_only

    # Validate C++ binary exists
    if run_cpp and not mttest_path.exists():
        print(f"Error: C++ mttest binary not found at {mttest_path}", file=sys.stderr)
        print(f"Build it with: cd {project_dir / 'reference'} && make", file=sys.stderr)
        sys.exit(1)

    # Header
    mode = "C++ vs Rust" if args.compare else ("Rust" if args.rust_only else "C++")
    pin_str = ", pinned" if args.pin else ""
    alloc_str = "" if use_mimalloc else ", system alloc"
    print(
        f"{mode} Masstree mttest - {args.threads} threads, {args.duration}s{pin_str}{alloc_str}"
    )

    if args.compare:
        # Comparison mode: run both and show table
        results: List[Tuple[str, Optional[float], Optional[float]]] = []
        for i, test in enumerate(args.tests):
            cpp_ops = run_cpp_benchmark(
                mttest_path,
                args.threads,
                args.duration,
                test,
                args.pin,
                args.quiet,
                args.verbose,
            )
            rust_ops = run_rust_benchmark(
                project_dir,
                args.threads,
                args.duration,
                test,
                args.pin,
                args.quiet,
                args.verbose,
                use_mimalloc,
            )
            results.append((test, cpp_ops, rust_ops))
            # Gap between tests for thermal/system state normalization
            if i < len(args.tests) - 1:
                time.sleep(2)

        print_comparison_table(results, args.markdown, use_color)

    elif args.rust_only:
        # Rust only
        if not args.quiet:
            print(file=sys.stderr)
        for i, test in enumerate(args.tests):
            rust_ops = run_rust_benchmark(
                project_dir,
                args.threads,
                args.duration,
                test,
                args.pin,
                quiet=True,
                verbose=args.verbose,
                use_mimalloc=use_mimalloc,
            )
            if rust_ops is not None:
                print(f"{test + ':':<8} {rust_ops / 1e6:>6.2f} Mops/s")
            else:
                print(f"{test + ':':<8} error")
            # Gap between tests for thermal/system state normalization
            if i < len(args.tests) - 1:
                time.sleep(2)

    else:
        # C++ only (original behavior)
        if not args.quiet:
            print(file=sys.stderr)
        for i, test in enumerate(args.tests):
            cpp_ops = run_cpp_benchmark(
                mttest_path,
                args.threads,
                args.duration,
                test,
                args.pin,
                quiet=True,
                verbose=args.verbose,
            )
            if cpp_ops is not None:
                print(f"{test + ':':<8} {cpp_ops / 1e6:>6.2f} Mops/s")
            else:
                print(f"{test + ':':<8} error")
            # Gap between tests for thermal/system state normalization
            if i < len(args.tests) - 1:
                time.sleep(2)


if __name__ == "__main__":
    main()
