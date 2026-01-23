#!/usr/bin/env python3
"""
Compare C++ and Rust mttest benchmark results from saved output files.
"""

import json
import re
from pathlib import Path
from typing import Dict, Optional

# ANSI colors
GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
BOLD = "\033[1m"
RESET = "\033[0m"


def parse_cpp_output(filepath: Path) -> Optional[float]:
    """Parse C++ mttest output and return total ops/sec."""
    test_name = filepath.stem

    ops_per_sec_values = []
    puts_per_sec_values = []
    gets_per_sec_values = []

    with open(filepath, "r") as f:
        for line in f:
            brace_idx = line.find("{")
            if brace_idx == -1:
                continue
            try:
                data = json.loads(line[brace_idx:])
                if data.get("test") != test_name:
                    continue
                if "ops_per_sec" in data:
                    ops_per_sec_values.append(float(data["ops_per_sec"]))
                if "puts_per_sec" in data:
                    puts_per_sec_values.append(float(data["puts_per_sec"]))
                if "gets_per_sec" in data:
                    gets_per_sec_values.append(float(data["gets_per_sec"]))
            except (json.JSONDecodeError, ValueError, KeyError):
                continue

    if ops_per_sec_values:
        return sum(ops_per_sec_values)
    elif puts_per_sec_values:
        return sum(puts_per_sec_values)
    elif gets_per_sec_values:
        return sum(gets_per_sec_values)
    return None


def parse_rust_output(filepath: Path) -> Optional[float]:
    """Parse Rust mttest output and return total ops/sec."""
    with open(filepath, "r") as f:
        content = f.read()

    # Find the TOTAL line
    total_pattern = re.compile(
        r"^\s*TOTAL\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)",
        re.MULTILINE,
    )
    match = total_pattern.search(content)
    if not match:
        return None

    total_gets = int(match.group(3))
    total_ops_per_sec = int(match.group(6))
    total_puts_per_sec = int(match.group(2))

    # For multi-phase tests (has gets), use ops_per_sec
    # For single-phase write tests, use puts_per_sec
    if total_gets > 0:
        return float(total_ops_per_sec)
    else:
        return float(total_puts_per_sec)


def main():
    project_dir = Path(__file__).parent.parent
    cpp_dir = project_dir / "cpp_benches"
    rust_dir = project_dir / "rust_benches"

    tests = ["rw1", "rw2g98", "rw3", "rw4", "same", "uscale", "wscale"]

    print("=" * 65)
    print("C++ vs Rust Masstree mttest Comparison (12 threads, 10s)")
    print("=" * 65)
    print()

    # Table header
    print(
        f"{'Benchmark':<10} {'C++ (Mops/s)':>12} {'Rust (Mops/s)':>14} {'Rust/C++':>10}"
    )
    print("-" * 48)

    results = []
    for test in tests:
        cpp_file = cpp_dir / f"{test}.txt"
        rust_file = rust_dir / f"{test}.txt"

        cpp_ops = parse_cpp_output(cpp_file) if cpp_file.exists() else None
        rust_ops = parse_rust_output(rust_file) if rust_file.exists() else None

        cpp_mops = cpp_ops / 1e6 if cpp_ops else None
        rust_mops = rust_ops / 1e6 if rust_ops else None

        cpp_str = f"{cpp_mops:.2f}" if cpp_mops else "N/A"
        rust_str = f"{rust_mops:.2f}" if rust_mops else "N/A"

        if cpp_mops and rust_mops:
            ratio = rust_mops / cpp_mops * 100
            if ratio >= 100:
                ratio_str = f"{GREEN}{BOLD}{ratio:.0f}%{RESET}"
            elif ratio >= 80:
                ratio_str = f"{YELLOW}{ratio:.0f}%{RESET}"
            else:
                ratio_str = f"{RED}{ratio:.0f}%{RESET}"
        else:
            ratio_str = "N/A"

        print(f"{test:<10} {cpp_str:>12} {rust_str:>14} {ratio_str:>10}")
        results.append((test, cpp_mops, rust_mops))

    print("-" * 48)
    print()

    # Markdown table
    print("Markdown table:")
    print()
    print("| Benchmark | C++ (Mops/s) | Rust (Mops/s) | Rust/C++ |")
    print("|-----------|--------------|---------------|----------|")
    for test, cpp_mops, rust_mops in results:
        cpp_str = f"{cpp_mops:.2f}" if cpp_mops else "N/A"
        rust_str = f"{rust_mops:.2f}" if rust_mops else "N/A"
        if cpp_mops and rust_mops:
            ratio_str = f"{rust_mops / cpp_mops * 100:.0f}%"
        else:
            ratio_str = "N/A"
        print(f"| {test:<9} | {cpp_str:>12} | {rust_str:>13} | {ratio_str:>8} |")


if __name__ == "__main__":
    main()
