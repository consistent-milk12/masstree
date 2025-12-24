#!/usr/bin/env python3
"""
Comprehensive inline/inline(always) analysis for Rust codebases.

Analyzes placement of #[inline] and #[inline(always)] attributes to identify:
- Oversized functions with inline(always)
- Cold-path functions that shouldn't force inlining
- Missing inline hints on small hot-path functions
- Code bloat estimation from call site multiplication

Usage:
    python scripts/analyze_inline.py [OPTIONS] [PATH]

Examples:
    python scripts/analyze_inline.py src/
    python scripts/analyze_inline.py src/ --issues-only
    python scripts/analyze_inline.py src/ --json
    python scripts/analyze_inline.py src/ --recommend
"""

import argparse
import json
import re
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Optional


class InlineType(Enum):
    """Type of inline attribute."""
    NONE = "none"
    INLINE = "#[inline]"
    INLINE_ALWAYS = "#[inline(always)]"
    INLINE_NEVER = "#[inline(never)]"


class PathType(Enum):
    """Classification of function as hot or cold path."""
    HOT = "hot"
    COLD = "cold"
    UNKNOWN = "unknown"


class Severity(Enum):
    """Issue severity level."""
    ERROR = "error"      # Definitely wrong
    WARNING = "warning"  # Likely suboptimal
    INFO = "info"        # Worth reviewing
    OK = "ok"            # No issues


@dataclass
class FunctionInfo:
    """Information about a function with inline attributes."""
    file: str
    line: int
    name: str
    inline_type: InlineType
    lines: int
    is_generic: bool = False
    is_unsafe: bool = False
    is_const: bool = False
    is_async: bool = False
    is_pub: bool = False
    is_trait_impl: bool = False
    call_sites: int = 0
    path_type: PathType = PathType.UNKNOWN

    @property
    def code_bloat(self) -> int:
        """Estimated code bloat if inlined at all call sites."""
        if self.inline_type == InlineType.INLINE_ALWAYS:
            return self.lines * max(1, self.call_sites)
        return 0

    @property
    def location(self) -> str:
        """File:line location string."""
        return f"{self.file}:{self.line}"


@dataclass
class Issue:
    """An identified issue with inline placement."""
    func: FunctionInfo
    severity: Severity
    reason: str
    recommendation: str


@dataclass
class AnalysisResult:
    """Complete analysis result."""
    functions: list[FunctionInfo] = field(default_factory=list)
    issues: list[Issue] = field(default_factory=list)
    stats: dict = field(default_factory=dict)


# Pattern matching for function classification
COLD_PATTERNS = [
    r'^new$', r'^new_', r'^create_', r'^build_', r'^make_',  # Construction
    r'^default$', r'^from_', r'^into_', r'^try_from', r'^try_into',  # Conversion
    r'_new$', r'_create$', r'_init$',  # Construction suffixes
    r'^drop$', r'^clone$',  # Lifecycle
    r'^debug$', r'^display$', r'^fmt$',  # Formatting
    r'^error', r'^fail', r'^panic', r'^abort',  # Error handling
    r'split', r'merge', r'rebalance', r'compact',  # Tree operations (rare)
    r'^serialize', r'^deserialize',  # Serialization
    r'^alloc', r'^dealloc', r'^free',  # Allocation
    r'^retire', r'^reclaim',  # Memory reclamation
    r'^init', r'^setup', r'^teardown',  # Initialization
]

HOT_PATTERNS = [
    r'^get$', r'^get_', r'_get$',  # Getters
    r'^set$', r'^set_', r'_set$',  # Setters
    r'^is_', r'^has_', r'^can_',  # Predicates
    r'^find', r'^search', r'^lookup', r'^locate',  # Search
    r'^load$', r'^store$', r'^read$', r'^write$',  # Memory access
    r'^compare', r'^cmp', r'^eq$', r'^ne$', r'^lt$', r'^le$', r'^gt$', r'^ge$',  # Comparison
    r'^len$', r'^size$', r'^count$', r'^capacity$',  # Size queries
    r'^empty$', r'^is_empty$', r'^is_full$',  # State queries
    r'^ptr$', r'^as_ptr', r'^as_ref', r'^as_mut',  # Pointer access
    r'^version', r'^perm', r'^slot', r'^ikey', r'^child',  # Domain-specific hot paths
    r'^next$', r'^prev$', r'^first$', r'^last$',  # Iteration
    r'^push$', r'^pop$', r'^peek$',  # Stack/queue ops
    r'^insert$', r'^remove$', r'^update$',  # Modification (can be hot)
    r'^index$', r'^at$',  # Indexing
    r'^cast$', r'^transmute',  # Type conversion
    r'^prefetch',  # Prefetching
]

# Thresholds
SMALL_FUNCTION_LINES = 10      # Functions <= this are always OK to inline(always)
MEDIUM_FUNCTION_LINES = 25     # Functions <= this need justification for inline(always)
LARGE_FUNCTION_LINES = 50      # Functions > this should rarely use inline(always)
HIGH_CALL_SITES = 5            # Many call sites = higher bloat risk
# Bloat thresholds - only warn if BOTH function is large AND has many call sites
BLOAT_WARNING_LINES = 15       # Only consider bloat for functions >= this size
BLOAT_WARNING_CALL_SITES = 10  # Only consider bloat for functions with >= this many calls
BLOAT_WARNING_THRESHOLD = 500  # Warn if estimated bloat > this (lines * call_sites)


def classify_path_type(fn_name: str) -> PathType:
    """Classify function as likely hot or cold path based on name."""
    fn_lower = fn_name.lower()

    for pattern in COLD_PATTERNS:
        if re.search(pattern, fn_lower):
            return PathType.COLD

    for pattern in HOT_PATTERNS:
        if re.search(pattern, fn_lower):
            return PathType.HOT

    return PathType.UNKNOWN


def parse_inline_type(line: str) -> Optional[InlineType]:
    """Parse inline attribute from a line."""
    if '#[inline(always)]' in line:
        return InlineType.INLINE_ALWAYS
    if '#[inline(never)]' in line:
        return InlineType.INLINE_NEVER
    if '#[inline]' in line and 'inline(' not in line:
        return InlineType.INLINE
    return None


def extract_function_info(content: str, attr_line_idx: int, inline_type: InlineType) -> Optional[tuple]:
    """Extract function name and properties from content starting at attr_line_idx."""
    lines = content.split('\n')

    # Look for fn definition on the line after the attribute (or same line)
    for offset in range(0, 5):  # Check next few lines
        if attr_line_idx + offset >= len(lines):
            break

        fn_line = lines[attr_line_idx + offset]

        # Match function signature
        fn_match = re.search(
            r'(?P<pub>pub\s+)?(?P<unsafe>unsafe\s+)?(?P<const>const\s+)?(?P<async>async\s+)?fn\s+(?P<name>\w+)',
            fn_line
        )

        if fn_match:
            name = fn_match.group('name')
            is_pub = fn_match.group('pub') is not None
            is_unsafe = fn_match.group('unsafe') is not None
            is_const = fn_match.group('const') is not None
            is_async = fn_match.group('async') is not None
            is_generic = '<' in fn_line

            # Check if this is a trait impl by looking backwards
            is_trait_impl = False
            for back_offset in range(1, 20):
                if attr_line_idx - back_offset < 0:
                    break
                prev_line = lines[attr_line_idx - back_offset].strip()
                if prev_line.startswith('impl') and ' for ' in prev_line:
                    is_trait_impl = True
                    break
                if prev_line.startswith('impl') or prev_line.startswith('mod '):
                    break

            # Count function body lines
            fn_start = attr_line_idx + offset
            brace_count = 0
            started = False
            fn_end = fn_start

            for i, line in enumerate(lines[fn_start:], fn_start):
                for char in line:
                    if char == '{':
                        brace_count += 1
                        started = True
                    elif char == '}':
                        brace_count -= 1
                if started and brace_count == 0:
                    fn_end = i
                    break

            fn_lines = fn_end - fn_start + 1

            return (name, fn_lines, is_pub, is_unsafe, is_const, is_async, is_generic, is_trait_impl)

    return None


def count_call_sites(content: str, fn_name: str, exclude_definition: bool = True) -> int:
    """Count approximate call sites for a function."""
    # Match function calls: fn_name( or fn_name::<...>(
    pattern = rf'\b{re.escape(fn_name)}\s*(?:::<[^>]*>)?\s*\('

    matches = list(re.finditer(pattern, content))
    count = len(matches)

    if exclude_definition:
        # Subtract the definition itself (fn name(...))
        def_pattern = rf'fn\s+{re.escape(fn_name)}\s*(?:<[^>]*>)?\s*\('
        def_matches = len(list(re.finditer(def_pattern, content)))
        count = max(0, count - def_matches)

    return count


def analyze_file(file_path: Path, all_content: str) -> list[FunctionInfo]:
    """Analyze a single Rust file for inline attributes."""
    functions = []
    content = file_path.read_text()
    lines = content.split('\n')

    for i, line in enumerate(lines):
        inline_type = parse_inline_type(line)
        if inline_type is None:
            continue

        result = extract_function_info(content, i, inline_type)
        if result is None:
            continue

        name, fn_lines, is_pub, is_unsafe, is_const, is_async, is_generic, is_trait_impl = result

        # Count call sites across all content
        call_sites = count_call_sites(all_content, name)

        func = FunctionInfo(
            file=str(file_path),
            line=i + 1,
            name=name,
            inline_type=inline_type,
            lines=fn_lines,
            is_pub=is_pub,
            is_unsafe=is_unsafe,
            is_const=is_const,
            is_async=is_async,
            is_generic=is_generic,
            is_trait_impl=is_trait_impl,
            call_sites=call_sites,
            path_type=classify_path_type(name),
        )
        functions.append(func)

    return functions


def find_issues(func: FunctionInfo) -> list[Issue]:
    """Identify issues with a function's inline placement."""
    issues = []

    if func.inline_type == InlineType.INLINE_ALWAYS:
        # Issue: Very large function with inline(always) - always bad
        if func.lines > LARGE_FUNCTION_LINES:
            issues.append(Issue(
                func=func,
                severity=Severity.ERROR,
                reason=f"Very large function ({func.lines} lines) with #[inline(always)]",
                recommendation="Use #[inline] to let LLVM decide, or remove attribute entirely",
            ))

        # Issue: Cold path function with inline(always) - bad for icache
        elif func.path_type == PathType.COLD and func.lines > SMALL_FUNCTION_LINES:
            issues.append(Issue(
                func=func,
                severity=Severity.WARNING,
                reason=f"Cold-path function ({func.lines} lines) with #[inline(always)]",
                recommendation="Use #[inline] for cold paths - forced inlining wastes icache",
            ))

        # Issue: High code bloat - only warn for larger functions with many call sites
        # Small hot-path functions (getters, etc.) are fine even with many call sites
        elif (func.lines >= BLOAT_WARNING_LINES and
              func.call_sites >= BLOAT_WARNING_CALL_SITES and
              func.code_bloat > BLOAT_WARNING_THRESHOLD and
              func.path_type != PathType.HOT):
            issues.append(Issue(
                func=func,
                severity=Severity.WARNING,
                reason=f"High code bloat: {func.lines} lines × {func.call_sites} call sites = {func.code_bloat} lines",
                recommendation="Consider #[inline] instead if function is not hot-path critical",
            ))

        # Issue: Medium function that's not clearly hot path
        elif func.lines > MEDIUM_FUNCTION_LINES and func.path_type != PathType.HOT:
            issues.append(Issue(
                func=func,
                severity=Severity.INFO,
                reason=f"Medium function ({func.lines} lines) with #[inline(always)] - verify it's hot path",
                recommendation="Profile to confirm this is a hot path, otherwise use #[inline]",
            ))

    elif func.inline_type == InlineType.NONE:
        # Info: Small hot-path function without inline hint (only pub functions)
        if func.lines <= SMALL_FUNCTION_LINES and func.path_type == PathType.HOT and func.is_pub:
            issues.append(Issue(
                func=func,
                severity=Severity.INFO,
                reason=f"Small hot-path function ({func.lines} lines) without #[inline]",
                recommendation="Consider adding #[inline] for cross-crate optimization",
            ))

    return issues


def analyze_directory(path: Path) -> AnalysisResult:
    """Analyze all Rust files in a directory."""
    result = AnalysisResult()

    # Collect all content for call site counting
    all_files = list(path.rglob('*.rs'))
    all_content = '\n'.join(f.read_text() for f in all_files if f.is_file())

    # Analyze each file
    for file_path in all_files:
        if not file_path.is_file():
            continue

        functions = analyze_file(file_path, all_content)
        result.functions.extend(functions)

        for func in functions:
            issues = find_issues(func)
            result.issues.extend(issues)

    # Calculate stats
    inline_always_count = sum(1 for f in result.functions if f.inline_type == InlineType.INLINE_ALWAYS)
    inline_count = sum(1 for f in result.functions if f.inline_type == InlineType.INLINE)
    inline_never_count = sum(1 for f in result.functions if f.inline_type == InlineType.INLINE_NEVER)

    total_bloat = sum(f.code_bloat for f in result.functions)

    by_severity = defaultdict(int)
    for issue in result.issues:
        by_severity[issue.severity.value] += 1

    result.stats = {
        'total_functions': len(result.functions),
        'inline_always': inline_always_count,
        'inline': inline_count,
        'inline_never': inline_never_count,
        'total_issues': len(result.issues),
        'issues_by_severity': dict(by_severity),
        'estimated_total_bloat': total_bloat,
        'files_analyzed': len(all_files),
    }

    return result


def print_table(headers: list[str], rows: list[list[str]], col_widths: Optional[list[int]] = None):
    """Print a formatted table."""
    if not rows:
        return

    if col_widths is None:
        col_widths = [
            max(len(str(row[i])) for row in [headers] + rows)
            for i in range(len(headers))
        ]

    # Header
    header_line = ' | '.join(h.ljust(w) for h, w in zip(headers, col_widths))
    separator = '-+-'.join('-' * w for w in col_widths)

    print(header_line)
    print(separator)

    for row in rows:
        print(' | '.join(str(cell).ljust(w) for cell, w in zip(row, col_widths)))


def format_output(result: AnalysisResult, args: argparse.Namespace) -> str:
    """Format analysis result for output."""
    if args.json:
        data = {
            'stats': result.stats,
            'issues': [
                {
                    'location': issue.func.location,
                    'function': issue.func.name,
                    'severity': issue.severity.value,
                    'reason': issue.reason,
                    'recommendation': issue.recommendation,
                }
                for issue in result.issues
            ],
            'functions': [
                {
                    'location': f.location,
                    'name': f.name,
                    'inline_type': f.inline_type.value,
                    'lines': f.lines,
                    'call_sites': f.call_sites,
                    'path_type': f.path_type.value,
                    'code_bloat': f.code_bloat,
                }
                for f in result.functions
            ] if args.verbose else [],
        }
        return json.dumps(data, indent=2)

    output = []

    # Header
    output.append("=" * 70)
    output.append("INLINE ATTRIBUTE ANALYSIS")
    output.append("=" * 70)
    output.append("")

    # Stats
    output.append("STATISTICS")
    output.append("-" * 40)
    output.append(f"Files analyzed:        {result.stats['files_analyzed']}")
    output.append(f"Functions with inline: {result.stats['total_functions']}")
    output.append(f"  #[inline(always)]:   {result.stats['inline_always']}")
    output.append(f"  #[inline]:           {result.stats['inline']}")
    output.append(f"  #[inline(never)]:    {result.stats['inline_never']}")
    output.append(f"Estimated code bloat:  {result.stats['estimated_total_bloat']} lines")
    output.append("")

    # Issues summary
    output.append("ISSUES SUMMARY")
    output.append("-" * 40)
    for severity in [Severity.ERROR, Severity.WARNING, Severity.INFO]:
        count = result.stats['issues_by_severity'].get(severity.value, 0)
        if count > 0:
            output.append(f"  {severity.value.upper():10} {count}")
    output.append("")

    # Filter issues
    issues = result.issues
    if args.issues_only:
        issues = [i for i in issues if i.severity in [Severity.ERROR, Severity.WARNING]]

    if not issues:
        output.append("No issues found!")
        return '\n'.join(output)

    # Group by severity
    for severity in [Severity.ERROR, Severity.WARNING, Severity.INFO]:
        severity_issues = [i for i in issues if i.severity == severity]
        if not severity_issues:
            continue

        output.append(f"\n{severity.value.upper()} ({len(severity_issues)})")
        output.append("-" * 70)

        for issue in sorted(severity_issues, key=lambda x: -x.func.lines):
            output.append(f"\n{issue.func.location}")
            output.append(f"  fn {issue.func.name}() - {issue.func.lines} lines, {issue.func.call_sites} call sites")
            output.append(f"  Current: {issue.func.inline_type.value}")
            output.append(f"  Issue: {issue.reason}")
            if args.recommend:
                output.append(f"  Recommendation: {issue.recommendation}")

    # Top bloat analysis
    if args.top_bloat > 0:
        output.append("\n" + "=" * 70)
        output.append(f"TOP {args.top_bloat} FUNCTIONS BY CODE BLOAT")
        output.append("=" * 70)
        output.append("")

        bloat_funcs = sorted(
            [f for f in result.functions if f.inline_type == InlineType.INLINE_ALWAYS],
            key=lambda x: -x.code_bloat
        )[:args.top_bloat]

        output.append(f"{'Location':<45} {'Function':<25} {'Lines':>6} {'Calls':>6} {'Bloat':>8}")
        output.append("-" * 95)

        for f in bloat_funcs:
            output.append(
                f"{f.location:<45} {f.name[:24]:<25} {f.lines:>6} {f.call_sites:>6} {f.code_bloat:>8}"
            )

    # Cold path analysis
    if args.cold_path:
        output.append("\n" + "=" * 70)
        output.append("COLD-PATH FUNCTIONS WITH #[inline(always)]")
        output.append("=" * 70)
        output.append("")

        cold_funcs = [
            f for f in result.functions
            if f.inline_type == InlineType.INLINE_ALWAYS and f.path_type == PathType.COLD
        ]
        cold_funcs.sort(key=lambda x: -x.lines)

        if cold_funcs:
            output.append(f"{'Location':<45} {'Function':<25} {'Lines':>6}")
            output.append("-" * 80)
            for f in cold_funcs:
                output.append(f"{f.location:<45} {f.name[:24]:<25} {f.lines:>6}")
        else:
            output.append("No cold-path functions with #[inline(always)] found.")

    # Verbose: show all functions
    if args.verbose and not args.issues_only:
        output.append("\n" + "=" * 70)
        output.append("ALL FUNCTIONS WITH INLINE ATTRIBUTES")
        output.append("=" * 70)

        headers = ['Location', 'Function', 'Type', 'Lines', 'Calls', 'Bloat', 'Path']
        rows = []

        for f in sorted(result.functions, key=lambda x: -x.lines):
            rows.append([
                f.location,
                f.name[:30],
                f.inline_type.value.replace('#[', '').replace(']', ''),
                str(f.lines),
                str(f.call_sites),
                str(f.code_bloat),
                f.path_type.value,
            ])

        output.append("")
        # Print as simple list for long lists
        for row in rows[:50]:  # Limit to 50
            output.append(f"{row[0]:40} {row[1]:25} {row[2]:15} {row[3]:>5} lines")

        if len(rows) > 50:
            output.append(f"... and {len(rows) - 50} more")

    return '\n'.join(output)


def main():
    parser = argparse.ArgumentParser(
        description='Analyze inline attribute placement in Rust code',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s src/                    # Analyze src/ directory
  %(prog)s src/ --issues-only      # Only show errors and warnings
  %(prog)s src/ --recommend        # Include recommendations
  %(prog)s src/ --json             # Output as JSON
  %(prog)s src/ --verbose          # Show all functions
        """
    )

    parser.add_argument(
        'path',
        nargs='?',
        default='src',
        help='Directory to analyze (default: src/)',
    )

    parser.add_argument(
        '--issues-only', '-i',
        action='store_true',
        help='Only show errors and warnings (skip info)',
    )

    parser.add_argument(
        '--recommend', '-r',
        action='store_true',
        help='Include recommendations for fixing issues',
    )

    parser.add_argument(
        '--json', '-j',
        action='store_true',
        help='Output as JSON',
    )

    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Show all functions, not just issues',
    )

    parser.add_argument(
        '--top-bloat', '-b',
        type=int,
        metavar='N',
        default=0,
        help='Show top N functions by estimated code bloat',
    )

    parser.add_argument(
        '--cold-path', '-c',
        action='store_true',
        help='Show all cold-path functions with inline(always)',
    )

    args = parser.parse_args()

    path = Path(args.path)
    if not path.exists():
        print(f"Error: Path '{path}' does not exist", file=sys.stderr)
        sys.exit(1)

    if not path.is_dir():
        print(f"Error: Path '{path}' is not a directory", file=sys.stderr)
        sys.exit(1)

    result = analyze_directory(path)
    output = format_output(result, args)
    print(output)

    # Exit with error code if there are errors
    error_count = result.stats['issues_by_severity'].get('error', 0)
    if error_count > 0:
        sys.exit(1)


if __name__ == '__main__':
    main()
