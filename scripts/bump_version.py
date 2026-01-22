#!/usr/bin/env python3
"""Bump patch version (0.X.Y -> 0.X.(Y+1)) in Cargo.toml and README.md."""

import re
import sys
from pathlib import Path


def bump_version(version: str) -> str:
    """Increment the patch version."""
    match = re.match(r"^(\d+)\.(\d+)\.(\d+)$", version)
    if not match:
        raise ValueError(f"Invalid version format: {version}")
    major, minor, patch = match.groups()
    return f"{major}.{minor}.{int(patch) + 1}"


def update_cargo_toml(path: Path, dry_run: bool = False) -> tuple[str, str]:
    """Update version in Cargo.toml, return (old_version, new_version)."""
    content = path.read_text()

    # Match version = "X.Y.Z" at the start of lines (package section)
    pattern = r'^(version\s*=\s*")(\d+\.\d+\.\d+)(")$'
    match = re.search(pattern, content, re.MULTILINE)

    if not match:
        raise ValueError("Could not find version in Cargo.toml")

    old_version = match.group(2)
    new_version = bump_version(old_version)

    new_content = re.sub(
        pattern, rf"\g<1>{new_version}\g<3>", content, count=1, flags=re.MULTILINE
    )

    if not dry_run:
        path.write_text(new_content)

    return old_version, new_version


def update_readme(
    path: Path, old_version: str, new_version: str, dry_run: bool = False
) -> int:
    """Update version references in README.md, return count of replacements."""
    content = path.read_text()

    # Replace all occurrences of the old version
    new_content, count = re.subn(re.escape(old_version), new_version, content)

    if count > 0 and not dry_run:
        path.write_text(new_content)

    return count


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Bump patch version in Cargo.toml and README.md"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be changed without modifying files",
    )
    args = parser.parse_args()

    # Find project root (where Cargo.toml is)
    script_dir = Path(__file__).parent
    project_root = script_dir.parent

    cargo_toml = project_root / "Cargo.toml"
    readme = project_root / "README.md"

    if not cargo_toml.exists():
        print(f"Error: {cargo_toml} not found", file=sys.stderr)
        sys.exit(1)

    if not readme.exists():
        print(f"Error: {readme} not found", file=sys.stderr)
        sys.exit(1)

    try:
        old_version, new_version = update_cargo_toml(cargo_toml, dry_run=args.dry_run)
        readme_count = update_readme(
            readme, old_version, new_version, dry_run=args.dry_run
        )

        if args.dry_run:
            print(f"Would bump: {old_version} -> {new_version}")
            print("  Cargo.toml: 1 occurrence")
            print(f"  README.md: {readme_count} occurrence(s)")
        else:
            print(f"Bumped: {old_version} -> {new_version}")
            print("  Updated Cargo.toml")
            print(f"  Updated README.md ({readme_count} occurrence(s))")

    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
