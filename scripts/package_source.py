#!/usr/bin/env python3
"""Create a clean source archive for this repository."""

from __future__ import annotations

import argparse
import fnmatch
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile


EXCLUDED_PART_NAMES = {
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    ".tox",
    ".nox",
    "build",
    "dist",
}

EXCLUDED_NAME_PATTERNS = (
    "*.pyc",
    "*.pyo",
    "*.egg-info",
)


def should_exclude(path: Path, root: Path) -> bool:
    """Return whether a repository-relative path is a generated artifact."""
    relative = path.relative_to(root)
    if any(part in EXCLUDED_PART_NAMES for part in relative.parts):
        return True
    return any(fnmatch.fnmatch(part, pattern) for part in relative.parts for pattern in EXCLUDED_NAME_PATTERNS)


def iter_archive_files(root: Path) -> list[Path]:
    """Return source files that should be included in the archive."""
    return [
        path
        for path in sorted(root.rglob("*"))
        if path.is_file() and not should_exclude(path, root)
    ]


def create_source_archive(root: Path, output: Path) -> int:
    """Write a clean zip archive and return the number of archived files."""
    root = root.resolve()
    output = output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)

    files = [path for path in iter_archive_files(root) if path.resolve() != output]
    with ZipFile(output, "w", ZIP_DEFLATED) as archive:
        for path in files:
            archive.write(path, path.relative_to(root).as_posix())
    return len(files)


def main() -> int:
    parser = argparse.ArgumentParser(description="Create a clean sam3dbody source archive.")
    parser.add_argument("output", type=Path, help="Output zip path")
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="Repository root to package",
    )
    args = parser.parse_args()

    count = create_source_archive(args.root, args.output)
    print(f"Wrote {args.output} with {count} files")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
