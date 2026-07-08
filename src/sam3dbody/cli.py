"""Command line interface for wrapper diagnostics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from sam3dbody.adapters import Sam3DBodyUpstreamAdapter


def build_parser() -> argparse.ArgumentParser:
    """Create the command line parser."""
    parser = argparse.ArgumentParser(prog="sam3dbody")
    subcommands = parser.add_subparsers(dest="command")

    inspect_deps = subcommands.add_parser(
        "inspect-deps",
        help="statically inspect upstream import requirements",
    )
    inspect_deps.add_argument(
        "--upstream-root",
        type=Path,
        default=None,
        help="path to an upstream sam-3d-body source tree",
    )
    inspect_deps.add_argument(
        "--json",
        action="store_true",
        help="print the dependency report as JSON",
    )

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the command line interface."""
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "inspect-deps":
        if args.upstream_root is None:
            adapter = Sam3DBodyUpstreamAdapter.from_source_tree()
        else:
            adapter = Sam3DBodyUpstreamAdapter.from_repository_root(args.upstream_root)
        report = adapter.inspect_dependencies()
        if args.json:
            print(json.dumps(report.to_dict(), indent=2, sort_keys=True))
        else:
            print(f"repository_root: {report.repository_root}")
            print(f"files_scanned: {report.files_scanned}")
            print("external_modules:")
            for module in report.external_modules:
                print(f"  - {module}")
        return 0

    parser.print_help()
    return 0
