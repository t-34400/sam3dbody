"""Command line interface for wrapper diagnostics and inference."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from sam3dbody.adapters import Sam3DBodyUpstreamAdapter
from sam3dbody.model import Sam3DBodyModel


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

    infer = subcommands.add_parser(
        "infer",
        help="run single-image SAM 3D Body inference",
    )
    infer.add_argument(
        "image",
        type=Path,
        help="path to an input image",
    )
    infer.add_argument(
        "--weights",
        type=Path,
        required=True,
        help="path to the upstream SAM 3D Body checkpoint",
    )
    infer.add_argument(
        "--output",
        type=Path,
        default=None,
        help="path to write the JSON prediction result; stdout is used when omitted",
    )
    infer.add_argument(
        "--device",
        default="cuda",
        help="device passed to the upstream loader; prediction currently requires CUDA",
    )
    infer.add_argument(
        "--mhr-path",
        type=Path,
        default=None,
        help="optional path to the upstream MHR model checkpoint",
    )
    infer.add_argument(
        "--bbox-thr",
        type=float,
        default=0.5,
        help="upstream detector bbox threshold in [0, 1]",
    )
    infer.add_argument(
        "--nms-thr",
        type=float,
        default=0.3,
        help="upstream detector NMS threshold in [0, 1]",
    )
    infer.add_argument(
        "--inference-type",
        choices=("full", "body", "hand"),
        default="full",
        help="upstream inference mode",
    )

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the command line interface."""
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "inspect-deps":
        return _run_inspect_deps(args)

    if args.command == "infer":
        return _run_infer(args)

    parser.print_help()
    return 0


def _run_inspect_deps(args: argparse.Namespace) -> int:
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


def _run_infer(args: argparse.Namespace) -> int:
    config = {}
    if args.mhr_path is not None:
        config["mhr_path"] = args.mhr_path
    model = Sam3DBodyModel.from_pretrained(
        args.weights,
        device=args.device,
        config=config or None,
    )
    session = model.load()
    result = session.predict(
        args.image,
        bbox_thr=args.bbox_thr,
        nms_thr=args.nms_thr,
        inference_type=args.inference_type,
    )
    payload = json.dumps(_json_safe(result.to_dict()), indent=2, sort_keys=True)
    if args.output is None:
        print(payload)
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload + "\n")
    return 0


def _json_safe(value: object) -> object:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    tolist = getattr(value, "tolist", None)
    if callable(tolist):
        return _json_safe(tolist())
    item = getattr(value, "item", None)
    if callable(item):
        return _json_safe(item())
    return repr(value)
