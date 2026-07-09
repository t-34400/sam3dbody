"""Command line interface for wrapper diagnostics and inference."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

from sam3dbody.adapters import Sam3DBodyUpstreamAdapter
from sam3dbody.environment import check_environment
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


    check_env = subcommands.add_parser(
        "check-env",
        help="check local prerequisites for real upstream inference",
    )
    check_env.add_argument(
        "--upstream-root",
        type=Path,
        default=None,
        help="path to an upstream sam-3d-body source tree",
    )
    check_env.add_argument(
        "--weights",
        type=Path,
        default=None,
        help="optional path to the upstream SAM 3D Body checkpoint to verify",
    )
    check_env.add_argument(
        "--mhr-path",
        type=Path,
        default=None,
        help="optional path to the upstream MHR asset to verify",
    )
    check_env.add_argument(
        "--json",
        action="store_true",
        help="print the environment report as JSON",
    )
    check_env.add_argument(
        "--strict",
        action="store_true",
        help="exit with status 1 when the environment is not ready for inference",
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
        "--bboxes-json",
        type=Path,
        default=None,
        help="optional JSON file containing bboxes as [[x1, y1, x2, y2], ...] or {'bboxes': ...}",
    )
    infer.add_argument(
        "--cam-int-json",
        type=Path,
        default=None,
        help="optional JSON file containing a 3x3 camera intrinsics matrix or {'cam_int': ...}",
    )
    infer.add_argument(
        "--masks-json",
        type=Path,
        default=None,
        help="optional JSON file containing masks as [[[...]], ...] or {'masks': ...}",
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

    if args.command == "check-env":
        return _run_check_env(args)

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



def _run_check_env(args: argparse.Namespace) -> int:
    report = check_environment(
        upstream_root=args.upstream_root,
        weights_path=args.weights,
        mhr_path=args.mhr_path,
    )
    payload = report.to_dict()
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print(f"upstream_root: {payload['upstream_root']}")
        print(f"upstream_exists: {payload['upstream_exists']}")
        print(f"upstream_package_exists: {payload['upstream_package_exists']}")
        print(f"weights_path: {payload['weights_path']}")
        print(f"weights_exists: {payload['weights_exists']}")
        print(f"mhr_path: {payload['mhr_path']}")
        print(f"mhr_exists: {payload['mhr_exists']}")
        print(f"torch_cuda_available: {payload['torch_cuda_available']}")
        print(f"ready_for_inference: {payload['ready_for_inference']}")
        missing_requirements = payload.get("missing_requirements", [])
        if missing_requirements:
            print("missing_requirements:")
            for requirement in missing_requirements:
                print(f"  - {requirement}")
        else:
            print("missing_requirements: none")
        print("modules:")
        for module, available in payload["modules"].items():
            print(f"  - {module}: {available}")
    if args.strict and not report.ready_for_inference:
        return 1
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
        bboxes=_load_bboxes_json(args.bboxes_json),
        masks=_load_masks_json(args.masks_json),
        cam_int=_load_cam_int_json(args.cam_int_json),
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


def _load_bboxes_json(path: Path | None) -> Any | None:
    if path is None:
        return None
    payload = _read_json_file(path)
    if isinstance(payload, dict):
        if "bboxes" not in payload:
            raise ValueError(f"bboxes JSON object must contain a 'bboxes' key: {path}")
        return payload["bboxes"]
    return payload


def _load_masks_json(path: Path | None) -> Any | None:
    if path is None:
        return None
    payload = _read_json_file(path)
    if isinstance(payload, dict):
        if "masks" not in payload:
            raise ValueError(f"masks JSON object must contain a 'masks' key: {path}")
        return payload["masks"]
    return payload


def _load_cam_int_json(path: Path | None) -> Any | None:
    if path is None:
        return None
    payload = _read_json_file(path)
    if isinstance(payload, dict):
        if "cam_int" not in payload:
            raise ValueError(f"cam_int JSON object must contain a 'cam_int' key: {path}")
        payload = payload["cam_int"]
    try:
        import torch  # type: ignore[import-not-found]
    except ImportError as exc:
        raise RuntimeError("--cam-int-json requires torch so the matrix can be passed to upstream as a tensor-like object.") from exc
    return torch.as_tensor(payload, dtype=torch.float32)


def _read_json_file(path: Path) -> Any:
    return json.loads(path.read_text())


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
