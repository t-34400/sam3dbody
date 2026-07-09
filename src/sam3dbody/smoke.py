"""Real-inference smoke test helpers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .adapters.upstream import Sam3DBodyUpstreamAdapter
from .environment import check_environment
from .model import Sam3DBodyModel
from .result import Sam3DBodyPrediction, Sam3DBodyResult


@dataclass(frozen=True)
class Sam3DBodySmokeTestConfig:
    """Inputs for one real-inference smoke test run."""

    image: Path
    weights_path: Path
    device: str = "cuda"
    mhr_path: Path | None = None
    upstream_root: Path | None = None
    repeat: int = 0
    skip_env_check: bool = False


def run_smoke_test(config: Sam3DBodySmokeTestConfig) -> dict[str, Any]:
    """Run a minimal real-inference smoke test and return a JSON-ready report."""
    environment = check_environment(
        upstream_root=config.upstream_root,
        weights_path=config.weights_path,
        mhr_path=config.mhr_path,
    )
    report: dict[str, Any] = {
        "smoke_schema_version": "0.1",
        "image": str(config.image),
        "weights_path": str(config.weights_path),
        "device": config.device,
        "mhr_path": str(config.mhr_path) if config.mhr_path is not None else None,
        "upstream_root": str(config.upstream_root) if config.upstream_root is not None else None,
        "environment": environment.to_dict(),
        "single": None,
        "batch": None,
    }
    if not config.skip_env_check and not environment.ready_for_inference:
        report["success"] = False
        report["message"] = "environment is not ready for real upstream inference"
        return report

    model_config = {"mhr_path": config.mhr_path} if config.mhr_path is not None else None
    adapter = (
        Sam3DBodyUpstreamAdapter.from_repository_root(config.upstream_root)
        if config.upstream_root is not None
        else None
    )
    model = Sam3DBodyModel(
        weights_path=config.weights_path,
        device=config.device,
        config=model_config,
        adapter=adapter,
    )
    session = model.load()
    single_result = session.predict(config.image)
    report["single"] = summarize_result(single_result)

    if config.repeat > 0:
        batch_results = session.predict_many([config.image] * config.repeat)
        report["batch"] = {
            "requested_count": config.repeat,
            "result_count": len(batch_results),
            "results": [summarize_result(result) for result in batch_results],
        }
    report["success"] = True
    report["message"] = "real upstream inference smoke test completed"
    return report


def summarize_result(result: Sam3DBodyResult) -> dict[str, Any]:
    """Summarize a wrapper result without embedding large tensors or arrays."""
    return {
        "body_count": len(result.bodies),
        "metadata": result.metadata.to_dict(),
        "bodies": [_summarize_body(body) for body in result.bodies],
    }


def _summarize_body(body: Sam3DBodyPrediction) -> dict[str, Any]:
    return {
        "score": body.score,
        "bbox_xyxy": body.bbox_xyxy,
        "vertices": _value_summary(body.vertices),
        "faces": _value_summary(body.faces),
        "joints": _value_summary(body.joints),
        "extra_keys": sorted(body.extra.keys()),
        "extra": {key: _value_summary(value) for key, value in sorted(body.extra.items())},
    }


def _value_summary(value: Any) -> dict[str, Any]:
    if value is None:
        return {"present": False, "type": "NoneType"}
    summary: dict[str, Any] = {
        "present": True,
        "type": type(value).__name__,
    }
    shape = getattr(value, "shape", None)
    if shape is not None:
        summary["shape"] = [int(item) for item in shape]
    elif isinstance(value, (list, tuple)):
        summary["shape"] = list(_nested_shape(value))
        summary["length"] = len(value)
    dtype = getattr(value, "dtype", None)
    if dtype is not None:
        summary["dtype"] = str(dtype)
    if isinstance(value, (str, int, float, bool)):
        summary["value"] = value
    return summary


def _nested_shape(value: Any) -> tuple[int, ...]:
    if not isinstance(value, (list, tuple)):
        return ()
    if not value:
        return (0,)
    return (len(value), *_nested_shape(value[0]))
