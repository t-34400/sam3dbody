"""Opt-in real upstream inference smoke test.

This test is intentionally skipped by default. It is for local validation with
real upstream source, CUDA, weights, and a sample image.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from sam3dbody.smoke import Sam3DBodySmokeTestConfig, run_smoke_test


@pytest.mark.real_inference
def test_real_upstream_inference_smoke_from_environment() -> None:
    if os.environ.get("SAM3DBODY_RUN_REAL_SMOKE") != "1":
        pytest.skip("set SAM3DBODY_RUN_REAL_SMOKE=1 to run real upstream inference")

    image = _required_path_env("SAM3DBODY_SMOKE_IMAGE")
    weights = _required_path_env("SAM3DBODY_SMOKE_WEIGHTS")
    upstream_root = _optional_path_env("SAM3DBODY_SMOKE_UPSTREAM_ROOT")
    mhr_path = _optional_path_env("SAM3DBODY_SMOKE_MHR_PATH")
    repeat = int(os.environ.get("SAM3DBODY_SMOKE_REPEAT", "2"))

    report = run_smoke_test(
        Sam3DBodySmokeTestConfig(
            image=image,
            weights_path=weights,
            upstream_root=upstream_root,
            mhr_path=mhr_path,
            repeat=repeat,
        )
    )

    assert report["success"] is True, report
    assert report["single"]["body_count"] >= 0
    assert report["batch"]["requested_count"] == repeat
    assert report["batch"]["result_count"] == repeat


def _required_path_env(name: str) -> Path:
    value = os.environ.get(name)
    if not value:
        pytest.fail(f"{name} must be set when SAM3DBODY_RUN_REAL_SMOKE=1")
    path = Path(value)
    if not path.exists():
        pytest.fail(f"{name} does not exist: {path}")
    return path


def _optional_path_env(name: str) -> Path | None:
    value = os.environ.get(name)
    if not value:
        return None
    path = Path(value)
    if not path.exists():
        pytest.fail(f"{name} does not exist: {path}")
    return path
