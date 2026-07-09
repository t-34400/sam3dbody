from pathlib import Path
import sys

import pytest

from sam3dbody.adapters import (
    Sam3DBodyLoadConfig,
    Sam3DBodyUpstreamAdapter,
    Sam3DBodyUpstreamLoadError,
    Sam3DBodyUpstreamRepository,
)
from sam3dbody.adapters.loader import Sam3DBodyUpstreamLoader
from sam3dbody.model import Sam3DBodyModel


def test_load_config_preserves_explicit_values() -> None:
    config = Sam3DBodyLoadConfig.from_values(
        "checkpoints/model.ckpt",
        device="cpu",
        mhr_path="assets/mhr_model.pt",
        extra={"threshold": 0.5},
    )

    assert config.checkpoint_path == Path("checkpoints/model.ckpt")
    assert config.device == "cpu"
    assert config.mhr_path == Path("assets/mhr_model.pt")
    assert config.extra == {"threshold": 0.5}
    assert config.to_upstream_kwargs() == {
        "checkpoint_path": "checkpoints/model.ckpt",
        "device": "cpu",
        "mhr_path": "assets/mhr_model.pt",
    }


def test_loader_rejects_missing_repository(tmp_path: Path) -> None:
    loader = Sam3DBodyUpstreamLoader(Sam3DBodyUpstreamRepository(tmp_path / "missing"))

    with pytest.raises(Sam3DBodyUpstreamLoadError, match="Upstream repository does not exist"):
        loader.load(Sam3DBodyLoadConfig.from_values(tmp_path / "model.ckpt"))


def test_loader_rejects_missing_checkpoint(tmp_path: Path) -> None:
    repo = tmp_path / "sam-3d-body"
    (repo / "sam_3d_body").mkdir(parents=True)
    loader = Sam3DBodyUpstreamLoader(Sam3DBodyUpstreamRepository(repo))

    with pytest.raises(Sam3DBodyUpstreamLoadError, match="Checkpoint does not exist"):
        loader.load(Sam3DBodyLoadConfig.from_values(tmp_path / "missing.ckpt"))


def test_loader_imports_upstream_lazily_and_restores_sys_path(tmp_path: Path) -> None:
    repo = tmp_path / "sam-3d-body"
    package = repo / "sam_3d_body"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text(
        "def load_sam_3d_body(checkpoint_path, device='cuda', mhr_path=''):\n"
        "    return ({'checkpoint_path': checkpoint_path, 'device': device, 'mhr_path': mhr_path}, {'ok': True})\n"
    )
    checkpoint = tmp_path / "model.ckpt"
    checkpoint.write_text("fake")
    mhr = tmp_path / "mhr_model.pt"
    mhr.write_text("fake")
    original_sys_path = list(sys.path)

    loaded = Sam3DBodyUpstreamLoader(Sam3DBodyUpstreamRepository(repo)).load(
        Sam3DBodyLoadConfig.from_values(checkpoint, device="cpu", mhr_path=mhr)
    )

    assert loaded.model == {
        "checkpoint_path": str(checkpoint),
        "device": "cpu",
        "mhr_path": str(mhr),
    }
    assert loaded.model_config == {"ok": True}
    assert sys.path == original_sys_path


def test_adapter_exposes_load_boundary(tmp_path: Path) -> None:
    repo = tmp_path / "sam-3d-body"
    package = repo / "sam_3d_body"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text(
        "def load_sam_3d_body(checkpoint_path, device='cuda', mhr_path=''):\n"
        "    return ('model', 'config')\n"
    )
    checkpoint = tmp_path / "model.ckpt"
    checkpoint.write_text("fake")

    adapter = Sam3DBodyUpstreamAdapter.from_repository_root(repo)
    loaded = adapter.load(Sam3DBodyLoadConfig.from_values(checkpoint, device="cpu"))

    assert loaded.model == "model"
    assert loaded.model_config == "config"


def test_public_model_load_requires_weights_path() -> None:
    model = Sam3DBodyModel.from_pretrained(device="cpu")

    with pytest.raises(ValueError, match="weights_path is required"):
        model.load()
