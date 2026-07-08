from pathlib import Path

import pytest

from sam3dbody import (
    Sam3DBodyMetadata,
    Sam3DBodyModel,
    Sam3DBodyNotImplementedError,
    Sam3DBodyPrediction,
    Sam3DBodyResult,
)


def test_public_api_imports() -> None:
    assert Sam3DBodyModel.__name__ == "Sam3DBodyModel"
    assert Sam3DBodyResult.__name__ == "Sam3DBodyResult"


def test_model_from_pretrained_preserves_explicit_settings() -> None:
    model = Sam3DBodyModel.from_pretrained(
        weights_path="weights/model.pt",
        device="cpu",
        config={"threshold": 0.5},
    )

    assert model.weights_path == Path("weights/model.pt")
    assert model.device == "cpu"
    assert model.config == {"threshold": 0.5}


def test_result_to_dict_uses_wrapper_owned_schema() -> None:
    result = Sam3DBodyResult(
        bodies=[Sam3DBodyPrediction(score=0.9, bbox_xyxy=(1.0, 2.0, 3.0, 4.0))],
        metadata=Sam3DBodyMetadata(model_name="sam-3d-body", device="cpu"),
    )

    assert result.to_dict() == {
        "bodies": [
            {
                "score": 0.9,
                "bbox_xyxy": (1.0, 2.0, 3.0, 4.0),
                "vertices": None,
                "faces": None,
                "joints": None,
                "extra": {},
            }
        ],
        "metadata": {
            "model_name": "sam-3d-body",
            "device": "cpu",
            "extra": {},
        },
    }


def test_predict_reports_missing_inference_adapter() -> None:
    model = Sam3DBodyModel.from_pretrained(device="cpu")

    with pytest.raises(Sam3DBodyNotImplementedError):
        model.predict(object())


def test_python310_compatible_typing() -> None:
    model_source = Path("src/sam3dbody/model.py").read_text()

    assert "typing import Any, Self" not in model_source
    assert "typing import Self" not in model_source
    assert "-> Self" not in model_source


def test_gitignore_excludes_generated_artifacts() -> None:
    gitignore = Path(".gitignore").read_text()

    for pattern in ("__pycache__/", ".pytest_cache/", "*.egg-info/", "build/", "dist/"):
        assert pattern in gitignore


def test_model_uses_upstream_adapter_boundary() -> None:
    model = Sam3DBodyModel.from_pretrained(device="cpu")

    assert model.adapter is not None
    assert model.adapter.repository.root.as_posix().endswith("third_party/sam-3d-body")
    assert model.adapter.repository.exists


def test_public_model_does_not_import_third_party_directly() -> None:
    model_source = Path("src/sam3dbody/model.py").read_text()

    assert "third_party" not in model_source
    assert "sam-3d-body" not in model_source


def test_upstream_dependency_inspection_is_static() -> None:
    from sam3dbody.adapters import Sam3DBodyUpstreamAdapter

    report = Sam3DBodyUpstreamAdapter.from_source_tree().inspect_dependencies()

    assert report.files_scanned > 0
    assert "torch" in report.external_modules
    assert "cv2" in report.external_modules
    assert "sam_3d_body" in report.import_modules
    assert "sam_3d_body" not in report.external_modules


def test_upstream_dependency_report_is_serializable() -> None:
    from sam3dbody.adapters import Sam3DBodyUpstreamAdapter

    payload = Sam3DBodyUpstreamAdapter.from_source_tree().inspect_dependencies().to_dict()

    assert isinstance(payload["repository_root"], str)
    assert isinstance(payload["external_modules"], list)

def test_source_packaging_script_excludes_generated_artifacts(tmp_path: Path) -> None:
    from scripts.package_source import create_source_archive
    from zipfile import ZipFile

    Path("src/sam3dbody/__pycache__").mkdir(parents=True, exist_ok=True)
    Path("src/sam3dbody/__pycache__/generated.pyc").write_bytes(b"cache")
    Path(".pytest_cache").mkdir(exist_ok=True)
    Path(".pytest_cache/README.md").write_text("cache")
    Path("src/sam3dbody.egg-info").mkdir(exist_ok=True)
    Path("src/sam3dbody.egg-info/PKG-INFO").write_text("metadata")

    archive_path = tmp_path / "source.zip"
    create_source_archive(Path.cwd(), archive_path)

    with ZipFile(archive_path) as archive:
        names = archive.namelist()

    assert not any("__pycache__" in name for name in names)
    assert not any(".pytest_cache" in name for name in names)
    assert not any(name.endswith(".pyc") for name in names)
    assert not any(".egg-info" in name for name in names)
    assert "scripts/package_source.py" in names

def test_upstream_adapter_accepts_explicit_repository_root(tmp_path: Path) -> None:
    from sam3dbody.adapters import Sam3DBodyUpstreamAdapter

    upstream_root = tmp_path / "sam-3d-body"
    upstream_root.mkdir()
    (upstream_root / "sample.py").write_text("import torch\n")

    report = Sam3DBodyUpstreamAdapter.from_repository_root(upstream_root).inspect_dependencies()

    assert report.repository_root == upstream_root
    assert report.external_modules == ("torch",)


def test_cli_inspect_deps_outputs_json(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    import json

    from sam3dbody.cli import main

    upstream_root = tmp_path / "sam-3d-body"
    upstream_root.mkdir()
    (upstream_root / "sample.py").write_text("import torch\nimport cv2\n")

    exit_code = main(["inspect-deps", "--upstream-root", str(upstream_root), "--json"])

    payload = json.loads(capsys.readouterr().out)
    assert exit_code == 0
    assert payload["repository_root"] == str(upstream_root)
    assert payload["external_modules"] == ["cv2", "torch"]


def test_pyproject_exposes_valid_cli_entrypoint() -> None:
    pyproject = Path("pyproject.toml").read_text()

    assert "[project.scripts]" in pyproject
    assert 'sam3dbody = "sam3dbody.cli:main"' in pyproject

