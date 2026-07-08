from pathlib import Path

from sam3dbody.adapters import Sam3DBodyUpstreamAdapter


def test_upstream_dependency_inspection_is_static() -> None:
    report = Sam3DBodyUpstreamAdapter.from_source_tree().inspect_dependencies()

    assert report.files_scanned > 0
    assert "torch" in report.external_modules
    assert "cv2" in report.external_modules
    assert "sam_3d_body" in report.import_modules
    assert "sam_3d_body" not in report.external_modules


def test_upstream_dependency_report_is_serializable() -> None:
    payload = Sam3DBodyUpstreamAdapter.from_source_tree().inspect_dependencies().to_dict()

    assert isinstance(payload["repository_root"], str)
    assert isinstance(payload["external_modules"], list)


def test_upstream_adapter_accepts_explicit_repository_root(tmp_path: Path) -> None:
    upstream_root = tmp_path / "sam-3d-body"
    upstream_root.mkdir()
    (upstream_root / "sample.py").write_text("import torch\n")

    report = Sam3DBodyUpstreamAdapter.from_repository_root(upstream_root).inspect_dependencies()

    assert report.repository_root == upstream_root
    assert report.external_modules == ("torch",)
