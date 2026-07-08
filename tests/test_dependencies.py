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

def _read_pyproject() -> str:
    return Path("pyproject.toml").read_text()


def _extract_toml_array(text: str, key: str) -> list[str]:
    marker = f"{key} = ["
    start = text.index(marker) + len(marker)
    end = text.index("]", start)
    values: list[str] = []
    for line in text[start:end].splitlines():
        stripped = line.strip().rstrip(",")
        if stripped.startswith('"') and stripped.endswith('"'):
            values.append(stripped.strip('"'))
    return values


def test_base_dependencies_remain_lightweight() -> None:
    text = _read_pyproject()

    assert _extract_toml_array(text, "dependencies") == []


def test_inference_extra_is_declared_for_upstream_dependencies() -> None:
    text = _read_pyproject()
    inference = _extract_toml_array(text, "inference")

    assert "opencv-python" in inference
    assert "huggingface_hub" in inference
    assert "networkx==3.2.1" in inference


def test_platform_specific_upstream_dependencies_are_not_base_requirements() -> None:
    text = _read_pyproject()
    base_dependencies = _extract_toml_array(text, "dependencies")
    inference = _extract_toml_array(text, "inference")

    assert "torch" not in base_dependencies
    assert "detectron2" not in inference
    assert not any("git+https://github.com/facebookresearch/sam3" in dep for dep in inference)
