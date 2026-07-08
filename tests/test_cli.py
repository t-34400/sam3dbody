import json
from pathlib import Path

import pytest

from sam3dbody.cli import main


def test_cli_inspect_deps_outputs_json(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
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
