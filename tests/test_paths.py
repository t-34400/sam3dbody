from pathlib import Path

import pytest

from sam3dbody import paths


def test_default_upstream_root_uses_source_checkout_when_pyproject_present() -> None:
    assert paths.default_upstream_root().as_posix().endswith("third_party/sam-3d-body")


def test_default_upstream_root_uses_cwd_local_for_installed_layout(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    fake_site = tmp_path / ".venv" / "lib" / "python3.10" / "site-packages"
    workdir = tmp_path / "work"
    fake_site.mkdir(parents=True)
    workdir.mkdir()
    monkeypatch.setattr(paths, "_installed_source_root", lambda: fake_site)

    resolved = paths.default_upstream_root(cwd=workdir)

    assert resolved == workdir / ".local" / "upstream" / "sam-3d-body"
    assert ".venv" not in resolved.as_posix()
    assert "site-packages" not in resolved.as_posix()
