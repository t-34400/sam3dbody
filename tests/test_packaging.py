from pathlib import Path
from zipfile import ZipFile


def test_gitignore_excludes_generated_artifacts() -> None:
    gitignore = Path(".gitignore").read_text()

    for pattern in ("__pycache__/", ".pytest_cache/", "*.egg-info/", "build/", "dist/"):
        assert pattern in gitignore


def test_source_packaging_script_excludes_generated_artifacts(tmp_path: Path) -> None:
    from scripts.package_source import create_source_archive

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


def test_source_packaging_script_excludes_upstream_checkout(tmp_path: Path) -> None:
    from scripts.package_source import create_source_archive

    upstream_root = Path("third_party/sam-3d-body")
    upstream_root.mkdir(parents=True, exist_ok=True)
    (upstream_root / "sam_3d_body").mkdir(parents=True, exist_ok=True)
    (upstream_root / "sam_3d_body" / "__init__.py").write_text("# upstream")
    archive_path = tmp_path / "source.zip"
    create_source_archive(Path.cwd(), archive_path)

    with ZipFile(archive_path) as archive:
        names = archive.namelist()

    assert ".gitmodules" in names
    assert not any(name.startswith("third_party/sam-3d-body/") for name in names)
    assert not any("/.git/" in name or name.startswith(".git/") for name in names)
