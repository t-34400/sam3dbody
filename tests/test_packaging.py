from pathlib import Path
from zipfile import ZipFile


def test_gitignore_excludes_generated_artifacts() -> None:
    gitignore = Path(".gitignore").read_text()

    for pattern in ("__pycache__/", ".pytest_cache/", "*.egg-info/", "build/", "dist/"):
        assert pattern in gitignore


def _write_file(path: Path, content: str | bytes = "content") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(content, bytes):
        path.write_bytes(content)
    else:
        path.write_text(content)


def test_source_packaging_script_excludes_generated_artifacts(tmp_path: Path) -> None:
    from scripts.package_source import create_source_archive

    repo_root = tmp_path / "repo"
    _write_file(repo_root / "scripts" / "package_source.py")
    _write_file(repo_root / "src" / "sam3dbody" / "__init__.py")
    _write_file(repo_root / "src" / "sam3dbody" / "__pycache__" / "generated.pyc", b"cache")
    _write_file(repo_root / ".pytest_cache" / "README.md", "cache")
    _write_file(repo_root / "src" / "sam3dbody.egg-info" / "PKG-INFO", "metadata")

    archive_path = tmp_path / "source.zip"
    create_source_archive(repo_root, archive_path)

    with ZipFile(archive_path) as archive:
        names = archive.namelist()

    assert "src/sam3dbody/__init__.py" in names
    assert not any("__pycache__" in name for name in names)
    assert not any(".pytest_cache" in name for name in names)
    assert not any(name.endswith(".pyc") for name in names)
    assert not any(".egg-info" in name for name in names)
    assert "scripts/package_source.py" in names


def test_source_packaging_script_excludes_upstream_checkout(tmp_path: Path) -> None:
    from scripts.package_source import create_source_archive

    repo_root = tmp_path / "repo"
    _write_file(repo_root / ".gitmodules")
    _write_file(repo_root / "README.md")
    _write_file(repo_root / "third_party" / "sam-3d-body" / "sam_3d_body" / "__init__.py", "# upstream")
    _write_file(repo_root / "third_party" / "sam-3d-body" / ".git" / "HEAD", "ref: main")

    archive_path = tmp_path / "source.zip"
    create_source_archive(repo_root, archive_path)

    with ZipFile(archive_path) as archive:
        names = archive.namelist()

    assert ".gitmodules" in names
    assert "README.md" in names
    assert not any(name.startswith("third_party/sam-3d-body/") for name in names)
    assert not any("/.git/" in name or name.startswith(".git/") for name in names)
