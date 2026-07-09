"""Path resolution helpers for wrapper-managed local assets."""

from __future__ import annotations

from pathlib import Path

DEVELOPMENT_UPSTREAM_RELATIVE = Path("third_party/sam-3d-body")
USER_UPSTREAM_RELATIVE = Path(".local/upstream/sam-3d-body")


def default_upstream_root(*, cwd: str | Path | None = None) -> Path:
    """Return the safe default upstream source location for the current install layout."""
    source_root = _installed_source_root()
    if _looks_like_source_checkout(source_root):
        return source_root / DEVELOPMENT_UPSTREAM_RELATIVE
    base = Path.cwd() if cwd is None else Path(cwd)
    return base / USER_UPSTREAM_RELATIVE


def _installed_source_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _looks_like_source_checkout(path: Path) -> bool:
    return (path / "pyproject.toml").is_file() and (path / "src" / "sam3dbody").is_dir()
