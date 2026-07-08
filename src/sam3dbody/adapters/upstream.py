"""Adapter boundary for the upstream SAM 3D Body implementation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from sam3dbody.exceptions import Sam3DBodyNotImplementedError
from .dependencies import UpstreamDependencyReport, inspect_upstream_dependencies


@dataclass(frozen=True)
class Sam3DBodyUpstreamRepository:
    """Location of the upstream SAM 3D Body source tree."""

    root: Path

    @classmethod
    def from_source_tree(cls) -> "Sam3DBodyUpstreamRepository":
        """Return the default upstream repository location in this source tree."""
        project_root = Path(__file__).resolve().parents[3]
        return cls(root=project_root / "third_party" / "sam-3d-body")

    @property
    def exists(self) -> bool:
        """Return whether the upstream source tree is present."""
        return self.root.is_dir()


@dataclass(frozen=True)
class Sam3DBodyUpstreamAdapter:
    """Internal boundary for upstream model access.

    This adapter intentionally avoids importing upstream modules until the
    inference contract is specified. It records where the upstream source tree
    lives and provides the single future integration point for model calls.
    """

    repository: Sam3DBodyUpstreamRepository

    @classmethod
    def from_source_tree(cls) -> "Sam3DBodyUpstreamAdapter":
        """Create an adapter for the default source-tree upstream location."""
        return cls(repository=Sam3DBodyUpstreamRepository.from_source_tree())

    @classmethod
    def from_repository_root(cls, repository_root: str | Path) -> "Sam3DBodyUpstreamAdapter":
        """Create an adapter for an explicit upstream source-tree location."""
        return cls(repository=Sam3DBodyUpstreamRepository(root=Path(repository_root)))

    def inspect_dependencies(self) -> UpstreamDependencyReport:
        """Return a static dependency report for the upstream source tree."""
        return inspect_upstream_dependencies(self.repository.root)

    def predict(self, image: Any) -> Any:
        """Call upstream inference once the integration contract is specified."""
        raise Sam3DBodyNotImplementedError(
            "SAM 3D Body upstream adapter prediction is not implemented yet."
        )
