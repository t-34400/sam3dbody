"""Lazy upstream model loading helpers."""

from __future__ import annotations

import importlib
import sys
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator, TYPE_CHECKING

from sam3dbody.exceptions import Sam3DBodyError

if TYPE_CHECKING:
    from .upstream import Sam3DBodyUpstreamRepository


@dataclass(frozen=True)
class Sam3DBodyLoadConfig:
    """Explicit settings required to load the upstream model."""

    checkpoint_path: Path
    device: str = "cuda"
    mhr_path: Path | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_values(
        cls,
        checkpoint_path: str | Path,
        *,
        device: str | None = None,
        mhr_path: str | Path | None = None,
        extra: dict[str, Any] | None = None,
    ) -> "Sam3DBodyLoadConfig":
        return cls(
            checkpoint_path=Path(checkpoint_path),
            device=device or "cuda",
            mhr_path=Path(mhr_path) if mhr_path is not None else None,
            extra=dict(extra) if extra is not None else {},
        )

    def to_upstream_kwargs(self) -> dict[str, Any]:
        kwargs: dict[str, Any] = {
            "checkpoint_path": str(self.checkpoint_path),
            "device": self.device,
        }
        if self.mhr_path is not None:
            kwargs["mhr_path"] = str(self.mhr_path)
        return kwargs


@dataclass(frozen=True)
class Sam3DBodyLoadedModel:
    """Wrapper-owned handle for opaque upstream loaded objects."""

    model: Any
    model_config: Any
    load_config: Sam3DBodyLoadConfig
    repository: "Sam3DBodyUpstreamRepository"


class Sam3DBodyUpstreamLoadError(Sam3DBodyError):
    """Raised when the upstream model cannot be loaded."""


class Sam3DBodyUpstreamLoader:
    """Load upstream SAM 3D Body objects behind the adapter boundary."""

    def __init__(self, repository: "Sam3DBodyUpstreamRepository") -> None:
        self.repository = repository

    def load(self, config: Sam3DBodyLoadConfig) -> Sam3DBodyLoadedModel:
        self._validate_repository()
        self._validate_checkpoint(config.checkpoint_path)
        if config.mhr_path is not None and not config.mhr_path.exists():
            raise Sam3DBodyUpstreamLoadError(f"MHR asset does not exist: {config.mhr_path}")

        try:
            with _prepend_sys_path(self.repository.root), _isolated_upstream_modules("sam_3d_body"):
                module = importlib.import_module("sam_3d_body")
                load_fn = getattr(module, "load_sam_3d_body")
                model, model_config = load_fn(**config.to_upstream_kwargs())
        except Exception as exc:  # pragma: no cover - exercised with real upstream deps/checkpoints.
            raise Sam3DBodyUpstreamLoadError(f"Failed to load upstream SAM 3D Body model: {exc}") from exc

        return Sam3DBodyLoadedModel(
            model=model,
            model_config=model_config,
            load_config=config,
            repository=self.repository,
        )

    def _validate_repository(self) -> None:
        if not self.repository.exists:
            raise Sam3DBodyUpstreamLoadError(f"Upstream repository does not exist: {self.repository.root}")
        if not (self.repository.root / "sam_3d_body").is_dir():
            raise Sam3DBodyUpstreamLoadError(
                f"Upstream repository is missing sam_3d_body package: {self.repository.root}"
            )

    @staticmethod
    def _validate_checkpoint(checkpoint_path: Path) -> None:
        if not checkpoint_path.is_file():
            raise Sam3DBodyUpstreamLoadError(f"Checkpoint does not exist: {checkpoint_path}")


@contextmanager
def _prepend_sys_path(path: Path) -> Iterator[None]:
    resolved = str(path.resolve())
    original = list(sys.path)
    if resolved in sys.path:
        sys.path.remove(resolved)
    sys.path.insert(0, resolved)
    try:
        yield
    finally:
        sys.path[:] = original


@contextmanager
def _isolated_upstream_modules(package_name: str) -> Iterator[None]:
    saved = {
        name: module
        for name, module in sys.modules.items()
        if name == package_name or name.startswith(f"{package_name}.")
    }
    for name in list(saved):
        sys.modules.pop(name, None)
    try:
        yield
    finally:
        for name in [
            module_name
            for module_name in sys.modules
            if module_name == package_name or module_name.startswith(f"{package_name}.")
        ]:
            sys.modules.pop(name, None)
        sys.modules.update(saved)
