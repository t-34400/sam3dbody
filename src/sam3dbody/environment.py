"""Environment diagnostics for upstream SAM 3D Body integration."""

from __future__ import annotations

import importlib.util
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

from sam3dbody.adapters import Sam3DBodyUpstreamRepository


DEFAULT_INFERENCE_MODULES = (
    "torch",
    "cv2",
    "yacs",
    "skimage",
    "einops",
    "timm",
    "pandas",
    "rich",
    "hydra",
    "huggingface_hub",
)


@dataclass(frozen=True)
class Sam3DBodyEnvironmentReport:
    """Wrapper-owned diagnostic report for inference environment readiness."""

    upstream_root: Path
    upstream_exists: bool
    upstream_package_exists: bool
    weights_path: Path | None
    weights_exists: bool | None
    mhr_path: Path | None
    mhr_exists: bool | None
    modules: Mapping[str, bool]
    torch_cuda_available: bool | None

    @property
    def ready_for_inference(self) -> bool:
        """Return whether required local preconditions appear satisfied."""
        return (
            self.upstream_exists
            and self.upstream_package_exists
            and self.weights_exists is True
            and all(self.modules.values())
            and self.torch_cuda_available is True
            and self.mhr_exists is not False
        )

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serializable report."""
        return {
            "upstream_root": str(self.upstream_root),
            "upstream_exists": self.upstream_exists,
            "upstream_package_exists": self.upstream_package_exists,
            "weights_path": str(self.weights_path) if self.weights_path is not None else None,
            "weights_exists": self.weights_exists,
            "mhr_path": str(self.mhr_path) if self.mhr_path is not None else None,
            "mhr_exists": self.mhr_exists,
            "modules": dict(self.modules),
            "torch_cuda_available": self.torch_cuda_available,
            "ready_for_inference": self.ready_for_inference,
        }


def check_environment(
    *,
    upstream_root: str | Path | None = None,
    weights_path: str | Path | None = None,
    mhr_path: str | Path | None = None,
    modules: tuple[str, ...] = DEFAULT_INFERENCE_MODULES,
) -> Sam3DBodyEnvironmentReport:
    """Check local prerequisites without importing upstream code or loading weights."""
    repository = (
        Sam3DBodyUpstreamRepository.from_source_tree()
        if upstream_root is None
        else Sam3DBodyUpstreamRepository(root=Path(upstream_root))
    )
    resolved_weights = Path(weights_path) if weights_path is not None else None
    resolved_mhr = Path(mhr_path) if mhr_path is not None else None
    module_status = {module: _module_available(module) for module in modules}
    return Sam3DBodyEnvironmentReport(
        upstream_root=repository.root,
        upstream_exists=repository.exists,
        upstream_package_exists=(repository.root / "sam_3d_body").is_dir(),
        weights_path=resolved_weights,
        weights_exists=resolved_weights.is_file() if resolved_weights is not None else None,
        mhr_path=resolved_mhr,
        mhr_exists=resolved_mhr.exists() if resolved_mhr is not None else None,
        modules=module_status,
        torch_cuda_available=_torch_cuda_available() if module_status.get("torch") else None,
    )


def _module_available(module_name: str) -> bool:
    try:
        return importlib.util.find_spec(module_name) is not None
    except (ImportError, ValueError):
        return False


def _torch_cuda_available() -> bool | None:
    try:
        import torch  # type: ignore[import-not-found]
    except Exception:
        return None
    try:
        return bool(torch.cuda.is_available())
    except Exception:
        return None
