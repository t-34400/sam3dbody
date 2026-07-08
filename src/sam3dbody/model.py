"""Public model interface for the sam3dbody wrapper."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Self

from .exceptions import Sam3DBodyNotImplementedError
from .result import Sam3DBodyResult


@dataclass(frozen=True)
class Sam3DBodyModel:
    """Wrapper-owned model handle.

    This class intentionally does not call the upstream implementation yet.
    It establishes the public construction and prediction boundary first.
    """

    weights_path: Path | None = None
    device: str | None = None
    config: dict[str, Any] | None = None

    @classmethod
    def from_pretrained(
        cls,
        weights_path: str | Path | None = None,
        *,
        device: str | None = None,
        config: dict[str, Any] | None = None,
    ) -> Self:
        """Create a wrapper model handle from explicit model settings."""
        resolved_weights = Path(weights_path) if weights_path is not None else None
        copied_config = dict(config) if config is not None else None
        return cls(weights_path=resolved_weights, device=device, config=copied_config)

    def predict(self, image: Any) -> Sam3DBodyResult:
        """Run single-image prediction.

        Actual upstream inference is intentionally not implemented in this
        initial package skeleton.
        """
        raise Sam3DBodyNotImplementedError(
            "SAM 3D Body inference adapter is not implemented yet."
        )
