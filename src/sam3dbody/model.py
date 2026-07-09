"""Public model interface for the sam3dbody wrapper."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .adapters.upstream import Sam3DBodyUpstreamAdapter
from .adapters.loader import Sam3DBodyLoadConfig, Sam3DBodyLoadedModel
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
    adapter: Sam3DBodyUpstreamAdapter | None = None

    @classmethod
    def from_pretrained(
        cls,
        weights_path: str | Path | None = None,
        *,
        device: str | None = None,
        config: dict[str, Any] | None = None,
    ) -> "Sam3DBodyModel":
        """Create a wrapper model handle from explicit model settings."""
        resolved_weights = Path(weights_path) if weights_path is not None else None
        copied_config = dict(config) if config is not None else None
        return cls(
            weights_path=resolved_weights,
            device=device,
            config=copied_config,
            adapter=Sam3DBodyUpstreamAdapter.from_source_tree(),
        )

    def load(self) -> Sam3DBodyLoadedModel:
        """Load upstream model weights without running prediction."""
        if self.weights_path is None:
            raise ValueError("weights_path is required to load SAM 3D Body upstream weights.")
        adapter = self.adapter or Sam3DBodyUpstreamAdapter.from_source_tree()
        load_config = Sam3DBodyLoadConfig.from_values(
            self.weights_path,
            device=self.device,
            mhr_path=self.config.get("mhr_path") if self.config else None,
            extra=self.config,
        )
        return adapter.load(load_config)

    def predict(self, image: Any) -> Sam3DBodyResult:
        """Run single-image prediction.

        Actual upstream inference is intentionally not implemented in this
        initial package skeleton.
        """
        if self.adapter is None:
            adapter = Sam3DBodyUpstreamAdapter.from_source_tree()
        else:
            adapter = self.adapter
        return adapter.predict(image)
