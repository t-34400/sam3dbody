"""Public model interface for the sam3dbody wrapper."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .adapters.loader import Sam3DBodyLoadConfig
from .adapters.upstream import Sam3DBodyPredictionOptions, Sam3DBodyUpstreamAdapter
from .result import Sam3DBodyResult
from .session import Sam3DBodySession


@dataclass(frozen=True)
class Sam3DBodyModel:
    """Wrapper-owned model configuration handle."""

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
        """Create a wrapper model configuration from explicit settings."""
        resolved_weights = Path(weights_path) if weights_path is not None else None
        copied_config = dict(config) if config is not None else None
        return cls(
            weights_path=resolved_weights,
            device=device,
            config=copied_config,
            adapter=Sam3DBodyUpstreamAdapter.from_source_tree(),
        )

    def load(self) -> Sam3DBodySession:
        """Load upstream weights and return a persistent inference session."""
        if self.weights_path is None:
            raise ValueError("weights_path is required to load SAM 3D Body upstream weights.")
        adapter = self.adapter or Sam3DBodyUpstreamAdapter.from_source_tree()
        load_config = Sam3DBodyLoadConfig.from_values(
            self.weights_path,
            device=self.device,
            mhr_path=self.config.get("mhr_path") if self.config else None,
            extra=self.config,
        )
        loaded_model = adapter.load(load_config)
        return Sam3DBodySession(adapter=adapter, loaded_model=loaded_model)

    def predict(
        self,
        image: str | Path | Any,
        *,
        bboxes: Any | None = None,
        masks: Any | None = None,
        cam_int: Any | None = None,
        bbox_thr: float = 0.5,
        nms_thr: float = 0.3,
        use_mask: bool = False,
        inference_type: str = "full",
    ) -> Sam3DBodyResult:
        """Run single-image prediction using a temporary loaded session."""
        session = self.load()
        return session.predict(
            image,
            bboxes=bboxes,
            masks=masks,
            cam_int=cam_int,
            bbox_thr=bbox_thr,
            nms_thr=nms_thr,
            use_mask=use_mask,
            inference_type=inference_type,
        )

    def predict_many(
        self,
        images: Any,
        *,
        bboxes: Any | None = None,
        masks: Any | None = None,
        cam_int: Any | None = None,
        bbox_thr: float = 0.5,
        nms_thr: float = 0.3,
        use_mask: bool = False,
        inference_type: str = "full",
    ) -> list[Sam3DBodyResult]:
        """Run ordered repeated prediction using a temporary loaded session."""
        session = self.load()
        return session.predict_many(
            images,
            bboxes=bboxes,
            masks=masks,
            cam_int=cam_int,
            bbox_thr=bbox_thr,
            nms_thr=nms_thr,
            use_mask=use_mask,
            inference_type=inference_type,
        )
