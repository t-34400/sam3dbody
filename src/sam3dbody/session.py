"""Persistent loaded inference session for SAM 3D Body."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from collections.abc import Iterable
from typing import Any

from .adapters.loader import Sam3DBodyLoadedModel
from .adapters.upstream import Sam3DBodyPredictionOptions, Sam3DBodyUpstreamAdapter
from .result import Sam3DBodyResult
from .validation import Sam3DBodyInputError


@dataclass
class Sam3DBodySession:
    """Loaded wrapper-owned session that can run repeated predictions."""

    adapter: Sam3DBodyUpstreamAdapter
    loaded_model: Sam3DBodyLoadedModel
    _estimator: Any | None = field(default=None, init=False, repr=False)

    @property
    def device(self) -> str:
        """Return the configured execution device for this loaded session."""
        return self.loaded_model.load_config.device

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
        """Run single-image prediction without reloading upstream weights."""
        options = Sam3DBodyPredictionOptions(
            bboxes=bboxes,
            masks=masks,
            cam_int=cam_int,
            bbox_thr=bbox_thr,
            nms_thr=nms_thr,
            use_mask=use_mask,
            inference_type=inference_type,
        )
        return self.adapter.predict(
            image,
            self.loaded_model,
            options=options,
            estimator=self._get_estimator(),
        )


    def predict_many(
        self,
        images: Iterable[str | Path | Any],
        *,
        bboxes: Any | None = None,
        masks: Any | None = None,
        cam_int: Any | None = None,
        bbox_thr: float = 0.5,
        nms_thr: float = 0.3,
        use_mask: bool = False,
        inference_type: str = "full",
    ) -> list[Sam3DBodyResult]:
        """Run ordered repeated single-image prediction with one loaded session."""
        if isinstance(images, (str, Path)):
            raise Sam3DBodyInputError("images must be an iterable of image inputs, not a single path-like image.")
        try:
            image_list = list(images)
        except TypeError as exc:
            raise Sam3DBodyInputError("images must be an iterable of image inputs.") from exc

        estimator = self._get_estimator()
        results: list[Sam3DBodyResult] = []
        for image in image_list:
            options = Sam3DBodyPredictionOptions(
                bboxes=bboxes,
                masks=masks,
                cam_int=cam_int,
                bbox_thr=bbox_thr,
                nms_thr=nms_thr,
                use_mask=use_mask,
                inference_type=inference_type,
            )
            results.append(
                self.adapter.predict(
                    image,
                    self.loaded_model,
                    options=options,
                    estimator=estimator,
                )
            )
        return results

    def _get_estimator(self) -> Any:
        if self._estimator is None:
            self._estimator = self.adapter.create_estimator(self.loaded_model)
        return self._estimator
