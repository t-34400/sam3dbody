"""Adapter boundary for the upstream SAM 3D Body implementation."""

from __future__ import annotations

import importlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .dependencies import UpstreamDependencyReport, inspect_upstream_dependencies
from .loader import (
    Sam3DBodyLoadConfig,
    Sam3DBodyLoadedModel,
    Sam3DBodyUpstreamLoader,
    _prepend_sys_path,
)
from sam3dbody.result import Sam3DBodyMetadata, Sam3DBodyPrediction, Sam3DBodyResult


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
class Sam3DBodyPredictionOptions:
    """Explicit options forwarded to the upstream single-image estimator."""

    bboxes: Any | None = None
    masks: Any | None = None
    cam_int: Any | None = None
    bbox_thr: float = 0.5
    nms_thr: float = 0.3
    use_mask: bool = False
    inference_type: str = "full"

    def to_upstream_kwargs(self) -> dict[str, Any]:
        return {
            "bboxes": self.bboxes,
            "masks": self.masks,
            "cam_int": self.cam_int,
            "bbox_thr": self.bbox_thr,
            "nms_thr": self.nms_thr,
            "use_mask": self.use_mask,
            "inference_type": self.inference_type,
        }


@dataclass(frozen=True)
class Sam3DBodyUpstreamAdapter:
    """Internal boundary for upstream model access."""

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

    def load(self, config: Sam3DBodyLoadConfig) -> Sam3DBodyLoadedModel:
        """Load upstream model objects without running image inference."""
        return Sam3DBodyUpstreamLoader(self.repository).load(config)

    def predict(
        self,
        image: str | Path | Any,
        loaded_model: Sam3DBodyLoadedModel,
        *,
        options: Sam3DBodyPredictionOptions | None = None,
    ) -> Sam3DBodyResult:
        """Run upstream single-image inference and convert to wrapper results."""
        prediction_options = options or Sam3DBodyPredictionOptions()
        estimator = self._build_estimator(loaded_model)
        upstream_image = str(image) if isinstance(image, Path) else image
        upstream_outputs = estimator.process_one_image(
            upstream_image,
            **prediction_options.to_upstream_kwargs(),
        )
        return _convert_upstream_outputs(
            upstream_outputs,
            faces=getattr(estimator, "faces", None),
            loaded_model=loaded_model,
        )

    def _build_estimator(self, loaded_model: Sam3DBodyLoadedModel) -> Any:
        with _prepend_sys_path(self.repository.root):
            module = importlib.import_module("sam_3d_body")
            estimator_cls = getattr(module, "SAM3DBodyEstimator")
            return estimator_cls(
                sam_3d_body_model=loaded_model.model,
                model_cfg=loaded_model.model_config,
                human_detector=None,
                human_segmentor=None,
                fov_estimator=None,
            )


def _convert_upstream_outputs(
    upstream_outputs: Any,
    *,
    faces: Any,
    loaded_model: Sam3DBodyLoadedModel,
) -> Sam3DBodyResult:
    bodies = []
    for output in upstream_outputs or []:
        bodies.append(_convert_one_body(output, faces=faces))
    return Sam3DBodyResult(
        bodies=bodies,
        metadata=Sam3DBodyMetadata(
            model_name="sam-3d-body",
            device=loaded_model.load_config.device,
            extra={
                "upstream_fields": sorted(upstream_outputs[0].keys())
                if upstream_outputs and isinstance(upstream_outputs[0], dict)
                else [],
            },
        ),
    )


def _convert_one_body(output: dict[str, Any], *, faces: Any) -> Sam3DBodyPrediction:
    extra = dict(output)
    bbox = extra.pop("bbox", None)
    vertices = extra.pop("pred_vertices", None)
    joints = extra.pop("pred_keypoints_3d", None)
    return Sam3DBodyPrediction(
        score=None,
        bbox_xyxy=_as_float_tuple4(bbox),
        vertices=vertices,
        faces=faces,
        joints=joints,
        extra=extra,
    )


def _as_float_tuple4(value: Any) -> tuple[float, float, float, float] | None:
    if value is None:
        return None
    try:
        values = value.tolist()
    except AttributeError:
        values = list(value)
    if len(values) != 4:
        raise ValueError(f"Expected bbox with four values, got {values!r}")
    return tuple(float(item) for item in values)  # type: ignore[return-value]
