"""Structured result objects for public inference outputs."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class Sam3DBodyMetadata:
    """Metadata associated with one wrapper prediction result."""

    model_name: str | None = None
    device: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_name": self.model_name,
            "device": self.device,
            "extra": dict(self.extra),
        }


@dataclass(frozen=True)
class Sam3DBodyPrediction:
    """Prediction for one detected body."""

    score: float | None = None
    bbox_xyxy: tuple[float, float, float, float] | None = None
    vertices: Any | None = None
    faces: Any | None = None
    joints: Any | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "score": self.score,
            "bbox_xyxy": self.bbox_xyxy,
            "vertices": self.vertices,
            "faces": self.faces,
            "joints": self.joints,
            "extra": dict(self.extra),
        }


@dataclass(frozen=True)
class Sam3DBodyResult:
    """Prediction result for one input image."""

    bodies: list[Sam3DBodyPrediction] = field(default_factory=list)
    metadata: Sam3DBodyMetadata = field(default_factory=Sam3DBodyMetadata)

    def to_dict(self) -> dict[str, Any]:
        return {
            "bodies": [body.to_dict() for body in self.bodies],
            "metadata": self.metadata.to_dict(),
        }
