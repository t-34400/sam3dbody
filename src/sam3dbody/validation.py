"""Public prediction input validation helpers."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

from .exceptions import Sam3DBodyError


class Sam3DBodyInputError(Sam3DBodyError, ValueError):
    """Raised when public prediction inputs violate the wrapper contract."""


SUPPORTED_INFERENCE_TYPES = {"full", "body", "hand"}


def validate_prediction_inputs(
    image: str | Path | Any,
    *,
    bboxes: Any | None = None,
    masks: Any | None = None,
    cam_int: Any | None = None,
    bbox_thr: float = 0.5,
    nms_thr: float = 0.3,
    inference_type: str = "full",
    device: str | None = None,
) -> None:
    """Validate the public single-image prediction contract before upstream calls."""
    _validate_cuda_device(device)
    _validate_image(image)
    bbox_count = _validate_bboxes(bboxes)
    mask_count = _validate_masks(masks)
    if masks is not None and bboxes is None:
        raise Sam3DBodyInputError("masks require bboxes because upstream mask-conditioned inference requires boxes.")
    if bbox_count is not None and mask_count is not None and bbox_count != mask_count:
        raise Sam3DBodyInputError(f"bboxes and masks must have the same first dimension, got {bbox_count} and {mask_count}.")
    _validate_tensor_like_cam_int(cam_int)
    _validate_unit_interval("bbox_thr", bbox_thr)
    _validate_unit_interval("nms_thr", nms_thr)
    if inference_type not in SUPPORTED_INFERENCE_TYPES:
        supported = ", ".join(sorted(SUPPORTED_INFERENCE_TYPES))
        raise Sam3DBodyInputError(f"inference_type must be one of: {supported}; got {inference_type!r}.")


def _validate_cuda_device(device: str | None) -> None:
    if device is None:
        return
    normalized = device.lower()
    if normalized == "cuda" or normalized.startswith("cuda:"):
        return
    raise Sam3DBodyInputError(
        "Real upstream prediction currently requires a CUDA device because upstream moves inference batches to 'cuda'. "
        f"Configured device was {device!r}."
    )


def _validate_image(image: str | Path | Any) -> None:
    if isinstance(image, (str, Path)):
        path = Path(image)
        if not path.is_file():
            raise Sam3DBodyInputError(f"image path does not exist: {path}")
        return
    shape = getattr(image, "shape", None)
    if shape is None or len(shape) != 3:
        raise Sam3DBodyInputError("image must be a filesystem path or an array-like RGB image with shape H x W x C.")
    if shape[2] != 3:
        raise Sam3DBodyInputError(f"image array must have 3 channels in the last dimension, got shape {tuple(shape)!r}.")


def _validate_bboxes(bboxes: Any | None) -> int | None:
    if bboxes is None:
        return None
    shape = _shape_of(bboxes)
    if len(shape) != 2 or shape[1] != 4:
        raise Sam3DBodyInputError(f"bboxes must have shape N x 4, got shape {shape!r}.")
    return int(shape[0])


def _validate_masks(masks: Any | None) -> int | None:
    if masks is None:
        return None
    shape = _shape_of(masks)
    if len(shape) != 3:
        raise Sam3DBodyInputError(f"masks must have shape N x H x W, got shape {shape!r}.")
    return int(shape[0])


def _validate_tensor_like_cam_int(cam_int: Any | None) -> None:
    if cam_int is None:
        return
    shape = _shape_of(cam_int)
    if len(shape) != 2 or shape != (3, 3):
        raise Sam3DBodyInputError(f"cam_int must have shape 3 x 3, got shape {shape!r}.")
    if not hasattr(cam_int, "to"):
        raise Sam3DBodyInputError("cam_int must be tensor-like and provide .to(...) for upstream device conversion.")


def _validate_unit_interval(name: str, value: float) -> None:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise Sam3DBodyInputError(f"{name} must be a numeric value in [0, 1], got {value!r}.")
    if value < 0 or value > 1:
        raise Sam3DBodyInputError(f"{name} must be in [0, 1], got {value!r}.")


def _shape_of(value: Any) -> tuple[int, ...]:
    shape = getattr(value, "shape", None)
    if shape is not None:
        return tuple(int(item) for item in shape)
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise Sam3DBodyInputError(f"value must be sequence-like or expose .shape, got {type(value).__name__}.")
    return _nested_sequence_shape(value)


def _nested_sequence_shape(value: Sequence[Any]) -> tuple[int, ...]:
    length = len(value)
    if length == 0:
        return (0,)
    first = value[0]
    if not isinstance(first, Sequence) or isinstance(first, (str, bytes)):
        return (length,)
    inner = _nested_sequence_shape(first)
    for item in value[1:]:
        if not isinstance(item, Sequence) or isinstance(item, (str, bytes)):
            raise Sam3DBodyInputError("nested sequences must be rectangular.")
        if _nested_sequence_shape(item) != inner:
            raise Sam3DBodyInputError("nested sequences must be rectangular.")
    return (length, *inner)
