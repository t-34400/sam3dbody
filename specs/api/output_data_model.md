# Output Data Model

## Purpose

This document defines the initial policy for wrapper-owned inference result objects.

## Status

The exact output schema is not stable yet.

This document defines the direction and constraints for the first structured result model.

## Ownership Rule

Public inference results should be represented by wrapper-owned data structures.

The public API should not expose raw upstream result dictionaries as the default result format.

Raw upstream outputs may be exposed only through an explicitly marked diagnostic or debug field if a later specification allows it.

## Design Goals

Output objects should be:

* explicit about coordinate conventions
* serializable when practical
* easy for downstream adapters to consume
* stable across upstream implementation changes
* capable of representing zero, one, or multiple detected bodies

## Initial Result Shape

The initial output model should distinguish between:

* a full prediction result for one input image
* per-body predictions inside that result
* optional metadata about model execution

A directional shape is:

```text
Sam3DBodyResult
    bodies: list[Sam3DBodyPrediction]
    metadata: Sam3DBodyMetadata
```

Exact class names and fields may change before the public API stabilizes.

## Per-Body Prediction Fields

A per-body prediction may later include:

* body mesh vertices
* mesh faces or topology reference
* joints or keypoints
* pose parameters if available
* shape parameters if available
* camera or projection parameters if available
* confidence or score if available
* bounding box if available

A field must not be treated as available until the upstream output has been inspected and the field is specified.

## Coordinate Convention Requirement

Every geometric field must declare its coordinate system before being used as stable API.

At minimum, future specifications should define whether each field is expressed in:

* original image coordinates
* model preprocessed image coordinates
* camera coordinates
* world coordinates
* model-local coordinates

## Serialization Policy

Result objects should support conversion to plain Python data structures when practical.

Serialization should not require GPU tensors, live model objects, or third-party implementation objects.

## Missing Data Policy

The output model must allow unavailable optional fields to be represented explicitly.

Downstream code must not infer missing fields from undocumented upstream behavior.

## Initial Output Mapping

The initial wrapper-owned conversion maps each upstream body dictionary to `Sam3DBodyPrediction` as follows:

* upstream `bbox` -> `bbox_xyxy` as a four-float tuple;
* upstream mesh faces from the estimator -> `faces`;
* upstream `pred_vertices` -> `vertices`;
* upstream `pred_keypoints_3d` -> `joints`;
* all other supported upstream fields -> `extra`.

The initial upstream output does not expose a stable confidence score from the core model path, so `score` remains `None` unless a later detector integration specifies score propagation.

The result metadata must include the execution device and may include diagnostic information such as upstream field names. The wrapper must not expose the raw upstream result list as the top-level public result.

## Initial Public Field Semantics

The initial wrapper-owned result schema is versioned as `0.1` in `Sam3DBodyMetadata.extra["output_schema_version"]`.

The initial public field names are:

* `Sam3DBodyResult.bodies`: ordered list of body predictions for one input image;
* `Sam3DBodyResult.metadata`: wrapper-owned metadata for that input image;
* `Sam3DBodyPrediction.bbox_xyxy`: four float values when upstream provides `bbox`;
* `Sam3DBodyPrediction.vertices`: upstream `pred_vertices` value without additional coordinate conversion;
* `Sam3DBodyPrediction.faces`: mesh topology faces exposed by the upstream estimator;
* `Sam3DBodyPrediction.joints`: upstream `pred_keypoints_3d` value without additional coordinate conversion;
* `Sam3DBodyPrediction.extra`: remaining converted upstream body fields.

The wrapper records coordinate labels in `Sam3DBodyMetadata.extra["coordinate_conventions"]`:

* `bbox_xyxy`: `upstream_output_image_xyxy_pixels`;
* `vertices`: `upstream_model_3d_coordinates_unverified`;
* `joints`: `upstream_model_3d_coordinates_unverified`;
* `faces`: `mesh_topology_indices`.

These labels are intentionally conservative. The wrapper does not yet perform coordinate correction or claim world/camera coordinate semantics for vertices or joints. Real-model smoke testing must inspect upstream outputs before these labels are promoted to a stable downstream contract.

## Observed Real Smoke Test Output

A real CUDA smoke test against the SAM 3D Body DINOv3 checkpoint and MHR asset has validated the initial wrapper mapping with one full-image body prediction and repeated inference. The observed stable summaries were:

* `success`: `true`;
* `environment.ready_for_inference`: `true`;
* single-image `body_count`: `1`;
* repeated inference with `repeat=3`: `requested_count` and `result_count` were both `3`;
* `vertices`: `float32`, shape `[18439, 3]`;
* `joints`: `float32`, shape `[70, 3]`;
* `faces`: `int64`, shape `[36874, 3]`;
* `bbox_xyxy`: `[0.0, 0.0, 1600.0, 1600.0]` for the full-image smoke input.

These observations validate that the wrapper can load the real upstream model and summarize outputs without embedding large arrays. They do not yet promote vertex or joint coordinates to a stable camera/world coordinate contract; the conservative coordinate labels remain authoritative.
