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
