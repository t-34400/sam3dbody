# Public API

## Purpose

This document defines the current policy and initial shape for downstream-facing Python APIs.

## Status

The public API is not stable yet.

The project should converge toward a small documented API before downstream projects depend on it broadly.

## Access Rule

Downstream projects must import from `sam3dbody`, not from `third_party/sam-3d-body`.

Public API modules must not require downstream projects to know the upstream repository layout.

## API Design Goals

The public API should:

* provide a small entry point for model loading
* provide a simple inference entry point
* return structured wrapper-owned result objects
* make device, weight, and configuration choices explicit
* hide upstream demo script internals
* avoid leaking raw third-party implementation objects by default

## Initial Intended API Shape

The initial API should converge toward this usage pattern:

```python
from sam3dbody import Sam3DBodyModel

model = Sam3DBodyModel.from_pretrained(weights_path="...")
result = model.predict(image)
```

This example is directional. Exact argument names, accepted image types, and result fields must be defined in the model interface and output data model specifications before being treated as stable.

## Public API Surface

The initial public API may expose:

* `Sam3DBodyModel`
* model loading helpers
* image inference methods
* batch inference methods
* structured result classes
* lightweight configuration classes

The public API should remain smaller than the internal wrapper implementation.

## Stability Policy

Until the first stable release, public API names may change.

When downstream projects begin depending on the package, API changes should be accompanied by:

1. specification updates
2. tests for observable behavior
3. migration notes when practical
