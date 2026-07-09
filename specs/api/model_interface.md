# Model Interface

## Purpose

This document defines the initial model-facing interface direction.

## Status

Detailed inference behavior is not fully specified yet.

This document defines the expected responsibilities and boundaries for the first wrapper model interface.

## Interface Responsibility

The wrapper model interface should hide upstream implementation details from downstream projects.

Downstream projects should not need to understand upstream demo script internals, repository paths, or ad hoc runtime setup.

## Initial Model Object

The initial model interface should be centered on a wrapper-owned model object.

The preferred public shape is:

```python
model = Sam3DBodyModel.from_pretrained(...)
result = model.predict(image)
```

The model object is responsible for:

* loading or initializing upstream model components
* selecting the execution device
* applying wrapper-defined input normalization rules
* invoking the upstream implementation through an internal adapter
* converting upstream outputs into wrapper-owned result objects

## Construction Requirements

Model construction should make the following choices explicit:

* model weights location
* execution device
* optional configuration values
* cache or temporary directory behavior, if needed

Implicit global setup should be avoided when practical.

Model construction may create a wrapper-owned handle without immediately loading upstream weights. Loading upstream weights should be explicit and may be exposed through `load()` before prediction is implemented.

## Prediction Requirements

Prediction methods should accept image input through documented input forms.

The initial implementation may support only one image input form, but the supported form must be documented before downstream use.

Prediction methods should return wrapper-owned result objects defined by the output data model specification.

Prediction methods should not return raw upstream dictionaries as the default public result.

## Coordinate and Image Convention Requirements

Input image convention must be explicit before results are consumed by downstream projects.

The model interface specification or a later coordinate-system specification must define:

* accepted image array layout
* color channel convention
* image coordinate origin
* coordinate units
* whether outputs refer to original input image coordinates or preprocessed model coordinates

Until these conventions are specified, downstream projects should treat inference outputs as experimental.

## Batching Policy

Single-image inference should be specified before batch inference.

Batch inference may be added later, but must define ordering, error handling, and result grouping before being treated as stable.

## Error Handling Policy

The wrapper should prefer explicit wrapper-owned exceptions or clear Python exceptions over leaking obscure upstream failures.

Detailed exception classes are not specified yet.

## Initial Skeleton Behavior

Until the upstream inference adapter is implemented, the public model object may be constructible without running inference.

Calling prediction before the adapter is available must fail explicitly through a wrapper-owned exception rather than silently returning incomplete results.

## Initial Prediction Input Contract

The first prediction implementation supports single-image inference only.

Accepted image inputs are:

* a filesystem path expressed as `str` or `pathlib.Path`;
* an in-memory array object accepted by the upstream estimator, intended to be an RGB `H x W x C` image.

When a path is supplied, upstream loads it with OpenCV and interprets it as BGR internally before conversion to RGB. When an array is supplied, upstream expects RGB input.

The initial public method may expose optional keyword arguments that map directly to the wrapper adapter prediction contract. These options are experimental until coordinate and detector integration specifications are stabilized.

Loaded weights must not be cached inside the immutable public model configuration object. `Sam3DBodyModel.load()` returns a wrapper-owned persistent `Sam3DBodySession` that owns the loaded upstream handle and can be reused for repeated predictions.

`Sam3DBodyModel.predict(...)` remains available as a convenience compatibility method. It may create a temporary session internally and therefore may reload upstream weights for each call. Repeated inference should use:

```python
model = Sam3DBodyModel.from_pretrained(...)
session = model.load()
result = session.predict(image)
```

The session is the preferred API for real upstream inference because checkpoint loading is expensive.
