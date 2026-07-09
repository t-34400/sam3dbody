# Inference Pipeline

## Purpose

This document defines the initial wrapper-level inference pipeline responsibilities.

## Status

The detailed inference algorithm is not specified yet.

This document defines responsibility boundaries for later implementation.

## Pipeline Boundary

The wrapper inference pipeline is responsible for connecting public API inputs to upstream model execution and converting outputs back into wrapper-owned result objects.

The upstream implementation remains responsible for model-specific internals.

## Initial Pipeline Stages

The pipeline should be organized around these stages:

```text
input loading or validation
    ->
wrapper preprocessing
    ->
upstream adapter call
    ->
wrapper postprocessing
    ->
structured result creation
```

Each stage should have a clear responsibility.

## Input Validation

The wrapper should validate public inputs before passing them to the upstream implementation when practical.

Validation should provide errors that are useful to downstream users.

## Preprocessing Policy

Preprocessing that affects observable output coordinates must be specified before downstream code depends on the result.

If the upstream implementation performs preprocessing internally, the wrapper must document whether public results are expressed in original input coordinates or upstream preprocessed coordinates.

## Upstream Adapter Policy

Calls into `third_party/sam-3d-body` should be concentrated in wrapper-owned adapter modules.

The adapter boundary should absorb upstream path, import, configuration, and result-format quirks.

The adapter may implement an upstream loader before prediction is implemented. The loader may resolve the upstream repository, temporarily expose it on `sys.path`, import the upstream model builder lazily, and call the upstream model-loading function with explicit checkpoint, device, and MHR asset settings.

The loader must not run image inference. It returns a wrapper-owned load result containing opaque upstream model/config objects and the resolved load settings.

Prediction may be implemented only after the upstream call contract and output conversion rules are specified. The initial prediction adapter may call the upstream `SAM3DBodyEstimator.process_one_image` API after loading the upstream model objects through the wrapper loader.

## Postprocessing Policy

Postprocessing should convert upstream outputs into wrapper-owned result objects.

Postprocessing should avoid silently dropping meaningful upstream data unless the dropped data is documented as unsupported.

## Diagnostics Policy

Diagnostic outputs may expose additional upstream details when useful, but diagnostic behavior must remain separate from the stable public result format.

## Initial Upstream Prediction Contract

The initial upstream prediction integration uses the current upstream `SAM3DBodyEstimator` class.

The wrapper adapter must construct the estimator behind the adapter boundary from a wrapper-owned loaded model handle. The initial integration supports the core SAM 3D Body model only:

* human detector: unsupported by the wrapper unless explicitly injected by a future adapter;
* human segmentor: unsupported by the wrapper unless explicitly injected by a future adapter;
* FOV estimator: unsupported by the wrapper unless explicitly injected by a future adapter.

Without a detector, upstream behavior falls back to full-image inference when no bounding boxes are supplied. Public callers may provide bounding boxes to avoid relying on detector integration.

The adapter prediction call may forward the following explicitly documented upstream options:

* `bboxes`
* `masks`
* `cam_int`
* `bbox_thr`
* `nms_thr`
* `use_mask`
* `inference_type`

The adapter must keep upstream imports lazy and must not make package import depend on upstream inference dependencies.

## Persistent Session Contract

A persistent wrapper session owns a loaded upstream model handle and may reuse an upstream estimator across repeated calls. The session must not reload checkpoint weights for every `predict()` call.

The public immutable model object represents configuration. It may construct a temporary session for convenience prediction, but the session API is the stable path for repeated inference.

Session prediction uses the same single-image input contract and output conversion rules as the adapter prediction call.

## Initial Prediction Validation Contract

The wrapper must validate public single-image prediction inputs before calling the upstream estimator.

The initial validation contract is intentionally conservative:

* filesystem image inputs must refer to an existing file;
* in-memory image inputs must expose an `H x W x C` shape with `C == 3` and are treated as RGB;
* `bboxes`, when supplied, must have shape `N x 4`;
* `masks`, when supplied, must have shape `N x H x W`;
* masks require bboxes because upstream mask-conditioned inference requires explicit boxes;
* if both bboxes and masks are supplied, their first dimension must match;
* `cam_int`, when supplied, must have shape `3 x 3` and must be tensor-like with `.to(...)` because upstream moves it to the batch device;
* `bbox_thr` and `nms_thr` must be numeric values in `[0, 1]`;
* `inference_type` must be one of `full`, `body`, or `hand`.

Invalid public inputs must fail with a wrapper-owned input error before upstream inference is called.

## CUDA Requirement

Real upstream prediction currently requires CUDA because the upstream estimator moves inference batches to the literal device string `"cuda"` internally.

The wrapper may continue to accept a device value at model-load time because the upstream load function accepts one. However, public prediction must not imply working CPU inference. Until upstream device handling is wrapped or patched, prediction with a non-CUDA configured device must fail before calling upstream inference.

## Initial Batch Pipeline

The first `predict_many()` pipeline is ordered repeated single-image inference.

For each input image, the wrapper must run the same validation and conversion path used by `predict()`. The implementation may reuse the loaded upstream estimator but must not bypass single-image validation.

If one image fails validation or upstream inference, the method may raise immediately and does not need to return partial results in the initial contract.
