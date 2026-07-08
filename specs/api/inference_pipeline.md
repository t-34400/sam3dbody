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

The initial adapter implementation may only locate the upstream source tree and provide an explicit not-implemented prediction boundary. It must not perform real inference until the upstream call contract and output conversion rules are specified.

## Postprocessing Policy

Postprocessing should convert upstream outputs into wrapper-owned result objects.

Postprocessing should avoid silently dropping meaningful upstream data unless the dropped data is documented as unsupported.

## Diagnostics Policy

Diagnostic outputs may expose additional upstream details when useful, but diagnostic behavior must remain separate from the stable public result format.
