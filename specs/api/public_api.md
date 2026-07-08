# Public API

## Purpose

This document defines the current policy for downstream-facing Python APIs.

## API Stability

The public API is not stable yet.

The project should converge toward a small documented API before downstream projects depend on it broadly.

## Access Rule

Downstream projects must import from `sam3dbody`, not from `third_party/sam-3d-body`.

## Intended Future Shape

The future public API may include:

* model loading
* image inference
* batch inference
* structured result objects
* lightweight configuration objects

Exact names and schemas are not defined yet.
