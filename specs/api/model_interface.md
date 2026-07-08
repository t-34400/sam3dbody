# Model Interface

## Purpose

This document records the initial direction for model-facing interfaces.

## Status

Detailed model inference behavior is not specified yet.

## Initial Direction

The wrapper should provide a model interface that hides upstream implementation details from downstream projects.

The interface should make input and output coordinate conventions explicit before they are relied on by downstream projects.

The interface should avoid requiring downstream projects to understand upstream demo script internals.

## Future Topics

Future specifications should define:

* model weight discovery
* model initialization
* device selection
* image preprocessing expectations
* output schema
* coordinate systems
* batching behavior
* error handling
