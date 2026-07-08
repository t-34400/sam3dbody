# Testing Guidelines

## Purpose

This document defines the initial testing policy.

## Lightweight Tests

Default tests should avoid requiring model weights, GPU access, or large datasets.

## Model-Dependent Tests

Tests that require model weights or GPU execution must be isolated from lightweight tests.

## Priority

Initial tests should verify:

* package import behavior
* public API surface
* CLI availability if a CLI is provided
* wrapper boundary behavior
