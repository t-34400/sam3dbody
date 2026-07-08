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

## Test Organization

Tests should be grouped by observable responsibility.

Preferred test modules include:

* `test_public_api.py` for import behavior and public API contracts
* `test_cli.py` for command line behavior
* `test_dependencies.py` for upstream dependency inspection behavior
* `test_packaging.py` for source archive and generated artifact rules

Avoid accumulating unrelated test responsibilities in a single large test file.
