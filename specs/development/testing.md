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

## Test Isolation

Tests must not create generated artifacts, third-party checkouts, caches, or package metadata in the repository working tree when an isolated temporary directory can exercise the same behavior.

Packaging tests that validate archive filtering should build synthetic source trees under temporary directories and pass those directories explicitly to packaging helpers.

## Test Organization

Tests should be grouped by observable responsibility.

Preferred test modules include:

* `test_public_api.py` for import behavior and public API contracts
* `test_cli.py` for command line behavior
* `test_dependencies.py` for upstream dependency inspection behavior
* `test_packaging.py` for source archive and generated artifact rules

Avoid accumulating unrelated test responsibilities in a single large test file.
