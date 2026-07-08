# Dependency Rules

## Purpose

This document defines allowed dependency directions.

## Allowed Dependency Direction

```text
Downstream projects
    ->
Public API / CLI
    ->
Internal wrapper modules
    ->
third_party/sam-3d-body
```

## Prohibited Dependency Direction

Third-party code must not depend on wrapper modules.

Downstream projects must not import from `third_party/sam-3d-body` directly.

## Boundary Rule

Imports from third-party modules should be concentrated in adapter or integration modules owned by the wrapper.

Public API modules should avoid exposing raw third-party implementation details unless a later specification explicitly allows it.
