# Architecture Overview

## Purpose

This document defines the high-level architecture for the project.

## Architecture Shape

```text
+----------------------+
|  Downstream Project  |
+----------+-----------+
           |
           | Public API / CLI
           v
+----------+-----------+
|      sam3dbody       |
|  Wrapper Library     |
+----------+-----------+
           |
           | Internal adapter boundary
           v
+----------+-----------+
|      third_party     |
|     sam-3d-body      |
+----------------------+
```

## Responsibilities

The wrapper library owns downstream-facing behavior.

The upstream implementation owns model-specific internals.

Downstream projects should only use the wrapper's documented API or CLI.

## Initial Scope

The initial implementation scope is:

* package installation
* dependency organization
* wrapper-level import boundaries
* public API skeleton
* CLI skeleton
* downstream integration readiness

Detailed inference behavior is intentionally left for later specifications.
