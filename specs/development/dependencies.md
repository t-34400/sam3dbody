# Dependency Policy

## Purpose

This document defines how Python dependencies are classified, declared, and validated.

## Dependency Classes

The project distinguishes wrapper dependencies from upstream inference dependencies.

```text
wrapper dependencies
    Dependencies required to import and use the wrapper-owned public API.

upstream inference dependencies
    Dependencies required to execute the original SAM 3D Body implementation.
```

## Wrapper Dependency Policy

The base package should keep wrapper dependencies minimal.

Base installation must support importing `sam3dbody` and using non-inference wrapper objects without installing the full upstream inference stack.

## Upstream Dependency Policy

Dependencies required only for upstream inference should not be added to the base dependency list unless they are required by wrapper code.

Upstream inference dependencies should be documented separately and may be exposed through optional extras once the installation contract is finalized.

## Upstream Import Policy

The wrapper must not import upstream modules at package import time.

Imports of upstream modules must be isolated behind adapter code so missing upstream inference dependencies do not break base wrapper imports.

## Dependency Inspection

The wrapper may provide a lightweight dependency inspection helper that reports observed upstream import requirements without importing upstream modules.

Inspection helpers must use static source inspection or other non-invasive checks.

They must not execute upstream inference code.

## Current Upstream Installation Source

The upstream repository currently documents its inference setup in:

```text
third_party/sam-3d-body/INSTALL.md
```

This file is the observed source for upstream dependency investigation until a wrapper-owned optional dependency contract is specified.
