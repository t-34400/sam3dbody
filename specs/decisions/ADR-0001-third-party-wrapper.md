# ADR-0001: Use a Wrapper Repository Around SAM 3D Body

## Status

Accepted.

## Context

The project needs SAM 3D Body to be usable as an installable Python package.

The upstream repository is primarily model and demo oriented. Downstream projects need a stable package boundary, API, and CLI rather than direct dependence on upstream demo internals.

## Decision

Use this repository as a wrapper package.

Keep the upstream SAM 3D Body implementation under `third_party/sam-3d-body`.

Implement package management, public API, CLI, adapters, and downstream integration behavior in `src/sam3dbody`.

## Consequences

Downstream projects can depend on this wrapper instead of upstream internals.

The upstream implementation can be updated separately from wrapper APIs.

The wrapper must preserve upstream license requirements and attribution.

Some integration work may require adapter modules that translate between wrapper-owned interfaces and upstream implementation details.
