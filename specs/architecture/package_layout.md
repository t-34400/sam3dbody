# Package Layout

## Purpose

This document defines the intended repository layout.

## Required Layout

```text
sam3dbody/
├── pyproject.toml
├── README.md
├── specs/
├── scripts/
├── src/
│   └── sam3dbody/
└── third_party/
    └── sam-3d-body/
```

## Wrapper Package

Project-owned Python code must live under:

```text
src/sam3dbody/
```

Wrapper-owned upstream integration modules must live under:

```text
src/sam3dbody/adapters/
```

## Third-Party Code

The upstream SAM 3D Body implementation must live under:

```text
third_party/sam-3d-body/
```

## Future Layout

The project includes:

```text
tests/
scripts/
```

The `scripts/` directory contains repository maintenance tools.

The project may later add:

```text
docs/
examples/
```

These directories should be added only when their responsibility is clear.
