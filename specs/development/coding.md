# Coding Guidelines

## Purpose

This document defines project-specific coding guidance.

## Package Layout

Use the `src/` layout.

Project-owned modules must be under `src/sam3dbody/`.

## Module Design

Prefer focused modules with explicit responsibilities.

Avoid turning `__init__.py` into a large implementation file.

## Third-Party Imports

Keep direct third-party imports near wrapper integration boundaries.

Avoid scattering direct imports from upstream internals across public API modules.
