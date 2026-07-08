# COMMON_RULES.md

## Purpose

This document defines project-wide principles, specification authority, architecture expectations, complexity management rules, and change management requirements.

Project-specific specifications must follow these rules.

---

# Specification Authority

When information conflicts, use the following priority:

```text
COMMON_RULES.md
    >
Other specifications
    >
Tests
    >
Implementation
```

Implementation behavior must not override written specifications.

Tests validate specifications but do not replace them.

---

# Read Before Modify

Before making changes:

1. Identify affected behavior.
2. Identify relevant specifications.
3. Identify affected tests.
4. Identify compatibility implications.

Do not begin implementation until the relevant specifications have been reviewed.

---

# Single Source of Truth

Each rule should have exactly one authoritative location.

Avoid duplicating behavioral requirements across documents.

INDEX documents are navigation documents.

INDEX documents must not define behavioral requirements.

---

# Specification-Driven Development

Implementation follows specifications.

When intentionally changing behavior:

1. Update specifications.
2. Update tests.
3. Update implementation.

Keep these changes in the same change set whenever practical.

---

# Explicit Behavior

Important behavior must be documented.

Do not rely on:

* developer assumptions
* undocumented conventions
* historical implementation quirks
* prior discussions

If behavior matters, specify it.

---

# Traceability

Important requirements should remain traceable.

Preferred chain:

```text
Requirement
    ->
Specification
    ->
Test
    ->
Implementation
```

Changes should preserve this relationship whenever practical.

---

# Specification Organization

Specifications must remain maintainable.

Prefer:

* small focused documents
* one responsibility per document
* explicit references between documents
* topic-oriented organization

Avoid:

* monolithic specifications
* duplicated requirements
* mixed responsibilities
* large unstructured documents

When a specification becomes difficult to navigate, split it into smaller specifications.

---

# Code Organization

Source code must remain maintainable.

Prefer:

* small focused modules
* explicit interfaces
* clear ownership boundaries
* responsibility-oriented structure

Avoid:

* god objects
* multi-purpose modules
* hidden coupling
* excessive file growth

Guidelines:

* keep source files near 300 lines or less when practical
* split files before 500 lines unless strongly justified
* keep functions near 50 lines or less when practical

These are guidelines, not hard limits.

---

# Complexity Management

When adding functionality:

1. Extend an existing specification if the responsibility matches.
2. Create a new specification if a new responsibility is introduced.
3. Avoid expanding unrelated specifications.

When adding code:

1. Extend an existing module if the responsibility matches.
2. Create a new module if a new responsibility is introduced.
3. Avoid accumulating unrelated responsibilities in a single file.

Prefer introducing new focused components over growing existing multi-purpose components.

---

# Architecture Principles

Unless a more specific specification states otherwise:

* prefer modular design
* prefer explicit interfaces
* prefer deterministic behavior
* prefer testable components
* separate responsibilities clearly
* avoid hidden coupling
* avoid unnecessary complexity

---

# Testing Principles

Tests verify observable behavior.

Tests should focus on:

* expected outputs
* public interfaces
* documented requirements
* regression protection

Avoid testing implementation details unless necessary.

Tests are validation artifacts, not primary specifications.

---

# Documentation Principles

Documentation should describe:

* behavior
* interfaces
* assumptions
* constraints

Avoid duplicating information maintained elsewhere.

Keep documentation synchronized with behavior changes.

---

# Compatibility Policy

The project must explicitly define compatibility requirements.

Examples:

* strict compatibility
* behavioral compatibility
* API compatibility
* migration compatibility
* breaking changes allowed

Project-specific compatibility requirements must be documented in project specifications.

---

# Decision Records

Long-term architectural decisions should be documented separately.

Decision records should explain:

* context
* decision
* consequences

Avoid embedding architectural rationale directly into behavioral specifications.

---

# Project-Specific Rules

## Project Purpose

This project provides a reusable Python package for SAM 3D Body.

The project wraps the original SAM 3D Body implementation and exposes stable interfaces suitable for downstream applications, diagnostics, and integration projects.

The initial goal is not to redesign the upstream model implementation. The initial goal is to make it installable, usable, and maintainable as a Python library.

---

## Domain Concepts

Original implementation
: The upstream SAM 3D Body repository managed under `third_party/sam-3d-body`.

Wrapper
: Code implemented by this project under `src/sam3dbody`.

Public API
: Stable Python interfaces intended for downstream projects.

CLI
: Command line entry points provided by this project.

Third-party code
: External code imported into this repository through `third_party`.

Downstream project
: Any project that uses this package through the public API or CLI.

---

## Architecture Constraints

The original implementation must be kept under `third_party/`.

Wrapper code must live under `src/sam3dbody/`.

Downstream projects must not depend on `third_party/sam-3d-body` directly.

New project-specific behavior should be implemented in the wrapper whenever practical.

Local modifications to third-party source files should be minimized and documented.

The wrapper is responsible for:

* dependency management
* public API design
* CLI design
* configuration boundaries
* adapters for downstream projects
* compatibility behavior

---

## Coding Rules

Use the `src/` package layout.

Keep third-party implementation details behind project-owned modules.

Prefer typed, explicit interfaces for public entry points.

Avoid importing from deeply nested third-party modules in downstream-facing code unless no stable alternative exists.

---

## Compatibility Requirements

The project should prefer stable public APIs over exposing upstream internals.

Before the first stable release, breaking changes are allowed when they simplify the architecture or API.

After a public API is documented as stable, compatibility requirements must be documented before changing it.

---

## Testing Requirements

Tests should prioritize wrapper behavior and public interfaces.

Tests should not require large model weights by default.

Model-dependent tests must be isolated from lightweight package tests.

---

## Performance Requirements

No project-wide performance target is defined yet.

Performance-sensitive behavior must be specified before optimization work begins.

---

## Safety Requirements

The package must preserve upstream license notices and attribution requirements.

Generated outputs, downloaded weights, and user-provided data must not be committed unless explicitly allowed by a specification.

---

## Data Rules

Large model files must not be stored in the repository by default.

Paths to model weights, input data, and output directories should be explicit.

Sample data policy is not defined yet and must be specified before adding sample assets.
