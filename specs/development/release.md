# Release Policy

## Purpose

This document defines the initial release policy.

## Status

No stable release policy is defined yet.

## Initial Versioning

Before the first stable release, breaking changes are allowed when documented.

## Distribution Goal

The project should become installable with standard Python packaging tools.

Supported installation commands should be documented before release.

## Source Archive Cleanliness

Project source archives must not include local cache, build, or packaging artifacts.

Excluded artifacts include at least:

* `__pycache__/`
* `.pytest_cache/`
* `*.egg-info/`
* `build/`
* `dist/`

Generated artifacts should be recreated by local tooling rather than stored in source archives.

