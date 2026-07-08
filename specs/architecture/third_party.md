# Third-Party Code

## Purpose

This document defines how upstream code is managed.

## Source

The upstream SAM 3D Body implementation is managed under:

```text
third_party/sam-3d-body/
```

## Modification Policy

Local modifications to third-party source files should be avoided when practical.

If a third-party modification is unavoidable:

1. Keep the change isolated.
2. Document the reason.
3. Prefer a wrapper-side solution first.
4. Prefer contributing generally useful fixes upstream.

## License Policy

Upstream license files and attribution notices must be preserved.

Wrapper repository license and notice files must not imply that upstream license requirements are removed.

## Update Policy

Upstream updates should be reviewed before adoption.

Wrapper compatibility must be validated after updating the third-party implementation.
