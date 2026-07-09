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

## Source Archive Creation

Source archives must be created through the repository packaging script once that script exists.

The packaging script must exclude generated artifacts documented in this release policy.

Manual zip creation is discouraged because it can accidentally include local cache or build outputs.

## Git Installation and Submodules

Installing this project from a Git URL with standard Python packaging tools must not assume that Git submodules are initialized by the installer.

In particular, installation forms such as:

```bash
pip install git+https://example.com/sam3dbody.git
```

should be treated as wrapper package installation only. They must not be documented as sufficient for real upstream inference unless an explicit upstream setup step is also documented and validated.

The project must not rely on `pip install git+...` to populate `third_party/sam-3d-body/`.

## Future Upstream Setup TODO

Before documenting Git URL installation as an inference-ready workflow, define and implement an explicit upstream setup path. Candidate approaches include:

* a setup CLI such as `sam3dbody install-upstream`;
* a documented manual `git submodule update --init --recursive` workflow for source checkouts;
* a cache-based external upstream clone managed by the wrapper;
* vendoring upstream source into distributable archives if license and maintenance requirements allow it.

The selected approach must specify:

* where upstream source is stored;
* how upstream version compatibility is verified;
* how missing upstream source is reported to users;
* whether checkpoints and MHR assets are downloaded, discovered, or supplied explicitly;
* how source archives and wheels avoid accidentally depending on uninitialized Git submodules.

Until this TODO is resolved, source checkouts with initialized submodules are the only supported development layout for upstream integration work.
