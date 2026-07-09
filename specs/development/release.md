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
* repository `.git/` directories

Source archives must also exclude the local upstream checkout under `third_party/sam-3d-body/`. The upstream source tree is prepared explicitly through `sam3dbody install-upstream` and must not be bundled accidentally into wrapper source archives.

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

## Upstream Source Setup

Installing the wrapper from a Git URL remains wrapper-only installation. Real inference additionally requires explicit upstream source setup.

The initial supported explicit setup path is:

```bash
sam3dbody install-upstream
```

This command prepares upstream source code under the wrapper default upstream location. Source checkouts use `third_party/sam-3d-body`; installed packages use `.local/upstream/sam-3d-body` under the current working directory so setup does not mutate `site-packages` or a virtual environment. It does not make the environment inference-ready by itself because checkpoints, MHR assets, platform-specific dependencies, and CUDA availability remain separate requirements.

The selected approach specifies:

* where upstream source is stored;
* how upstream version compatibility is verified;
* how missing upstream source is reported to users;
* whether checkpoints and MHR assets are downloaded, discovered, or supplied explicitly;
* how source archives and wheels avoid accidentally depending on uninitialized Git submodules.

Source checkouts with initialized submodules remain a supported development layout for upstream integration work. Git URL installs should document `sam3dbody install-upstream` as the explicit source setup step rather than implying that packaging tools initialize submodules automatically.

A diagnostic command may report missing upstream source, missing checkpoints, missing inference dependencies, and CUDA availability. Such diagnostics are validation aids and do not download assets or install dependencies.

For CI or scripted validation, diagnostic commands may provide a strict mode that returns a non-zero exit status when required inference prerequisites are missing. Strict diagnostics are still validation aids only and must not mutate the filesystem or install upstream assets.

Human-readable environment diagnostics should summarize the missing inference prerequisites that prevent strict readiness so users can distinguish missing paths, missing Python modules, and unavailable CUDA without inspecting JSON manually.

A non-mutating upstream setup planning command may be provided before a mutating installer exists. Such a command may suggest clone and submodule commands, but it must not be documented as completing upstream setup by itself.
