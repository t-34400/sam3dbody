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

The base package should keep wrapper dependencies modest but practical for the documented inference setup workflow.

Base installation must support importing `sam3dbody`, using non-inference wrapper objects, running CLI diagnostics, and satisfying the PyPI-installable representative runtime modules checked by `check-env`, except for platform-specific packages.

The following representative PyPI packages are base runtime dependencies because they are ordinary packaging-tool-installable prerequisites reported by the wrapper environment check that do not pull PyTorch transitively:

```text
opencv-python
yacs
scikit-image
einops
pandas
rich
hydra-core
huggingface_hub
braceexpand
roma
termcolor
```

These base dependencies do not make the environment inference-ready by themselves. They only remove ordinary PyPI dependency setup from the manual post-install checklist when those dependencies do not pull PyTorch transitively. Packages such as `timm` and `pytorch-lightning` are real-inference prerequisites but are excluded from base dependencies because they depend on PyTorch and can cause packaging tools to select an environment-inappropriate Torch build.

## Upstream Dependency Policy

Platform-specific, non-PyPI, pinned Git, optional upstream, or asset-like dependencies required only for real upstream inference should not be added to the base dependency list unless wrapper-owned runtime code requires them.

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

## Optional Dependency Contract

The base project dependency list should remain modest and must not include platform-specific, non-PyPI, pinned Git, optional upstream, or asset-like dependencies from the upstream inference stack.

The wrapper may declare optional dependency groups in `pyproject.toml` for documented integration stages.

### `inference` extra

The `inference` extra contains additional PyPI-installable packages observed in the upstream installation guide that may be needed before real SAM 3D Body inference integration can be attempted. Some representative packages may also appear in the base dependency list when they are part of the documented wrapper setup experience.

The `inference` extra is allowed to be incomplete with respect to non-PyPI or platform-specific requirements. In particular:

* PyTorch installation remains user/environment specific and should follow the official PyTorch installation selector.
* `timm` remains outside the base dependencies because it depends on PyTorch and may cause package managers to install an unintended Torch build. Install it as part of the explicit real-inference environment after selecting a Torch build.
* `pytorch-lightning` remains outside the base dependencies because it depends on PyTorch and should be installed after the user selects a Torch build.
* `braceexpand`, `roma`, and `termcolor` are ordinary PyPI packages observed during real upstream model import and are included in the base dependency list.
* Detectron2 remains outside the `inference` extra because upstream currently requires a pinned Git installation with custom installer flags.
* MoGe remains outside the `inference` extra because upstream marks it optional.
* SAM3 remains outside the `inference` extra because upstream requires cloning and editable installation of a separate repository.
* Model checkpoints and MHR assets are never packaged in this wrapper and must be supplied explicitly or downloaded through authenticated Hugging Face access.

The `inference` extra is a wrapper-owned installation convenience, not a guarantee that model inference will run without separately installed platform-specific dependencies and checkpoint access.

### Observed optional upstream packages

Static inspection of the current upstream source shows that `wandb` is imported only inside logging helper methods under the upstream Lightning base module. It is not part of the wrapper base dependency set.

`seaborn` is listed in the upstream installation guide but is not imported by the current upstream Python source tree. It is not part of the wrapper base dependency set.

These packages may be added to a future development, training, logging, or visualization extra if wrapper-owned behavior requires them.

## Dependency Declaration Requirements

Base dependencies in `pyproject.toml` must include only wrapper-owned runtime requirements and the documented ordinary PyPI inference prerequisites listed in this specification. They must not include platform-specific or non-PyPI upstream requirements.

Optional dependencies must be tested as declared metadata. Lightweight tests must not install or import the heavy upstream inference stack.

Dependency tests should verify that:

* base dependencies include the documented ordinary PyPI prerequisites that do not pull PyTorch transitively;
* the `inference` extra is declared;
* platform-specific and non-PyPI upstream-only packages do not become base requirements;
* static dependency inspection remains non-invasive.

