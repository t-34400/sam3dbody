# sam3dbody

Python wrapper for SAM 3D Body.

`sam3dbody` provides a wrapper-owned Python API and CLI around the upstream SAM 3D Body implementation. Downstream projects should depend on this package instead of importing directly from an upstream checkout.

The core project policy is:

- keep the upstream SAM 3D Body source unmodified;
- keep wrapper code and upstream code separated;
- install the wrapper with standard Python packaging tools;
- prepare upstream source explicitly after wrapper installation;
- install Torch-family dependencies explicitly for the target CUDA environment.

## Quick Start

```bash
uv venv --python 3.10 .venv
source .venv/bin/activate
uv pip install git+https://github.com/t-34400/sam3dbody.git

sam3dbody check-env
sam3dbody install-upstream

# Choose the Torch command that matches your CUDA / driver / platform first.
# Example only; use the official PyTorch selector for the real command.
uv pip install torch torchvision --index-url <TORCH_INDEX_URL>
uv pip install timm pytorch-lightning

sam3dbody check-env \
  --weights .local/models/sam-3d-body-dinov3/model.ckpt \
  --mhr-path .local/models/sam-3d-body-dinov3/assets/mhr_model.pt

sam3dbody smoke-test /path/to/image.png \
  --weights .local/models/sam-3d-body-dinov3/model.ckpt \
  --mhr-path .local/models/sam-3d-body-dinov3/assets/mhr_model.pt \
  --repeat 3 \
  --output smoke-report.json
```

A successful real smoke test verifies wrapper installation, upstream source availability, dependency readiness, CUDA availability, checkpoint access, single-image inference, repeated `predict_many()` inference, and JSON report generation.

## Installation model

There are three separate layers:

1. **Wrapper package**: this repository and its lightweight CLI/API.
2. **Upstream source**: the original SAM 3D Body repository, prepared explicitly with `sam3dbody install-upstream`.
3. **Real inference environment**: CUDA, Torch, torchvision, timm, PyTorch Lightning, checkpoints, MHR assets, and any upstream-specific packages required by the selected setup.

The base package intentionally does not install Torch, torchvision, timm, PyTorch Lightning, Detectron2, SAM3, checkpoints, or MHR assets. These pieces are environment-specific and should be installed explicitly for the target machine.

## Install the wrapper

From GitHub:

```bash
uv venv --python 3.10 .venv
source .venv/bin/activate
uv pip install git+https://github.com/t-34400/sam3dbody.git
```

From a source checkout for development:

```bash
uv venv --python 3.10 .venv
source .venv/bin/activate
uv pip install -e .
```

The base install includes ordinary PyPI dependencies used by wrapper diagnostics and the documented setup flow, including OpenCV, scikit-image, pandas, rich, Hydra, Hugging Face Hub, braceexpand, roma, and termcolor. It deliberately excludes dependencies that can pull Torch transitively, especially `timm` and `pytorch-lightning`.

## Prepare upstream source

Real inference requires the upstream SAM 3D Body source tree in addition to this wrapper. The wrapper does not rely on Git submodule initialization during `pip install`; `install-upstream` is the explicit setup path.

```bash
sam3dbody plan-upstream-setup
sam3dbody install-upstream
```

Default target behavior:

- source checkout: `third_party/sam-3d-body`
- Git URL / wheel install: `.local/upstream/sam-3d-body` under the current working directory

You can choose an explicit target:

```bash
sam3dbody install-upstream --target .local/upstream/sam-3d-body
```

`install-upstream` prepares source code only. It does not download checkpoints, download MHR assets, install Python dependencies, import upstream modules, or run inference.

Source archives produced by `scripts/package_source.py` intentionally exclude `third_party/sam-3d-body/` and Git metadata. Recreate upstream source locally with `sam3dbody install-upstream` after installing or unpacking the wrapper.

## Download official model files

Real inference requires the official SAM 3D Body DINOv3 checkpoint and MHR asset. Download them from the official gated Hugging Face repository after your access request has been approved:

```text
https://huggingface.co/facebook/sam-3d-body-dinov3/tree/main
```

Use the upstream file layout expected by the wrapper examples:

```text
.local/models/
└── sam-3d-body-dinov3/
    ├── model.ckpt
    └── assets/
        └── mhr_model.pt
```

The corresponding command paths are:

```text
--weights .local/models/sam-3d-body-dinov3/model.ckpt
--mhr-path .local/models/sam-3d-body-dinov3/assets/mhr_model.pt
```

The wrapper does not bundle, mirror, or automatically download these files.

## Check the environment

Run the diagnostic before and after preparing the real inference environment:

```bash
sam3dbody check-env
```

After a base install and upstream setup, it is normal for `check-env` to still report missing requirements such as:

```text
weights path was not provided
mhr path was not provided
missing importable modules: torch, timm, pytorch_lightning
CUDA is not available through torch
```

That means the wrapper and upstream source may be present, but the real inference environment is not complete yet.

Use strict mode in scripts when missing inference prerequisites should fail the command:

```bash
sam3dbody check-env \
  --weights .local/models/sam-3d-body-dinov3/model.ckpt \
  --mhr-path .local/models/sam-3d-body-dinov3/assets/mhr_model.pt \
  --strict
```

## Install real inference dependencies

Install Torch explicitly for your CUDA / driver / platform combination. Choose the correct command from the official PyTorch selector before continuing.

After Torch is installed, install remaining real-inference packages that depend on it:

```bash
uv pip install timm pytorch-lightning
```

The base wrapper install already includes ordinary upstream-import dependencies observed during real smoke testing, including `braceexpand`, `roma`, and `termcolor`.

Then re-check with real model paths:

```bash
sam3dbody check-env \
  --weights .local/models/sam-3d-body-dinov3/model.ckpt \
  --mhr-path .local/models/sam-3d-body-dinov3/assets/mhr_model.pt
```

Model checkpoints and MHR assets are not bundled into this wrapper. Provide them explicitly after downloading the official files through approved Hugging Face access.

## Python API

```python
from sam3dbody import Sam3DBodyModel

model = Sam3DBodyModel.from_pretrained(
    weights_path=".local/models/sam-3d-body-dinov3/model.ckpt",
    device="cuda",
    config={"mhr_path": ".local/models/sam-3d-body-dinov3/assets/mhr_model.pt"},
)
session = model.load()
result = session.predict("/path/to/image.png")
print(result.to_dict())
```

Repeated inference should use a loaded session so checkpoint weights and the upstream estimator are reused:

```python
results = session.predict_many([
    "/path/to/frame_000.png",
    "/path/to/frame_001.png",
])
```

`predict_many()` currently performs ordered repeated single-image inference. It is not tensor-batched inference.

## CLI inference

```bash
sam3dbody infer image.png \
  --weights .local/models/sam-3d-body-dinov3/model.ckpt \
  --mhr-path .local/models/sam-3d-body-dinov3/assets/mhr_model.pt \
  --output result.json
```

`infer` runs one image through the public wrapper API and writes a wrapper-owned JSON result. Public callers may also provide bounding boxes, masks, and camera intrinsics through the documented JSON options.

## Real inference smoke test

After preparing upstream source, dependencies, CUDA, weights, MHR assets, and a sample image, run an explicit smoke test:

```bash
sam3dbody smoke-test image.png \
  --weights .local/models/sam-3d-body-dinov3/model.ckpt \
  --mhr-path .local/models/sam-3d-body-dinov3/assets/mhr_model.pt \
  --repeat 3 \
  --output smoke-report.json
```

The smoke report records environment readiness, body counts, output keys, and shape/dtype summaries. It avoids embedding full tensors or arrays. When `--output` is used, the CLI still prints the report path and prints a concise failure summary to stderr if the smoke test fails.

A validated real smoke test with the SAM 3D Body DINOv3 checkpoint and MHR asset produced one body for both single-image inference and `--repeat 3`. The observed output summaries were:

```text
vertices: float32 [18439, 3]
joints:   float32 [70, 3]
faces:    int64   [36874, 3]
```

These shapes are useful sanity checks, but vertex and joint coordinates are still labeled as upstream model coordinates until a dedicated coordinate convention validation is completed.

The pytest real-inference smoke test is skipped by default. To run it explicitly, provide real paths:

```bash
SAM3DBODY_RUN_REAL_SMOKE=1 \
SAM3DBODY_SMOKE_IMAGE=/path/to/image.png \
SAM3DBODY_SMOKE_WEIGHTS=.local/models/sam-3d-body-dinov3/model.ckpt \
SAM3DBODY_SMOKE_MHR_PATH=.local/models/sam-3d-body-dinov3/assets/mhr_model.pt \
PYTHONPATH=src:. pytest -q tests/test_real_inference_smoke.py
```

## Development notes

- The upstream source tree is treated as external code and should remain unmodified when practical.
- Wrapper imports are lazy with respect to upstream inference modules, so missing upstream dependencies should not break `import sam3dbody`.
- Base dependency checks are diagnostic; `check-env` does not clone repositories, install packages, download checkpoints, import upstream modules, or run inference.
- `smoke-test` is the integration validation command for real upstream inference environments.

## License and upstream terms

This wrapper is intended to be used with upstream SAM 3D Body materials. The upstream SAM 3D Body code, checkpoints, MHR assets, and related materials are distributed separately by Meta and are subject to the SAM License and the upstream access terms.

This repository includes a copy of the SAM License in [`LICENSE`](LICENSE) so users can review the terms that apply to SAM 3D Body materials. Before downloading or using the official checkpoint and MHR asset, also review the terms presented by the upstream repository and the gated Hugging Face model page:

- Upstream source: `https://github.com/facebookresearch/sam-3d-body`
- Official model files: `https://huggingface.co/facebook/sam-3d-body-dinov3/tree/main`

`sam3dbody install-upstream` prepares source code only. It does not grant model access, download checkpoints, download MHR assets, or change the license obligations for upstream materials.
