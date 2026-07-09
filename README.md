# sam3dbody

Python wrapper for SAM 3D Body.

This package provides a wrapper-owned Python API and CLI around the upstream SAM 3D Body implementation. Downstream projects should import `sam3dbody` instead of importing directly from an upstream checkout.

## Installation model

There are three separate layers:

1. **Wrapper package**: this repository and its lightweight CLI/API.
2. **Upstream source**: the original SAM 3D Body repository, prepared explicitly with `sam3dbody install-upstream`.
3. **Real inference environment**: CUDA, Torch, timm, PyTorch Lightning, Detectron2, SAM3, checkpoints, and optional assets.

The base package intentionally does not install Torch, torchvision, timm, PyTorch Lightning, Detectron2, SAM3, checkpoints, or MHR assets. Those pieces are environment-specific and should be installed explicitly for the target machine.

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

The base install includes ordinary PyPI dependencies used by the wrapper diagnostics and the documented setup flow, such as OpenCV, scikit-image, pandas, rich, Hydra, Hugging Face Hub, braceexpand, roma, and termcolor. It deliberately excludes dependencies that can pull Torch transitively, especially `timm` and `pytorch-lightning`.

## Prepare upstream source

Real inference requires the upstream SAM 3D Body source tree in addition to this wrapper.

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

Source archives produced by `scripts/package_source.py` intentionally exclude `third_party/sam-3d-body/` and Git metadata. Recreate upstream source locally with `sam3dbody install-upstream` after installing or unpacking the wrapper.

## Check the environment

Before installing real inference dependencies:

```bash
sam3dbody check-env
```

After a base install and upstream setup, it is normal for `check-env` to still report missing requirements such as:

```text
weights path was not provided
missing importable modules: torch, timm, pytorch_lightning
CUDA is not available through torch
```

That means the wrapper and upstream source are present, but the real inference environment is not complete yet.

Use strict mode in scripts when missing inference prerequisites should fail the command:

```bash
sam3dbody check-env --weights /path/to/checkpoint.ckpt --strict
```

## Install real inference dependencies

Install Torch explicitly for your CUDA / driver / platform combination. For example, choose the correct command from the official PyTorch selector before continuing.

After Torch is installed, install remaining real-inference packages that depend on it, such as `timm` and `pytorch-lightning`, plus upstream-specific packages such as Detectron2 and SAM3 following the upstream instructions.

A minimal next step after selecting Torch is typically:

```bash
uv pip install timm pytorch-lightning
```

The base wrapper install already includes ordinary upstream-import dependencies observed during real smoke testing, including `braceexpand`, `roma`, and `termcolor`. Then re-check:

```bash
sam3dbody check-env --weights /path/to/checkpoint.ckpt
```

Model checkpoints and MHR assets are not bundled into this wrapper. Provide them explicitly or obtain them through the upstream-authorized distribution path.

## Python API

```python
from sam3dbody import Sam3DBodyModel

model = Sam3DBodyModel.from_pretrained(
    weights_path="/path/to/checkpoint.ckpt",
    device="cuda",
)
session = model.load()
result = session.predict("/path/to/image.png")
print(result.to_dict())
```

Repeated inference should use a loaded session so weights and the upstream estimator are reused:

```python
results = session.predict_many([
    "/path/to/frame_000.png",
    "/path/to/frame_001.png",
])
```

`predict_many()` currently performs ordered repeated single-image inference. It is not yet optimized tensor batching.

## CLI inference

```bash
sam3dbody infer image.png \
  --weights /path/to/checkpoint.ckpt \
  --output result.json
```

## Real inference smoke test

After preparing upstream source, dependencies, CUDA, weights, and a sample image, run an explicit smoke test:

```bash
sam3dbody smoke-test image.png \
  --weights /path/to/checkpoint.ckpt \
  --repeat 2 \
  --output smoke-report.json
```

The smoke report records environment readiness, body counts, output keys, and shape/dtype summaries. It avoids embedding full tensors or arrays.

The pytest real-inference smoke test is skipped by default. To run it explicitly, provide real paths:

```bash
SAM3DBODY_RUN_REAL_SMOKE=1 \
SAM3DBODY_SMOKE_IMAGE=/path/to/image.png \
SAM3DBODY_SMOKE_WEIGHTS=/path/to/checkpoint.ckpt \
PYTHONPATH=src:. pytest -q tests/test_real_inference_smoke.py
```
