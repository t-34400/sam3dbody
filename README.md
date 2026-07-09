# sam3dbody

Python wrapper for SAM 3D Body.

This repository provides a small wrapper-owned API and CLI around the upstream SAM 3D Body implementation. The wrapper is designed so downstream projects import `sam3dbody` instead of importing directly from `third_party/sam-3d-body`.

## Install

For wrapper development from a source checkout:

```bash
pip install -e .
```

For Git URL installation, standard Python packaging tools install this wrapper package plus ordinary PyPI runtime prerequisites such as OpenCV, scikit-image, timm, pandas, rich, Hydra, and Hugging Face Hub. They do not initialize Git submodules and should not be treated as enough for real upstream inference.

```bash
pip install git+https://example.com/sam3dbody.git
```

PyTorch, Detectron2, SAM3, checkpoints, MHR assets, CUDA drivers, and authenticated model access remain environment-specific and must be prepared separately.

## Prepare upstream source

Real inference requires the upstream SAM 3D Body source tree in addition to this wrapper. Use the explicit setup command:

```bash
sam3dbody install-upstream
```

By default this prepares `third_party/sam-3d-body` in a source checkout. When the wrapper is installed from a wheel or Git URL, the default is `.local/upstream/sam-3d-body` under the current working directory, not inside the virtual environment. You can always choose an explicit target:

```bash
sam3dbody install-upstream --target .local/upstream/sam-3d-body
```

To inspect what would be done without mutating the filesystem:

```bash
sam3dbody plan-upstream-setup
```

## Check the environment

```bash
sam3dbody check-env --weights /path/to/checkpoint.ckpt --mhr-path /path/to/mhr.pt
```

Use strict mode in scripts when missing inference prerequisites should fail the command:

```bash
sam3dbody check-env --weights /path/to/checkpoint.ckpt --strict
```

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

## CLI inference

```bash
sam3dbody infer image.png \
  --weights /path/to/checkpoint.ckpt \
  --output result.json
```

Checkpoints, MHR assets, CUDA, PyTorch, Detectron2, SAM3, and other platform-specific or non-PyPI upstream inference requirements are external runtime prerequisites. They are not bundled into this wrapper package.

Source archives produced by `scripts/package_source.py` intentionally exclude `third_party/sam-3d-body/` and Git metadata. Recreate upstream source locally with `sam3dbody install-upstream` after installing or unpacking the wrapper.

Repeated inference should use a loaded session so weights and the upstream estimator are reused:

```python
results = session.predict_many([
    "/path/to/frame_000.png",
    "/path/to/frame_001.png",
])
```

`predict_many()` currently performs ordered repeated single-image inference. It is not yet optimized tensor batching.

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
