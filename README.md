# sam3dbody

Python wrapper for SAM 3D Body.

This repository provides a small wrapper-owned API and CLI around the upstream SAM 3D Body implementation. The wrapper is designed so downstream projects import `sam3dbody` instead of importing directly from `third_party/sam-3d-body`.

## Install

For wrapper development from a source checkout:

```bash
pip install -e .
```

For Git URL installation, standard Python packaging tools install only this wrapper package. They do not initialize Git submodules and should not be treated as enough for real upstream inference.

```bash
pip install git+https://example.com/sam3dbody.git
```

## Prepare upstream source

Real inference requires the upstream SAM 3D Body source tree in addition to this wrapper. Use the explicit setup command:

```bash
sam3dbody install-upstream
```

By default this prepares:

```text
third_party/sam-3d-body
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

Checkpoints, MHR assets, CUDA, PyTorch, Detectron2, SAM3, and other upstream inference requirements are external runtime prerequisites. They are not bundled into this wrapper package.
