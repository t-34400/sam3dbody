# sam3dbody

Python wrapper for SAM 3D Body.

`sam3dbody` provides a wrapper-owned Python API and CLI around the upstream SAM 3D Body implementation. The wrapper keeps the upstream source separate and unmodified.

## Quick Start

```bash
uv venv --python 3.10 .venv
source .venv/bin/activate

uv pip install git+https://github.com/t-34400/sam3dbody.git

sam3dbody install-upstream

# Download the official model files after your access request has been approved:
# https://huggingface.co/facebook/sam-3d-body-dinov3/tree/main
#
# Required files:
# - model.ckpt
# - assets/mhr_model.pt

# Install the appropriate Torch packages for your platform.
uv pip install torch torchvision --index-url <TORCH_INDEX_URL>
uv pip install timm pytorch-lightning

sam3dbody check-env

sam3dbody smoke-test image.png \
    --weights /path/to/model.ckpt \
    --mhr-path /path/to/assets/mhr_model.pt
```

## CLI commands

| Command | Purpose |
| --- | --- |
| `check-env` | Diagnose the installation and runtime environment |
| `plan-upstream-setup` | Show where the upstream source will be installed |
| `install-upstream` | Clone and prepare the upstream source |
| `infer` | Run inference |
| `smoke-test` | Validate the complete installation |

## Python API

```python
from sam3dbody import Sam3DBodyModel

model = Sam3DBodyModel(...)
result = model.predict(image)
```

## License and upstream terms

This repository contains the wrapper only.

The upstream SAM 3D Body source code and official model assets are distributed separately by Meta and remain subject to their respective license terms.

- Upstream: https://github.com/facebookresearch/sam-3d-body
- Models: https://huggingface.co/facebook/sam-3d-body-dinov3/tree/main
