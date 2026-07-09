# sam3dbody

Installable Python package for the upstream SAM 3D Body project.

`sam3dbody` provides a stable Python API and CLI for the official Meta SAM 3D Body implementation (https://github.com/facebookresearch/sam-3d-body). The wrapper keeps the upstream source separate and unmodified, allowing applications to depend on `sam3dbody` instead of an upstream checkout.

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

# Verify that the runtime environment is ready.
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

Create a model handle with `Sam3DBodyModel.from_pretrained()`.

```python
from sam3dbody import Sam3DBodyModel

model = Sam3DBodyModel.from_pretrained(
    weights_path="/path/to/model.ckpt",
    mhr_path="/path/to/assets/mhr_model.pt",
    device="cuda",
)
```

For one-off inference, use `model.predict()`.

```python
result = model.predict("image.png")
print(result.to_dict())
```

For repeated inference, create a reusable session once with `model.load()`. This avoids reloading the upstream model for every image.

```python
session = model.load()

result = session.predict("image.png")
results = session.predict_many(["image1.png", "image2.png"])
```

`predict_many()` performs ordered repeated single-image inference. It is **not** optimized tensor batching and returns one `Sam3DBodyResult` per input image in the same order.

The wrapper exposes the following public result types:

- `Sam3DBodyResult`
- `Sam3DBodyPrediction`
- `Sam3DBodyMetadata`

`Sam3DBodyResult.to_dict()` returns a serializable wrapper result containing `bodies` and `metadata`.

## License and upstream terms

The wrapper code in this repository is dedicated to the public domain under CC0 1.0.

The upstream SAM 3D Body source code and official model assets are distributed separately by Meta and remain subject to their own license terms.

See `licenses/SAM-LICENSE.txt` for the upstream license notice.

- Upstream: https://github.com/facebookresearch/sam-3d-body
- Models: https://huggingface.co/facebook/sam-3d-body-dinov3/tree/main
