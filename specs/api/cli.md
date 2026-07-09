# CLI

## Purpose

This document defines the current command line interface contract.

## Stability

The CLI is experimental unless a command is explicitly marked stable.

## Console Command

The package provides a `sam3dbody` console command after installation.

## Dependency Inspection Command

```text
sam3dbody inspect-deps [--upstream-root PATH] [--json]
```

`inspect-deps` statically inspects the upstream SAM 3D Body source tree and reports observed top-level import requirements.

The command must not import upstream modules or execute upstream inference code.

When `--upstream-root` is omitted, the command uses the source-tree upstream repository under:

```text
third_party/sam-3d-body
```

When `--json` is provided, the command prints a JSON object using the wrapper-owned dependency report schema.

Without `--json`, the command prints a human-readable summary.

## Single Image Inference Command

```text
sam3dbody infer IMAGE --weights PATH [--output PATH] [--device DEVICE] [--mhr-path PATH]
```

`infer` runs single-image inference through the public wrapper API and writes a wrapper-owned JSON result.

The command must use `Sam3DBodyModel.from_pretrained(...).load().predict(...)` rather than importing upstream modules directly.

The command requires an explicit `--weights` path. Checkpoints are external runtime inputs and must not be bundled into the wrapper package.

When `--output` is provided, the command writes JSON to that path and creates missing parent directories. When `--output` is omitted, the command prints JSON to stdout.

The command accepts:

* `--device`, defaulting to `cuda`
* `--mhr-path`, forwarded as `config["mhr_path"]`
* `--bbox-thr`, defaulting to `0.5`
* `--nms-thr`, defaulting to `0.3`
* `--inference-type`, one of `full`, `body`, or `hand`

Real upstream prediction currently requires CUDA for the same reason documented in [inference_pipeline.md](inference_pipeline.md).

The JSON output must be derived from `Sam3DBodyResult.to_dict()`. Values that are not directly JSON serializable, such as tensors or arrays, may be converted to lists when supported or represented using a deterministic string fallback.

## Future Commands

Future CLI commands may include:

* upstream setup
* checkpoint download or verification
* batch inference
* output conversion
* diagnostics

Future command behavior must be specified before being treated as stable.
