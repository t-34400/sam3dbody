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


## Environment Check Command

```text
sam3dbody check-env [--upstream-root PATH] [--weights PATH] [--mhr-path PATH] [--json] [--strict]
```

`check-env` reports local prerequisites for real upstream inference without importing upstream modules, loading checkpoints, or running inference.

The command checks at least:

* whether the upstream source tree exists;
* whether the upstream `sam_3d_body` package directory exists;
* whether an explicitly supplied `--weights` path exists;
* whether an explicitly supplied `--mhr-path` exists;
* whether representative inference dependency modules are importable;
* whether CUDA appears available through torch when torch is importable.

When `--upstream-root` is omitted, the command uses the source-tree upstream repository under:

```text
third_party/sam-3d-body
```

When `--json` is provided, the command prints a JSON object using the wrapper-owned environment report schema. Without `--json`, the command prints a human-readable summary. The human-readable summary must include a `missing_requirements` section that lists the readiness blockers used to decide `ready_for_inference`, or `none` when no blockers are present.

By default, `check-env` exits with status code `0` after reporting diagnostics, even when the environment is not inference-ready. When `--strict` is provided, the command exits with status code `0` only if the report is ready for inference, and exits with status code `1` otherwise.

`check-env` is diagnostic only. It must not attempt to initialize Git submodules, clone upstream repositories, download checkpoints, import upstream modules, or mutate the filesystem.

## Upstream Setup Planning Command

```text
sam3dbody plan-upstream-setup [--target PATH] [--source-url URL] [--revision REV] [--no-recursive] [--json]
```

`plan-upstream-setup` prints a non-mutating plan for preparing the upstream SAM 3D Body source tree.

The command must not clone repositories, initialize Git submodules, download checkpoints, install dependencies, import upstream modules, or otherwise mutate the filesystem. It is intended to make the future setup workflow explicit before an installing command is implemented.

When `--target` is omitted, the command uses the development-layout upstream location:

```text
third_party/sam-3d-body
```

The command reports whether the target exists, whether the upstream `sam_3d_body` package directory exists, and a suggested command list. When the upstream package directory already exists, the command reports status `ready`. When the target is missing, it reports status `missing`. When the target exists but does not contain the upstream package directory, it reports status `incomplete`.

When `--json` is provided, the command prints a JSON object using the wrapper-owned upstream setup plan schema. Without `--json`, the command prints a human-readable summary.

`plan-upstream-setup` always exits with status code `0` because it is advisory and non-mutating. Readiness enforcement remains the responsibility of `check-env --strict`.

## Single Image Inference Command

```text
sam3dbody infer IMAGE --weights PATH [--output PATH] [--device DEVICE] [--mhr-path PATH] [--bboxes-json PATH] [--masks-json PATH] [--cam-int-json PATH]
```

`infer` runs single-image inference through the public wrapper API and writes a wrapper-owned JSON result.

The command must use `Sam3DBodyModel.from_pretrained(...).load().predict(...)` rather than importing upstream modules directly.

The command requires an explicit `--weights` path. Checkpoints are external runtime inputs and must not be bundled into the wrapper package.

When `--output` is provided, the command writes JSON to that path and creates missing parent directories. When `--output` is omitted, the command prints JSON to stdout.

The command accepts:

* `--device`, defaulting to `cuda`
* `--mhr-path`, forwarded as `config["mhr_path"]`
* `--bboxes-json`, a JSON file containing either `[[x1, y1, x2, y2], ...]` or `{"bboxes": [[...], ...]}`
* `--masks-json`, a JSON file containing either mask data shaped like `N x H x W` or `{"masks": ...}`
* `--cam-int-json`, a JSON file containing either a `3 x 3` camera intrinsics matrix or `{"cam_int": [[...], ...]}`
* `--bbox-thr`, defaulting to `0.5`
* `--nms-thr`, defaulting to `0.3`
* `--inference-type`, one of `full`, `body`, or `hand`

Real upstream prediction currently requires CUDA for the same reason documented in [inference_pipeline.md](inference_pipeline.md).

`--bboxes-json` is forwarded to `session.predict(..., bboxes=...)` after JSON parsing. The wrapper does not reinterpret coordinate systems at the CLI boundary; values must already follow the public prediction input contract.

`--masks-json` is forwarded to `session.predict(..., masks=...)` after JSON parsing. The wrapper does not reinterpret mask coordinate systems or dimensions at the CLI boundary. Mask shape compatibility and the requirement that masks are supplied with bboxes are enforced by the public prediction validation contract.

`--cam-int-json` is converted to a `torch.float32` tensor before prediction because upstream expects `cam_int` to provide tensor methods such as `.to(...)` and `.clone()`. The option therefore requires the inference dependency stack to make `torch` importable.

The JSON output must be derived from `Sam3DBodyResult.to_dict()`. Values that are not directly JSON serializable, such as tensors or arrays, may be converted to lists when supported or represented using a deterministic string fallback.

## Future Commands

Future CLI commands may include:

* mutating upstream setup
* checkpoint download
* batch inference
* output conversion
* diagnostics

Future command behavior must be specified before being treated as stable.
