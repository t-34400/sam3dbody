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

When `--upstream-root` is omitted, the command uses the wrapper default upstream source location. In a source checkout, this is `third_party/sam-3d-body`. In an installed wheel or Git URL installation, this is `.local/upstream/sam-3d-body` relative to the current working directory. The default must not resolve inside `site-packages` or a virtual environment.

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
  The representative module list includes ordinary PyPI prerequisites and real-inference prerequisites observed during upstream import, including `braceexpand`, `roma`, `pytorch_lightning`, and `termcolor`.
* whether CUDA appears available through torch when torch is importable.

When `--upstream-root` is omitted, the command uses the wrapper default upstream source location. In a source checkout, this is `third_party/sam-3d-body`. In an installed wheel or Git URL installation, this is `.local/upstream/sam-3d-body` relative to the current working directory. The default must not resolve inside `site-packages` or a virtual environment.

When `--json` is provided, the command prints a JSON object using the wrapper-owned environment report schema. Without `--json`, the command prints a human-readable summary. The human-readable summary must include a `missing_requirements` section that lists the readiness blockers used to decide `ready_for_inference`, or `none` when no blockers are present.

By default, `check-env` exits with status code `0` after reporting diagnostics, even when the environment is not inference-ready. When `--strict` is provided, the command exits with status code `0` only if the report is ready for inference, and exits with status code `1` otherwise.

`check-env` is diagnostic only. It must not attempt to initialize Git submodules, clone upstream repositories, download checkpoints, import upstream modules, or mutate the filesystem.

## Upstream Setup Planning Command

```text
sam3dbody plan-upstream-setup [--target PATH] [--source-url URL] [--revision REV] [--no-recursive] [--json]
```

`plan-upstream-setup` prints a non-mutating plan for preparing the upstream SAM 3D Body source tree.

The command must not clone repositories, initialize Git submodules, download checkpoints, install dependencies, import upstream modules, or otherwise mutate the filesystem. It is intended to make the future setup workflow explicit before an installing command is implemented.

When `--target` is omitted, the command uses the wrapper default upstream source location. In a source checkout, this is `third_party/sam-3d-body`. In an installed wheel or Git URL installation, this is `.local/upstream/sam-3d-body` relative to the current working directory. The default must not resolve inside `site-packages` or a virtual environment.

The command reports whether the target exists, whether the upstream `sam_3d_body` package directory exists, and a suggested command list. Command strings must be rendered as shell-readable commands with explicit argument separation. When the upstream package directory already exists, the command reports status `ready`. When the target is missing, it reports status `missing`. When the target exists but does not contain the upstream package directory, it reports status `incomplete`.

When `--json` is provided, the command prints a JSON object using the wrapper-owned upstream setup plan schema. Without `--json`, the command prints a human-readable summary.

`plan-upstream-setup` always exits with status code `0` because it is advisory and non-mutating. Readiness enforcement remains the responsibility of `check-env --strict`.

## Upstream Installation Command

```text
sam3dbody install-upstream [--target PATH] [--source-url URL] [--revision REV] [--no-recursive] [--json]
```

`install-upstream` is the explicit mutating setup path for preparing the upstream SAM 3D Body source tree. It may create parent directories for the target, clone the upstream repository when the target is missing, optionally check out a requested revision, and optionally initialize upstream submodules recursively.

When `--target` is omitted, the command uses the wrapper default upstream source location. In a source checkout, this is `third_party/sam-3d-body`. In an installed wheel or Git URL installation, this is `.local/upstream/sam-3d-body` relative to the current working directory. The default must not resolve inside `site-packages` or a virtual environment.

When the target already contains the upstream `sam_3d_body` package directory, the command treats the source tree as ready. With recursive setup enabled, it may run recursive submodule initialization against the ready tree. With `--no-recursive`, it performs no Git command for an already-ready target.

When the target exists but does not contain the upstream `sam_3d_body` package directory, the command must refuse to overwrite or repair it automatically and must report failure. Users should move, remove, or inspect the existing directory manually.

When `--json` is provided, the command prints a JSON object using the wrapper-owned upstream install result schema. Without `--json`, the command prints a human-readable summary including the previous status, final status, success flag, message, and commands run. Command strings in `commands_run` must be rendered as shell-readable commands with explicit argument separation.

The command exits with status code `0` only when the final target contains the upstream `sam_3d_body` package directory. It exits with status code `1` when setup fails or the target remains incomplete.

`install-upstream` prepares source code only. It must not download checkpoints, download MHR assets, install Python dependencies, import upstream modules, or run inference.

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

## Real Inference Smoke Test Command

```text
sam3dbody smoke-test IMAGE --weights PATH [--output PATH] [--device DEVICE] [--mhr-path PATH] [--upstream-root PATH] [--repeat N] [--skip-env-check]
```

`smoke-test` is an explicitly requested real-inference validation command. It is intended for local or CI environments that have upstream source code, model weights, CUDA, and inference dependencies available.

The command first runs the same non-mutating prerequisite checks as `check-env`. Unless `--skip-env-check` is supplied, the command must not attempt model loading or inference when the environment report is not ready for inference. In that case it writes a smoke report with `success: false` and exits with status code `1`.

When prerequisites are ready, or when `--skip-env-check` is supplied, the command loads the wrapper model through the public wrapper API and runs one `session.predict(IMAGE)` call. When `--repeat N` is greater than zero, the command must also run `session.predict_many([IMAGE] * N)` to exercise the ordered repeated-inference path. Runtime failures during model loading or inference must be captured in the smoke report as `success: false` rather than escaping as an unstructured traceback.

The command writes a wrapper-owned JSON report. The report must include:

* a smoke report schema version;
* input paths and device settings, including the resolved upstream source root;
* the environment readiness report;
* single-image result summary;
* optional repeated batch result summary;
* success flag and message.

Result summaries must avoid embedding full tensors or arrays. They should record body counts, metadata, field presence, field type, shape, dtype when available, bbox values, and extra output keys. The top-level `upstream_root` field must contain the resolved upstream source root used by the environment check, even when `--upstream-root` was omitted. This command is a diagnostic aid and does not define stable model accuracy expectations.

`--repeat` must be greater than or equal to zero. The command exits with status code `0` only when the smoke test completes successfully. When `--output` is provided, the command must still print the report path on success or failure. When the smoke report has `success: false`, the command must also print a concise failure summary to stderr so users are not left with a silent non-zero exit.

The underlying `run_smoke_test()` API must enforce the same non-negative `repeat` requirement so direct API callers cannot bypass CLI validation.


## Future Commands

Future CLI commands may include:

* checkpoint download
* output conversion
* additional diagnostics

Future command behavior must be specified before being treated as stable.
