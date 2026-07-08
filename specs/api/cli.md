# CLI

## Purpose

This document defines the current command line interface contract.

## Stability

The CLI is experimental unless a command is explicitly marked stable.

## Initial Command

The package provides a `sam3dbody` console command after installation.

The initial supported subcommand is:

```text
sam3dbody inspect-deps [--upstream-root PATH] [--json]
```

## Dependency Inspection Command

`inspect-deps` statically inspects the upstream SAM 3D Body source tree and reports observed top-level import requirements.

The command must not import upstream modules or execute upstream inference code.

When `--upstream-root` is omitted, the command uses the source-tree upstream repository under:

```text
third_party/sam-3d-body
```

When `--json` is provided, the command prints a JSON object using the wrapper-owned dependency report schema.

Without `--json`, the command prints a human-readable summary.

## Future Commands

Future CLI commands may include:

* model inspection
* single image inference
* batch inference
* output conversion
* diagnostics

Future command behavior must be specified before being treated as stable.
