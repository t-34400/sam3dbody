"""Upstream setup planning and installation for SAM 3D Body source assets."""

from __future__ import annotations

import subprocess
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from sam3dbody.paths import default_upstream_root


DEFAULT_UPSTREAM_SOURCE_URL = "https://github.com/facebookresearch/sam-3d-body.git"

GitRunner = Callable[[Sequence[str]], Any]


@dataclass(frozen=True)
class Sam3DBodyUpstreamSetupPlan:
    """Wrapper-owned description of a non-mutating upstream setup plan."""

    target: Path
    source_url: str
    revision: str | None
    recursive: bool
    target_exists: bool
    upstream_package_exists: bool

    @property
    def needs_clone(self) -> bool:
        return not self.target_exists

    @property
    def needs_verification(self) -> bool:
        return self.target_exists and not self.upstream_package_exists

    @property
    def commands(self) -> tuple[str, ...]:
        return tuple(" ".join(command) for command in self.command_argvs)

    @property
    def command_argvs(self) -> tuple[tuple[str, ...], ...]:
        commands: list[tuple[str, ...]] = []
        target = str(self.target)
        if self.needs_clone:
            commands.append(("git", "clone", self.source_url, target))
            if self.revision is not None:
                commands.append(("git", "-C", target, "checkout", self.revision))
            if self.recursive:
                commands.append(("git", "-C", target, "submodule", "update", "--init", "--recursive"))
        elif self.recursive:
            commands.append(("git", "-C", target, "submodule", "update", "--init", "--recursive"))
        return tuple(commands)

    @property
    def status(self) -> str:
        if self.upstream_package_exists:
            return "ready"
        if self.needs_clone:
            return "missing"
        return "incomplete"

    def to_dict(self) -> dict[str, object]:
        return {
            "target": str(self.target),
            "source_url": self.source_url,
            "revision": self.revision,
            "recursive": self.recursive,
            "target_exists": self.target_exists,
            "upstream_package_exists": self.upstream_package_exists,
            "status": self.status,
            "commands": list(self.commands),
        }


@dataclass(frozen=True)
class Sam3DBodyUpstreamInstallResult:
    """Wrapper-owned result for a mutating upstream source setup attempt."""

    target: Path
    source_url: str
    revision: str | None
    recursive: bool
    before_status: str
    target_exists: bool
    upstream_package_exists: bool
    commands_run: tuple[str, ...]
    success: bool
    message: str

    @property
    def status(self) -> str:
        if self.upstream_package_exists:
            return "ready"
        if self.target_exists:
            return "incomplete"
        return "missing"

    def to_dict(self) -> dict[str, object]:
        return {
            "target": str(self.target),
            "source_url": self.source_url,
            "revision": self.revision,
            "recursive": self.recursive,
            "before_status": self.before_status,
            "status": self.status,
            "target_exists": self.target_exists,
            "upstream_package_exists": self.upstream_package_exists,
            "commands_run": list(self.commands_run),
            "success": self.success,
            "message": self.message,
        }


def plan_upstream_setup(
    *,
    target: Path | None = None,
    source_url: str = DEFAULT_UPSTREAM_SOURCE_URL,
    revision: str | None = None,
    recursive: bool = True,
) -> Sam3DBodyUpstreamSetupPlan:
    """Create a non-mutating plan for preparing upstream source code."""
    setup_target = target if target is not None else default_upstream_root()
    return Sam3DBodyUpstreamSetupPlan(
        target=setup_target,
        source_url=source_url,
        revision=revision,
        recursive=recursive,
        target_exists=setup_target.exists(),
        upstream_package_exists=(setup_target / "sam_3d_body").is_dir(),
    )


def install_upstream_source(
    *,
    target: Path | None = None,
    source_url: str = DEFAULT_UPSTREAM_SOURCE_URL,
    revision: str | None = None,
    recursive: bool = True,
    runner: GitRunner | None = None,
) -> Sam3DBodyUpstreamInstallResult:
    """Prepare upstream source code by executing the planned Git commands."""
    before = plan_upstream_setup(
        target=target,
        source_url=source_url,
        revision=revision,
        recursive=recursive,
    )
    if before.status == "incomplete":
        return _install_result(
            before,
            commands_run=(),
            success=False,
            message="target exists but does not contain sam_3d_body; refusing to overwrite it",
        )

    command_runner = runner if runner is not None else _run_git_command
    commands_run: list[str] = []
    try:
        if before.needs_clone:
            before.target.parent.mkdir(parents=True, exist_ok=True)
        for command in before.command_argvs:
            command_runner(command)
            commands_run.append(" ".join(command))
    except subprocess.CalledProcessError as exc:
        return _install_result(
            before,
            commands_run=tuple(commands_run),
            success=False,
            message=f"git command failed with exit status {exc.returncode}",
        )

    after = plan_upstream_setup(
        target=before.target,
        source_url=source_url,
        revision=revision,
        recursive=recursive,
    )
    if after.upstream_package_exists:
        if commands_run:
            message = "upstream source is ready"
        else:
            message = "upstream source was already ready"
        return _install_result(before, commands_run=tuple(commands_run), success=True, message=message)
    return _install_result(
        before,
        commands_run=tuple(commands_run),
        success=False,
        message="commands completed but sam_3d_body was not found under the target",
    )


def _run_git_command(command: Sequence[str]) -> None:
    subprocess.run(list(command), check=True)


def _install_result(
    before: Sam3DBodyUpstreamSetupPlan,
    *,
    commands_run: tuple[str, ...],
    success: bool,
    message: str,
) -> Sam3DBodyUpstreamInstallResult:
    return Sam3DBodyUpstreamInstallResult(
        target=before.target,
        source_url=before.source_url,
        revision=before.revision,
        recursive=before.recursive,
        before_status=before.status,
        target_exists=before.target.exists(),
        upstream_package_exists=(before.target / "sam_3d_body").is_dir(),
        commands_run=commands_run,
        success=success,
        message=message,
    )
