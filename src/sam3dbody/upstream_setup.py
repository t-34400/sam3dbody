"""Upstream setup planning for SAM 3D Body source assets."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


DEFAULT_UPSTREAM_SOURCE_URL = "https://github.com/facebookresearch/sam-3d-body.git"
DEFAULT_UPSTREAM_TARGET = Path("third_party/sam-3d-body")


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
        commands: list[str] = []
        target = str(self.target)
        if self.needs_clone:
            clone_command = f"git clone {self.source_url} {target}"
            commands.append(clone_command)
            if self.revision is not None:
                commands.append(f"git -C {target} checkout {self.revision}")
            if self.recursive:
                commands.append(f"git -C {target} submodule update --init --recursive")
        elif self.recursive:
            commands.append(f"git -C {target} submodule update --init --recursive")
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


def plan_upstream_setup(
    *,
    target: Path | None = None,
    source_url: str = DEFAULT_UPSTREAM_SOURCE_URL,
    revision: str | None = None,
    recursive: bool = True,
) -> Sam3DBodyUpstreamSetupPlan:
    """Create a non-mutating plan for preparing upstream source code."""
    setup_target = target if target is not None else DEFAULT_UPSTREAM_TARGET
    return Sam3DBodyUpstreamSetupPlan(
        target=setup_target,
        source_url=source_url,
        revision=revision,
        recursive=recursive,
        target_exists=setup_target.exists(),
        upstream_package_exists=(setup_target / "sam_3d_body").is_dir(),
    )
