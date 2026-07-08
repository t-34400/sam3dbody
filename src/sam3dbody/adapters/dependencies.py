"""Static dependency inspection for the upstream source tree."""

from __future__ import annotations

import ast
from dataclasses import dataclass, field
from pathlib import Path


_STDLIB_MODULES = {
    "__future__",
    "abc",
    "argparse",
    "collections",
    "concurrent",
    "functools",
    "glob",
    "itertools",
    "json",
    "logging",
    "math",
    "os",
    "pathlib",
    "pickle",
    "random",
    "re",
    "shutil",
    "sys",
    "tempfile",
    "time",
    "typing",
    "warnings",
}

_LOCAL_MODULES = {
    "_init_paths",
    "config",
    "lib",
    "notebook",
    "sam_3d_body",
    "tools",
}


@dataclass(frozen=True)
class UpstreamDependencyReport:
    """Static import summary for the upstream repository."""

    repository_root: Path
    import_modules: tuple[str, ...] = field(default_factory=tuple)
    external_modules: tuple[str, ...] = field(default_factory=tuple)
    files_scanned: int = 0

    def to_dict(self) -> dict[str, object]:
        return {
            "repository_root": str(self.repository_root),
            "import_modules": list(self.import_modules),
            "external_modules": list(self.external_modules),
            "files_scanned": self.files_scanned,
        }


def inspect_upstream_dependencies(repository_root: str | Path) -> UpstreamDependencyReport:
    """Return a static import summary without importing upstream modules."""
    root = Path(repository_root)
    modules: set[str] = set()
    files_scanned = 0

    for path in sorted(root.rglob("*.py")):
        if ".git" in path.parts:
            continue
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except SyntaxError:
            continue
        files_scanned += 1
        modules.update(_top_level_imports(tree))

    external = tuple(
        sorted(module for module in modules if module not in _STDLIB_MODULES and module not in _LOCAL_MODULES)
    )
    return UpstreamDependencyReport(
        repository_root=root,
        import_modules=tuple(sorted(modules)),
        external_modules=external,
        files_scanned=files_scanned,
    )


def _top_level_imports(tree: ast.AST) -> set[str]:
    modules: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.update(alias.name.split(".", 1)[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            modules.add(node.module.split(".", 1)[0])
    return modules
