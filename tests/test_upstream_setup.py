from pathlib import Path

from sam3dbody.upstream_setup import install_upstream_source, plan_upstream_setup


def test_plan_upstream_setup_reports_missing_target(tmp_path: Path) -> None:
    target = tmp_path / "sam-3d-body"

    plan = plan_upstream_setup(target=target, source_url="https://example.com/upstream.git", revision="abc123")

    assert plan.status == "missing"
    assert plan.target_exists is False
    assert plan.upstream_package_exists is False
    assert plan.commands == (
        f"git clone https://example.com/upstream.git {target}",
        f"git -C {target} checkout abc123",
        f"git -C {target} submodule update --init --recursive",
    )
    assert plan.to_dict()["status"] == "missing"


def test_plan_upstream_setup_reports_ready_target(tmp_path: Path) -> None:
    target = tmp_path / "sam-3d-body"
    (target / "sam_3d_body").mkdir(parents=True)

    plan = plan_upstream_setup(target=target)

    assert plan.status == "ready"
    assert plan.target_exists is True
    assert plan.upstream_package_exists is True
    assert plan.commands == (f"git -C {target} submodule update --init --recursive",)


def test_plan_upstream_setup_can_omit_recursive_command(tmp_path: Path) -> None:
    target = tmp_path / "sam-3d-body"

    plan = plan_upstream_setup(target=target, recursive=False)

    assert plan.commands == (f"git clone https://github.com/facebookresearch/sam-3d-body.git {target}",)


def test_install_upstream_source_clones_missing_target(tmp_path: Path) -> None:
    target = tmp_path / "sam-3d-body"
    commands: list[tuple[str, ...]] = []

    def fake_runner(command):
        commands.append(tuple(command))
        if tuple(command[:2]) == ("git", "clone"):
            (target / "sam_3d_body").mkdir(parents=True)

    result = install_upstream_source(
        target=target,
        source_url="https://example.com/upstream.git",
        revision="abc123",
        runner=fake_runner,
    )

    assert result.success is True
    assert result.status == "ready"
    assert commands == [
        ("git", "clone", "https://example.com/upstream.git", str(target)),
        ("git", "-C", str(target), "checkout", "abc123"),
        ("git", "-C", str(target), "submodule", "update", "--init", "--recursive"),
    ]
    assert result.commands_run == tuple(" ".join(command) for command in commands)


def test_install_upstream_source_refuses_incomplete_target(tmp_path: Path) -> None:
    target = tmp_path / "sam-3d-body"
    target.mkdir()
    commands: list[tuple[str, ...]] = []

    result = install_upstream_source(target=target, runner=lambda command: commands.append(tuple(command)))

    assert result.success is False
    assert result.status == "incomplete"
    assert commands == []
    assert "refusing" in result.message


def test_install_upstream_source_reports_already_ready_without_recursive(tmp_path: Path) -> None:
    target = tmp_path / "sam-3d-body"
    (target / "sam_3d_body").mkdir(parents=True)
    commands: list[tuple[str, ...]] = []

    result = install_upstream_source(target=target, recursive=False, runner=lambda command: commands.append(tuple(command)))

    assert result.success is True
    assert result.status == "ready"
    assert result.commands_run == ()
    assert commands == []


def test_default_upstream_root_uses_source_checkout_layout() -> None:
    from sam3dbody.paths import default_upstream_root

    assert default_upstream_root().as_posix().endswith("third_party/sam-3d-body")
