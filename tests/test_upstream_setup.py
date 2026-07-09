from pathlib import Path

from sam3dbody.upstream_setup import plan_upstream_setup


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
