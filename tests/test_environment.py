from pathlib import Path

from sam3dbody.environment import check_environment


def test_check_environment_does_not_require_weights_path(tmp_path: Path) -> None:
    upstream_root = tmp_path / "sam-3d-body"
    (upstream_root / "sam_3d_body").mkdir(parents=True)

    report = check_environment(upstream_root=upstream_root, modules=())

    assert report.upstream_root == upstream_root
    assert report.upstream_exists is True
    assert report.upstream_package_exists is True
    assert report.weights_path is None
    assert report.weights_exists is None
    assert report.modules == {}
    assert report.torch_cuda_available is None
    assert report.ready_for_inference is False


def test_check_environment_reports_ready_when_all_required_checks_pass(tmp_path: Path) -> None:
    upstream_root = tmp_path / "sam-3d-body"
    (upstream_root / "sam_3d_body").mkdir(parents=True)
    weights = tmp_path / "model.ckpt"
    weights.write_text("fake")

    report = check_environment(upstream_root=upstream_root, weights_path=weights, modules=())

    assert report.weights_exists is True
    assert report.ready_for_inference is False


def test_environment_report_serializes_paths(tmp_path: Path) -> None:
    upstream_root = tmp_path / "missing"
    weights = tmp_path / "missing.ckpt"

    payload = check_environment(upstream_root=upstream_root, weights_path=weights, modules=()).to_dict()

    assert payload["upstream_root"] == str(upstream_root)
    assert payload["weights_path"] == str(weights)
    assert payload["weights_exists"] is False
