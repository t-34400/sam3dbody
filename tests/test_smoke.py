from pathlib import Path
import sys

import pytest

from sam3dbody.smoke import Sam3DBodySmokeTestConfig, run_smoke_test, summarize_result
from sam3dbody.result import Sam3DBodyMetadata, Sam3DBodyPrediction, Sam3DBodyResult


def test_summarize_result_reports_shapes_and_extra_keys() -> None:
    result = Sam3DBodyResult(
        bodies=[
            Sam3DBodyPrediction(
                bbox_xyxy=(1.0, 2.0, 3.0, 4.0),
                vertices=[[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]],
                joints=[[6.0, 7.0, 8.0]],
                faces=[[0, 1, 2]],
                extra={"raw": [[1, 2], [3, 4]]},
            )
        ],
        metadata=Sam3DBodyMetadata(model_name="sam-3d-body", device="cuda", extra={"schema": "test"}),
    )

    summary = summarize_result(result)

    assert summary["body_count"] == 1
    assert summary["metadata"]["extra"] == {"schema": "test"}
    assert summary["bodies"][0]["vertices"]["shape"] == [2, 3]
    assert summary["bodies"][0]["joints"]["shape"] == [1, 3]
    assert summary["bodies"][0]["extra_keys"] == ["raw"]
    assert summary["bodies"][0]["extra"]["raw"]["shape"] == [2, 2]


def test_run_smoke_test_uses_real_api_path_with_fake_upstream(tmp_path: Path) -> None:
    upstream_root = tmp_path / "sam-3d-body"
    package = upstream_root / "sam_3d_body"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text(
        "def load_sam_3d_body(checkpoint_path, device='cuda', mhr_path=''):\n"
        "    return {'device': device}, {'checkpoint_path': checkpoint_path}\n"
        "class FakeArray:\n"
        "    def __init__(self, values, dtype='float32'):\n"
        "        self.values = values\n"
        "        self.shape = (len(values), len(values[0]))\n"
        "        self.dtype = dtype\n"
        "    def tolist(self):\n"
        "        return self.values\n"
        "class SAM3DBodyEstimator:\n"
        "    def __init__(self, sam_3d_body_model, model_cfg, human_detector=None, human_segmentor=None, fov_estimator=None):\n"
        "        self.faces = FakeArray([[0, 1, 2]], dtype='int64')\n"
        "    def process_one_image(self, img, **kwargs):\n"
        "        return [{'bbox': [1, 2, 3, 4], 'pred_vertices': FakeArray([[0.0, 1.0, 2.0]]), 'pred_keypoints_3d': FakeArray([[3.0, 4.0, 5.0]]), 'image_arg': img}]\n"
    )
    image = tmp_path / "image.png"
    weights = tmp_path / "model.ckpt"
    image.write_bytes(b"fake")
    weights.write_text("fake")

    sys.modules.pop("sam_3d_body", None)

    report = run_smoke_test(
        Sam3DBodySmokeTestConfig(
            image=image,
            weights_path=weights,
            upstream_root=upstream_root,
            repeat=2,
            skip_env_check=True,
        )
    )

    assert report["success"] is True
    assert report["single"]["body_count"] == 1
    assert report["single"]["bodies"][0]["vertices"]["shape"] == [1, 3]
    assert report["single"]["bodies"][0]["vertices"]["dtype"] == "float32"
    assert report["batch"]["requested_count"] == 2
    assert report["batch"]["result_count"] == 2


def test_run_smoke_test_rejects_negative_repeat(tmp_path: Path) -> None:
    image = tmp_path / "image.png"
    weights = tmp_path / "model.ckpt"
    image.write_bytes(b"fake")
    weights.write_text("fake")

    with pytest.raises(ValueError, match="repeat"):
        run_smoke_test(
            Sam3DBodySmokeTestConfig(
                image=image,
                weights_path=weights,
                repeat=-1,
                skip_env_check=True,
            )
        )
