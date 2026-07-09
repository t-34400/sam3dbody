from pathlib import Path

from sam3dbody.adapters import (
    Sam3DBodyLoadConfig,
    Sam3DBodyPredictionOptions,
    Sam3DBodyUpstreamAdapter,
)
from sam3dbody.model import Sam3DBodyModel


def _write_fake_upstream(repo: Path) -> None:
    package = repo / "sam_3d_body"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text(
        "class FakeModel:\n"
        "    device = 'cpu'\n"
        "\n"
        "def load_sam_3d_body(checkpoint_path, device='cuda', mhr_path=''):\n"
        "    model = FakeModel()\n"
        "    model.device = device\n"
        "    return model, {'image_size': 256}\n"
        "\n"
        "class SAM3DBodyEstimator:\n"
        "    def __init__(self, sam_3d_body_model, model_cfg, human_detector=None, human_segmentor=None, fov_estimator=None):\n"
        "        self.model = sam_3d_body_model\n"
        "        self.cfg = model_cfg\n"
        "        self.detector = human_detector\n"
        "        self.sam = human_segmentor\n"
        "        self.fov_estimator = fov_estimator\n"
        "        self.faces = [[0, 1, 2]]\n"
        "\n"
        "    def process_one_image(self, img, **kwargs):\n"
        "        return [{\n"
        "            'bbox': [1, 2, 3, 4],\n"
        "            'pred_vertices': [[0.0, 0.0, 0.0]],\n"
        "            'pred_keypoints_3d': [[1.0, 1.0, 1.0]],\n"
        "            'pred_cam_t': [0.1, 0.2, 0.3],\n"
        "            'image_arg': img,\n"
        "            'inference_type': kwargs['inference_type'],\n"
        "            'bbox_thr': kwargs['bbox_thr'],\n"
        "        }]\n"
    )


def test_adapter_predict_calls_upstream_estimator_and_converts_result(tmp_path: Path) -> None:
    repo = tmp_path / "sam-3d-body"
    _write_fake_upstream(repo)
    checkpoint = tmp_path / "model.ckpt"
    checkpoint.write_text("fake")
    adapter = Sam3DBodyUpstreamAdapter.from_repository_root(repo)
    loaded = adapter.load(Sam3DBodyLoadConfig.from_values(checkpoint, device="cpu"))

    result = adapter.predict(
        Path("image.png"),
        loaded,
        options=Sam3DBodyPredictionOptions(bbox_thr=0.9, inference_type="body"),
    )

    assert result.metadata.model_name == "sam-3d-body"
    assert result.metadata.device == "cpu"
    assert len(result.bodies) == 1
    body = result.bodies[0]
    assert body.bbox_xyxy == (1.0, 2.0, 3.0, 4.0)
    assert body.vertices == [[0.0, 0.0, 0.0]]
    assert body.faces == [[0, 1, 2]]
    assert body.joints == [[1.0, 1.0, 1.0]]
    assert body.extra["image_arg"] == "image.png"
    assert body.extra["inference_type"] == "body"
    assert body.extra["bbox_thr"] == 0.9


def test_public_model_predict_loads_and_uses_adapter_boundary(tmp_path: Path) -> None:
    repo = tmp_path / "sam-3d-body"
    _write_fake_upstream(repo)
    checkpoint = tmp_path / "model.ckpt"
    checkpoint.write_text("fake")
    model = Sam3DBodyModel(
        weights_path=checkpoint,
        device="cpu",
        adapter=Sam3DBodyUpstreamAdapter.from_repository_root(repo),
    )

    result = model.predict("image.png", inference_type="body")

    assert result.bodies[0].bbox_xyxy == (1.0, 2.0, 3.0, 4.0)
    assert result.bodies[0].extra["inference_type"] == "body"


def test_predict_options_document_forwarded_kwargs() -> None:
    options = Sam3DBodyPredictionOptions(bbox_thr=0.7, nms_thr=0.2, use_mask=True)

    assert options.to_upstream_kwargs()["bbox_thr"] == 0.7
    assert options.to_upstream_kwargs()["nms_thr"] == 0.2
    assert options.to_upstream_kwargs()["use_mask"] is True


def test_public_model_load_returns_reusable_session(tmp_path: Path) -> None:
    repo = tmp_path / "sam-3d-body"
    _write_fake_upstream(repo)
    checkpoint = tmp_path / "model.ckpt"
    checkpoint.write_text("fake")
    model = Sam3DBodyModel(
        weights_path=checkpoint,
        device="cpu",
        adapter=Sam3DBodyUpstreamAdapter.from_repository_root(repo),
    )

    session = model.load()
    first = session.predict("first.png")
    second = session.predict("second.png", inference_type="body")

    assert session.device == "cpu"
    assert first.bodies[0].extra["image_arg"] == "first.png"
    assert second.bodies[0].extra["image_arg"] == "second.png"
    assert second.bodies[0].extra["inference_type"] == "body"


def test_session_does_not_reload_weights_for_repeated_prediction(tmp_path: Path) -> None:
    repo = tmp_path / "sam-3d-body"
    marker = tmp_path / "load_calls.txt"
    package = repo / "sam_3d_body"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text(
        "from pathlib import Path\n"
        f"MARKER = Path({str(marker)!r})\n"
        "def load_sam_3d_body(checkpoint_path, device='cuda', mhr_path=''):\n"
        "    with MARKER.open('a') as f:\n"
        "        f.write('load\\n')\n"
        "    return {'device': device}, {'ok': True}\n"
        "class SAM3DBodyEstimator:\n"
        "    def __init__(self, sam_3d_body_model, model_cfg, human_detector=None, human_segmentor=None, fov_estimator=None):\n"
        "        self.faces = []\n"
        "    def process_one_image(self, img, **kwargs):\n"
        "        return [{'bbox': [0, 0, 1, 1], 'image_arg': img}]\n"
    )
    checkpoint = tmp_path / "model.ckpt"
    checkpoint.write_text("fake")
    model = Sam3DBodyModel(
        weights_path=checkpoint,
        device="cpu",
        adapter=Sam3DBodyUpstreamAdapter.from_repository_root(repo),
    )

    session = model.load()
    session.predict("a.png")
    session.predict("b.png")

    assert marker.read_text().splitlines() == ["load"]
