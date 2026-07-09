import sys
from pathlib import Path

import pytest

from sam3dbody import Sam3DBodyInputError
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
    loaded = adapter.load(Sam3DBodyLoadConfig.from_values(checkpoint, device="cuda"))

    image_path = tmp_path / "image.png"
    image_path.write_bytes(b"fake")

    result = adapter.predict(
        image_path,
        loaded,
        options=Sam3DBodyPredictionOptions(bbox_thr=0.9, inference_type="body"),
    )

    assert result.metadata.model_name == "sam-3d-body"
    assert result.metadata.device == "cuda"
    assert len(result.bodies) == 1
    body = result.bodies[0]
    assert body.bbox_xyxy == (1.0, 2.0, 3.0, 4.0)
    assert body.vertices == [[0.0, 0.0, 0.0]]
    assert body.faces == [[0, 1, 2]]
    assert body.joints == [[1.0, 1.0, 1.0]]
    assert body.extra["image_arg"] == str(image_path)
    assert body.extra["inference_type"] == "body"
    assert body.extra["bbox_thr"] == 0.9


def test_public_model_predict_loads_and_uses_adapter_boundary(tmp_path: Path) -> None:
    repo = tmp_path / "sam-3d-body"
    _write_fake_upstream(repo)
    checkpoint = tmp_path / "model.ckpt"
    checkpoint.write_text("fake")
    model = Sam3DBodyModel(
        weights_path=checkpoint,
        device="cuda",
        adapter=Sam3DBodyUpstreamAdapter.from_repository_root(repo),
    )

    image_path = tmp_path / "image.png"
    image_path.write_bytes(b"fake")

    result = model.predict(image_path, inference_type="body")

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
        device="cuda",
        adapter=Sam3DBodyUpstreamAdapter.from_repository_root(repo),
    )

    session = model.load()
    first_path = tmp_path / "first.png"
    second_path = tmp_path / "second.png"
    first_path.write_bytes(b"fake")
    second_path.write_bytes(b"fake")

    first = session.predict(first_path)
    second = session.predict(second_path, inference_type="body")

    assert session.device == "cuda"
    assert first.bodies[0].extra["image_arg"] == str(first_path)
    assert second.bodies[0].extra["image_arg"] == str(second_path)
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
        device="cuda",
        adapter=Sam3DBodyUpstreamAdapter.from_repository_root(repo),
    )

    session = model.load()
    a_path = tmp_path / "a.png"
    b_path = tmp_path / "b.png"
    a_path.write_bytes(b"fake")
    b_path.write_bytes(b"fake")

    session.predict(a_path)
    session.predict(b_path)

    assert marker.read_text().splitlines() == ["load"]


def test_predict_rejects_missing_image_before_upstream_call(tmp_path: Path) -> None:
    repo = tmp_path / "sam-3d-body"
    _write_fake_upstream(repo)
    checkpoint = tmp_path / "model.ckpt"
    checkpoint.write_text("fake")
    model = Sam3DBodyModel(
        weights_path=checkpoint,
        device="cuda",
        adapter=Sam3DBodyUpstreamAdapter.from_repository_root(repo),
    )
    session = model.load()

    with pytest.raises(Sam3DBodyInputError, match="image path does not exist"):
        session.predict(tmp_path / "missing.png")


def test_predict_rejects_non_cuda_device_before_upstream_call(tmp_path: Path) -> None:
    repo = tmp_path / "sam-3d-body"
    _write_fake_upstream(repo)
    checkpoint = tmp_path / "model.ckpt"
    image_path = tmp_path / "image.png"
    checkpoint.write_text("fake")
    image_path.write_bytes(b"fake")
    model = Sam3DBodyModel(
        weights_path=checkpoint,
        device="cpu",
        adapter=Sam3DBodyUpstreamAdapter.from_repository_root(repo),
    )
    session = model.load()

    with pytest.raises(Sam3DBodyInputError, match="requires a CUDA device"):
        session.predict(image_path)


def test_predict_validates_bboxes_masks_and_inference_type(tmp_path: Path) -> None:
    repo = tmp_path / "sam-3d-body"
    _write_fake_upstream(repo)
    checkpoint = tmp_path / "model.ckpt"
    image_path = tmp_path / "image.png"
    checkpoint.write_text("fake")
    image_path.write_bytes(b"fake")
    model = Sam3DBodyModel(
        weights_path=checkpoint,
        device="cuda",
        adapter=Sam3DBodyUpstreamAdapter.from_repository_root(repo),
    )
    session = model.load()

    with pytest.raises(Sam3DBodyInputError, match="bboxes must have shape N x 4"):
        session.predict(image_path, bboxes=[1, 2, 3, 4])

    with pytest.raises(Sam3DBodyInputError, match="masks require bboxes"):
        session.predict(image_path, masks=[[[1]]])

    with pytest.raises(Sam3DBodyInputError, match="inference_type must be one of"):
        session.predict(image_path, inference_type="invalid")


def test_session_predict_many_preserves_order_and_reuses_estimator(tmp_path: Path) -> None:
    repo = tmp_path / "sam-3d-body"
    marker = tmp_path / "estimator_calls.txt"
    package = repo / "sam_3d_body"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text(
        "from pathlib import Path\n"
        f"MARKER = Path({str(marker)!r})\n"
        "def load_sam_3d_body(checkpoint_path, device='cuda', mhr_path=''):\n"
        "    return {'device': device}, {'ok': True}\n"
        "class SAM3DBodyEstimator:\n"
        "    def __init__(self, sam_3d_body_model, model_cfg, human_detector=None, human_segmentor=None, fov_estimator=None):\n"
        "        with MARKER.open('a') as f:\n"
        "            f.write('init\\n')\n"
        "        self.faces = []\n"
        "    def process_one_image(self, img, **kwargs):\n"
        "        return [{'bbox': [0, 0, 1, 1], 'image_arg': img, 'inference_type': kwargs['inference_type']}]\n"
    )
    checkpoint = tmp_path / "model.ckpt"
    checkpoint.write_text("fake")
    paths = [tmp_path / "a.png", tmp_path / "b.png", tmp_path / "c.png"]
    for path in paths:
        path.write_bytes(b"fake")
    sys.modules.pop("sam_3d_body", None)
    model = Sam3DBodyModel(
        weights_path=checkpoint,
        device="cuda",
        adapter=Sam3DBodyUpstreamAdapter.from_repository_root(repo),
    )

    session = model.load()
    results = session.predict_many(paths, inference_type="body")

    assert [result.bodies[0].extra["image_arg"] for result in results] == [str(path) for path in paths]
    assert [result.bodies[0].extra["inference_type"] for result in results] == ["body", "body", "body"]
    assert marker.read_text().splitlines() == ["init"]


def test_predict_many_rejects_single_path_argument(tmp_path: Path) -> None:
    repo = tmp_path / "sam-3d-body"
    _write_fake_upstream(repo)
    checkpoint = tmp_path / "model.ckpt"
    image_path = tmp_path / "image.png"
    checkpoint.write_text("fake")
    image_path.write_bytes(b"fake")
    model = Sam3DBodyModel(
        weights_path=checkpoint,
        device="cuda",
        adapter=Sam3DBodyUpstreamAdapter.from_repository_root(repo),
    )
    session = model.load()

    with pytest.raises(Sam3DBodyInputError, match="iterable of image inputs"):
        session.predict_many(image_path)


def test_predict_many_empty_iterable_returns_empty_list(tmp_path: Path) -> None:
    repo = tmp_path / "sam-3d-body"
    _write_fake_upstream(repo)
    checkpoint = tmp_path / "model.ckpt"
    checkpoint.write_text("fake")
    model = Sam3DBodyModel(
        weights_path=checkpoint,
        device="cuda",
        adapter=Sam3DBodyUpstreamAdapter.from_repository_root(repo),
    )
    session = model.load()

    assert session.predict_many([]) == []


def test_result_metadata_records_initial_schema_and_coordinate_labels(tmp_path: Path) -> None:
    repo = tmp_path / "sam-3d-body"
    _write_fake_upstream(repo)
    checkpoint = tmp_path / "model.ckpt"
    image_path = tmp_path / "image.png"
    checkpoint.write_text("fake")
    image_path.write_bytes(b"fake")
    model = Sam3DBodyModel(
        weights_path=checkpoint,
        device="cuda",
        adapter=Sam3DBodyUpstreamAdapter.from_repository_root(repo),
    )

    result = model.predict(image_path)

    assert result.metadata.extra["output_schema_version"] == "0.1"
    assert result.metadata.extra["coordinate_conventions"] == {
        "bbox_xyxy": "upstream_output_image_xyxy_pixels",
        "vertices": "upstream_model_3d_coordinates_unverified",
        "joints": "upstream_model_3d_coordinates_unverified",
        "faces": "mesh_topology_indices",
    }
