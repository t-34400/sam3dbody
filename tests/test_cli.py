import json
from pathlib import Path

import pytest

from sam3dbody.cli import main


def test_cli_inspect_deps_outputs_json(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    upstream_root = tmp_path / "sam-3d-body"
    upstream_root.mkdir()
    (upstream_root / "sample.py").write_text("import torch\nimport cv2\n")

    exit_code = main(["inspect-deps", "--upstream-root", str(upstream_root), "--json"])

    payload = json.loads(capsys.readouterr().out)
    assert exit_code == 0
    assert payload["repository_root"] == str(upstream_root)
    assert payload["external_modules"] == ["cv2", "torch"]


def test_pyproject_exposes_valid_cli_entrypoint() -> None:
    pyproject = Path("pyproject.toml").read_text()

    assert "[project.scripts]" in pyproject
    assert 'sam3dbody = "sam3dbody.cli:main"' in pyproject


def test_cli_infer_writes_json_output(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    image = tmp_path / "image.png"
    image.write_bytes(b"not a real image")
    weights = tmp_path / "model.ckpt"
    weights.write_bytes(b"checkpoint")
    output = tmp_path / "nested" / "result.json"
    bboxes_json = tmp_path / "bboxes.json"
    bboxes_json.write_text(json.dumps({"bboxes": [[1, 2, 3, 4]]}))
    calls: dict[str, object] = {}

    class FakeResult:
        def to_dict(self) -> dict[str, object]:
            return {
                "bodies": [
                    {
                        "score": None,
                        "bbox_xyxy": (1.0, 2.0, 3.0, 4.0),
                        "vertices": FakeArray([[0.0, 1.0, 2.0]]),
                    }
                ],
                "metadata": {"device": "cuda", "extra": {}},
            }

    class FakeArray:
        def __init__(self, values: list[list[float]]) -> None:
            self._values = values

        def tolist(self) -> list[list[float]]:
            return self._values

    class FakeSession:
        def predict(self, image_arg: Path, **kwargs: object) -> FakeResult:
            calls["image"] = image_arg
            calls["predict_kwargs"] = kwargs
            return FakeResult()

    class FakeModel:
        def __init__(self, weights_arg: Path, *, device: str, config: dict[str, object] | None) -> None:
            calls["weights"] = weights_arg
            calls["device"] = device
            calls["config"] = config

        def load(self) -> FakeSession:
            calls["loaded"] = True
            return FakeSession()

    def fake_from_pretrained(
        weights_arg: Path,
        *,
        device: str | None = None,
        config: dict[str, object] | None = None,
    ) -> FakeModel:
        return FakeModel(weights_arg, device=device or "", config=config)

    monkeypatch.setattr("sam3dbody.cli.Sam3DBodyModel.from_pretrained", fake_from_pretrained)

    exit_code = main(
        [
            "infer",
            str(image),
            "--weights",
            str(weights),
            "--output",
            str(output),
            "--mhr-path",
            str(tmp_path / "mhr.ckpt"),
            "--bboxes-json",
            str(bboxes_json),
            "--bbox-thr",
            "0.7",
            "--nms-thr",
            "0.2",
            "--inference-type",
            "body",
        ]
    )

    payload = json.loads(output.read_text())
    assert exit_code == 0
    assert calls["weights"] == weights
    assert calls["device"] == "cuda"
    assert calls["config"] == {"mhr_path": tmp_path / "mhr.ckpt"}
    assert calls["image"] == image
    assert calls["predict_kwargs"] == {
        "bboxes": [[1, 2, 3, 4]],
        "masks": None,
        "cam_int": None,
        "bbox_thr": 0.7,
        "nms_thr": 0.2,
        "inference_type": "body",
    }
    assert payload["bodies"][0]["bbox_xyxy"] == [1.0, 2.0, 3.0, 4.0]
    assert payload["bodies"][0]["vertices"] == [[0.0, 1.0, 2.0]]


def test_cli_infer_prints_json_to_stdout(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    image = tmp_path / "image.png"
    image.write_bytes(b"not a real image")
    weights = tmp_path / "model.ckpt"
    weights.write_bytes(b"checkpoint")

    class FakeResult:
        def to_dict(self) -> dict[str, object]:
            return {"bodies": [], "metadata": {"device": "cuda", "extra": {}}}

    class FakeSession:
        def predict(self, image_arg: Path, **kwargs: object) -> FakeResult:
            return FakeResult()

    class FakeModel:
        def load(self) -> FakeSession:
            return FakeSession()

    monkeypatch.setattr(
        "sam3dbody.cli.Sam3DBodyModel.from_pretrained",
        lambda *args, **kwargs: FakeModel(),
    )

    exit_code = main(["infer", str(image), "--weights", str(weights)])

    payload = json.loads(capsys.readouterr().out)
    assert exit_code == 0
    assert payload == {"bodies": [], "metadata": {"device": "cuda", "extra": {}}}


def test_cli_infer_loads_cam_int_json_as_torch_tensor(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    image = tmp_path / "image.png"
    image.write_bytes(b"not a real image")
    weights = tmp_path / "model.ckpt"
    weights.write_bytes(b"checkpoint")
    cam_int_json = tmp_path / "cam_int.json"
    cam_int_json.write_text(json.dumps({"cam_int": [[1, 0, 2], [0, 1, 3], [0, 0, 1]]}))
    calls: dict[str, object] = {}

    class FakeTensor:
        def __init__(self, values: object, dtype: object) -> None:
            self.values = values
            self.dtype = dtype

        def tolist(self) -> object:
            return self.values

    class FakeTorch:
        float32 = "float32"

        @staticmethod
        def as_tensor(values: object, *, dtype: object) -> FakeTensor:
            return FakeTensor(values, dtype)

    class FakeResult:
        def to_dict(self) -> dict[str, object]:
            return {"bodies": [], "metadata": {}}

    class FakeSession:
        def predict(self, image_arg: Path, **kwargs: object) -> FakeResult:
            calls.update(kwargs)
            return FakeResult()

    class FakeModel:
        def load(self) -> FakeSession:
            return FakeSession()

    monkeypatch.setitem(__import__("sys").modules, "torch", FakeTorch)
    monkeypatch.setattr(
        "sam3dbody.cli.Sam3DBodyModel.from_pretrained",
        lambda *args, **kwargs: FakeModel(),
    )

    exit_code = main(["infer", str(image), "--weights", str(weights), "--cam-int-json", str(cam_int_json)])

    assert exit_code == 0
    assert isinstance(calls["cam_int"], FakeTensor)
    assert calls["cam_int"].tolist() == [[1, 0, 2], [0, 1, 3], [0, 0, 1]]
    assert calls["cam_int"].dtype == "float32"


def test_cli_infer_rejects_bboxes_json_object_without_bboxes_key(tmp_path: Path) -> None:
    from sam3dbody.cli import _load_bboxes_json

    bboxes_json = tmp_path / "bboxes.json"
    bboxes_json.write_text(json.dumps({"boxes": []}))

    with pytest.raises(ValueError, match="bboxes"):
        _load_bboxes_json(bboxes_json)


def test_cli_infer_loads_masks_json(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    image = tmp_path / "image.png"
    image.write_bytes(b"not a real image")
    weights = tmp_path / "model.ckpt"
    weights.write_bytes(b"checkpoint")
    masks_json = tmp_path / "masks.json"
    masks_json.write_text(json.dumps({"masks": [[[0, 1], [1, 0]]]}))
    calls: dict[str, object] = {}

    class FakeResult:
        def to_dict(self) -> dict[str, object]:
            return {"bodies": [], "metadata": {}}

    class FakeSession:
        def predict(self, image_arg: Path, **kwargs: object) -> FakeResult:
            calls.update(kwargs)
            return FakeResult()

    class FakeModel:
        def load(self) -> FakeSession:
            return FakeSession()

    monkeypatch.setattr(
        "sam3dbody.cli.Sam3DBodyModel.from_pretrained",
        lambda *args, **kwargs: FakeModel(),
    )

    exit_code = main(["infer", str(image), "--weights", str(weights), "--masks-json", str(masks_json)])

    assert exit_code == 0
    assert calls["masks"] == [[[0, 1], [1, 0]]]


def test_cli_infer_rejects_masks_json_object_without_masks_key(tmp_path: Path) -> None:
    from sam3dbody.cli import _load_masks_json

    masks_json = tmp_path / "masks.json"
    masks_json.write_text(json.dumps({"mask": []}))

    with pytest.raises(ValueError, match="masks"):
        _load_masks_json(masks_json)
