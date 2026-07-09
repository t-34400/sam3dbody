from pathlib import Path

import pytest

from sam3dbody import (
    Sam3DBodyMetadata,
    Sam3DBodyModel,
    Sam3DBodyPrediction,
    Sam3DBodyResult,
)


def test_public_api_imports() -> None:
    assert Sam3DBodyModel.__name__ == "Sam3DBodyModel"
    assert Sam3DBodyResult.__name__ == "Sam3DBodyResult"


def test_model_from_pretrained_preserves_explicit_settings() -> None:
    model = Sam3DBodyModel.from_pretrained(
        weights_path="weights/model.pt",
        device="cpu",
        config={"threshold": 0.5},
    )

    assert model.weights_path == Path("weights/model.pt")
    assert model.device == "cpu"
    assert model.config == {"threshold": 0.5}


def test_result_to_dict_uses_wrapper_owned_schema() -> None:
    result = Sam3DBodyResult(
        bodies=[Sam3DBodyPrediction(score=0.9, bbox_xyxy=(1.0, 2.0, 3.0, 4.0))],
        metadata=Sam3DBodyMetadata(model_name="sam-3d-body", device="cpu"),
    )

    assert result.to_dict() == {
        "bodies": [
            {
                "score": 0.9,
                "bbox_xyxy": (1.0, 2.0, 3.0, 4.0),
                "vertices": None,
                "faces": None,
                "joints": None,
                "extra": {},
            }
        ],
        "metadata": {
            "model_name": "sam-3d-body",
            "device": "cpu",
            "extra": {},
        },
    }


def test_predict_requires_weights_before_upstream_inference() -> None:
    model = Sam3DBodyModel.from_pretrained(device="cpu")

    with pytest.raises(ValueError, match="weights_path is required"):
        model.predict(object())


def test_python310_compatible_typing() -> None:
    model_source = Path("src/sam3dbody/model.py").read_text()

    assert "typing import Any, Self" not in model_source
    assert "typing import Self" not in model_source
    assert "-> Self" not in model_source


def test_model_uses_upstream_adapter_boundary() -> None:
    model = Sam3DBodyModel.from_pretrained(device="cpu")

    assert model.adapter is not None
    assert model.adapter.repository.root.as_posix().endswith("third_party/sam-3d-body")
    assert model.adapter.repository.exists


def test_public_model_does_not_import_third_party_directly() -> None:
    model_source = Path("src/sam3dbody/model.py").read_text()

    assert "third_party" not in model_source
    assert "sam-3d-body" not in model_source
