"""postprocess.py 테스트 파일

테스트 목적:
- score threshold보다 낮은 예측 제거
- score 기준 내림차순 정렬
- max_detections 개수 제한
- 빈 예측 결과의 shape / dtype 유지
- 프로젝트 공용 dummy prediction fixture와의 호환성 확인
"""

import torch

from src.engine.postprocess import (
  PostprocessConfig,
  postprocess_prediction,
  postprocess_predictions,
  postprocess_raw_outputs,
)
from tests.dummy_model import make_dummy_pred_dict, make_dummy_raw_output

def make_mock_prediction() -> dict:
 """후처리 규칙 명확히 롹인하기 위한 가짜 예측 결과를 생성
  scores를 [0.9, 0.2, 0.7]로 고정해서 threshold filtering과
    score 정렬이 제대로 동작하는지 확인한다.
 """
 return {
        "image_id": 0,
        "boxes": torch.tensor(
            [
                [10.0, 20.0, 50.0, 80.0],
                [15.0, 25.0, 55.0, 85.0],
                [100.0, 120.0, 180.0, 220.0],
            ],
            dtype=torch.float32,
        ),
        "labels": torch.tensor([1, 1, 2], dtype=torch.int64),
        "scores": torch.tensor([0.9, 0.2, 0.7], dtype=torch.float32),
    }

def test_postprocess_prediction() -> None:
    """이미지 1장 prediction의 필터링 / 정렬 / 개수 제한을 확인."""

    config = PostprocessConfig(
        conf_threshold=0.25,
        iou_threshold=0.7,
        max_detections=2,
    )

    processed = postprocess_prediction(make_mock_prediction(), config)

    assert processed["image_id"] == 0
    assert processed["boxes"].shape == (2, 4)
    assert processed["labels"].shape == (2,)
    assert processed["scores"].shape == (2,)

    # score 기준 내림차순 정렬 확인
    assert processed["scores"][0] >= processed["scores"][1]

    # score 0.2 예측은 threshold보다 낮아서 제거되어야 한다.
    assert torch.all(processed["scores"] >= 0.25)

    # interfaces.md 형식 확인
    assert processed["boxes"].dtype == torch.float32
    assert processed["labels"].dtype == torch.int64
    assert processed["scores"].dtype == torch.float32

def test_postprocess_predictions() -> None:
    """여러 이미지 prediction을 list 단위로 후처리하는 helper를 확인."""

    config = PostprocessConfig(conf_threshold=0.25, max_detections=2)

    processed_list = postprocess_predictions([make_mock_prediction()], config)

    assert len(processed_list) == 1
    assert processed_list[0]["image_id"] == 0
    assert processed_list[0]["boxes"].shape == (2, 4)

def test_postprocess_empty_prediction() -> None:
    """threshold를 통과한 예측이 없을 때도 빈 prediction 형식이 유지되는지 확인"""

    prediction = {
        "image_id": 1,
        "boxes": torch.tensor([[10.0, 20.0, 50.0, 80.0]], dtype=torch.float32),
        "labels": torch.tensor([1], dtype=torch.int64),
        "scores": torch.tensor([0.1], dtype=torch.float32),
    }

    processed = postprocess_prediction(
        prediction,
        PostprocessConfig(conf_threshold=0.25),
    )

    assert processed["image_id"] == 1
    assert processed["boxes"].shape == (0, 4)
    assert processed["labels"].shape == (0,)
    assert processed["scores"].shape == (0,)
    assert processed["boxes"].dtype == torch.float32
    assert processed["labels"].dtype == torch.int64
    assert processed["scores"].dtype == torch.float32

def test_postprocess_with_project_dummy_prediction() -> None:
    """공용 dummy prediction fixture도 postprocess.py 입력 형식과 맞는지 확인한다."""

    predictions = make_dummy_pred_dict(
        batch_size=1,
        num_classes=10,
        num_detections=3,
        start_image_id=2,
    )
    prediction = predictions[0]

    processed = postprocess_prediction(
        prediction,
        PostprocessConfig(conf_threshold=0.0, max_detections=4),
    )

    assert processed["image_id"] == 2
    assert processed["boxes"].shape[1] == 4
    assert processed["labels"].ndim == 1
    assert processed["scores"].ndim == 1
    assert processed["boxes"].dtype == torch.float32
    assert processed["labels"].dtype == torch.int64
    assert processed["scores"].dtype == torch.float32

def test_postprocess_raw_outputs_from_dummy_model_format() -> None:
    """DummyModel raw output을 Prediction 리스트로 변환할 수 있는지 확인"""

    raw_outputs = (
        make_dummy_raw_output(
            batch_size=2,
            num_classes=10,
        ),
    )

    processed_list = postprocess_raw_outputs(
        raw_outputs=raw_outputs,
        image_ids=[10, 11],
        config=PostprocessConfig(
            conf_threshold=0.99,
            iou_threshold=0.7,
            max_detections=4,
        ),
    )

    assert len(processed_list) == 2
    assert processed_list[0]["image_id"] == 10
    assert processed_list[1]["image_id"] == 11

    for processed in processed_list:
        assert processed["boxes"].shape[1] == 4
        assert processed["labels"].ndim == 1
        assert processed["scores"].ndim == 1
        assert processed["boxes"].shape[0] <= 4
        assert processed["labels"].shape[0] <= 4
        assert processed["scores"].shape[0] <= 4
        assert processed["boxes"].dtype == torch.float32
        assert processed["labels"].dtype == torch.int64
        assert processed["scores"].dtype == torch.float32


# --- class_probs: NMS가 고른 앵커의 클래스 점수를 합 1로 정규화 ---------------------

def make_raw_output_with_known_anchors() -> torch.Tensor:
    """앵커 5개, 클래스 3개짜리 raw output [4 + nc, num_anchors]. 열 하나가 앵커 하나다.

    앵커 0: 박스 A, 클래스 0 (0.90)
    앵커 1: 박스 A와 거의 같은 위치 → NMS에서 제거
    앵커 2: 박스 B, 클래스 0 (0.60) — A와 같은 클래스지만 클래스 1 점수도 높다
    앵커 3: 박스 C, 클래스 2 (0.70)
    앵커 4: 모든 점수가 낮아 conf threshold에서 제거
    """
    boxes_cxcywh = torch.tensor(
        [
            [50.0, 50.0, 20.0, 20.0],
            [51.0, 50.0, 20.0, 20.0],
            [200.0, 200.0, 20.0, 20.0],
            [400.0, 100.0, 30.0, 30.0],
            [300.0, 300.0, 10.0, 10.0],
        ]
    )
    class_scores = torch.tensor(
        [
            [0.90, 0.05, 0.05],
            [0.80, 0.10, 0.05],
            [0.60, 0.35, 0.05],
            [0.02, 0.03, 0.70],
            [0.01, 0.01, 0.01],
        ]
    )
    return torch.cat([boxes_cxcywh, class_scores], dim=1).T.contiguous()


def test_class_probs_come_from_each_box_own_anchor() -> None:
    """같은 클래스 박스가 둘이어도 각 박스가 자기 앵커의 클래스 분포를 받는지 확인한다.

    예전 구현은 "같은 클래스 중 점수가 가장 높은 앵커"를 골라서
    박스 B(0.60)가 박스 A(0.90)의 분포를 받았다.
    """
    raw = make_raw_output_with_known_anchors()
    config = PostprocessConfig(conf_threshold=0.25, iou_threshold=0.7, max_detections=4)

    (prediction,) = postprocess_raw_outputs((raw.unsqueeze(0),), image_ids=[7], config=config)

    assert prediction["image_id"] == 7
    assert prediction["boxes"].shape == (3, 4)

    # score → 원래 앵커의 클래스 점수 (합 1로 정규화한 값이 기대값)
    expected_by_score = {
        0.90: torch.tensor([0.90, 0.05, 0.05]),
        0.60: torch.tensor([0.60, 0.35, 0.05]),
        0.70: torch.tensor([0.02, 0.03, 0.70]),
    }
    for score, probs in zip(prediction["scores"].tolist(), prediction["class_probs"]):
        expected = expected_by_score[round(score, 2)]
        assert torch.allclose(probs, expected / expected.sum(), atol=1e-6)


def test_class_probs_are_normalized_and_agree_with_labels() -> None:
    """랜덤 raw output에서도 class_probs가 [N, nc], 행 합 1, argmax == label인지 확인한다."""
    torch.manual_seed(0)
    num_classes = 10
    raw_outputs = (make_dummy_raw_output(batch_size=2, num_classes=num_classes),)

    processed_list = postprocess_raw_outputs(
        raw_outputs=raw_outputs,
        image_ids=[0, 1],
        config=PostprocessConfig(conf_threshold=0.9, iou_threshold=0.7, max_detections=4),
    )

    for processed in processed_list:
        class_probs = processed["class_probs"]
        assert class_probs.shape == (processed["labels"].shape[0], num_classes)
        assert class_probs.dtype == torch.float32
        assert torch.allclose(class_probs.sum(dim=1), torch.ones(len(class_probs)), atol=1e-5)
        assert torch.equal(class_probs.argmax(dim=1), processed["labels"])


def test_postprocess_raw_outputs_does_not_modify_input() -> None:
    """NMS가 박스 좌표를 제자리에서 바꾸지만, 호출한 쪽의 raw output은 그대로여야 한다."""
    raw = make_raw_output_with_known_anchors().unsqueeze(0)
    before = raw.clone()

    postprocess_raw_outputs((raw,), image_ids=[0], config=PostprocessConfig(conf_threshold=0.25))

    assert torch.equal(raw, before)


def test_empty_raw_prediction_keeps_class_probs_shape() -> None:
    """threshold를 넘는 박스가 없어도 class_probs는 [0, nc] 형식을 유지한다."""
    raw = make_raw_output_with_known_anchors().unsqueeze(0)

    (prediction,) = postprocess_raw_outputs(
        (raw,), image_ids=[0], config=PostprocessConfig(conf_threshold=0.95)
    )

    assert prediction["boxes"].shape == (0, 4)
    assert prediction["class_probs"].shape == (0, 3)