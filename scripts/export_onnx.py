#!/usr/bin/env python3
"""학습한 체크포인트(.pt)를 서빙용 ONNX로 내보낸다.

    python scripts/export_onnx.py --config configs/yolo11x_1024.yaml \\
        --checkpoint outputs/checkpoints/best_model.pt --use-ema --output outputs/export/model.onnx

출력 ONNX 형식 (pill-detect-serving이 그대로 읽는다):
    입력  images  [1, 3, S, S]  RGB, 0~1 (학습과 같은 리사이즈)
    출력  output0 [1, 4 + nc, A] cx, cy, w, h (입력 픽셀) + sigmoid 클래스 점수

Ultralytics exporter(v8.4)와 같은 방식으로 Conv+BN을 합치고 Detect head를 export 모드로 바꾼 뒤
TorchScript 기반 exporter(dynamo=False)로 내보낸다. 클래스 매핑 파일도 옆에 복사한다.
onnxruntime이 설치돼 있으면 같은 입력에서 PyTorch 출력과 ONNX 출력의 차이를 확인한다.
"""

from __future__ import annotations

import argparse
import copy
import shutil
import sys
from pathlib import Path

import torch
from torch import nn

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from ultralytics.nn.modules import Detect  # noqa: E402

from src.models.baseline import build_model  # noqa: E402
from src.utils.config import load_config  # noqa: E402


class RawOutput(nn.Module):
    """eval 출력 tuple에서 추론 텐서 [B, 4 + nc, A]만 꺼낸다."""

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.model(x)
        return out[0] if isinstance(out, (tuple, list)) else out


def load_trained_model(cfg: dict, checkpoint_path: Path, use_ema: bool) -> nn.Module:
    model = build_model(cfg["data"]["nc"], model_name=cfg["model"]["name"])
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    key = "ema_state" if use_ema and checkpoint.get("ema_state") is not None else "model_state"
    model.load_state_dict(checkpoint[key])
    print(f"가중치: {checkpoint_path.name} [{key}] (epoch {checkpoint.get('epoch')}, val mAP {checkpoint.get('val_mAP')})")
    return model.float().eval()


def prepare_for_export(model: nn.Module, img_size: int, max_det: int) -> nn.Module:
    """Ultralytics exporter와 같은 준비: 복사 → Conv+BN fuse → Detect head export 모드."""
    model = copy.deepcopy(model)
    for p in model.parameters():
        p.requires_grad = False
    model = model.fuse()
    for m in model.modules():
        if isinstance(m, Detect):
            m.dynamic = False
            m.export = True
            m.format = "onnx"
            anchors = sum(int(img_size / s) ** 2 for s in model.stride.tolist())
            m.max_det = min(max_det, anchors)
            m.xyxy = False  # cx, cy, w, h 그대로 (NMS는 서빙 쪽에서)
    return RawOutput(model).eval()


def verify(onnx_path: Path, reference: nn.Module, img_size: int) -> None:
    try:
        import onnxruntime as ort
    except ImportError:
        print("onnxruntime이 없어 출력 비교를 건너뜁니다 (pip install onnxruntime).")
        return

    x = torch.rand(1, 3, img_size, img_size)
    with torch.no_grad():
        expected = reference(x).numpy()
    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    actual = session.run(None, {session.get_inputs()[0].name: x.numpy()})[0]

    if actual.shape != expected.shape:
        raise SystemExit(f"출력 shape이 다릅니다: ONNX {actual.shape} vs PyTorch {expected.shape}")
    box_diff = float(abs(actual[:, :4] - expected[:, :4]).max())
    score_diff = float(abs(actual[:, 4:] - expected[:, 4:]).max())
    print(f"검증: 출력 {list(actual.shape)} · 최대 차이 박스 {box_diff:.2e}px, 점수 {score_diff:.2e}")
    if box_diff > 1e-1 or score_diff > 1e-3:
        raise SystemExit("ONNX 출력이 PyTorch와 너무 다릅니다. opset이나 모델 설정을 확인하세요.")


def main() -> None:
    parser = argparse.ArgumentParser(description="체크포인트 → 서빙용 ONNX")
    parser.add_argument("--config", default="configs/default.yaml", help="학습에 쓴 설정 파일")
    parser.add_argument("--checkpoint", type=Path, required=True, help="train.py가 저장한 .pt")
    parser.add_argument("--use-ema", action="store_true", help="EMA 가중치 사용 (ema_state)")
    parser.add_argument("--output", type=Path, default=Path("outputs/export/model.onnx"))
    parser.add_argument("--img-size", type=int, default=None, help="입력 크기 (기본: 설정의 train.img_size)")
    parser.add_argument("--opset", type=int, default=17)
    args = parser.parse_args()

    cfg = load_config(args.config)
    img_size = args.img_size or cfg["train"]["img_size"]
    max_det = cfg["postprocess"]["max_detections"]

    model = load_trained_model(cfg, args.checkpoint, args.use_ema)
    export_model = prepare_for_export(model, img_size, max_det)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    dummy = torch.zeros(1, 3, img_size, img_size)
    torch.onnx.export(
        export_model,
        dummy,
        str(args.output),
        opset_version=args.opset,
        input_names=["images"],
        output_names=["output0"],
        do_constant_folding=True,
        dynamo=False,
    )
    size_mb = args.output.stat().st_size / 1e6
    print(f"ONNX 저장: {args.output} ({size_mb:.1f} MB, 입력 {img_size}px, opset {args.opset})")

    mapping_src = Path(cfg["data"]["class_mapping"])
    if not mapping_src.is_absolute():
        mapping_src = ROOT / mapping_src
    mapping_dst = args.output.parent / "class_mapping.json"
    shutil.copyfile(mapping_src, mapping_dst)
    print(f"클래스 매핑 복사: {mapping_dst}")

    # 기준은 export 모드로 바꾸기 전, 학습·평가 때와 같은 eval 모델의 출력이다.
    verify(args.output, RawOutput(model).eval(), img_size)


if __name__ == "__main__":
    main()
