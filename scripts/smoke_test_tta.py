"""Smoke test: 验证 predict_one / predict_one_tta 代码路径 (CPU, 随机权重)。"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from src.config import load_config
from src.infer import predict_one, predict_one_tta
from src.model import build_model


def main():
    cfg = load_config()
    torch.manual_seed(0)
    model = build_model(cfg, pretrained="").cpu()  # 不加载权重, CPU 上验证逻辑
    model.eval()

    img = torch.randn(5, 640, 640)
    meta = (1.0, 0, 0, 640, 640)  # 无 letterbox, r=1, 无 pad

    preds = predict_one(model, img, meta)
    preds_tta = predict_one_tta(model, img, meta)
    print(f"[推理] predict_one 输出 {len(preds)} 个框")
    print(f"[推理] predict_one_tta 输出 {len(preds_tta)} 个框")

    for name, p in [("one", preds), ("tta", preds_tta)]:
        for item in p:
            cls, cx, cy, w, h, conf = item
            assert isinstance(cls, int) and 0 <= cls < 12, f"cls 非法: {item}"
            assert 0.0 <= cx <= 1.0 and 0.0 <= cy <= 1.0, f"坐标越界: {item}"
            assert w > 0 and h > 0, f"宽高非法: {item}"
        print(f"[推理] {name}: 输出格式合法 (cls/cx/cy/w/h/conf)")

    print("[TTA] 验证通过")


if __name__ == "__main__":
    main()
