"""对比: 纯 COCO 预训练模型 vs EMA 3-epoch 模型的输出 scores 分布。

若 COCO 预训练 scores 正常 (~0.5) 而 EMA 3-epoch 极低 (~0.001),
说明训练把 logits 推负了 (早期背景抑制, 正常); 若两者都极低, 则有构建/加载 bug。
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from src.config import load_config, resolve_data_root
from src.dataset import TriModalDataset
from src.infer import predict_raw
from src.model import build_model, load_checkpoint


def main():
    cfg = load_config()
    root = resolve_data_root(cfg)
    device = torch.device("cpu")

    val_full = TriModalDataset(root, imgsz=640, max_depth=cfg["depth"]["max_val"], train=False)
    s = val_full[0]

    # 1. 纯 COCO 预训练 (未训练, 加载 yolov8s.pt)
    model_init = build_model(cfg).cpu().eval()
    _, scores0 = predict_raw(model_init, s["img"])
    print(f"[COCO预训练] scores max={scores0.max().item():.4f} mean={scores0.mean().item():.4f}")

    # 2. last.pt (EMA 训练 3 epoch)
    model_ema = build_model(cfg, pretrained="").cpu()
    model_ema = load_checkpoint(model_ema, "runs/train/last.pt", device).eval()
    _, scores1 = predict_raw(model_ema, s["img"])
    print(f"[EMA 3epoch]  scores max={scores1.max().item():.4f} mean={scores1.mean().item():.4f}")


if __name__ == "__main__":
    main()
