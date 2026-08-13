"""诊断: 检查 last.pt 模型原始输出的 scores 分布, 定位 mAP=0 根因。"""
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

    model = build_model(cfg, pretrained="")
    model = load_checkpoint(model, "runs/train/last.pt", device)
    model.eval()

    val_full = TriModalDataset(root, imgsz=640, max_depth=cfg["depth"]["max_val"], train=False)

    for i in [0, 5, 20]:
        s = val_full[i]
        boxes, scores = predict_raw(model, s["img"])
        print(f"[样本 {i}] boxes({boxes.shape}) 范围 [{boxes.min().item():.1f}, {boxes.max().item():.1f}]")
        print(f"          scores({scores.shape})  min={scores.min().item():.4f} "
              f"max={scores.max().item():.4f} mean={scores.mean().item():.4f}")
        max_per_cls = scores.max(dim=0).values
        print(f"          每类最大分: {[f'{v:.3f}' for v in max_per_cls.tolist()]}")


if __name__ == "__main__":
    main()
