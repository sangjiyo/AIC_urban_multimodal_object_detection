"""诊断: 用 640 训练的原版 best.pt 直接在不同 imgsz 下推理 val, 量化尺度差异损失。

对比:
  - imgsz 640 (训练同款, 应 ≈ 0.2562)
  - imgsz 832
  - imgsz 960
  - imgsz 1280 (已知崩 ~0.04)

据此判断「高分辨率精修」该从哪个尺度切入、损失有多大。
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from src.config import load_config, resolve_data_root
from src.dataset import TriModalDataset
from src.infer import predict_one
from src.metrics import compute_map
from src.model import build_fusion_model, load_checkpoint


def main():
    cfg = load_config("configs/aic_trimodal_s.yaml")
    root = resolve_data_root(cfg)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = build_fusion_model(cfg, pretrained="")
    model = load_checkpoint(model, "runs/train_s/best.pt", device)
    model.eval()

    # 复现训练 val 划分 (seed 42, val_ratio 0.2), 取前 200 张加速
    full = TriModalDataset(root, imgsz=640, max_depth=cfg["depth"]["max_val"],
                           train=True, seed=cfg["data"].get("seed", 42))
    n = len(full)
    n_val = max(1, int(round(n * cfg["data"].get("val_ratio", 0.2))))
    indices = torch.randperm(n, generator=torch.Generator().manual_seed(cfg["data"].get("seed", 42))).tolist()
    val_idx = indices[:n_val][:200]

    for imgsz in (640, 832, 960, 1280):
        ds = TriModalDataset(root, imgsz=imgsz, max_depth=cfg["depth"]["max_val"], train=False)
        preds_per_image, gts_per_image = [], []
        for idx in val_idx:
            sample = ds[idx]
            preds = predict_one(model, sample["img"].to(device), sample["meta"], 0.25, 0.7)
            labels = ds._load_labels(ds.samples[idx])
            gts = [(int(l[0]), l[1], l[2], l[3], l[4]) for l in labels]
            preds_per_image.append(preds)
            gts_per_image.append(gts)
        m = compute_map(preds_per_image, gts_per_image)
        print(f"[imgsz {imgsz:4d}] mAP@50-95 = {m['map50_95']:.4f}  (200 样本, 640 训练权重直推)")


if __name__ == "__main__":
    main()
