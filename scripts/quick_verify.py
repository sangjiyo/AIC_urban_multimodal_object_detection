"""快速验证修复后的推理: 加载 last.pt, 在验证集前 N 张上算 mAP。"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import load_config, resolve_data_root
from src.dataset import TriModalDataset
from src.infer import predict_one
from src.metrics import compute_map
from src.model import build_model, load_checkpoint


def main():
    cfg = load_config()
    root = resolve_data_root(cfg)
    device = torch.device("cuda:0")

    model = build_model(cfg, pretrained="")
    model = load_checkpoint(model, "runs/train/last.pt", device)
    model.eval()

    ds = TriModalDataset(root, imgsz=640, max_depth=cfg["depth"]["max_val"], train=False)
    n = len(ds)
    val_ratio = cfg["data"].get("val_ratio", 0.2)
    n_val = max(1, int(round(n * val_ratio)))
    indices = torch.randperm(n, generator=torch.Generator().manual_seed(cfg["data"].get("seed", 42))).tolist()
    val_idx = indices[:n_val]

    N = 100
    preds_per_image, gts_per_image = [], []
    total_boxes = 0
    for idx in val_idx[:N]:
        sample = ds[idx]
        preds = predict_one(model, sample["img"].to(device), sample["meta"], 0.25, 0.7)
        labels = ds._load_labels(ds.samples[idx])
        gts = [(int(l[0]), l[1], l[2], l[3], l[4]) for l in labels]
        preds_per_image.append(preds)
        gts_per_image.append(gts)
        total_boxes += len(preds)

    print(f"total boxes over {N} imgs: {total_boxes}")
    if preds_per_image[0]:
        print("sample pred (cls, cx, cy, w, h, conf):", preds_per_image[0][:3])

    m = compute_map(preds_per_image, gts_per_image)
    print(f"mAP@50-95 ({N} imgs): {m['map50_95']:.4f}, map50: {m['map50']:.4f}")


if __name__ == "__main__":
    main()
