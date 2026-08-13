"""诊断 mAP=0: 用当前 last.pt (EMA 权重) 在验证集上对比不同 conf 阈值的 mAP。

若 conf=0.001 时 mAP 明显 >0 且预测框置信度在缓慢抬升, 说明模型正常学习中
(只是置信度还没越过 0.25 阈值); 若 conf=0.001 仍为 0, 需排查坐标/标签/验证逻辑。
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
from src.model import build_model, load_checkpoint


def main():
    cfg = load_config()
    root = resolve_data_root(cfg)
    device = torch.device("cpu")  # 用 CPU, 不影响后台 GPU 训练

    model = build_model(cfg, pretrained="")
    model = load_checkpoint(model, "runs/train/last.pt", device)
    model.eval()

    # 权重统计 (确认 EMA 权重正常, 非全 0/NaN)
    w = next(model.parameters())
    print(f"[权重] last.pt 首层权重: mean={w.mean().item():.4f} std={w.std().item():.4f} "
          f"min={w.min().item():.4f} max={w.max().item():.4f}")

    val_full = TriModalDataset(root, imgsz=640, max_depth=cfg["depth"]["max_val"], train=False)
    full = TriModalDataset(root, imgsz=640, max_depth=cfg["depth"]["max_val"], train=True)
    n = len(full)
    val_ratio = cfg["data"].get("val_ratio", 0.2)
    n_val = max(1, int(round(n * val_ratio)))
    indices = torch.randperm(n, generator=torch.Generator().manual_seed(42)).tolist()
    val_idx = indices[:n_val]

    # 诊断取前 60 个验证样本
    val_idx = val_idx[:60]

    for conf in [0.25, 0.01]:
        preds_per_image, gts_per_image = [], []
        n_boxes = 0
        max_conf_seen = 0.0
        for idx in val_idx:
            s = val_full[idx]
            preds = predict_one(model, s["img"], s["meta"], conf_thres=conf)
            labels = val_full._load_labels(val_full.samples[idx])
            gts = [(int(l[0]), l[1], l[2], l[3], l[4]) for l in labels]
            preds_per_image.append(preds)
            gts_per_image.append(gts)
            n_boxes += len(preds)
            if preds:
                max_conf_seen = max(max_conf_seen, max(p[5] for p in preds))
        m = compute_map(preds_per_image, gts_per_image)
        print(f"[诊断] conf={conf:<6} mAP@50-95={m['map50_95']:.4f}  "
              f"mAP@50={m.get('map50', 0):.4f}  总预测框={n_boxes}  最大置信度={max_conf_seen:.4f}")


if __name__ == "__main__":
    main()
