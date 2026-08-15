"""诊断: 对比不同推理设置下的 val mAP, 定位 7 分暴跌根因。

对比 4 种设置:
  (1) no-TTA + imgsz 640   (训练时 validate 同款, 应 ≈ 0.2453)
  (2) TTA    + imgsz 640
  (3) no-TTA + imgsz 1280
  (4) TTA    + imgsz 1280   (提交同款)

若 (1)≈0.2453 而 (4) 暴跌, 则 TTA 或 1280 推理路径有 bug。
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from src.config import load_config, resolve_data_root
from src.dataset import TriModalDataset
from src.infer import predict_one, predict_one_tta
from src.metrics import compute_map
from src.model import build_fusion_model, load_checkpoint


def main():
    cfg = load_config()
    root = resolve_data_root(cfg)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = build_fusion_model(cfg, pretrained="")
    model = load_checkpoint(model, "runs/train/best.pt", device)
    model.eval()

    # 复现训练时的 val 划分 (seed 42, val_ratio 0.2)
    full = TriModalDataset(root, imgsz=640, max_depth=cfg["depth"]["max_val"],
                           train=True, seed=cfg["data"].get("seed", 42))
    n = len(full)
    n_val = max(1, int(round(n * cfg["data"].get("val_ratio", 0.2))))
    indices = torch.randperm(n, generator=torch.Generator().manual_seed(cfg["data"].get("seed", 42))).tolist()
    val_idx = indices[:n_val]

    # 子集加速: 取前 100 个 val 样本即可分辨差异
    val_idx = val_idx[:100]

    val640 = TriModalDataset(root, imgsz=640, max_depth=cfg["depth"]["max_val"], train=False)
    val1280 = TriModalDataset(root, imgsz=1280, max_depth=cfg["depth"]["max_val"], train=False)

    def eval_setting(dataset, pred_fn, label):
        preds_per_image, gts_per_image = [], []
        for idx in val_idx:
            sample = dataset[idx]
            preds = pred_fn(model, sample["img"].to(device), sample["meta"], 0.25, 0.7)
            labels = dataset._load_labels(dataset.samples[idx])
            gts = [(int(l[0]), l[1], l[2], l[3], l[4]) for l in labels]
            preds_per_image.append(preds)
            gts_per_image.append(gts)
        m = compute_map(preds_per_image, gts_per_image)
        print(f"[{label}] mAP@50-95 = {m['map50_95']:.4f}  (100 样本)")

    eval_setting(val640, lambda m, i, meta, c, u: predict_one(m, i, meta, c, u), "no-TTA 640")
    eval_setting(val640, lambda m, i, meta, c, u: predict_one_tta(m, i, meta, c, u), "TTA    640")
    eval_setting(val1280, lambda m, i, meta, c, u: predict_one(m, i, meta, c, u), "no-TTA 1280")
    eval_setting(val1280, lambda m, i, meta, c, u: predict_one_tta(m, i, meta, c, u), "TTA    1280")


if __name__ == "__main__":
    main()
