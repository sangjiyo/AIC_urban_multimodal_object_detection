"""诊断: 640 下 no-TTA vs 水平翻转 TTA 的 val mAP 收益。

用 runs/train_s/best.pt (36.4 分权重), 200 个 val 样本, 判断 TTA 是否值得在提交推理中启用。
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
    cfg = load_config("configs/aic_trimodal_s.yaml")
    root = resolve_data_root(cfg)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = build_fusion_model(cfg, pretrained="")
    model = load_checkpoint(model, "runs/train_s/best.pt", device)
    model.eval()

    full = TriModalDataset(root, imgsz=640, max_depth=cfg["depth"]["max_val"],
                           train=True, seed=cfg["data"].get("seed", 42))
    n = len(full)
    n_val = max(1, int(round(n * cfg["data"].get("val_ratio", 0.2))))
    indices = torch.randperm(n, generator=torch.Generator().manual_seed(cfg["data"].get("seed", 42))).tolist()
    val_idx = indices[:n_val][:200]

    ds = TriModalDataset(root, imgsz=640, max_depth=cfg["depth"]["max_val"], train=False)

    gts_per_image = []
    for idx in val_idx:
        labels = ds._load_labels(ds.samples[idx])
        gts_per_image.append([(int(l[0]), l[1], l[2], l[3], l[4]) for l in labels])

    for label, pred_fn in (("no-TTA", predict_one), ("TTA", predict_one_tta)):
        preds_per_image = []
        for idx in val_idx:
            sample = ds[idx]
            preds = pred_fn(model, sample["img"].to(device), sample["meta"], 0.25, 0.7)
            preds_per_image.append(preds)
        m = compute_map(preds_per_image, gts_per_image)
        print(f"[{label:6s}] mAP@50-95 = {m['map50_95']:.4f}  (200 样本, 640)")


if __name__ == "__main__":
    main()
