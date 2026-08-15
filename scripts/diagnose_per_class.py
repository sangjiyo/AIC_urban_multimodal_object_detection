"""per-class 诊断: 用 yolov8s best.pt 在完整验证集上评测每类 AP, 定位拖后腿的类别。

输出:
  - 总 mAP@50-95
  - 每类: GT 框数 / 预测框数 / AP@50-95 / AP@50
按 AP@50-95 升序排列 (最差在前), 便于识别困难类别与数据不平衡。
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
    names = cfg["names"]
    root = resolve_data_root(cfg)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = build_fusion_model(cfg, pretrained="")
    model = load_checkpoint(model, "runs/train_s/best.pt", device)
    model.eval()

    # 复现训练时的 val 划分 (seed 42, val_ratio 0.2)
    full = TriModalDataset(root, imgsz=640, max_depth=cfg["depth"]["max_val"],
                           train=True, seed=cfg["data"].get("seed", 42))
    n = len(full)
    n_val = max(1, int(round(n * cfg["data"].get("val_ratio", 0.2))))
    indices = torch.randperm(n, generator=torch.Generator().manual_seed(cfg["data"].get("seed", 42))).tolist()
    val_idx = indices[:n_val]

    val640 = TriModalDataset(root, imgsz=640, max_depth=cfg["depth"]["max_val"], train=False)

    preds_per_image, gts_per_image = [], []
    gt_count = {}   # 每类 GT 框数
    pred_count = {} # 每类预测框数 (conf>=0.25)
    for idx in val_idx:
        sample = val640[idx]
        preds = predict_one(model, sample["img"].to(device), sample["meta"], 0.25, 0.7)
        labels = val640._load_labels(val640.samples[idx])
        gts = [(int(l[0]), l[1], l[2], l[3], l[4]) for l in labels]
        preds_per_image.append(preds)
        gts_per_image.append(gts)
        for p in preds:
            c = int(p[0])
            pred_count[c] = pred_count.get(c, 0) + 1
        for g in gts:
            c = int(g[0])
            gt_count[c] = gt_count.get(c, 0) + 1

    m = compute_map(preds_per_image, gts_per_image)
    per_class = m["per_class"]

    print(f"总 mAP@50-95 = {m['map50_95']:.4f}   (val {len(val_idx)} 张)")
    print(f"{'类别':<14}{'GT框':>6}{'Pred框':>7}{'AP50-95':>9}")
    print("-" * 40)
    for c in sorted(per_class, key=lambda x: per_class[x]):
        nm = names.get(c, names.get(str(c), str(c)))
        print(f"{nm:<14}{gt_count.get(c, 0):>6}{pred_count.get(c, 0):>7}{per_class[c]:>9.4f}")

    # 数据不平衡概览
    total_gt = sum(gt_count.values())
    print("\n[数据分布] 各类 GT 占比 (top 由多到少):")
    for c in sorted(gt_count, key=lambda x: -gt_count[x]):
        nm = names.get(c, names.get(str(c), str(c)))
        print(f"  {nm:<14} {gt_count[c]:>4}  ({gt_count[c]/total_gt*100:.1f}%)")


if __name__ == "__main__":
    main()
