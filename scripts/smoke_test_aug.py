"""Smoke test: 验证增强 + yolov8s 前向/loss + EMA。"""
from __future__ import annotations

import copy
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from src.config import load_config, resolve_data_root
from src.dataset import TriModalDataset


def main():
    cfg = load_config()
    root = resolve_data_root(cfg)

    # ---------- 数据增强检查 ----------
    ds = TriModalDataset(root, imgsz=640, max_depth=cfg["depth"]["max_val"], train=True, seed=42)
    n_mosaic_hit = 0
    for i in range(20):
        s = ds[i]
        img, boxes, cls = s["img"], s["boxes"], s["cls"]
        assert img.shape == (5, 640, 640), f"img shape {img.shape}"
        assert len(boxes) == len(cls)
        if len(boxes):
            assert boxes.min() >= -0.02 and boxes.max() <= 1.02, f"boxes 越界: {boxes.min():.3f}~{boxes.max():.3f}"
            assert (boxes[:, 0] >= 0).all() and (boxes[:, 0] <= 1).all(), "cx 越界"
            assert (boxes[:, 2] > 0).all() and (boxes[:, 3] > 0).all(), "w/h 非法"
        if s["meta"][0] == 1.0 and s["meta"][1] == 0:  # mosaic 分支的 meta 特征
            n_mosaic_hit += 1
    print(f"[数据] 增强 smoke test 通过 (20 样本, 疑似 mosaic {n_mosaic_hit} 个)")

    ds_val = TriModalDataset(root, imgsz=640, max_depth=cfg["depth"]["max_val"], train=False)
    assert ds_val[0]["img"].shape == (5, 640, 640)
    print("[数据] val 数据集正常 (无增强)")

    # ---------- yolov8s 前向 + loss ----------
    from src.model import build_model, build_criterion
    device = torch.device("cuda:0")
    model = build_model(cfg).to(device)
    criterion = build_criterion(model)

    batch = TriModalDataset.collate_fn([ds[i] for i in range(4)])
    batch = {k: (v.to(device) if isinstance(v, torch.Tensor) else v)
             for k, v in batch.items() if k in ("img", "batch_idx", "cls", "bboxes")}
    preds = model(batch["img"])
    loss, _ = criterion(preds, batch)
    print(f"[模型] yolov8s forward + loss OK, loss = {loss.sum().item():.3f}")

    # ---------- EMA 更新 ----------
    from src.train import update_ema
    ema = copy.deepcopy(model).eval()
    update_ema(ema, model)
    p0 = next(model.parameters()).data
    e0 = next(ema.parameters()).data
    assert torch.equal(e0, p0) or torch.allclose(e0, p0), "EMA 初始同步失败"
    print("[EMA] update_ema 正常")


if __name__ == "__main__":
    main()
