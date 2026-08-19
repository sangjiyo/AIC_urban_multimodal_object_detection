"""伪标签生成: 用当前最优模型对测试集推理, 高置信度框作为伪标签混入训练集。

输出 YOLO 训练格式 [class_id, cx, cy, w, h] (不带 conf), 写到 --out-labels 目录。
之后把测试集三模态图 + 伪标签组装成训练目录结构, 即可作为 extra 训练数据。

用法:
    py -m scripts.pseudo_label --config configs/aic_trimodal_m.yaml \
        --weights runs/ms_m_1024/best.pt --imgsz 1024 --tta \
        --conf 0.5 --out-labels pseudo_train/labels
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from src.config import load_config, resolve_test_root
from src.dataset import TriModalDataset
from src.infer import predict_one, predict_one_tta
from src.model import build_fusion_model, load_checkpoint


def parse_args():
    p = argparse.ArgumentParser(description="测试集伪标签生成")
    p.add_argument("--config", type=str, required=True)
    p.add_argument("--weights", type=str, required=True)
    p.add_argument("--imgsz", type=int, default=1024)
    p.add_argument("--conf", type=float, default=0.5, help="伪标签置信度阈值")
    p.add_argument("--iou", type=float, default=0.7)
    p.add_argument("--device", type=int, default=0)
    p.add_argument("--tta", action="store_true", help="启用水平翻转 TTA")
    p.add_argument("--out-labels", type=str, required=True, help="伪标签 labels 输出目录")
    return p.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args.config)
    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")
    root = resolve_test_root(cfg)

    model = build_fusion_model(cfg, pretrained="")
    model = load_checkpoint(model, args.weights, device)
    model.eval()

    dataset = TriModalDataset(root, imgsz=args.imgsz, max_depth=cfg["depth"]["max_val"], train=False)
    out_labels = Path(args.out_labels)
    out_labels.mkdir(parents=True, exist_ok=True)

    pred_fn = predict_one_tta if args.tta else predict_one
    n_boxes = 0
    n_img_with_box = 0
    for idx in range(len(dataset)):
        sample = dataset[idx]
        preds = pred_fn(model, sample["img"].to(device), sample["meta"],
                        args.conf, args.iou, 100)
        # YOLO 训练格式 [cls, cx, cy, w, h] (无 conf)
        lines = [f"{cls} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}"
                 for cls, cx, cy, w, h, conf in preds]
        if lines:
            n_img_with_box += 1
            n_boxes += len(lines)
        (out_labels / (sample["stem"] + ".txt")).write_text(
            "\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
        if (idx + 1) % 200 == 0:
            print(f"  已处理 {idx + 1}/{len(dataset)}", flush=True)

    print(f"[完成] 伪标签 {n_boxes} 个框 / {n_img_with_box} 张图 (conf>{args.conf}) -> {out_labels}")


if __name__ == "__main__":
    main()
