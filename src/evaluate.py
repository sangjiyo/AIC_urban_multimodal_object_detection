"""本地评测: 对比预测目录与真实标签目录, 计算 mAP@50-95。

用法示例:
    py -m src.evaluate --pred-dir runs/predict --gt-dir 数据目录/labels
"""
from __future__ import annotations

import argparse
from pathlib import Path

from .metrics import evaluate_dirs


def parse_args():
    p = argparse.ArgumentParser(description="本地 mAP@50-95 评测")
    p.add_argument("--pred-dir", type=str, required=True, help="预测 txt 目录")
    p.add_argument("--gt-dir", type=str, required=True, help="真实标签 txt 目录")
    p.add_argument("--max-det", type=int, default=100)
    return p.parse_args()


def main():
    args = parse_args()
    pred_dir = Path(args.pred_dir)
    gt_dir = Path(args.gt_dir)
    if not pred_dir.is_absolute():
        pred_dir = Path.cwd() / pred_dir
    if not gt_dir.is_absolute():
        gt_dir = Path.cwd() / gt_dir

    res = evaluate_dirs(pred_dir, gt_dir, max_det=args.max_det)
    print(f"mAP@50-95 : {res['map50_95']:.4f}")
    print(f"mAP@50    : {res['map50']:.4f}")
    print("per-threshold:", {k: round(v, 4) for k, v in res["per_threshold"].items()})
    print("per-class   :", {k: round(v, 4) for k, v in sorted(res["per_class"].items())})


if __name__ == "__main__":
    main()
