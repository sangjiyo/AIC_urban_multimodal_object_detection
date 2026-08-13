"""推理: 对测试集三模态数据生成预测 txt。

用法示例:
    py -m src.predict --weights runs/train/best.pt --out-dir runs/predict
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch

from .config import load_config, resolve_data_root, resolve_test_root
from .dataset import TriModalDataset
from .infer import run_inference
from .model import build_model, load_checkpoint


def parse_args():
    p = argparse.ArgumentParser(description="三模态目标检测推理")
    p.add_argument("--config", type=str, default=None)
    p.add_argument("--weights", type=str, required=True, help="训练好的权重 .pt")
    p.add_argument("--data-root", type=str, default=None,
                   help="数据目录 (默认用配置里的 test_root, 即测试集)")
    p.add_argument("--out-dir", type=str, default="runs/predict", help="预测 txt 输出目录")
    p.add_argument("--imgsz", type=int, default=None)
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument("--iou", type=float, default=0.7)
    p.add_argument("--max-det", type=int, default=100)
    p.add_argument("--device", type=int, default=None)
    return p.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args.config)
    ic = cfg["infer"]

    imgsz = args.imgsz or ic.get("imgsz", 640)
    device_id = args.device if args.device is not None else ic.get("device", 0)
    device = torch.device(f"cuda:{device_id}" if (torch.cuda.is_available() and device_id >= 0) else "cpu")

    # 默认推理测试集; 可用 --data-root 覆盖 (如用于本地评测训练集)
    if args.data_root:
        root = Path(args.data_root)
        if not root.is_absolute():
            root = (Path(__file__).resolve().parent.parent / root)
    else:
        root = resolve_test_root(cfg)

    model = build_model(cfg, pretrained="")  # 推理用 checkpoint, 不加载预训练
    model = load_checkpoint(model, args.weights, device)
    model.eval()

    dataset = TriModalDataset(root, imgsz=imgsz, max_depth=cfg["depth"]["max_val"], train=False)
    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = Path(__file__).resolve().parent.parent / out_dir

    run_inference(model, dataset, out_dir,
                  conf_thres=args.conf, iou_thres=args.iou,
                  max_det=args.max_det, device=device)
    print(f"[完成] 预测结果已写出到 {out_dir} (共 {len(dataset)} 张)")


if __name__ == "__main__":
    main()
