"""三模态 YOLOv8 训练脚本。

用法示例:
    py -m src.train --epochs 100 --batch-size 8 --imgsz 640
    py -m src.train --epochs 1 --batch-size 2 --imgsz 320   # smoke test
"""
from __future__ import annotations

import argparse
import copy
import functools
import math
import time
from pathlib import Path

# 重定向到文件时也实时刷新输出, 便于后台训练时监控进度
print = functools.partial(print, flush=True)

import torch
from torch.utils.data import DataLoader, Subset

from .config import load_config, resolve_data_root
from .dataset import TriModalDataset
from .infer import predict_one
from .metrics import compute_map
from .model import build_criterion, build_fusion_model, build_model


def parse_args():
    p = argparse.ArgumentParser(description="三模态目标检测训练")
    p.add_argument("--config", type=str, default=None, help="配置文件路径")
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--imgsz", type=int, default=None)
    p.add_argument("--lr0", type=float, default=None)
    p.add_argument("--device", type=int, default=None)
    p.add_argument("--workers", type=int, default=None)
    p.add_argument("--no-amp", action="store_true", help="禁用混合精度")
    p.add_argument("--fusion", action="store_true", help="使用双流模态融合模型")
    return p.parse_args()


def set_lr(optimizer, lr: float):
    for pg in optimizer.param_groups:
        pg["lr"] = lr


def update_ema(ema_model, model, decay: float = 0.9999):
    """指数移动平均: 参数 ema = decay*ema + (1-decay)*model, buffer (BN) 直接同步。"""
    with torch.no_grad():
        for p_ema, p in zip(ema_model.parameters(), model.parameters()):
            p_ema.data.mul_(decay).add_(p.data, alpha=1.0 - decay)
        for b_ema, b in zip(ema_model.buffers(), model.buffers()):
            b_ema.copy_(b)


def validate(model, val_dataset, val_indices, device, conf_thres=0.25, iou_thres=0.7):
    """在验证集上计算 mAP@50-95。

    Args:
        val_dataset: TriModalDataset (train=False)。
        val_indices: 验证集样本下标。
    """
    model.eval()
    preds_per_image, gts_per_image = [], []
    for idx in val_indices:
        sample = val_dataset[idx]
        preds = predict_one(model, sample["img"].to(device), sample["meta"], conf_thres, iou_thres)
        labels = val_dataset._load_labels(val_dataset.samples[idx])
        gts = [(int(l[0]), l[1], l[2], l[3], l[4]) for l in labels]
        preds_per_image.append(preds)
        gts_per_image.append(gts)
    return compute_map(preds_per_image, gts_per_image)


def main():
    args = parse_args()
    cfg = load_config(args.config)
    tc = cfg["train"]

    epochs = args.epochs or tc.get("epochs", 100)
    batch_size = args.batch_size or tc.get("batch_size", 8)
    imgsz = args.imgsz or tc.get("imgsz", 640)
    lr0 = args.lr0 or tc.get("lr0", 0.01)
    device_id = args.device if args.device is not None else tc.get("device", 0)
    workers = args.workers if args.workers is not None else tc.get("workers", 0)

    device = torch.device(f"cuda:{device_id}" if (torch.cuda.is_available() and device_id >= 0) else "cpu")
    amp = device.type == "cuda" and not args.no_amp

    root = resolve_data_root(cfg)
    save_dir = Path(tc.get("save_dir", "runs/train"))
    if not save_dir.is_absolute():
        save_dir = Path(__file__).resolve().parent.parent / save_dir
    save_dir.mkdir(parents=True, exist_ok=True)

    # ---------- 数据 ----------
    full = TriModalDataset(root, imgsz=imgsz, max_depth=cfg["depth"]["max_val"],
                           train=True, seed=cfg["data"].get("seed", 42))
    n = len(full)
    val_ratio = cfg["data"].get("val_ratio", 0.2)
    n_val = max(1, int(round(n * val_ratio)))
    indices = torch.randperm(n, generator=torch.Generator().manual_seed(cfg["data"].get("seed", 42))).tolist()
    val_idx, train_idx = indices[:n_val], indices[n_val:]

    train_dataset = Subset(full, train_idx)
    # 验证集禁用增强 (重建一个 train=False 的 dataset 用于 val)
    val_full = TriModalDataset(root, imgsz=imgsz, max_depth=cfg["depth"]["max_val"], train=False)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              num_workers=workers, collate_fn=TriModalDataset.collate_fn,
                              pin_memory=(device.type == "cuda"))

    print(f"[数据] 总样本 {n}, 训练 {len(train_idx)}, 验证 {len(val_idx)}, imgsz {imgsz}")

    # ---------- 模型 ----------
    if args.fusion:
        model = build_fusion_model(cfg).to(device)
        print("[模型] 使用双流模态融合模型 (RGB backbone + IR/Depth 残差分支)")
    else:
        model = build_model(cfg).to(device)
    criterion = build_criterion(model)
    model.args = model.args  # criterion 已引用

    # EMA 模型副本 (验证/保存用, 不参与反向传播)
    ema = copy.deepcopy(model).eval()

    # ---------- 优化器 & 调度 ----------
    optimizer = torch.optim.SGD(model.parameters(), lr=lr0,
                                momentum=tc.get("momentum", 0.937),
                                weight_decay=tc.get("weight_decay", 0.0005),
                                nesterov=True)
    warmup_epochs = tc.get("warmup_epochs", 3)
    lrf = tc.get("lrf", 0.01)
    scaler = torch.amp.GradScaler(device.type, enabled=amp)

    best_map = -1.0
    print(f"[训练] epochs {epochs}, batch {batch_size}, lr0 {lr0}, amp {amp}, device {device}")

    for epoch in range(epochs):
        model.train()
        # 学习率: warmup + cosine
        if epoch < warmup_epochs:
            lr = lr0 * (epoch + 1) / max(1, warmup_epochs)
        else:
            progress = (epoch - warmup_epochs) / max(1, epochs - warmup_epochs)
            lr = lr0 * lrf + (lr0 - lr0 * lrf) * (1 + math.cos(math.pi * progress)) / 2
        set_lr(optimizer, lr)

        t0 = time.time()
        total_loss = 0.0
        nb = len(train_loader)
        for i, batch in enumerate(train_loader):
            batch = {k: (v.to(device) if isinstance(v, torch.Tensor) else v)
                     for k, v in batch.items() if k in ("img", "batch_idx", "cls", "bboxes")}

            with torch.amp.autocast(device.type, enabled=amp):
                preds = model(batch["img"])
                loss, loss_items = criterion(preds, batch)
                loss = loss.sum()

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            update_ema(ema, model)

            total_loss += loss.item()
            if (i + 1) % max(1, nb // 5) == 0 or i == nb - 1:
                print(f"  epoch {epoch + 1}/{epochs}  [{i + 1}/{nb}]  "
                      f"loss {loss.item():.3f}  lr {lr:.5f}")

        # ---------- 验证 (用 EMA 权重) ----------
        m = validate(ema, val_full, val_idx, device)
        cur_map = m["map50_95"]
        print(f"[epoch {epoch + 1}/{epochs}] train_loss {total_loss / max(1, nb):.3f}  "
              f"val mAP@50-95 {cur_map:.4f}  ({time.time() - t0:.1f}s)")

        # 保存 best / last (保存 EMA 权重)
        ckpt = {"model": ema.state_dict(), "epoch": epoch, "best_map": cur_map, "cfg": cfg}
        torch.save(ckpt, save_dir / "last.pt")
        if cur_map > best_map:
            best_map = cur_map
            torch.save(ckpt, save_dir / "best.pt")
            print(f"  ✓ 保存 best.pt (mAP {best_map:.4f})")

    print(f"[完成] best mAP@50-95 = {best_map:.4f}, 权重保存在 {save_dir}")


if __name__ == "__main__":
    main()
