"""mAP@50-95 评测模块。

严格按赛题「性能指标要求」实现:
  - IoU 阈值 T = {0.50, 0.55, ..., 0.95} 共 10 个
  - 每个类别/阈值按置信度降序匹配, 计算 TP/FP/FN
  - 单类 AP 采用 101 点插值 (recall 在 [0,1] 均匀取 101 个点)
  - mAP(t) = 各类 AP 均值;  mAP@50-95 = 10 个阈值 mAP 均值
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

IOU_THRESHOLDS = np.arange(0.50, 0.96, 0.05)  # 10 个阈值


def xywh_to_xyxy(box) -> np.ndarray:
    """[cx, cy, w, h] (归一化) -> [x1, y1, x2, y2]。"""
    cx, cy, w, h = box
    return np.array([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2])


def box_iou(a, b) -> float:
    """两个 xyxy 框的 IoU。"""
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def compute_ap_101(recall: np.ndarray, precision: np.ndarray) -> float:
    """101 点插值法计算 AP。"""
    ap = 0.0
    for r in np.linspace(0.0, 1.0, 101):
        mask = recall >= r
        p = float(precision[mask].max()) if mask.any() else 0.0
        ap += p
    return ap / 101.0


def _class_ap(preds, gts, t: float) -> float | None:
    """单个类别在单个 IoU 阈值下的 AP; 无 GT 时返回 None (跳过)。

    preds: list[(img_idx, cx, cy, w, h, conf)]
    gts:   list[(img_idx, cx, cy, w, h)]
    """
    n_gt = len(gts)
    if n_gt == 0:
        return None

    # 按置信度降序
    preds = sorted(preds, key=lambda x: -x[5])
    matched = [False] * n_gt
    tp = np.zeros(len(preds))
    fp = np.zeros(len(preds))

    for i, pred in enumerate(preds):
        img_idx, cx, cy, w, h, conf = pred
        pbox = xywh_to_xyxy((cx, cy, w, h))
        best_iou, best_j = 0.0, -1
        for j, gt in enumerate(gts):
            gt_img_idx = gt[0]
            if gt_img_idx == img_idx and not matched[j]:
                iou = box_iou(pbox, xywh_to_xyxy(gt[1:5]))
                if iou > best_iou:
                    best_iou, best_j = iou, j
        if best_j >= 0 and best_iou >= t:
            tp[i] = 1.0
            matched[best_j] = True
        else:
            fp[i] = 1.0

    cum_tp = np.cumsum(tp)
    cum_fp = np.cumsum(fp)
    recall = cum_tp / n_gt
    precision = cum_tp / np.maximum(cum_tp + cum_fp, 1e-16)
    return compute_ap_101(recall, precision)


def compute_map(preds_per_image, gts_per_image, iou_thresholds=IOU_THRESHOLDS) -> dict:
    """计算 mAP@50-95。

    Args:
        preds_per_image: list[list[tuple]], 每张图一个 list, 元素 (cls, cx, cy, w, h, conf)
        gts_per_image:   list[list[tuple]], 每张图一个 list, 元素 (cls, cx, cy, w, h)

    Returns:
        dict: {'map50_95', 'map50', 'per_class', 'per_threshold'}
    """
    classes = set()
    for img_gts in gts_per_image:
        for g in img_gts:
            classes.add(int(g[0]))
    for img_preds in preds_per_image:
        for p in img_preds:
            classes.add(int(p[0]))

    # 收集每个类别: preds -> (img_idx, cx, cy, w, h, conf), gts -> (img_idx, cx, cy, w, h)
    class_preds, class_gts = {}, {}
    for c in classes:
        cp, cg = [], []
        for img_idx, img_preds in enumerate(preds_per_image):
            for p in img_preds:
                if int(p[0]) == c:
                    cp.append((img_idx, p[1], p[2], p[3], p[4], p[5]))
        for img_idx, img_gts in enumerate(gts_per_image):
            for g in img_gts:
                if int(g[0]) == c:
                    cg.append((img_idx, g[1], g[2], g[3], g[4]))
        class_preds[c], class_gts[c] = cp, cg

    # per_class_aps[c] = 类别 c 在 10 个阈值 AP 的平均
    # per_threshold_maps[t] = 阈值 t 下所有类别 AP 的平均
    per_class_aps = {}
    per_threshold_maps = {}
    ap_matrix = {c: [] for c in classes}  # c -> list of AP per threshold
    for t in iou_thresholds:
        aps_t = []
        for c in sorted(classes):
            ap = _class_ap(class_preds[c], class_gts[c], float(t))
            ap_matrix[c].append(ap)
            if ap is not None:
                aps_t.append(ap)
        per_threshold_maps[round(float(t), 2)] = float(np.mean(aps_t)) if aps_t else 0.0

    for c in sorted(classes):
        valid = [a for a in ap_matrix[c] if a is not None]
        per_class_aps[c] = float(np.mean(valid)) if valid else 0.0

    map50_95 = float(np.mean(list(per_class_aps.values()))) if per_class_aps else 0.0
    return {
        "map50_95": map50_95,
        "map50": per_threshold_maps.get(0.5, 0.0),
        "per_class": per_class_aps,
        "per_threshold": per_threshold_maps,
    }


# ---------------- 文件级评测 ----------------

def read_predictions(pred_path: Path, max_det: int = 100):
    """读取一个预测 txt, 返回 list[(cls, cx, cy, w, h, conf)]。

    过滤非法类别/坐标/置信度缺失 (与赛题一致, 非法预测不参与计算)。
    """
    preds = []
    if not pred_path.exists():
        return preds
    for line in pred_path.read_text().strip().splitlines():
        parts = line.split()
        if len(parts) < 6:
            continue
        try:
            cls = int(float(parts[0]))
            cx, cy, w, h, conf = (float(x) for x in parts[1:6])
        except ValueError:
            continue
        if not (0 <= cx <= 1 and 0 <= cy <= 1 and 0 <= w <= 1 and 0 <= h <= 1):
            continue
        if not (0.0 <= conf <= 1.0):
            continue
        preds.append((cls, cx, cy, w, h, conf))
    preds.sort(key=lambda x: -x[5])
    return preds[:max_det]


def read_ground_truth(gt_path: Path):
    """读取一个 GT txt, 返回 list[(cls, cx, cy, w, h)]。"""
    gts = []
    if not gt_path.exists():
        return gts
    for line in gt_path.read_text().strip().splitlines():
        parts = line.split()
        if len(parts) < 5:
            continue
        try:
            cls = int(float(parts[0]))
            cx, cy, w, h = (float(x) for x in parts[1:5])
        except ValueError:
            continue
        gts.append((cls, cx, cy, w, h))
    return gts


def evaluate_dirs(pred_dir: Path, gt_dir: Path, max_det: int = 100) -> dict:
    """对两个目录下的同名 txt 文件评测 (预测目录 vs 真实标签目录)。"""
    pred_dir, gt_dir = Path(pred_dir), Path(gt_dir)
    gts_per_image, preds_per_image = [], []

    gt_files = sorted(gt_dir.glob("*.txt"))
    for gt_path in gt_files:
        stem = gt_path.stem
        gts_per_image.append(read_ground_truth(gt_path))
        preds_per_image.append(read_predictions(pred_dir / (stem + ".txt"), max_det))

    return compute_map(preds_per_image, gts_per_image)
