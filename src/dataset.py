"""三模态数据集加载与预处理。

将 RGB(可见光) + Infrared(红外) + Depth(深度) 三模态拼接为 5 通道输入:
    channel 0~2 : RGB   (float, ImageNet 标准化)
    channel 3   : IR    (float, [0,1], 由 3 通道红外取灰度)
    channel 4   : Depth (float, [0,1], 16bit 毫米值归一化)

标签为 YOLO 格式 [class_id, cx, cy, w, h] (归一化)。
"""
from __future__ import annotations

import math
import random
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

# ImageNet 均值/方差, 用于 RGB 通道标准化 (与 COCO 预训练权重一致)
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def letterbox(h0: int, w0: int, imgsz: int):
    """计算 letterbox 几何参数 (保持宽高比, 中心填充)。

    Returns:
        r, top, left, bottom, right, new_h, new_w
    """
    r = min(imgsz / h0, imgsz / w0)
    new_h, new_w = int(round(h0 * r)), int(round(w0 * r))
    pad_h, pad_w = imgsz - new_h, imgsz - new_w
    top, left = pad_h // 2, pad_w // 2
    bottom, right = pad_h - top, pad_w - left
    return r, top, left, bottom, right, new_h, new_w


def apply_letterbox(img: np.ndarray, top: int, bottom: int, left: int, right: int,
                    new_h: int, new_w: int, pad_value: int) -> np.ndarray:
    """对图像做 resize + 边框填充 (支持 uint8 / uint16)。"""
    interp = cv2.INTER_LINEAR if img.ndim == 2 or img.shape[2] == 3 else cv2.INTER_LINEAR
    img = cv2.resize(img, (new_w, new_h), interpolation=interp)
    return cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=pad_value)


def imread_unicode(path: Path, flags: int = cv2.IMREAD_COLOR) -> np.ndarray:
    """读取图像 (兼容中文/非 ASCII 路径, cv2.imread 在 Windows 下不支持中文路径)。"""
    data = np.fromfile(str(path), dtype=np.uint8)
    img = cv2.imdecode(data, flags)
    if img is None:
        raise FileNotFoundError(f"无法读取图像: {path}")
    return img


def load_depth(path: Path, max_val: int = 19999) -> np.ndarray:
    """加载深度图并归一化到 [0,1] float32 单通道。

    16bit PNG 单位毫米 (有效 [0, max_val]) 为主;
    若意外为彩色图 (如 jpg) 则取灰度并 /255。
    """
    img = imread_unicode(path, cv2.IMREAD_UNCHANGED)
    if img.ndim == 3:  # 意外为彩色图, 取灰度
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = img.astype(np.float32)
    if img.max() > 255.0:  # 16bit 深度
        img = np.clip(img, 0.0, float(max_val)) / float(max_val)
    else:  # 8bit 灰度
        img = img / 255.0
    return img


def load_infrared_gray(path: Path) -> np.ndarray:
    """加载红外图并归一化到 [0,1] float32 单通道。

    红外为 3 通道灰度堆叠, 取均值作为单通道。
    """
    img = imread_unicode(path, cv2.IMREAD_UNCHANGED)
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img.astype(np.float32) / 255.0


def normalize_rgb(img_bgr: np.ndarray) -> np.ndarray:
    """BGR uint8 -> RGB float, ImageNet 标准化。"""
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    return (rgb - IMAGENET_MEAN) / IMAGENET_STD


def box_to_letterbox(boxes: np.ndarray, r: float, left: int, top: int,
                     h0: int, w0: int, imgsz: int) -> np.ndarray:
    """将 [cx, cy, w, h] 归一化(相对原图) 变换到 letterbox 后 (相对 imgsz)。"""
    boxes = boxes.copy().astype(np.float32)
    # 转像素
    boxes[:, 0] *= w0
    boxes[:, 1] *= h0
    boxes[:, 2] *= w0
    boxes[:, 3] *= h0
    # 缩放 + 偏移
    boxes[:, 0] = boxes[:, 0] * r + left
    boxes[:, 1] = boxes[:, 1] * r + top
    boxes[:, 2] *= r
    boxes[:, 3] *= r
    # 归一化到 imgsz
    boxes /= imgsz
    return boxes


def box_from_letterbox(boxes: np.ndarray, r: float, left: int, top: int,
                       h0: int, w0: int, imgsz: int) -> np.ndarray:
    """letterbox 后 (相对 imgsz) 的 [cx, cy, w, h] 反变换回原图归一化。"""
    boxes = boxes.copy().astype(np.float32)
    boxes *= imgsz  # 转像素
    boxes[:, 0] = (boxes[:, 0] - left) / r
    boxes[:, 1] = (boxes[:, 1] - top) / r
    boxes[:, 2] /= r
    boxes[:, 3] /= r
    boxes[:, 0] /= w0
    boxes[:, 1] /= h0
    boxes[:, 2] /= w0
    boxes[:, 3] /= h0
    return boxes


def random_affine(imgs, labels, imgsz: int = 640, degrees: float = 10.0,
                  translate: float = 0.1, scale: float = 0.5, shear: float = 2.0):
    """对三模态图做随机仿射 (旋转+缩放+平移+剪切), 标签同步变换。

    Args:
        imgs: [rgb(uint8,H,W,3), ir(float32,H,W), depth(float32,H,W)] (已 letterbox/mosaic)
        labels: (N,5) [cls, cx, cy, w, h] 相对 imgsz 归一化
    Returns:
        (imgs', labels') 越界/退化目标被过滤, labels' 为 (M,5)
    """
    w = h = imgsz
    angle = random.uniform(-degrees, degrees)
    shear_rad = math.radians(random.uniform(-shear, shear))
    sc = random.uniform(1 - scale, 1 + scale)
    dx = random.uniform(-translate, translate) * w
    dy = random.uniform(-translate, translate) * h

    # 仿射矩阵 = 平移 @ 剪切 @ 旋转缩放 @ 中心平移
    a = math.radians(angle)
    M = np.array([
        [sc * math.cos(a), -sc * math.sin(a), 0.0],
        [sc * math.sin(a), sc * math.cos(a), 0.0],
        [0.0, 0.0, 1.0],
    ])
    S = np.array([[1.0, math.tan(shear_rad), 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    M = S @ M
    M[0, 2] += dx + w / 2 - (M[0, 0] * w / 2 + M[0, 1] * h / 2)
    M[1, 2] += dy + h / 2 - (M[1, 0] * w / 2 + M[1, 1] * h / 2)
    M2 = M[:2].astype(np.float32)

    rgb = cv2.warpAffine(imgs[0], M2, (w, h), flags=cv2.INTER_LINEAR,
                         borderMode=cv2.BORDER_CONSTANT, borderValue=(114, 114, 114))
    ir = cv2.warpAffine(imgs[1], M2, (w, h), flags=cv2.INTER_LINEAR,
                        borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    depth = cv2.warpAffine(imgs[2], M2, (w, h), flags=cv2.INTER_NEAREST,
                           borderMode=cv2.BORDER_CONSTANT, borderValue=0)

    # 标签: 4 角点经仿射变换后取包围盒
    new_labels = []
    for lab in labels:
        cls, cx, cy, bw, bh = lab
        x1 = (cx - bw / 2) * w
        y1 = (cy - bh / 2) * h
        x2 = (cx + bw / 2) * w
        y2 = (cy + bh / 2) * h
        pts = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.float32)
        pts = (M2[:, :2] @ pts.T).T + M2[:, 2]
        nx1, ny1 = pts.min(0)
        nx2, ny2 = pts.max(0)
        nx1, nx2 = max(0.0, min(nx1, w)), max(0.0, min(nx2, w))
        ny1, ny2 = max(0.0, min(ny1, h)), max(0.0, min(ny2, h))
        ncx = (nx1 + nx2) / 2 / w
        ncy = (ny1 + ny2) / 2 / h
        nbw = (nx2 - nx1) / w
        nbh = (ny2 - ny1) / h
        if nbw > 0.01 and nbh > 0.01:
            new_labels.append([cls, ncx, ncy, nbw, nbh])
    labels = np.array(new_labels, dtype=np.float32).reshape(-1, 5) if new_labels else np.zeros((0, 5), dtype=np.float32)
    return [rgb, ir, depth], labels


def mosaic4(raw_imgs, raw_labels_list, imgsz: int = 640):
    """4 张原图拼成 imgsz x imgsz 的 mosaic。

    Args:
        raw_imgs: list of 4 个 (rgb, ir, depth) 原图 (未 letterbox)
        raw_labels_list: list of 4 个 (N,5) 标签, 原图归一化
    Returns:
        (rgb, ir, depth) mosaic 图, merged_labels (M,5) 相对 imgsz 归一化
    """
    cx = int(random.uniform(imgsz * 0.3, imgsz * 0.7))
    cy = int(random.uniform(imgsz * 0.3, imgsz * 0.7))
    regions = [
        (0, 0, cx, cy),                      # 左上
        (cx, 0, imgsz - cx, cy),             # 右上
        (0, cy, cx, imgsz - cy),             # 左下
        (cx, cy, imgsz - cx, imgsz - cy),    # 右下
    ]
    canvas_rgb = np.full((imgsz, imgsz, 3), 114, dtype=np.uint8)
    canvas_ir = np.zeros((imgsz, imgsz), dtype=np.float32)
    canvas_depth = np.zeros((imgsz, imgsz), dtype=np.float32)
    merged = []

    for (rgb, ir, depth), labels, (x0, y0, rw, rh) in zip(raw_imgs, raw_labels_list, regions):
        h0, w0 = rgb.shape[:2]
        r = min(rw / w0, rh / h0)
        nw, nh = int(round(w0 * r)), int(round(h0 * r))
        rgb_s = cv2.resize(rgb, (nw, nh), interpolation=cv2.INTER_LINEAR)
        ir_s = cv2.resize(ir, (nw, nh), interpolation=cv2.INTER_LINEAR)
        depth_s = cv2.resize(depth, (nw, nh), interpolation=cv2.INTER_NEAREST)
        px0, py0 = x0 + (rw - nw) // 2, y0 + (rh - nh) // 2
        canvas_rgb[py0:py0 + nh, px0:px0 + nw] = rgb_s
        canvas_ir[py0:py0 + nh, px0:px0 + nw] = ir_s
        canvas_depth[py0:py0 + nh, px0:px0 + nw] = depth_s
        if len(labels):
            lab = labels.copy()
            lab[:, 1] = (lab[:, 1] * w0 * r + px0) / imgsz
            lab[:, 2] = (lab[:, 2] * h0 * r + py0) / imgsz
            lab[:, 3] = lab[:, 3] * w0 * r / imgsz
            lab[:, 4] = lab[:, 4] * h0 * r / imgsz
            merged.append(lab)
    merged = np.concatenate(merged, 0) if merged else np.zeros((0, 5), dtype=np.float32)
    return canvas_rgb, canvas_ir, canvas_depth, merged


class TriModalDataset(Dataset):
    """三模态目标检测数据集 (训练/验证)。

    Args:
        root: 数据根目录, 含 visible/infrared/depth/labels 子目录。
        imgsz: letterbox 目标尺寸。
        names: 类别字典 {id: name}。
        train: True 用于训练 (启用水平翻转增强), False 用于验证。
    """

    def __init__(self, root: Path, imgsz: int = 640, max_depth: int = 19999,
                 train: bool = True, seed: int = 42,
                 mosaic_prob: float = 0.5, affine_prob: float = 0.5, flip_prob: float = 0.5,
                 affine_degrees: float = 10.0, affine_translate: float = 0.1,
                 affine_scale: float = 0.5, affine_shear: float = 2.0):
        self.root = Path(root)
        self.imgsz = imgsz
        self.max_depth = max_depth
        self.train = train

        visible_dir = self.root / "visible"
        infrared_dir = self.root / "infrared"
        depth_dir = self.root / "depth"
        label_dir = self.root / "labels"

        # 以 visible 目录为基准收集样本 stem (三模态+标签按文件名 stem 对齐)
        self.samples = sorted(p.stem for p in visible_dir.glob("*")
                              if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
        if len(self.samples) == 0:
            raise FileNotFoundError(f"未在 {visible_dir} 找到任何图像")

        self._visible_dir = visible_dir
        self._infrared_dir = infrared_dir
        self._depth_dir = depth_dir
        self._label_dir = label_dir

        self.rng = random.Random(seed)
        self.mosaic_prob = mosaic_prob   # Mosaic 增强概率
        self.affine_prob = affine_prob   # 随机仿射增强概率
        self.flip_prob = flip_prob       # 水平翻转增强概率
        self.affine_degrees = affine_degrees   # 仿射旋转角 (±deg)
        self.affine_translate = affine_translate  # 仿射平移 (±比例)
        self.affine_scale = affine_scale   # 仿射缩放 (±比例)
        self.affine_shear = affine_shear   # 仿射剪切 (±deg)

    def __len__(self) -> int:
        return len(self.samples)

    def _find_file(self, directory: Path, stem: str) -> Path:
        """在目录中按 stem 查找文件 (扩展名可能不同)。"""
        for p in directory.glob(stem + ".*"):
            return p
        raise FileNotFoundError(f"未找到 {stem} 在 {directory}")

    def _load_labels(self, stem: str) -> np.ndarray:
        """加载 YOLO 标签, 返回 (N, 5) [cls, cx, cy, w, h] 归一化。

        测试集无标签目录时返回空数组 (无目标)。
        """
        try:
            label_path = self._find_file(self._label_dir, stem)
        except FileNotFoundError:
            return np.zeros((0, 5), dtype=np.float32)
        boxes = []
        if label_path.exists() and label_path.stat().st_size > 0:
            for line in label_path.read_text().strip().splitlines():
                parts = line.split()
                if len(parts) >= 5:
                    cls = int(float(parts[0]))
                    cx, cy, w, h = (float(x) for x in parts[1:5])
                    boxes.append([cls, cx, cy, w, h])
        return np.array(boxes, dtype=np.float32).reshape(-1, 5) if boxes else np.zeros((0, 5), dtype=np.float32)

    def _load_raw(self, stem: str):
        """加载三模态原图 + 原图归一化标签 (不做 letterbox/增强)。"""
        rgb = imread_unicode(self._find_file(self._visible_dir, stem))
        h0, w0 = rgb.shape[:2]
        ir = load_infrared_gray(self._find_file(self._infrared_dir, stem))
        depth = load_depth(self._find_file(self._depth_dir, stem), self.max_depth)
        labels = self._load_labels(stem)  # (N,5) [cls, cx, cy, w, h] 原图归一化
        return rgb, ir, depth, labels, h0, w0

    def __getitem__(self, idx: int):
        stem = self.samples[idx]

        if self.train and self.rng.random() < self.mosaic_prob:
            # Mosaic: 随机取 4 张图拼接 (三模态同步)
            stems = [stem] + [self.samples[self.rng.randint(0, len(self))] for _ in range(3)]
            raws = [self._load_raw(s) for s in stems]
            rgb, ir, depth, labels = mosaic4(
                [(r[0], r[1], r[2]) for r in raws],
                [r[3] for r in raws],
                self.imgsz,
            )
            meta = (1.0, 0, 0, self.imgsz, self.imgsz)  # 训练时 meta 不用于推理
        else:
            rgb, ir, depth, labels, h0, w0 = self._load_raw(stem)
            r, top, left, bottom, right, new_h, new_w = letterbox(h0, w0, self.imgsz)
            rgb = apply_letterbox(rgb, top, bottom, left, right, new_h, new_w, pad_value=114)
            ir = apply_letterbox(ir, top, bottom, left, right, new_h, new_w, pad_value=0)
            depth = apply_letterbox(depth, top, bottom, left, right, new_h, new_w, pad_value=0)
            if len(labels):
                labels[:, 1:] = box_to_letterbox(labels[:, 1:], r, left, top, h0, w0, self.imgsz)
            meta = (r, left, top, h0, w0)

        # 随机仿射增强 (train)
        if self.train and self.rng.random() < self.affine_prob:
            (rgb, ir, depth), labels = random_affine(
                [rgb, ir, depth], labels, self.imgsz,
                degrees=self.affine_degrees, translate=self.affine_translate,
                scale=self.affine_scale, shear=self.affine_shear)

        # 水平翻转增强 (train)
        if self.train and self.rng.random() < self.flip_prob:
            rgb = np.ascontiguousarray(rgb[:, ::-1])
            ir = np.ascontiguousarray(ir[:, ::-1])
            depth = np.ascontiguousarray(depth[:, ::-1])
            if len(labels):
                labels[:, 1] = 1.0 - labels[:, 1]

        # 归一化 + 拼接为 5 通道 (C, H, W)
        rgb_norm = normalize_rgb(rgb)          # (H, W, 3)
        ir_norm = ir[..., None]                 # (H, W, 1)
        depth_norm = depth[..., None]           # (H, W, 1)
        img = np.concatenate([rgb_norm, ir_norm, depth_norm], axis=-1)  # (H, W, 5)
        img = torch.from_numpy(np.ascontiguousarray(img.transpose(2, 0, 1)))

        cls = torch.from_numpy(labels[:, 0].astype(np.int64)) if len(labels) else torch.empty(0, dtype=torch.int64)
        boxes = torch.from_numpy(labels[:, 1:5].astype(np.float32)) if len(labels) else torch.zeros((0, 4), dtype=torch.float32)

        return {"img": img, "cls": cls, "boxes": boxes, "stem": stem, "meta": meta}

    @staticmethod
    def collate_fn(batch):
        """组装一个 batch, 输出与 v8DetectionLoss 期望一致。

        Returns:
            img: (bs, 5, H, W)
            batch_idx: (M, 1)
            cls: (M, 1)
            bboxes: (M, 4)  xywh 归一化
            (以及 stems / metas 用于推理)
        """
        imgs = torch.stack([b["img"] for b in batch], 0)
        batch_idx_list, cls_list, box_list = [], [], []
        for i, b in enumerate(batch):
            n = b["cls"].shape[0]
            if n:
                batch_idx_list.append(torch.full((n, 1), i, dtype=torch.int64))
                cls_list.append(b["cls"].view(-1, 1))
                box_list.append(b["boxes"])
        if batch_idx_list:
            batch_idx = torch.cat(batch_idx_list, 0)
            cls = torch.cat(cls_list, 0)
            bboxes = torch.cat(box_list, 0)
        else:
            batch_idx = torch.zeros((0, 1), dtype=torch.int64)
            cls = torch.zeros((0, 1), dtype=torch.int64)
            bboxes = torch.zeros((0, 4), dtype=torch.float32)
        return {"img": imgs, "batch_idx": batch_idx, "cls": cls, "bboxes": bboxes,
                "stems": [b["stem"] for b in batch], "metas": [b["meta"] for b in batch]}
