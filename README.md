# 面向城市场景的视觉多模态目标检测

全球校园人工智能算法精英大赛 · 算法挑战赛 —— 基于 **RGB(可见光) + Infrared(红外) + Depth(深度)** 三模态数据融合的目标检测。

## 赛题要点

- **任务**：对三模态空间对齐图像中的关键目标做精准定位与分类。
- **类别**（12 类，编号 0~11）：`person, boat, animal, seat, sign, bicycle, car, ball, light, garbage can, uav, tricycle`
- **标签格式**：YOLO `[class_id, cx, cy, w, h]`（归一化）
- **输出格式**：`[class_id, cx, cy, w, h, confidence]`，每张图一个同名 txt，最大 100 个框
- **评测指标**：`mAP@50-95`（IoU 0.50~0.95 步长 0.05，101 点插值 AP，12 类等权平均）
- **约束**：禁止在线服务/API、禁止手工标注、禁止多模型简单集成投票/平均；允许 ImageNet/COCO/Objects365 预训练权重

## 数据说明

| 模态 | 格式 | 数值范围 |
|------|------|----------|
| RGB 可见光 | 3 通道 uint8 | [0, 255] |
| Infrared 红外 | 3 通道 uint8（灰度堆叠，无彩色语义） | [0, 255]，越大越热 |
| Depth 深度 | 单通道 uint16（mm） | 有效 [0, 19999]，0 为无效 |

数据目录结构：`visible/` `infrared/` `depth/` `labels/` 四个子目录，按文件名 stem 对齐。

训练集 2000 张（内部按 seed=42 划 1600 train / 400 val），测试集 1000 张（无标签）。

## 方案设计

**双流模态融合 Stem**（当前最优，`--fusion` 开启）：

```
输入 5 通道 [RGB(3), IR(1), Depth(1)]
        │
        ├── rgb_stem = Conv(3→32)   ← 复用 COCO 预训练权重
        │
        └── aux_stem = Conv(2→32)   ← IR/Depth 残差分支, BN gamma 置 0 从零起步
                │
             相加融合 (add) → 后续 backbone/neck/head 与标准 yolov8 完全一致
```

- **RGB 流**直接继承 COCO 预训练特征，**IR/Depth 流**从 0 开始逐步补充互补信息，训练初期等价于纯 RGB 模型，避免破坏预训练。
- 相比「简单 5 通道拼接」更稳（可对照 `build_model` vs `build_fusion_model`）。

## 项目结构

```
configs/aic_trimodal.yaml    # yolov8n 配置 (类别/数据路径/训练推理超参)
configs/aic_trimodal_s.yaml  # yolov8s 配置 (batch 4, 独立保存 runs/train_s)
src/
  config.py                  # 配置加载
  dataset.py                 # 三模态加载 / letterbox / 增强(可开关) / 5 通道拼接
  model.py                   # 双流融合 Stem + 预训练权重加载
  metrics.py                 # mAP@50-95 评测 (101 点插值)
  infer.py                   # 推理 + NMS 后处理 + 水平翻转 TTA
  train.py                   # 训练 (--fusion 开关, EMA/增强可配)
  predict.py                 # 测试集推理
  evaluate.py                # 本地评测
  pack.py                    # 打包提交
scripts/
  diagnose_per_class.py      # per-class AP 诊断 (定位拖后腿类别)
  diagnose_tta.py            # TTA/分辨率对照诊断
  diagnose_compare.py / diagnose_map.py / diagnose_raw.py   # 早期融合/指标诊断
  smoke_test_*.py            # 快速冒烟验证
```

## 快速开始

### 1. 安装依赖

```bash
# CUDA 版 torch (按显卡 CUDA 版本选择, RTX 30 系可用 cu121)
py -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
py -m pip install -r requirements.txt
```

### 2. 训练（双流融合）

```bash
# yolov8n (4GB 显存可 batch 8)
py -m src.train --fusion
# yolov8s (batch 4, 显式指定配置)
py -m src.train --config configs/aic_trimodal_s.yaml --fusion
```

### 3. 推理 + 打包提交

```bash
py -m src.predict --config configs/aic_trimodal_s.yaml \
    --weights runs/train_s/best.pt --fusion --tta --imgsz 640 \
    --conf 0.25 --out-dir runs/predict_s
py -m src.pack --pred-dir runs/predict_s --out submission.zip
```

> ⚠️ 推理必须显式传 `--config` 指定与权重一致的模型架构（yolov8s 权重配 `aic_trimodal_s.yaml`），否则会误用 yolov8n 架构导致权重不匹配。

### 4. 本地评测（有标签时）

```bash
py -m src.evaluate --pred-dir runs/predict_s --gt-dir 数据目录/labels
```

## 关键经验（踩坑记录）

这些坑都曾让提交分数暴跌，务必遵守：

1. **别在小数据集上叠加强增强 + 减半 lr**：Mosaic+仿射 + `lr0=0.005` 会让 val 从 0.2168 回退到 0.1588（提交 32→15 分）。正确配方是 **batch8/lr0.01/关 Mosaic+仿射/只留水平翻转**（batch4 时 lr0 仍保持 0.01）。
2. **推理分辨率锁死 640**：YOLOv8 的 DFL 头框尺寸是「网格单位」，训练固定 640 且无尺度增强时，换 1280 输入框中心会放大但宽高不缩放，mAP 从 0.31 崩到 0.04（提交 7 分）。**1280 是陷阱，不是提升手段**（除非先做尺度增强训练）。
3. **conf_thres 固定 0.25，勿按 val mAP 调**：降到 0.001 会让 val mAP 微升，但提交从 33.799 暴跌到 27.524（验证集与测试集置信度校准不一致，低置信框在 test 上多为误检）。
4. **EMA 默认关闭**：decay 0.9999 太慢导致验证滞后 ~25 epoch，选中欠训练权重。

## 当前进度与分数

| 方案 | val mAP@50-95 | 提交分数 |
|------|---------------|----------|
| baseline yolov8n | 0.2168 | 32.989 |
| yolov8n + 融合 | 0.2453 | 33.799 |
| **yolov8s + 融合** | **0.2562** | **36.4260** |

**per-class 诊断结论**（12 类等权平均，每类 +0.01 ≈ 总 mAP +0.0008）：

- **小目标类拖后腿，高分辨率是解药**：`person`(39% GT, AP 0.20)、`sign`(0.197)、`bicycle`(0.122)、`ball`(0.183)——正是 640 分辨率下小目标漏检/框不准的重灾区。
- **语义难题**：`animal`(22% GT, AP 0.19)——猫狗鸟等塞一个类，类内差异巨大，需更强模型/更多数据。
- **小样本类**：`tricycle`(5框)、`ball`(16框)、`boat`(22框)，可过采样。

## 后续优化方向（按性价比）

1. **尺度增强训练 → 解锁高分辨率推理**（最大杠杆）：训练加温和 scale 抖动（不 rotate/shear），让模型学会跨尺度，再试 832/960 推理，主攻 person/sign/bicycle/ball 小目标。
2. **少数类过采样**：tricycle/ball/boat 放大采样权重，可与 1 一起做。
3. **伪标签自训练**：用当前模型对 1000 张测试集打高置信度伪标签混入训练，1600→2600 且覆盖测试分布。
4. **融合升级**：残差相加 → 通道注意力 / 跨模态 Transformer（复杂度高，等前面榨干再上）。
5. **深度图处理**：对数映射或反距离（1/z）编码，弱化远距离噪声。
