#!/bin/bash
# yolov8l 多尺度渐进训练 (复刻 yolov8m 成功路径, 模型换 l)。
# 起点 train_l/best.pt (640), 每步 +128 渐进到 1024。
# 每步 6 epoch, lr0=0.001, 不冻结; batch 随尺度递减避免 OOM (l 比 m 更大)。
# 用法: bash run_multiscale_l.sh [device_id]   (默认 device 0)
set -e
cd /data1/mazc/sangjiyo/AIC_urban_multimodal_object_detection
PY=/home/mazc/anaconda3/bin/python
DEVICE=${1:-0}

PREV="runs/train_l/best.pt"
for STAGE in 768 896 1024; do
  case $STAGE in
    768)  BATCH=8;;
    896)  BATCH=6;;
    1024) BATCH=4;;
  esac
  SAVE="runs/ms_l_$STAGE"
  echo "=============================================================="
  echo "== yolov8l 渐进 imgsz=$STAGE  batch=$BATCH  init=$PREV"
  echo "=============================================================="
  $PY -m src.train --config configs/aic_trimodal_l.yaml --fusion \
    --imgsz "$STAGE" --epochs 6 --lr0 0.001 --init-weights "$PREV" \
    --save-dir "$SAVE" --batch-size "$BATCH" --device "$DEVICE"
  PREV="$SAVE/best.pt"
done

echo "=============================================================="
echo "== 多尺度渐进完成: 最终权重 $PREV"
echo "=============================================================="
