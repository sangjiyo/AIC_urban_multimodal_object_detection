#!/bin/bash
# yolov8m 多尺度渐进训练 (服务器 4×3090 24GB)。
# 起点 cw_m/best.pt (640, val 0.3220), 每步 +128 渐进到 1280。
# 每步 6 epoch, lr0=0.001, 不冻结; batch 随尺度递减避免 OOM。
set -e
cd /data1/mazc/sangjiyo/AIC_urban_multimodal_object_detection
PY=/home/mazc/anaconda3/bin/python

PREV="runs/cw_m/best.pt"
for STAGE in 768 896 1024 1152 1280; do
  case $STAGE in
    768)  BATCH=12;;
    896)  BATCH=8;;
    1024) BATCH=8;;
    1152) BATCH=4;;
    1280) BATCH=4;;
  esac
  SAVE="runs/ms_m_$STAGE"
  echo "=============================================================="
  echo "== yolov8m 渐进 imgsz=$STAGE  batch=$BATCH  init=$PREV"
  echo "=============================================================="
  $PY -m src.train --config configs/aic_trimodal_m.yaml --fusion \
    --imgsz "$STAGE" --epochs 6 --lr0 0.001 --init-weights "$PREV" \
    --save-dir "$SAVE" --batch-size "$BATCH"
  PREV="$SAVE/best.pt"
done

echo "=============================================================="
echo "== 多尺度渐进完成: 最终权重 $PREV"
echo "=============================================================="
