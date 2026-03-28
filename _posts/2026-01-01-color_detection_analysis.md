---
title: 'YOLOv8 Red & Green Object Detection Analysis'
date: 01-01-2026
permalink: /posts/2026-01-01-object_detection_analysis
tags:
  - cool posts
  - category1
  - category2
---

#  YOLOv8 Red & Green Object Detection  

## 📌 Project Overview
This project uses **YOLOv8** to detect and classify two object classes:

- 🟩 `greenbox`
- 🟥 `redbox`

## ⚙️ Training Configuration
After running 

```
yolo detect train model=yolov8n.pt data=data.yaml imgsz=320 epochs=10 batch=16 device=0

```
| Parameter   | Value            |
|------------|------------------|
| Model      | yolov8n          |
| Image Size | 320              |
| Epochs     | 40               |
| Batch Size | 16               |
| Device     | GPU (device=0)   |



## 📁 Important Output Files
Inside the training folder (`runs/detect/train/`), the most important files are:

- `results.png` ✅ (main training graph)
- `confusion_matrix.png` ✅
- `PR_curve.png` ✅
- `F1_curve.png` ✅
- `weights/best.pt` ✅

### 📊 results.png (Main Graph)
Shows:
- Training & validation loss
- Precision and recall
- mAP50 and mAP50-95  
✔ What to check:
-Loss decreases over time → model is learning
-Validation follows training → no overfitting
-mAP increases → performance improves