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
| Epochs     | 10               |
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
- Loss decreases over time → model is learning
- Validation follows training → no overfitting
- mAP increases → performance improves


[Result](/_posts/color_detection_image/results.png)

Throught the Graph we can analyze

#### TRAINING LOSS ANALYSIS

1. train/box_loss   
- Decreases from ~0.75 → ~0.53  
- Smooth and consistent  

**Interpretation:**
- Model is improving bounding box localization 

2. 🔹 train/cls_loss

- Drops rapidly from ~1.4 → ~0.28  
**Interpretation:**
- Model quickly learns to classify `redbox` vs `greenbox`  
- Task is relatively easy (distinct colors)  


### 📉 confusion_matrix.png

Shows classification performance:

| True \ Predicted | greenbox | redbox |
|------------------|----------|--------|
| greenbox         | ✅ correct | ❌ wrong |
| redbox           | ❌ wrong  | ✅ correct |

**What to check:**
- Strong diagonal → good model  
- Off-diagonal → classification errors  

---

### 📈 PR_curve.png (Precision–Recall Curve)
Shows tradeoff between precision and recall.

**What to check:**
- Curve near top-right → excellent model  
- Large area under curve → high performance  

---

### 📊 F1_curve.png
Shows best balance between precision and recall.

**What to check:**
- Peak value → best performance  
- Confidence at peak → optimal threshold  

---

### 🧠 weights/best.pt
The final trained model.

Use it for inference:

```bash
yolo detect predict model=weights/best.pt source=your_image.jpg conf=0.8
```
