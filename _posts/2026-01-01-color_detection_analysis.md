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

## 📊 results.png (Main Graph)
Shows:
- Training & validation loss
- Precision and recall
- mAP50 and mAP50-95  
✔ What to check:
- Loss decreases over time → model is learning
- Validation follows training → no overfitting
- mAP increases → performance improves


![Result](/_posts/color_detection_image/results.png)

Throught the Graph we can analyze

### 1. TRAINING LOSS ANALYSIS

1. train/box_loss   
- Decreases from ~0.75 → ~0.53  
- Smooth and consistent  

**Interpretation:**
- Model is improving bounding box localization 
---
2. train/cls_loss

- Drops rapidly from ~1.4 → ~0.28  
**Interpretation:**
- Model quickly learns to classify `redbox` vs `greenbox`  
- Task is relatively easy (distinct colors)  

--- 
3. train/dfl_loss
- Gradual decrease  

**Interpretation:**
- Model is refining bounding box precision  

---

### 2. Validation Loss Analysis

1. val/box_loss
- Smooth decrease  

**Interpretation:**
- Good generalization to unseen validation data  

---
2.  val/cls_loss
- Spike at early epoch (~3)  
- Then decreases steadily  

**Interpretation:**
- Early training instability (normal)  
- Model stabilizes afterward  

---

3. val/dfl_loss
- Smooth decreasing trend  

**Interpretation:**
- Bounding box quality improves on validation data  

---
### 🚨 Key Concept: Overfitting Check

| Training Loss | Validation Loss | Meaning |
|--------------|----------------|--------|
| ↓ | ↓ | ✅ Good (your case) |
| ↓ | ↑ | ❌ Overfitting |
| ↑ | ↑ | ❌ Poor training |


**Conclusion:**
- No overfitting observed  
- Model generalizes well  

---

### 3. Precision Analysis

- Starts ~0.94  
- Drops briefly  
- Converges to ~1.0  

**Interpretation:**
- Very few false positives  
- Temporary fluctuation is normal  

---

### 4. Recall Analysis

- Similar behavior to precision  
- Ends near 1.0  

**Interpretation:**
- Model detects almost all objects  
- Very few missed detections  

---

### 📊 5. mAP Analysis (Most Important Metric)

1. mAP50
- Final value ≈ 0.995  

**Interpretation:**
- Nearly perfect detection at IoU = 0.5  

---

2.  mAP50-95
- Improves from ~0.81 → ~0.93  

**Interpretation:**
- Strong performance under stricter evaluation  
- Indicates robust bounding box quality  

---
### ⚠️ 6. Early Epoch Instability

Observed at epoch ~3:
- Drop in precision, recall, and mAP  

**Reason:**
- Random weight initialization  
- Learning adjustment phase  

**Important:**
- This is normal behavior in training  
- Focus on overall trend, not individual fluctuations  

---
### 📉 7. Trend vs Raw Values

- Blue line: actual values  
- Orange line: smoothed trend  

**Interpretation:**
- Smoothed curve shows true learning behavior  
- Model trend is stable and improving  

---


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
