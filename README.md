# Satellite Imagery Building Segmentation (SpaceNet-6)

This project implements an end-to-end deep learning pipeline for **building footprint segmentation from high-resolution satellite imagery**, using the SpaceNet-6 dataset.

The goal is to train a robust semantic segmentation model that generalizes to unseen urban areas, following industry-standard computer vision and remote sensing practices.

---

## 📌 Problem Statement

Accurate extraction of building footprints from satellite imagery is crucial for:
- urban planning
- disaster response
- population and infrastructure analysis

This project focuses on **pixel-wise building segmentation** from RGB satellite tiles using deep learning.

---

## 📂 Dataset

- **Dataset**: SpaceNet-6 (Rotterdam AOI)
- **Inputs**: RGB GeoTIFF tiles (PS-RGB)
- **Labels**: Building footprints (vector → rasterized masks)
- **Resolution**: ~0.5m per pixel

Vector building annotations were rasterized into binary mask GeoTIFFs aligned with the input imagery.

---

## 🧠 Model

- **Architecture**: U-Net
- **Encoder**: ResNet-34 (ImageNet pretrained)
- **Framework**: PyTorch
- **Library**: segmentation-models-pytorch

---

## ⚙️ Training Pipeline

### Data Handling
- On-the-fly patch sampling (256×256)
- Class imbalance handled by probabilistic sampling of empty patches
- No pre-saved patches → efficient memory usage

### Split Strategy
- Train / Validation split performed **by image tiles**
- Prevents spatial data leakage

### Loss & Metrics
- **Loss**: Dice Loss
- **Metrics**:
  - Dice Coefficient
  - Intersection over Union (IoU)

---

## 📊 Results

| Metric | Value |
|------|------|
| Best Validation Dice | ~0.71 |
| Best Validation IoU  | ~0.64 |

The best model checkpoint is selected based on **validation IoU**.

---

## 🖥️ Hardware

- GPU-accelerated training (CUDA)
- Tested on consumer NVIDIA GPU

---


