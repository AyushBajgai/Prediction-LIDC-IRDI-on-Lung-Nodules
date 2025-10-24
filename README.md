# 🫁 LIDC-IDRI Lung Nodule Prediction

A complete pipeline for **preprocessing, training, and evaluating** deep learning models for **lung nodule segmentation and prediction** using the **LIDC-IDRI** dataset.

---

## Overview

This repository provides a **reproducible workflow** for automated lung nodule detection and segmentation using **U-Net** and **Nested U-Net (U-Net++)** architectures.  
The system consists of two main components:

1. **Preprocessing Pipeline** – Converts raw LIDC-IDRI DICOM CT scans into standardized NumPy arrays with segmentation masks  
2. **Segmentation & Prediction Pipeline** – Trains and evaluates deep learning models for lung nodule segmentation and classification

---

## Repository Structure
```
Prediction-LIDC-IRDI-on-Lung-Nodules/
│
├── LIDC-IDRI-Preprocessing/
│ ├── config_file_create.py
│ ├── prepare_dataset.py
│ ├── utils.py
│ ├── requirements.txt
│ ├── ResNet.ipynb
│ ├── Classification.ipynb
│ ├── Classification2.ipynb
│ ├── Classification3.ipynb
│
├── LIDC-IDRI-Segmentation/
│ ├── train.py
│ ├── validate.py
│ ├── dataset.py
│ ├── losses.py
│ ├── metrics.py
│ ├── Unet/
│ ├── UnetNested/
│ ├── requirements.txt

```

---

## Quick Start

### 1. Environment Setup

**Requirements:**
- Python 3.10 or 3.11  
- CUDA 11+ (recommended for GPU acceleration)

Install preprocessing dependencies:
---

### Installation

```bash
cd LIDC-IDRI-Preprocessing
pip install -r requirements.txt
```

```bash
cd ../LIDC-IDRI-Segmentation
pip install -r requirements.txt
```

Or in Google Colab:
```python
!pip install -r requirements.txt
```
---

### Data Preprocessing

Run the preprocessing scripts to prepare the dataset:

```bash
cd LIDC-IDRI-Preprocessing
python config_file_create.py
python prepare_dataset.py
```

This step will:

1) Load DICOM CT scans using pylidc

2) Generate segmentation masks from radiologist annotations

3)Extract clean (nodule-free) and nodule-positive slices

4) Store preprocessed data in the data/ directory


---

### Model Training

Train a segmentation or prediction model:

```bash
cd ../LIDC-IDRI-Segmentation
python train.py --name UNET --epochs 100 --batch_size 8 --optimizer Adam --augmentation True
```
---

### Model Evaluation

```bash
python validate.py --name UNET --augmentation False
```
Metrics such as Dice coefficient, IoU, False Positive Rate, and Sensitivity are reported for segmentation performance.
---

### Features

Preprocessing Pipeline

1) DICOM loading using pylidc

2) Automated lung region segmentation

3) Consensus mask generation from multiple radiologists

4) Extraction of clean (nodule-free) CT slices

Segmentation & Classification Pipeline

1) Supports U-Net, U-Net++, and ResNet-based CNNs

2) Combined BCE + Dice loss or CrossEntropy losses

3) Early stopping based on validation Dice/IoU

4) Comprehensive evaluation metrics:

5) Dice, IoU, Sensitivity, Specificity, False Positive Rate

---

### Citation
“LIDC-IDRI Lung Nodule Prediction: Deep Learning-Based Segmentation and Classification Pipeline”
© 2025 — Based on open-source frameworks for medical imaging analysis.

---

### Notes

- Download the full **LIDC-IDRI dataset** (~1010 patients) from [link here](https://www.cancerimagingarchive.net/collection/lidc-idri/).  
- `.npy` files can be used directly in PyTorch, TensorFlow, or NumPy pipelines.
---