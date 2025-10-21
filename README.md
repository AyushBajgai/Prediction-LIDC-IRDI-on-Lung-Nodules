# LIDC-IDRI Lung Nodule Prediction

A complete pipeline for preprocessing, training, and evaluating deep learning models for lung nodule segmentation using the LIDC-IDRI dataset. <cite />

## Overview

This repository provides a reproducible workflow for automated lung nodule segmentation using U-Net and Nested U-Net architectures. <cite /> The system consists of two main components:

1. **Preprocessing Pipeline** - Converts raw LIDC-IDRI DICOM CT scans into standardized NumPy arrays with segmentation masks Prediction-LIDC-IRDI-on-Lung-Nodules:24-40 
2. **Segmentation Pipeline** - Trains and evaluates deep learning models for nodule detection Prediction-LIDC-IRDI-on-Lung-Nodules:32-47 

## Repository Structure

```
Prediction-LIDC-IRDI-on-Lung-Nodules/
├── LIDC-IDRI-Preprocessing/
│   ├── config_file_create.py
│   ├── prepare_dataset.py
│   ├── utils.py
│   └── requirements.txt
├── LIDC-IDRI-Segmentation/
│   ├── train.py
│   ├── validate.py
│   ├── dataset.py
│   ├── losses.py
│   ├── metrics.py
│   ├── Unet/
│   └── UnetNested/
└── README.md
``` Prediction-LIDC-IRDI-on-Lung-Nodules:8-18 Prediction-LIDC-IRDI-on-Lung-Nodules:8-26 

## Quick Start

### 1. Environment Setup

**Requirements:** Python 3.10 or 3.11, CUDA 11+ (recommended) Prediction-LIDC-IRDI-on-Lung-Nodules:54-56 

```bash
# Install preprocessing dependencies
cd LIDC-IDRI-Preprocessing
pip install -r requirements.txt

# Install segmentation dependencies
cd ../LIDC-IDRI-Segmentation
pip install -r requirements.txt
``` Prediction-LIDC-IRDI-on-Lung-Nodules:48-51 

### 2. Data Preprocessing

Generate configuration and preprocess the LIDC-IDRI dataset: <cite />

```bash
cd LIDC-IDRI-Preprocessing
python config_file_create.py
python prepare_dataset.py
``` Prediction-LIDC-IRDI-on-Lung-Nodules:62-65 Prediction-LIDC-IRDI-on-Lung-Nodules:88-90 

This creates processed CT slices, segmentation masks, and metadata in the `data/` directory. Prediction-LIDC-IRDI-on-Lung-Nodules:92-104 

### 3. Model Training

Train a U-Net model: <cite />

```bash
cd ../LIDC-IDRI-Segmentation
python train.py --name UNET --epochs 100 --batch_size 8 --optimizer Adam --augmentation True
``` Prediction-LIDC-IRDI-on-Lung-Nodules:94-96 

### 4. Model Evaluation

Evaluate the trained model: <cite />

```bash
python validate.py --name UNET --augmentation False
``` Prediction-LIDC-IRDI-on-Lung-Nodules:122-124 

## Features

### Preprocessing Pipeline
- DICOM file loading using `pylidc` Prediction-LIDC-IRDI-on-Lung-Nodules:26-27 
- Automated lung region segmentation Prediction-LIDC-IRDI-on-Lung-Nodules:29-30 
- Consensus mask generation from radiologist annotations Prediction-LIDC-IRDI-on-Lung-Nodules:32-33 
- Clean (nodule-free) sample extraction Prediction-LIDC-IRDI-on-Lung-Nodules:35-36 

### Segmentation Pipeline
- U-Net and Nested U-Net (U-Net++) architectures Prediction-LIDC-IRDI-on-Lung-Nodules:34-36 
- Combined BCE + Dice loss function Prediction-LIDC-IRDI-on-Lung-Nodules:36-36 
- Early stopping based on validation metrics Prediction-LIDC-IRDI-on-Lung-Nodules:38-40 
- Comprehensive evaluation metrics (Dice, IoU, False Positive Rate) Prediction-LIDC-IRDI-on-Lung-Nodules:143-150 

## Dataset

Download the LIDC-IDRI dataset (~1010 patients) from [TCIA](https://wiki.cancerimagingarchive.net/display/Public/LIDC-IDRI). Prediction-LIDC-IRDI-on-Lung-Nodules:148-148 

## Citation

- **Jaeho Kim (2020)** – LIDC-IDRI Segmentation (U-Net, Nested U-Net)  
  GitHub: [https://github.com/jaeho3690/LIDC-IDRI-Segmentation](https://github.com/jaeho3690/LIDC-IDRI-Segmentation) Prediction-LIDC-IRDI-on-Lung-Nodules:186-187 

- **Jaeho Kim (Kaggle Contributor)** – LIDC-IDRI Preprocessing Code (2017), Data Science Bowl 2017 Tutorial  
  https://www.kaggle.com/c/data-science-bowl-2017 Prediction-LIDC-IRDI-on-Lung-Nodules:140-141 

---

## Notes

Both pipelines have their own detailed README files with comprehensive documentation: <cite />
- See `LIDC-IDRI-Preprocessing/README.md` for preprocessing details Prediction-LIDC-IRDI-on-Lung-Nodules:1-1 
- See `LIDC-IDRI-Segmentation/README.md` for training and evaluation details Prediction-LIDC-IRDI-on-Lung-Nodules:1-1 

All outputs are saved as `.npy` arrays for interoperability with NumPy, PyTorch, and TensorFlow workflows. Prediction-LIDC-IRDI-on-Lung-Nodules:151-151 

Wiki pages you might want to explore:
- [Overview (AyushBajgai/Prediction-LIDC-IRDI-on-Lung-Nodules)](/wiki/AyushBajgai/Prediction-LIDC-IRDI-on-Lung-Nodules#1)
### Citations
**File:** LIDC-IDRI-Preprocessing/README.md (L1-4)
```markdown
# LIDC-IDRI Lung CT Preprocessing Pipeline

This repository provides a complete and reproducible preprocessing workflow for the **LIDC-IDRI dataset**, one of the largest publicly available CT datasets for lung nodule analysis.  
It prepares CT image slices, segmentation masks, and metadata for deep learning tasks such as **nodule detection, segmentation, and malignancy classification**.
```
**File:** LIDC-IDRI-Preprocessing/README.md (L8-18)
```markdown
## Project Structure

```
LIDC-IDRI-Preprocessing/
├── config_file_create.py     # Generates 'lung.conf' configuration file
├── prepare_dataset.py        # Main dataset builder and preprocessing script
├── utils.py                  # Helper functions (lung segmentation, directory validation, etc.)
├── requirements.txt          # Python dependencies
├── lung.conf                 # Configuration file generated automatically
└── README.md                 # Project documentation
```
```
**File:** LIDC-IDRI-Preprocessing/README.md (L24-40)
```markdown
This pipeline performs:

1. **Load LIDC-IDRI Dataset**  
   - Reads DICOM files using `pylidc` and extracts annotations.

2. **Lung Region Segmentation**  
   - Implemented in `utils.py → segment_lung()`; adapted from Kaggle Data Science Bowl 2017.

3. **Consensus Mask Generation**  
   - Combines radiologist annotations using `pylidc.utils.consensus()`.

4. **Clean Data Extraction**  
   - Creates 50 clean lung slices for patients without nodules.

5. **Metadata Export**  
   - Saves per-slice details to `meta_info.csv`.

```
**File:** LIDC-IDRI-Preprocessing/README.md (L48-51)
```markdown
### Installation
```bash
pip install -r requirements.txt
```
```
**File:** LIDC-IDRI-Preprocessing/README.md (L62-65)
```markdown
Before running the preprocessing script, create the configuration file:
```bash
python config_file_create.py
```
```
**File:** LIDC-IDRI-Preprocessing/README.md (L88-90)
```markdown
```bash
python prepare_dataset.py
```
```
**File:** LIDC-IDRI-Preprocessing/README.md (L92-104)
```markdown
### Output Structure
```
data/
├── Image/           # Processed lung slices (.npy)
├── Mask/            # Corresponding segmentation masks (.npy)
├── Clean/           # Clean (nodule-free) slices
│   ├── Image/
│   └── Mask/
└── Meta/
    └── meta_info.csv
```

Each patient’s data is stored under its own folder (e.g., `LIDC-IDRI-0001`).
```

**File:** LIDC-IDRI-Preprocessing/README.md (L148-148)
```markdown
- Download the full **LIDC-IDRI dataset** (~1010 patients) from [TCIA](https://www.cancerimagingarchive.net/collection/lidc-idri/).  
```
**File:** LIDC-IDRI-Preprocessing/README.md (L151-151)
```markdown
- `.npy` files can be used directly in PyTorch, TensorFlow, or NumPy pipelines.
```
**File:** LIDC-IDRI-Segmentation/README.md (L1-4)
```markdown
# LIDC-IDRI Lung CT Segmentation and Evaluation Pipeline

This repository implements the **training, validation, and evaluation** pipeline for lung nodule segmentation using **U-Net** and **Nested U-Net (U-Net++)** architectures on the **LIDC-IDRI dataset**.  
It follows the preprocessing stage and focuses on model learning, inference, and quantitative assessment of segmentation quality.
```
**File:** LIDC-IDRI-Segmentation/README.md (L8-26)
```markdown
## Project Structure

```
LIDC-IDRI-Segmentation/
├── train.py                  # Train and validate segmentation models
├── validate.py               # Evaluate and test trained models
├── dataset.py                # PyTorch Dataset for LIDC-IDRI slices and masks
├── losses.py                 # BCE + Dice combined loss function
├── metrics.py                # IoU, Dice, and related evaluation metrics
├── utils.py                  # Utility helpers (argument parsing, logging)
├── Unet/
│   ├── unet_model.py         # U-Net model definition
│   └── unet_parts.py         # Encoder/decoder components
├── UnetNested/
│   └── Nested_Unet.py        # Nested U-Net (U-Net++) implementation
├── model_outputs/            # Stores models, configs, and logs
├── requirements.txt          # Python dependencies
└── README.md                 # Documentation
```
```
**File:** LIDC-IDRI-Segmentation/README.md (L32-47)
```markdown
This pipeline supports:

1. **Model Training**  
   - Trains U-Net or Nested U-Net models on the processed LIDC-IDRI slices.  
   - Uses mixed loss (BCE + Dice) and configurable optimizers.

2. **Validation and Early Stopping**  
   - Monitors Dice and IoU scores per epoch.  
   - Stops automatically when performance stabilizes.

3. **Testing and Evaluation**  
   - Generates binary mask predictions for both *Test* and *Clean* datasets.  
   - Calculates segmentation metrics and false positive statistics.

4. **Output Management**  
   - Saves trained weights, metrics logs, and model configuration for reproducibility.
```
**File:** LIDC-IDRI-Segmentation/README.md (L54-56)
```markdown
Recommended versions:
- **Python 3.10 or 3.11**
- **CUDA 11+ (if available)**
```
**File:** LIDC-IDRI-Segmentation/README.md (L94-96)
```markdown
```bash
python train.py --name UNET --epochs 100 --batch_size 8 --optimizer Adam --augmentation True
```
```
**File:** LIDC-IDRI-Segmentation/README.md (L122-124)
```markdown
```bash
python validate.py --name UNET --augmentation False
```
```
**File:** LIDC-IDRI-Segmentation/README.md (L143-150)
```markdown
| Metric | Description |
|---------|-------------|
| **Dice Coefficient** | Measures the overlap between predicted and true regions |
| **IoU (Jaccard Index)** | Intersection over Union for segmentation |
| **False Positive Rate** | Average false detections per patient scan |
| **Confusion Matrix** | Reports TP, FP, FN, and TN counts |

Metrics are computed for both *nodule-present* and *clean* datasets.
``` 
```