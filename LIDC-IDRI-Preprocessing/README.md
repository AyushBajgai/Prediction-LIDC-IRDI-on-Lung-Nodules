# LIDC-IDRI Lung CT Preprocessing Pipeline

This repository provides a complete and reproducible preprocessing workflow for the **LIDC-IDRI dataset**, one of the largest publicly available CT datasets for lung nodule analysis.  
It prepares CT image slices, segmentation masks, and metadata for deep learning tasks such as **nodule detection, segmentation, and malignancy classification**.

---

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

---

## 1. Overview

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

---

## 2. Environment Setup

### Python Environment
Recommended Python version: **3.10 or 3.11**

### Installation
```bash
pip install -r requirements.txt
```

Or in Google Colab:
```python
!pip install -r requirements.txt
```

---

## 3. Configuration File (`lung.conf`)

Before running the preprocessing script, create the configuration file:
```bash
python config_file_create.py
```

Example configuration:

```ini
[prepare_dataset]
lidc_dicom_path = ./LIDC-IDRI
mask_path = ./data/Mask
image_path = ./data/Image
clean_path_image = ./data/Clean/Image
clean_path_mask = ./data/Clean/Mask
meta_path = ./data/Meta/
mask_threshold = 8

[pylidc]
confidence_level = 0.5
padding_size = 512
```

---

## 4. Running the Preprocessing

```bash
python prepare_dataset.py
```

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

---

## 5. Example Usage

```python
from prepare_dataset import MakeDataSet
from configparser import ConfigParser
import os

parser = ConfigParser()
parser.read('lung.conf')

LIDC_IDRI_list = sorted(
    [f for f in os.listdir(parser.get('prepare_dataset','LIDC_DICOM_PATH')) if not f.startswith('.')]
)

builder = MakeDataSet(
    LIDC_IDRI_list,
    parser.get('prepare_dataset','IMAGE_PATH'),
    parser.get('prepare_dataset','MASK_PATH'),
    parser.get('prepare_dataset','CLEAN_PATH_IMAGE'),
    parser.get('prepare_dataset','CLEAN_PATH_MASK'),
    parser.get('prepare_dataset','META_PATH'),
    parser.getint('prepare_dataset','Mask_Threshold'),
    parser.getint('pylidc','padding_size'),
    parser.getfloat('pylidc','confidence_level')
)
builder.prepare_dataset()
```


## 6. Notes

- Download the full **LIDC-IDRI dataset** (~1010 patients) from [TCIA](https://www.cancerimagingarchive.net/collection/lidc-idri/).  
- On the first run, `pylidc` will create a local SQLite index (takes a few minutes).  
- Missing scans are automatically skipped.  
- `.npy` files can be used directly in PyTorch, TensorFlow, or NumPy pipelines.

---
