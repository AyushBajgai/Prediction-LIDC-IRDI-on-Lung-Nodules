# LIDC-IDRI Lung CT Segmentation and Evaluation Pipeline

This repository implements the **training, validation, and evaluation** pipeline for lung nodule segmentation using **U-Net** and **Nested U-Net (U-Net++)** architectures on the **LIDC-IDRI dataset**.  
It follows the preprocessing stage and focuses on model learning, inference, and quantitative assessment of segmentation quality.

---

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

---

## 1. Overview

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

---

## 2. Environment Setup

### Python Environment
Recommended versions:
- **Python 3.10 or 3.11**
- **CUDA 11+ (if available)**

### Installation
```bash
pip install -r requirements.txt
```

---

## 3. Dataset Structure

Before running this pipeline, prepare data using the **LIDC-IDRI Preprocessing Pipeline**.  
Expected directory layout:

```
/home/LUNG_DATA/
├── Image/                   # CT slices (.npy)
├── Mask/                    # Corresponding segmentation masks (.npy)
├── Clean/                   # Clean (nodule-free) samples
│   ├── Image/
│   └── Mask/
└── meta_csv/
    ├── meta.csv             # Main dataset metadata
    └── clean_meta.csv       # Clean dataset metadata
```

Each `.csv` file contains:
- `patient_id`
- `original_image`
- `mask_image`
- `data_split` (Train / Validation / Test)

---

## 4. Model Training

Start training with the following command:

```bash
python train.py --name UNET --epochs 100 --batch_size 8 --optimizer Adam --augmentation True
```

### Key Arguments
| Parameter | Description | Example               |
|------------|-------------|-----------------------|
| `--name` | Model type | `UNET` / `NestedUNET` |
| `--epochs` | Number of training epochs | `100`                 |
| `--batch_size` | Mini-batch size | `8`                   |
| `--optimizer` | Optimizer choice | `Adam` / `SGD`        |
| `--augmentation` | Enable data augmentation | `True` / `False`      |
| `--early_stopping` | Stop after N epochs without improvement | `50`                  |

### Output Files
```
model_outputs/{MODEL_NAME}/
├── model.pth          # Trained model weights
├── config.yml         # Configuration parameters
└── log.csv            # Training log with loss, Dice, IoU
```

---

## 5. Inference and Evaluation

After training, evaluate your model using:

```bash
python validate.py --name UNET --augmentation False
```

### Functions
- Loads trained weights automatically.  
- Runs inference on **Test** and **Clean** datasets.  
- Saves predicted binary masks into structured folders.  
- Computes Dice, IoU, and False Positive metrics.

### Output Directory
```
/home/LUNG_DATA/Segmentation_output/
├── UNET/                   # Predictions on test dataset
└── CLEAN_UNET/             # Predictions on clean dataset
```

---

## 6. Evaluation Metrics

| Metric | Description |
|---------|-------------|
| **Dice Coefficient** | Measures the overlap between predicted and true regions |
| **IoU (Jaccard Index)** | Intersection over Union for segmentation |
| **False Positive Rate** | Average false detections per patient scan |
| **Confusion Matrix** | Reports TP, FP, FN, and TN counts |

Metrics are computed for both *nodule-present* and *clean* datasets.

---

## 7. Example Usage (Programmatic)

```python
from Unet.unet_model import UNet
import torch, numpy as np

# Load model
model = UNet(n_channels=1, n_classes=1)
model.load_state_dict(torch.load('model_outputs/UNET/model.pth'))
model.eval()

# Example inference
img = np.load('/home/LUNG_DATA/Image/LIDC-IDRI-0001_slice001.npy')
input_tensor = torch.tensor(img).unsqueeze(0).unsqueeze(0).float()

with torch.no_grad():
    output = torch.sigmoid(model(input_tensor))
```

---

## 8. Notes

- Ensure dataset preprocessing is complete before training.  
- Maintain identical directory names for compatibility (`Image`, `Mask`, `Clean`, `meta_csv`).  
- GPU acceleration is highly recommended for training.  
- All outputs are saved as `.npy` arrays for interoperability with NumPy and PyTorch workflows.

---

## 9. Citation

- **Jaeho Kim (2020)** – *LIDC-IDRI Segmentation (U-Net, Nested U-Net)*  
  GitHub: [https://github.com/jaeho3690/LIDC-IDRI-Segmentation](https://github.com/jaeho3690/LIDC-IDRI-Segmentation)
