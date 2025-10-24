import os
import argparse
from collections import OrderedDict

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import yaml
from tqdm import tqdm
from scipy import ndimage as ndi
from scipy.ndimage import label, generate_binary_structure

from dataset import MyLidcDataset
from metrics import iou_score, dice_coef, dice_coef2
from utils import AverageMeter, str2bool

from Unet.unet_model import UNet
from UnetNested.Nested_Unet import NestedUNet
from MAWNet_dual_encoder import MAWNetDualEncoder


# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--name",
        default="UNET",
        choices=["UNET", "NestedUNET", "MAWNET"],
        help="Model name: UNET | NestedUNET | MAWNET",
    )
    parser.add_argument(
        "--augmentation",
        default=False,
        type=str2bool,
        help="Use augmented model output directory or not",
    )
    return parser.parse_args()


def save_output(output, output_dir, image_paths, counter):
    """Save model predictions as .npy files."""
    for i in range(output.shape[0]):
        label_name = image_paths[counter][-23:]
        label_name = label_name.replace("NI", "PD")
        np.save(os.path.join(output_dir, label_name), output[i, :, :])
        counter += 1
    return counter


def calculate_fp(prediction_dir, mask_dir, distance_threshold=80):
    """Calculate confusion matrix for positive test set."""
    confusion = [0, 0, 0, 0]  # [TP, TN, FP, FN]
    s = generate_binary_structure(2, 2)
    print("Number of predictions:", len(os.listdir(prediction_dir)))

    for pred_name in os.listdir(prediction_dir):
        pid = "LIDC-IDRI-" + pred_name[:4]
        mask_name = pred_name.replace("PD", "MA")

        mask = np.load(os.path.join(mask_dir, pid, mask_name))
        predict = np.load(os.path.join(prediction_dir, pred_name))
        mask_com = np.array(ndi.center_of_mass(mask))

        labeled, num_features = label(predict, structure=s)
        if num_features == 0:
            confusion[3] += 1  # FN
            continue

        tp_found = False
        for n in range(num_features):
            region = (labeled == (n + 1)).astype(np.uint8)
            predict_com = np.array(ndi.center_of_mass(region))
            if np.linalg.norm(predict_com - mask_com, 2) < distance_threshold:
                tp_found = True
            else:
                confusion[2] += 1  # FP
        confusion[0 if tp_found else 3] += 1

    return np.array(confusion)


def calculate_fp_clean_dataset(prediction_dir, distance_threshold=80):
    """Calculate confusion matrix for clean (negative) test set."""
    confusion = [0, 0, 0, 0]
    s = generate_binary_structure(2, 2)

    for pred_name in os.listdir(prediction_dir):
        predict = np.load(os.path.join(prediction_dir, pred_name))
        labeled, num_features = label(predict, structure=s)

        if num_features == 0:
            confusion[1] += 1  # TN
            continue

        previous_com = np.array([-1, -1])
        for n in range(num_features):
            region = (labeled == (n + 1)).astype(np.uint8)
            predict_com = np.array(ndi.center_of_mass(region))
            if previous_com[0] == -1:
                confusion[2] += 1  # FP
                previous_com = predict_com
            elif np.linalg.norm(previous_com - predict_com, 2) > distance_threshold:
                confusion[2] += 1
                previous_com = predict_com
    return np.array(confusion)


def main():
    args = vars(parse_args())

    name = f"{args['name']}_with_augmentation" if args["augmentation"] else f"{args['name']}_base"

    # Load configuration
    with open(f"model_outputs/{name}/config.yml", "r") as f:
        config = yaml.safe_load(f)

    print("-" * 20)
    for k, v in config.items():
        print(f"{k}: {v}")
    print("-" * 20)

    cudnn.benchmark = True

    # Model setup
    print(f"=> Loading model: {name}")
    model_type = config["name"].upper()
    if model_type in ["MAWNET", "MAW-NET", "MAW"]:
        model = MAWNetDualEncoder(in_channels=1, out_channels=1)
    elif model_type == "NESTEDUNET":
        model = NestedUNet(num_classes=1)
    elif model_type == "UNET":
        model = UNet(n_channels=1, n_classes=1, bilinear=True)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    # Multi-GPU
    if torch.cuda.is_available() and torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs")
        model = nn.DataParallel(model)

    print(f"Loading checkpoint from {name}")
    state_dict = torch.load(f"model_outputs/{name}/model.pth", map_location="cpu")
    model.load_state_dict(state_dict)
    model = model.to(device)

    # Test data paths
    IMAGE_DIR = "/content/drive/MyDrive/LUNG_DATA/Image/"
    MASK_DIR = "/content/drive/MyDrive/LUNG_DATA/Mask/"
    meta = pd.read_csv("/content/drive/MyDrive/LUNG_DATA/meta_csv/meta.csv")

    meta["original_image"] = meta["original_image"].apply(lambda x: IMAGE_DIR + x + ".npy")
    meta["mask_image"] = meta["mask_image"].apply(lambda x: MASK_DIR + x + ".npy")
    test_meta = meta[meta["data_split"] == "Test"]

    test_imgs = list(test_meta["original_image"])
    test_masks = list(test_meta["mask_image"])
    total_patients = len(test_meta.groupby("patient_id"))

    print("*" * 50)
    print(f"Test images: {len(test_imgs)}, Test masks: {len(test_masks)}")
    print(f"Total patients: {total_patients}")

    OUTPUT_DIR = f"/content/drive/MyDrive/LUNG_DATA/Segmentation_output/{name}"
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"Saving outputs to: {OUTPUT_DIR}")

    test_dataset = MyLidcDataset(test_imgs, test_masks)
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=config.get("batch_size", 8),
        shuffle=False,
        pin_memory=True,
        drop_last=False,
        num_workers=config.get("num_workers", 6),
    )

    # Evaluation
    model.eval()
    print("Preview sample test files:")
    print(test_imgs[:5])

    avg_meters = {"iou": AverageMeter(), "dice": AverageMeter()}

    with torch.no_grad():
        counter = 0
        pbar = tqdm(total=len(test_loader))
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)

            iou = iou_score(outputs, targets)
            dice = dice_coef2(outputs, targets)

            avg_meters["iou"].update(iou, inputs.size(0))
            avg_meters["dice"].update(dice, inputs.size(0))

            preds = torch.sigmoid(outputs)
            preds = (preds > 0.5).float().cpu().numpy()
            preds = np.squeeze(preds, axis=1)
            counter = save_output(preds, OUTPUT_DIR, test_imgs, counter)

            pbar.set_postfix({"iou": avg_meters["iou"].avg, "dice": avg_meters["dice"].avg})
            pbar.update(1)
        pbar.close()

    print("=" * 50)
    print(f"IoU: {avg_meters['iou'].avg:.4f}")
    print(f"DICE: {avg_meters['dice'].avg:.4f}")

    confusion = calculate_fp(OUTPUT_DIR, MASK_DIR)
    print("=" * 50)
    print(f"TP: {confusion[0]} FP: {confusion[2]}")
    print(f"FN: {confusion[3]} TN: {confusion[1]}")
    print(f"{confusion[2] / max(1, total_patients):.2f} FP per scan")
    print("=" * 50)

    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()


