import argparse
import os
from pathlib import Path
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


#Argument parser
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--name', default="UNET", choices=['UNET', 'NestedUNET'])
    p.add_argument('--augmentation', default=False, type=str2bool)
    return vars(p.parse_args())


#Helper function
def save_output(pred_batch, out_dir, src_paths, start_idx):
    """Save binarized predictions, following the original naming rule."""
    count = start_idx
    for i in range(pred_batch.shape[0]):
        # last 23 chars, replace NI->PD
        label = src_paths[count][-23:].replace('NI', 'PD')
        np.save(str(Path(out_dir) / label), pred_batch[i, :, :])
        count += 1
    return count


#FP positivity clean function
def calculate_fp(pred_dir, mask_root, distance_threshold=80):
    """TP/TN/FP/FN on positive set; distance filter between COMs."""
    cm = [0, 0, 0, 0]  # TP, TN, FP, FN
    conn = generate_binary_structure(2, 2)
    files = list(Path(pred_dir).iterdir())
    print('Number of predicted masks:', len(files))

    for p in files:
        prediction = np.load(str(p))
        pid = 'LIDC-IDRI-' + p.name[:4]
        mask_id = p.name.replace('PD', 'MA')
        mask = np.load(str(Path(mask_root) / pid / mask_id))

        ans_com = np.array(ndi.center_of_mass(mask))
        patience = 0

        labeled, n_regions = label(prediction, structure=conn)
        if n_regions > 0:
            for n in range(n_regions):
                # isolate component n+1
                comp = (labeled == (n + 1)).astype(np.uint8)
                pred_com = np.array(ndi.center_of_mass(comp))
                if np.linalg.norm(pred_com - ans_com, 2) < distance_threshold:
                    patience += 1
                else:
                    cm[2] += 1  # FP
            if patience > 0:
                cm[0] += 1  # TP
            else:
                cm[3] += 1  # FN
        else:
            cm[3] += 1  # FN when nothing predicted
    return np.array(cm)

#FP cleaning for distance thresoldf
def calculate_fp_clean_dataset(pred_dir, distance_threshold=80):
    """FP/TN on clean set (no positives expected)."""
    cm = [0, 0, 0, 0]  # TP, TN, FP, FN (TP/FN unused here)
    conn = generate_binary_structure(2, 2)

    for p in Path(pred_dir).iterdir():
        pred = np.load(str(p))
        labeled, n_regions = label(pred, structure=conn)

        if n_regions > 0:
            patience = 0
            prev_com = np.array([-1, -1])
            for n in range(n_regions):
                comp = (labeled == (n + 1)).astype(np.uint8)
                cur_com = np.array(ndi.center_of_mass(comp))
                if prev_com[0] == -1:
                    cm[2] += 1  # first component is FP
                    prev_com = cur_com
                    continue
                # treat distant extra components as additional FPs
                if np.linalg.norm(prev_com - cur_com, 2) > distance_threshold:
                    if patience == 0:
                        cm[2] += 1
                        patience += 1
        else:
            cm[1] += 1  # TN
    return np.array(cm)


#Splitting for evaluation
def run_eval_split(model, loader, device, out_dir, save_paths, title='EVAL'):
    meters = {'iou': AverageMeter(), 'dice': AverageMeter()}
    idx = 0
    model.eval()

    with torch.no_grad():
        pbar = tqdm(loader, total=len(loader), desc=f"[{title}]")
        for imgs, gts in pbar:
            imgs = imgs.to(device, non_blocking=True)
            gts = gts.to(device, non_blocking=True)

            logits = model(imgs)
            iou = iou_score(logits, gts)
            dice = dice_coef2(logits, gts)

            meters['iou'].update(iou, imgs.size(0))
            meters['dice'].update(dice, imgs.size(0))

            # binarize and save
            probs = torch.sigmoid(logits)
            preds = (probs > 0.5).float().cpu().numpy()
            preds = np.squeeze(preds, axis=1)  # [B, H, W]

            idx = save_output(preds, out_dir, save_paths, idx)
            pbar.set_postfix(OrderedDict(iou=meters['iou'].avg, dice=meters['dice'].avg))
    return meters


#Main for argument parsing
def main():
    args = parse_args()

    # choose run name
    NAME = f"{args['name']}_{'with_augmentation' if args['augmentation'] else 'base'}"

    # load saved config
    cfg_path = Path('model_outputs') / NAME / 'config.yml'
    with open(cfg_path, 'r') as f:
        config = yaml.safe_load(f)

    print('-' * 20)
    for k, v in config.items():
        print(f"{k}: {v}")
    print('-' * 20)

    # device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    cudnn.benchmark = True

    # build model
    print(f"=> building model: {NAME}")
    if config['name'] == 'NestedUNET':
        model = NestedUNet(num_classes=1)
    else:
        model = UNet(n_channels=1, n_classes=1, bilinear=True)

    if torch.cuda.device_count() > 1:
        print(f"Using DataParallel on {torch.cuda.device_count()} GPUs")
        model = nn.DataParallel(model)

    weights = Path('model_outputs') / NAME / 'model.pth'
    print(f"Loading weights from: {weights}")
    model.load_state_dict(torch.load(str(weights), map_location=device))
    model = model.to(device)

    # Prepare test set
    #IMAGE_DIR = Path('/content/drive/MyDrive/LUNG_DATA/Image/')
    #MASK_DIR = Path('/content/drive/MyDrive/LUNG_DATA/Mask/')
    #meta = pd.read_csv('/content/drive/MyDrive/LUNG_DATA/meta_csv/meta.csv')

    IMAGE_DIR = "../data/Image/"
    MASK_DIR = "../data/Mask/"
    meta = pd.read_csv('../data/Meta/meta.csv')

    meta['original_image'] = meta['original_image'].apply(lambda x: str(IMAGE_DIR / f"{x}.npy"))
    meta['mask_image'] = meta['mask_image'].apply(lambda x: str(MASK_DIR / f"{x}.npy"))
    test_meta = meta[meta['data_split'] == 'Test']

    test_images = test_meta['original_image'].tolist()
    test_masks = test_meta['mask_image'].tolist()
    n_patients = len(test_meta.groupby('patient_id'))

    print("*" * 50)
    print(f"[TEST] images={len(test_images)}, masks={len(test_masks)}")
    print(f"[TEST] patients={n_patients}")

    out_dir = Path('/content/drive/MyDrive/LUNG_DATA/Segmentation_output') / NAME
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[TEST] saving predictions to: {out_dir}")

    test_ds = MyLidcDataset(test_images, test_masks)
    test_loader = torch.utils.data.DataLoader(
        test_ds, batch_size=config['batch_size'],
        shuffle=False, pin_memory=True, drop_last=False, num_workers=6
    )

    print("\n[DEBUG] sample image paths:", test_images[:5])
    print("[DEBUG] sample mask  paths:", test_masks[:5])

    # Preparing clean test
    #CLEAN_IMG = Path('/content/drive/MyDrive/LUNG_DATA/Clean/Image/')
    #CLEAN_MSK = Path('/content/drive/MyDrive/LUNG_DATA/Mask/')
    #clean_meta = pd.read_csv('/content/drive/MyDrive/LUNG_DATA/meta_csv/clean_meta.csv')

    CLEAN_IMG = "../LUNG_DATA/Clean/Image/"
    CLEAN_MSK = "../LUNG_DATA/Mask/"
    clean_meta = pd.read_csv('../LUNG_DATA/Meta/clean_meta.csv')

    clean_meta['original_image'] = clean_meta['original_image'].apply(lambda x: str(CLEAN_IMG / f"{x}.npy"))
    clean_meta['mask_image'] = clean_meta['mask_image'].apply(lambda x: str(CLEAN_MSK / f"{x}.npy"))
    clean_test_meta = clean_meta[clean_meta['data_split'] == 'Test']

    clean_images = clean_test_meta['original_image'].tolist()
    clean_masks = clean_test_meta['mask_image'].tolist()
    clean_patients = len(clean_test_meta.groupby('patient_id'))

    print("*" * 50)
    print(f"[CLEAN] images={len(clean_images)}, masks={len(clean_masks)}")
    print(f"[CLEAN] patients={clean_patients}")

    clean_name = f"CLEAN_{NAME}"
    clean_out_dir = Path('/content/drive/MyDrive/LUNG_DATA/Segmentation_output') / clean_name
    clean_out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[CLEAN] saving predictions to: {clean_out_dir}")

    clean_ds = MyLidcDataset(clean_images, clean_masks)
    clean_loader = torch.utils.data.DataLoader(
        clean_ds, batch_size=config['batch_size'],
        shuffle=False, pin_memory=True, drop_last=False, num_workers=6
    )

    # Evaluate both splits explicitly
    test_m = run_eval_split(model, test_loader, device, out_dir, test_images, title='TEST')
    print("\n" + "=" * 50)
    print(f"[TEST] IoU : {test_m['iou'].avg:.4f}")
    print(f"[TEST] Dice: {test_m['dice'].avg:.4f}")

    cm_test = calculate_fp(out_dir, MASK_DIR, distance_threshold=80)
    print("=" * 50)
    print(f"[TEST] TP={cm_test[0]} FP={cm_test[2]} | FN={cm_test[3]} TN={cm_test[1]}")
    print(f"[TEST] FP/scan: {cm_test[2] / max(1, n_patients):.2f}")

    print("\n[INFO] Evaluating clean split...")
    clean_m = run_eval_split(model, clean_loader, device, clean_out_dir, clean_images, title='CLEAN')
    print("\n" + "=" * 50)
    print(f"[CLEAN] IoU : {clean_m['iou'].avg:.4f}")
    print(f"[CLEAN] Dice: {clean_m['dice'].avg:.4f}")

    cm_clean = calculate_fp_clean_dataset(clean_out_dir)
    print(f"[CLEAN] confusion: {cm_clean}")

    # total summary
    cm_total = cm_clean + cm_test
    total_patients = n_patients + clean_patients

    print("=" * 50)
    print(f"[ALL] TP={cm_total[0]} FP={cm_total[2]} | FN={cm_total[3]} TN={cm_total[1]}")
    print(f"[ALL] FP/scan: {cm_total[2] / max(1, total_patients):.2f}")
    print(f"[ALL] total patients={total_patients}, clean patients={clean_patients}")
    print("=" * 50 + "\n")

    torch.cuda.empty_cache()


if __name__ == '__main__':
    main()

