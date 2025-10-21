import argparse
import os
from collections import OrderedDict
from glob import glob

import pandas as pd
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
import yaml
from torch.optim import lr_scheduler
from tqdm import tqdm

from dataset import MyLidcDataset
from losses import BCEDiceLoss
from metrics import iou_score, dice_coef
from utils import AverageMeter, str2bool
from Unet.unet_model import UNet
from UnetNested.Nested_Unet import NestedUNet



# parsing argument here
def build_args():
    """Define and parse command-line arguments."""
    p = argparse.ArgumentParser()

    # Model configuration done
    p.add_argument('--name', default='UNET', choices=['UNET', 'NestedUNET'])
    p.add_argument('--epochs', type=int, default=100)
    p.add_argument('-b', '--batch_size', type=int, default=8)
    p.add_argument('--early_stopping', type=int, default=50)
    p.add_argument('--num_workers', type=int, default=8)

    # Optimizer configuration
    p.add_argument('--optimizer', default='Adam', choices=['Adam', 'SGD'])
    p.add_argument('--lr', '--learning_rate', dest='lr', type=float, default=1e-5)
    p.add_argument('--momentum', type=float, default=0.9)
    p.add_argument('--weight_decay', type=float, default=1e-4)
    p.add_argument('--nesterov', type=str2bool, default=False)

    # Data settings for aug
    p.add_argument('--augmentation', type=str2bool, default=False, choices=[True, False])
    return p.parse_args()



# Train/validation epoch runner
def _run_epoch(data_loader, model, criterion, device, mode='train', optimizer=None):
    """
    Run one epoch for either training or validation.
    Computes loss, IoU, and Dice metrics.
    """
    is_train = (mode == 'train')
    model.train(mode=is_train)

    meters = {'loss': AverageMeter(), 'iou': AverageMeter(), 'dice': AverageMeter()}
    bar = tqdm(data_loader, total=len(data_loader), leave=False)

    with torch.set_grad_enabled(is_train):
        for images, masks in bar:
            images = images.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)

            # Forward pass
            logits = model(images)
            loss = criterion(logits, masks)
            iou = iou_score(logits, masks)
            dice = dice_coef(logits, masks)

            # Backpropagation only in training mode
            if is_train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

            # Metric tracking
            meters['loss'].update(loss.item(), images.size(0))
            meters['iou'].update(iou, images.size(0))
            meters['dice'].update(dice, images.size(0))

            # Progress display
            bar.set_description(
                f"{mode} | loss:{meters['loss'].avg:.4f} "
                f"iou:{meters['iou'].avg:.4f} dice:{meters['dice'].avg:.4f}"
            )

    return OrderedDict([
        ('loss', meters['loss'].avg),
        ('iou', meters['iou'].avg),
        ('dice', meters['dice'].avg),
    ])


# training mai pipeline
def main():
    cfg = vars(build_args())

    # Prepare output directory
    tag = f"{cfg['name']}_{'with_augmentation' if cfg['augmentation'] else 'base'}"
    out_dir = os.path.join('model_outputs', tag)
    os.makedirs(out_dir, exist_ok=True)

    print(f"[Init] Output dir: {out_dir}")
    print("[Config]")
    for k, v in cfg.items():
        print(f"  - {k}: {v}")

    # Save configurations
    with open(os.path.join(out_dir, 'config.yml'), 'w') as f:
        yaml.safe_dump(cfg, f)

    # Setup device and CUD
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    cudnn.benchmark = True

    # Initializeing model
    print(f"[Model] Building {cfg['name']}")
    model = NestedUNet(num_classes=1) if cfg['name'] == 'NestedUNET' else UNet(n_channels=1, n_classes=1, bilinear=True)

    # Multi-GPU checks
    if torch.cuda.device_count() > 1:
        print(f"[Model] Using DataParallel (GPUs: {torch.cuda.device_count()})")
        model = nn.DataParallel(model)
    model = model.to(device)

    # Optimizer setups
    params = (p for p in model.parameters() if p.requires_grad)
    if cfg['optimizer'] == 'Adam':
        optimizer = optim.Adam(params, lr=cfg['lr'], weight_decay=cfg['weight_decay'])
    else:
        optimizer = optim.SGD(
            params, lr=cfg['lr'],
            momentum=cfg['momentum'],
            nesterov=cfg['nesterov'],
            weight_decay=cfg['weight_decay']
        )

    # Loss function
    criterion = BCEDiceLoss().to(device)

# Dataset loading

    #IMAGE_DIR = '/content/drive/MyDrive/LUNG_DATA/Image/'
    #MASK_DIR = '/content/drive/MyDrive/LUNG_DATA/Mask/'
    #meta = pd.read_csv('/content/drive/MyDrive/LUNG_DATA/meta_csv/meta.csv')
    IMAGE_DIR = "../data/Image/"
    MASK_DIR = "../data/Mask/"
    meta = pd.read_csv('../data/Meta/meta.csv')


    meta['original_image'] = meta['original_image'].apply(lambda x: os.path.join(IMAGE_DIR, f"{x}.npy"))
    meta['mask_image'] = meta['mask_image'].apply(lambda x: os.path.join(MASK_DIR, f"{x}.npy"))

    # Split into train/validation
    train_meta = meta[meta['data_split'] == 'Train']
    val_meta = meta[meta['data_split'] == 'Validation']

    train_images, train_masks = train_meta['original_image'].tolist(), train_meta['mask_image'].tolist()
    val_images, val_masks = val_meta['original_image'].tolist(), val_meta['mask_image'].tolist()

    print("=" * 60)
    print(f"[Data] Train: {len(train_images)} images, {len(train_masks)} masks")
    print(f"[Data] Val  : {len(val_images)} images, {len(val_masks)} masks")
    print(f"[Data] Val/Train ratio: {len(val_images) / max(1, len(train_images)):.2f}")
    print("=" * 60)

    # Build dataset and dataloaders
    train_ds = MyLidcDataset(train_images, train_masks, cfg['augmentation'])
    val_ds = MyLidcDataset(val_images, val_masks, cfg['augmentation'])
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=cfg['batch_size'],
                                               shuffle=True, num_workers=cfg['num_workers'],
                                               pin_memory=True, drop_last=True)
    val_loader = torch.utils.data.DataLoader(val_ds, batch_size=cfg['batch_size'],
                                             shuffle=False, num_workers=cfg['num_workers'],
                                             pin_memory=True, drop_last=False)

    #Trying training loop
    best_dice = -1.0
    epochs_no_improve = 0
    logs = []  # collect logs for CSV output

    for ep in range(1, cfg['epochs'] + 1):
        train_log = _run_epoch(train_loader, model, criterion, device, mode='train', optimizer=optimizer)
        val_log = _run_epoch(val_loader, model, criterion, device, mode='val')

        print(f"[Epoch {ep}/{cfg['epochs']}] "
              f"Train -> loss:{train_log['loss']:.4f}, dice:{train_log['dice']:.4f}, iou:{train_log['iou']:.4f} | "
              f"Val -> loss:{val_log['loss']:.4f}, dice:{val_log['dice']:.4f}, iou:{val_log['iou']:.4f}")

        # Store each epoch results
        logs.append({
            'epoch': ep, 'lr': cfg['lr'],
            'loss': train_log['loss'], 'iou': train_log['iou'], 'dice': train_log['dice'],
            'val_loss': val_log['loss'], 'val_iou': val_log['iou'], 'val_dice': val_log['dice']
        })

        # Save the best model
        if val_log['dice'] > best_dice:
            best_dice = val_log['dice']
            epochs_no_improve = 0
            torch.save(model.state_dict(), os.path.join(out_dir, 'model.pth'))
            print(f"[Checkpoint] Best model saved (Dice={best_dice:.4f})")
        else:
            epochs_no_improve += 1

        # Early stopping
        if 0 <= cfg['early_stopping'] <= epochs_no_improve:
            print(f"[Early Stop] No improvement for {epochs_no_improve} epochs.")
            break

        torch.cuda.empty_cache()

    # Write logs to CSV
    log_df = pd.DataFrame(logs, columns=['epoch', 'lr', 'loss', 'iou', 'dice', 'val_loss', 'val_iou', 'val_dice'])
    log_path = os.path.join(out_dir, 'log.csv')
    log_df.to_csv(log_path, index=False)
    print(f"[Finish] Training complete. Log saved to: {log_path}")


if __name__ == '__main__':
    main()
