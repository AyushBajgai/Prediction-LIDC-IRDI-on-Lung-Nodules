import os
from pathlib import Path
from collections import OrderedDict
from typing import Dict, Tuple, List

import argparse
import yaml
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn

from torch.optim import lr_scheduler  # kept for parity (not used)
from tqdm import tqdm

import albumentations as albu  # kept for parity (import side-effects not required)

from losses import BCEDiceLoss
from dataset import MyLidcDataset
from metrics import iou_score, dice_coef
from utils import AverageMeter, str2bool

from Unet.unet_model import UNet
from UnetNested.Nested_Unet import NestedUNet
from MAWNet_dual_encoder import MAWNetDualEncoder


# ------------------------------
# Device
# ------------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ------------------------------
# Args
# ------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    # model
    parser.add_argument(
        "--name", default="UNET",
        choices=["UNET", "NestedUNET", "MAWNET"],
        help="model name: UNET | NestedUNET | MAWNET"
    )
    parser.add_argument("--epochs", default=100, type=int, metavar="N", help="number of total epochs to run")
    parser.add_argument("-b", "--batch_size", default=8, type=int, metavar="N", help="mini-batch size (default: 8)")
    parser.add_argument("--early_stopping", default=50, type=int, metavar="N", help="early stopping (default: 50)")
    parser.add_argument("--num_workers", default=8, type=int)

    # optimizer
    parser.add_argument(
        "--optimizer", default="Adam", choices=["Adam", "SGD"],
        help="optimizer: Adam | SGD (default: Adam)"
    )
    parser.add_argument("--lr", "--learning_rate", default=1e-5, type=float, metavar="LR", help="initial learning rate")
    parser.add_argument("--momentum", default=0.9, type=float, help="momentum (SGD)")
    parser.add_argument("--weight_decay", default=1e-4, type=float, help="weight decay")
    parser.add_argument("--nesterov", default=False, type=str2bool, help="nesterov (SGD)")

    # data
    parser.add_argument("--augmentation", type=str2bool, default=False, choices=[True, False])

    return parser.parse_args()


# ------------------------------
# Train / Validate
# ------------------------------
def train_one_epoch(
    loader: torch.utils.data.DataLoader,
    model: nn.Module,
    criterion: nn.Module,
    optimizer: optim.Optimizer
) -> OrderedDict:
    meters = {"loss": AverageMeter(), "iou": AverageMeter(), "dice": AverageMeter()}
    model.train()

    pbar = tqdm(total=len(loader))
    for images, targets in loader:
        images = images.to(DEVICE)
        targets = targets.to(DEVICE)

        outputs = model(images)
        loss = criterion(outputs, targets)
        iou = iou_score(outputs, targets)
        dice = dice_coef(outputs, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_size = images.size(0)
        meters["loss"].update(loss.item(), batch_size)
        meters["iou"].update(iou, batch_size)
        meters["dice"].update(dice, batch_size)

        pbar.set_postfix(OrderedDict([
            ("loss", meters["loss"].avg),
            ("iou",  meters["iou"].avg),
            ("dice", meters["dice"].avg),
        ]))
        pbar.update(1)
    pbar.close()

    return OrderedDict([
        ("loss", meters["loss"].avg),
        ("iou",  meters["iou"].avg),
        ("dice", meters["dice"].avg),
    ])


@torch.no_grad()
def validate_one_epoch(
    loader: torch.utils.data.DataLoader,
    model: nn.Module,
    criterion: nn.Module
) -> OrderedDict:
    meters = {"loss": AverageMeter(), "iou": AverageMeter(), "dice": AverageMeter()}
    model.eval()

    pbar = tqdm(total=len(loader))
    for images, targets in loader:
        images = images.to(DEVICE)
        targets = targets.to(DEVICE)

        outputs = model(images)
        loss = criterion(outputs, targets)
        iou = iou_score(outputs, targets)
        dice = dice_coef(outputs, targets)

        batch_size = images.size(0)
        meters["loss"].update(loss.item(), batch_size)
        meters["iou"].update(iou, batch_size)
        meters["dice"].update(dice, batch_size)

        pbar.set_postfix(OrderedDict([
            ("loss", meters["loss"].avg),
            ("iou",  meters["iou"].avg),
            ("dice", meters["dice"].avg),
        ]))
        pbar.update(1)
    pbar.close()

    return OrderedDict([
        ("loss", meters["loss"].avg),
        ("iou",  meters["iou"].avg),
        ("dice", meters["dice"].avg),
    ])


# ------------------------------
# Builders
# ------------------------------
def build_model(name: str) -> nn.Module:
    name_upper = name.upper()
    if name_upper in ["MAWNET", "MAW-NET", "MAW"]:
        return MAWNetDualEncoder(in_channels=1, out_channels=1)
    if name_upper == "NESTEDUNET":
        return NestedUNet(num_classes=1)
    if name_upper == "UNET":
        return UNet(n_channels=1, n_classes=1, bilinear=True)
    raise ValueError(f"Unknown model name: {name}")


def build_optimizer(
    model: nn.Module,
    opt_name: str,
    lr: float,
    weight_decay: float,
    momentum: float,
    nesterov: bool
) -> optim.Optimizer:
    params = filter(lambda p: p.requires_grad, model.parameters())

    if opt_name == "Adam":
        return optim.Adam(params, lr=lr, weight_decay=weight_decay)

    if opt_name == "SGD":
        return optim.SGD(
            params, lr=lr, momentum=momentum,
            nesterov=nesterov, weight_decay=weight_decay
        )

    raise NotImplementedError(f"Unsupported optimizer: {opt_name}")


def build_dataloaders(
    config: Dict
) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    # Directories created in preprocessing stage
    image_root = "/content/drive/MyDrive/LUNG_DATA/Image/"
    mask_root  = "/content/drive/MyDrive/LUNG_DATA/Mask/"
    meta_csv   = "/content/drive/MyDrive/LUNG_DATA/meta_csv/meta.csv"

    meta = pd.read_csv(meta_csv)

    # Build full paths
    meta["original_image"] = meta["original_image"].apply(lambda x: f"{image_root}{x}.npy")
    meta["mask_image"]     = meta["mask_image"].apply(lambda x: f"{mask_root}{x}.npy")

    train_meta = meta[meta["data_split"] == "Train"]
    val_meta   = meta[meta["data_split"] == "Validation"]

    train_img_paths = list(train_meta["original_image"])
    train_msk_paths = list(train_meta["mask_image"])
    val_img_paths   = list(val_meta["original_image"])
    val_msk_paths   = list(val_meta["mask_image"])

    print("*" * 50)
    print(f"The length of image: {len(train_img_paths)}, mask folders: {len(train_msk_paths)} for train")
    print(f"The length of image: {len(val_img_paths)}, mask folders: {len(val_msk_paths)} for validation")
    ratio = len(val_img_paths) / max(1, len(train_img_paths))
    print(f"Ratio between Val/Train is {ratio:.4f}")
    print("*" * 50)

    # Datasets (augmentation flag behavior unchanged)
    train_dataset = MyLidcDataset(train_img_paths, train_msk_paths, config["augmentation"])
    val_dataset   = MyLidcDataset(val_img_paths,   val_msk_paths,   config["augmentation"])

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config["batch_size"],
        shuffle=True,
        pin_memory=True,
        drop_last=True,
        num_workers=config["num_workers"],
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=config["batch_size"],
        shuffle=False,
        pin_memory=True,
        drop_last=False,
        num_workers=config["num_workers"],
    )

    return train_loader, val_loader


# ------------------------------
# Main
# ------------------------------
def main():
    # Config
    args = parse_args()
    config = vars(args)

    # Output directory name (unchanged logic)
    file_name = f"{config['name']}_with_augmentation" if config["augmentation"] else f"{config['name']}_base"
    out_dir = Path("model_outputs") / file_name
    out_dir.mkdir(parents=True, exist_ok=True)
    print("Creating directory called", file_name)

    # Print config
    print("-" * 20)
    print("Configuration Setting as follow")
    for k in config:
        print(f"{k}: {config[k]}")
    print("-" * 20)

    # Save config
    with open(out_dir / "config.yml", "w") as f:
        yaml.dump(config, f)

    # Loss & backend
    criterion = BCEDiceLoss().to(DEVICE)
    cudnn.benchmark = True

    # Model
    print("=> creating model")
    model = build_model(config["name"]).to(DEVICE)

    # Multi-GPU
    if torch.cuda.is_available() and torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = nn.DataParallel(model)

    # Optimizer
    optimizer = build_optimizer(
        model=model,
        opt_name=config["optimizer"],
        lr=config["lr"],
        weight_decay=config["weight_decay"],
        momentum=config["momentum"],
        nesterov=config["nesterov"],
    )

    # Data
    train_loader, val_loader = build_dataloaders(config)

    # Log dataframe (same columns/order)
    log_cols = ["epoch", "lr", "loss", "iou", "dice", "val_loss", "val_iou", "val_dice"]
    log_df = pd.DataFrame(index=[], columns=log_cols)

    best_dice = 0.0
    trigger = 0

    for epoch in range(config["epochs"]):
        train_log = train_one_epoch(train_loader, model, criterion, optimizer)
        val_log   = validate_one_epoch(val_loader, model, criterion)

        print(
            "Training epoch [{}/{}], Training BCE+DICE loss:{:.4f}, Training DICE:{:.4f}, Training IOU:{:.4f}, "
            "Validation BCE+DICE loss:{:.4f}, Validation Dice:{:.4f}, Validation IOU:{:.4f}".format(
                epoch + 1, config["epochs"],
                train_log["loss"], train_log["dice"], train_log["iou"],
                val_log["loss"],   val_log["dice"],   val_log["iou"]
            )
        )

        row = pd.Series(
            [
                epoch,
                config["lr"],
                train_log["loss"],
                train_log["iou"],
                train_log["dice"],
                val_log["loss"],
                val_log["iou"],
                val_log["dice"],
            ],
            index=log_cols,
        )

        log_df = pd.concat([log_df, row.to_frame().T], ignore_index=True)
        log_df.to_csv(out_dir / "log.csv", index=False)

        trigger += 1

        # Save best by validation Dice (unchanged)
        if val_log["dice"] > best_dice:
            torch.save(model.state_dict(), out_dir / "model.pth")
            best_dice = val_log["dice"]
            print("=> saved best model as validation DICE is greater than previous best DICE")
            trigger = 0

        # Early stopping (unchanged)
        if config["early_stopping"] >= 0 and trigger >= config["early_stopping"]:
            print("=> early stopping")
            break

        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
