import numpy as np
import torch
import torch.nn.functional as F


def iou_score(output: torch.Tensor, target: torch.Tensor) -> float:

    smooth = 1e-5

    # Convert to numpy boolean masks
    if torch.is_tensor(output):
        output = torch.sigmoid(output).detach().cpu().numpy()
    if torch.is_tensor(target):
        target = target.detach().cpu().numpy()

    output_bin = output > 0.5
    target_bin = target > 0.5

    intersection = np.logical_and(output_bin, target_bin).sum()
    union = np.logical_or(output_bin, target_bin).sum()

    return float((intersection + smooth) / (union + smooth))


def dice_coef(output: torch.Tensor, target: torch.Tensor) -> float:
    smooth = 1e-5

    # Apply sigmoid since U-Net outputs logits
    output = torch.sigmoid(output).view(-1).detach().cpu().numpy()
    target = target.view(-1).detach().cpu().numpy()

    intersection = (output * target).sum()
    dice = (2. * intersection + smooth) / (output.sum() + target.sum() + smooth)

    return float(dice)


def dice_coef2(output: torch.Tensor, target: torch.Tensor) -> float:

    smooth = 1e-5

    output = (output.view(-1) > 0.5).float().cpu().numpy()
    target = target.view(-1).detach().cpu().numpy()

    intersection = (output * target).sum()
    dice = (2. * intersection + smooth) / (output.sum() + target.sum() + smooth)

    return float(dice)
