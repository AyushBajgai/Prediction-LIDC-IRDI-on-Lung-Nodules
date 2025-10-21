import numpy as np
import torch
import torch.nn.functional as F

def iou_score(pred, true, eps=1e-5):
    if torch.is_tensor(pred):
        pred = torch.sigmoid(pred).detach().cpu().numpy()
    if torch.is_tensor(true):
        true = true.detach().cpu().numpy()

    pred_bin = pred > 0.5
    true_bin = true > 0.5

    inter = np.logical_and(pred_bin, true_bin).sum()
    union = np.logical_or(pred_bin, true_bin).sum()
    return (inter + eps) / (union + eps)


def dice_coef(pred, true, eps=1e-5):
    pred = torch.sigmoid(pred).flatten().detach().cpu().numpy()
    true = true.flatten().detach().cpu().numpy()
    inter = np.sum(pred * true)
    return (2 * inter + eps) / (pred.sum() + true.sum() + eps)


def dice_coef_binary(pred, true, eps=1e-5):
    pred = (pred.flatten() > 0.5).float().cpu().numpy()
    true = true.flatten().detach().cpu().numpy()
    inter = np.sum(pred * true)
    return (2 * inter + eps) / (pred.sum() + true.sum() + eps)
