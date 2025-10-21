import torch
import torch.nn as nn
import torch.nn.functional as F

class BCEDiceLoss(nn.Module):
    def __init__(self, smooth=1e-5, bce_weight=0.5):
        super().__init__()
        self.smooth = smooth
        self.bce_weight = bce_weight

    def forward(self, preds, targets):
        bce_loss = F.binary_cross_entropy_with_logits(preds, targets)
        preds = torch.sigmoid(preds)
        preds = preds.flatten(1)
        targets = targets.flatten(1)

        intersection = (preds * targets).sum(1)
        dice_loss = 1 - (2 * intersection + self.smooth) / (
            preds.sum(1) + targets.sum(1) + self.smooth
        )
        return self.bce_weight * bce_loss + dice_loss.mean()
