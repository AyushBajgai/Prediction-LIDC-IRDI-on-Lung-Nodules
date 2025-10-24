import torch
import torch.nn as nn
import torch.nn.functional as F


class BCEDiceLoss(nn.Module):
    """
    Combined Binary Cross-Entropy (BCE) and Soft Dice loss.
    Used for binary segmentation tasks (e.g., nodule vs background).

    Features:
    - from_logits: applies sigmoid internally if True
    - smooth term avoids division by zero
    - weight_bce / weight_dice allow easy tuning of both terms
    """
    def __init__(self, from_logits: bool = True, weight_bce: float = 0.5, weight_dice: float = 1.0, smooth: float = 1e-6):
        super().__init__()
        self.from_logits = from_logits
        self.weight_bce = weight_bce
        self.weight_dice = weight_dice
        self.smooth = smooth

    def forward(self, preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            preds: (N, 1, H, W) raw logits or probabilities
            targets: (N, 1, H, W) binary masks
        """
        # Convert logits → probabilities if needed
        probs = torch.sigmoid(preds) if self.from_logits else preds

        # BCE part
        bce_loss = F.binary_cross_entropy_with_logits(preds, targets) if self.from_logits else F.binary_cross_entropy(probs, targets)

        # Dice part
        probs_flat = probs.contiguous().view(probs.size(0), -1)
        targets_flat = targets.contiguous().view(targets.size(0), -1)

        intersection = (probs_flat * targets_flat).sum(1)
        dice_score = (2. * intersection + self.smooth) / (
            probs_flat.sum(1) + targets_flat.sum(1) + self.smooth
        )
        dice_loss = 1 - dice_score.mean()

        # Weighted sum
        total_loss = self.weight_bce * bce_loss + self.weight_dice * dice_loss
        return total_loss
