import os
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import albumentations as albu
from albumentations.pytorch import ToTensorV2


class MyLidcDataset(Dataset):
    """Custom dataset for LIDC lung CT images and masks."""

    def __init__(self, image_paths, mask_paths, use_augmentation=False):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.use_augmentation = use_augmentation

        # Albumentations-based data augmentation
        self.albu_transform = albu.Compose([
            albu.ElasticTransform(alpha=1.1, alpha_affine=0.5, sigma=5, p=0.15),
            albu.HorizontalFlip(p=0.15),
            ToTensorV2()
        ])

        # Simple tensor conversion (no augmentation)
        self.basic_transform = transforms.Compose([
            transforms.ToTensor()
        ])

    def _apply_transform(self, image, mask):
        """Apply augmentation or basic transform."""
        if self.use_augmentation:
            # Albumentations expects HWC format and uint8 masks
            image = image.reshape(512, 512, 1)
            mask = mask.reshape(512, 512, 1).astype(np.uint8)

            augmented = self.albu_transform(image=image, mask=mask)
            image, mask = augmented['image'], augmented['mask']
            mask = mask.unsqueeze(0)  # reshape to [1, H, W]
        else:
            image = self.basic_transform(image)
            mask = self.basic_transform(mask)

        return image.float(), mask.float()

    def __getitem__(self, index):
        """Load one image-mask pair."""
        image = np.load(self.image_paths[index])
        mask = np.load(self.mask_paths[index])
        image, mask = self._apply_transform(image, mask)
        return image, mask

    def __len__(self):
        """Return dataset length."""
        return len(self.image_paths)

