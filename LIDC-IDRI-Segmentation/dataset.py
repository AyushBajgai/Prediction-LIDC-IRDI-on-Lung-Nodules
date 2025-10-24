import os
import numpy as np
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T
import albumentations as A
from albumentations.pytorch import ToTensorV2
from typing import List, Sequence, Optional, Tuple


class MyLidcDataset(Dataset):
    """
    LIDC-IDRI style dataset for .npy images & masks.

    Behavior:
      - If use_albumentation=True: uses Albumentations pipeline (HWC numpy -> CHW tensor)
      - If use_albumentation=False: uses torchvision pipeline for images and direct tensor conversion for masks
      - Keeps the same input/output behavior as the original version
    """
    def __init__(self,
                 image_paths: Sequence[str],
                 mask_paths: Sequence[str],
                 use_albumentation: bool = False,
                 albumentation_transforms: Optional[A.BasicTransform] = None,
                 torch_transforms: Optional[T.Compose] = None) -> None:
        assert len(image_paths) == len(mask_paths), "Images and masks must align"
        self.image_paths: List[str] = list(image_paths)
        self.mask_paths: List[str] = list(mask_paths)
        self.use_albu: bool = bool(use_albumentation)

        if albumentation_transforms is None:
            self.albu_transform = A.Compose([
                A.ElasticTransform(alpha=1.1, alpha_affine=0.5, sigma=5, p=0.15),
                A.HorizontalFlip(p=0.15),
                A.Normalize(mean=[0.485], std=[0.229]),  # single-channel normalization
                ToTensorV2()
            ])
        else:
            self.albu_transform = albumentation_transforms

        if torch_transforms is None:
            self.torch_transform = T.Compose([
                T.ToTensor(),
                T.Normalize(mean=[0.485], std=[0.229]),
            ])
        else:
            self.torch_transform = torch_transforms

    def __len__(self) -> int:
        return len(self.image_paths)

    def _load_pair(self, idx: int) -> Tuple[np.ndarray, np.ndarray]:
        """Load image and mask from disk."""
        image = np.load(self.image_paths[idx])
        mask = np.load(self.mask_paths[idx])

        if image.ndim == 2:
            image = image[..., None]
        if mask.ndim == 2:
            mask = mask[..., None]

        if image.dtype != np.float32:
            image = image.astype(np.float32, copy=False)
        if not image.flags.c_contiguous:
            image = np.ascontiguousarray(image)

        if mask.dtype != np.float32:
            mask = mask.astype(np.float32, copy=False)
        if not mask.flags.c_contiguous:
            mask = np.ascontiguousarray(mask)

        return image, mask

    def __getitem__(self, idx: int):
        image, mask = self._load_pair(idx)

        if self.use_albu:
            augmented = self.albu_transform(image=image, mask=mask)
            image_t = augmented["image"]           # Tensor C,H,W
            mask_t = augmented["mask"]             # Tensor C,H,W
        else:
            image_t = self.torch_transform(image)
            mask_t = torch.from_numpy(mask).float()
            if mask_t.ndim == 2:
                mask_t = mask_t.unsqueeze(0)       # 1,H,W

        return image_t.float(), mask_t.float()


