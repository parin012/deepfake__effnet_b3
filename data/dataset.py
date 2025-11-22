#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
from pathlib import Path
from PIL import Image, UnidentifiedImageError

import torch
from torch.utils.data import Dataset, DataLoader

from configs.config import (
    TRAIN_ROOTS, VAL_ROOTS, CROP_CACHE_DIR,
    IMG_SIZE, BATCH_SIZE, SKIP_IF_NO_CACHE
)
from data.preprocess import get_transforms
from train.utils import gather_image_pairs, cache_path_for


class CachedFaceDataset(Dataset):
    def __init__(self, roots, transform, skip_if_no_cache=True):
        self.transform = transform
        self.skip_if_no_cache = skip_if_no_cache

        srcs = gather_image_pairs(roots)
        samples = []

        for p, y in srcs:
            cp = cache_path_for(p)
            if Path(cp).exists():
                samples.append((cp, y))
            elif not skip_if_no_cache:
                samples.append((p, y))  

        if len(samples) == 0:
            raise RuntimeError(
                f"No samples found. Check CROP_CACHE_DIR={CROP_CACHE_DIR}"
            )

        self.samples = samples

    def __len__(self):
        return len(self.samples)

    @staticmethod
    def _safe_load(path: str):
        try:
            return Image.open(path).convert("RGB")
        except (UnidentifiedImageError, OSError, FileNotFoundError):
            return Image.fromarray(np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8))

    def __getitem__(self, idx):
        path, y = self.samples[idx]
        img = self._safe_load(path)
        x = self.transform(img)
        return x, torch.tensor([y], dtype=torch.float32)


def create_dataloaders():
    tf_train = get_transforms(is_train=True)
    tf_val = get_transforms(is_train=False)

    train_ds = CachedFaceDataset(
        TRAIN_ROOTS, transform=tf_train, skip_if_no_cache=SKIP_IF_NO_CACHE
    )
    val_ds = CachedFaceDataset(
        VAL_ROOTS, transform=tf_val, skip_if_no_cache=SKIP_IF_NO_CACHE
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        prefetch_factor=4,
        persistent_workers=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        prefetch_factor=4,
        persistent_workers=True,
    )

    return train_loader, val_loader, train_ds, val_ds

