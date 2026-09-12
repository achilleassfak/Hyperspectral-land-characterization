"""Dataset loading and augmentation for the hyperspectral crop-health patches."""

import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


class SatelliteDataset(Dataset):
    """
    Loads (hyperspectral patch, health-class mask) pairs from a
    band-exclusion dataset folder (see notebooks/01_ground_truth_generation.ipynb),
    with optional spatial and spectral augmentation for training.
    """

    def __init__(self, data_root, num_classes, augment=False):
        self.rs_dir      = Path(data_root) / 'rs'
        self.gt_dir      = Path(data_root) / 'gt'
        self.filenames   = sorted(f.name for f in self.rs_dir.glob('*.npy'))
        self.num_classes = num_classes
        self.augment     = augment
        sample           = np.load(self.rs_dir / self.filenames[0])
        self.num_bands   = sample.shape[-1]

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        fname = self.filenames[idx]
        rs = np.nan_to_num(np.load(self.rs_dir / fname).astype(np.float32))
        gt = np.load(self.gt_dir / fname)

        if self.augment:
            # Spatial augmentations applied identically to RS and GT
            if random.random() > 0.5:
                rs = np.flip(rs, axis=1); gt = np.flip(gt, axis=1)
            if random.random() > 0.5:
                rs = np.flip(rs, axis=0); gt = np.flip(gt, axis=0)
            k = random.randint(0, 3)
            if k > 0:
                rs = np.rot90(rs, k=k, axes=(0, 1))
                gt = np.rot90(gt, k=k, axes=(0, 1))

            # CutMix: swap a random patch from another sample
            if random.random() > 0.7:
                idx2 = random.randint(0, len(self.filenames) - 1)
                rs2  = np.nan_to_num(np.load(self.rs_dir / self.filenames[idx2]).astype(np.float32))
                gt2  = np.load(self.gt_dir / self.filenames[idx2])
                cy, cx = random.randint(0, 64), random.randint(0, 64)
                h,  w  = random.randint(8, 32),  random.randint(8, 32)
                rs[cy:cy+h, cx:cx+w, :] = rs2[cy:cy+h, cx:cx+w, :]
                gt[cy:cy+h, cx:cx+w]    = gt2[cy:cy+h, cx:cx+w]

            # Spectral augmentations applied to RS only
            if random.random() > 0.5:
                rs += np.random.normal(0, 0.01, rs.shape).astype(np.float32)

            if random.random() > 0.7:
                n_drop = max(1, int(self.num_bands * 0.05))
                drop_idx = random.sample(range(self.num_bands), n_drop)
                rs[:, :, drop_idx] = 0.0

        gt   = np.clip(np.round(gt).astype(np.int64), 0, self.num_classes - 1)
        rs_t = torch.from_numpy(rs.copy()).permute(2, 0, 1)
        gt_t = torch.from_numpy(gt.copy()).long()
        if rs_t.max() > 1.0:
            rs_t = rs_t / rs_t.max()
        return rs_t, gt_t


def get_loaders(root_dir, dataset_base_name, level, num_classes, batch_size=16):
    """Builds train/val/test DataLoaders for a given band-exclusion level."""
    base = Path(root_dir) / dataset_base_name if level == 0 else Path(root_dir) / f"{dataset_base_name}_L{level}"
    kw   = dict(batch_size=batch_size, num_workers=0, pin_memory=True)
    train_ds = SatelliteDataset(base / 'Train' / 'Training',   num_classes, augment=True)
    val_ds   = SatelliteDataset(base / 'Train' / 'Validation', num_classes, augment=False)
    test_ds  = SatelliteDataset(base / 'Test',                 num_classes, augment=False)
    return (
        DataLoader(train_ds, shuffle=True,  **kw),
        DataLoader(val_ds,   shuffle=False, **kw),
        DataLoader(test_ds,  shuffle=False, **kw),
    )
