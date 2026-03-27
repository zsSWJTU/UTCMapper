"""
Utility functions and configurations for UTCMapper.

Includes:
- Dataset configuration
- Label mapping
- Loss tracking
- Checkpoint loading (exclude head)
"""

import csv
import os
from collections import defaultdict

import numpy as np
import torch


# =========================
# Image normalization
# =========================
IMAGE_MEANS = np.array([101.16, 103.08, 87.93])
IMAGE_STDS = np.array([63.32, 56.17, 53.20])


# =========================
# Label definition
# =========================
LABEL_CLASSES = [0, 2]

LABEL_CLASS_COLORMAP = {
    0: (0, 0, 0),
    2: (153, 255, 204)
}

LABEL_IDX_COLORMAP = {
    idx: LABEL_CLASS_COLORMAP[c]
    for idx, c in enumerate(LABEL_CLASSES)
}


def get_label_class_to_idx_map():
    """Map raw label values to continuous indices."""
    label_to_idx_map = []
    idx = 0
    for i in range(LABEL_CLASSES[-1] + 1):
        if i in LABEL_CLASSES:
            label_to_idx_map.append(idx)
            idx += 1
        else:
            label_to_idx_map.append(0)
    return np.array(label_to_idx_map).astype(np.int64)


LABEL_CLASS_TO_IDX_MAP = get_label_class_to_idx_map()


# =========================
# Dataset configuration
# =========================
dataset_config = {
    'Shanghai-center-train': {
        'list_dir': 'dataset/CSV_list/Shanghai-center-train.csv',
        'image_dir': 'dataset/Shanghai-center-train/image_tiles',
        'num_classes': 2
    },
    'Shanghai-center-test': {
        'list_dir': 'dataset/CSV_list/Shanghai-center-test.csv',
        'image_dir': 'dataset/Shanghai-center-train/image_tiles',
        'num_classes': 2
    },
}


# =========================
# Loss tracker
# =========================
class LossTracker:
    """Track, print, and save averaged losses per epoch."""

    def __init__(self, snapshot_path):
        self.loss_sums = {}
        self.counts = {}
        self.file_path = os.path.join(snapshot_path, 'losses.csv')
        self.epoch_losses = []

        # Initialize CSV file with header
        if not os.path.exists(self.file_path):
            with open(self.file_path, 'w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(['epoch', 'consistent_loss', 'inconsistent_loss'])

    def update(self, *loss_dicts):
        """Accumulate losses from multiple dicts."""
        for loss_dict in loss_dicts:
            if loss_dict is None:
                continue
            for key, value in loss_dict.items():
                if isinstance(value, torch.Tensor):
                    if torch.isnan(value):
                        continue
                    value = value.item()
                if np.isnan(value):
                    continue

                self.loss_sums[key] = self.loss_sums.get(key, 0) + value
                self.counts[key] = self.counts.get(key, 0) + 1

    def print_and_save_losses(self, epoch):
        """Compute average losses, print, and append to CSV."""
        avg_losses = {
            k: self.loss_sums[k] / self.counts[k]
            for k in self.loss_sums
        }

        loss_str = ', '.join(f"{k}: {v:.4f}" for k, v in avg_losses.items())
        print(f"Epoch {epoch}: {loss_str}")

        row = [epoch] + [avg_losses.get(k, 0) for k in self.loss_sums]
        self.epoch_losses.append(row)

        with open(self.file_path, 'a', newline='') as file:
            csv.writer(file).writerow(row)

        # Reset after each epoch
        self.loss_sums.clear()
        self.counts.clear()


# =========================
# Loss utils
# =========================
def sum_losses(loss_dict):
    """Sum all valid tensor losses in a dict."""
    total_loss = 0.0
    if loss_dict is None:
        return total_loss

    for loss in loss_dict.values():
        if isinstance(loss, torch.Tensor) and not torch.isnan(loss):
            total_loss += loss
        else:
            print('Invalid loss value detected.')

    return total_loss


# =========================
# Checkpoint loading
# =========================
def load_pretrained_exclude_head(model, path):
    """Load checkpoint while excluding layers containing 'head'."""
    ckpt = torch.load(path, map_location='cpu')

    if isinstance(ckpt, dict) and 'state_dict' in ckpt:
        ckpt = ckpt['state_dict']

    # Remove DataParallel prefix
    ckpt = {k.replace('module.', ''): v for k, v in ckpt.items()}

    # Exclude head layers
    ckpt = {k: v for k, v in ckpt.items() if 'head' not in k}

    model.load_state_dict(ckpt, strict=False)
    return model