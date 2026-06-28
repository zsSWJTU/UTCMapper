# -*- coding: utf-8 -*-
"""
Shared utilities for UTCMapper training and inference.

This module intentionally avoids project-specific filesystem paths. Dataset
entries only declare public dataset names and class counts; CSV paths should be
provided through the YAML config or command-line arguments.
"""

import csv
import os

import numpy as np
import torch


# =========================================================
# Image normalization
# =========================================================

# Default RGB statistics used by the release configuration. Override them in
# configs/default.yaml when training on a different image distribution.
IMAGE_MEANS = np.array([101.16, 103.08, 87.93])
IMAGE_STDS = np.array([63.32, 56.17, 53.20])

DATASET_IMAGE_STATS = {
    "Shanghai-center-train": {
        "IMAGE_MEANS": IMAGE_MEANS,
        "IMAGE_STDS": IMAGE_STDS,
    },
    "Shanghai-center-test": {
        "IMAGE_MEANS": IMAGE_MEANS,
        "IMAGE_STDS": IMAGE_STDS,
    },
}


# =========================================================
# Dataset registry
# =========================================================

# Keep this registry path-free for open-source release. The training and
# inference scripts can still resolve num_classes from the dataset name, while
# users pass list_dir explicitly in the config or CLI.
dataset_config = {
    "Shanghai-center-train": {
        "num_classes": 2,
    },
    "Shanghai-center-test": {
        "num_classes": 2,
    },
}


# =========================================================
# Loss tracking
# =========================================================

class LossTrackerV1:
    """Accumulate per-epoch loss values and append their averages to CSV."""

    def __init__(self, snapshot_path, suffix=""):
        self.loss_sums = {}
        self.counts = {}
        self.file_path = os.path.join(snapshot_path, f"losses_{suffix}.csv")

        if not os.path.exists(self.file_path):
            with open(self.file_path, "w", newline="") as f:
                csv.writer(f).writerow(
                    ["epoch", "consistent_loss", "inconsistent_loss"]
                )

    def update(self, *loss_dicts):
        """Add one or more loss dictionaries from a training step."""
        for loss_dict in loss_dicts:
            if loss_dict is None:
                continue

            for key, value in loss_dict.items():
                if isinstance(value, torch.Tensor):
                    value = value.item()

                if np.isnan(value):
                    continue

                self.loss_sums[key] = self.loss_sums.get(key, 0.0) + value
                self.counts[key] = self.counts.get(key, 0) + 1

    def print_and_save_losses(self, epoch):
        """Print averaged losses and write one CSV row for the epoch."""
        avg = {
            key: self.loss_sums[key] / self.counts[key]
            for key in self.loss_sums
            if self.counts[key] > 0
        }

        print("Loss:", avg)

        with open(self.file_path, "a", newline="") as f:
            csv.writer(f).writerow([epoch] + list(avg.values()))

        self.loss_sums.clear()
        self.counts.clear()


# =========================================================
# Loss helpers
# =========================================================

def sum_losses(loss_dict):
    """Sum tensor losses while ignoring missing dictionaries and NaNs."""
    if loss_dict is None:
        return 0.0

    total = 0.0

    for value in loss_dict.values():
        if isinstance(value, torch.Tensor) and not torch.isnan(value):
            total += value

    return total


def weight_loss_dict(loss_dict, weights):
    """Apply scalar weights to a loss dictionary in insertion order."""
    if len(loss_dict) != len(weights):
        raise ValueError(
            f"loss/weight length mismatch: {len(loss_dict)} losses, "
            f"{len(weights)} weights"
        )

    return {
        key: value * weight
        for (key, value), weight in zip(loss_dict.items(), weights)
    }


# =========================================================
# Prediction colormap
# =========================================================

# Binary output classes:
#   0 = background
#   1 = foreground urban tree canopy
LABEL_CLASSES = [0, 1]
LABEL_CLASS_COLORMAP = {
    0: (231, 230, 230),
    1: (51, 160, 44),
}

LABEL_IDX_COLORMAP = {
    idx: LABEL_CLASS_COLORMAP[c]
    for idx, c in enumerate(LABEL_CLASSES)
}
