# -*- coding: utf-8 -*-
"""
Trainer for MESNet - Config Version

Features:
1. MESNet only.
2. Single-band label training.
3. Normalization from config:
       normalization.image_means
       normalization.image_stds

4. Fast training:
       - multi-worker DataLoader
       - persistent_workers
       - prefetch_factor
       - pin_memory
       - AMP mixed precision
       - non_blocking CUDA transfer
       - optimizer.zero_grad(set_to_none=True)

5. Dataset sampling:
       - random chips per tile
"""

import argparse
import logging
import os
import platform
import random
import sys
from functools import partial
from types import SimpleNamespace

os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")

import albumentations as A
import numpy as np
import pandas as pd
import rasterio
import torch
import torch.optim as optim
import yaml

from rasterio.errors import RasterioError
from rasterio.windows import Window
from tensorboardX import SummaryWriter
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from torch.utils.data import get_worker_info
from torch.utils.data.dataset import IterableDataset
from tqdm import tqdm

import utils
from networks.loss import CCGLoss
from networks.MESNet_UltraFast import MESNet
from utils import sum_losses, weight_loss_dict
from pathlib import Path
from cached_dataset import CachedGeospatialDataset
import torch.multiprocessing as mp
mp.set_sharing_strategy("file_system")

# =============================================================================
# 0. Helpers
# =============================================================================

MY_DEVICE_ID = "DESKTOP-RK6BJ9T"

try:
    device_id = platform.node()
except Exception:
    device_id = None


def dict_to_namespace(d):
    if isinstance(d, dict):
        return SimpleNamespace(
            **{
                k: dict_to_namespace(v)
                for k, v in d.items()
            }
        )

    if isinstance(d, list):
        return [
            dict_to_namespace(v)
            for v in d
        ]

    return d


def namespace_to_dict(ns):
    if isinstance(ns, SimpleNamespace):
        return {
            k: namespace_to_dict(v)
            for k, v in vars(ns).items()
        }

    if isinstance(ns, list):
        return [
            namespace_to_dict(v)
            for v in ns
        ]

    return ns


def load_config(config_path):
    if config_path is None or config_path == "":
        raise ValueError("config_path is empty.")

    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config not found: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    if cfg is None:
        cfg = {}

    return dict_to_namespace(cfg)


def get_arg(args, name, default=None):
    return getattr(args, name, default)


def get_nested_arg(args, group_name, name, default=None):
    group = getattr(args, group_name, None)

    if group is None:
        return default

    return getattr(group, name, default)


def setup_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def normalize_runtime_config(args):
    """
    Keep old configs with optim.compile/checkpoint/channels_last working.
    MESNet.from_config reads args.runtime for execution options.
    """
    if hasattr(args, "runtime") and getattr(args, "runtime") is not None:
        return

    optim_cfg = getattr(args, "optim", None)
    if optim_cfg is None:
        return

    runtime_keys = [
        "checkpoint",
        "channels_last",
        "compile",
        "compile_mode",
    ]

    runtime_values = {
        key: getattr(optim_cfg, key)
        for key in runtime_keys
        if hasattr(optim_cfg, key)
    }

    if runtime_values:
        args.runtime = SimpleNamespace(**runtime_values)




# =============================================================================
# 2. Augmentation and transforms
# =============================================================================

aug_transforms = A.Compose(
    [
        A.RandomBrightnessContrast(
            brightness_limit=0.3,
            contrast_limit=0.3,
            p=0.5,
        ),
        A.RGBShift(
            r_shift_limit=30,
            g_shift_limit=30,
            b_shift_limit=30,
            p=0.5,
        ),
    ]
)


class ImageTransform:
    def __init__(
        self,
        image_means,
        image_stds,
        image_band=3,
    ):
        self.image_band = image_band
        self.image_means = np.asarray(
            image_means,
            dtype=np.float32,
        ).reshape(1, 1, -1)
        self.image_stds = np.asarray(
            image_stds,
            dtype=np.float32,
        ).reshape(1, 1, -1)

        if np.any(self.image_stds == 0):
            raise ValueError(
                f"image_stds contains zero values: {self.image_stds}"
            )

        if self.image_means.shape[-1] != self.image_band:
            raise ValueError(
                f"image_means channel mismatch: "
                f"expected {self.image_band}, "
                f"got {self.image_means.shape[-1]}"
            )

        if self.image_stds.shape[-1] != self.image_band:
            raise ValueError(
                f"image_stds channel mismatch: "
                f"expected {self.image_band}, "
                f"got {self.image_stds.shape[-1]}"
            )

    def __call__(self, img, aug=False):
        if img.ndim != 3:
            raise ValueError(
                f"Expected image shape [H, W, C], got {img.shape}"
            )

        if img.shape[-1] < self.image_band:
            raise ValueError(
                f"Image has {img.shape[-1]} bands, "
                f"but image_band={self.image_band}"
            )

        img = img[:, :, :self.image_band]

        if aug:
            if img.dtype != np.uint8:
                img_aug = np.clip(img, 0, 255).astype(np.uint8)
            else:
                img_aug = img

            img = aug_transforms(image=img_aug)["image"]

        img = img.astype(np.float32)
        img = (img - self.image_means) / self.image_stds

        img = np.moveaxis(img, 2, 0)
        img = np.ascontiguousarray(
            img,
            dtype=np.float32,
        )

        return torch.from_numpy(img)


def build_image_transform(
    image_means,
    image_stds,
    image_band=3,
):
    """
    Build a pickle-safe image transform from config.
    """
    return ImageTransform(
        image_means=image_means,
        image_stds=image_stds,
        image_band=image_band,
    )


def label_transforms(labels: np.ndarray) -> torch.Tensor:
    """
    Preserve raw label channels.

    Input:
        labels:
            [H, W] or [C, H, W] or [H, W, C]

    Output:
        Tensor [C, H, W]

    """
    if labels.ndim == 2:
        labels = np.expand_dims(labels, axis=0)

    elif labels.ndim == 3:
        # rasterio read is usually [C, H, W].
        # If accidentally [H, W, C], convert to [C, H, W].
        if labels.shape[0] > 16 and labels.shape[-1] <= 16:
            labels = np.moveaxis(labels, -1, 0)

    else:
        raise RuntimeError(f"Unsupported label shape: {labels.shape}")

    labels = np.ascontiguousarray(labels)

    return torch.from_numpy(labels).long()


def nodata_check(
    img,
    labels,
    skip_nt=False,
    ignore_ratio_threshold=1.0,
    positive_values=(2,),
    ignore_value=255,
):
    """
    No-data check.

    labels:
        [H, W] or [C, H, W]

    Return True means skip this chip.
    """
    if labels is None:
        return True

    if labels.ndim == 2:
        labels_for_check = labels[np.newaxis, :, :]
    elif labels.ndim == 3:
        labels_for_check = labels
    else:
        return True

    ignore_ratio = np.mean(labels_for_check == ignore_value)

    if ignore_ratio > ignore_ratio_threshold:
        return True

    if np.all(img == 255):
        return True

    if np.all(img == 0):
        return True

    if not np.isfinite(img).all():
        return True

    if skip_nt:
        has_tree = np.zeros(
            labels_for_check.shape[1:],
            dtype=bool,
        )

        for v in positive_values:
            has_tree |= np.any(labels_for_check == v, axis=0)

        if not np.any(has_tree):
            return True

    return False


# =============================================================================
# 3. Label target builder
# =============================================================================

def torch_isin(x, values):
    """
    Compatible torch.isin for older PyTorch versions.
    """
    mask = torch.zeros_like(
        x,
        dtype=torch.bool,
    )

    for v in values:
        mask |= (x == int(v))

    return mask


def build_target_from_label_batch(args, label_batch):
    """
    Build a binary target from a single-band label batch.

    Input:
        label_batch: [B, C, H, W] or [B, H, W]

    Output:
        target: [B, H, W]

    Final target:
        0   = background
        1   = foreground
        255 = ignore
    """
    if label_batch.ndim == 3:
        label_raw = label_batch
    elif label_batch.ndim == 4:
        label_raw = label_batch[:, 0, :, :]
    else:
        raise RuntimeError(
            f"label_batch should be [B, C, H, W] or [B, H, W], got {label_batch.shape}"
        )

    positive_values = get_nested_arg(
        args,
        "label",
        "positive_values",
        [2],
    )

    if not isinstance(positive_values, (list, tuple, set)):
        positive_values = [positive_values]

    ignore_value = get_nested_arg(
        args,
        "label",
        "ignore_value",
        255,
    )

    foreground = torch_isin(
        label_raw,
        positive_values,
    )

    ignore = label_raw == ignore_value

    target = torch.zeros_like(
        label_raw,
        dtype=torch.long,
    )

    target[foreground] = 1
    target[ignore] = ignore_value

    return target


# =============================================================================
# 4. Worker seed
# =============================================================================

def worker_init_fn(worker_id, seed=1234):
    worker_seed = seed + worker_id

    random.seed(worker_seed)
    np.random.seed(worker_seed)
    torch.manual_seed(worker_seed)


# =============================================================================
# 5. Scheduler
# =============================================================================

def get_basic_scheduler(
    optimizer,
    total_epochs,
    base_lr,
):
    """Use a simple cosine decay schedule based on optimizer learning rates."""
    if total_epochs <= 1:
        return None

    return CosineAnnealingLR(
        optimizer,
        T_max=total_epochs,
        eta_min=base_lr * 0.001,
    )


# =============================================================================
# 6. Optimizer
# =============================================================================
# 6. Optimizer
# =============================================================================

def build_optimizer(args, model):
    optimizer_type = get_nested_arg(
        args,
        "optimizer",
        "type",
        "SGD",
    )

    weight_decay = get_nested_arg(
        args,
        "optimizer",
        "weight_decay",
        1e-4,
    )

    params = filter(
        lambda p: p.requires_grad,
        model.parameters(),
    )

    if optimizer_type.lower() == "sgd":
        momentum = get_nested_arg(
            args,
            "optimizer",
            "momentum",
            0.9,
        )

        nesterov = get_nested_arg(
            args,
            "optimizer",
            "nesterov",
            False,
        )

        optimizer = optim.SGD(
            params,
            lr=args.base_lr,
            momentum=momentum,
            weight_decay=weight_decay,
            nesterov=nesterov,
        )

    elif optimizer_type.lower() == "adamw":
        betas = get_nested_arg(
            args,
            "optimizer",
            "betas",
            [0.9, 0.999],
        )

        optimizer = optim.AdamW(
            params,
            lr=args.base_lr,
            betas=tuple(betas),
            weight_decay=weight_decay,
        )

    else:
        raise ValueError(
            f"Unsupported optimizer type: {optimizer_type}"
        )

    return optimizer


# =============================================================================
# 7. Pretrained
# =============================================================================

def load_pretrained_if_needed(model, pretrained_path):
    if pretrained_path is None or pretrained_path == "":
        return model

    if not os.path.exists(pretrained_path):
        print(f"Pretrained weights not found, skipping load: {pretrained_path}")
        return model

    print(f"Loading pretrained weights: {pretrained_path}")

    state = torch.load(
        pretrained_path,
        map_location="cpu",
    )

    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]

    new_state = {}

    for k, v in state.items():
        if k.startswith("module."):
            k = k[len("module."):]
        new_state[k] = v

    missing, unexpected = model.load_state_dict(
        new_state,
        strict=False,
    )

    print(f"Missing keys: {len(missing)}")
    print(f"Unexpected keys: {len(unexpected)}")

    return model


# =============================================================================
# 8. Trainer
# =============================================================================

def trainer_dataset(args, model, snapshot_path):
    os.makedirs(
        snapshot_path,
        exist_ok=True,
    )

    logging.basicConfig(
        filename=os.path.join(snapshot_path, "log.txt"),
        level=logging.INFO,
        format="[%(asctime)s.%(msecs)03d] %(message)s",
        datefmt="%H:%M:%S",
        force=True,
    )

    logging.info("Training arguments: %s", str(args))
    logging.info("Model: %s", type(model).__name__)

    # -------------------------------------------------------------------------
    # Basic config
    # -------------------------------------------------------------------------
    seed = get_arg(args, "seed", 1234)
    setup_seed(seed)

    dataset_name = get_arg(args, "dataset", "unknown")
    list_dir = get_arg(args, "list_dir", None)

    if list_dir is None:
        raise ValueError("args.list_dir is required.")

    base_lr = get_arg(args, "base_lr", 0.01)
    batch_size = get_arg(args, "batch_size", 2)
    max_epochs = get_arg(args, "max_epochs", 10)

    chip_size = get_arg(args, "chip_size", 224)

    num_chips_per_tile = get_arg(
        args,
        "num_chips_per_tile",
        get_arg(args, "n_chips", 50),
    )

    ratio = get_arg(args, "ratio", 1.0)
    aug = get_arg(args, "aug", False)
    skip_nt = get_arg(args, "skip_nt", True)

    image_band = get_arg(args, "image_band", 3)

    model_cfg = getattr(args, "model", None)

    if model_cfg is not None:
        image_band = getattr(model_cfg, "image_band", image_band)

    pretrained_model = get_arg(
        args,
        "pretrained_model",
        get_arg(args, "last_model", ""),
    )

    normalize_runtime_config(args)

    # -------------------------------------------------------------------------
    # Label config
    # -------------------------------------------------------------------------
    ignore_ratio_threshold = get_arg(
        args,
        "ignore_ratio_threshold",
        get_nested_arg(
            args,
            "label",
            "ignore_ratio_threshold",
            1.0,
        ),
    )

    positive_values = get_nested_arg(
        args,
        "label",
        "positive_values",
        [2],
    )

    ignore_value = get_nested_arg(
        args,
        "label",
        "ignore_value",
        255,
    )

    # -------------------------------------------------------------------------
    # Dataloader config
    # -------------------------------------------------------------------------
    num_workers = get_nested_arg(
        args,
        "dataloader",
        "num_workers",
        get_arg(args, "num_workers", 6),
    )

    pin_memory = get_nested_arg(
        args,
        "dataloader",
        "pin_memory",
        get_arg(args, "pin_memory", True),
    )

    persistent_workers = get_nested_arg(
        args,
        "dataloader",
        "persistent_workers",
        get_arg(args, "persistent_workers", True),
    )

    prefetch_factor = get_nested_arg(
        args,
        "dataloader",
        "prefetch_factor",
        get_arg(args, "prefetch_factor", 2),
    )

    drop_last = get_nested_arg(
        args,
        "dataloader",
        "drop_last",
        get_arg(args, "drop_last", True),
    )

    windowed_sampling = get_nested_arg(
        args,
        "dataloader",
        "windowed_sampling",
        get_arg(args, "windowed_sampling", True),
    )

    # -------------------------------------------------------------------------
    # Device / AMP
    # -------------------------------------------------------------------------
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    use_amp = get_nested_arg(
        args,
        "amp",
        "enabled",
        get_arg(args, "amp", True),
    )

    if device.type != "cuda":
        use_amp = False
        pin_memory = False

    if num_workers <= 0:
        persistent_workers = False
        prefetch_factor = None

    # -------------------------------------------------------------------------
    # Normalization from config
    # -------------------------------------------------------------------------
    image_means = get_nested_arg(
        args,
        "normalization",
        "image_means",
        None,
    )

    image_stds = get_nested_arg(
        args,
        "normalization",
        "image_stds",
        None,
    )

    if image_means is None:
        if hasattr(utils, "DATASET_IMAGE_STATS"):
            if dataset_name in utils.DATASET_IMAGE_STATS:
                image_means = utils.DATASET_IMAGE_STATS[dataset_name]["IMAGE_MEANS"]
            else:
                raise ValueError(
                    "normalization.image_means is not provided "
                    f"and no image statistics are registered for {dataset_name}."
                )
        else:
            raise ValueError(
                "normalization.image_means is not provided."
            )

    if image_stds is None:
        if hasattr(utils, "DATASET_IMAGE_STATS"):
            if dataset_name in utils.DATASET_IMAGE_STATS:
                image_stds = utils.DATASET_IMAGE_STATS[dataset_name]["IMAGE_STDS"]
            else:
                raise ValueError(
                    "normalization.image_stds is not provided "
                    f"and no image statistics are registered for {dataset_name}."
                )
        else:
            raise ValueError(
                "normalization.image_stds is not provided."
            )

    image_transform = build_image_transform(
        image_means=image_means,
        image_stds=image_stds,
        image_band=image_band,
    )

    # -------------------------------------------------------------------------
    # Logging config
    # -------------------------------------------------------------------------
    logging.info("Device: %s", str(device))
    logging.info("Dataset: %s", dataset_name)
    logging.info("List dir: %s", list_dir)
    logging.info("Batch size: %d", batch_size)
    logging.info("Base LR: %.8f", base_lr)
    logging.info("Max epochs: %d", max_epochs)
    logging.info("Chip size: %d", chip_size)
    logging.info("Num chips per tile: %d", num_chips_per_tile)
    logging.info("Ratio: %.4f", ratio)
    logging.info("Aug: %s", str(aug))
    logging.info("Main loss type: dice")
    logging.info("Skip no-tree: %s", str(skip_nt))
    logging.info("Positive values: %s", str(positive_values))
    logging.info("Ignore value: %s", str(ignore_value))
    logging.info("Ignore ratio threshold: %.4f", ignore_ratio_threshold)
    logging.info("Ignore ratio threshold: %.4f", ignore_ratio_threshold)
    logging.info("Windowed sampling: %s", str(windowed_sampling))
    logging.info("Num workers: %d", num_workers)
    logging.info("Pin memory: %s", str(pin_memory))
    logging.info("Persistent workers: %s", str(persistent_workers))
    logging.info("Prefetch factor: %s", str(prefetch_factor))
    logging.info("AMP enabled: %s", str(use_amp))
    logging.info("Image means: %s", str(image_means))
    logging.info("Image stds : %s", str(image_stds))

    # -------------------------------------------------------------------------
    # Data
    # -------------------------------------------------------------------------
    input_dataframe = pd.read_csv(list_dir)

    if "image_fn" not in input_dataframe.columns:
        raise ValueError("CSV must contain column: image_fn")

    if "label_fn" not in input_dataframe.columns:
        raise ValueError("CSV must contain column: label_fn")

    image_fns = input_dataframe["image_fn"].values
    label_fns = input_dataframe["label_fn"].values

    nodata_checker = partial(
        nodata_check,
        skip_nt=skip_nt,
        ignore_ratio_threshold=ignore_ratio_threshold,
        positive_values=positive_values,
        ignore_value=ignore_value,
    )

    db_train = CachedGeospatialDataset(
        imagery_fns=image_fns,
        label_fns=label_fns,

        chip_size=chip_size,
        num_chips_per_tile=num_chips_per_tile,

        dataset_name=dataset_name,

        image_transform=image_transform,
        label_transform=label_transforms,
        nodata_check=nodata_checker,

        cache_dir=None,
        rebuild_cache=False,
        seed=seed,

        verbose=False,
    )


    sampled_tile_count = len(db_train)




    logging.info("Sampled tiles: %d", sampled_tile_count)
    print(f"Sampled tiles: {sampled_tile_count}")



    # -------------------------------------------------------------------------
    # DataLoader
    # -------------------------------------------------------------------------
    worker_init = partial(
        worker_init_fn,
        seed=seed,
    )

    print(
        "\nResolved runtime: "
        f"device={device}, "
        f"workers={num_workers}, "
        f"amp={use_amp}, "
        f"pin_memory={pin_memory}"
    )

    trainloader = DataLoader(
        db_train,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        prefetch_factor=prefetch_factor,
        drop_last=drop_last,
        worker_init_fn=worker_init,
    )

    steps_per_epoch = len(trainloader)
    # -------------------------------------------------------------------------
    # Model / pretrained / loss / optimizer
    # -------------------------------------------------------------------------
    model = model.to(device)

    if pretrained_model:
        model = load_pretrained_if_needed(
            model,
            pretrained_model,
        )

    criterion = CCGLoss(
        alpha=1.0,
        ignore_index=ignore_value,
    )


    loss_weights = get_nested_arg(
        args,
        "loss",
        "weights",
        [1, 1],
    )

    loss_tracker = utils.LossTrackerV1(
        snapshot_path=snapshot_path,
        suffix="train",
    )

    optimizer = build_optimizer(
        args,
        model,
    )

    initial_lrs = [
        group["lr"]
        for group in optimizer.param_groups
    ]

    print("\nInitial learning rates:")
    for i, lr in enumerate(initial_lrs):
        print(f"  Param group {i}: lr = {lr:.6f}")

    scheduler = get_basic_scheduler(
        optimizer,
        max_epochs,
        base_lr,
    )

    clip_grad = get_nested_arg(
        args,
        "optimizer",
        "clip_grad",
        get_arg(args, "clip_grad", 1.0),
    )

    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=(use_amp and device.type == "cuda"),
    )

    writer = SummaryWriter(
        os.path.join(snapshot_path, "log")
    )

    # -------------------------------------------------------------------------
    # Checkpoint config
    # -------------------------------------------------------------------------
    save_interval = get_nested_arg(
        args,
        "checkpoint",
        "save_interval",
        get_arg(args, "save_interval", 20),
    )

    save_last = get_nested_arg(
        args,
        "checkpoint",
        "save_last",
        True,
    )

    # -------------------------------------------------------------------------
    # Training loop
    # -------------------------------------------------------------------------
    iter_num = 0

    for epoch_num in range(max_epochs):
        model.train()

        epoch_loss = 0.0
        valid_batches = 0

        pbar = tqdm(
            trainloader,
            desc=f"Epoch {epoch_num + 1}/{max_epochs}",
        )

        for image_batch, label_batch in pbar:

            image_batch = image_batch.to(
                device,
                non_blocking=True,
            )

            label_batch = label_batch.to(
                device,
                non_blocking=True,
            )

            # -----------------------------------------------------------------
            # Build final target from raw label values
            # -----------------------------------------------------------------
            target = build_target_from_label_batch(
                args=args,
                label_batch=label_batch,
            )

            # =========================================================
            # Check target values before loss
            # Allowed values:
            #   0 ~ num_classes - 1
            #   ignore_value
            # =========================================================
            num_classes = get_arg(
                args,
                "num_classes",
                2,
            )

            invalid_mask = (
                    (target != ignore_value)
                    & (
                            (target < 0)
                            | (target >= num_classes)
                    )
            )

            if invalid_mask.any():
                bad_values = torch.unique(
                    target[invalid_mask].detach().cpu()
                )

                raise RuntimeError(
                    f"Invalid target values detected: {bad_values.tolist()} | "
                    f"Allowed values: 0~{num_classes - 1}, "
                    f"ignore_value={ignore_value}"
                )

            lr_ = optimizer.param_groups[0]["lr"]

            optimizer.zero_grad(set_to_none=True)

            # =========================================================
            # 1. Model forward with AMP
            # =========================================================
            with torch.amp.autocast(
                    "cuda",
                    enabled=(use_amp and device.type == "cuda"),
            ):
                outputs = model(image_batch)

            # =========================================================
            # 2. Loss computation in float32
            #    Important:
            #    CCGLoss includes softmax / log_softmax / topk / dice / masks.
            #    These are safer outside autocast.
            # =========================================================
            with torch.amp.autocast(
                    "cuda",
                    enabled=False,
            ):
                loss_dict = criterion(
                    outputs.float(),
                    target,
                )

                loss_dict = weight_loss_dict(
                    loss_dict,
                    loss_weights,
                )

                loss = sum_losses(loss_dict)

            scaler.scale(loss).backward()

            if clip_grad is not None and clip_grad > 0:
                scaler.unscale_(optimizer)

                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    max_norm=clip_grad,
                )

            scaler.step(optimizer)
            scaler.update()

            loss_tracker.update(loss_dict)

            iter_num += 1
            valid_batches += 1
            epoch_loss += loss.item()

            writer.add_scalar(
                "train/loss",
                loss.item(),
                iter_num,
            )

            writer.add_scalar(
                "train/lr",
                lr_,
                iter_num,
            )

            pbar.set_postfix(
                {
                    "loss": f"{loss.item():.4f}",
                    "lr": f"{lr_:.6f}",
                }
            )

        avg_epoch_loss = epoch_loss / max(valid_batches, 1)

        print(f"\nEpoch {epoch_num + 1}/{max_epochs}")
        print(f"Average loss: {avg_epoch_loss:.6f}")

        writer.add_scalar(
            "train/epoch_loss",
            avg_epoch_loss,
            epoch_num + 1,
        )

        print("loss_dict:")
        loss_tracker.print_and_save_losses(epoch_num + 1)

        if scheduler is not None:
            scheduler.step()

        current_lr = optimizer.param_groups[0]["lr"]
        print(f"Current LR: {current_lr:.6f}")

        # ---------------------------------------------------------------------
        # Save checkpoint
        # ---------------------------------------------------------------------
        should_save = False

        if save_interval > 0 and (epoch_num + 1) % save_interval == 0:
            should_save = True

        if save_last and epoch_num == max_epochs - 1:
            should_save = True

        if should_save:
            save_mode_path = os.path.join(
                snapshot_path,
                f"epoch_{epoch_num + 1}.pth",
            )

            torch.save(
                model.state_dict(),
                save_mode_path,
            )

            # ===== FINAL MODEL (for inference) =====
            if epoch_num == max_epochs - 1:
                final_path = os.path.join(snapshot_path, "final.pth")
                torch.save(model.state_dict(), final_path)
                logging.info(f"save final model to {final_path}")



            logging.info(f"save model to {save_mode_path}")

    writer.close()

    return "Training Finished!"


# =============================================================================
# 9. Main
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to YAML config file.",
    )

    parser.add_argument(
        "--gpu",
        type=str,
        default=None,
        help="Override GPU id. If None, use config.gpu.",
    )

    cli_args = parser.parse_args()

    args = load_config(cli_args.config)
    normalize_runtime_config(args)

    if cli_args.gpu is not None:
        args.gpu = cli_args.gpu

    gpu = get_arg(args, "gpu", "0")
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)

    snapshot_path = get_arg(
        args,
        "work_dir",
        "work_dirs/mesnet_config",
    )

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    net = MESNet.from_config(args)
    net = net.to(device)
    net = net.build_runtime()

    trainer_dataset(
        args=args,
        model=net,
        snapshot_path=snapshot_path,
    )
