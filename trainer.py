"""
Trainer for UTCMapper

This script sets up training routines

Referenced from:
L2HNet and Paraformer: https://github.com/LiZhuoHong/Paraformer
"""

import logging
import os
import sys
from functools import partial
import pandas as pd
import torch.optim as optim
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from tqdm import tqdm
import random
import numpy as np
import torch
from torch.utils.data import IterableDataset
import rasterio
from rasterio.windows import Window
from rasterio.errors import RasterioError
import utils as utils
from networks.loss import CCGLoss
from utils import sum_losses




class StreamingGeospatialDataset(IterableDataset):
    """
    StreamingGeospatialDataset for on-the-fly chip extraction from geospatial imagery.

    Features:
        - Iterable dataset for multi-worker loading
        - Optional label handling
        - Windowed or full-tile sampling
        - Transformations for imagery and labels
        - No-data checking per chip
    """

    def __init__(self, imagery_fns, label_fns=None,
                 chip_size=256, num_chips_per_tile=200, windowed_sampling=False,
                 image_transform=None, label_transform=None, nodata_check=None,
                 verbose=False):
        self.use_labels = label_fns is not None
        self.fns = list(zip(imagery_fns, label_fns)) if self.use_labels else imagery_fns

        self.chip_size = chip_size
        self.num_chips_per_tile = num_chips_per_tile
        self.windowed_sampling = windowed_sampling
        self.image_transform = image_transform
        self.label_transform = label_transform
        self.nodata_check = nodata_check
        self.verbose = verbose

        if self.verbose:
            print("Constructed StreamingGeospatialDataset")

    def stream_tile_fns(self):
        worker_info = torch.utils.data.get_worker_info()
        worker_id = 0 if worker_info is None else worker_info.id
        num_workers = 1 if worker_info is None else worker_info.num_workers

        if worker_id == 0:
            np.random.shuffle(self.fns)

        N = len(self.fns)
        num_files_per_worker = int(np.ceil(N / num_workers))
        lower_idx = worker_id * num_files_per_worker
        upper_idx = min(N, (worker_id + 1) * num_files_per_worker)

        for idx in range(lower_idx, upper_idx):
            if self.use_labels:
                img_fn, label_fn = self.fns[idx]
            else:
                img_fn, label_fn = self.fns[idx], None

            if self.verbose:
                print(f"Worker {worker_id}, yielding file {idx}")

            yield img_fn, label_fn

    def stream_chips(self):
        for img_fn, label_fn in self.stream_tile_fns():
            num_skipped_chips = 0

            # Open file pointers
            img_fp = rasterio.open(img_fn, "r")
            label_fp = rasterio.open(label_fn, "r") if self.use_labels else None

            height, width = img_fp.shape
            if self.use_labels:
                t_height, t_width = label_fp.shape
                assert height == t_height and width == t_width

            img_data, label_data = None, None
            try:
                if not self.windowed_sampling:
                    img_data = np.rollaxis(img_fp.read(3), 0, 3)
                    if self.use_labels:
                        label_data = label_fp.read().squeeze()
            except RasterioError:
                print(f"WARNING: Error reading {img_fn}, skipping file")
                continue

            for _ in range(self.num_chips_per_tile):
                x = np.random.randint(0, width - self.chip_size)
                y = np.random.randint(0, height - self.chip_size)

                # Extract chip
                if self.windowed_sampling:
                    try:
                        img = np.rollaxis(img_fp.read(window=Window(x, y, self.chip_size, self.chip_size)), 0, 3)
                        labels = label_fp.read(
                            window=Window(x, y, self.chip_size, self.chip_size)).squeeze() if self.use_labels else None
                    except RasterioError:
                        continue
                else:
                    img = img_data[y:y + self.chip_size, x:x + self.chip_size, :]
                    labels = label_data[y:y + self.chip_size, x:x + self.chip_size] if self.use_labels else None

                # No-data check
                if self.nodata_check is not None:
                    skip_chip = self.nodata_check(img, labels) if self.use_labels else self.nodata_check(img)
                    if skip_chip:
                        num_skipped_chips += 1
                        continue

                # Apply transformations
                img = self.image_transform(img) if self.image_transform else torch.from_numpy(img).squeeze()
                if self.use_labels:
                    labels = self.label_transform(labels) if self.label_transform else torch.from_numpy(
                        labels).squeeze()
                    yield img, labels
                else:
                    yield img

            img_fp.close()
            if self.use_labels:
                label_fp.close()

            if num_skipped_chips > 0 and self.verbose:
                print(f"Skipped {num_skipped_chips} chips for {img_fn}")

    def __iter__(self):
        if self.verbose:
            print("Creating a new StreamingGeospatialDataset iterator")
        return iter(self.stream_chips())


def image_transforms(img):
    img = (img - utils.IMAGE_MEANS) / utils.IMAGE_STDS
    img = np.rollaxis(img, 2, 0).astype(np.float32)
    img = torch.from_numpy(img)
    return img

def label_transforms(labels,positive_value=2,target_value=1):
    labels=np.where(labels==target_value,positive_value,0)
    labels = utils.LABEL_CLASS_TO_IDX_MAP[labels]
    labels = torch.from_numpy(labels)
    return labels

def nodata_check(img, labels):
    return np.all(labels == 0) or np.all(np.sum(img == 0, axis=2) == 4) or np.any(labels == 255)

def worker_init_fn(worker_id,seed=None):
    random.seed(seed + worker_id)


def trainer_dataset(args, model, snapshot_path):
    """
    Training function

    Args:
        args: argparse.Namespace, contains training parameters like batch_size, max_epochs, base_lr, etc.
        model: torch.nn.Module, the model to train
        snapshot_path: str, directory to save logs and checkpoints

    Returns:
        str: completion message
    """

    # ---------------------------
    # Logging setup
    # ---------------------------
    logging.basicConfig(
        filename=os.path.join(snapshot_path, "log.txt"),
        level=logging.INFO,
        format='[%(asctime)s.%(msecs)03d] %(message)s',
        datefmt='%H:%M:%S'
    )
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info("Training arguments: %s", str(args))
    logging.info("Model: %s", type(model).__name__)

    # ---------------------------
    # Load input data
    # ---------------------------
    # Number of chips sampled per tile
    # To maximize sample usage, set as 6000*6000/224*224 ≈ 716 to approximately cover the full tile
    # For lightweight training, set to 200
    NUM_CHIPS_PER_TILE = 200
    input_df = pd.read_csv(args.list_dir)
    image_fns = input_df["image_fn"].values
    label_fns = input_df["label_fn"].values

    db_train = StreamingGeospatialDataset(
        imagery_fns=image_fns,
        label_fns=label_fns,
        chip_size=224,
        num_chips_per_tile=NUM_CHIPS_PER_TILE,
        windowed_sampling=True,
        verbose=False,
        image_transform=image_transforms,
        label_transform=label_transforms,
        nodata_check=nodata_check
    )

    print(f"The length of train set is: {len(image_fns) * NUM_CHIPS_PER_TILE}")

    # ---------------------------
    # DataLoader
    # ---------------------------
    worker_init = partial(worker_init_fn, seed=args.seed)
    trainloader = DataLoader(
        db_train,
        batch_size=args.batch_size,
        num_workers=0,
        pin_memory=True,
        worker_init_fn=worker_init,
        drop_last=True
    )

    # ---------------------------
    # Model, loss, optimizer
    # ---------------------------
    model.train()
    loss_function = CCGLoss()
    optimizer = optim.SGD(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.base_lr,
        momentum=0.9,
        weight_decay=1e-4
    )
    loss_tracker = utils.LossTracker(snapshot_path=snapshot_path)
    writer = SummaryWriter(os.path.join(snapshot_path, 'log'))

    # ---------------------------
    # Training loop
    # ---------------------------
    iter_num = 0
    max_epoch = args.max_epochs
    max_iterations = max_epoch * len(image_fns) * NUM_CHIPS_PER_TILE
    logging.info("%d iterations per epoch, %d max iterations", len(image_fns) * NUM_CHIPS_PER_TILE, max_iterations)

    for epoch_num in range(max_epoch):
        for i_batch, (image_batch, label_batch) in tqdm(
                enumerate(trainloader),
                total=int(len(image_fns) * NUM_CHIPS_PER_TILE / args.batch_size)
        ):
            image_batch, label_batch = image_batch.cuda(), label_batch.cuda()
            outputs = model(image_batch)

            # Loss computation
            loss_total = loss_function(outputs, label_batch)
            loss_tracker.update(loss_total)
            loss = sum_losses(loss_total)

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            # Learning rate schedule
            lr_ = args.base_lr * (1.0 - iter_num / max_iterations) ** 0.9
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_
            iter_num += 1

        # ---------------------------
        # Checkpointing
        # ---------------------------
        if (epoch_num + 1) % 20 == 0 or epoch_num == max_epoch - 1:
            save_path = os.path.join(snapshot_path, f'epoch_{epoch_num}.pth')
            torch.save(model.state_dict(), save_path)
            logging.info("Saved model checkpoint: %s", save_path)

        # Save average losses
        loss_tracker.print_and_save_losses(epoch_num + 1)

    writer.close()
    return "Training Finished!"