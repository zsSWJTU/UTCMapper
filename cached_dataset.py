# ==========================================================
# Standard library
# ==========================================================
import logging
import math
import os
import random
import sys

from functools import partial

# ==========================================================
# Third-party libraries
# ==========================================================
import numpy as np
import pandas as pd
import rasterio
import torch
import torch.optim as optim
import yaml
import json
from pathlib import Path

from rasterio.errors import RasterioError
from rasterio.windows import Window
from torch.utils.data import Dataset
from tqdm import tqdm

class CachedGeospatialDataset(Dataset):
    """
    CachedGeospatialDataset

    First run:
        - Randomly sample chips from GeoTIFFs
        - Apply nodata_check
        - Save valid image chips and label chips into np.memmap files

    Later runs:
        - Directly read chips from memmap cache
        - No rasterio random window reading during training

    Default cache format:
        dataset/cache/{dataset_name}/
            meta.json
            images_uint8.dat
            labels_uint8.dat
    """

    def __init__(
        self,
        imagery_fns,
        label_fns=None,
        chip_size=224,
        num_chips_per_tile=200,

        image_transform=None,
        label_transform=None,
        nodata_check=None,
        dataset_name="default_dataset",
        cache_dir=None,
        rebuild_cache=False,
        seed=42,
        max_attempts_multiplier=20,
        image_cache_dtype="uint8",
        label_cache_dtype="uint8",
        verbose=True,
    ):
        self.use_labels = label_fns is not None

        if self.use_labels:
            self.fns = list(
                zip(
                    list(imagery_fns),
                    list(label_fns)
                )
            )
        else:
            self.fns = [
                (img_fn, None)
                for img_fn in list(imagery_fns)
            ]

        self.dataset_name = str(dataset_name)

        self.chip_size = int(chip_size)
        self.num_chips_per_tile = int(num_chips_per_tile)

        self.image_transform = image_transform
        self.label_transform = label_transform
        self.nodata_check = nodata_check

        # ======================================================
        # Default cache location:
        # dataset/cache/{dataset_name}
        # ======================================================
        if cache_dir is None:
            cache_dir = Path("dataset") / "cache" / self.dataset_name
        else:
            cache_dir = Path(cache_dir)

        self.cache_dir = cache_dir
        self.cache_dir.mkdir(
            parents=True,
            exist_ok=True
        )

        self.rebuild_cache = bool(rebuild_cache)
        self.seed = int(seed)
        self.max_attempts_multiplier = int(max_attempts_multiplier)

        self.image_cache_dtype = str(image_cache_dtype)
        self.label_cache_dtype = str(label_cache_dtype)

        self.verbose = verbose

        self.meta_path = self.cache_dir / "meta.json"
        self.image_cache_path = self.cache_dir / f"images_{self.image_cache_dtype}.dat"
        self.label_cache_path = self.cache_dir / f"labels_{self.label_cache_dtype}.dat"

        # memmap handles are opened lazily in each worker
        self._images = None
        self._labels = None

        # ======================================================
        # Check cache before building
        # ======================================================
        cache_ready = self._cache_exists_and_compatible()

        if self.rebuild_cache:

            if self.verbose:
                print(
                    f"[CachedGeospatialDataset] rebuild_cache=True, "
                    f"rebuilding cache at: {self.cache_dir}"
                )

            self._build_cache()

        elif cache_ready:

            if self.verbose:
                print(
                    f"[CachedGeospatialDataset] Found compatible cache: "
                    f"{self.cache_dir}"
                )

        else:

            if self.verbose:
                print(
                    f"[CachedGeospatialDataset] No compatible cache found. "
                    f"Building cache at: {self.cache_dir}"
                )

            self._build_cache()

        self._load_meta()

        if self.verbose:
            print(
                f"[CachedGeospatialDataset] "
                f"dataset_name={self.dataset_name} | "
                f"cache_dir={self.cache_dir} | "
                f"num_samples={self.num_samples} | "
                f"chip_size={self.chip_size}"
            )

    # ==========================================================
    # Pickle safety for Windows DataLoader workers
    # ==========================================================
    def __getstate__(self):

        state = self.__dict__.copy()

        state["_images"] = None
        state["_labels"] = None

        return state

    # ==========================================================
    # Check whether cache exists and matches current settings
    # ==========================================================
    def _cache_exists_and_compatible(self):

        if not self.meta_path.exists():
            return False

        try:
            with open(
                self.meta_path,
                "r",
                encoding="utf-8"
            ) as f:
                meta = json.load(f)

        except Exception:
            return False

        required_keys = [
            "num_samples",
            "max_samples",
            "chip_size",
            "num_chips_per_tile",
            "use_labels",
            "image_cache_dtype",
            "label_cache_dtype",
            "image_cache_path",
            "label_cache_path",
            "image_shape",
            "label_shape",
            "seed",
            "dataset_name"
        ]

        for key in required_keys:
            if key not in meta:
                return False

        if int(meta["num_samples"]) <= 0:
            return False

        if int(meta["chip_size"]) != self.chip_size:
            return False

        if int(meta["num_chips_per_tile"]) != self.num_chips_per_tile:
            return False

        if bool(meta["use_labels"]) != self.use_labels:
            return False

        if str(meta["image_cache_dtype"]) != self.image_cache_dtype:
            return False

        if str(meta["label_cache_dtype"]) != self.label_cache_dtype:
            return False

        if int(meta["seed"]) != self.seed:
            return False

        if str(meta["dataset_name"]) != self.dataset_name:
            return False

        image_cache_path = Path(
            meta["image_cache_path"]
        )

        label_cache_path = Path(
            meta["label_cache_path"]
        )

        # If meta paths are stale, fall back to the current cache directory.
        if not image_cache_path.exists():
            image_cache_path = self.image_cache_path

        if not image_cache_path.exists():
            return False

        if self.use_labels:

            if not label_cache_path.exists():
                label_cache_path = self.label_cache_path

            if not label_cache_path.exists():
                return False

        return True

    # ==========================================================
    # Basic RGB reader
    # ==========================================================
    @staticmethod
    def read_rgb(
        src,
        window=None
    ):

        if src.count >= 3:

            img = src.read(
                [1, 2, 3],
                window=window
            )

            img = np.moveaxis(
                img,
                0,
                -1
            )

        elif src.count == 1:

            img = src.read(
                1,
                window=window
            )

            img = np.repeat(
                img[:, :, None],
                3,
                axis=2
            )

        else:

            raise ValueError(
                f"Unsupported band count: {src.count}"
            )

        return img

    # ==========================================================
    # Convert image to cache dtype
    # ==========================================================
    def _convert_image_for_cache(
        self,
        img
    ):

        if self.image_cache_dtype == "uint8":

            if img.dtype != np.uint8:
                img = np.clip(
                    img,
                    0,
                    255
                ).astype(
                    np.uint8
                )

            return img

        elif self.image_cache_dtype == "float32":

            return img.astype(
                np.float32
            )

        elif self.image_cache_dtype == "float16":

            return img.astype(
                np.float16
            )

        else:

            raise ValueError(
                f"Unsupported image_cache_dtype: {self.image_cache_dtype}"
            )

    # ==========================================================
    # Convert label to cache dtype
    # ==========================================================
    def _convert_label_for_cache(
        self,
        labels
    ):

        if self.label_cache_dtype == "uint8":

            if labels.dtype != np.uint8:
                labels = np.clip(
                    labels,
                    0,
                    255
                ).astype(
                    np.uint8
                )

            return labels

        elif self.label_cache_dtype == "int16":

            return labels.astype(
                np.int16
            )

        elif self.label_cache_dtype == "int32":

            return labels.astype(
                np.int32
            )

        else:

            raise ValueError(
                f"Unsupported label_cache_dtype: {self.label_cache_dtype}"
            )

    # ==========================================================
    # Pad label bands to cache band count
    # ==========================================================
    def _pad_labels_to_band_count(
            self,
            labels,
            target_bands
    ):
        """
        labels:
            C x H x W or H x W

        target_bands:
            fixed cache label band count

        If labels has fewer bands than target_bands, pad extra
        bands with 0. If labels has more bands than target_bands,
        keep the first target_bands bands.
        """

        if labels is None:
            return labels

        labels = np.asarray(
            labels
        )

        if labels.ndim == 2:
            labels = labels[None, :, :]

        if labels.ndim != 3:
            raise ValueError(
                f"Unsupported label ndim for padding: {labels.ndim}"
            )

        current_bands = int(
            labels.shape[0]
        )

        target_bands = int(
            target_bands
        )

        if current_bands == target_bands:
            return labels

        if current_bands > target_bands:
            return labels[:target_bands]

        pad_shape = (
            target_bands - current_bands,
            labels.shape[1],
            labels.shape[2]
        )

        pad = np.zeros(
            pad_shape,
            dtype=labels.dtype
        )

        labels = np.concatenate(
            [
                labels,
                pad
            ],
            axis=0
        )

        return labels

    # ==========================================================
    # Randomly sample candidate windows and keep the best one
    # ==========================================================
    def _sample_window(
            self,
            img_fp,
            label_fp_ctx,
            width,
            height,
            rng
    ):
        """Randomly sample one valid image/label window."""

        x = int(
            rng.integers(
                0,
                width - self.chip_size + 1
            )
        )

        y = int(
            rng.integers(
                0,
                height - self.chip_size + 1
            )
        )

        window = Window(
            x,
            y,
            self.chip_size,
            self.chip_size
        )

        try:
            img = self.read_rgb(
                img_fp,
                window=window
            )

            if self.use_labels:
                labels = label_fp_ctx.read(
                    window=window
                )
            else:
                labels = None

        except Exception:
            return None, None, 1, 0

        if self.nodata_check is not None:

            if self.use_labels:
                skip_chip = self.nodata_check(
                    img,
                    labels
                )

            else:
                skip_chip = self.nodata_check(
                    img
                )

            if skip_chip:
                return None, None, 0, 1

        return img, labels, 0, 0

    def _build_tile_chip_plan(
            self
    ):
        """Use the same number of random chips for every tile."""
        return [
            int(self.num_chips_per_tile)
            for _ in self.fns
        ]

    # ==========================================================
    # Build memmap cache
    # ==========================================================
    def _build_cache(self):

        label_bands = None
        label_band_counts = []

        # ======================================================
        # Pre-scan all label files to determine cache band count.
        #
        # This supports mixed label formats:
        #   some labels: 1 band
        #   some labels: 2/3 bands
        # Cache uses the maximum band count, and fewer-band labels
        # are padded with 0 before writing memmap.
        # ======================================================
        if self.use_labels:

            for _, label_fn in self.fns:

                if label_fn is None:
                    continue

                with rasterio.open(
                    label_fn
                ) as f:

                    band_count = int(
                        f.count
                    )

                if band_count <= 0:
                    raise ValueError(
                        f"Label has no band: {label_fn}"
                    )

                label_band_counts.append(
                    band_count
                )

            if len(label_band_counts) <= 0:
                raise ValueError(
                    "Cannot determine label bands."
                )

            label_bands = int(
                max(
                    label_band_counts
                )
            )

            if self.verbose:
                unique_band_counts = sorted(
                    set(
                        label_band_counts
                    )
                )

                print(
                    f"[Cache Build] label band counts detected: "
                    f"{unique_band_counts}; "
                    f"cache label_bands={label_bands}"
                )

        rng = np.random.default_rng(
            self.seed
        )

        # ======================================================
        # Build per-tile chip allocation plan
        # ======================================================
        tile_chip_counts = self._build_tile_chip_plan()

        max_samples = int(
            sum(tile_chip_counts)
        )

        if max_samples <= 0:
            raise RuntimeError(
                "No files were provided for cache building."
            )

        if self.verbose:
            print(
                f"[Cache Build] Start building chip cache..."
            )
            print(
                f"[Cache Build] dataset_name={self.dataset_name}"
            )
            print(
                f"[Cache Build] cache_dir={self.cache_dir}"
            )
            print(
                f"[Cache Build] max_samples={max_samples}"
            )
        image_shape = (
            max_samples,
            self.chip_size,
            self.chip_size,
            3
        )

        label_shape = (
            max_samples,
            self.chip_size,
            self.chip_size,
            int(label_bands) if self.use_labels else 1
        )

        images_mm = np.memmap(
            self.image_cache_path,
            dtype=self.image_cache_dtype,
            mode="w+",
            shape=image_shape
        )

        if self.use_labels:
            labels_mm = np.memmap(
                self.label_cache_path,
                dtype=self.label_cache_dtype,
                mode="w+",
                shape=label_shape
            )
        else:
            labels_mm = None

        sample_count = 0
        skipped_tiles = 0
        skipped_nodata = 0
        skipped_errors = 0

        file_iter = tqdm(
            enumerate(self.fns),
            total=len(self.fns),
            desc="[Cache Build]",
            unit="tile",
            file=sys.__stdout__
        )

        for tile_idx, (img_fn, label_fn) in file_iter:

            chips_this_tile = int(
                tile_chip_counts[tile_idx]
            )

            if chips_this_tile <= 0:
                continue

            try:

                with rasterio.open(img_fn, "r") as img_fp:

                    height, width = img_fp.shape

                    if (
                        width < self.chip_size
                        or height < self.chip_size
                    ):
                        skipped_tiles += 1
                        continue

                    if self.use_labels:

                        if label_fn is None:
                            raise ValueError(
                                "label_fn is None but use_labels=True"
                            )

                        label_fp_ctx = rasterio.open(
                            label_fn,
                            "r"
                        )

                    else:

                        label_fp_ctx = None

                    try:

                        if self.use_labels:

                            label_height, label_width = label_fp_ctx.shape

                            if (
                                height != label_height
                                or width != label_width
                            ):
                                raise ValueError(
                                    f"Image-label shape mismatch: "
                                    f"image=({height}, {width}), "
                                    f"label=({label_height}, {label_width}), "
                                    f"file={img_fn}"
                                )

                        valid_this_tile = 0
                        attempts = 0
                        max_attempts = (
                                chips_this_tile
                                * self.max_attempts_multiplier
                        )

                        while (
                                valid_this_tile < chips_this_tile
                                and attempts < max_attempts
                                and sample_count < max_samples
                        ):

                            attempts += 1

                            (
                                img,
                                labels,
                                err_count,
                                nodata_count
                            ) = self._sample_window(
                                img_fp=img_fp,
                                label_fp_ctx=label_fp_ctx,
                                width=width,
                                height=height,
                                rng=rng
                            )

                            skipped_errors += int(
                                err_count
                            )
                            skipped_nodata += int(
                                nodata_count
                            )

                            if img is None:
                                continue

                            img = self._convert_image_for_cache(
                                img
                            )

                            images_mm[sample_count] = img

                            if self.use_labels:

                                labels = self._pad_labels_to_band_count(
                                    labels=labels,
                                    target_bands=label_bands
                                )

                                labels = self._convert_label_for_cache(
                                    labels
                                )

                                labels_mm[sample_count] = np.moveaxis(
                                    labels,
                                    0,
                                    -1
                                )

                            sample_count += 1
                            valid_this_tile += 1

                        file_iter.set_postfix(
                            {
                                "chips": sample_count,
                                "skip_nodata": skipped_nodata,
                                "skip_tiles": skipped_tiles
                            }
                        )

                    finally:

                        if label_fp_ctx is not None:
                            label_fp_ctx.close()

            except RasterioError:

                skipped_tiles += 1
                continue

            except Exception:

                skipped_tiles += 1
                continue

        if sample_count == 0:
            raise RuntimeError(
                "No valid chips were cached. "
                "Please check paths, chip_size, label values, and nodata_check."
            )

        images_mm.flush()

        if labels_mm is not None:
            labels_mm.flush()

        meta = {
            "dataset_name": self.dataset_name,
            "num_samples": int(sample_count),
            "max_samples": int(max_samples),
            "chip_size": int(self.chip_size),
            "num_chips_per_tile": int(self.num_chips_per_tile),
            "tile_chip_counts": [
                int(x)
                for x in tile_chip_counts
            ],
            "use_labels": bool(self.use_labels),
            "seed": int(self.seed),
            "max_attempts_multiplier": int(self.max_attempts_multiplier),
            "image_cache_dtype": self.image_cache_dtype,
            "label_cache_dtype": self.label_cache_dtype,
            "image_cache_path": str(self.image_cache_path),
            "label_cache_path": str(self.label_cache_path),
            "image_shape": [
                int(max_samples),
                int(self.chip_size),
                int(self.chip_size),
                3
            ],
            "label_shape": [
                int(max_samples),
                int(self.chip_size),
                int(self.chip_size),
                int(label_bands) if self.use_labels else 1
            ],
            "label_band_counts_detected": sorted(
                list(
                    set(
                        label_band_counts
                    )
                )
            ) if self.use_labels else [],
            "skipped_tiles": int(skipped_tiles),
            "skipped_nodata": int(skipped_nodata),
            "skipped_errors": int(skipped_errors),
        }

        with open(
            self.meta_path,
            "w",
            encoding="utf-8"
        ) as f:

            json.dump(
                meta,
                f,
                indent=2
            )

        if self.verbose:

            print(
                f"[Cache Build] Finished."
            )
            print(
                f"[Cache Build] valid samples={sample_count}"
            )
            print(
                f"[Cache Build] skipped_tiles={skipped_tiles}"
            )
            print(
                f"[Cache Build] skipped_nodata={skipped_nodata}"
            )
            print(
                f"[Cache Build] skipped_errors={skipped_errors}"
            )

    # ==========================================================
    # Load cache metadata
    # ==========================================================
    def _load_meta(self):

        with open(
            self.meta_path,
            "r",
            encoding="utf-8"
        ) as f:

            self.meta = json.load(
                f
            )

        self.num_samples = int(
            self.meta["num_samples"]
        )

        self.max_samples = int(
            self.meta["max_samples"]
        )

        self.chip_size = int(
            self.meta["chip_size"]
        )

        self.num_chips_per_tile = int(
            self.meta["num_chips_per_tile"]
        )

        self.use_labels = bool(
            self.meta["use_labels"]
        )

        self.image_shape = tuple(
            self.meta["image_shape"]
        )

        self.label_shape = tuple(
            self.meta["label_shape"]
        )

        self.image_cache_dtype = self.meta["image_cache_dtype"]
        self.label_cache_dtype = self.meta["label_cache_dtype"]

        image_cache_path = Path(
            self.meta["image_cache_path"]
        )

        label_cache_path = Path(
            self.meta["label_cache_path"]
        )

        # Support cache directories moved after creation.
        if not image_cache_path.exists():
            image_cache_path = self.cache_dir / f"images_{self.image_cache_dtype}.dat"

        if not label_cache_path.exists():
            label_cache_path = self.cache_dir / f"labels_{self.label_cache_dtype}.dat"

        self.image_cache_path = image_cache_path
        self.label_cache_path = label_cache_path

    # ==========================================================
    # Lazy open memmaps in each worker
    # ==========================================================
    def _ensure_memmap_open(self):

        if self._images is None:

            self._images = np.memmap(
                self.image_cache_path,
                dtype=self.image_cache_dtype,
                mode="r",
                shape=self.image_shape
            )

        if self.use_labels and self._labels is None:

            self._labels = np.memmap(
                self.label_cache_path,
                dtype=self.label_cache_dtype,
                mode="r",
                shape=self.label_shape
            )

    # ==========================================================
    # Dataset length
    # ==========================================================
    def __len__(self):

        return self.num_samples

    # ==========================================================
    # Get item
    # ==========================================================
    def __getitem__(
        self,
        idx
    ):

        self._ensure_memmap_open()

        img = np.array(
            self._images[idx],
            copy=True
        )

        if self.use_labels:

            labels = np.array(self._labels[idx], copy=True)
            labels = np.moveaxis(labels, -1, 0)  # HWC → CHW

        else:

            labels = None

        # ------------------------------------------------------
        # Image transform
        # ------------------------------------------------------
        if self.image_transform is not None:

            img = self.image_transform(
                img
            )

        else:

            img = np.ascontiguousarray(
                img.transpose(
                    2,
                    0,
                    1
                )
            )

            img = torch.from_numpy(
                img
            ).float()

        # ------------------------------------------------------
        # Label transform
        # ------------------------------------------------------
        if self.use_labels:

            if self.label_transform is not None:

                labels = self.label_transform(
                    labels
                )

            else:

                labels = torch.from_numpy(
                    labels
                ).long()

            return img, labels

        else:

            return img
