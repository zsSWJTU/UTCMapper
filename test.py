# -*- coding: utf-8 -*-
"""
Inference script for MESNet / UTCMapper - Config Version

This script performs tile-wise inference on input raster datasets using MESNet.

Compatible with config-driven training scripts:
    - normalization.image_means
    - normalization.image_stds
    - model.width
    - model.image_band
    - model.num_blocks
    - inference.batch_size
    - inference.chip_size
    - inference.stride
    - inference.windowed_sampling
    - inference.num_workers
    - inference.amp
"""

import argparse
import os
import random
from types import SimpleNamespace

import numpy as np
import pandas as pd
import rasterio
import torch
import torch.backends.cudnn as cudnn
import yaml

from rasterio.errors import RasterioIOError
from rasterio.windows import Window
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

import utils
from networks.MESNet_UltraFast import MESNet


# =============================================================================
# 0. Config helpers
# =============================================================================

def dict_to_namespace(d):
    """
    Recursively convert dict to SimpleNamespace.
    """
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


def load_config(config_path):
    """
    Load YAML config file.
    """
    if config_path is None or config_path == "":
        raise ValueError("Config path is empty.")

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


def set_arg_if_missing(args, name, value):
    if not hasattr(args, name) or getattr(args, name) is None:
        setattr(args, name, value)


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


def setup_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# =============================================================================
# 1. Image transform
# =============================================================================

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

    def __call__(self, img):
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
    Build a pickle-safe image normalization transform from config.
    """
    return ImageTransform(
        image_means=image_means,
        image_stds=image_stds,
        image_band=image_band,
    )


# =============================================================================
# 2. Dataset
# =============================================================================

class TileInferenceDataset(Dataset):
    """
    Tile-wise inference dataset.

    Output:
        img: Tensor [C, chip_size, chip_size]
        coords: Tensor / ndarray [2], meaning [y, x]
    """

    def __init__(
        self,
        fn,
        chip_size,
        stride,
        transform=None,
        windowed_sampling=True,
        image_band=3,
        verbose=False,
    ):
        self.fn = fn
        self.chip_size = chip_size
        self.stride = stride
        self.transform = transform
        self.windowed_sampling = windowed_sampling
        self.image_band = image_band
        self.verbose = verbose

        with rasterio.open(self.fn) as f:
            self.height = f.height
            self.width = f.width
            self.num_channels = f.count
            self.dtype = f.profile["dtype"]

            if self.num_channels < self.image_band:
                raise ValueError(
                    f"Image has {self.num_channels} bands, "
                    f"but image_band={self.image_band}: {self.fn}"
                )

            if not self.windowed_sampling:
                data = f.read(
                    indexes=list(range(1, self.image_band + 1))
                )
                self.data = np.moveaxis(data, 0, -1)

        self.chip_coordinates = self._build_chip_coordinates()
        self.num_chips = len(self.chip_coordinates)

        if self.verbose:
            print(
                f"Constructed TileInferenceDataset -- "
                f"{self.height}x{self.width}, "
                f"{self.num_channels} channels ({self.dtype}). "
                f"Total chips: {self.num_chips}"
            )

    def _build_chip_coordinates(self):
        """
        Generate top-left coordinates for sliding-window inference.
        """
        if self.height <= self.chip_size:
            y_list = [0]
        else:
            y_list = list(
                range(
                    0,
                    self.height - self.chip_size,
                    self.stride,
                )
            )
            y_list.append(self.height - self.chip_size)

        if self.width <= self.chip_size:
            x_list = [0]
        else:
            x_list = list(
                range(
                    0,
                    self.width - self.chip_size,
                    self.stride,
                )
            )
            x_list.append(self.width - self.chip_size)

        coords = [
            (y, x)
            for y in y_list
            for x in x_list
        ]

        return coords

    def __getitem__(self, idx):
        y, x = self.chip_coordinates[idx]

        if self.windowed_sampling:
            try:
                with rasterio.open(self.fn) as f:
                    img = f.read(
                        indexes=list(range(1, self.image_band + 1)),
                        window=Window(
                            x,
                            y,
                            self.chip_size,
                            self.chip_size,
                        ),
                    )
                    img = np.moveaxis(img, 0, -1)

            except RasterioIOError:
                print(f"Reading chip {idx} failed, returning zeros")
                img = np.zeros(
                    (
                        self.chip_size,
                        self.chip_size,
                        self.image_band,
                    ),
                    dtype=np.uint8,
                )

        else:
            img = self.data[
                y:y + self.chip_size,
                x:x + self.chip_size,
                :,
            ]

        if self.transform is not None:
            img = self.transform(img)
        else:
            img = np.moveaxis(img, 2, 0)
            img = torch.from_numpy(
                np.ascontiguousarray(
                    img,
                    dtype=np.float32,
                )
            )

        return img, np.array((y, x), dtype=np.int64)

    def __len__(self):
        return self.num_chips


# =============================================================================
# 3. Model
# =============================================================================

def build_mesnet_from_config(args):
    """
    Build MESNet from config.

    Supports:
        model:
          width: 64
          image_band: 3
          num_blocks: 5

    Tries new signature first:
        MESNet(width, image_band)

    Falls back to:
        MESNet(num_blocks=num_blocks)
    """
    model_cfg = getattr(args, "model", None)

    width = get_arg(args, "width", 64)
    image_band = get_arg(args, "image_band", 3)
    model_cfg = getattr(args, "model", None)

    if model_cfg is not None:
        image_band = getattr(model_cfg, "image_band", image_band)
    num_blocks = get_arg(args, "length", 5)

    if model_cfg is not None:
        width = getattr(model_cfg, "width", width)
        image_band = getattr(model_cfg, "image_band", image_band)
        num_blocks = getattr(model_cfg, "num_blocks", num_blocks)

    args.width = width
    args.image_band = image_band
    args.length = num_blocks

    try:
        net = MESNet(
            width,
            image_band,
        )
        print(
            f"Initialized MESNet with "
            f"width={width}, image_band={image_band}"
        )

    except TypeError:
        net = MESNet(
            num_blocks=num_blocks,
        )
        print(
            f"Initialized MESNet with "
            f"num_blocks={num_blocks}"
        )

    return net


def load_model_weights(model, model_path, device):
    """
    Load model weights.

    Supports:
        - pure state_dict
        - checkpoint with key 'state_dict'
        - DataParallel 'module.' prefix
    """
    if model_path is None or model_path == "":
        raise ValueError("model_path is empty.")

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"model_path not found: {model_path}")

    print(f"Loading model weights: {model_path}")

    state = torch.load(
        model_path,
        map_location=device,
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

    print(f"Missing keys   : {len(missing)}")
    print(f"Unexpected keys: {len(unexpected)}")

    if len(missing) > 0:
        print("First missing keys:", missing[:10])

    if len(unexpected) > 0:
        print("First unexpected keys:", unexpected[:10])

    return model


# =============================================================================
# 4. Inference
# =============================================================================

def get_output_name(image_fn, output_suffix="_pred"):
    """
    Build output filename.

    Examples:
        xxx.tif -> xxx_pred.tif
    """
    base = os.path.basename(image_fn)
    stem, ext = os.path.splitext(base)

    if ext == "":
        ext = ".tif"

    return f"{stem}{output_suffix}{ext}"

def extract_city_from_image_fn(
    image_fn,
    multi_word_cities=None,
):
    """
    Extract city name from image basename.

    Expected basename format:
        {city}_xxx.tif

    Special cases:
        taizhou and taizhou_zhejiang are different cities.

    multi_word_cities:
        Optional list of city names that contain underscores.
        Longer names are checked first.
    """

    base = os.path.basename(
        str(image_fn)
    )

    stem = os.path.splitext(
        base
    )[0]

    if multi_word_cities is None:
        multi_word_cities = [
            "taizhou_zhejiang",
            "taizhou_zj",
            "taizhou_js",
        ]

    # --------------------------------------------------
    # Check longer city names first.
    # Otherwise taizhou_zhejiang_xxx.tif may be treated
    # as taizhou.
    # --------------------------------------------------
    multi_word_cities = sorted(
        multi_word_cities,
        key=len,
        reverse=True,
    )

    for city in multi_word_cities:
        prefix = f"{city}_"

        if stem.startswith(prefix):
            return city

    return stem.split("_")[0]


def inference(args, model, test_save_path):
    os.makedirs(
        test_save_path,
        exist_ok=True,
    )

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    model = model.to(device)
    model.eval()

    # -------------------------------------------------------------------------
    # Config
    # -------------------------------------------------------------------------
    batch_size = get_nested_arg(
        args,
        "inference",
        "batch_size",
        get_arg(args, "batch_size", 32),
    )

    chip_size = get_nested_arg(
        args,
        "inference",
        "chip_size",
        get_arg(args, "chip_size", 224),
    )

    stride = get_nested_arg(
        args,
        "inference",
        "stride",
        None,
    )

    if stride is None:
        stride = chip_size // 2
        padding = chip_size - stride
        half_padding = padding // 2
    else:
        padding = chip_size - stride
        half_padding = padding // 2

    num_workers = get_nested_arg(
        args,
        "inference",
        "num_workers",
        0,
    )

    pin_memory = get_nested_arg(
        args,
        "inference",
        "pin_memory",
        True,
    )

    windowed_sampling = get_nested_arg(
        args,
        "inference",
        "windowed_sampling",
        True,
    )

    use_amp = get_nested_arg(
        args,
        "inference",
        "amp",
        True,
    )

    output_suffix = get_nested_arg(
        args,
        "inference",
        "output_suffix",
        "_pred",
    )

    skip_existing = get_nested_arg(
        args,
        "inference",
        "skip_existing",
        True,
    )

    save_prob = get_nested_arg(
        args,
        "inference",
        "save_prob",
        False,
    )

    # -------------------------------------------------------------------------
    # Save predictions into city-named subfolders when multiple cities exist
    # -------------------------------------------------------------------------
    separate_by_city = get_nested_arg(
        args,
        "inference",
        "separate_by_city",
        True,
    )

    multi_word_cities = get_nested_arg(
        args,
        "inference",
        "multi_word_cities",
        [
            "taizhou_zhejiang",
            "taizhou_zj",
            "taizhou_js",
        ],
    )

    image_band = get_arg(args, "image_band", 3)

    num_classes = get_arg(args, "num_classes", 2)

    if device.type != "cuda":
        use_amp = False
        pin_memory = False

    # -------------------------------------------------------------------------
    # Normalization
    # -------------------------------------------------------------------------
    image_means = get_nested_arg(args, "normalization", "image_means", None)
    image_stds = get_nested_arg(args, "normalization", "image_stds", None)

    image_transform = build_image_transform(
        image_means=image_means,
        image_stds=image_stds,
        image_band=image_band,
    )

    # -------------------------------------------------------------------------
    # Load image list
    # -------------------------------------------------------------------------
    input_dataframe = pd.read_csv(args.list_dir)

    if "image_fn" not in input_dataframe.columns:
        raise ValueError("CSV must contain column: image_fn")

    image_fns = input_dataframe["image_fn"].values

    # -------------------------------------------------------------------------
    # Detect cities in this inference list
    # -------------------------------------------------------------------------
    image_cities = [
        extract_city_from_image_fn(
            image_fn,
            multi_word_cities=multi_word_cities,
        )
        for image_fn in image_fns
    ]

    unique_cities = sorted(
        set(image_cities)
    )

    use_city_subfolders = (
            bool(separate_by_city)
            and len(unique_cities) > 1
    )

    if use_city_subfolders:
        print(
            f"Detected {len(unique_cities)} cities. "
            f"Predictions will be saved into city subfolders."
        )
        print(
            "Cities:",
            unique_cities,
        )
    else:
        print(
            f"Detected {len(unique_cities)} city. "
            f"Predictions will be saved directly into save_path."
        )

    print("\n========== Inference Config ==========")
    print(f"List dir        : {args.list_dir}")
    print(f"Save path       : {test_save_path}")
    print(f"Batch size      : {batch_size}")
    print(f"Chip size       : {chip_size}")
    print(f"Stride          : {stride}")
    print(f"Num workers     : {num_workers}")
    print(f"Windowed        : {windowed_sampling}")
    print(f"AMP             : {use_amp}")
    print(f"Image band      : {image_band}")
    print(f"Num classes     : {num_classes}")
    print(f"Image means     : {image_means}")
    print(f"Image stds      : {image_stds}")
    print(f"Save prob       : {save_prob}")
    print(f"Separate city   : {separate_by_city}")
    print(f"City subfolders : {use_city_subfolders}")

    # -------------------------------------------------------------------------
    # Inference loop
    # -------------------------------------------------------------------------
    for idx, image_fn in enumerate(
            tqdm(
                image_fns,
                desc="Images",
                unit="image",
            )
    ):
        city_name = image_cities[idx]

        if use_city_subfolders:
            current_save_path = os.path.join(
                test_save_path,
                city_name,
            )
        else:
            current_save_path = test_save_path

        os.makedirs(
            current_save_path,
            exist_ok=True,
        )

        output_name = get_output_name(
            image_fn,
            output_suffix=output_suffix,
        )

        output_fn = os.path.join(
            current_save_path,
            output_name,
        )

        prob_output_fn = os.path.join(
            current_save_path,
            os.path.splitext(output_name)[0] + "_prob.tif",
        )

        if skip_existing and os.path.exists(output_fn):
            print(f"{output_fn} already exists, skipping...")
            continue

        print(
            f"({idx + 1}/{len(image_fns)}) Processing {image_fn} ..."
        )

        try:
            with rasterio.open(image_fn) as f:
                input_width = f.width
                input_height = f.height
                input_profile = f.profile.copy()

            dataset = TileInferenceDataset(
                image_fn,
                chip_size=chip_size,
                stride=stride,
                transform=image_transform,
                windowed_sampling=windowed_sampling,
                image_band=image_band,
                verbose=False,
            )

        except Exception as e:
            print(f"Failed to prepare image: {image_fn}")
            print(e)
            continue

        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=False,
        )

        output = np.zeros(
            (
                num_classes,
                input_height,
                input_width,
            ),
            dtype=np.float32,
        )

        counts = np.zeros(
            (
                input_height,
                input_width,
            ),
            dtype=np.float32,
        )

        kernel = np.ones(
            (
                chip_size,
                chip_size,
            ),
            dtype=np.float32,
        )

        if half_padding > 0:
            kernel[
                half_padding:-half_padding,
                half_padding:-half_padding,
            ] = 5.0

        with torch.no_grad():
            for data, coords in tqdm(
                dataloader,
                desc="Chips",
                leave=False,
            ):
                data = data.to(
                    device,
                    non_blocking=True,
                )

                with torch.amp.autocast(
                    "cuda",
                    enabled=(use_amp and device.type == "cuda"),
                ):
                    logits = model(data)

                    # ---------------------------------------------------------
                    # For 2-class segmentation, use softmax.
                    # If your MESNet outputs one channel, use sigmoid fallback.
                    # ---------------------------------------------------------
                    if logits.shape[1] == 1:
                        prob_tree = torch.sigmoid(logits)
                        prob_bg = 1.0 - prob_tree
                        prob = torch.cat(
                            [prob_bg, prob_tree],
                            dim=1,
                        )
                    else:
                        prob = torch.softmax(
                            logits,
                            dim=1,
                        )

                prob_np = prob.float().cpu().numpy()

                coords_np = coords.numpy()

                for j in range(prob_np.shape[0]):
                    y = int(coords_np[j][0])
                    x = int(coords_np[j][1])

                    output[
                        :,
                        y:y + chip_size,
                        x:x + chip_size,
                    ] += prob_np[j] * kernel

                    counts[
                        y:y + chip_size,
                        x:x + chip_size,
                    ] += kernel

        counts[counts == 0] = 1.0
        output = output / counts[np.newaxis, :, :]

        output_hard = output.argmax(axis=0).astype(np.uint8)

        # ---------------------------------------------------------------------
        # Save hard prediction
        # ---------------------------------------------------------------------
        output_profile = input_profile.copy()

        output_profile.update(
            driver="GTiff",
            dtype="uint8",
            count=1,
            nodata=0,
            compress="lzw",
        )

        with rasterio.open(
            output_fn,
            "w",
            **output_profile,
        ) as f:
            f.write(
                output_hard,
                1,
            )

            colormap = getattr(utils, "LABEL_IDX_COLORMAP", None)
            if colormap is not None:
                f.write_colormap(1, colormap)

        # ---------------------------------------------------------------------
        # Optional: save probability
        # ---------------------------------------------------------------------
        if save_prob:
            prob_profile = input_profile.copy()

            prob_profile.update(
                driver="GTiff",
                dtype="float32",
                count=num_classes,
                nodata=None,
                compress="lzw",
            )

            with rasterio.open(
                prob_output_fn,
                "w",
                **prob_profile,
            ) as f:
                f.write(
                    output.astype(np.float32),
                )

        print(f"Saved: {output_fn}")

    return "Testing Finished!"


# =============================================================================
# 5. Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Inference MESNet for UTC mapping"
    )

    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to YAML config file.",
    )

    parser.add_argument(
        "--model_path",
        type=str,
        default=None,
        help="Override model_path in config.",
    )

    parser.add_argument(
        "--save_path",
        type=str,
        default=None,
        help="Override save_path in config.",
    )

    parser.add_argument(
        "--list_dir",
        type=str,
        default=None,
        help="Override list_dir in config.",
    )

    parser.add_argument(
        "--gpu",
        type=str,
        default=None,
        help="Override GPU id in config.",
    )

    cli_args = parser.parse_args()

    args = load_config(cli_args.config)
    normalize_runtime_config(args)

    # -------------------------------------------------------------------------
    # CLI overrides
    # -------------------------------------------------------------------------
    if cli_args.gpu is not None:
        args.gpu = cli_args.gpu

    if cli_args.model_path is not None:
        args.model_path = cli_args.model_path

    if cli_args.save_path is not None:
        args.save_path = cli_args.save_path

    if cli_args.list_dir is not None:
        args.list_dir = cli_args.list_dir

    # -------------------------------------------------------------------------
    # Defaults
    # -------------------------------------------------------------------------
    set_arg_if_missing(args, "dataset", "")
    set_arg_if_missing(args, "gpu", "0")
    set_arg_if_missing(args, "seed", 1234)
    set_arg_if_missing(args, "batch_size", 32)
    set_arg_if_missing(args, "chip_size", 224)
    set_arg_if_missing(args, "model_path", "")
    set_arg_if_missing(args, "save_path", "")
    set_arg_if_missing(args, "list_dir", None)
    set_arg_if_missing(args, "num_classes", 2)

    # -------------------------------------------------------------------------
    # GPU / seed
    # -------------------------------------------------------------------------
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    cudnn.benchmark = True
    cudnn.deterministic = False

    setup_seed(args.seed)

    # -------------------------------------------------------------------------
    # Dataset config fallback
    # -------------------------------------------------------------------------
    if args.dataset != "" and hasattr(utils, "dataset_config"):
        print("Available datasets:", list(utils.dataset_config.keys()))

        if args.dataset in utils.dataset_config:
            dataset_cfg = utils.dataset_config[args.dataset]

            if args.list_dir is None:
                args.list_dir = dataset_cfg.get(
                    "list_dir",
                    None,
                )

            if not hasattr(args, "num_classes") or args.num_classes is None:
                args.num_classes = dataset_cfg.get(
                    "num_classes",
                    2,
                )

        else:
            print(
                f"WARNING: dataset {args.dataset} not found in utils.dataset_config. "
                f"Use list_dir from config."
            )

    if args.list_dir is None:
        raise ValueError(
            "list_dir is None. Please set list_dir in config "
            "or pass --list_dir."
        )

    if args.model_path is None or args.model_path == "":
        raise ValueError(
            "model_path is required. Please set model_path in config "
            "or pass --model_path."
        )

    # -------------------------------------------------------------------------
    # Make save path
    # -------------------------------------------------------------------------
    if args.save_path is not None and args.save_path != "":
        test_save_path = args.save_path
    else:
        snapshot = args.model_path
        parent_name = os.path.basename(
            os.path.dirname(snapshot)
        )

        if parent_name == "":
            task_name = "MESNet"
        else:
            parts = parent_name.split("_")
            task_name = "_".join(parts[1:]) if len(parts) > 1 else parent_name

        dataset_name = args.dataset if args.dataset != "" else "dataset"

        test_save_path = os.path.join(
            "dataset",
            dataset_name,
            f"Prediction_{task_name}",
        )

    os.makedirs(
        test_save_path,
        exist_ok=True,
    )

    print("Results will be saved to:", test_save_path)

    # -------------------------------------------------------------------------
    # Model
    # -------------------------------------------------------------------------
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    net = MESNet.from_config(args)
    net = net.to(device)
    net = net.build_runtime()
    model  = load_model_weights(
        net,
        args.model_path,
        device=device,
    )

    # -------------------------------------------------------------------------
    # Inference
    # -------------------------------------------------------------------------
    inference(
        args,
        model ,
        test_save_path,
    )


if __name__ == "__main__":
    main()
