# -*- coding: utf-8 -*-
"""
Train MESNet for UTCMapper

Config-driven entry script.

This script:
1. Loads training config from YAML.
2. Initializes MESNet.
3. Resolves dataset/list_dir/work_dir.
4. Starts trainer_dataset.

Compatible with trainer that supports:
    - MESNet only
    - single-band label training
"""

import argparse
import datetime
import os
import random
from types import SimpleNamespace

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import yaml

from networks.MESNet_UltraFast import MESNet
from trainer import trainer_dataset, dict_to_namespace
from utils import dataset_config
import shutil
import sys
import inspect
from pathlib import Path

# =============================================================================
# 0. Config helpers
# =============================================================================
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
    """
    Set args.name = value only when args does not have this field
    or the current value is None.
    """
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
# 2. Experiment record
# =============================================================================
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

def snapshot_experiment(args, cli_args, snapshot_path):
    """
    Save full training snapshot for reproducibility.
    """

    os.makedirs(snapshot_path, exist_ok=True)

    # ----------------------------------
    # 1. save config (yaml style)
    # ----------------------------------
    config_path = os.path.join(snapshot_path, "config.yaml")
    with open(config_path, "w", encoding="utf-8") as f:
        yaml.dump(namespace_to_dict(args), f, allow_unicode=True)

    # ----------------------------------
    # 2. save command line
    # ----------------------------------
    cmd_path = os.path.join(snapshot_path, "command.txt")
    with open(cmd_path, "w", encoding="utf-8") as f:
        f.write(" ".join(sys.argv) + "\n")

    # ----------------------------------
    # 3. save source files
    # ----------------------------------
    def copy_file(src, dst_dir):
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(snapshot_path, dst_dir))

    # root files
    copy_file("train.py", ".")
    copy_file("trainer.py", ".")
    copy_file("test.py", ".")
    copy_file("utils.py", ".")

    # networks
    os.makedirs(os.path.join(snapshot_path, "networks"), exist_ok=True)
    copy_file("networks/MESNet_UltraFast.py", "networks")
    copy_file("networks/loss.py", "networks")

    # ----------------------------------
    # 4. git commit (optional)
    # ----------------------------------
    try:
        import subprocess
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
        ).decode().strip()

        with open(os.path.join(snapshot_path, "git_commit.txt"), "w") as f:
            f.write(commit + "\n")
    except Exception:
        pass


def print_config_table(title, rows):
    rows = [
        (str(key), str(value))
        for key, value in rows
    ]

    if not rows:
        return

    key_width = max(len(key) for key, _ in rows)
    value_width = max(len(value) for _, value in rows)
    title_text = f" {title} "
    inner_width = max(
        key_width + value_width + 5,
        len(title_text),
    )

    print()
    print("+" + title_text.center(inner_width, "=") + "+")
    print(
        "| "
        + "Key".ljust(key_width)
        + " | "
        + "Value".ljust(value_width)
        + " |"
    )
    print(
        "+"
        + "-" * (key_width + 2)
        + "+"
        + "-" * (value_width + 2)
        + "+"
    )

    for key, value in rows:
        print(
            "| "
            + key.ljust(key_width)
            + " | "
            + value.ljust(value_width)
            + " |"
        )

    print(
        "+"
        + "-" * (key_width + 2)
        + "+"
        + "-" * (value_width + 2)
        + "+"
    )

# =============================================================================
# 2. Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Train MESNet for UTC mapping"
    )

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

    parser.add_argument(
        "--work_dir",
        type=str,
        default=None,
        help="Override work_dir in config.",
    )

    parser.add_argument(
        "--list_dir",
        type=str,
        default=None,
        help="Override list_dir in config.",
    )

    parser.add_argument(
        "--last_model",
        type=str,
        default=None,
        help="Override last_model / pretrained_model in config.",
    )

    parser.add_argument(
        "--positive_values",
        nargs="+",
        type=int,
        default=None,
        help="Override label.positive_values (PSD predictions use value 1).",
    )

    cli_args = parser.parse_args()

    # -------------------------------------------------------------------------
    # Load config
    # -------------------------------------------------------------------------
    args = load_config(cli_args.config)

    # -------------------------------------------------------------------------
    # CLI overrides
    # -------------------------------------------------------------------------
    if cli_args.gpu is not None:
        args.gpu = cli_args.gpu

    if cli_args.work_dir is not None:
        args.work_dir = cli_args.work_dir

    if cli_args.list_dir is not None:
        args.list_dir = cli_args.list_dir

    if cli_args.last_model is not None:
        args.last_model = cli_args.last_model
        args.pretrained_model = cli_args.last_model

    if cli_args.positive_values is not None:
        if not hasattr(args, "label") or args.label is None:
            args.label = SimpleNamespace()
        args.label.positive_values = list(cli_args.positive_values)

    # -------------------------------------------------------------------------
    # Basic defaults
    # -------------------------------------------------------------------------
    set_arg_if_missing(args, "dataset", "Shanghai-0.3m-center-merge-sm-train")
    set_arg_if_missing(args, "max_epochs", 100)
    set_arg_if_missing(args, "batch_size", 16)
    set_arg_if_missing(args, "base_lr", 0.01)
    set_arg_if_missing(args, "seed", 1234)
    set_arg_if_missing(args, "gpu", "0")
    set_arg_if_missing(args, "work_dir", None)
    set_arg_if_missing(args, "list_dir", None)
    set_arg_if_missing(args, "last_model", None)
    set_arg_if_missing(args, "pretrained_model", None)

    if args.pretrained_model is None and args.last_model is not None:
        args.pretrained_model = args.last_model

    # Keep compatibility with older trainer.
    set_arg_if_missing(args, "length", 5)
    set_arg_if_missing(args, "num_workers", 6)
    set_arg_if_missing(args, "chip_size", 224)
    if (
        not hasattr(args, "n_chips")
        and not hasattr(args, "num_chips_per_tile")
    ):
        args.num_chips_per_tile = 50
    set_arg_if_missing(args, "ratio", 1.0)
    set_arg_if_missing(args, "aug", False)
    set_arg_if_missing(args, "skip_nt", True)
    set_arg_if_missing(args, "save_interval", 20)
    set_arg_if_missing(args, "clip_grad", 1.0)

    normalize_runtime_config(args)

    # -------------------------------------------------------------------------
    # GPU setup and seeds
    # -------------------------------------------------------------------------
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    cudnn.benchmark = True
    cudnn.deterministic = False

    setup_seed(args.seed)

    # -------------------------------------------------------------------------
    # Dataset configuration fallback
    # -------------------------------------------------------------------------
    dataset_name = args.dataset

    if dataset_name in dataset_config:
        if not hasattr(args, "num_classes") or args.num_classes is None:
            args.num_classes = dataset_config[dataset_name].get(
                "num_classes",
                2,
            )

        if args.list_dir is None:
            args.list_dir = dataset_config[dataset_name].get(
                "list_dir",
                None,
            )

    else:
        print(
            f"WARNING: Dataset {dataset_name} not found in dataset_config. "
            f"Will use list_dir from config."
        )

        set_arg_if_missing(args, "num_classes", 2)

    if args.list_dir is None:
        raise ValueError(
            "list_dir is None. Please set list_dir in config "
            "or define it in dataset_config."
        )

    # -------------------------------------------------------------------------
    # Snapshot path
    # -------------------------------------------------------------------------
    if args.work_dir is not None:
        snapshot_path = args.work_dir
    else:
        snapshot_path = os.path.join(
            "work_dirs",
            f"{args.dataset}_MESNet_"
            f"{datetime.datetime.now().strftime('%Y-%m-%d')}",
        )

    os.makedirs(
        snapshot_path,
        exist_ok=True,
    )

    snapshot_experiment(args, cli_args, snapshot_path)

    # -------------------------------------------------------------------------
    # Print key config
    # -------------------------------------------------------------------------
    print_config_table(
        "Training Config",
        [
            ("Dataset", args.dataset),
            ("List dir", args.list_dir),
            ("Work dir", snapshot_path),
            ("GPU", args.gpu),
            ("Seed", args.seed),
            ("Max epochs", args.max_epochs),
            ("Batch size", args.batch_size),
            ("Chip size", args.chip_size),
            ("Base LR", args.base_lr),
            (
                "Num chips per tile",
                getattr(
                    args,
                    "num_chips_per_tile",
                    getattr(args, "n_chips", 50),
                ),
            ),
        ],
    )

    label_cfg = getattr(args, "label", None)

    if label_cfg is not None:
        print_config_table(
            "Label Config",
            [
                ("positive_values", getattr(label_cfg, "positive_values", [2])),
                ("ignore_value", getattr(label_cfg, "ignore_value", 255)),
            ],
        )

    # =========================
    # Model
    # =========================
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    net = MESNet.from_config(args)
    net = net.to(device)
    net = net.build_runtime()


    # -------------------------------------------------------------------------
    # Start training
    # -------------------------------------------------------------------------
    trainer_dataset(
        args,
        net,
        snapshot_path,
    )


if __name__ == "__main__":
    main()
