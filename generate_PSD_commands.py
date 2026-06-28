"""
Generate commands for Progressive Self-Distillation (PSD).

The script is path-free by default: provide config, CSV, image, and output
locations when calling ``generate_psd_commands``. Only the public release
dataset names ``Shanghai-center-train`` and ``Shanghai-center-test`` are
accepted.
"""

import datetime
import os


TRAIN_DATASET = "Shanghai-center-train"
TEST_DATASET = "Shanghai-center-test"
ALLOWED_DATASETS = {TRAIN_DATASET, TEST_DATASET}


def create_folder_if_not_exists(path):
    """Create the parent folder for an output command file."""
    directory = os.path.dirname(path)
    if directory and not os.path.exists(directory):
        os.makedirs(directory)


def _validate_dataset(dataset):
    """Reject internal or unsupported dataset names."""
    if dataset not in ALLOWED_DATASETS:
        raise ValueError(
            f"Unsupported dataset: {dataset}. "
            f"Expected one of: {sorted(ALLOWED_DATASETS)}"
        )


def _quote(value):
    """Quote command values so paths with spaces remain valid."""
    return f'"{value}"'


def generate_commands(
    cycle=1,
    total_cycles=1,
    dataset=TRAIN_DATASET,
    config_path=None,
    train_list_path=None,
    test_list_path=None,
    image_folder=None,
    work_root=None,
    prediction_root=None,
    last_model=None,
):
    """
    Generate train, inference, and optional CSV-refresh commands for one cycle.

    Parameters are intentionally explicit so this file does not contain private
    or machine-specific paths.
    """
    _validate_dataset(dataset)

    required = {
        "config_path": config_path,
        "train_list_path": train_list_path,
        "test_list_path": test_list_path,
        "image_folder": image_folder,
        "work_root": work_root,
        "prediction_root": prediction_root,
    }
    missing = [name for name, value in required.items() if not value]
    if missing:
        raise ValueError(f"Missing required PSD arguments: {missing}")

    day = datetime.datetime.now().strftime("%Y-%m-%d")
    run_name = (
        f"{dataset}_cycle{cycle}_{day}"
        if total_cycles > 1
        else f"{dataset}_{day}"
    )

    work_dir = os.path.join(work_root, run_name)
    model_path = os.path.join(work_dir, "final.pth")
    prediction_dir = os.path.join(
        prediction_root,
        f"cycle{cycle}_of_{total_cycles}_{day}",
    )
    next_list_path = os.path.join(work_dir, "list.csv")

    last_model_arg = f" --last_model {_quote(last_model)}" if last_model else ""

    train_command = (
        f"python train.py --config {_quote(config_path)} "
        f"--work_dir {_quote(work_dir)} "
        f"--list_dir {_quote(train_list_path)}"
        f"{last_model_arg}"
    )

    inference_list_path = test_list_path if cycle == total_cycles else train_list_path

    test_command = (
        f"python test.py --config {_quote(config_path)} "
        f"--model_path {_quote(model_path)} "
        f"--save_path {_quote(prediction_dir)} "
        f"--list_dir {_quote(inference_list_path)}"
    )

    if cycle < total_cycles:
        csv_command = (
            f"python generate_dataset_csv.py "
            f"--image_folder {_quote(image_folder)} "
            f"--label_folder {_quote(prediction_dir)} "
            f"--new_file_path {_quote(next_list_path)}"
        )
    else:
        csv_command = None

    return [train_command, test_command, csv_command], next_list_path, model_path


def generate_psd_commands(
    total_cycles=2,
    dataset=TRAIN_DATASET,
    config_path=None,
    train_list_path=None,
    test_list_path=None,
    image_folder=None,
    work_root=None,
    prediction_root=None,
    command_root=None,
):
    """
    Generate all PSD commands and save them as Linux shell and Windows batch files.

    The first cycle trains from ``train_list_path``. Later cycles use the CSV
    generated from the previous cycle's predictions. The final inference command
    runs on ``Shanghai-center-test`` through ``test_list_path``.
    """
    _validate_dataset(dataset)

    if total_cycles < 1:
        raise ValueError("total_cycles must be >= 1")

    if command_root is None:
        raise ValueError("command_root is required.")

    day = datetime.datetime.now().strftime("%Y-%m-%d")
    command_list = []

    txt_file = os.path.join(command_root, f"{dataset}_PSD_{day}.txt")
    bat_file = os.path.join(command_root, f"{dataset}_PSD_{day}.bat")
    create_folder_if_not_exists(txt_file)

    current_train_list = train_list_path
    last_model_path = None

    for cycle in range(1, total_cycles + 1):
        commands, current_train_list, last_model_path = generate_commands(
            cycle=cycle,
            total_cycles=total_cycles,
            dataset=dataset,
            config_path=config_path,
            train_list_path=current_train_list,
            test_list_path=test_list_path,
            image_folder=image_folder,
            work_root=work_root,
            prediction_root=prediction_root,
            last_model=last_model_path,
        )
        command_list.extend(cmd for cmd in commands if cmd)

    with open(txt_file, "w", encoding="utf-8") as f:
        f.write("\n".join(command_list))
        f.write("\n")

    with open(bat_file, "w", encoding="utf-8") as f:
        f.write("@echo off\n")
        f.write("\n".join(command_list))
        f.write("\n")

    print(f"Windows commands: {bat_file}")
    print(f"Linux commands: {txt_file}")
    return command_list


if __name__ == "__main__":
    raise SystemExit(
        "Import this module and call generate_psd_commands(...) with explicit "
        "paths for config_path, train_list_path, test_list_path, image_folder, "
        "work_root, prediction_root, and command_root."
    )
