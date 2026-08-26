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

# Default preserves the original release behavior. Set this to
# "union_original" (or pass label_mode explicitly) when coarse annotations
# have high precision but incomplete foreground coverage.
PSD_LABEL_MODE = "pseudo_only"
ORIGINAL_POSITIVE_VALUES = (2,)


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
    label_mode=PSD_LABEL_MODE,
    original_train_list=None,
    original_positive_values=ORIGINAL_POSITIVE_VALUES,
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
        "work_root": work_root,
        "prediction_root": prediction_root,
    }
    missing = [name for name, value in required.items() if not value]
    if missing:
        raise ValueError(f"Missing required PSD arguments: {missing}")
    if label_mode not in {"pseudo_only", "union_original"}:
        raise ValueError(
            "label_mode must be 'pseudo_only' or 'union_original', "
            f"got {label_mode!r}."
        )
    if label_mode == "union_original" and not original_train_list:
        raise ValueError("union_original requires original_train_list.")

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

    pseudo_label_arg = " --positive_values 1" if cycle > 1 else ""
    train_command = (
        f"python train.py --config {_quote(config_path)} "
        f"--work_dir {_quote(work_dir)} "
        f"--list_dir {_quote(train_list_path)}"
        f"{last_model_arg}"
        f"{pseudo_label_arg}"
    )

    inference_list_path = test_list_path if cycle == total_cycles else train_list_path

    test_command = (
        f"python test.py --config {_quote(config_path)} "
        f"--model_path {_quote(model_path)} "
        f"--save_path {_quote(prediction_dir)} "
        f"--list_dir {_quote(inference_list_path)}"
    )

    if cycle < total_cycles:
        merged_label_folder = os.path.join(work_dir, "labels_union_original")
        original_args = ""
        if label_mode == "union_original":
            if not original_positive_values:
                raise ValueError(
                    "union_original requires original_positive_values."
                )
            positive_values = " ".join(str(v) for v in original_positive_values)
            original_args = (
                f" --original_list {_quote(original_train_list)}"
                f" --original_positive_values {positive_values}"
                f" --merged_label_folder {_quote(merged_label_folder)}"
            )
        csv_command = (
            f"python generate_psd_csv.py "
            f"--input_list {_quote(train_list_path)} "
            f"--prediction_folder {_quote(prediction_dir)} "
            f"--output_list {_quote(next_list_path)} "
            f"--label_mode {label_mode}"
            f"{original_args}"
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
    label_mode=PSD_LABEL_MODE,
    original_positive_values=ORIGINAL_POSITIVE_VALUES,
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
    original_train_list = train_list_path
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
            label_mode=label_mode,
            original_train_list=original_train_list,
            original_positive_values=original_positive_values,
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
