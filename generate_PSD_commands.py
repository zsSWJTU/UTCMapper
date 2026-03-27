"""
PSD Command Generator

This script implements the Progressive Self-Distillation (PSD) strategy for multi-cycle training and prediction.
Each cycle:
    1. Trains the model using the current labels (initially coarse labels).
    2. Uses the trained model to predict and update labels for the next cycle.
    3. Generates training, testing, and dataset CSV commands for each cycle.
It outputs Linux (.txt) and Windows (.bat) command files.
"""

import os
import datetime
from utils import dataset_config

dataset_info = dataset_config


def create_folder_if_not_exists(path):
    """Ensure the parent directory of a path exists."""
    directory = os.path.dirname(path)
    if not os.path.exists(directory):
        os.makedirs(directory)


def generate_commands(cycle=1, total_cycles=1, dataset='Shanghai-train',
                      batch_size=1, max_epochs=50, length=5, lr=0.01,
                      total_epochs=100, last_model=None):
    """
    Generate commands for a single PSD cycle.
    last_model: path to previous cycle's model (None for first cycle)
    """
    day = datetime.datetime.now().strftime('%Y-%m-%d')
    work_dir = f'work_dirs/{dataset}_cycle{cycle}_{day}' if total_cycles > 1 else f'work_dirs/{dataset}_{day}'
    model_path = os.path.join(work_dir, f'epoch_{max_epochs - 1}.pth')  # last epoch model
    image_folder = dataset_info[dataset]['image_dir']
    new_file_path = os.path.join(work_dir, 'list.csv')

    # Add --last_model if provided (not first cycle)
    last_model_arg = f" --last_model {last_model}" if last_model else ""

    train_command = (
        f"python train.py --dataset {dataset} "
        f"--batch_size {batch_size} --max_epochs {max_epochs} "
        f"--work_dir {work_dir} --length {length} --base_lr {lr}"
        f"{last_model_arg}"
    )

    # Determine test dataset
    test_dataset = dataset.replace('train', 'test') if cycle == total_cycles else dataset
    save_path = f'dataset/{test_dataset}/Prediction_cycle{cycle}_total_{total_cycles}x_{total_epochs}e_{day}/' \
        if total_cycles > 1 else f'dataset/{dataset}/Prediction_{day}/'
    label_folder = save_path
    test_command = (
        f"python test.py --dataset {test_dataset} "
        f"--model_path {model_path} --save_path {save_path} --length {length}"
    )

    if cycle < total_cycles:
        csv_command = (
            f"python generate_dataset_csv.py --image_folder {image_folder} "
            f"--label_folder {label_folder} --new_file_path {new_file_path}"
        )
    else:
        csv_command = None

    return [train_command, test_command, csv_command], new_file_path, model_path


def generate_psd_commands(total_cycles=2, total_epochs=100, dataset='Shanghai-train',
                          batch_size=1, length=5, lr=0.01, foot_dir='commands'):
    """
    Generate all commands for PSD strategy.
    Each cycle (after the first) uses the previous cycle's last model as --last_model.
    """
    day = datetime.datetime.now().strftime('%Y-%m-%d')
    command_list = []
    each_cycle_epoch = total_epochs // total_cycles

    txt_file = os.path.join(foot_dir, f'{dataset}_PSD_{day}.txt')
    bat_file = txt_file.replace('.txt', '.bat')
    create_folder_if_not_exists(txt_file)

    last_model_path = None
    last_list_path = None

    for cycle in range(1, total_cycles + 1):
        commands, last_list_path, last_model_path = generate_commands(
            cycle=cycle,
            total_cycles=total_cycles,
            dataset=dataset,
            batch_size=batch_size,
            max_epochs=each_cycle_epoch,
            length=length,
            lr=lr,
            total_epochs=total_epochs,
            last_model=last_model_path  # pass previous model to next cycle
        )
        command_list.extend(commands)

    # Write to txt (Linux) and bat (Windows)
    with open(txt_file, 'w') as f:
        for cmd in command_list:
            if isinstance(cmd, str):
                f.write(cmd + '\n')

    with open(bat_file, 'w') as f:
        f.write('@echo off\n')
        for cmd in command_list:
            if isinstance(cmd, str):
                f.write(cmd + '\n')

    print(f"Run commands in Windows: {bat_file}")
    print(f"Run commands in Linux: {txt_file}")
    return command_list


if __name__ == '__main__':
    commands = generate_psd_commands(
        total_cycles=2,
        total_epochs=100,
        dataset='Shanghai-center-train',
        batch_size=8,
        length=5,
        lr=0.01
    )