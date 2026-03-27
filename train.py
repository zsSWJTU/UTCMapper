"""
Train UTCMapper

This script initializes datasets, networks, and training configurations
for the UTCMapper project.
"""

import argparse
import datetime
import os
import random
import numpy as np
import torch
import torch.backends.cudnn as cudnn

from networks.MESNet import MESNet
from trainer import trainer_dataset
from utils import dataset_config, load_pretrained_exclude_head

# =====================================
# 1. Argument parsing
# =====================================
parser = argparse.ArgumentParser(description="Train MESNet for UTC mapping")

parser.add_argument('--dataset', type=str, default='Shanghai-0.3m-center-merge-sm-train',
                    help='Name of the dataset / experiment')
parser.add_argument('--max_epochs', type=int, default=100,
                    help='Maximum number of training epochs')
parser.add_argument('--batch_size', type=int, default=16,
                    help='Batch size per GPU')
parser.add_argument('--base_lr', type=float, default=0.01,
                    help='Learning rate for segmentation network')
parser.add_argument('--work_dir', type=str, default=None,
                    help='Directory to save checkpoints and logs')
parser.add_argument('--list_dir', type=str, default=None,
                    help='Directory containing CSV file lists for dataset')
parser.add_argument('--length', type=int, default=5,
                    help='Number of MESNet blocks')
parser.add_argument('--last_model', type=str, default=None,
                    help='Path of the trained model weight in last cycle')
parser.add_argument('--gpu', type=str, default='0',
                    help='GPU index to use')
parser.add_argument('--seed', type=int, default=1234,
                    help='Random seed for reproducibility')


args = parser.parse_args()

# =====================================
# 2. GPU setup and seeds
# =====================================
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
cudnn.benchmark = True
cudnn.deterministic = False
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)

# =====================================
# 3. Dataset configuration
# =====================================
dataset_name = args.dataset
if dataset_name not in dataset_config:
    raise ValueError(f"Dataset {dataset_name} not found in dataset_config")

args.num_classes = dataset_config[dataset_name]['num_classes']
if args.list_dir is None:
    args.list_dir = dataset_config[dataset_name]['list_dir']

# =====================================
# 4. Checkpoint / snapshot path
# =====================================
snapshot_path = args.work_dir if args.work_dir else os.path.join(
    'work_dirs',
    f"{args.dataset}_MESNet_{datetime.datetime.now().strftime('%Y-%m-%d')}"
)
os.makedirs(snapshot_path, exist_ok=True)

# =====================================
# 5. Model initialization
# =====================================
net = MESNet(num_blocks=args.length).cuda()

if args.last_model is not None:
    net = load_pretrained_exclude_head(net, args.last_model)

# =====================================
# 6. Start training
# =====================================
trainer_dataset(args, net, snapshot_path)