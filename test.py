"""
Inference script for UTCMapper

This script performs tile-wise inference on input raster datasets using MESNet,
including dataset preparation, tiling, and saving predicted outputs.

Referenced from:
L2HNet and Paraformer: https://github.com/LiZhuoHong/Paraformer
"""

import argparse
import os
import random
import numpy as np
import pandas as pd
import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import Dataset, DataLoader
import rasterio
from rasterio.errors import RasterioIOError
from tqdm import tqdm

import utils
from networks.MESNet import MESNet

# ----------------------------
# Utility functions
# ----------------------------
def image_transforms(img: np.ndarray) -> torch.Tensor:
    """
    Normalize and convert HWC image to CHW tensor
    """
    img = (img - utils.IMAGE_MEANS) / utils.IMAGE_STDS
    img = np.rollaxis(img, 2, 0).astype(np.float32)
    return torch.from_numpy(img)


# ----------------------------
# Dataset
# ----------------------------
class TileInferenceDataset(Dataset):
    def __init__(self, fn, chip_size, stride, transform=None, windowed_sampling=False, verbose=False):
        self.fn = fn
        self.chip_size = chip_size
        self.transform = transform
        self.windowed_sampling = windowed_sampling
        self.verbose = verbose

        with rasterio.open(self.fn) as f:
            self.height, self.width = f.height, f.width
            self.num_channels = f.count
            self.dtype = f.profile["dtype"]
            if not windowed_sampling:
                self.data = np.rollaxis(f.read(), 0, 3)  # HWC

        # Generate chip coordinates
        self.chip_coordinates = [
            (y, x)
            for y in list(range(0, self.height - self.chip_size, stride)) + [self.height - self.chip_size]
            for x in list(range(0, self.width - self.chip_size, stride)) + [self.width - self.chip_size]
        ]
        self.num_chips = len(self.chip_coordinates)

        if self.verbose:
            print(f"Constructed TileInferenceDataset -- {self.height}x{self.width} file with "
                  f"{self.num_channels} channels ({self.dtype}). Total chips: {self.num_chips}")

    def __getitem__(self, idx):
        y, x = self.chip_coordinates[idx]

        if self.windowed_sampling:
            try:
                with rasterio.open(self.fn) as f:
                    img = np.rollaxis(
                        f.read(window=rasterio.windows.Window(x, y, self.chip_size, self.chip_size)), 0, 3
                    )
            except RasterioIOError:
                print(f"Reading chip {idx} failed, returning zeros")
                img = np.zeros((self.chip_size, self.chip_size, self.num_channels), dtype=np.uint8)
        else:
            img = self.data[y:y + self.chip_size, x:x + self.chip_size, :3]

        if self.transform is not None:
            img = self.transform(img)

        return img, np.array((y, x))

    def __len__(self):
        return self.num_chips


# ----------------------------
# Argument parser
# ----------------------------
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='', help='dataset_name')
parser.add_argument('--batch_size', type=int, default=32, help='batch size per GPU')
parser.add_argument('--chip_size', type=int, default=224, help='network input patch size')
parser.add_argument('--save_path', type=str, default='')
parser.add_argument('--model_path', type=str, default='')
parser.add_argument('--gpu', type=str, default='0', help='GPU number')
parser.add_argument('--length', type=int, default=5, help='number of MESNet blocks')
parser.add_argument('--seed', type=int, default=1234, help='random seed')
args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

CHIP_SIZE = args.chip_size
PADDING = CHIP_SIZE // 2
assert PADDING % 2 == 0
HALF_PADDING = PADDING // 2
CHIP_STRIDE = CHIP_SIZE - PADDING


# ----------------------------
# Inference function
# ----------------------------
def inference(args, model, test_save_path: str = None):
    if test_save_path and not os.path.exists(test_save_path):
        os.makedirs(test_save_path)
        print(f"Folder '{test_save_path}' created.")

    model.eval()
    input_dataframe = pd.read_csv(args.list_dir)
    image_fns = input_dataframe["image_fn"].values

    for idx, image_fn in enumerate(tqdm(image_fns, desc="Images", unit="image")):
        output_fn = os.path.basename(image_fn).replace("tile", "tile_predictions")
        output_fn = os.path.join(test_save_path, output_fn)

        if os.path.exists(output_fn):
            print(f'{output_fn} already exists, skipping...')
            continue

        print(f"({idx + 1}/{len(image_fns)}) Processing {image_fn} ... ", end='')

        # Load dataset
        try:
            with rasterio.open(image_fn) as f:
                input_width, input_height = f.width, f.height
                input_profile = f.profile.copy()
                dataset = TileInferenceDataset(
                    image_fn, chip_size=CHIP_SIZE, stride=CHIP_STRIDE, transform=image_transforms
                )
        except Exception as e:
            print(e)
            continue

        dataloader = DataLoader(dataset, batch_size=args.batch_size, num_workers=0, pin_memory=True)

        # Prepare output buffer
        output = np.zeros((args.num_classes, input_height, input_width), dtype=np.float32)
        kernel = np.ones((CHIP_SIZE, CHIP_SIZE), dtype=np.float32)
        kernel[HALF_PADDING:-HALF_PADDING, HALF_PADDING:-HALF_PADDING] = 5
        counts = np.zeros((input_height, input_width), dtype=np.float32)

        # Run inference
        for data, coords in dataloader:
            data = data.cuda()
            with torch.no_grad():
                t_output = torch.sigmoid(model(data)).cpu().numpy()

            for j in range(t_output.shape[0]):
                y, x = coords[j]
                output[:, y:y + CHIP_SIZE, x:x + CHIP_SIZE] += t_output[j] * kernel
                counts[y:y + CHIP_SIZE, x:x + CHIP_SIZE] += kernel

        output /= counts
        output_hard = output.argmax(axis=0).astype(np.uint8)

        # Save output
        output_profile = input_profile.copy()
        output_profile.update(driver="GTiff", dtype="uint8", count=1, nodata=0)
        with rasterio.open(output_fn, "w", **output_profile) as f:
            f.write(output_hard, 1)
            f.write_colormap(1, utils.LABEL_IDX_COLORMAP)

    return "Testing Finished!"


# ----------------------------
# Main
# ----------------------------
if __name__ == "__main__":
    cudnn.benchmark = True
    cudnn.deterministic = False
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    # Dataset config
    print('Available datasets:', list(utils.dataset_config.keys()))
    args.num_classes = utils.dataset_config[args.dataset]['num_classes']
    args.list_dir = utils.dataset_config[args.dataset]['list_dir']

    # MESNet initialization
    net = MESNet(num_blocks=args.length).cuda()
    snapshot = args.model_path
    net.load_state_dict(torch.load(snapshot))
    task_name = '_'.join(os.path.basename(os.path.dirname(snapshot)).split('_')[1:])
    test_save_path = args.save_path or f'dataset/{args.dataset}/Prediction_{task_name}'
    os.makedirs(test_save_path, exist_ok=True)
    print('Results will be saved to:', test_save_path)

    inference(args, net, test_save_path)