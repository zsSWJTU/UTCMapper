# UTCMapper for 0.3 m Urban Tree Canopy Mapping

![Python](https://img.shields.io/badge/Python-3.9%2B-3776AB)
![PyTorch](https://img.shields.io/badge/PyTorch-Training-EE4C2C)
![GeoTIFF](https://img.shields.io/badge/GeoTIFF-Inference-2E7D32)
![License](https://img.shields.io/badge/License-MIT-lightgrey)
![Last Updated](https://img.shields.io/badge/Updated-2026--06--28-blue)

UTCMapper is a deep-learning workflow for very-fine-resolution urban tree canopy
(UTC) mapping based on coarse labels. This release uses the
accelerated `MESNet_UltraFast` implementation for training and inference.

```text
Very-high-resolution imagery
          |
          v
  CSV dataset list  --->  cached geospatial chips
          |                         |
          v                         v
   MESNet_UltraFast training  --->  final.pth
          |
          v
  tile-wise GeoTIFF inference  --->  0.3 m UTC map
```

## What's New in This Release

| Area | Update |
|---|---|
| Speed | Training is several times faster through `MESNet_UltraFast`, cached geospatial chips, mixed precision, optional channels-last memory format, optional `torch.compile`, and gradient checkpointing. |
| Network | The active training and inference path uses `networks/MESNet_UltraFast.py`; `networks/MESNet.py` is kept as a legacy reference. |
| Labels | Single-band binary setup: background = `0`, foreground UTC = `1`, ignored pixels = `255`; raw values in `label.positive_values` are mapped to foreground. |
| Paths | Dataset paths are no longer hard-coded in `utils.py` or the PSD command generator. Provide CSV paths and output folders through YAML or CLI arguments. |

> Speed note: the core MESNet idea is preserved, but the release pipeline uses
> a more efficient implementation for practical large-area training.

## Repository Layout

| File | Role |
|---|---|
| `configs/default.yaml` | Default training configuration |
| `train.py` | Training entry point |
| `trainer.py` | Training loop and data pipeline |
| `test.py` | Tile-wise inference entry point |
| `cached_dataset.py` | Memmap-backed geospatial chip cache |
| `generate_dataset_csv.py` | Build image/label CSV lists |
| `generate_PSD_commands.py` | Parameterized PSD command generator |
| `generate_psd_csv.py` | Validate predictions and build pseudo-only or original-union PSD labels |
| `utils.py` | Shared utilities and dataset registry |
| `networks/MESNet_UltraFast.py` | Active accelerated MESNet implementation |
| `networks/MESNet.py` | Legacy/reference MESNet implementation |
| `networks/loss.py` | CCG loss with Dice consistent loss |

`networks/MESNet.py` is kept as a reference implementation, but the provided
training and inference scripts use `networks/MESNet_UltraFast.py`.

## Installation

Create a Python environment, then install the required packages:

```bash
pip install -r requirements.txt
```

Install the PyTorch build that matches your CUDA version if you plan to train
on GPU.

## Data Preparation

```text
Training CSV:  image_fn + label_fn
Testing CSV :  image_fn
```

Prepare two CSV files:

- `Shanghai-center-train`: training images and labels
- `Shanghai-center-test`: test images

Each training CSV should contain:

```text
image_fn,label_fn
path/to/image_001.tif,path/to/label_001.tif
```

Each inference-only CSV should contain:

```text
image_fn
path/to/image_001.tif
```

You can generate a CSV with:

```bash
python generate_dataset_csv.py \
  --image_folder /path/to/image_tiles \
  --label_folder /path/to/label_tiles \
  --new_file_path /path/to/Shanghai-center-train.csv
```

For inference-only CSV generation, omit `--label_folder` or pass an empty value.

## Configuration

Start from:

```text
configs/default.yaml
```

At minimum, set:

```yaml
dataset: Shanghai-center-train
list_dir: /path/to/Shanghai-center-train.csv
work_dir: /path/to/output_work_dir
```

Important current-label settings:

```yaml
label:
  positive_values: [2]
  ignore_value: 255
  ignore_ratio_threshold: 1.0
```

`positive_values` are values in the raw label raster that will be mapped to the
foreground UTC class during training.

## Training

```text
config + training CSV
        |
        v
cached chips + MESNet_UltraFast
        |
        v
final.pth
```

Run:

```bash
python train.py \
  --config configs/default.yaml \
  --list_dir /path/to/Shanghai-center-train.csv \
  --work_dir /path/to/output_work_dir
```

The final model is saved as:

```text
/path/to/output_work_dir/final.pth
```

## Inference

```text
final.pth + testing CSV
        |
        v
sliding-window inference
        |
        v
GeoTIFF UTC predictions
```

Run:

```bash
python test.py \
  --config configs/default.yaml \
  --model_path /path/to/output_work_dir/final.pth \
  --list_dir /path/to/Shanghai-center-test.csv \
  --save_path /path/to/prediction_output
```

Predictions are written as GeoTIFF files. If multiple city names are detected
from filenames, outputs can be separated into city subfolders.

## Progressive Self-Distillation Commands

Import it and pass all paths explicitly:

```python
from generate_PSD_commands import generate_psd_commands

generate_psd_commands(
    total_cycles=2,
    dataset="Shanghai-center-train",
    config_path="/path/to/configs/default.yaml",
    train_list_path="/path/to/Shanghai-center-train.csv",
    test_list_path="/path/to/Shanghai-center-test.csv",
    image_folder="/path/to/train_image_tiles",
    work_root="/path/to/work_dirs",
    prediction_root="/path/to/predictions",
    command_root="/path/to/commands",
    label_mode="union_original",
    original_positive_values=[2],
)
```

This writes both `.txt` and `.bat` command files.

### CCG error-pixel selection

CCG now selects false negatives (FN) and false positives (FP) independently
inside their own confidence distributions. Configure the retained confidence
quantiles under `loss`:

```yaml
loss:
  alpha: 1.0
  fn_quantile: 0.5
  fp_quantile: 0.5
```

The quantiles are in `[0, 1]`. Lower values send more of that error type to
BootLoss; higher values retain only the most confident errors. Use a lower
`fp_quantile` when foreground is often missing from the annotation: a model
prediction over missing foreground appears as an FP. Use a lower
`fn_quantile` when foreground is often annotated over true background. Keeping
the two thresholds separate is especially useful when one annotation-error
type is much more common than the other.

This selection is binary-only because FN and FP are defined here for
background `0` and UTC foreground `1`.

### PSD label modes

`generate_psd_commands` supports two label modes:

- `pseudo_only` (default) uses the model prediction as the next-cycle label
  and preserves the original UTCMapper PSD behavior.
- `union_original` uses `prediction foreground OR original foreground`. It is
  appropriate when original annotations are incomplete but their existing
  foreground pixels are trusted and should not be forgotten by later cycles.

For `union_original`, `original_positive_values` must describe foreground in
the original raster (for the public configuration, `[2]`). Generated PSD
labels contain only `0/1`, so the command generator automatically adds
`--positive_values 1` to every training cycle after the first.

The union utility validates that predictions are binary and that pseudo and
original rasters have matching shape, transform, and CRS. Do not use union mode
when original foreground contains many false positives: union cannot remove
those errors and may propagate them through every PSD cycle.

## Produced Shanghai UTC Map

The produced Shanghai 0.3 m UTC map can be downloaded here:

[Download Shanghai 0.3 m UTC Map](https://zenodo.org/records/19445966)

## Citation

If you use this work, please cite:

```bibtex
@article{zhang2026utcmapper,
  title   = {{UTCMapper} for 0.3 m urban tree canopy mapping: A case study in Shanghai},
  author  = {Zhang, Shuang and Wang, Qunming and Yang, Qiquan and Tong, Xiaohua and Atkinson, Peter M.},
  journal = {Remote Sensing of Environment},
  volume  = {340},
  pages   = {115435},
  year    = {2026},
  doi     = {10.1016/j.rse.2026.115435}
}
```
