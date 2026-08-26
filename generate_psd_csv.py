"""Build the next UTCMapper PSD training CSV.

``pseudo_only`` keeps the original UTCMapper PSD behavior. ``union_original``
writes binary labels whose foreground is the union of the current prediction
and the foreground encoded in the original annotation.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio


PSD_LABEL_MODES = ("pseudo_only", "union_original")


def _prediction_path(root, image_path):
    name = Path(image_path).name
    candidates = (root / name, root / f"label_{name}")
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        f"No PSD prediction for {image_path}; tried: "
        + ", ".join(str(path) for path in candidates)
    )


def _check_frame(frame, name):
    missing = {"image_fn", "label_fn"} - set(frame.columns)
    if missing:
        raise ValueError(f"{name} CSV is missing columns: {sorted(missing)}")


def _validate_binary(path):
    with rasterio.open(path) as source:
        for _, window in source.block_windows(1):
            values = set(np.unique(source.read(1, window=window)).tolist())
            invalid = values - {0, 1}
            if invalid:
                raise ValueError(
                    f"PSD prediction {path} contains non-binary values: "
                    f"{sorted(invalid)}"
                )


def _union_labels(pseudo_paths, original_paths, output_folder, positive_values):
    positives = np.asarray(list(positive_values))
    if positives.size == 0:
        raise ValueError("original_positive_values must not be empty.")

    output_root = Path(output_folder)
    output_root.mkdir(parents=True, exist_ok=True)
    results = []
    foreground_pixels = 0

    for pseudo_path, original_path in zip(pseudo_paths, original_paths):
        destination = output_root / pseudo_path.name
        with rasterio.open(pseudo_path) as pseudo, rasterio.open(original_path) as original:
            if pseudo.shape != original.shape:
                raise ValueError(
                    f"Shape mismatch for {pseudo_path.name}: "
                    f"{pseudo.shape} != {original.shape}"
                )
            if pseudo.transform != original.transform or pseudo.crs != original.crs:
                raise ValueError(
                    f"Georeferencing mismatch for {pseudo_path.name}."
                )

            profile = pseudo.profile.copy()
            profile.update(dtype="uint8", count=1, nodata=None, compress="lzw")
            with rasterio.open(destination, "w", **profile) as output:
                for _, window in pseudo.block_windows(1):
                    predicted = pseudo.read(1, window=window) == 1
                    annotated = np.isin(original.read(1, window=window), positives)
                    merged = np.logical_or(predicted, annotated).astype(np.uint8)
                    foreground_pixels += int(merged.sum())
                    output.write(merged, 1, window=window)
        results.append(destination)

    if foreground_pixels == 0:
        raise ValueError("Union labels contain no foreground pixels.")
    return results


def build_psd_csv(
    input_list,
    prediction_folder,
    output_list,
    label_mode="pseudo_only",
    original_list=None,
    original_positive_values=None,
    merged_label_folder=None,
):
    label_mode = str(label_mode).lower()
    if label_mode not in PSD_LABEL_MODES:
        raise ValueError(
            f"Unsupported label_mode={label_mode!r}; choose from {PSD_LABEL_MODES}."
        )

    frame = pd.read_csv(input_list)
    _check_frame(frame, "Input")
    prediction_root = Path(prediction_folder)
    pseudo_paths = [
        _prediction_path(prediction_root, image_path)
        for image_path in frame["image_fn"].astype(str)
    ]
    for path in pseudo_paths:
        _validate_binary(path)

    output_paths = pseudo_paths
    if label_mode == "union_original":
        if not original_list or not merged_label_folder:
            raise ValueError(
                "union_original requires original_list and merged_label_folder."
            )
        original = pd.read_csv(original_list)
        _check_frame(original, "Original")
        if len(frame) != len(original):
            raise ValueError("Input and original CSV row counts differ.")
        current_images = frame["image_fn"].astype(str).map(lambda p: Path(p).as_posix())
        original_images = original["image_fn"].astype(str).map(lambda p: Path(p).as_posix())
        if not current_images.equals(original_images):
            raise ValueError("Input and original CSV image paths/order differ.")
        original_paths = [Path(path) for path in original["label_fn"].astype(str)]
        missing = [str(path) for path in original_paths if not path.is_file()]
        if missing:
            raise FileNotFoundError(f"Original labels are missing: {missing[:10]}")
        output_paths = _union_labels(
            pseudo_paths,
            original_paths,
            merged_label_folder,
            original_positive_values or [],
        )

    result = frame.copy()
    result["label_fn"] = [path.as_posix() for path in output_paths]
    destination = Path(output_list)
    destination.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(destination, index=False)
    print(f"PSD CSV written: {destination} ({len(result)} rows, mode={label_mode})")
    return destination.as_posix()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input_list", required=True)
    parser.add_argument("--prediction_folder", required=True)
    parser.add_argument("--output_list", required=True)
    parser.add_argument("--label_mode", choices=PSD_LABEL_MODES, default="pseudo_only")
    parser.add_argument("--original_list")
    parser.add_argument("--original_positive_values", nargs="+", type=int)
    parser.add_argument("--merged_label_folder")
    args = parser.parse_args()
    build_psd_csv(**vars(args))


if __name__ == "__main__":
    main()
