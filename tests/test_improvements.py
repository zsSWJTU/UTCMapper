import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
import torch
from rasterio.transform import from_origin

from generate_psd_csv import build_psd_csv
from networks.loss import CCGLoss


class CCGSelectionTest(unittest.TestCase):
    def test_fn_and_fp_use_independent_quantiles(self):
        outputs = torch.tensor([[[[2.0, 4.0, 0.0, 0.0]], [[0.0, 0.0, 3.0, 5.0]]]])
        target = torch.tensor([[[1, 1, 0, 0]]])

        all_errors = CCGLoss(fn_quantile=0, fp_quantile=0)
        top_errors = CCGLoss(fn_quantile=1, fp_quantile=1)

        self.assertEqual(all_errors.find_inconsistent_mask(outputs, target).sum(), 4)
        self.assertEqual(top_errors.find_inconsistent_mask(outputs, target).sum(), 2)


class PSDUnionTest(unittest.TestCase):
    def test_union_preserves_original_and_predicted_foreground(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            prediction_root = root / "predictions"
            prediction_root.mkdir()
            profile = {
                "driver": "GTiff",
                "height": 2,
                "width": 2,
                "count": 1,
                "dtype": "uint8",
                "crs": "EPSG:4326",
                "transform": from_origin(0, 2, 1, 1),
            }

            with rasterio.open(prediction_root / "tile.tif", "w", **profile) as output:
                output.write(np.array([[1, 0], [0, 0]], dtype=np.uint8), 1)
            original_path = root / "original.tif"
            with rasterio.open(original_path, "w", **profile) as output:
                output.write(np.array([[0, 2], [0, 0]], dtype=np.uint8), 1)

            input_list = root / "input.csv"
            pd.DataFrame(
                {"image_fn": ["tile.tif"], "label_fn": [str(original_path)]}
            ).to_csv(input_list, index=False)

            build_psd_csv(
                input_list=input_list,
                prediction_folder=prediction_root,
                output_list=root / "next.csv",
                label_mode="union_original",
                original_list=input_list,
                original_positive_values=[2],
                merged_label_folder=root / "merged",
            )

            with rasterio.open(root / "merged" / "tile.tif") as source:
                merged = source.read(1)
            np.testing.assert_array_equal(merged, [[1, 1], [0, 0]])


if __name__ == "__main__":
    unittest.main()
