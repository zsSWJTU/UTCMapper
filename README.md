# UTCMapper for 0.3 m urban tree canopy mapping: a case study in Shanghai  

## 1️⃣ Data Preparation

| Step | Description | Path Example |
|------|------------|--------------|
| 1 | Place VFR images | `dataset/Shanghai-center-train/image_tiles` |
| 2 | Place coarse labels | `dataset/Shanghai-center-train/label_tiles` |
| 3 | Notes | - The **UBGG-3 m** coarse labels can be downloaded [here](https://drive.google.com/file/d/12JRQE4NMvyj1_Wr_Lln9KSIYSeZoxLpZ/view?usp=drive_link).<br>- VFR images can be obtained via **Bing Maps**, **Google Earth**, or **ESRI World Imagery** APIs or platforms. |

> 💡 Tip: Keep the folder structure consistent with the example above to avoid path errors during registration and training.
>
## 2️⃣ Dataset Registration
### a. Generate CSV List

```bash
python generate_dataset_csv.py \
    --image_folder 'dataset/Shanghai-center-train/image_tiles' \
    --label_folder 'dataset/Shanghai-center-train/label_tiles' \ #optional
    --new_file_path 'dataset/CSV_list/Shanghai-center-train.csv'
```
### b. Register Dataset in utils.dataset_config

```python
'Shanghai-train': {  # default dataset example
    'list_dir': 'dataset/CSV_list/Shanghai-center-train',  # path to *.csv
    'image_dir': 'dataset/Shanghai-center-train/image_tiles',
    'num_classes': 2
}
```
---
## **3️⃣ Training**

### a. Generate PSD Training Commands

```bash
python generate_PSD_commands.py
```
Optional: specify cycles and epochs.
### b. Run Training
Execute the generated command file, e.g.:
```bash
commands/Shanghai-center-train_PSD_2026-03-26.bat
```
---
## **4️⃣ Model Testing**
| Step | Command Example |
|------|----------------|
| 1 | Prepare test dataset (same as training steps 1–2) |
| 2 | Run test script ( ```python test.py --dataset {your_dataset_name} --model_path {trained_model.pth} --save_path {prediction_output_folder} ```) |
## 🔔 Quick Reference
| Task | Command / Path |
|------|----------------|
| Generate CSV list | `python generate_dataset_csv.py ...` |
| Generate PSD commands | `python generate_PSD_commands.py` |
| Run training | `commands/{dataset}_PSD_{date}.bat` |
| Test model | `python test.py --dataset ... --model_path ... --save_path ...` |

## 🔗 Download the Produced Shanghai 0.3 m UTC Map

You can directly download the produced urban tree canopy (UTC) map for Shanghai (0.3 m resolution) here:

[Download Shanghai 0.3 m UTC Map](https://zenodo.org/records/19445966)

## 📄 Citation

If you use this work, please cite:

```bibtex
@article{zhang2026utcmapper,
  title={UTCMapper for 0.3 m urban tree canopy mapping: A case study in Shanghai},
  author={Zhang, Shuang and Wang, Qian and Yang, Qian and Tong, Xiaohua and Atkinson, Peter M.},
  journal={Remote Sensing of Environment},
  volume={340},
  pages={115435},
  year={2026},
  doi={10.1016/j.rse.2026.115435}
}

