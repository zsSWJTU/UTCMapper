import argparse
import os
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument('--image_folder', type=str, required=True, help="Path to the folder containing images")
parser.add_argument('--label_folder', type=str, default='', help="Path to the folder containing labels (optional, can be None)")
parser.add_argument('--new_file_path', type=str, required=True, help="Path where the output CSV file will be saved")
args = parser.parse_args()


def generate_dataset_csv(fr_image_folder: str, cr_label_folder: str, new_file_path: str):
    # Handle relative paths if needed
    if fr_image_folder.split('/')[0] == 'dataset':
        fr_image_folder = './' + fr_image_folder

    # Get fine-resolution image files
    hr_image_files = sorted([f for f in os.listdir(fr_image_folder) if f.endswith('.tif')])
    image_paths = [os.path.join(fr_image_folder, f) for f in hr_image_files]

    data = {'image_fn': []}
    if cr_label_folder:
        data['label_fn'] = []

    # Only process labels if label folder exists and is not empty
    if cr_label_folder and os.path.exists(cr_label_folder) and os.path.isdir(cr_label_folder):
        if cr_label_folder.split('/')[0] == 'dataset':
            cr_label_folder = './' + cr_label_folder

        lr_label_files = sorted([f for f in os.listdir(cr_label_folder) if f.endswith('.tif')])
        label_paths = {f: os.path.join(cr_label_folder, f) for f in lr_label_files}

        # Match images with labels by filename
        matched_image_paths = []
        matched_label_paths = []

        for img_file, img_path in zip(hr_image_files, image_paths):

            if 'label_'+img_file in label_paths.keys() :
                matched_image_paths.append(img_path)
                matched_label_paths.append(label_paths['label_'+img_file])
            elif img_file in label_paths.keys():
                matched_image_paths.append(img_path)
                matched_label_paths.append(label_paths[img_file])


        data['image_fn'] = matched_image_paths
        if matched_label_paths:
            data['label_fn'] = matched_label_paths

        unmatched_images = len(hr_image_files) - len(matched_image_paths)
        unmatched_labels = len(lr_label_files) - len(matched_label_paths)
        if unmatched_images > 0 or unmatched_labels > 0:
            print(f"Warning: Found {unmatched_images} images and {unmatched_labels} labels without pairs. "
                  f"Only {len(matched_image_paths)} paired samples are saved.")

    else:
        # No label folder: only save images
        data['image_fn'] = image_paths
        print("No label folder provided. Only image paths will be saved.")

    # Create DataFrame and save to CSV
    df = pd.DataFrame(data)
    df.to_csv(new_file_path, index=False)
    print(f"CSV file saved to: {new_file_path} with {len(df)} paired records.")


if __name__ == '__main__':
    generate_dataset_csv(args.image_folder, args.label_folder, args.new_file_path)
    #example:
    # python generate_dataset_csv.py --image_folder 'dataset/Shanghai_tiny/HR_image/' --label_folder 'dataset/Shanghai_tiny/LR_label/' --new_file_path 'dataset/CSV_list/Shanghai_tiny.csv'

