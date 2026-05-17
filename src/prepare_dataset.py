import os
import shutil
import pandas as pd
import cv2
import argparse
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_excel(file_path):
    return pd.read_excel(file_path)

def create_dirs(path, dirs):
    for d in dirs:
        os.makedirs(os.path.join(path, d), exist_ok=True)

def copy_palpebral_images(df, original_path, new_dataset_path):
    anemia_counter = 1
    normal_counter = 1

    for _, row in df.iterrows():
        folder_number = str(int(row["Number"]))
        hgb = row["Hgb"]
        gender = row["Gender"]
        label = "anemia" if (gender == "F" and hgb < 12) or (gender == "M" and hgb < 13) else "normal"

        for country in ["India", "Italy"]:
            patient_folder = os.path.join(original_path, country, folder_number)
            if not os.path.exists(patient_folder):
                continue

            for img in os.listdir(patient_folder):
                if "palpebral" not in img.lower():
                    continue
                src = os.path.join(patient_folder, img)
                ext = os.path.splitext(img)[1]
                if label == "anemia":
                    new_name = f"kaggle_anemia_{anemia_counter}{ext}"
                    anemia_counter += 1
                else:
                    new_name = f"kaggle_normal_{normal_counter}{ext}"
                    normal_counter += 1
                dst = os.path.join(new_dataset_path, label, new_name)
                shutil.copy(src, dst)
                break

def merge_datasets(datasets, combined_path, classes):
    for dataset_path, prefix in datasets:
        for cls in classes:
            source_folder = os.path.join(dataset_path, cls)
            dest_folder = os.path.join(combined_path, cls)
            os.makedirs(dest_folder, exist_ok=True)
            if not os.path.exists(source_folder):
                continue
            for img in os.listdir(source_folder):
                if not img.lower().endswith(('.jpg', '.jpeg', '.png')):
                    continue
                src = os.path.join(source_folder, img)
                ext = os.path.splitext(img)[1]
                new_name = f"{prefix}_{cls}_{os.path.splitext(img)[0]}{ext}"
                dst = os.path.join(dest_folder, new_name)
                shutil.copy(src, dst)

def augment_image(img):
    aug_images = []

    # Rotation
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w/2, h/2), 10, 1)
    rotated = cv2.warpAffine(img, M, (w, h))
    aug_images.append(rotated)

    # Horizontal flip
    flipped = cv2.flip(img, 1)
    aug_images.append(flipped)

    # Brightness adjustment
    bright = cv2.convertScaleAbs(img, alpha=1.1, beta=10)
    aug_images.append(bright)

    # Slight zoom
    crop = img[int(0.05*h):int(0.95*h), int(0.05*w):int(0.95*w)]
    zoom = cv2.resize(crop, (w, h))
    aug_images.append(zoom)

    return aug_images

def process_and_augment(dataset_path, output_path, prefix, counters):
    classes = ["anemia", "normal"]
    for cls in classes:
        folder = os.path.join(dataset_path, cls)
        for img_name in os.listdir(folder):
            if not img_name.lower().endswith(('.jpg', '.jpeg', '.png')):
                continue
            img_path = os.path.join(folder, img_name)
            img = cv2.imread(img_path)
            if img is None:
                continue
            augmented = augment_image(img)
            for aug in augmented:
                name_idx = counters[cls]
                new_name = f"{prefix}_aug_{cls}_{name_idx}.jpg"
                save_path = os.path.join(output_path, cls, new_name)
                cv2.imwrite(save_path, aug)
                counters[cls] += 1

def combine_datasets(source_dirs, combined_dir, classes):
    os.makedirs(combined_dir, exist_ok=True)
    for cls in classes:
        os.makedirs(os.path.join(combined_dir, cls), exist_ok=True)

    def copy_from_source(source):
        for cls in classes:
            src_folder = os.path.join(source, cls)
            dst_folder = os.path.join(combined_dir, cls)
            if not os.path.exists(src_folder):
                continue
            for img in os.listdir(src_folder):
                src_path = os.path.join(src_folder, img)
                dst_path = os.path.join(dst_folder, img)
                if os.path.isfile(src_path):
                    shutil.copy(src_path, dst_path)

    for src in source_dirs:
        copy_from_source(src)

def add_kaggle2_images(source_path, combined_path, classes):
    counters = {cls: 1 for cls in classes}
    for cls in classes:
        src_folder = os.path.join(source_path, cls)
        dst_folder = os.path.join(combined_path, cls)
        if not os.path.exists(src_folder):
            continue
        for img in os.listdir(src_folder):
            if not img.lower().endswith(('.jpg', '.jpeg', '.png')):
                continue
            src = os.path.join(src_folder, img)
            ext = os.path.splitext(img)[1]
            new_name = f"kaggle2_{cls}_{counters[cls]}{ext}"
            dst = os.path.join(dst_folder, new_name)
            shutil.copy(src, dst)
            counters[cls] += 1

def main():
    parser = argparse.ArgumentParser(description="Prepare dataset: organize, augment, merge.")
    parser.add_argument("--excel_path", default="India.xlsx")
    parser.add_argument("--original_path", default="dataset anemia")
    parser.add_argument("--new_dataset_path", default="processed1_anemia_dataset")
    parser.add_argument("--roboflow_path", default="Roboflow anemia detection.folder")
    parser.add_argument("--dataset2_path", default="Kaggle dataset 2")
    parser.add_argument("--combined_path", default="combined_images_dataset")
    parser.add_argument("--augmented_path", default="augmented_images_dataset")
    args = parser.parse_args()

    # Step 1: Load data and create folders
    df = load_excel(args.excel_path)
    create_dirs(args.new_dataset_path, ["anemia", "normal"])
    copy_palpebral_images(df, args.original_path, args.new_dataset_path)

    # Step 2: Merge datasets
    merge_datasets([
        (args.roboflow_path, "roboflow"),
        (args.new_dataset_path, "kaggle")
    ], args.combined_path, ["anemia", "normal"])

    # Step 3: Augmentation
    counters = {"anemia": 1, "normal": 1}
    process_and_augment(args.combined_path, args.augmented_path, "master", counters)

    # Step 4: Add Kaggle Dataset 2
    add_kaggle2_images(args.dataset2_path, args.combined_path, ["anemia", "normal"])

    logging.info("All steps completed successfully.")

if __name__ == "__main__":
    main()
