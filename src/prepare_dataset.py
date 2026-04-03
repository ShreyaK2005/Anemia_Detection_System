import os
import shutil
import pandas as pd
import cv2
import random
import numpy as np

# Load Excel
df = pd.read_excel("India.xlsx")

# Paths
original_path = "dataset anemia"
new_dataset_path = "processed1_anemia_dataset"

# Create new folders
os.makedirs(os.path.join(new_dataset_path, "anemia"), exist_ok=True)
os.makedirs(os.path.join(new_dataset_path, "normal"), exist_ok=True)

# Counters for renaming
anemia_counter = 1
normal_counter = 1

for index, row in df.iterrows():

    folder_number = str(int(row["Number"]))
    hgb = row["Hgb"]
    gender = row["Gender"]

    # WHO anemia rule
    if (gender == "F" and hgb < 12) or (gender == "M" and hgb < 13):
        label = "anemia"
    else:
        label = "normal"

    # Search inside India and Italy
    for country in ["India", "Italy"]:

        patient_folder = os.path.join(original_path, country, folder_number)

        if not os.path.exists(patient_folder):
            continue

        # Find palpebral image
        for img in os.listdir(patient_folder):

            if "palpebral" not in img.lower():
                continue

            src = os.path.join(patient_folder, img)

            # get extension
            ext = os.path.splitext(img)[1]

            if label == "anemia":

                new_name = f"kaggle_anemia_{anemia_counter}{ext}"
                anemia_counter += 1

            else:

                new_name = f"kaggle_normal_{normal_counter}{ext}"
                normal_counter += 1

            dst = os.path.join(new_dataset_path, label, new_name)

            shutil.copy(src, dst)

            # stop after copying the palpebral image
            break

print("Processed1 Kaggle dataset created successfully.")
# Original Roboflow dataset
roboflow_path = "Roboflow anemia detection.folder"

# New merged Roboflow dataset
output_path = "Roboflow_final_dataset"

# Create folders
os.makedirs(os.path.join(output_path, "anemia"), exist_ok=True)
os.makedirs(os.path.join(output_path, "normal"), exist_ok=True)

splits = ["train", "test", "valid"]
classes = ["anemia", "normal"]

anemia_counter = 1
normal_counter = 1

for split in splits:
    for cls in classes:

        source_folder = os.path.join(roboflow_path, split, cls)

        if not os.path.exists(source_folder):
            continue

        for img in os.listdir(source_folder):

            if not img.lower().endswith((".jpg", ".jpeg", ".png")):
                continue

            src = os.path.join(source_folder, img)

            ext = os.path.splitext(img)[1]

            if cls == "anemia":

                new_name = f"roboflow_anemia_{anemia_counter}{ext}"
                anemia_counter += 1

            else:

                new_name = f"roboflow_normal_{normal_counter}{ext}"
                normal_counter += 1

            dst = os.path.join(output_path, cls, new_name)

            shutil.copy(src, dst)

print("Roboflow dataset merged successfully.")


# IMAGE AUGMENTATION
# Source datasets
kaggle_path = "processed1_anemia_dataset"
roboflow_path = "Roboflow_final_dataset"

# Output dataset
output_path = "augmented_images_dataset"

classes = ["anemia", "normal"]

# Create output folders
for cls in classes:
    os.makedirs(os.path.join(output_path, cls), exist_ok=True)


def augment_image(img):

    aug_images = []

    # 1 Rotation
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w/2, h/2), 10, 1)
    rotated = cv2.warpAffine(img, M, (w, h))
    aug_images.append(rotated)

    # 2 Horizontal flip
    flipped = cv2.flip(img, 1)
    aug_images.append(flipped)

    # 3 Brightness adjustment
    bright = cv2.convertScaleAbs(img, alpha=1.1, beta=10)
    aug_images.append(bright)

    # 4 Slight zoom
    crop = img[int(0.05*h):int(0.95*h), int(0.05*w):int(0.95*w)]
    zoom = cv2.resize(crop, (w, h))
    aug_images.append(zoom)

    return aug_images


# Counters for naming
kaggle_counter = {"anemia":1, "normal":1}
roboflow_counter = {"anemia":1, "normal":1}


def process_dataset(dataset_path, prefix, counters):

    for cls in classes:

        folder = os.path.join(dataset_path, cls)

        for img_name in os.listdir(folder):

            if not img_name.lower().endswith((".jpg",".jpeg",".png")):
                continue

            img_path = os.path.join(folder, img_name)
            img = cv2.imread(img_path)

            if img is None:
                continue

            augmented = augment_image(img)

            for aug in augmented:

                num = counters[cls]

                new_name = f"{prefix}_aug_{cls}_{num}.jpg"

                save_path = os.path.join(output_path, cls, new_name)

                cv2.imwrite(save_path, aug)

                counters[cls] += 1


# Process Kaggle images
process_dataset(kaggle_path, "kaggle", kaggle_counter)

# Process Roboflow images
process_dataset(roboflow_path, "roboflow", roboflow_counter)

print("Augmentation completed successfully.")

#COMBINING IMAGES FROM ROBOFLOW AND PROCESSED1 BEFORE SPLITTING IN MASTER DATASET

# Source datasets
dataset1 = "Roboflow_final_dataset"
dataset2 = "processed1_anemia_dataset"

# Destination dataset
combined_dataset = "combined_images_dataset"

classes = ["anemia", "normal"]

# Create destination folders
for cls in classes:
    os.makedirs(os.path.join(combined_dataset, cls), exist_ok=True)

# Function to copy images
def copy_images(source_dataset):

    for cls in classes:

        source_folder = os.path.join(source_dataset, cls)
        dest_folder = os.path.join(combined_dataset, cls)

        if not os.path.exists(source_folder):
            continue

        for img in os.listdir(source_folder):

            src = os.path.join(source_folder, img)
            dst = os.path.join(dest_folder, img)

            if os.path.isfile(src):
                shutil.copy(src, dst)

# Copy images from both datasets
copy_images(dataset1)
copy_images(dataset2)

print("Datasets combined successfully.")



#ADDING KAGGLE DATASET 2 IMAGES TO COMBINED IMAGES DATASET


# Paths
dataset2_path = "Kaggle dataset 2"
combined_path = "combined_images_dataset"

classes = ["anemia", "normal"]

# Counters for renaming
counters = {
    "anemia": 1,
    "normal": 1
}

for cls in classes:

    source_folder = os.path.join(dataset2_path, cls)
    dest_folder = os.path.join(combined_path, cls)

    if not os.path.exists(source_folder):
        print(f"{source_folder} not found, skipping...")
        continue

    for img_name in os.listdir(source_folder):

        if not img_name.lower().endswith((".jpg", ".jpeg", ".png")):
            continue

        src = os.path.join(source_folder, img_name)

        ext = os.path.splitext(img_name)[1]

        # New name format
        new_name = f"kaggle2_{cls}_{counters[cls]}{ext}"

        dst = os.path.join(dest_folder, new_name)

        shutil.copy(src, dst)

        counters[cls] += 1

print("Kaggle Dataset 2 added and renamed successfully.")
