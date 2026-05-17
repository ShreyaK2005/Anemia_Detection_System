import os
import cv2
import numpy as np
import random
import argparse
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def augment_image(img, rotation_range=(-15, 15), zoom_scale=(1.0, 1.2), brightness_shift=(-30, 30)):
    """
    Apply augmentation techniques to an image:
    - Rotation
    - Horizontal flip
    - Brightness change
    - Slight zoom (crop + resize)
    Returns a list of augmented images.
    """
    h, w = img.shape[:2]
    augmented_images = []

    # 1. Rotation
    angle = random.randint(*rotation_range)
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1)
    rotated = cv2.warpAffine(img, M, (w, h))
    augmented_images.append(rotated)

    # 2. Horizontal Flip
    flipped = cv2.flip(img, 1)
    augmented_images.append(flipped)

    # 3. Brightness Change
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    v = hsv[:, :, 2].astype(np.int16)
    v += random.randint(*brightness_shift)
    v = np.clip(v, 0, 255)
    hsv[:, :, 2] = v.astype(np.uint8)
    bright = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    augmented_images.append(bright)

    # 4. Slight Zoom (Crop + Resize)
    crop_fraction = 0.1  # 10% crop
    y1, y2 = int(crop_fraction * h), int((1 - crop_fraction) * h)
    x1, x2 = int(crop_fraction * w), int((1 - crop_fraction) * w)
    crop = img[y1:y2, x1:x2]
    zoom = cv2.resize(crop, (w, h))
    augmented_images.append(zoom)

    return augmented_images

def process_dataset(dataset_path, classes):
    """
    Process the dataset directory:
    - For each class folder, augment images.
    - Save augmented images with _aug suffix.
    """
    for cls in classes:
        folder = os.path.join(dataset_path, cls)
        if not os.path.exists(folder):
            logging.warning(f"Folder not found: {folder}")
            continue

        for img_name in os.listdir(folder):
            # Skip already augmented images
            if "_aug" in img_name:
                continue
            # Skip non-image files
            if not img_name.lower().endswith((".jpg", ".jpeg", ".png")):
                continue

            img_path = os.path.join(folder, img_name)
            img = cv2.imread(img_path)
            if img is None:
                logging.warning(f"Failed to read image: {img_path}")
                continue

            name, ext = os.path.splitext(img_name)
            aug_images = augment_image(img)

            for i, aug in enumerate(aug_images, start=1):
                new_name = f"{name}_aug{i}{ext}"
                save_path = os.path.join(folder, new_name)
                cv2.imwrite(save_path, aug)
        logging.info(f"Augmentation completed for class '{cls}' in folder '{folder}'.")

def main():
    parser = argparse.ArgumentParser(description="Augment training images for dataset expansion.")
    parser.add_argument("--dataset_path", type=str, required=True, help="Path to the dataset root directory.")
    parser.add_argument("--classes", nargs='+', default=["anemia", "normal"], help="List of class folder names.")
    args = parser.parse_args()

    process_dataset(args.dataset_path, args.classes)
    logging.info("Dataset augmentation completed successfully.")

if __name__ == "__main__":
    main()
