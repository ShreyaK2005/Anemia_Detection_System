import os
import cv2
import numpy as np
import random

dataset_path = "master_dataset/train"

classes = ["anemia", "normal"]

def augment_image(img):

    augmentations = []

    # rotation
    angle = random.randint(-15, 15)
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w//2, h//2), angle, 1)
    rotated = cv2.warpAffine(img, M, (w, h))

    # horizontal flip
    flipped = cv2.flip(img, 1)

    # brightness change
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    brightness = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    # slight zoom
    scale = random.uniform(1.0, 1.2)
    resized = cv2.resize(img, None, fx=scale, fy=scale)
    zh, zw = resized.shape[:2]
    zoom = resized[0:h, 0:w]

    augmentations.extend([rotated, flipped, brightness, zoom])

    return augmentations


for cls in classes:

    folder = os.path.join(dataset_path, cls)

    for img_name in os.listdir(folder):

        if "_aug" in img_name:
            continue

        img_path = os.path.join(folder, img_name)

        img = cv2.imread(img_path)

        if img is None:
            continue

        name, ext = os.path.splitext(img_name)

        aug_images = augment_image(img)

        for i, aug in enumerate(aug_images, start=1):

            new_name = f"{name}_aug{i}{ext}"

            save_path = os.path.join(folder, new_name)

            cv2.imwrite(save_path, aug)

print("Train augmentation completed.")






#AUGMENT TRAIN IMAGES IN MASTER DATASET 1

import os
import cv2
import numpy as np
import random

train_path = "master_dataset_1/train"

classes = ["anemia", "normal"]

def augment_image(img):

    h, w = img.shape[:2]

    augmented_images = []

    # 1. Rotation
    angle = random.randint(-12, 12)
    M = cv2.getRotationMatrix2D((w/2, h/2), angle, 1)
    rotated = cv2.warpAffine(img, M, (w, h))
    augmented_images.append(rotated)

    # 2. Horizontal Flip
    flipped = cv2.flip(img, 1)
    augmented_images.append(flipped)

    # 3. Brightness Change (FIXED)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Convert ONLY V channel to int16
    v = hsv[:, :, 2].astype(np.int16)

    v = v + random.randint(-30, 30)
    v = np.clip(v, 0, 255)

    hsv[:, :, 2] = v.astype(np.uint8)

    bright = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    augmented_images.append(bright)

    # 4. Slight Zoom
    crop = img[int(0.05*h):int(0.95*h), int(0.05*w):int(0.95*w)]
    zoom = cv2.resize(crop, (w, h))
    augmented_images.append(zoom)

    return augmented_images


for cls in classes:

    folder = os.path.join(train_path, cls)

    for img_name in os.listdir(folder):

        # Skip already augmented images
        if "_aug" in img_name:
            continue

        if not img_name.lower().endswith((".jpg",".jpeg",".png")):
            continue

        img_path = os.path.join(folder, img_name)

        img = cv2.imread(img_path)

        if img is None:
            continue

        name, ext = os.path.splitext(img_name)

        aug_images = augment_image(img)

        for i, aug in enumerate(aug_images, start=1):

            new_name = f"{name}_aug{i}{ext}"

            save_path = os.path.join(folder, new_name)

            cv2.imwrite(save_path, aug)

print("Train dataset augmentation completed successfully.")
