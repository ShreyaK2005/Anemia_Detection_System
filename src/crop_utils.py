import cv2
import os
import numpy as np
import argparse
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load Haar cascade for eye detection
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

def crop_conjunctiva(img):
    """
    Detects eye region, extends downward to include conjunctiva, applies mask, and crops.
    Returns a resized (224x224) image of the eye region.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    eyes = eye_cascade.detectMultiScale(gray, 1.3, 5)

    if len(eyes) > 0:
        x, y, w_eye, h_eye = eyes[0]
        pad = int(0.4 * h_eye)
        y_new = max(y + h_eye // 2 - pad, 0)
        h_new = min(h_eye + pad, img.shape[0] - y_new)
        eye_crop = img[y_new:y_new + h_new, x:x + w_eye]
    else:
        # Fallback to whole image if eye not detected
        eye_crop = img

    # Convert to HSV and create mask for redness
    hsv = cv2.cvtColor(eye_crop, cv2.COLOR_BGR2HSV)
    lower = np.array([0, 70, 100])
    upper = np.array([20, 255, 255])
    mask = cv2.inRange(hsv, lower, upper)

    # Morphological operations to clean mask
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        # Filter small contours
        contours = [c for c in contours if cv2.contourArea(c) > 500]
        if contours:
            c = max(contours, key=cv2.contourArea)
            x, y, w_box, h_box = cv2.boundingRect(c)
            pad = 15
            x = max(x - pad, 0)
            y = max(y - pad, 0)
            h_e, w_e = eye_crop.shape[:2]
            w_box = min(w_box + 2 * pad, w_e - x)
            h_box = min(h_box + 2 * pad, h_e - y)
            cropped = eye_crop[y:y + h_box, x:x + w_box]
            return cv2.resize(cropped, (224, 224))

    # Fallback: center crop
    h_e, w_e = eye_crop.shape[:2]
    cx, cy = w_e // 2, h_e // 2
    size = min(h_e, w_e) // 2
    x1 = max(cx - size // 2, 0)
    y1 = max(cy - size // 2, 0)
    x2 = x1 + size
    y2 = y1 + size
    cropped = eye_crop[y1:y2, x1:x2]
    return cv2.resize(cropped, (224, 224))

def process_dataset(input_path, output_path, classes):
    """
    Processes dataset images: crops eyes with conjunctiva and saves to output folder.
    """
    for cls in classes:
        folder = os.path.join(input_path, cls)
        output_cls_folder = os.path.join(output_path, cls)
        os.makedirs(output_cls_folder, exist_ok=True)

        for img_name in os.listdir(folder):
            if not img_name.lower().endswith((".jpg", ".jpeg", ".png")):
                continue
            img_path = os.path.join(folder, img_name)
            img = cv2.imread(img_path)

            if img is None:
                logging.warning(f"Failed to read image: {img_path}")
                continue

            cropped = crop_conjunctiva(img)
            save_path = os.path.join(output_cls_folder, img_name)
            cv2.imwrite(save_path, cropped)
        logging.info(f"Processed class '{cls}' in folder '{folder}'.")

def main():
    parser = argparse.ArgumentParser(description="Crop eye region with conjunctiva from dataset images.")
    parser.add_argument("--input_path", type=str, default="combined_images_dataset", help="Path to input dataset.")
    parser.add_argument("--output_path", type=str, default="cropped_dataset", help="Path to save cropped images.")
    parser.add_argument("--classes", nargs='+', default=["anemia", "normal"], help="List of class folder names.")
    args = parser.parse_args()

    process_dataset(args.input_path, args.output_path, args.classes)
    logging.info("Cropping completed successfully.")

if __name__ == "__main__":
    main()
