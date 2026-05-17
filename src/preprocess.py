import cv2
import numpy as np
import os
import argparse
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load Haar cascade for eye detection
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

def crop_conjunctiva(img):
    """
    Detects eye, expands downward to include conjunctiva, applies mask, and crops.
    Returns a resized (224x224) image.
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
        # Fallback to entire image if eye not detected
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

    # Contour-based crop
    if contours:
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

def process_images(input_folder, output_folder):
    """
    Processes all images in input_folder, crops eyes, and saves to output_folder.
    """
    for img_name in os.listdir(input_folder):
        if not img_name.lower().endswith(('.jpg', '.jpeg', '.png')):
            continue
        img_path = os.path.join(input_folder, img_name)
        img = cv2.imread(img_path)
        if img is None:
            logging.warning(f"Failed to read image: {img_path}")
            continue
        cropped_img = crop_conjunctiva(img)
        save_path = os.path.join(output_folder, img_name)
        cv2.imwrite(save_path, cropped_img)
        logging.info(f"Processed and saved: {save_path}")

def main():
    parser = argparse.ArgumentParser(description="Crop eye regions with conjunctiva from images.")
    parser.add_argument("--input_folder", type=str, required=True, help="Folder with input images.")
    parser.add_argument("--output_folder", type=str, required=True, help="Folder to save cropped images.")
    args = parser.parse_args()

    os.makedirs(args.output_folder, exist_ok=True)
    process_images(args.input_folder, args.output_folder)
    logging.info("Processing completed successfully.")

if __name__ == "__main__":
    main()
