import os
from PIL import Image
import argparse
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def clean_dataset(dataset_path):
    removed = 0
    for root, dirs, files in os.walk(dataset_path):
        for file in files:
            if file.lower().endswith((".jpg", ".jpeg", ".png")):
                path = os.path.join(root, file)
                try:
                    with Image.open(path) as img:
                        img.verify()
                except Exception:
                    logging.info(f"Removing corrupted image: {path}")
                    os.remove(path)
                    removed += 1
    return removed

def main():
    parser = argparse.ArgumentParser(description="Clean dataset by removing corrupted images.")
    parser.add_argument("--dataset_path", type=str, required=True, help="Path to the dataset folder.")
    args = parser.parse_args()

    total_removed = clean_dataset(args.dataset_path)
    print(f"Cleaning finished. Total removed: {total_removed}")

if __name__ == "__main__":
    main()
