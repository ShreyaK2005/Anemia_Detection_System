#CLEANING MASTER DATASET
from PIL import Image
import os

dataset_path = "master_dataset"

removed = 0

for root, dirs, files in os.walk(dataset_path):
    for file in files:

        if file.lower().endswith((".jpg",".jpeg",".png")):

            path = os.path.join(root,file)

            try:
                img = Image.open(path)
                img.verify()

            except:
                print("Removing corrupted image:", path)
                os.remove(path)
                removed += 1

print("Cleaning finished.")
print("Total removed:", removed)

#CLEANING MASTER DATASET 1
from PIL import Image
import os

dataset_path = "master_dataset_1"

removed = 0

for root, dirs, files in os.walk(dataset_path):
    for file in files:

        if file.lower().endswith((".jpg",".jpeg",".png")):

            path = os.path.join(root,file)

            try:
                img = Image.open(path)
                img.verify()

            except:
                print("Removing corrupted image:", path)
                os.remove(path)
                removed += 1

print("Cleaning finished.")
print("Total removed:", removed)
