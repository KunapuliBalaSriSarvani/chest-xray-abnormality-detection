import os
import shutil

TRAIN_IMG = "dataset/train/images"
VAL_IMG = "dataset/val/images"
IMG_DIR = "dataset/images"

os.makedirs(IMG_DIR, exist_ok=True)

# move train images back
for img in os.listdir(TRAIN_IMG):
    shutil.move(os.path.join(TRAIN_IMG, img), IMG_DIR)

# move val images back
for img in os.listdir(VAL_IMG):
    shutil.move(os.path.join(VAL_IMG, img), IMG_DIR)

print("Images restored to dataset/images")