import os
import random
import shutil

SRC_IMG = "dataset/train/images"
SRC_LBL = "dataset/train/labels"

DST_IMG = "dataset/demo/images"
DST_LBL = "dataset/demo/labels"

os.makedirs(DST_IMG, exist_ok=True)
os.makedirs(DST_LBL, exist_ok=True)

labels = os.listdir(SRC_LBL)

# choose only images that have labels
sample = random.sample(labels, 50)

for lbl in sample:
    img = lbl.replace(".txt", ".jpg")

    shutil.copy(os.path.join(SRC_IMG, img), DST_IMG)
    shutil.copy(os.path.join(SRC_LBL, lbl), DST_LBL)

print("Demo dataset created with labeled images")