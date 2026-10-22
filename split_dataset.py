import os
import random
import shutil

IMG_DIR = "dataset/images"
LBL_DIR = "dataset/labels"

TRAIN_IMG = "dataset/train/images"
TRAIN_LBL = "dataset/train/labels"
VAL_IMG = "dataset/val/images"
VAL_LBL = "dataset/val/labels"

for path in [TRAIN_IMG, TRAIN_LBL, VAL_IMG, VAL_LBL]:
    os.makedirs(path, exist_ok=True)

images = os.listdir(IMG_DIR)
random.shuffle(images)

split = int(len(images) * 0.8)

train_imgs = images[:split]
val_imgs = images[split:]

def move_files(img_list, dest_img, dest_lbl):
    for img in img_list:

        src_img = os.path.join(IMG_DIR, img)
        dst_img = os.path.join(dest_img, img)

        shutil.move(src_img, dst_img)

        base = os.path.splitext(img)[0]
        label_file = base + ".txt"

        src_lbl = os.path.join(LBL_DIR, label_file)

        if os.path.exists(src_lbl):
            shutil.move(src_lbl, os.path.join(dest_lbl, label_file))


move_files(train_imgs, TRAIN_IMG, TRAIN_LBL)
move_files(val_imgs, VAL_IMG, VAL_LBL)

print("Dataset split complete!")