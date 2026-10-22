import os
import pandas as pd

IMG_DIR = "dataset/images"
LABEL_DIR = "dataset/labels"
CSV_PATH = "train.csv"

os.makedirs(LABEL_DIR, exist_ok=True)

df = pd.read_csv(CSV_PATH)

# remove "No finding"
df = df[df["class_name"] != "No finding"]

classes = sorted(df["class_name"].unique())
class_map = {name: i for i, name in enumerate(classes)}

IMG_SIZE = 512

for img_id in df["image_id"].unique():

    # support both jpg and png
    img_path_jpg = os.path.join(IMG_DIR, f"{img_id}.jpg")
    img_path_png = os.path.join(IMG_DIR, f"{img_id}.png")

    if os.path.exists(img_path_jpg):
        img_path = img_path_jpg
    elif os.path.exists(img_path_png):
        img_path = img_path_png
    else:
        continue

    records = df[df["image_id"] == img_id]

    label_path = os.path.join(LABEL_DIR, f"{img_id}.txt")

    with open(label_path, "w") as f:

        for _, row in records.iterrows():

            x_min = row["x_min"] / IMG_SIZE
            y_min = row["y_min"] / IMG_SIZE
            x_max = row["x_max"] / IMG_SIZE
            y_max = row["y_max"] / IMG_SIZE

            x_center = (x_min + x_max) / 2
            y_center = (y_min + y_max) / 2
            width = x_max - x_min
            height = y_max - y_min

            class_id = class_map[row["class_name"]]

            if 0 < width <= 1 and 0 < height <= 1:

                f.write(f"{class_id} {x_center} {y_center} {width} {height}\n")

print("Labels generated successfully.")