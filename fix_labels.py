import os

folders = ["dataset/train/labels", "dataset/val/labels"]

for folder in folders:

    if not os.path.exists(folder):
        continue

    for file in os.listdir(folder):

        path = os.path.join(folder, file)
        new_lines = []

        with open(path) as f:
            for line in f:

                parts = line.strip().split()

                if len(parts) != 5:
                    continue

                cls, x, y, w, h = parts

                x = float(x)
                y = float(y)
                w = float(w)
                h = float(h)

                # clamp values safely inside YOLO range
                x = min(max(x, 0.001), 0.999)
                y = min(max(y, 0.001), 0.999)
                w = min(max(w, 0.001), 0.999)
                h = min(max(h, 0.001), 0.999)

                new_lines.append(f"{cls} {x} {y} {w} {h}\n")

        with open(path, "w") as f:
            f.writelines(new_lines)

print("Labels fixed correctly.")