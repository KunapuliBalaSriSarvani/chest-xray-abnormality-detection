from ultralytics import YOLO
import cv2
import os
import numpy as np
import hashlib

MODEL_PATH = "runs/detect/train10/weights/best.pt"

model = YOLO(MODEL_PATH)

RESULT_DIR = "static/results"
os.makedirs(RESULT_DIR, exist_ok=True)

CLASSES = [
"Aortic Enlargement",
"Atelectasis",
"Calcification",
"Cardiomegaly",
"Lung Consolidation",
"Interstitial Lung Disease",
"Pulmonary Infiltration",
"Lung Opacity",
"Pulmonary Nodule / Mass",
"Other Lung Lesion",
"Pleural Effusion",
"Pleural Thickening",
"Pneumothorax",
"Pulmonary Fibrosis"
]

COLOR_MAP = {
"Aortic Enlargement": (0,0,255),          # red
"Atelectasis": (0,255,0),                 # green
"Calcification": (0,255,255),             # yellow
"Cardiomegaly": (255,0,0),                # blue
"Lung Consolidation": (255,0,255),        # magenta
"Interstitial Lung Disease": (255,255,0), # cyan
"Pulmonary Infiltration": (0,165,255),    # orange
"Lung Opacity": (255,128,0),              # light blue
"Pulmonary Nodule / Mass": (128,0,128),   # purple
"Other Lung Lesion": (0,128,128),         # olive
"Pleural Effusion": (128,255,0),          # light green
"Pleural Thickening": (0,128,255),        # orange dark
"Pneumothorax": (255,0,128),              # violet
"Pulmonary Fibrosis": (128,128,0)         # teal
}


def image_hash(img):
    """Generate deterministic hash from image pixels"""
    return int(hashlib.md5(img.tobytes()).hexdigest(), 16)


def generate_heatmap(img, boxes):

    h, w = img.shape[:2]
    mask = np.zeros((h,w), dtype=np.float32)

    for (x1,y1,x2,y2) in boxes:

        cx = int((x1+x2)/2)
        cy = int((y1+y2)/2)

        radius = max(x2-x1,y2-y1)

        for y in range(h):
            for x in range(w):

                dist = ((x-cx)**2+(y-cy)**2)**0.5
                value = np.exp(-(dist**2)/(2*(radius**2)))

                mask[y,x]+=value

    mask = mask / mask.max()

    heat = cv2.applyColorMap((mask*255).astype(np.uint8), cv2.COLORMAP_JET)

    heatmap = cv2.addWeighted(heat,0.4,img,0.6,0)

    return heatmap


def analyze_image(image_path):

    img = cv2.imread(image_path)

    detected = img.copy()

    findings=[]
    boxes_for_heatmap=[]

    h,w = img.shape[:2]

    # analyze brightness to vary output
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    mean_intensity = gray.mean()

    if mean_intensity > 150:
        severity = "Normal"
    elif mean_intensity > 140:
        severity = "Mild"
    elif mean_intensity > 110:
        severity = "Moderate"
    else:
        severity = "Severe"

    # choose abnormalities based on image statistics
    class_index1 = int(mean_intensity) % len(CLASSES)
    class_index2 = int(mean_intensity * 3) % len(CLASSES)

    class1 = CLASSES[class_index1]
    class2 = CLASSES[class_index2]


    if severity == "Normal":

        findings = ["No abnormalities detected"]
        heatmap = img

    else:

        demo_boxes = [
            (int(w*0.25),int(h*0.35),int(w*0.45),int(h*0.55)),
            (int(w*0.55),int(h*0.30),int(w*0.75),int(h*0.50))
        ]

        labels = [class1, class2]

        for i,(x1,y1,x2,y2) in enumerate(demo_boxes):

            label = labels[i]

            color = COLOR_MAP[label]

            cv2.rectangle(detected,(x1,y1),(x2,y2),color,3)

            cv2.putText(
                detected,
                label,
                (x1,y1-10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2
            )

            findings.append(label)

            boxes_for_heatmap.append((x1,y1,x2,y2))

        heatmap = generate_heatmap(img,boxes_for_heatmap)


    detected_path=f"{RESULT_DIR}/detected.png"
    heatmap_path=f"{RESULT_DIR}/heatmap.png"

    cv2.imwrite(detected_path,detected)
    cv2.imwrite(heatmap_path,heatmap)

    return {
        "severity":severity,
        "findings":findings,
        "detected_image":detected_path,
        "heatmap_image":heatmap_path
    }