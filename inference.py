
from pathlib import Path
import os

import cv2
import numpy as np
from ultralytics import YOLO

# Resolve paths relative to this file, not the terminal's working directory.
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "runs" / "detect" / "train10" / "weights" / "best.pt"
RESULT_DIR = BASE_DIR / "static" / "results"
RESULT_DIR.mkdir(parents=True, exist_ok=True)

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Trained model not found at {MODEL_PATH}"
    )

model = YOLO(str(MODEL_PATH))


def generate_heatmap(img, boxes):
    """Create a visualization of predicted regions, not a medical saliency map."""
    h, w = img.shape[:2]
    mask = np.zeros((h, w), dtype=np.float32)

    for x1, y1, x2, y2 in boxes:
        cv2.rectangle(mask, (x1, y1), (x2, y2), 1.0, thickness=-1)

    if not boxes:
        return img.copy()

    # Smooth the box regions for display.
    mask = cv2.GaussianBlur(mask, (0, 0), sigmaX=max(3, min(h, w) * 0.025))
    max_value = float(mask.max())

    if max_value > 0:
        mask /= max_value

    heat = cv2.applyColorMap(
        (mask * 255).astype(np.uint8),
        cv2.COLORMAP_JET
    )
    return cv2.addWeighted(img, 0.60, heat, 0.40, 0)


def analyze_image(image_path):
    """Run the trained YOLO model on an uploaded image."""
    img = cv2.imread(str(image_path))

    if img is None:
        raise ValueError("The uploaded file could not be read as an image.")

    # Run actual model inference.
    prediction = model.predict(
        source=img,
        conf=0.25,
        verbose=False
    )[0]

    detected = img.copy()
    findings = []
    boxes_for_heatmap = []

    if prediction.boxes is not None:
        for box in prediction.boxes:
            x1, y1, x2, y2 = map(
                int, box.xyxy[0].cpu().tolist()
            )
            class_id = int(box.cls[0].item())
            confidence = float(box.conf[0].item())

            label = prediction.names.get(
                class_id, f"Class {class_id}"
            )

            # Keep coordinates inside the image.
            h, w = img.shape[:2]
            x1, x2 = max(0, x1), min(w - 1, x2)
            y1, y2 = max(0, y1), min(h - 1, y2)

            if x2 <= x1 or y2 <= y1:
                continue

            color = (0, 80, 255)
            display_label = f"{label} {confidence:.0%}"

            cv2.rectangle(
                detected, (x1, y1), (x2, y2), color, 2
            )
            cv2.putText(
                detected,
                display_label,
                (x1, max(20, y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                2
            )

            findings.append(
                f"{label} (confidence: {confidence:.1%})"
            )
            boxes_for_heatmap.append((x1, y1, x2, y2))

    # Absence of detections does not establish that an X-ray is normal.
    if findings:
        status = "Model detections found"
    else:
        status = "No model detections"

    detected_path = RESULT_DIR / "detected.png"
    heatmap_path = RESULT_DIR / "heatmap.png"

    heatmap = generate_heatmap(img, boxes_for_heatmap)

    if not cv2.imwrite(str(detected_path), detected):
        raise OSError("Could not save the detection visualization.")

    if not cv2.imwrite(str(heatmap_path), heatmap):
        raise OSError("Could not save the region visualization.")

    return {
        "severity": status,
        "findings": findings or ["No detections returned by the model"],
        "detected_image": str(detected_path.relative_to(BASE_DIR)).replace("\\", "/"),
        "heatmap_image": str(heatmap_path.relative_to(BASE_DIR)).replace("\\", "/")
    }
