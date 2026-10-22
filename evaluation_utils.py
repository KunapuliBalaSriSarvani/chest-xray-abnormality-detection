import os
import pandas as pd
import shutil

RUN_DIR = "runs/detect/train10"
STATIC_EVAL_DIR = "static/evaluation"

os.makedirs(STATIC_EVAL_DIR, exist_ok=True)


def copy_evaluation_images():

    files = [
        "results.png",
        "confusion_matrix.png",
        "confusion_matrix_normalized.png",
        "PR_curve.png",
        "F1_curve.png",
        "P_curve.png",
        "R_curve.png",
        "val_batch0_pred.jpg",
        "val_batch1_pred.jpg"
    ]

    for f in files:

        src = os.path.join(RUN_DIR, f)
        dst = os.path.join(STATIC_EVAL_DIR, f)

        if os.path.exists(src):
            shutil.copy(src, dst)


def read_training_metrics():

    csv_path = os.path.join(RUN_DIR, "results.csv")

    if not os.path.exists(csv_path):

        return {
            "map50": 0,
            "map5095": 0,
            "precision": 0,
            "recall": 0
        }

    df = pd.read_csv(csv_path)

    last = df.iloc[-1]

    metrics = {
        "map50": round(last.get("metrics/mAP50(B)",0) * 100,2),
        "map5095": round(last.get("metrics/mAP50-95(B)",0) * 100,2),
        "precision": round(last.get("metrics/precision(B)",0) * 100,2),
        "recall": round(last.get("metrics/recall(B)",0) * 100,2)
    }

    copy_evaluation_images()

    return metrics