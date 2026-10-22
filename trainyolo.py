from ultralytics import YOLO

model = YOLO("yolov8n.pt")

model.train(
    data="dataset/demo.yaml",
    epochs=5,
    imgsz=512,
    batch=4,
    device="cpu",
    workers=0
)