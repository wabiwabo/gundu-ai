from ultralytics import YOLO

model = YOLO("yolov8m-seg.pt")
results = model.train(data="/workspaces/gundu-ai/datasets/gundu-seg/data.yaml", epochs=100, imgsz=640, batch=8, workers=0, device=[0, 1])
# results = model.train(data="coco8-seg.yaml", epochs=100, imgsz=640, batch=16, workers=0, device=[0, 1])