from ultralytics import YOLO

# Load a YOLOv8n PyTorch model
model = YOLO("/home/valid/Documents/Visto/gundu-ai/models/best-6-class-tuned.pt")

# Export the model
model.export(format="openvino")  # creates 'yolov8n_openvino_model/'