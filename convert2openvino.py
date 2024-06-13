from ultralytics import YOLO
import os
import dotenv

dotenv.load_dotenv()

# Load a YOLOv8n PyTorch model
model = YOLO(os.getenv('BASE_PATH') + '/models/best-6-class-tuned.pt')

# Export the model
model.export(format="openvino")  # creates 'yolov8n_openvino_model/'

# Load a YOLOv8n PyTorch model
model = YOLO(os.getenv('BASE_PATH') + '/models/best-6-class-tuned-seg.pt')

# Export the model
model.export(format="openvino")  # creates 'yolov8n_openvino_model/'