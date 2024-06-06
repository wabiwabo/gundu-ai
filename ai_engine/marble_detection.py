from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator
from ai_engine.rabbitmq import rabbitmq_declare_queue, rabbitmq_publish
import numpy as np
import os
import dotenv
import cv2
import time
import json
import logging
from logging.handlers import RotatingFileHandler

# Create a logger
logger = logging.getLogger("my_logger")
logger.setLevel(logging.DEBUG)

# Create a handler that writes log messages to a file with a limit on the file size and backup count
handler = RotatingFileHandler("error.log", maxBytes=1024*1024*25, backupCount=3)  # 5MB per file, 3 backups
handler.setLevel(logging.DEBUG)

# Create a formatter and set it for the handler
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)

# Add the handler to the logger
logger.addHandler(handler)

dotenv.load_dotenv()

class MarbleDetection:
    def __init__(self):
        self.model = YOLO(os.getenv('BASE_PATH') + '/models/best-6-class-tuned.pt')
        self.source = os.getenv('SOURCE_CAM')
        try:
            self.source = int(self.source)
        except ValueError:
            self.source = str(self.source)
        self.cap = cv2.VideoCapture(self.source)
        self.width  = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.min_width = int(self.width * float(os.getenv('FINISH_LINE_LEFT')))
        self.min_height = int(self.height * float(os.getenv('FINISH_LINE_TOP')))
        self.max_width = int(self.width * float(os.getenv('FINISH_LINE_RIGHT')))
        self.max_height = int(self.height * float(os.getenv('FINISH_LINE_BOTTOM')))
        self.orientation = os.getenv('ORIENTATION')
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.fontScale = 1
        self.fontColor = (255, 0, 0)
        self.thickness = 2
        self.lineType  = 1

        self.connection = None
        self.channel = None
        self.out = None

        if isinstance(self.source, str):
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            self.out = cv2.VideoWriter('output.mp4', fourcc, self.fps, (self.width, self.height))

        self.cap.release()

        self.rank_queue_name = os.getenv('SERVER_ID') + os.getenv('RANK_QUEUE_POSTFIX')
        self.finish_status_queue_name = os.getenv('SERVER_ID') + os.getenv('FINISH_STATUS_QUEUE_POSTFIX')

        try:
            rabbitmq_declare_queue(self.rank_queue_name)
            rabbitmq_declare_queue(self.finish_status_queue_name)
        except Exception as e:
            logger.error(e, stack_info=True, exc_info=True)

    def predict_color(self, img, bbox):
        array = np.ascontiguousarray(img[int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2])])
        # array = cv2.cvtColor(array, cv2.COLOR_BGR2RGB)
        filename= os.getenv('BASE_PATH') + '/marble.jpg'
        cv2.imwrite(filename, array) 
        print(bbox)
        color, colors, probs = self.color_model.predict(array, 0.01)
        return (color, colors, probs)

    def predict(self, img, publish):
        results = self.model.predict(img, conf = float(os.getenv('CONF')), iou = float(os.getenv('IOU')), agnostic_nms=True)
        names = self.model.names
        rank = []
        for r in results:
            annotator = Annotator(img)
            boxes = r.boxes
            confs = r.boxes.conf
            classes = r.boxes.cls
            for box, conf, cls in zip(boxes, confs, classes):
                b = box.xyxy[0]  # get box coordinates in (left, top, right, bottom) format
                bbox = b.cpu().numpy()
                name = names[int(cls)][8:].upper()
                if bbox[0] >= self.min_width and bbox[2] <= self.max_width and bbox[1] >= self.min_height and bbox[3] <= self.max_height:
                    if self.orientation == "LEFT" or self.orientation == "RIGHT":
                        rank.append(((int(bbox[0] + bbox[2]) / 2.0), name, bbox, conf))
                    elif self.orientation == "TOP" or self.orientation == "BOTTOM":
                        rank.append(((int(bbox[1] + bbox[3]) / 2.0), name, bbox, conf))

                annotator.box_label(b, name)

        img = annotator.result()
        
        if self.orientation == "LEFT" or self.orientation == "TOP":
            rank.sort(reverse=False)
        elif self.orientation == "RIGHT" or self.orientation == "BOTTOM":
            rank.sort(reverse=True)

        rank_message = {}
        timestamp = time.time()
        marbles = []
        idx = 1
        for r in rank:
            marble = {"name": r[1], "rank": str(idx)}
            idx += 1
            marbles.append(marble)

        rank_message['marbles'] = marbles
        rank_message['timestamp'] = timestamp

        finish_notif = {}
        finish_status = False
        if len(rank) == int(os.getenv('NUM_MARBLES')):
            finish_status = True

        finish_notif['finish_status'] = finish_status
        finish_notif['timestamp'] = timestamp

        rank_message = json.dumps(rank_message, indent = 4)
        finish_notif = json.dumps(finish_notif, indent = 4)

        if publish == True:
            if len(marbles) == 0:
                print("EMPTY")
            else:
                try:
                    rabbitmq_publish(rank_message, self.rank_queue_name)
                    rabbitmq_publish(finish_notif, self.finish_status_queue_name)
                except Exception as e:
                    logger.error(e, stack_info=True, exc_info=True)

        x = 50
        y = 100
        idx = 1
        img = cv2.putText(img, "LEADERBOARD", (x, y), self.font, self.fontScale,  
                    self.fontColor, self.thickness, cv2.LINE_AA, False)
        y += 50
        for r in rank:
            s = str(idx) + ". " + r[1] 
            img = cv2.putText(img, s, (x, y), self.font, self.fontScale,  
                    self.fontColor, self.thickness, cv2.LINE_AA, False)
            y += 50
            idx += 1

        color = (0, 255, 0)
        thickness = 2
        start_point = (int(float(os.getenv('FINISH_LINE_LEFT')) * img.shape[1]), int(float(os.getenv('FINISH_LINE_TOP')) * img.shape[0]))
        end_point = (int(float(os.getenv('FINISH_LINE_RIGHT')) * img.shape[1]), int(float(os.getenv('FINISH_LINE_BOTTOM')) * img.shape[0]))
        img = cv2.rectangle(img, start_point, end_point, color, thickness)

        print(marbles)

        return img