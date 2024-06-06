import os
import dotenv
import cv2
from ai_engine.marble_detection import MarbleDetection
from ai_engine.helper import enhance_image, image_resize
import numpy as np
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
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

np.random.seed(345)

marble_detector = MarbleDetection()

source = os.getenv('SOURCE_CAM')
try:
    source = int(source)
except ValueError:
    source = str(source)

cap = cv2.VideoCapture(source)
width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)
min_width = int(width * float(os.getenv('FINISH_LINE_LEFT')))
min_height = int(height * float(os.getenv('FINISH_LINE_TOP')))
max_width = int(width * float(os.getenv('FINISH_LINE_RIGHT')))
max_height = int(height * float(os.getenv('FINISH_LINE_BOTTOM')))

out = None
if isinstance(source, str):
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter('output.mp4', fourcc, fps, (width, height))

fps = int(os.getenv('FPS'))
cnt = 0
while True:
    _, img = cap.read()

    if _ == False:
        break
    
    cnt += 1
    flag = False
    if cnt % fps == 0:
        cnt = 0
        flag = True

    # img = enhance_image(img)
    results = None
    try:
        results = marble_detector.predict(img, flag)
        results = image_resize(results, height=int(os.getenv('OUTPUT_HEIGHT')))
    except Exception as e:
        logger.error(e, stack_info=True, exc_info=True)

    if results is not None:
        if out is not None:
            out.write(results)
        else:
            cv2.imshow(os.getenv('WINDOW_NAME'), results)
            k = cv2.waitKey(1)

cv2.destroyAllWindows()
cap.release()
if isinstance(source, str):
    out.release()

