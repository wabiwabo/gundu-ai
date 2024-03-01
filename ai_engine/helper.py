import numpy as np
import cv2
from PIL import Image as im

def enhance_image(image):
    # Convert to HSV for better color manipulation
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Increase saturation for vibrance
    h, s, v = cv2.split(hsv)
    s = cv2.add(s, np.array([50.0]))  # Increase this value to increase saturation
    hsv_vibrant = cv2.merge([h, s, v])
    image_vibrant = cv2.cvtColor(hsv_vibrant, cv2.COLOR_HSV2BGR)

    # Increase contrast using CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    lab = cv2.cvtColor(image_vibrant, cv2.COLOR_BGR2Lab)
    l, a, b = cv2.split(lab)
    l = clahe.apply(l)
    lab_contrast = cv2.merge([l, a, b])
    image_contrast = cv2.cvtColor(lab_contrast, cv2.COLOR_Lab2BGR)

    # The final image is vibrant, has increased contrast, and enhanced colors
    final_image = image_contrast

    return final_image

def image_resize(image, width = None, height = None, inter = cv2.INTER_AREA):
    # initialize the dimensions of the image to be resized and
    # grab the image size
    dim = None
    (h, w) = image.shape[:2]

    # if both the width and height are None, then return the
    # original image
    if width is None and height is None:
        return image

    # check to see if the width is None
    if width is None:
        # calculate the ratio of the height and construct the
        # dimensions
        r = height / float(h)
        dim = (int(w * r), height)

    # otherwise, the height is None
    else:
        # calculate the ratio of the width and construct the
        # dimensions
        r = width / float(w)
        dim = (width, int(h * r))

    # resize the image
    resized = cv2.resize(image, dim, interpolation = inter)

    # return the resized image
    return resized