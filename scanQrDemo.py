from pyzbar.pyzbar import decode
import numpy as np
import cv2
from PIL import Image
import base64
import matplotlib.pyplot as plt


def scanQr(img):
    decoded_objects = decode(img)
    if decoded_objects:
        for code in decoded_objects:
            decoded_data = code.data.decode("utf-8")
            rect_pts = code.polygon
            rect_pts = np.array(rect_pts, dtype=np.int32)
            # print(decoded_data)
            if decoded_data:
                return decoded_data, rect_pts
            else:
                return "None", "[]"
    else:
        return "None", "[]"
