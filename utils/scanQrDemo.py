import cv2
from pyzbar.pyzbar import decode
import numpy as np


def scanQr(img):
    for code in decode(img):
        decoded_data = code.data.decode("utf-8")
        rect_pts = code.polygon
        rect_pts = np.array(rect_pts, dtype=np.int32)
        # print(decoded_data)
        if decoded_data and rect_pts:
            return decoded_data, rect_pts
        else:
            return None, None
