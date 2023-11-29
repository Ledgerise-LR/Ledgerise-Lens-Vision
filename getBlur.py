import sys
import cv2
import numpy as np
import base64
import fileinput
import requests

import cv2
from preprocess.blurImage import blurAidParcelBackground
from process_custom import process_custom

# input = ""

# for line in fileinput.input(encoding="utf-8"):
#     input += line.strip()

# response = requests.get(input)
# image_array = np.frombuffer(response.content, dtype=np.uint8)

# image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

image = cv2.imread("./preprocess/data/img8.png")

if image is not None:
    found_status, coordinates = process_custom(image)

    [x, w, y, h] = coordinates
    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 5)
    res_image: np.ndarray = blurAidParcelBackground(image)
    res_image_base64 = base64.b64encode(cv2.imencode(".png", res_image)[1]).decode()  # type: ignore

    cv2.imshow("res", res_image)
    cv2.waitKey()
    # chunk_size = 1000
    # for i in range(0, len(res_image_base64), chunk_size):
    #     chunk = res_image_base64[i : i + chunk_size]
    #     sys.stdout.write(chunk)
    #     sys.stdout.flush()
    # sys.stdout.write("end")
else:
    pass
