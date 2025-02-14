import cv2
import numpy as np
import base64
import requests

import cv2
from preprocess.blurImage import blurAidParcelBackground
from process_custom import process_custom


def getBlur(tokenUri: str, bounds: object):
    response = requests.get(tokenUri)
    image_array = np.frombuffer(response.content, dtype=np.uint8)

    image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

    # image = cv2.imread("./preprocess/data/img18.png")

    if image is not None:
        found_status, coordinates = process_custom(image, bounds)

        [x, w, y, h] = coordinates
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 5)
        res_image: np.ndarray = blurAidParcelBackground(image, bounds)
        res_image_base64 = base64.b64encode(cv2.imencode(".png", res_image)[1]).decode()  # type: ignore

        return {"image": res_image_base64}
        # cv2.imshow("res", res_image)
        # cv2.waitKey()
    else:
        pass
    
# getBlur("")

