import sys
import cv2
import numpy as np
import base64
import fileinput

from blurImage import blurAidParcelBackground

input = ""

for line in fileinput.input(encoding="utf-8"):
    input += line.strip()

image_data = base64.b64decode(input)

image_np_array = np.frombuffer(image_data, np.uint8)

image = cv2.imdecode(image_np_array, cv2.IMREAD_COLOR)

if image is not None:
    # image = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
    res_image = blurAidParcelBackground(image)
    res_image_base64 = base64.b64encode(res_image)  # type: ignore

    results = {
        "res_image": str(res_image_base64),
    }
    sys.stdout.write(str(results))
else:
    pass
