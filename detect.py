from process_custom import process_custom
import sys
import cv2
import numpy as np
import base64
import fileinput

input = ""

for line in fileinput.input(encoding="utf-8"):
    input += line.strip()

image_data = base64.b64decode(input)

image_np_array = np.frombuffer(image_data, np.uint8)

image = cv2.imdecode(image_np_array, cv2.IMREAD_COLOR)

width = int(375 * 0.941)
height = int(650 * 0.941)

# dsize
dsize = (width, height)


if image is not None:
    image = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
    img_rgb, user_info, found_status, coordinates = process_custom(image)

    results = {
        "coordinates_array": coordinates,
        "found_status": found_status,
        "user_info": user_info,
    }
    sys.stdout.write(str(results))
    sys.stdout.flush()
else:
    pass
