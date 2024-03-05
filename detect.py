from process_custom import process_custom
import cv2
import numpy as np
import base64

# input = ""

# for line in fileinput.input(encoding="utf-8"):
#     input += line.strip()


def processImage(base64Input: str, bounds):
    image_data = base64.b64decode(base64Input)

    image_np_array = np.frombuffer(image_data, np.uint8)

    image = cv2.imdecode(image_np_array, cv2.IMREAD_COLOR)

    if image is not None:
        # image = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
        found_status, coordinates = process_custom(image, bounds)

        results = {
            "coordinates_array": coordinates,
            "found_status": found_status,
        }
        return results
        # sys.stdout.write(str(results))
    else:
        pass
