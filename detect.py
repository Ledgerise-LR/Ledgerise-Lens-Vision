from process_custom import process_custom
import sys
import cv2
import numpy as np
import base64

# Read the image data from stdin
base64_image_data = sys.argv[0]

# Decode the base64 image data
image_data = base64.b64decode(base64_image_data)

# Convert the image data to a NumPy array
image_array = np.frombuffer(image_data, np.uint8)

# Decode the image using OpenCV
image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

img_rgb, user_info, found_status = process_custom(image)

processed_image_data = base64.b64encode(cv2.imencode(".png", img_rgb)[1]).decode("utf-8")  # type: ignore

results = {
    "img_rgb": processed_image_data,
    "user_info": user_info,
    "found_status": found_status,
}

print(results)
