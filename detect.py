from process_custom import process_custom
import sys
import cv2
import numpy as np
import base64


image = cv2.imread(sys.argv[1])
if image is not None:
    img_rgb, user_info, found_status = process_custom(image)
    processed_image_data = base64.b64encode(cv2.imencode(".png", img_rgb)[1]).decode("utf-8")  # type: ignore
    results = {
        "img_rgb": processed_image_data,
        "user_info": user_info,
        "found_status": found_status,
    }
    with open("../Nft-Fundraising-nodejs-backend/res_image.png", "wb") as f:
        f.write(base64.b64decode(processed_image_data))
    sys.stdout.write("done")
    sys.stdout.flush()
else:
    pass
