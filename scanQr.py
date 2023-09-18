import cv2
from pyzbar.pyzbar import decode
import numpy as np

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()

    if ret:
        # alpha = 1.5  # Contrast control
        # beta = 15  # Brightness control 1.4 10

        # adjusted = cv2.convertScaleAbs(frame, alpha=alpha, beta=beta)

        # gray = cv2.cvtColor(adjusted, cv2.COLOR_RGB2GRAY)
        # _, thresh = cv2.threshold(gray, 200, 200, cv2.THRESH_BINARY)

        for code in decode(frame):
            decoded_data = code.data.decode("utf-8")
            rect_pts = code.polygon

            rect_pts = np.array(rect_pts, dtype=np.int32)

            # print(decoded_data)
            cv2.polylines(frame, [rect_pts], True, (0, 255, 0), 3)

        cv2.imshow("frame", frame)

    if cv2.waitKey(1) == ord("q"):
        break
