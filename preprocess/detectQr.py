
import cv2
import numpy as np

img = cv2.imread("./preprocess/data/img15.png")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

qr_code_detector = cv2.QRCodeDetector()

decoded_data, points, _ = qr_code_detector.detectAndDecode(gray)

if points is not None:
  for i in range(len(points)):
      # Convert the points to integers
      pts = np.array(points[i], np.int32)
      pts = pts.reshape((-1, 1, 2))

      # Draw the bounding box around the QR code
      cv2.polylines(img, [pts], True, (0, 255, 0), thickness=2)

  # Display the decoded data
  print("Decoded Data:", decoded_data)

  # Display the image with bounding boxes
  cv2.imshow("QR Code Detection", img)
  cv2.waitKey(0)
  cv2.destroyAllWindows()
else:
  print("No QR code detected")
