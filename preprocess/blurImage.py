import cv2
import numpy as np
import matplotlib.pyplot as plt
from .utils.draw_contours import draw_contours

# img = cv2.imread("./preprocess/data/blurImg.png")


def blurAidParcelBackground(img: np.ndarray, bounds) -> np.ndarray:
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    mask_green = cv2.inRange(img_hsv, (36, 25, 25), (70, 255, 255))  # type: ignore

    imask_green = mask_green > 0
    green = np.zeros_like(img, np.uint8)
    green[imask_green] = img[imask_green]

    green_mask_grey = cv2.cvtColor(green, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(green_mask_grey, 10, 255, cv2.THRESH_BINARY)

    (contour_image, rect_images_first_case) = draw_contours(img, thresh, 0, 0)

    x, y, w, h = rect_images_first_case[0]

    cv2.rectangle(thresh, (x, y), (x + w, y + h), (255, 255, 255), -1)
    cv2.rectangle(contour_image, (x, y), (x + w, y + h), (255, 255, 255), -1)

    blurred_img_rgb = cv2.GaussianBlur(contour_image, (33, 33), 10)
    # print(rect_images_first_case)

    filter = cv2.bitwise_and(img_rgb, img_rgb, mask=thresh)
    filter = cv2.cvtColor(filter, cv2.COLOR_BGR2RGB)

    blurred_img_rgb[y : y + h, x : x + w] = filter[y : y + h, x : x + w]

    res = blurred_img_rgb

    cv2.circle(res, (int(bounds["x"]), int(bounds["y"])), 2, (0, 255, 0), 2)

    return res
