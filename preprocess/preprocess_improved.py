import cv2
import numpy as np
import matplotlib.pyplot as plt

from .utils.draw_contours import draw_contours
from .utils.draw_corners import draw_corners


def preprocessv2(img):
    sat = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)[:, :, 1]
    # sat = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    sat = cv2.equalizeHist(sat)

    # hist = cv2.calcHist([sat], [0], None, [256], [0, 256])

    avg = np.median(np.median(sat, axis=0), axis=0)

    _, thresh = cv2.threshold(sat, int(avg) + 10, 255, 0)

    OPEN_KERNEL, CLOSE_KERNEL = np.zeros((20, 20), np.uint8), np.zeros((5, 5), np.uint8)

    morph = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, OPEN_KERNEL)
    morph = cv2.morphologyEx(morph, cv2.MORPH_CLOSE, CLOSE_KERNEL)

    (contour_image_first_case, rect_images_first_case) = draw_contours(
        cv2.cvtColor(img, cv2.COLOR_BGR2RGB), morph
    )
    (contour_image_second_case, rect_images_second_case) = draw_contours(
        cv2.cvtColor(img, cv2.COLOR_BGR2RGB), cv2.bitwise_not(morph)
    )

    # images = [
    #     img,
    #     sat,
    #     thresh,
    #     morph,
    #     contour_image_first_case,
    #     contour_image_second_case,
    # ]
    # titles = [
    #     "img",
    #     "sat",
    #     "thresh",
    #     "morph",
    #     "rect_images_first_case",
    #     "rect_images_second_case",
    # ]

    # plt.figure(figsize=(10, 7))
    # for i in range(len(images)):
    #     plt.subplot(2, 3, i + 1)
    #     plt.imshow(images[i])
    #     plt.title(titles[i])
    #     plt.axis("off")

    # plt.show()

    return [
        rect_images_first_case + rect_images_second_case,
        cv2.cvtColor(img, cv2.COLOR_BGR2RGB),
    ]
