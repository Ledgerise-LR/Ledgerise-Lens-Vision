import cv2
import numpy as np
import matplotlib.pyplot as plt

from .utils.draw_contours import draw_contours
from .utils.draw_corners import draw_corners


def preprocessv2():
    img = cv2.imread("./preprocess/data/img4.png")
    sat = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)[:, :, 0]

    _, thresh = cv2.threshold(sat, 90, 255, 0)

    OPEN_KERNEL, CLOSE_KERNEL = np.ones((7, 7), np.uint8), np.ones((13, 13), np.uint8)

    morph = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, OPEN_KERNEL)
    morph = cv2.morphologyEx(morph, cv2.MORPH_CLOSE, CLOSE_KERNEL)

    rect_images_first_case = draw_contours(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), morph)[
        1
    ]
    rect_images_second_case = draw_contours(
        cv2.cvtColor(img, cv2.COLOR_BGR2RGB), cv2.bitwise_not(morph)
    )[1]

    images = [img, sat, thresh, morph, rect_images_first_case, rect_images_second_case]
    titles = [
        "img",
        "sat",
        "thresh",
        "morph",
        "rect_images_first_case",
        "rect_images_second_case",
    ]

    # plt.figure(figsize=(10, 7))
    # for i in range(len(images)):
    #     plt.subplot(2, 3, i + 1)
    #     plt.imshow(images[i])
    #     plt.title(titles[i])
    #     plt.axis("off")

    # plt.show()

    return (
        (rect_images_first_case + rect_images_second_case),
        cv2.cvtColor(img, cv2.COLOR_BGR2RGB),
    )
