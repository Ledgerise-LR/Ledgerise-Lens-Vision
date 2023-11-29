import cv2
import numpy as np
import matplotlib.pyplot as plt

from .utils.draw_contours import draw_contours
from .utils.draw_corners import draw_corners


def preprocessv2(img):
    alpha = 1.2
    beta = 10
    adjusted = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)

    sat = cv2.cvtColor(adjusted, cv2.COLOR_BGR2HSV)[:, :, 1]
    # sat = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    sat = cv2.equalizeHist(sat)

    # hist = cv2.calcHist([sat], [0], None, [256], [0, 256])

    avg = np.median(np.median(sat, axis=0), axis=0)

    _, thresh_down = cv2.threshold(sat, int(avg) * 0.8, 255, cv2.THRESH_BINARY)
    _, thresh_up = cv2.threshold(sat, int(avg) * 1.2, 255, cv2.THRESH_BINARY)

    thresh = cv2.bitwise_or(thresh_up, thresh_down)

    OPEN_KERNEL, CLOSE_KERNEL = np.zeros((3, 3), np.uint8), np.ones((30, 30), np.uint8)

    morph = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, OPEN_KERNEL)
    morph = cv2.morphologyEx(morph, cv2.MORPH_CLOSE, CLOSE_KERNEL)

    (contour_image_first_case, rect_images_first_case) = draw_contours(
        cv2.cvtColor(img, cv2.COLOR_BGR2RGB), morph, 0, 0
    )
    (contour_image_second_case, rect_images_second_case) = draw_contours(
        cv2.cvtColor(img, cv2.COLOR_BGR2RGB), cv2.bitwise_not(morph), 0, 0
    )

    for i in range(len(rect_images_first_case)):
        x, y, w, h = rect_images_first_case[i]
        rect_img = sat[y : y + h, x : x + w]

        avg = np.median(np.median(rect_img, axis=0), axis=0)

        _, thresh = cv2.threshold(rect_img, int(avg), 255, cv2.THRESH_BINARY)
        OPEN_KERNEL, CLOSE_KERNEL = np.zeros((2, 2), np.uint8), np.ones(
            (10, 10), np.uint8
        )

        morph = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, OPEN_KERNEL)
        morph = cv2.morphologyEx(morph, cv2.MORPH_CLOSE, CLOSE_KERNEL)

        (contour_image_first_case, rect_rect_images_first_case) = draw_contours(
            cv2.cvtColor(rect_img, cv2.COLOR_BGR2RGB), morph, x, y
        )
        (contour_image_second_case, rect_rect_images_second_case) = draw_contours(
            cv2.cvtColor(rect_img, cv2.COLOR_BGR2RGB), cv2.bitwise_not(morph), x, y
        )

        for i in rect_rect_images_first_case:
            rect_images_first_case.append(i)
        for i in rect_rect_images_second_case:
            rect_images_first_case.append(i)

    for i in range(len(rect_images_second_case)):
        x, y, w, h = rect_images_second_case[i]
        rect_img = sat[y : y + h, x : x + w]

        avg = np.median(np.median(rect_img, axis=0), axis=0)

        _, thresh = cv2.threshold(rect_img, int(avg), 255, cv2.THRESH_BINARY)
        OPEN_KERNEL, CLOSE_KERNEL = np.zeros((3, 3), np.uint8), np.ones(
            (10, 10), np.uint8
        )

        morph = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, OPEN_KERNEL)
        morph = cv2.morphologyEx(morph, cv2.MORPH_CLOSE, CLOSE_KERNEL)

        (contour_image_first_case, rect_rect_images_first_case) = draw_contours(
            cv2.cvtColor(rect_img, cv2.COLOR_BGR2RGB), morph, x, y
        )
        (contour_image_second_case, rect_rect_images_second_case) = draw_contours(
            cv2.cvtColor(rect_img, cv2.COLOR_BGR2RGB), cv2.bitwise_not(morph), x, y
        )

        for i in rect_rect_images_first_case:
            rect_images_second_case.append(i)
        for i in rect_rect_images_second_case:
            rect_images_second_case.append(i)

    # print(rect_images_first_case + rect_images_second_case)

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
        (rect_images_first_case + rect_images_second_case),
        cv2.cvtColor(img, cv2.COLOR_BGR2RGB),
    ]
