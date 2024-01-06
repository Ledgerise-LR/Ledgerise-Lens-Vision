import cv2
import numpy as np
import matplotlib.pyplot as plt

from .utils.draw_contours import draw_contours
from .utils.draw_corners import draw_corners

contour_image_third_case = []
contour_image_fourth_case = []
contour_image_fifth_case = []
contour_image_sixth_case = []


def preprocessv2(img):
    alpha = 1.2
    beta = 20
    adjusted = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)

    sat = cv2.cvtColor(adjusted, cv2.COLOR_BGR2HSV)[:, :, 1]
    # sat = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    sat = cv2.equalizeHist(sat)

    # hist = cv2.calcHist([sat], [0], None, [256], [0, 256])

    avg = np.median(np.median(sat, axis=0), axis=0)

    _, thresh = cv2.threshold(sat, 127, 255, cv2.THRESH_BINARY)

    OPEN_KERNEL, CLOSE_KERNEL = np.zeros((3, 3), np.uint8), np.ones((30, 30), np.uint8)

    morph = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, OPEN_KERNEL)
    morph = cv2.morphologyEx(morph, cv2.MORPH_CLOSE, CLOSE_KERNEL)

    (contour_image_first_case, rect_images_first_case) = draw_contours(
        cv2.cvtColor(img, cv2.COLOR_BGR2RGB), morph, 0, 0
    )
    (contour_image_second_case, rect_images_second_case) = draw_contours(
        cv2.cvtColor(img, cv2.COLOR_BGR2RGB), cv2.bitwise_not(morph), 0, 0
    )

    total_rect_images = []

    for i in range(len(rect_images_first_case)):
        total_rect_images.append(rect_images_first_case[i])

    for i in range(len(rect_images_second_case)):
        total_rect_images.append(rect_images_second_case[i])

    for i in range(len(total_rect_images)):
        x, y, w, h = total_rect_images[i]
        rect_img = sat[y : y + h, x : x + w]

        sat_avg = np.median(np.median(sat, axis=0), axis=0)
        avg = np.median(np.median(rect_img, axis=0), axis=0)

        # print(avg, sat_avg)

        border = 190 if avg > sat_avg else 60

        _, thresh = cv2.threshold(rect_img, 60, 255, cv2.THRESH_BINARY)
        OPEN_KERNEL, CLOSE_KERNEL = np.zeros((1, 1), np.uint8), np.ones(
            (20, 20), np.uint8
        )

        morph = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, OPEN_KERNEL)
        morph = cv2.morphologyEx(morph, cv2.MORPH_CLOSE, CLOSE_KERNEL)

        (contour_image_third_case, rect_rect_images_first_case) = draw_contours(
            cv2.cvtColor(img, cv2.COLOR_BGR2RGB), morph, x, y
        )
        (contour_image_fourth_case, rect_rect_images_second_case) = draw_contours(
            cv2.cvtColor(img, cv2.COLOR_BGR2RGB), cv2.bitwise_not(morph), x, y
        )

        for i in rect_rect_images_first_case:
            total_rect_images.append(i)

        for i in rect_rect_images_second_case:
            total_rect_images.append(i)

    for i in range(len(total_rect_images)):
        x, y, w, h = total_rect_images[i]
        rect_img = sat[y : y + h, x : x + w]

        sat_avg = np.median(np.median(sat, axis=0), axis=0)
        avg = np.median(np.median(rect_img, axis=0), axis=0)

        # print(avg, sat_avg)

        border = 190 if avg > sat_avg else 60

        _, thresh = cv2.threshold(rect_img, 190, 255, cv2.THRESH_BINARY)
        OPEN_KERNEL, CLOSE_KERNEL = np.zeros((1, 1), np.uint8), np.ones(
            (20, 20), np.uint8
        )

        morph = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, OPEN_KERNEL)
        morph = cv2.morphologyEx(morph, cv2.MORPH_CLOSE, CLOSE_KERNEL)

        (contour_image_fifth_case, rect_rect_images_first_case) = draw_contours(
            cv2.cvtColor(img, cv2.COLOR_BGR2RGB), morph, x, y
        )
        (contour_image_sixth_case, rect_rect_images_second_case) = draw_contours(
            cv2.cvtColor(img, cv2.COLOR_BGR2RGB), cv2.bitwise_not(morph), x, y
        )

        for i in rect_rect_images_first_case:
            total_rect_images.append(i)
        for i in rect_rect_images_second_case:
            total_rect_images.append(i)

    # print(rect_images_first_case + rect_images_second_case)

    # images = [
    #     img,
    #     sat,
    #     thresh,
    #     morph,
    #     contour_image_first_case,
    #     contour_image_second_case,
    #     contour_image_third_case,
    #     contour_image_fourth_case,
    #     contour_image_fifth_case,
    #     contour_image_sixth_case,
    # ]
    # titles = [
    #     "img",
    #     "sat",
    #     "thresh",
    #     "morph",
    #     "rect_images_first_case",
    #     "rect_images_second_case",
    #     "rect_images_third_case",
    #     "rect_images_fourth_case",
    #     "rect_images_fifth_case",
    #     "rect_images_sixth_case",
    # ]

    # plt.figure(figsize=(10, 7))
    # for i in range(len(images)):
    #     plt.subplot(2, 5, i + 1)
    #     plt.imshow(images[i])
    #     plt.title(titles[i])
    #     plt.axis("off")

    # plt.show()

    return [
        (total_rect_images),
        cv2.cvtColor(img, cv2.COLOR_BGR2RGB),
    ]


# img = cv2.imread("./preprocess/data/img14.png")
# preprocessv2(img)
