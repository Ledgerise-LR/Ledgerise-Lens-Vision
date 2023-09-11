import cv2
import matplotlib.pyplot as plt
from pprint import pprint


def draw_contours(img, ext):
    # Find contours in the grayscale image
    contours, hierarchy = cv2.findContours(
        ext, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    rectangle_images = []
    contour_image = img.copy()

    min_contour_area = 5000

    try:
        hierarchy = hierarchy[0]
    except:
        hierarchy = []

    height, width, _ = img.shape
    min_x, min_y = width, height
    max_x = max_y = 0

    for contour, hier in zip(contours, hierarchy):
        if cv2.contourArea(contour) > min_contour_area:
            (x, y, w, h) = cv2.boundingRect(contour)
            min_x, max_x = min(x, min_x), max(x + w, max_x)
            min_y, max_y = min(y, min_y), max(y + h, max_y)
            cv2.rectangle(contour_image, (x, y), (x + w, y + h), (255, 0, 0), 2)
            extracted_rectangle = img[y : y + h, x : x + w]
            rectangle_images.append(list([x, y, w, h]))
            print(len(rectangle_images))

    if max_x - min_x > 0 and max_y - min_y > 0:
        cv2.rectangle(contour_image, (min_x, min_y), (max_x, max_y), (255, 0, 0), 2)

    print("aa", len(rectangle_images))

    # plt.figure(figsize=(10, 7))
    # for i, rect_image in enumerate(rectangle_images):
    #     plt.subplot(
    #         1,
    #         len(rectangle_images),
    #         i + 1,
    #     )
    #     plt.imshow(rectangle_images[i]["extracted_rectangle"])
    #     plt.axis("off")
    return contour_image, rectangle_images
