import cv2
import numpy as np
import matplotlib.pyplot as plt

from utils.draw_contours import draw_contours
from utils.draw_corners import draw_corners


def preprocess():
    img_read = cv2.imread("./preprocess/data/img3.png", cv2.IMREAD_UNCHANGED)
    img_rgb = cv2.cvtColor(img_read, cv2.COLOR_BGR2RGB)
    img = cv2.imread("./preprocess/data/img3.png", cv2.IMREAD_GRAYSCALE)
    canny = cv2.Canny(img, 127, 255)
    _, th1 = cv2.threshold(img, 70, 255, cv2.THRESH_BINARY)
    th2 = cv2.adaptiveThreshold(
        img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 10
    )
    th3 = cv2.adaptiveThreshold(
        img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 10
    )

    # define the alpha and beta
    alpha = 1.4  # Contrast control
    beta = 10  # Brightness control 1.4 10

    # call convertScaleAbs function
    adjusted = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)

    _, th_adjusted = cv2.threshold(adjusted, 95, 255, cv2.THRESH_BINARY)

    canny_adjusted = cv2.Canny(adjusted, 127, 255)

    black_mask_adjusted_treshold = cv2.bitwise_and(img, th_adjusted)

    # top_border = 5
    # bottom_border = 5
    # left_border = 5
    # right_border = 5

    # border_color = (0, 255, 0)

    # cv2.copyMakeBorder(
    #     img_rgb,
    #     top_border,
    #     bottom_border,
    #     left_border,
    #     right_border,
    #     cv2.BORDER_CONSTANT,
    #     value=border_color,
    # )

    # bordered_mask_adjusted_treshold = cv2.copyMakeBorder(
    #     black_mask_adjusted_treshold,
    #     top_border,
    #     bottom_border,
    #     left_border,
    #     right_border,
    #     cv2.BORDER_CONSTANT,
    #     value=border_color,
    # )

    bordered_mask_adjusted_treshold_canny = cv2.adaptiveThreshold(
        black_mask_adjusted_treshold,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        11,
        3,
    )

    blurred = cv2.GaussianBlur(black_mask_adjusted_treshold, (5, 5), 0)
    median_filtered = cv2.medianBlur(
        black_mask_adjusted_treshold, 5
    )  # Adjust the kernel size as needed
    bilateral_filtered = cv2.bilateralFilter(
        bordered_mask_adjusted_treshold_canny, d=4, sigmaColor=75, sigmaSpace=150
    )

    noise_reduction_image = cv2.adaptiveThreshold(
        bilateral_filtered,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        11,
        3,
    )

    blurred_out_x = cv2.GaussianBlur(noise_reduction_image, (3, 1), 0)
    blurred_out_y = cv2.GaussianBlur(noise_reduction_image, (1, 3), 0)

    CLOSE_RECT = 2
    OPEN_RECT = 15

    se1_x = cv2.getStructuringElement(cv2.MORPH_RECT, (CLOSE_RECT, CLOSE_RECT))
    se2_x = cv2.getStructuringElement(cv2.MORPH_RECT, (OPEN_RECT, OPEN_RECT))
    mask_x = cv2.morphologyEx(
        blurred_out_x.astype("uint8"),
        cv2.MORPH_CLOSE,
        se1_x,
    )
    mask_x = cv2.morphologyEx(mask_x, cv2.MORPH_OPEN, se2_x)

    mask_x = np.dstack([mask_x, mask_x, mask_x]) / 255
    out_x = img_rgb * mask_x

    se1_y = cv2.getStructuringElement(cv2.MORPH_RECT, (CLOSE_RECT, CLOSE_RECT))
    se2_y = cv2.getStructuringElement(cv2.MORPH_RECT, (OPEN_RECT, OPEN_RECT))
    mask_y = cv2.morphologyEx(
        blurred_out_y.astype("uint8"),
        cv2.MORPH_CLOSE,
        se1_y,
    )
    mask_y = cv2.morphologyEx(mask_y, cv2.MORPH_OPEN, se2_x)

    mask_y = np.dstack([mask_y, mask_y, mask_y]) / 255
    out_y = img_rgb * mask_y

    out_final = cv2.bitwise_and(out_x, out_y)
    out_final = cv2.cvtColor(np.uint8(out_final), cv2.COLOR_RGB2GRAY)
    _, out_final = cv2.threshold(out_final, 0, 255, cv2.THRESH_BINARY)

    images = [
        img,
        # canny,
        # th1,
        # th2,
        # th3,
        # adjusted,
        # th_adjusted,
        # canny_adjusted,
        # black_mask_adjusted_treshold,
        # bordered_mask_adjusted_treshold,
        bordered_mask_adjusted_treshold_canny,
        # blurred,
        # median_filtered,
        noise_reduction_image,
        cv2.bitwise_not(noise_reduction_image),
        out_x,
        out_y,
        draw_contours(img_rgb, cv2.bitwise_not(out_final))[0],
    ]
    titles = [
        "image",
        # "canny",
        # "th1",
        # "th2",
        # "th3",
        # "Adjusted",
        # "th_adjusted",  # black mask
        # "canny_adjusted",
        # "black_mask_adjusted_treshold",
        # "bordered_mask_adjusted_treshold",
        "canny",
        # "blurred",
        # "median_filtered",
        "final",
        "not",
        "out_x",
        "out_y",
        "out_final",
    ]

    plt.figure(figsize=(15, 10))
    for i in range(len(images)):
        plt.subplot(3, 3, i + 1)
        plt.title(titles[i])
        plt.imshow(images[i], cmap="gray")
        plt.axis("off")

    plt.show()

    results = {"rect_images": draw_contours(img_rgb, out_final)[1], "img_rgb": img_rgb}

    return results


preprocess()
