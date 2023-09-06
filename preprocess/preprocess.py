import cv2
import numpy as np
import matplotlib.pyplot as plt

from utils.draw_contours import *
from utils.draw_corners import *

img_read = cv2.imread("./preprocess/data/img1.png", cv2.IMREAD_UNCHANGED)
img_rgb = cv2.cvtColor(img_read, cv2.COLOR_BGR2RGB)
img = cv2.imread("./preprocess/data/img1.png", cv2.IMREAD_GRAYSCALE)
canny = cv2.Canny(img, 127, 255)
_, th1 = cv2.threshold(img, 70, 255, cv2.THRESH_BINARY)
th2 = cv2.adaptiveThreshold(
    img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 10
)
th3 = cv2.adaptiveThreshold(
    img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 10
)

# define the alpha and beta
alpha = 1.3  # Contrast control
beta = 20  # Brightness control 1.4 10

# call convertScaleAbs function
adjusted = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)

_, th_adjusted = cv2.threshold(adjusted, 95, 255, cv2.THRESH_BINARY)

canny_adjusted = cv2.Canny(adjusted, 127, 255)

black_mask_adjusted_treshold = cv2.bitwise_and(img, th_adjusted)

top_border = 5
bottom_border = 5
left_border = 5
right_border = 5

border_color = (0, 255, 0)

bordered_mask_adjusted_treshold = cv2.copyMakeBorder(
    black_mask_adjusted_treshold,
    top_border,
    bottom_border,
    left_border,
    right_border,
    cv2.BORDER_CONSTANT,
    value=border_color,
)

bordered_mask_adjusted_treshold_canny = cv2.adaptiveThreshold(
    bordered_mask_adjusted_treshold,
    255,
    cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
    cv2.THRESH_BINARY,
    11,
    10,
)

blurred = cv2.GaussianBlur(bordered_mask_adjusted_treshold, (5, 5), 0)
median_filtered = cv2.medianBlur(
    bordered_mask_adjusted_treshold, 5
)  # Adjust the kernel size as needed
bilateral_filtered = cv2.bilateralFilter(
    bordered_mask_adjusted_treshold, d=7, sigmaColor=75, sigmaSpace=75
)

noise_reduction_image = cv2.adaptiveThreshold(
    bilateral_filtered,
    255,
    cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
    cv2.THRESH_BINARY,
    11,
    2,
)

draw_corners(noise_reduction_image, img_rgb)

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
    draw_contours(img_rgb, cv2.bitwise_not(noise_reduction_image)),
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
    "contour",
]


for i in range(len(images)):
    plt.subplot(2, 3, i + 1)
    plt.title(titles[i])
    plt.imshow(images[i], cmap="gray")
    plt.axis("off")

plt.show()
