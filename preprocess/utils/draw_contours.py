import cv2


def draw_contours(img, ext):
    # Find contours in the grayscale image
    contours, _ = cv2.findContours(ext, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    print(len(contours))
    contour_image = img.copy()

    min_contour_area = 1

    for contour in contours:
        contour_area = cv2.contourArea(contour)
        if contour_area > min_contour_area:
            cv2.drawContours(
                contour_image, [contour], -1, (0, 255, 0), 2
            )  # Draw the contour in green

    return contour_image
