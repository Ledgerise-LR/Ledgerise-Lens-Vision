import cv2
import numpy as np


def draw_corners(gray, image):
    # Define the maximum number of corners to detect
    max_corners = 20  # You can adjust this number based on your needs

    # Define the quality level (0-1) - A higher value detects better-quality corners
    quality_level = 0.01

    # Define the minimum Euclidean distance between detected corners
    min_distance = 200  # Adjust as needed

    # Use the Shi-Tomasi corner detection method to detect corners
    corners = cv2.goodFeaturesToTrack(
        gray,
        maxCorners=max_corners,
        qualityLevel=quality_level,
        minDistance=min_distance,
    )

    # Convert corners to integers
    corners = np.int0(corners)

    # Draw circles at the detected corners
    for corner in corners:
        x, y = corner.ravel()
        cv2.circle(
            image, (x, y), 15, (0, 0, 255), -1
        )  # Draw a red circle at the corner location
