import math

import cv2
import numpy as np


GAUSSIAN_SMOOTH_FILTER_SIZE = (5, 5)
ADAPTIVE_THRESH_BLOCK_SIZE = 19
ADAPTIVE_THRESH_WEIGHT = 9


def preprocess(img_original):
    img_grayscale = extract_value(img_original)
    img_max_contrast = maximize_contrast(img_grayscale)

    img_blurred = cv2.GaussianBlur(
        img_max_contrast,
        GAUSSIAN_SMOOTH_FILTER_SIZE,
        0,
    )

    img_thresh = cv2.adaptiveThreshold(
        img_blurred,
        255.0,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        ADAPTIVE_THRESH_BLOCK_SIZE,
        ADAPTIVE_THRESH_WEIGHT,
    )

    return img_grayscale, img_thresh


def extract_value(img_original):
    img_hsv = cv2.cvtColor(img_original, cv2.COLOR_BGR2HSV)
    _, _, img_value = cv2.split(img_hsv)

    return img_value


def maximize_contrast(img_grayscale):
    structuring_element = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (3, 3),
    )

    img_top_hat = cv2.morphologyEx(
        img_grayscale,
        cv2.MORPH_TOPHAT,
        structuring_element,
        iterations=10,
    )

    img_black_hat = cv2.morphologyEx(
        img_grayscale,
        cv2.MORPH_BLACKHAT,
        structuring_element,
        iterations=10,
    )

    img_plus_top_hat = cv2.add(
        img_grayscale,
        img_top_hat,
    )

    result = cv2.subtract(
        img_plus_top_hat,
        img_black_hat,
    )

    return result


def rotation_angle(lines_p):
    angles = []

    for line in lines_p:
        l = line[0].astype(int)

        doi = l[1] - l[3]
        ke = abs(l[0] - l[2])

        if ke == 0:
            continue

        angle = math.atan(doi / ke) * (180.0 / math.pi)

        if abs(angle) > 45:
            angle = (90 - abs(angle)) * angle / abs(angle)

        angles.append(angle)

    angles = [
        x for x in angles
        if abs(x) > 3 and abs(x) < 15
    ]

    if not angles:
        return 0

    return float(np.mean(angles))


def rotate_lp(img, angle):
    height, width = img.shape[:2]

    center = (width / 2, height / 2)

    rotation_matrix = cv2.getRotationMatrix2D(
        center,
        -angle,
        1.0,
    )

    return cv2.warpAffine(
        img,
        rotation_matrix,
        (width, height),
    )


def hough_transform(threshold_image, nol=6):
    h, w = threshold_image.shape[:2]

    lines_p = cv2.HoughLinesP(
        threshold_image,
        1,
        np.pi / 180,
        50,
        None,
        50,
        10,
    )

    if lines_p is None:
        return []

    distances = []

    for line in lines_p:
        l = line[0]

        d = math.sqrt(
            (l[0] - l[2]) ** 2
            + (l[1] - l[3]) ** 2
        )

        if d < 0.5 * max(h, w):
            d = 0

        distances.append(d)

    distances = np.array(distances).reshape(-1, 1, 1)

    lines_p = np.concatenate(
        [lines_p, distances],
        axis=2,
    )

    lines_p = sorted(
        lines_p,
        key=lambda x: x[0][-1],
        reverse=True,
    )[:nol]

    return lines_p