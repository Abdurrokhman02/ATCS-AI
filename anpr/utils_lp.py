import cv2
import numpy as np

from .preprocess import (
    preprocess,
    hough_transform,
    rotation_angle,
    rotate_lp,
)


ALPHA_DICT = {
    0: "A",
    1: "B",
    2: "C",
    3: "D",
    4: "E",
    5: "F",
    6: "G",
    7: "H",
    8: "K",
    9: "L",
    10: "M",
    11: "N",
    12: "P",
    13: "R",
    14: "S",
    15: "T",
    16: "U",
    17: "V",
    18: "X",
    19: "Y",
    20: "Z",
    21: "0",
    22: "1",
    23: "2",
    24: "3",
    25: "4",
    26: "5",
    27: "6",
    28: "7",
    29: "8",
    30: "9",
    31: "Background",
}


def character_recog_cnn(model, img):
    img_roi = cv2.resize(
        img,
        (28, 28),
        interpolation=cv2.INTER_AREA,
    )

    img_roi = img_roi.reshape((28, 28, 1))
    img_roi = np.expand_dims(
        np.array(img_roi),
        axis=0,
    )

    result = model.predict(
        img_roi,
        verbose=0,
    )

    result_idx = np.argmax(
        result,
        axis=1,
    )

    return ALPHA_DICT[int(result_idx[0])]


def crop_n_rotate_lp(
    source_img,
    x1,
    y1,
    x2,
    y2,
):
    h_img, w_img = source_img.shape[:2]

    # Clamp bbox supaya tidak keluar frame
    x1 = max(0, min(int(x1), w_img - 1))
    y1 = max(0, min(int(y1), h_img - 1))
    x2 = max(x1 + 1, min(int(x2), w_img))
    y2 = max(y1 + 1, min(int(y2), h_img))

    w = x2 - x1
    h = y2 - y1

    if h <= 0:
        return None, None, None

    ratio = w / h

    # Rasio dari repo lama
    if not (0.8 <= ratio <= 1.5 or 3.5 <= ratio <= 6.5):
        return None, None, None

    cropped_lp = source_img[
        y1:y2,
        x1:x2,
    ].copy()

    if cropped_lp.size == 0:
        return None, None, None

    _, img_thresh_plate = preprocess(
        cropped_lp
    )

    canny_image = cv2.Canny(
        img_thresh_plate,
        250,
        255,
    )

    kernel = np.ones(
        (3, 3),
        np.uint8,
    )

    dilated_image = cv2.dilate(
        canny_image,
        kernel,
        iterations=2,
    )

    lines_p = hough_transform(
        dilated_image,
        nol=6,
    )

    angle = rotation_angle(lines_p)

    rotate_thresh = rotate_lp(
        img_thresh_plate,
        angle,
    )

    lp_rotated = rotate_lp(
        cropped_lp,
        angle,
    )

    return (
        angle,
        rotate_thresh,
        lp_rotated,
    )