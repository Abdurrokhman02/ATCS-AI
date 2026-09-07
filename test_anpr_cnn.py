import os

# PENTING:
# torch harus di-import sebelum paddle kalau environment
# masih menggunakan PaddleOCR juga.
import torch

import cv2
import numpy as np

from ultralytics import YOLO

from anpr.cnn_model import CNN_Model
from anpr.utils_lp import (
    crop_n_rotate_lp,
    character_recog_cnn,
)


VIDEO_PATH = "./samples/sampel plat.mp4"
PLATE_MODEL_PATH = "./models/best_plat.pt"
CHAR_WEIGHTS_PATH = "./anpr/weights/weight.h5"

OUTPUT_PATH = "./output/test_anpr_cnn.mp4"

PLATE_CONF = 0.25
MAX_FRAMES = 300


MIN_CHAR = 0.01
MAX_CHAR = 0.09


os.makedirs("./output", exist_ok=True)


# =========================================================
# LOAD YOLOv11 PLATE DETECTOR
# =========================================================

print("[INFO] Loading YOLOv11 plate detector...")

plate_model = YOLO(
    PLATE_MODEL_PATH
)

print("[INFO] Plate model loaded.")
print("[INFO] Classes:", plate_model.names)


# =========================================================
# LOAD CNN CHARACTER RECOGNIZER
# =========================================================

print("[INFO] Loading CNN character recognizer...")

cnn = CNN_Model()

cnn.model.load_weights(
    CHAR_WEIGHTS_PATH
)

print("[INFO] CNN weights loaded.")


# =========================================================
# VIDEO
# =========================================================

cap = cv2.VideoCapture(
    VIDEO_PATH
)

if not cap.isOpened():
    raise RuntimeError(
        f"Gagal membuka video: {VIDEO_PATH}"
    )


width = int(
    cap.get(cv2.CAP_PROP_FRAME_WIDTH)
)

height = int(
    cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
)

fps = cap.get(
    cv2.CAP_PROP_FPS
)

total_frames = int(
    cap.get(cv2.CAP_PROP_FRAME_COUNT)
)


print(f"[INFO] Video      : {VIDEO_PATH}")
print(f"[INFO] Resolution : {width}x{height}")
print(f"[INFO] FPS        : {fps:.2f}")
print(f"[INFO] Total      : {total_frames}")
print(f"[INFO] Processing : first {MAX_FRAMES} frames")


fourcc = cv2.VideoWriter_fourcc(
    *"mp4v"
)

writer = cv2.VideoWriter(
    OUTPUT_PATH,
    fourcc,
    fps,
    (width, height),
)


# =========================================================
# LOOP
# =========================================================

frame_idx = 0
total_plate_detections = 0
total_recognized = 0


while True:

    ret, frame = cap.read()

    if not ret:
        break

    frame_idx += 1

    if frame_idx > MAX_FRAMES:
        break

    # -----------------------------------------------------
    # YOLOv11 PLATE DETECTION
    # -----------------------------------------------------

    result = plate_model.predict(
        source=frame,
        conf=PLATE_CONF,
        verbose=False,
    )[0]

    if result.boxes is None:
        writer.write(frame)
        continue

    boxes = result.boxes

    for box in boxes:

        xyxy = box.xyxy[0].cpu().numpy()

        conf = float(
            box.conf[0].cpu().item()
        )

        x1, y1, x2, y2 = map(
            int,
            xyxy,
        )

        total_plate_detections += 1

        # -------------------------------------------------
        # DEBUG INFO UKURAN PLAT
        # -------------------------------------------------

        plate_w = x2 - x1
        plate_h = y2 - y1

        # -------------------------------------------------
        # CROP + ROTATE
        # -------------------------------------------------

        (
            angle,
            rotate_thresh,
            lp_rotated,
        ) = crop_n_rotate_lp(
            frame,
            x1,
            y1,
            x2,
            y2,
        )

        # Draw detector bbox regardless
        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            (0, 255, 255),
            2,
        )

        cv2.putText(
            frame,
            f"PLATE {conf:.2f} {plate_w}x{plate_h}",
            (x1, max(20, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 255),
            1,
        )

        if (
            rotate_thresh is None
            or lp_rotated is None
        ):
            continue

        # -------------------------------------------------
        # CHARACTER SEGMENTATION
        # -------------------------------------------------

        lp_rotated_copy = lp_rotated.copy()

        contours, _ = cv2.findContours(
            rotate_thresh,
            cv2.RETR_CCOMP,
            cv2.CHAIN_APPROX_SIMPLE,
        )

        contours = sorted(
            contours,
            key=cv2.contourArea,
            reverse=True,
        )[:17]

        char_x = []

        plate_h2, plate_w2 = (
            lp_rotated_copy.shape[:2]
        )

        roi_area = plate_h2 * plate_w2

        for cnt in contours:

            x, y, w, h = cv2.boundingRect(
                cnt
            )

            if h <= 0:
                continue

            ratio_char = w / h
            char_area = w * h

            if (
                MIN_CHAR * roi_area
                < char_area
                < MAX_CHAR * roi_area
                and 0.25 < ratio_char < 0.7
            ):
                char_x.append(
                    [x, y, w, h]
                )

        if not char_x:
            continue

        char_x = np.array(char_x)

        # -------------------------------------------------
        # CHARACTER ORDER
        # -------------------------------------------------

        threshold_12line = (
            char_x[:, 1].min()
            + char_x[:, 3].mean() / 2
        )

        char_x = sorted(
            char_x,
            key=lambda x: x[0],
        )

        first_line = ""
        second_line = ""

        # -------------------------------------------------
        # CNN CHARACTER RECOGNITION
        # -------------------------------------------------

        for char in char_x:

            x, y, w, h = char

            x = int(x)
            y = int(y)
            w = int(w)
            h = int(h)

            img_roi = rotate_thresh[
                y:y + h,
                x:x + w,
            ]

            text = character_recog_cnn(
                cnn.model,
                img_roi,
            )

            if text == "Background":
                text = ""

            if y < threshold_12line:
                first_line += text
            else:
                second_line += text

        plate_text = (
            first_line
            + second_line
        )

        if plate_text:

            total_recognized += 1

            cv2.putText(
                frame,
                plate_text,
                (x1, y2 + 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 0),
                2,
            )

            print(
                f"[PLATE] frame={frame_idx} "
                f"bbox={plate_w}x{plate_h} "
                f"conf={conf:.2f} "
                f"TEXT={plate_text}"
            )

    writer.write(frame)

    if frame_idx % 25 == 0:
        print(
            f"[PROGRESS] "
            f"{frame_idx}/{MAX_FRAMES} "
            f"| detections={total_plate_detections} "
            f"| recognized={total_recognized}"
        )


cap.release()
writer.release()


print()
print("[SELESAI]")
print(
    "Frames processed       :",
    frame_idx,
)
print(
    "Total plate detections :",
    total_plate_detections,
)
print(
    "CNN recognized         :",
    total_recognized,
)
print(
    "Output                 :",
    OUTPUT_PATH,
)
