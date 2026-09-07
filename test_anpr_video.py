import os
import json

import cv2
import numpy as np

# ============================================================
# IMPORTANT:
# Import torch BEFORE paddle to avoid Windows DLL conflict
# ============================================================
import torch
from ultralytics import YOLO

import paddle
from paddleocr import PaddleOCR


# ============================================================
# CONFIG
# ============================================================

VIDEO_PATH = "./samples/sampel plat.mp4"
MODEL_PLAT_PATH = "./models/best_plat.pt"
OUTPUT_PATH = "./output/test_anpr.mp4"

# Confidence detector plat
PLATE_CONF = 0.05

# Hanya proses 300 frame pertama untuk testing
MAX_FRAMES = 300

# OCR dilakukan setiap N frame
OCR_INTERVAL = 5

os.makedirs("./output", exist_ok=True)


# ============================================================
# PADDLE CONFIG
# ============================================================

paddle.set_flags({
    "FLAGS_use_mkldnn": False
})


# ============================================================
# LOAD MODEL
# ============================================================

print("[INFO] Loading plate detector...")

model_plat = YOLO(MODEL_PLAT_PATH)

print("[INFO] Plate detector loaded.")
print(f"[INFO] Classes: {model_plat.names}")

print("[INFO] Loading PaddleOCR...")

ocr = PaddleOCR(
    lang="en"
)

print("[INFO] Models loaded.")


# ============================================================
# VIDEO
# ============================================================

cap = cv2.VideoCapture(VIDEO_PATH)

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

fps = cap.get(cv2.CAP_PROP_FPS)

if fps <= 0:
    fps = 25.0

total_frames = int(
    cap.get(cv2.CAP_PROP_FRAME_COUNT)
)

print(f"[INFO] Video      : {VIDEO_PATH}")
print(f"[INFO] Resolution : {width}x{height}")
print(f"[INFO] FPS        : {fps:.2f}")
print(f"[INFO] Total      : {total_frames}")
print(f"[INFO] Processing : first {MAX_FRAMES} frames")


# ============================================================
# OUTPUT VIDEO
# ============================================================

fourcc = cv2.VideoWriter_fourcc(*"mp4v")

writer = cv2.VideoWriter(
    OUTPUT_PATH,
    fourcc,
    fps,
    (width, height)
)

if not writer.isOpened():
    raise RuntimeError(
        f"Gagal membuat output video: {OUTPUT_PATH}"
    )


# ============================================================
# OCR CACHE
# ============================================================

# Menyimpan hasil OCR terakhir berdasarkan
# posisi plate detector.
#
# Untuk test awal kita belum menghubungkan
# plate dengan Vehicle ID.

ocr_cache = []

frame_idx = 0

total_plate_detections = 0
total_ocr_success = 0


# ============================================================
# MAIN LOOP
# ============================================================

while True:

    ret, frame = cap.read()

    if not ret:
        break

    frame_idx += 1

    # --------------------------------------------------------
    # LIMIT FRAME
    # --------------------------------------------------------

    if frame_idx > MAX_FRAMES:
        break

    # --------------------------------------------------------
    # PLATE DETECTION
    # --------------------------------------------------------

    results = model_plat(
        frame,
        conf=PLATE_CONF,
        verbose=False
    )[0]

    boxes = results.boxes

    current_detections = []

    if boxes is not None and len(boxes) > 0:

        total_plate_detections += len(boxes)

        for i, box in enumerate(boxes.xyxy):

            # ------------------------------------------------
            # BBOX
            # ------------------------------------------------

            x1, y1, x2, y2 = (
                box.cpu()
                .numpy()
                .astype(int)
            )

            conf = float(
                boxes.conf[i]
                .cpu()
                .numpy()
            )

            # ------------------------------------------------
            # CLAMP BBOX
            # ------------------------------------------------

            x1 = max(
                0,
                min(x1, width - 1)
            )

            y1 = max(
                0,
                min(y1, height - 1)
            )

            x2 = max(
                0,
                min(x2, width)
            )

            y2 = max(
                0,
                min(y2, height)
            )

            if x2 <= x1 or y2 <= y1:
                continue

            # ------------------------------------------------
            # CROP PLATE
            # ------------------------------------------------

            plate_crop = frame[
                y1:y2,
                x1:x2
            ]

            if plate_crop.size == 0:
                continue

            text = ""
            ocr_conf = 0.0

            # ------------------------------------------------
            # OCR
            # ------------------------------------------------

            if frame_idx % OCR_INTERVAL == 0:

                try:

                    result = ocr.predict(
                        plate_crop
                    )

                    for res in result:

                        # ------------------------------------
                        # PaddleOCR 3.x
                        # ------------------------------------

                        data = res.json

                        if callable(data):
                            data = data()

                        if isinstance(data, str):
                            data = json.loads(data)

                        # ------------------------------------
                        # OCR RESULT
                        # ------------------------------------

                        rec_texts = data.get(
                            "rec_texts",
                            []
                        )

                        rec_scores = data.get(
                            "rec_scores",
                            []
                        )

                        if rec_texts:

                            text = "".join(
                                rec_texts
                            ).strip()

                            if rec_scores:

                                ocr_conf = max(
                                    rec_scores
                                )

                            if text:
                                total_ocr_success += 1

                        break

                except Exception as e:

                    print(
                        f"[OCR ERROR] "
                        f"frame={frame_idx}: {e}"
                    )

            # ------------------------------------------------
            # CACHE
            # ------------------------------------------------

            if text:

                current_detections.append({

                    "bbox": (
                        x1,
                        y1,
                        x2,
                        y2
                    ),

                    "plate": text,

                    "ocr_conf": ocr_conf,

                    "det_conf": conf,

                })

            else:

                # --------------------------------------------
                # Cari hasil OCR sebelumnya yang paling dekat
                # --------------------------------------------

                best_previous = None

                best_distance = float("inf")

                cx = (
                    x1 + x2
                ) / 2

                cy = (
                    y1 + y2
                ) / 2

                for prev in ocr_cache:

                    px1, py1, px2, py2 = (
                        prev["bbox"]
                    )

                    pcx = (
                        px1 + px2
                    ) / 2

                    pcy = (
                        py1 + py2
                    ) / 2

                    distance = np.sqrt(
                        (cx - pcx) ** 2 +
                        (cy - pcy) ** 2
                    )

                    if distance < best_distance:

                        best_distance = distance
                        best_previous = prev

                # --------------------------------------------
                # Gunakan cache jika cukup dekat
                # --------------------------------------------

                if (
                    best_previous is not None
                    and best_distance < 100
                ):

                    current_detections.append({

                        "bbox": (
                            x1,
                            y1,
                            x2,
                            y2
                        ),

                        "plate": (
                            best_previous["plate"]
                        ),

                        "ocr_conf": (
                            best_previous["ocr_conf"]
                        ),

                        "det_conf": conf,

                    })

    # ========================================================
    # UPDATE CACHE
    # ========================================================

    ocr_cache = current_detections


    # ========================================================
    # DRAW
    # ========================================================

    for detection in current_detections:

        x1, y1, x2, y2 = (
            detection["bbox"]
        )

        plate = detection["plate"]

        ocr_conf = detection["ocr_conf"]

        det_conf = detection["det_conf"]

        # ----------------------------------------------------
        # BBOX
        # ----------------------------------------------------

        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            2
        )

        # ----------------------------------------------------
        # LABEL
        # ----------------------------------------------------

        if plate:

            label = (
                f"{plate} "
                f"OCR:{ocr_conf:.2f} "
                f"DET:{det_conf:.2f}"
            )

        else:

            label = (
                f"PLATE "
                f"DET:{det_conf:.2f}"
            )

        # ----------------------------------------------------
        # LABEL SIZE
        # ----------------------------------------------------

        (
            tw,
            th
        ), _ = cv2.getTextSize(
            label,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            2
        )

        label_y = max(
            y1 - 8,
            th + 5
        )

        # ----------------------------------------------------
        # LABEL BACKGROUND
        # ----------------------------------------------------

        cv2.rectangle(
            frame,
            (
                x1,
                label_y - th - 5
            ),
            (
                x1 + tw + 5,
                label_y + 3
            ),
            (0, 255, 0),
            -1
        )

        # ----------------------------------------------------
        # TEXT
        # ----------------------------------------------------

        cv2.putText(
            frame,
            label,
            (
                x1 + 2,
                label_y
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 0, 0),
            2
        )


    # ========================================================
    # INFO
    # ========================================================

    info = (
        f"Frame: {frame_idx}/{MAX_FRAMES} | "
        f"Plates: {len(current_detections)}"
    )

    cv2.putText(
        frame,
        info,
        (20, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2
    )


    # ========================================================
    # WRITE
    # ========================================================

    writer.write(frame)


    # ========================================================
    # PROGRESS
    # ========================================================

    if frame_idx % 50 == 0:

        print(
            f"[PROGRESS] "
            f"{frame_idx}/{min(total_frames, MAX_FRAMES)} "
            f"| Current plates: "
            f"{len(current_detections)}"
        )


# ============================================================
# CLEANUP
# ============================================================

cap.release()
writer.release()


# ============================================================
# SUMMARY
# ============================================================

print()
print("=" * 60)
print("[SELESAI]")
print("=" * 60)

print(
    f"Frames processed       : {frame_idx}"
)

print(
    f"Total plate detections : {total_plate_detections}"
)

print(
    f"OCR successful         : {total_ocr_success}"
)

print(
    f"Output                 : {OUTPUT_PATH}"
)

print("=" * 60)