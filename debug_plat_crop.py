import os
import cv2
from ultralytics import YOLO


# ============================================================
# CONFIG
# ============================================================

VIDEO_PATH = "./samples/sampel plat.mp4"
PLATE_MODEL_PATH = "./models/best_plat.pt"

OUTPUT_DIR = "./output/debug_plat_crops"

PLATE_CONF = 0.20

# Maksimal crop yang disimpan
MAX_CROPS = 30

# Proses maksimal frame
MAX_FRAMES = 300

# Jarak tambahan di sekitar bbox
PADDING = 5


# ============================================================
# SETUP
# ============================================================

os.makedirs(OUTPUT_DIR, exist_ok=True)

print("=" * 60)
print("DEBUG PLATE CROP")
print("=" * 60)

print(f"[INFO] Video       : {VIDEO_PATH}")
print(f"[INFO] Model       : {PLATE_MODEL_PATH}")
print(f"[INFO] Output      : {OUTPUT_DIR}")
print(f"[INFO] Confidence  : {PLATE_CONF}")
print(f"[INFO] Max crops   : {MAX_CROPS}")
print(f"[INFO] Max frames  : {MAX_FRAMES}")
print("=" * 60)


# ============================================================
# LOAD MODEL
# ============================================================

print("[INFO] Loading plate detector...")

model = YOLO(PLATE_MODEL_PATH)

print("[INFO] Plate detector loaded.")
print(f"[INFO] Classes: {model.names}")


# ============================================================
# OPEN VIDEO
# ============================================================

cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():
    raise RuntimeError(
        f"Gagal membuka video: {VIDEO_PATH}"
    )

fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

print()
print(f"[INFO] Resolution : {width}x{height}")
print(f"[INFO] FPS        : {fps:.2f}")
print(f"[INFO] Total      : {total_frames}")
print()


# ============================================================
# PROCESS
# ============================================================

frame_idx = 0
crop_count = 0
total_detections = 0

while True:

    ret, frame = cap.read()

    if not ret:
        break

    frame_idx += 1

    if frame_idx > MAX_FRAMES:
        break

    # --------------------------------------------------------
    # YOLO PLATE DETECTION
    # --------------------------------------------------------

    results = model.predict(
        source=frame,
        conf=PLATE_CONF,
        verbose=False
    )[0]

    if results.boxes is None:
        continue

    # --------------------------------------------------------
    # LOOP DETECTIONS
    # --------------------------------------------------------

    for det_idx, box in enumerate(results.boxes):

        if crop_count >= MAX_CROPS:
            break

        # BBOX
        x1, y1, x2, y2 = map(
            int,
            box.xyxy[0].tolist()
        )

        conf = float(box.conf[0])

        cls_id = int(box.cls[0])

        class_name = model.names.get(
            cls_id,
            str(cls_id)
        )

        total_detections += 1

        # ----------------------------------------------------
        # ADD PADDING
        # ----------------------------------------------------

        crop_x1 = max(0, x1 - PADDING)
        crop_y1 = max(0, y1 - PADDING)

        crop_x2 = min(
            frame.shape[1],
            x2 + PADDING
        )

        crop_y2 = min(
            frame.shape[0],
            y2 + PADDING
        )

        # ----------------------------------------------------
        # CROP
        # ----------------------------------------------------

        plate_crop = frame[
            crop_y1:crop_y2,
            crop_x1:crop_x2
        ]

        if plate_crop.size == 0:
            continue

        crop_h, crop_w = plate_crop.shape[:2]

        # ----------------------------------------------------
        # SAVE ORIGINAL CROP
        # ----------------------------------------------------

        filename = (
            f"crop_{crop_count:03d}"
            f"_frame_{frame_idx:04d}"
            f"_{crop_w}x{crop_h}"
            f"_conf_{conf:.2f}.jpg"
        )

        output_path = os.path.join(
            OUTPUT_DIR,
            filename
        )

        cv2.imwrite(
            output_path,
            plate_crop
        )

        # ----------------------------------------------------
        # SAVE UPSCALED VERSION
        # ----------------------------------------------------
        #
        # Ini cuma untuk mempermudah mata kita melihat.
        # TIDAK dipakai untuk OCR.
        #

        scale = 5

        upscaled = cv2.resize(
            plate_crop,
            None,
            fx=scale,
            fy=scale,
            interpolation=cv2.INTER_CUBIC
        )

        upscaled_filename = (
            f"crop_{crop_count:03d}"
            f"_frame_{frame_idx:04d}"
            f"_{crop_w}x{crop_h}"
            f"_conf_{conf:.2f}_UP.jpg"
        )

        upscaled_path = os.path.join(
            OUTPUT_DIR,
            upscaled_filename
        )

        cv2.imwrite(
            upscaled_path,
            upscaled
        )

        # ----------------------------------------------------
        # PRINT INFO
        # ----------------------------------------------------

        aspect_ratio = crop_w / crop_h

        print(
            f"[CROP {crop_count + 1:02d}] "
            f"frame={frame_idx:04d} | "
            f"bbox={crop_w}x{crop_h} | "
            f"ratio={aspect_ratio:.2f} | "
            f"conf={conf:.2f} | "
            f"class={class_name}"
        )

        crop_count += 1

    # --------------------------------------------------------
    # PROGRESS
    # --------------------------------------------------------

    if frame_idx % 25 == 0:
        print(
            f"[PROGRESS] "
            f"frame={frame_idx}/{MAX_FRAMES} | "
            f"detections={total_detections} | "
            f"saved={crop_count}"
        )

    # --------------------------------------------------------
    # STOP IF ENOUGH CROPS
    # --------------------------------------------------------

    if crop_count >= MAX_CROPS:
        print()
        print(
            f"[INFO] Sudah mendapatkan "
            f"{MAX_CROPS} crop."
        )
        break


# ============================================================
# CLEANUP
# ============================================================

cap.release()


# ============================================================
# SUMMARY
# ============================================================

print()
print("=" * 60)
print("SELESAI")
print("=" * 60)

print(f"Frames processed : {frame_idx}")
print(f"Detections       : {total_detections}")
print(f"Crops saved      : {crop_count}")
print(f"Output directory : {OUTPUT_DIR}")

print("=" * 60)