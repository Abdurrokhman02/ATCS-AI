# test_crop.py
import os
import cv2
import numpy as np
from ultralytics import YOLO

# Buat folder khusus buat menampung hasil crop
OUTPUT_CROP_DIR = "./output/debug_plat_crops"
os.makedirs(OUTPUT_CROP_DIR, exist_ok=True)

# Load model plat nomor
MODEL_PLAT_PATH = "./best_plat.pt"
model_plat = YOLO(MODEL_PLAT_PATH)

# Ambil video sampel
SOURCE_VIDEO = "./samples/sampel 1.mp4"
cap = cv2.VideoCapture(SOURCE_VIDEO)

frame_idx = 0
saved_count = 0

print("[INFO] Memulai ekstraksi crop plat, silakan tunggu...")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Cek tiap 15 frame sekali biar gak kebanyakan sampah gambar
    if frame_idx % 15 == 0:
        results = model_plat(frame, conf=0.25, verbose=False)[0]
        boxes = getattr(results, "boxes", None)

        if boxes is not None and len(boxes) > 0:
            for i, box in enumerate(boxes.xyxy):
                x1, y1, x2, y2 = box.cpu().numpy().astype(int)
                conf = float(boxes.conf[i].cpu().numpy())

                h, w = frame.shape[:2]
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)

                if x2 > x1 and y2 > y1:
                    crop_plat = frame[y1:y2, x1:x2]
                    
                    # Simpan hasil crop ke folder lokal
                    filename = os.path.join(OUTPUT_CROP_DIR, f"frame_{frame_idx}_conf_{conf:.2f}.jpg")
                    cv2.imwrite(filename, crop_plat)
                    saved_count += 1

    frame_idx += 1
    # Batasi sampai 300 frame pertama aja buat sampel pengecekan
    if frame_idx > 300:
        break

cap.release()
print(f"[SELESAI] Berhasil menyimpan {saved_count} gambar crop plat di folder: {OUTPUT_CROP_DIR}")