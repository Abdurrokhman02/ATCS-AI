import os
import warnings

# ==========================================
# 0. FIX LINGKUNGAN ARCH LINUX / WAYLAND
# ==========================================
# Paksa Qt OpenCV menggunakan backend X11/XWayland agar tidak crash di Hyprland
os.environ["QT_QPA_PLATFORM"] = "xcb"
# Sembunyikan log font Qt yang berisik
os.environ["QT_LOGGING_RULES"] = "*.warning=false"
# Sembunyikan peringatan update API ByteTrack
warnings.filterwarnings("ignore", category=FutureWarning)

import cv2
import numpy as np
from collections import defaultdict
from ultralytics import YOLO
import supervision as sv

# ==========================================
# 1. KONFIGURASI KAMERA (Fisik / Lapangan)
# ==========================================
CAM_CONFIG = {
    "camera_id": "CCTV_Jalan_01",
    "video_source": "./samples/sampleuji.mp4", # Ganti ke '0' jika pakai webcam langsung
    "jarak_garis_meter": 20.0,
    "free_flow_speed_kmh": 40.0,
    "panjang_segmen_meter": 50.0,
    "jumlah_lajur": 2
}

# ==========================================
# 2. STATE FITUR (TOGGLE ON / OFF)
# ==========================================
flags = {
    "DETEKSI": True,     # Bounding Box
    "COUNTING": True,    # Hitung Kendaraan & Garis
    "KECEPATAN": True,   # Rata-Rata Kecepatan
    "KEMACETAN": True    # Tingkat Kemacetan
}

# Koordinat Tombol Menu (X_start, Y_start, X_end, Y_end)
BUTTONS = {
    "DETEKSI": (20, 15, 150, 45),
    "COUNTING": (160, 15, 290, 45),
    "KECEPATAN": (300, 15, 430, 45),
    "KEMACETAN": (440, 15, 580, 45)
}

def handle_mouse_click(event, x, y, flags_param, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        for feature_name, (x1, y1, x2, y2) in BUTTONS.items():
            if x1 <= x <= x2 and y1 <= y <= y2:
                flags[feature_name] = not flags[feature_name]
                print(f"🔄 Fitur {feature_name} : {'AKTIF' if flags[feature_name] else 'NONAKTIF'}")

# ==========================================
# 3. INISIALISASI VIDEO & DYNAMIC AREA
# ==========================================
cap = cv2.VideoCapture(CAM_CONFIG["video_source"])
if not cap.isOpened():
    print("❌ Error: Video tidak ditemukan!")
    exit()

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS) or 25.0

# Garis Otomatis Maksimal (Kiri 0 sampai Kanan penuh sesuai resolusi video)
garis1_y = int(height * 0.30) # Garis Entry 30% dari atas
garis2_y = int(height * 0.75) # Garis Exit 75% dari atas

L1_START, L1_END = sv.Point(0, garis1_y), sv.Point(width, garis1_y)
L2_START, L2_END = sv.Point(0, garis2_y), sv.Point(width, garis2_y)

LINE1 = sv.LineZone(start=L1_START, end=L1_END)
LINE2 = sv.LineZone(start=L2_START, end=L2_END)

MAX_ROI_POLYGON = np.array([
    [0, garis1_y],
    [width, garis1_y],
    [width, garis2_y],
    [0, garis2_y]
])

try:
    roi_zone = sv.PolygonZone(polygon=MAX_ROI_POLYGON, frame_resolution_wh=(width, height))
except TypeError:
    roi_zone = sv.PolygonZone(polygon=MAX_ROI_POLYGON)

# Output video hasil deteksi
OUTPUT_VIDEO_PATH = "./outputujivideo.mp4"
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out_video = cv2.VideoWriter(OUTPUT_VIDEO_PATH, fourcc, fps, (width, height))

# ==========================================
# 4. INISIALISASI MODEL AI & TRACKER
# ==========================================
model = YOLO('best.pt')
# Setup Tracker AI (wajib ada agar sistem ingat kendaraan dan ngitung kecepatan)
tracker = sv.ByteTrack(track_activation_threshold=0.25) 

box_annotator = sv.BoxAnnotator(thickness=2)
label_annotator = sv.LabelAnnotator(text_thickness=1, text_scale=0.5)
line1_annotator = sv.LineZoneAnnotator(thickness=2, text_thickness=1, color=sv.Color.GREEN)
line2_annotator = sv.LineZoneAnnotator(thickness=2, text_thickness=1, color=sv.Color.RED)

CLASS_NAMES = {0: "Orang", 1: "Mobil", 2: "Motor", 3: "Bus", 4: "Truk"}

# Data Ingatan Kendaraan
timestamp_line1 = {}
timestamp_line2 = {}
travel_times = []
speeds_kmh = []
class_counts = defaultdict(int)

# Setup Window GUI
window_name = f"CCTV Traffic Monitor - {CAM_CONFIG['camera_id']}"
cv2.namedWindow(window_name)
cv2.setMouseCallback(window_name, handle_mouse_click)

# ==========================================
# 5. LOOPING VIDEO UTAMA
# ==========================================
frame_idx = 0
print("✅ Memulai AI ATCS System... (Tekan 'q' untuk keluar)")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    t_sekarang = frame_idx / fps

    # Inference YOLO
    results = model(frame, conf=0.30, verbose=False)[0]
    detections = sv.Detections.from_ultralytics(results)
    
    # Filter hanya kendaraan (best.pt: Car=1, Motorcycle=2, Bus=3, Truck=4; Person=0 dilewati)
    detections = detections[np.isin(detections.class_id, [1, 2, 3, 4])]
    detections = tracker.update_with_detections(detections)

    # Cek objek menabrak garis (Masuk / Keluar)
    c1_in, c1_out = LINE1.trigger(detections=detections)
    c2_in, c2_out = LINE2.trigger(detections=detections)
    crossed_l1 = c1_in | c1_out
    crossed_l2 = c2_in | c2_out

    for i, (tid, cid) in enumerate(zip(detections.tracker_id, detections.class_id)):
        # Hitung Kendaraan Masuk Pertama Kali (Garis 1 atau Garis 2)
        if crossed_l1[i] and tid not in timestamp_line1:
            timestamp_line1[tid] = t_sekarang
            class_counts[cid] += 1
        if crossed_l2[i] and tid not in timestamp_line2:
            timestamp_line2[tid] = t_sekarang
            if not crossed_l1[i]: # Supaya tidak dihitung double
                class_counts[cid] += 1

        # Kalkulasi Hitung Kecepatan Dua Arah
        if crossed_l2[i] and tid in timestamp_line1 and tid not in timestamp_line2:
            tt = t_sekarang - timestamp_line1[tid]
            if tt > 0.2:
                speeds_kmh.append((CAM_CONFIG["jarak_garis_meter"] / tt) * 3.6)
        elif crossed_l1[i] and tid in timestamp_line2 and tid not in timestamp_line1:
            tt = t_sekarang - timestamp_line2[tid]
            if tt > 0.2:
                speeds_kmh.append((CAM_CONFIG["jarak_garis_meter"] / tt) * 3.6)

    # Siapkan Canvas Tampilan
    annotated = frame.copy()

    # --- RENDER FITUR 1: DETEKSI ---
    if flags["DETEKSI"]:
        annotated = box_annotator.annotate(scene=annotated, detections=detections)
        # Bikin tulisan lebih bersih tanpa menampilkan nomor ID kendaraan
        labels = [f"{CLASS_NAMES.get(cid,'?')}" for cid in detections.class_id]
        annotated = label_annotator.annotate(scene=annotated, detections=detections, labels=labels)

    # --- RENDER FITUR 2: COUNTING (Garis) ---
    if flags["COUNTING"]:
        try:
            annotated = line1_annotator.annotate(annotated, line_zone=LINE1)
            annotated = line2_annotator.annotate(annotated, line_zone=LINE2)
        except TypeError: # Fallback untuk supervision versi di bawah 0.19.0
            annotated = line1_annotator.annotate(annotated, line_counter=LINE1)
            annotated = line2_annotator.annotate(annotated, line_counter=LINE2)

    # --- RENDER TOMBOL INTERAKTIF UI ---
    for btn_name, (x1, y1, x2, y2) in BUTTONS.items():
        is_active = flags[btn_name]
        btn_color = (0, 180, 0) if is_active else (60, 60, 60) # Hijau = ON, Abu = OFF
        cv2.rectangle(annotated, (x1, y1), (x2, y2), btn_color, -1)
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (255, 255, 255), 1)
        
        status_txt = "ON" if is_active else "OFF"
        cv2.putText(annotated, f"{btn_name}: {status_txt}", (x1 + 8, y1 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1)

    # --- KALKULASI DATA DASHBOARD ---
    total_kendaraan = sum(class_counts.values())
    avg_speed = float(np.mean(speeds_kmh[-20:])) if speeds_kmh else 0.0

    in_roi_mask = roi_zone.trigger(detections=detections)
    jumlah_di_roi = int(in_roi_mask.sum()) if len(in_roi_mask) else 0
    density = (jumlah_di_roi / (CAM_CONFIG["panjang_segmen_meter"] / 1000.0)) / CAM_CONFIG["jumlah_lajur"]
    
    # Penentuan Kategori Macet
    if density <= 10:
        status_macet, color_macet = "LANCAR", (0, 255, 0)
    elif density <= 22:
        status_macet, color_macet = "RAMAI / SEDANG", (0, 255, 255)
    elif density <= 35:
        status_macet, color_macet = "PADAT", (0, 165, 255)
    else:
        status_macet, color_macet = "MACET TOTAL", (0, 0, 255)

    # --- RENDER PANEL DASHBOARD ---
    overlay = annotated.copy()
    cv2.rectangle(overlay, (20, 60), (400, 200), (0, 0, 0), -1)
    annotated = cv2.addWeighted(overlay, 0.7, annotated, 0.3, 0)

    cv2.putText(annotated, "DISHUB - TRAFFIC MONITORING", (35, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)
    
    # Teks Dashboard menyesuaikan ON/OFF Fitur
    val_count = f"{total_kendaraan} unit" if flags["COUNTING"] else "[NONAKTIF]"
    val_speed = f"{avg_speed:.1f} km/jam" if flags["KECEPATAN"] else "[NONAKTIF]"
    val_macet = status_macet if flags["KEMACETAN"] else "[NONAKTIF]"
    c_macet = color_macet if flags["KEMACETAN"] else (150, 150, 150)

    cv2.putText(annotated, f"1. Total Kendaraan : {val_count}", (35, 115), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.putText(annotated, f"2. Rata2 Kecepatan : {val_speed}", (35, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 255, 200), 1)
    cv2.putText(annotated, f"3. Status Kemacetan : {val_macet}", (35, 165), cv2.FONT_HERSHEY_SIMPLEX, 0.5, c_macet, 2)

    cv2.imshow(window_name, annotated)
    out_video.write(annotated)

    # ==========================================
    # CLEANUP MEMORY (Anti RAM jebol jalan 24/7)
    # Hapus data mobil yang sudah lewat lebih dari 10 detik
    # ==========================================
    if frame_idx % (int(fps) * 5) == 0:  # Bersihkan setiap 5 detik
        keys_to_del_1 = [k for k, v in timestamp_line1.items() if t_sekarang - v > 10]
        keys_to_del_2 = [k for k, v in timestamp_line2.items() if t_sekarang - v > 10]
        for k in keys_to_del_1: del timestamp_line1[k]
        for k in keys_to_del_2: del timestamp_line2[k]
        
        # Jaga agar list memori angka tidak membengkak tanpa batas
        if len(travel_times) > 100: travel_times = travel_times[-50:]
        if len(speeds_kmh) > 100: speeds_kmh = speeds_kmh[-50:]

    # Keyboard Control Shortcuts
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'): break
    elif key == ord('1'): flags["DETEKSI"] = not flags["DETEKSI"]
    elif key == ord('2'): flags["COUNTING"] = not flags["COUNTING"]
    elif key == ord('3'): flags["KECEPATAN"] = not flags["KECEPATAN"]
    elif key == ord('4'): flags["KEMACETAN"] = not flags["KEMACETAN"]

    frame_idx += 1

cap.release()
out_video.release()
cv2.destroyAllWindows()
print(f"✅ Video hasil tersimpan di: {OUTPUT_VIDEO_PATH}")
print("🛑 Selesai.")