# main.py
import os
import warnings
import cv2
import numpy as np
from collections import defaultdict
from ultralytics import YOLO
import supervision as sv

# --- IMPORT MODULE FITUR ---
from config import CAM_CONFIG, FLAGS, BUTTONS, CLASS_NAMES
from modules.fitur_counting import catat_dan_hitung, get_total_akumulasi, bersihkan_memori
from modules.fitur_kecepatan import hitung_kecepatan_kendaraan, get_rata_rata_kecepatan
from modules.fitur_kemacetan import cek_status_kemacetan

# Fix Environment Linux
os.environ["QT_QPA_PLATFORM"] = "xcb"
os.environ["QT_LOGGING_RULES"] = "*.warning=false"
warnings.filterwarnings("ignore", category=FutureWarning)

# --- SETUP MOUSE CLICK UI ---
def handle_mouse_click(event, x, y, flags_param, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        for feature_name, (x1, y1, x2, y2) in BUTTONS.items():
            if x1 <= x <= x2 and y1 <= y <= y2:
                FLAGS[feature_name] = not FLAGS[feature_name]

# --- INISIALISASI KAMERA & AREA ---
cap = cv2.VideoCapture(CAM_CONFIG["video_source"])
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS) or 25.0

garis1_y, garis2_y = int(height * 0.30), int(height * 0.75)
LINE1 = sv.LineZone(start=sv.Point(0, garis1_y), end=sv.Point(width, garis1_y))
LINE2 = sv.LineZone(start=sv.Point(0, garis2_y), end=sv.Point(width, garis2_y))

MAX_ROI = np.array([[0, garis1_y], [width, garis1_y], [width, garis2_y], [0, garis2_y]])
try:
    roi_zone = sv.PolygonZone(polygon=MAX_ROI, frame_resolution_wh=(width, height))
except TypeError:
    roi_zone = sv.PolygonZone(polygon=MAX_ROI)

# --- INISIALISASI AI & DATA STATE ---
model = YOLO('./models/yolov8m.pt')
tracker = sv.ByteTrack(track_activation_threshold=0.25)

box_annotator = sv.BoxAnnotator(thickness=2)
label_annotator = sv.LabelAnnotator(text_thickness=1, text_scale=0.5)
l1_annotator = sv.LineZoneAnnotator(thickness=2, color=sv.Color.GREEN)
l2_annotator = sv.LineZoneAnnotator(thickness=2, color=sv.Color.RED)

# State Variables
timestamp_l1, timestamp_l2 = {}, {}
speeds_kmh, travel_times = [], []
class_counts = defaultdict(int)

# Setup Window
cv2.namedWindow("CCTV Traffic Monitor")
cv2.setMouseCallback("CCTV Traffic Monitor", handle_mouse_click)

# ==========================================
# LOOPING UTAMA
# ==========================================
frame_idx = 0
while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break
    t_sekarang = frame_idx / fps

    # 1. AI Inference
    results = model(frame, conf=0.30, verbose=False)[0]
    detections = sv.Detections.from_ultralytics(results)
    detections = detections[np.isin(detections.class_id, [2, 3, 5, 7])]
    detections = tracker.update_with_detections(detections)

    # 2. Cek Trigger Garis
    c1_in, c1_out = LINE1.trigger(detections=detections)
    c2_in, c2_out = LINE2.trigger(detections=detections)
    crossed_l1 = c1_in | c1_out
    crossed_l2 = c2_in | c2_out

    # 3. Proses Logika Modul per Kendaraan
    for i, (tid, cid) in enumerate(zip(detections.tracker_id, detections.class_id)):
        # Hitung Kecepatan DULU sebelum nyatet waktu biar ngga nabrak logika
        hitung_kecepatan_kendaraan(tid, crossed_l1[i], crossed_l2[i], t_sekarang, timestamp_l1, timestamp_l2, speeds_kmh, CAM_CONFIG["jarak_garis_meter"])
        
        # Baru catat masuk & hitung jumlahnya
        catat_dan_hitung(tid, cid, crossed_l1[i], crossed_l2[i], t_sekarang, timestamp_l1, timestamp_l2, class_counts)

    # 4. Ambil Hasil dari Modul
    total_akumulasi = get_total_akumulasi(class_counts)
    kendaraan_live = len(detections)
    avg_speed = get_rata_rata_kecepatan(speeds_kmh)
    
    in_roi_mask = roi_zone.trigger(detections=detections)
    jumlah_di_roi = int(in_roi_mask.sum()) if len(in_roi_mask) else 0
    status_macet, color_macet = cek_status_kemacetan(jumlah_di_roi, CAM_CONFIG["panjang_segmen_meter"], CAM_CONFIG["jumlah_lajur"])

    # 5. RENDER KE LAYAR
    annotated = frame.copy()

    if FLAGS["DETEKSI"]:
        annotated = box_annotator.annotate(scene=annotated, detections=detections)
        labels = [f"{CLASS_NAMES.get(cid,'?')}" for cid in detections.class_id]
        annotated = label_annotator.annotate(scene=annotated, detections=detections, labels=labels)

    if FLAGS["COUNTING"]:
        try:
            annotated = l1_annotator.annotate(annotated, line_zone=LINE1)
            annotated = l2_annotator.annotate(annotated, line_zone=LINE2)
        except TypeError:
            annotated = l1_annotator.annotate(annotated, line_counter=LINE1)
            annotated = l2_annotator.annotate(annotated, line_counter=LINE2)

    # Render UI Tombol Menu
    for btn_name, (x1, y1, x2, y2) in BUTTONS.items():
        is_active = FLAGS[btn_name]
        color = (0, 180, 0) if is_active else (60, 60, 60)
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, -1)
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (255, 255, 255), 1)
        cv2.putText(annotated, f"{btn_name}: {'ON' if is_active else 'OFF'}", (x1+8, y1+20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255), 1)

    # Render Dashboard Panel
    overlay = annotated.copy()
    cv2.rectangle(overlay, (20, 60), (420, 225), (0, 0, 0), -1)
    annotated = cv2.addWeighted(overlay, 0.7, annotated, 0.3, 0)
    
    cv2.putText(annotated, "DISHUB - TRAFFIC MONITORING", (35, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)
    
    v_akum = f"{total_akumulasi} unit" if FLAGS["COUNTING"] else "[NONAKTIF]"
    v_live = f"{kendaraan_live} unit" if FLAGS["DETEKSI"] else "[NONAKTIF]"
    v_sped = f"{avg_speed:.1f} km/jam" if FLAGS["KECEPATAN"] else "[NONAKTIF]"
    v_mact = status_macet if FLAGS["KEMACETAN"] else "[NONAKTIF]"
    c_mact = color_macet if FLAGS["KEMACETAN"] else (150, 150, 150)

    cv2.putText(annotated, f"1. Total Akumulasi  : {v_akum}", (35, 115), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.putText(annotated, f"2. Live Frame       : {v_live}", (35, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.putText(annotated, f"3. Rata2 Kecepatan  : {v_sped}", (35, 165), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 255, 200), 1)
    cv2.putText(annotated, f"4. Status Kemacetan : {v_mact}", (35, 190), cv2.FONT_HERSHEY_SIMPLEX, 0.5, c_mact, 2)

    cv2.imshow("CCTV Traffic Monitor", annotated)

    # 6. Housekeeping / Bersih-bersih RAM
    if frame_idx % (int(fps) * 5) == 0:
        bersihkan_memori(timestamp_l1, timestamp_l2, speeds_kmh, travel_times, t_sekarang)

    # Hotkeys
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'): break
    elif key == ord('1'): FLAGS["DETEKSI"] = not FLAGS["DETEKSI"]
    elif key == ord('2'): FLAGS["COUNTING"] = not FLAGS["COUNTING"]
    elif key == ord('3'): FLAGS["KECEPATAN"] = not FLAGS["KECEPATAN"]
    elif key == ord('4'): FLAGS["KEMACETAN"] = not FLAGS["KEMACETAN"]

    frame_idx += 1

cap.release()
cv2.destroyAllWindows()