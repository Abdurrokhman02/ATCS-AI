# main.py
import os
import warnings
import cv2
import numpy as np
from collections import defaultdict
from ultralytics import YOLO
import supervision as sv

# --- IMPORT MODULE ---
from config import CAM_CONFIG, FLAGS, BUTTONS, CLASS_NAMES
from modules.fitur_counting import catat_dan_hitung, get_total_akumulasi, bersihkan_memori
from modules.fitur_kecepatan import hitung_kecepatan_kendaraan, get_rata_rata_kecepatan
from modules.fitur_kemacetan import cek_status_kemacetan
from modules.fitur_kecelakaan import pantau_kecelakaan # <--- IMPORT MODUL BARU

os.environ["QT_QPA_PLATFORM"] = "xcb"
os.environ["QT_LOGGING_RULES"] = "*.warning=false"
warnings.filterwarnings("ignore", category=FutureWarning)

def handle_mouse_click(event, x, y, flags_param, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        for feature_name, (x1, y1, x2, y2) in BUTTONS.items():
            if x1 <= x <= x2 and y1 <= y <= y2:
                FLAGS[feature_name] = not FLAGS[feature_name]

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

model = YOLO('models/modol.pt')
tracker = sv.ByteTrack(track_activation_threshold=0.25)

box_annotator = sv.BoxAnnotator(thickness=2)
label_annotator = sv.LabelAnnotator(text_thickness=1, text_scale=0.5)
l1_annotator = sv.LineZoneAnnotator(thickness=2, color=sv.Color.GREEN)
l2_annotator = sv.LineZoneAnnotator(thickness=2, color=sv.Color.RED)

# --- STATE VARIABLES ---
timestamp_l1, timestamp_l2 = {}, {}
speeds_kmh, travel_times = [], []
class_counts = defaultdict(int)

# State untuk Fitur Kecelakaan
frame_buffer = {}
box_history = defaultdict(list)
id_terlibat_kecelakaan = []

cv2.namedWindow("CCTV Traffic Monitor")
cv2.setMouseCallback("CCTV Traffic Monitor", handle_mouse_click)

frame_idx = 0
while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break
    t_sekarang = frame_idx / fps
    frame_idx += 1

    results = model(frame, conf=0.30, verbose=False)[0]
    detections = sv.Detections.from_ultralytics(results)
    allowed_class_ids = list(CLASS_NAMES.keys())
    detections = detections[np.isin(detections.class_id, allowed_class_ids)]
    detections = tracker.update_with_detections(detections)

    c1_in, c1_out = LINE1.trigger(detections=detections)
    c2_in, c2_out = LINE2.trigger(detections=detections)
    crossed_l1, crossed_l2 = c1_in | c1_out, c2_in | c2_out

    # --- EKSEKUSI FITUR KECELAKAAN ---
    ada_kecelakaan = False
    if FLAGS["KECELAKAAN"]:
        ada_kecelakaan = pantau_kecelakaan(
            frame, frame_idx, detections, frame_buffer, box_history, id_terlibat_kecelakaan, width, height
        )

    for i, (tid, cid) in enumerate(zip(detections.tracker_id, detections.class_id)):
        hitung_kecepatan_kendaraan(tid, crossed_l1[i], crossed_l2[i], t_sekarang, timestamp_l1, timestamp_l2, speeds_kmh, CAM_CONFIG["jarak_garis_meter"])
        catat_dan_hitung(tid, cid, crossed_l1[i], crossed_l2[i], t_sekarang, timestamp_l1, timestamp_l2, class_counts)

    # --- AGREGASI DATA ---
    total_akumulasi = get_total_akumulasi(class_counts)
    kendaraan_live = len(detections)
    avg_speed = get_rata_rata_kecepatan(speeds_kmh)
    
    in_roi_mask = roi_zone.trigger(detections=detections)
    jumlah_di_roi = int(in_roi_mask.sum()) if len(in_roi_mask) else 0
    status_macet, color_macet = cek_status_kemacetan(jumlah_di_roi, CAM_CONFIG["panjang_segmen_meter"], CAM_CONFIG["jumlah_lajur"])

    # --- RENDER KE LAYAR ---
    annotated = frame.copy()

    if FLAGS["DETEKSI"]:
        annotated = box_annotator.annotate(scene=annotated, detections=detections)
        # Modifikasi Label: Munculkan [AWAS!] jika kendaraan ini pernah terlibat tabrakan
        labels = [
            f"{CLASS_NAMES.get(cid,'?')} [AWAS!]" if tid in id_terlibat_kecelakaan else f"{CLASS_NAMES.get(cid,'?')}"
            for cid, tid in zip(detections.class_id, detections.tracker_id)
        ]
        annotated = label_annotator.annotate(scene=annotated, detections=detections, labels=labels)

    if FLAGS["COUNTING"]:
        try:
            annotated = l1_annotator.annotate(annotated, line_zone=LINE1)
            annotated = l2_annotator.annotate(annotated, line_zone=LINE2)
        except TypeError:
            annotated = l1_annotator.annotate(annotated, line_counter=LINE1)
            annotated = l2_annotator.annotate(annotated, line_counter=LINE2)

    # UI Tombol Menu
    for btn_name, (x1, y1, x2, y2) in BUTTONS.items():
        is_active = FLAGS[btn_name]
        color = (0, 180, 0) if is_active else (60, 60, 60)
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, -1)
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (255, 255, 255), 1)
        cv2.putText(annotated, f"{btn_name}: {'ON' if is_active else 'OFF'}", (x1+8, y1+20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255), 1)

    # --- OVERLAY PERINGATAN KECELAKAAN (BANNER MERAH) ---
    if ada_kecelakaan:
        cv2.rectangle(annotated, (0, 0), (width, 80), (0, 0, 255), -1)
        cv2.putText(annotated, "!!! PERINGATAN KECELAKAAN !!! MENGAMBIL BUKTI", (30, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)

    # --- DASHBOARD PANEL ---
    overlay = annotated.copy()
    cv2.rectangle(overlay, (20, 60), (450, 250), (0, 0, 0), -1) # Kotak diperlebar agar muat 5 baris
    annotated = cv2.addWeighted(overlay, 0.7, annotated, 0.3, 0)
    
    cv2.putText(annotated, "DISHUB - TRAFFIC MONITORING", (35, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)
    
    v_akum = f"{total_akumulasi} unit" if FLAGS["COUNTING"] else "[NONAKTIF]"
    v_live = f"{kendaraan_live} unit" if FLAGS["DETEKSI"] else "[NONAKTIF]"
    v_sped = f"{avg_speed:.1f} km/jam" if FLAGS["KECEPATAN"] else "[NONAKTIF]"
    v_mact = status_macet if FLAGS["KEMACETAN"] else "[NONAKTIF]"
    c_mact = color_macet if FLAGS["KEMACETAN"] else (150, 150, 150)
    
    v_crash = "TERDETEKSI" if len(id_terlibat_kecelakaan) > 0 else "AMAN"
    c_crash = (0, 0, 255) if len(id_terlibat_kecelakaan) > 0 else (0, 255, 0)
    if not FLAGS["KECELAKAAN"]: v_crash, c_crash = "[NONAKTIF]", (150, 150, 150)

    cv2.putText(annotated, f"1. Total Akumulasi  : {v_akum}", (35, 115), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.putText(annotated, f"2. Live Frame       : {v_live}", (35, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.putText(annotated, f"3. Rata2 Kecepatan  : {v_sped}", (35, 165), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 255, 200), 1)
    cv2.putText(annotated, f"4. Status Kemacetan : {v_mact}", (35, 190), cv2.FONT_HERSHEY_SIMPLEX, 0.5, c_mact, 2)
    cv2.putText(annotated, f"5. Status Kecelakaan: {v_crash}", (35, 215), cv2.FONT_HERSHEY_SIMPLEX, 0.5, c_crash, 2)

    cv2.imshow("CCTV Traffic Monitor", annotated)

    # --- HOUSEKEEPING / BERSIH-BERSIH RAM ---
    if frame_idx % (int(fps) * 5) == 0:
        bersihkan_memori(timestamp_l1, timestamp_l2, speeds_kmh, travel_times, t_sekarang)
        
        # Hapus histori posisi kendaraan lama agar RAM tidak penuh
        aktif_ids = set(detections.tracker_id) if detections.tracker_id is not None else set()
        for k in list(box_history.keys()):
            if k not in aktif_ids and k not in id_terlibat_kecelakaan:
                del box_history[k]

    # --- HOTKEYS ---
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'): break
    elif key == ord('1'): FLAGS["DETEKSI"] = not FLAGS["DETEKSI"]
    elif key == ord('2'): FLAGS["COUNTING"] = not FLAGS["COUNTING"]
    elif key == ord('3'): FLAGS["KECEPATAN"] = not FLAGS["KECEPATAN"]
    elif key == ord('4'): FLAGS["KEMACETAN"] = not FLAGS["KEMACETAN"]
    elif key == ord('5'): FLAGS["KECELAKAAN"] = not FLAGS["KECELAKAAN"]

cap.release()
cv2.destroyAllWindows()