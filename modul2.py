"""
MODUL B - TRAFFIC INTELLIGENCE
Dishub Bandung - AI ATCS
================================
Melanjutkan kerangka Modul 1 & Modul A (YOLOv8 + ByteTrack + supervision).
Menambahkan metrik:
  - Traffic Density        (kendaraan / km / lajur)
  - Vehicle Count           (per kategori, sudah ada dari Modul A)
  - Average Speed           (km/jam, estimasi dari 2 garis lintas)
  - Queue Length             (meter, dari kendaraan diam di ROI)
  - Delay Time                (detik, selisih travel time vs kondisi free-flow)
  - Travel Time               (detik, waktu tempuh antar 2 garis)
  - Congestion Index          (0-100%, dari rasio speed vs free-flow speed)
  - Level of Service (LOS)    (A-F, dari density, acuan mirip HCM)

CATATAN PENTING - WAJIB DIKALIBRASI SEBELUM DIPAKAI SERIUS:
  1. JARAK_GARIS_METER   -> ukur jarak riil garis1-garis2 di lapangan
                             (pakai Google Maps "measure distance" atau GPS)
  2. FREE_FLOW_SPEED_KMH -> kecepatan rata-rata saat jalan lengang/tengah malam
  3. PANJANG_SEGMEN_METER & JUMLAH_LAJUR -> untuk hitung density per km per lajur
  4. Threshold LOS di fungsi hitung_los() masih pakai acuan umum (HCM-like).
     Untuk laporan resmi Dishub, sebaiknya disesuaikan ke MKJI 1997 Indonesia.
"""

import cv2
import numpy as np
from collections import defaultdict
from ultralytics import YOLO
import supervision as sv

# ==========================================
# 1. INISIALISASI MODEL & TRACKER
# ==========================================
model = YOLO('yolov8m.pt')
tracker = sv.ByteTrack()

# ==========================================
# 2. KALIBRASI DUNIA NYATA (WAJIB DIISI SESUAI LOKASI!)
# ==========================================
JARAK_GARIS_METER = 20.0       # jarak riil antar garis1 & garis2 (meter)
FREE_FLOW_SPEED_KMH = 40.0     # kecepatan kondisi jalan lengang (km/jam)
PANJANG_SEGMEN_METER = 50.0    # panjang ruas jalan yang diamati (meter)
JUMLAH_LAJUR = 2                # jumlah lajur pada ruas yang diamati

# ==========================================
# 3. DUA GARIS VIRTUAL: ENTRY (garis1) & EXIT (garis2)
# ==========================================
garis1_y = 300   # entry - sesuaikan dengan video
garis2_y = 500   # exit  - sesuaikan dengan video

LINE1 = sv.LineZone(start=sv.Point(0, garis1_y), end=sv.Point(1280, garis1_y))
LINE2 = sv.LineZone(start=sv.Point(0, garis2_y), end=sv.Point(1280, garis2_y))

# ROI (poligon) untuk hitung density & antrian -> area di antara garis1 & garis2
ROI_POLYGON = np.array([
    [0, garis1_y],
    [1280, garis1_y],
    [1280, garis2_y],
    [0, garis2_y]
])
# NOTE: parameter PolygonZone bisa berbeda tergantung versi `supervision`.
# Kalau error, cek dokumentasi versi terpasang (`pip show supervision`).
roi_zone = sv.PolygonZone(polygon=ROI_POLYGON)

# ==========================================
# 4. VISUAL ANNOTATORS
# ==========================================
box_annotator = sv.BoxAnnotator(thickness=2)
label_annotator = sv.LabelAnnotator(text_thickness=1, text_scale=0.5)
line1_annotator = sv.LineZoneAnnotator(thickness=2, text_thickness=1, color=sv.Color.GREEN)
line2_annotator = sv.LineZoneAnnotator(thickness=2, text_thickness=1, color=sv.Color.RED)

CLASS_NAMES = {2: "Mobil Pribadi", 3: "Sepeda Motor", 5: "Bus Pariwisata", 7: "Truk Logistik"}

# ==========================================
# 5. STATE UNTUK PERHITUNGAN METRIK
# ==========================================
timestamp_line1 = {}            # tracker_id -> waktu (detik) saat lewat garis1
travel_times = []               # riwayat travel time (detik)
speeds_kmh = []                 # riwayat kecepatan (km/jam)
class_counts = defaultdict(int)
stationary_frames = defaultdict(int)   # tracker_id -> jumlah frame beruntun "diam"
last_position = {}              # tracker_id -> (cx, cy) posisi terakhir

STATIONARY_SPEED_THRESHOLD_PX = 3      # gerak < 3px/frame dianggap diam
QUEUE_STATIONARY_FRAMES = 15           # diam >= 15 frame berturut = "ngantri"
PANJANG_RATA2_KENDARAAN_M = 6          # asumsi panjang + jarak aman antrian (meter)

# ==========================================
# 6. PROSES VIDEO
# ==========================================
video_input = "sampleuji.mp4"
cap = cv2.VideoCapture(video_input)

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
out = cv2.VideoWriter("Modul_B_Traffic_Intelligence.mp4", cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

frame_idx = 0
print("⏳ Menjalankan AI Modul B (Traffic Intelligence)...")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    t_sekarang = frame_idx / fps

    results = model(frame, verbose=False)[0]
    detections = sv.Detections.from_ultralytics(results)
    detections = detections[np.isin(detections.class_id, [2, 3, 5, 7])]
    detections = tracker.update_with_detections(detections)

    # --- crossing garis 1 & 2 ---
    crossed_in_1, crossed_out_1 = LINE1.trigger(detections=detections)
    crossed_in_2, crossed_out_2 = LINE2.trigger(detections=detections)

    for i, (tid, cid) in enumerate(zip(detections.tracker_id, detections.class_id)):
        # Vehicle Count per kategori (dihitung saat lewat garis1)
        if crossed_in_1[i] or crossed_out_1[i]:
            class_counts[cid] += 1
            timestamp_line1[tid] = t_sekarang

        # Travel Time & Average Speed (saat lewat garis2, kalau sudah tercatat di garis1)
        if (crossed_in_2[i] or crossed_out_2[i]) and tid in timestamp_line1:
            tt = t_sekarang - timestamp_line1[tid]
            if tt > 0:
                travel_times.append(tt)
                speed = (JARAK_GARIS_METER / tt) * 3.6   # m/s -> km/jam
                speeds_kmh.append(speed)
            del timestamp_line1[tid]

    # --- Queue Length: kendaraan diam di dalam ROI ---
    in_roi_mask = roi_zone.trigger(detections=detections)
    queue_count = 0
    xyxy = detections.xyxy
    for i, tid in enumerate(detections.tracker_id):
        cx = (xyxy[i][0] + xyxy[i][2]) / 2
        cy = (xyxy[i][1] + xyxy[i][3]) / 2
        if tid in last_position:
            px, py = last_position[tid]
            gerak = float(np.hypot(cx - px, cy - py))
            if gerak < STATIONARY_SPEED_THRESHOLD_PX:
                stationary_frames[tid] += 1
            else:
                stationary_frames[tid] = 0
        last_position[tid] = (cx, cy)

        if in_roi_mask[i] and stationary_frames.get(tid, 0) >= QUEUE_STATIONARY_FRAMES:
            queue_count += 1

    # ==========================================
    # 7. AGREGASI METRIK
    # ==========================================
    jumlah_di_roi = int(in_roi_mask.sum()) if len(in_roi_mask) else 0
    density_veh_per_km_per_lane = (jumlah_di_roi / (PANJANG_SEGMEN_METER / 1000)) / JUMLAH_LAJUR

    avg_speed = float(np.mean(speeds_kmh[-30:])) if speeds_kmh else 0.0
    avg_travel_time = float(np.mean(travel_times[-30:])) if travel_times else 0.0

    free_flow_travel_time = JARAK_GARIS_METER / (FREE_FLOW_SPEED_KMH / 3.6) if FREE_FLOW_SPEED_KMH > 0 else 0.0
    delay_time = max(0.0, avg_travel_time - free_flow_travel_time) if avg_travel_time else 0.0

    congestion_index = max(0.0, min(1.0, 1 - (avg_speed / FREE_FLOW_SPEED_KMH))) if avg_speed and FREE_FLOW_SPEED_KMH else 0.0

    def hitung_los(density):
        # Acuan umum ala HCM (veh/km/lane). Sesuaikan ke MKJI 1997 untuk laporan resmi.
        if density <= 7:   return "A"
        elif density <= 11: return "B"
        elif density <= 16: return "C"
        elif density <= 22: return "D"
        elif density <= 28: return "E"
        else:                return "F"

    los = hitung_los(density_veh_per_km_per_lane)
    queue_length_m = queue_count * PANJANG_RATA2_KENDARAAN_M

    # ==========================================
    # 8. RENDER DASHBOARD
    # ==========================================
    annotated_frame = box_annotator.annotate(scene=frame.copy(), detections=detections)
    labels = [f"#{tid} {CLASS_NAMES.get(cid,'?')}" for cid, tid in zip(detections.class_id, detections.tracker_id)]
    annotated_frame = label_annotator.annotate(scene=annotated_frame, detections=detections, labels=labels)
    annotated_frame = line1_annotator.annotate(annotated_frame, line_counter=LINE1)
    annotated_frame = line2_annotator.annotate(annotated_frame, line_counter=LINE2)

    overlay = annotated_frame.copy()
    cv2.rectangle(overlay, (20, 20), (480, 320), (0, 0, 0), -1)
    annotated_frame = cv2.addWeighted(overlay, 0.7, annotated_frame, 0.3, 0)

    total_semua = sum(class_counts.values())
    baris = [
        ("TRAFFIC INTELLIGENCE DASHBOARD", (0, 255, 255)),
        (f"Vehicle Count    : {total_semua}", (255, 255, 255)),
        (f"Traffic Density  : {density_veh_per_km_per_lane:.1f} veh/km/lane", (255, 255, 255)),
        (f"Average Speed    : {avg_speed:.1f} km/jam", (200, 255, 200)),
        (f"Queue Length     : {queue_length_m} m ({queue_count} kendaraan)", (200, 255, 200)),
        (f"Delay Time       : {delay_time:.1f} detik", (200, 200, 255)),
        (f"Travel Time      : {avg_travel_time:.1f} detik", (200, 200, 255)),
        (f"Congestion Index : {congestion_index*100:.0f}%", (180, 220, 255)),
        (f"Level of Service : {los}", (0, 200, 255)),
    ]
    for i, (teks, warna) in enumerate(baris):
        cv2.putText(annotated_frame, teks, (40, 55 + i * 30), cv2.FONT_HERSHEY_SIMPLEX, 0.55, warna, 2)

    out.write(annotated_frame)
    frame_idx += 1

cap.release()
out.release()
print("✅ SELESAI! Video tersimpan: 'Modul_B_Traffic_Intelligence.mp4'")