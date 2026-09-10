# config.py
import numpy as np

# ==========================================
# KONFIGURASI KAMERA (Fisik / Lapangan)
# ==========================================
CAM_CONFIG = {
    "camera_id": "CCTV_Jalan_01",
    "video_source": "./samples/sampel 1.mp4", #ganti ke rtsp nanti
    "jarak_garis_meter": 5.5,
    "free_flow_speed_kmh": 40.0,
    "panjang_segmen_meter": 50.0,
    "jumlah_lajur": 2,
    # Arah normal arus lalu lintas pada frame kamera:
    #   "atas_ke_bawah" -> kendaraan bergerak dari atas layar ke bawah
    #   "bawah_ke_atas" -> kendaraan bergerak dari bawah layar ke atas
    "arah_lalu_lintas": "bawah_ke_atas"
}

# ==========================================
# MODEL DETEKSI
# ==========================================
MODEL_CONFIG = {
    "path": "models/best modol.pt",
    "conf": 0.5   # threshold moderat: lebih sensitif drpd 0.30, tidak senoisy 0.20
}

# ==========================================
# TRACKER (ByteTrack)
# ==========================================
TRACKER_CONFIG = {
    # aktivasi track baru jika confidence >= nilai ini (harus <= conf model)
    "track_activation_threshold": 0.25,
    # IoU matching lebih longgar -> kendaraan kecil/cepat tidak ganti ID.
    # Catatan: 0.3 lebih stabil utk kendaraan kecil di malam hari, namun ada
    # risiko ID melebur utk kendaraan berdampingan; sesuaikan per kamera.
    "minimum_matching_threshold": 0.3,
    # jumlah frame track "hilang" tetap disimpan sebelum dibuang (~3 detik di 30fps)
    "lost_track_buffer": 45,
}

# ==========================================
# STABILISASI BOX (DetectionsSmoother + BoxPersistence)
# ==========================================
SMOOTHER_CONFIG = {
    "length": 8,                # rata-rata posisi box dalam N frame terakhir (kurangi jitter)
    "persist_grace_frames": 25, # tahan box terakhir selama N frame saat deteksi sempat hilang (~0.6 dtk)
    "persist_decay": 0.7,       # faktor penurunan confidence box persisten
    "persist_min_detections": 2 # box baru dipersistenkan setelah terdeteksi >= N frame (anti hantu)
}

# ==========================================
# GARIS VIRTUAL (Counting Line) otomatis relatif thd tinggi frame
# ==========================================
GARIS = {
    "garis1_y_frac": 0.60,
    "garis2_y_frac": 0.45
}

# ==========================================
# STATUS KEMACETAN (LANCAR / PADAT / MACET)
# ==========================================
KEMACETAN_CONFIG = {
    "density_batas_lancar": 12.0,   # veh/km/lane di bawah ini + speed normal -> LANCAR
    "density_batas_padat": 25.0,    # di atas ini -> MACET
    "rasio_kecepatan_padat": 0.55,  # avg_speed/free_flow di bawah ini -> setidaknya PADAT
    "rasio_kecepatan_macet": 0.30   # avg_speed/free_flow di bawah ini -> MACET
}

# ==========================================
# DETEKSI LAWAN ARAH (Wrong-Way)
# ==========================================
LAWAN_ARAH_CONFIG = {
    "min_kecepatan_px": 3.0,     # kecepatan gerak min (px/detik) agar dianggap "bergerak"
    "frame_konfirmasi": 5,       # jumlah frame konsisten melawan arah sebelum flag
    "cooldown_detik": 20.0       # jeda antar alert per kendaraan
}

# ==========================================
# DETEKSI INSIDEN (Incident)
# ==========================================
INSIDEN_CONFIG = {
    "ambang_berhenti_px": 2.5,       # gerak < ambang (px/detik) = dianggap berhenti
    "durasi_berhenti_detik": 8.0,    # berhenti sekian detik -> event "kendaraan berhenti"
    "kecepatan_tinggi_px": 12.0,     # kecepatan awal utk deteksi pengereman mendadak
    "kecepatan_rendah_px": 2.0,      # kecepatan akhir utk deteksi pengereman mendadak
    "frame_hard_brake": 5,           # lebar window pengereman mendadak
    "ambang_iou_tabrakan": 0.25,     # IoU minimal antar box utk "potensi tabrakan"
    "min_seen_frames": 3,            # track harus terlihat N frame agar event dipertimbangkan
    "cooldown_detik": 30.0,
    "buffer_size": 90                # buffer frame history untuk time machine snapshot (~3.6s @25fps)
}

# ==========================================
# ANPR
# ==========================================
ANPR_CONFIG = {
    "enabled": True,
    "model_path": "./models/best_plat.pt",
    "model_conf": 0.25,
    "backend": "easyocr",
    "min_conf": 0.3,
    "cooldown_detik": 60.0,
    "min_w_plat": 40,
    "min_h_plat": 14
}

# ==========================================
# OUTPUT EVENT (Snapshot & Log)
# ==========================================
OUTPUT_CONFIG = {
    "event_dir": "./output/events",
    "simpan_snapshot": True
}

# ==========================================
# STATE FITUR & UI TOMBOL
# ==========================================
FLAGS = {
    "DETEKSI": True,
    "COUNTING": True,
    "KECEPATAN": True,
    "KEMACETAN": True,
    "LAWAN_ARAH": True,
    "INSIDEN": True,
    "ANPR": True
}

BUTTONS = {
    "DETEKSI": (20, 15, 150, 45),
    "COUNTING": (160, 15, 290, 45),
    "KECEPATAN": (300, 15, 430, 45),
    "KEMACETAN": (440, 15, 580, 45),
    "LAWAN_ARAH": (600, 15, 750, 45),
    "INSIDEN": (760, 15, 900, 45),
    "ANPR": (910, 15, 1040, 45)
}

CLASS_NAMES = {0: "Orang", 1: "Motor", 2: "Mobil", 3: "Bus", 4: "Truk"}
