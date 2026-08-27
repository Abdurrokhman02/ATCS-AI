# config.py
import numpy as np

# ==========================================
# KONFIGURASI KAMERA (Fisik / Lapangan)
# ==========================================
CAM_CONFIG = {
    "camera_id": "CCTV_Jalan_01",
    "video_source": "./samples/samplekecelakaan.mp4",
    "jarak_garis_meter": 20.0,
    "free_flow_speed_kmh": 40.0,
    "panjang_segmen_meter": 50.0,
    "jumlah_lajur": 2
}

# ==========================================
# STATE FITUR & UI TOMBOL
# ==========================================
FLAGS = {
    "DETEKSI": True,
    "COUNTING": True,
    "KECEPATAN": True,
    "KEMACETAN": True,
    "KECELAKAAN": True
}

BUTTONS = {
    "DETEKSI": (20, 15, 150, 45),
    "COUNTING": (160, 15, 290, 45),
    "KECEPATAN": (300, 15, 430, 45),
    "KEMACETAN": (440, 15, 580, 45),
    "KECELAKAAN": (590, 15, 730, 45)
}

CLASS_NAMES = {
    1: "Mobil Pribadi",
    2: "Sepeda Motor",
    3: "Bus Pariwisata",
    4: "Truk Logistik"
}