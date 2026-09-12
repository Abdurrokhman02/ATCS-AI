# ATCS Traffic Intelligence - CCTV Monitoring System

## 📋 Identitas Project
- **Nama:** ATCS Traffic Intelligence — Adaptive Traffic Control System
- **Tujuan:** Sistem monitoring lalu lintas real-time berbasis Computer Vision & AI. Input video CCTV → deteksi kendaraan (YOLOv8) → tracking (ByteTrack) → 7 fitur analitik (counting, speed, congestion, wrong-way, incident, ANPR) → output GUI real-time atau headless.
- **Tim Developer:** Abdurrokhman02 (Phase 1), Raihan-Brown (refactor Phase 2)
- **Stack:** Python 3.10+, Ultralytics YOLOv8, Supervision/ByteTrack, OpenCV, NumPy, EasyOCR/Pytesseract
- **OS Target:** Windows & Linux (Wayland/Hyprland workaround via `QT_QPA_PLATFORM=xcb`)

---

## 🧱 Arsitektur Pipeline

```
Video Source (file/RTSP/webcam)
        │
  YOLOv8 Inference → deteksi: Mobil(1), Motor(2), Bus(3), Truk(4)
        │
  ByteTrack → assign tracker ID unik per kendaraan
        │
  ┌─────┴──────┐
  │ Persistence │  ← fitur_stabilisasi.py (bounding box tahan saat deteksi hilang)
  │ Smoother    │  ← supervision DetectionsSmoother (rata-rata posisi 8 frame terakhir)
  └─────┬──────┘
        │
  ┌─────┼─────┬──────┬──────┬──────┬──────┐
  │     │     │      │      │      │      │
  ▼     ▼     ▼      ▼      ▼      ▼      ▼
Counting Speed Congestion WrongWay Incident ANPR  EventLogger
        │     │      │      │      │      │      │
  ┌─────┴─────┴──────┴──────┴──────┴──────┴──────┘
  │
  ▼
Render + UI (GUI mode: cv2 window + buttons + hotkeys)
  │
Headless mode: console progress + output video + JSON report
```

---

## ✅ Semua Fitur & Modul

### 1️⃣ Vehicle Detection + Tracking
- **File:** `main.py` (inference loop) + `fitur_stabilisasi.py`
- YOLOv8 via Ultralytics, filter class: Mobil(1), Motor(2), Bus(3), Truk(4)
- ByteTrack via Supervision: `track_activation_threshold=0.25`, `lost_track_buffer=90` frame (~3 detik)
- Output: `sv.Detections` dengan tracker ID

### 2️⃣ Box Persistence & Smoothing — `fitur_stabilisasi.py`
- **Kelas:** `BoxPersistence` — menyimpan bounding box selama `grace_frames=25` saat deteksi hilang
- **Anti-ghost filter:** butuh minimal `min_detections=2` sebelum box benar-benar muncul
- **Confidence decay:** keyakinan menurun tiap frame (`decay=0.7`) — box dihapus jika < threshold
- **Smoother:** `sv.DetectionsSmoother` dengan panjang `SMOOTHER_CONFIG.length=8`
- Persistence hanya untuk render visual, TIDAK mempengaruhi data analitik

### 3️⃣ Vehicle Counting — `fitur_counting.py`
- **Kelas:** `VehicleCounter`
- Dual virtual lines (Garis 1 = 45% tinggi frame, Garis 2 = 60%)
- Tracking koordinat Y tiap tracker, deteksi crossing dengan tolerance
- **Crossing logic:** tracker harus melewati garis secara berurutan (G1→G2 = masuk, G2→G1 = keluar)
- **Sekali hitung per tracker** — tidak akan dihitung ulang
- **Per-class breakdown:** `counter_per_class` dictionary
- **Memory cleanup:** `bersihkan_memori()` hapus tracker stale >90 frame
- **State:** `total`, `count_in`, `count_out`, `counter_per_class`

### 4️⃣ Speed Estimation — `fitur_kecepatan.py`
- **Kelas:** `SpeedEstimator`
- Hitung `jarak_garis_meter / waktu_tempuh_detik * 3.6`
- Waktu tempuh = selisih timestamp crossing G1 dan G2
- **Filter outlier:** speed < 2 km/h atau > 200 km/h diabaikan
- **Rolling average:** rata-rata dari max 20 sampel terakhir
- Butuh kalibrasi: `jarak_garis_meter` di config (default 20m)

### 5️⃣ Congestion Classification — `fitur_kemacetan.py`
- **Kelas:** `KemacetanAnalyst`
- **Density:** `jumlah_kendaraan_di_zone / (panjang_segmen_meter / 1000) / jumlah_lajur`
- **Speed ratio:** `rata_rata_speed / free_flow_speed`
- **Threshold LOS (Level of Service):**
  - `LANCAR` (hijau): density < 12 veh/km/lane ATAU speed ratio >= 0.75
  - `MACET` (merah): density > 25 veh/km/lane DAN speed ratio < 0.30
  - `PADAT` (oranye): di antaranya — density >= 12 DAN speed ratio < 0.55
- Butuh kalibrasi: `panjang_segmen_meter=50`, `jumlah_lajur=2`, `free_flow_speed=40` km/h

### 6️⃣ Wrong-Way Detection — `fitur_lawan_arah.py`
- **Kelas:** `WrongWayDetector`
- **Movement history:** deque posisi Y per tracker (max 30 frame)
- **Logic:** hitung slope (perubahan Y) dalam history; jika signifikan negatif/positif tergantung `arah_lalu_lintas` (default `bawah`)
- **Konfirmasi:** butuh 5 frame berturut-turut terdeteksi lawan arah sebelum trigger
- **Cooldown:** 20 detik per tracker setelah trigger
- **Min movement:** pergerakan < 3 pixel/frame diabaikan (anggap diam)

### 7️⃣ Incident Detection — `fitur_insiden.py`
- **Kelas:** `IncidentDetector`
- **3 sub-insiden:**
  1. **Kendaraan Berhenti:** tracker dengan pergerakan < 2.5 px/s selama >= 8 detik (20 frame @ ~3fps)
  2. **Pengereman Mendadak:** penurunan speed drastis dalam 5 frame — dari >= 12 px/frame jadi <= 2 px/frame
  3. **Potensi Tabrakan:** dua tracker berhenti (sub-insiden 1) dengan IoU bounding box > 0.25
- **Time machine:** simpan snapshot beberapa frame sebelum event via `frame_buffer` (deque 90 frame)
- **Cooldown:** 30 detik per tracker untuk kendaraan berhenti
- **Per-tracker state:** `last_positions` (rolling), `stopped_frames`, `recent_speeds`

### 8️⃣ ANPR (Automatic Number Plate Recognition) — `fitur_anpr.py`
- **Kelas:** `ANPRReader`
- **Lazy loading:** model YOLO `best_plat.pt` + OCR backend hanya di-load saat pertama dibutuhkan
- **Pipeline:**
  1. Ambil crop bounding box kendaraan dari frame
  2. Deteksi plat via YOLO pada crop (filter class plat)
  3. Crop plat → OCR via EasyOCR (primary) atau Pytesseract (fallback)
  4. Cleanup teks: hanya alfanumerik, uppercase, hapus spasi berlebih
- **Filter:** confidence OCR >= 0.3, lebar plat crop >= 80px
- **Cooldown:** 60 detik per tracker — tidak akan baca ulang plat yang sama
- **State:** `ocr_results` dictionary per tracker ID

### 9️⃣ Event Logging — `event_logger.py`
- **Kelas:** `EventLogger`
- **Format:** JSONL (`output/events/events.jsonl`) — satu baris per event
- **Field per event:** `timestamp`, `camera_id`, `frame`, `event_type`, `confidence`, `vehicle_id`, `vehicle_class`, `metadata` (sub-insiden, speed, plat, dll), `anpr_result` (jika ada)
- **Snapshot:** simpan JPG frame ke folder events untuk event tertentu
- **Auto-flush:** tiap event langsung di-write + di-`flush()` biar aman crash
- **Event types:** `LAWAN_ARAH`, `KENDARAAN_BERHENTI`, `PENGEREMAN_MENDADAK`, `POTENSI_TABRAKAN`, `ANPR`

---

## 📁 Struktur File Lengkap

```
ATCS-AI/
│
├── main.py                          # Entry point utama (444 baris)
├── config.py                        # Central config (~138 baris)
├── testing.py                       # Legacy Phase 1 prototype (~264 baris)
├── test_crop.py                     # Debug ekstraksi crop plat (~57 baris)
├── Project.md                       # Dokumentasi ini
├── Readme.md                        # Dokumentasi ringkas
├── requirements.txt                 # (file binary/tidak readable di repo)
├── .gitignore                       # Git exclusion rules
├── best_kendaraan.pt                # YOLO weights deteksi kendaraan
├── best_plat.pt                     # YOLO weights deteksi plat
│
├── .venv/                           # Virtual environment Python
├── __pycache__/                     # Bytecode cache
│
├── modules/                         # CORE MODULES
│   ├── __init__.py
│   ├── fitur_counting.py            # Vehicle counting dual garis
│   ├── fitur_kecepatan.py           # Speed estimation
│   ├── fitur_kemacetan.py           # Congestion classification
│   ├── fitur_lawan_arah.py          # Wrong-way detection
│   ├── fitur_insiden.py             # Incident detection (3 jenis)
│   ├── fitur_anpr.py                # ANPR dengan OCR backend
│   ├── fitur_stabilisasi.py         # Box persistence + smoother
│   └── event_logger.py              # JSONL event logging + snapshot
│
├── tests/                           # UNIT TESTS
│   ├── test_counting.py             # 54 baris
│   ├── test_kecepatan.py            # 53 baris
│   ├── test_kemacetan.py            # 64 baris
│   ├── test_lawan_arah.py           # 65 baris
│   ├── test_insiden.py              # 85 baris
│   ├── test_anpr.py                 # 66 baris
│   └── test_stabilisasi.py          # 54 baris
│
├── testing/                         # INTEGRATION TEST HARNESS
│   ├── README.md
│   ├── __init__.py
│   ├── _runner.py                   # Shared runner, set FLAGS + panggil main.py
│   └── video/                       # Test per fitur via video
│       ├── test_deteksi.py
│       ├── test_counting.py
│       ├── test_kecepatan.py
│       ├── test_kemacetan.py
│       ├── test_lawan_arah.py
│       ├── test_insiden.py
│       ├── test_anpr.py
│       └── test_all_features.py     # Run semua fitur berurutan
│
├── models/                          # YOLO WEIGHTS
│   ├── yolov8n.pt                   # Nano (~6.5MB)
│   ├── yolov8m.pt                   # Medium (~52MB)
│   └── best.pt                      # Fine-tuned model (~52MB)
│
├── samples/                         # VIDEO SAMPLE
│   ├── sample kendaraa berhenti.mp4
│   ├── sampel plat.mp4
│   └── sampel 1.mp4
│
└── output/                          # RUNTIME OUTPUT
    ├── events/
    │   ├── events.jsonl             # Log event (8 events recorded)
    │   └── *.jpg                    # Snapshot event
    ├── testing/
    │   ├── counting_report.json     # Report: 1192 frame, 19 kendaraan
    │   └── counting_output.mp4      # Video anotasi
    └── debug_plat_crops/            # Debug crop plat dari test_crop.py
```

---

## ⚙️ Konfigurasi Detail (`config.py`)

| Group | Key | Default | Deskripsi |
|-------|-----|---------|-----------|
| **CAM_CONFIG** | `video_source` | `./samples/sampel 1.mp4` | Sumber video |
| | `camera_id` | `CCTV_Jalan_01` | ID kamera untuk log |
| | `jarak_garis_meter` | `20.0` | Jarak fisik 2 garis (speed) |
| | `panjang_segmen_meter` | `50` | Panjang segmen (density) |
| | `jumlah_lajur` | `2` | Jumlah lajur (density) |
| | `free_flow_speed` | `40` | Kecepatan ideal km/h |
| | `arah_lalu_lintas` | `bawah` | Arah normal (`bawah`/`atas`) |
| **MODEL_CONFIG** | `path` | `models/best.pt` | Path YOLO utama |
| | `conf` | `0.25` | Confidence threshold |
| **TRACKER_CONFIG** | `track_activation_threshold` | `0.25` | ByteTrack threshold |
| | `minimum_matching_threshold` | `0.3` | ByteTrack matching |
| | `lost_track_buffer` | `90` | Frame buffer lost track (~3dtk) |
| **SMOOTHER_CONFIG** | `length` | `8` | Panjang smoother (frame) |
| **GARIS** | `y1` | `0.45` | Garis 1 (45% tinggi frame) |
| | `y2` | `0.60` | Garis 2 (60% tinggi frame) |
| **KEMACETAN_CONFIG** | `lancar_max_density` | `12` | Max veh/km/lane untuk lancar |
| | `macet_min_density` | `25` | Min untuk macet |
| | `padat_speed_ratio` | `0.55` | Threshold padat |
| | `macet_speed_ratio` | `0.30` | Threshold macet |
| **LAWAN_ARAH_CONFIG** | `min_movement` | `3.0` | Min px/frame |
| | `confirmation_frames` | `5` | Frame konfirmasi |
| | `cooldown` | `20` | Cooldown detik |
| **INSIDEN_CONFIG** | `stop_threshold` | `2.5` | px/s threshold berhenti |
| | `stop_duration` | `8.0` | Detik threshold berhenti |
| | `hard_brake_speed_high` | `12` | px/frame awal |
| | `hard_brake_speed_low` | `2` | px/frame akhir |
| | `hard_brake_window` | `5` | Frame window brake |
| | `crash_iou_threshold` | `0.25` | IoU threshold tabrakan |
| | `cooldown` | `30` | Cooldown detik |
| **ANPR_CONFIG** | `enabled` | `True` | Aktif/nonaktif |
| | `model_path` | `best_plat.pt` | Path YOLO plat |
| | `ocr_confidence` | `0.3` | Min OCR confidence |
| | `cooldown` | `60` | Cooldown detik per kendaraan |
| | `min_plate_width` | `80` | Min lebar crop plat (px) |
| **OUTPUT_CONFIG** | `event_dir` | `./output/events` | Folder output event |
| | `save_snapshot` | `True` | Simpan snapshot JPG |
| **FLAGS** | `show_deteksi: True` | toggle | Tampilkan bounding box |
| | `show_counting: True` | toggle | Tampilkan counting |
| | `show_kecepatan: True` | toggle | Tampilkan speed |
| | `show_kemacetan: True` | toggle | Tampilkan congestion |
| | `show_lawan_arah: True` | toggle | Tampilkan wrong-way |
| | `show_insiden: True` | toggle | Tampilkan insiden |
| | `show_anpr: True` | toggle | Tampilkan ANPR |

---

## 🖥️ CLI Arguments (`main.py`)

| Argument | Tipe | Default | Deskripsi |
|----------|------|---------|-----------|
| `--source` | str | dari config | Video source path |
| `--model` | str | dari config | YOLO model path |
| `--conf` | float | dari config | Confidence threshold |
| `--headless` | flag | False | Nonaktifkan GUI |
| `--max-frames` | int | inf | Batas frame proses |
| `--output-video` | str | None | Path output video |
| `--output-report` | str | None | Path output JSON report |
| `--no-lawan-arah` | flag | False | Matikan wrong-way |
| `--no-insiden` | flag | False | Matikan insiden |
| `--no-anpr` | flag | False | Matikan ANPR |

---

## 🎮 Mode Interaksi

### GUI Mode (default)
- **Hotkeys:**
  - `1` toggle deteksi | `2` toggle counting | `3` toggle speed
  - `4` toggle kemacetan | `5` toggle lawan arah | `6` toggle insiden
  - `7` toggle ANPR | `q` keluar
- **Mouse click:** tombol toggle di layar tanpa stop program
- **Dashboard panel:** informasi real-time (total kendaraan, speed rata-rata, status kemacetan, jumlah event)
- **Event strip:** deretan event terbaru di bagian atas frame

### Headless Mode (`--headless`)
- Console progress tiap 2 detik
- Output video anotasi jika `--output-video`
- JSON report jika `--output-report`
- Event logging tetap aktif

---

## 🧪 Testing

### Unit Tests (`tests/`)
Test logika modul (bukan pipeline video penuh), menggunakan input sintetis:
- `test_counting.py`: crossing lines, per-class counter, ghost filter
- `test_kecepatan.py`: speed calculation, outlier filter, rolling average
- `test_kemacetan.py`: density, speed ratio, LOS classification
- `test_lawan_arah.py`: movement direction, confirmation threshold, cooldown
- `test_insiden.py`: stop detection, hard brake, crash IoU, cooldown
- `test_anpr.py`: lazy loading, OCR result, text cleanup, cooldown
- `test_stabilisasi.py`: persistence, decay, ghost suppression

### Integration Tests (`testing/video/`)
Test dengan video sample, panggil `main.py` via `_runner.py`:
- `test_deteksi.py` — tes deteksi saja
- `test_counting.py` — counting
- `test_kecepatan.py` — speed
- `test_kemacetan.py` — congestion
- `test_lawan_arah.py` — wrong-way
- `test_insiden.py` — incident
- `test_anpr.py` — ANPR
- `test_all_features.py` — semua fitur berurutan

---

## 📜 Git History (12 commits)

| Hash | Author | Tanggal | Deskripsi |
|------|--------|---------|-----------|
| `649813c` | Raihan-Brown | 2026-09-01 | fix: remove samples/models from git tracking |
| `70765c5` | Raihan-Brown | 2026-09-01 | update (merge feature branch) |
| `955a1b7` | Raihan-Brown | 2026-09-01 | **Major refactor:** +2142/-214 lines — tambah Project.md, config, 9 module, 7 test, testing harness, event_logger, ANPR, stabilisasi, lawan_arah, insiden |
| `9e6747f` | Abdurrokhman02 | 2026-08-27 | feat: crash-detection (fitur_kecelakaan.py) |
| `a3c2c68` | Abdurrokhman02 | 2026-08-27 | add: tuned-model best.pt |
| `24a998a` | Abdurrokhman02 | 2026-08-27 | add: sample videos |
| `4925346` | Abdurrokhman02 | 2026-08-27 | edit: gitignore |
| `b484d18` | Abdurrokhman02 | 2026-08-14 | Merge pull request #1 dari develop |
| `1fbd799` | Abdurrokhman02 | 2026-08-14 | fix: readme.md |
| `5841dd1` | Abdurrokhman02 | 2026-08-14 | add: readme (98 baris) |
| `11190f9` | Abdurrokhman02 | - | Phase 1 initial implementation |
| `31a6600` | Abdurrokhman02 | - | first (initial commit) |

---

## ⚡ Catatan Teknis Penting

1. **Wayland/Hyprland:** `main.py` set `QT_QPA_PLATFORM=xcb` & filter `FutureWarning` biar OpenCV Qt aman di Linux Wayland
2. **Memory cleanup periodik:** tiap ~5 detik (`frame % 40 == 0`), semua module panggil `bersihkan_memori()` — hapus tracker stale > 90 frame
3. **Frame buffer time machine:** `fitur_insiden.py` simpan deque 90 frame terakhir buat snapshot beberapa frame sebelum event terjadi
4. **ANPR lazy loading:** model YOLO plat + EasyOCR cuma di-load saat ANPR pertama kali aktif, bukan saat startup
5. **Persistence ≠ analitik:** `BoxPersistence` hanya untuk render visual — modul analitik pakai data asli dari ByteTrack
6. **Event log format JSONL:** append-only, tiap baris JSON independen — aman untuk long-running & mudah diproses per baris
7. **.gitignore:** exclude `.venv/`, `.pt`, `models/`, `samples/`, `output/`, media files, cache, image/log/json files, IDE folders

---

## 🔮 Potensi Pengembangan

| Area | Deskripsi |
|------|-----------|
| **Dependency lock** | Tambah `requirements.txt` atau `pyproject.toml` biar instalasi konsisten |
| **Setup guide** | Dokumentasi setup per OS (Windows + Linux) |
| **Multi-camera** | Sample config untuk beberapa CCTV sekaligus |
| **Accuracy eval** | Evaluasi akurasi model & threshold per lokasi CCTV |
| **Integrasi test** | Test end-to-end `main.py` mode headless dengan sample video |
| **CI/CD** | GitHub Actions untuk auto test tiap push |
| **Dashboard web** | Export data ke API/web dashboard (real-time monitoring) |
| **Database** | Simpan historical traffic data ke database (PostgreSQL, InfluxDB) |
| **Notification** | Alert real-time (Telegram, WhatsApp, Email) untuk insiden |