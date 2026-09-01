# Project Overview

## Nama Project
ATCS Traffic Intelligence - CCTV Monitoring System

## Tujuan
Project ini sistem monitoring lalu lintas berbasis Computer Vision dan AI. Input utama video CCTV lalu lintas, lalu sistem jalankan deteksi kendaraan, tracking, counting, estimasi kecepatan, analisis kemacetan, deteksi lawan arah, deteksi insiden, dan ANPR. Output bisa tampil real-time di GUI, headless untuk testing/CI, simpan video anotasi, dan simpan event snapshot + log JSONL.

## Apa Yang Sudah Dikerjakan
- Deteksi kendaraan pakai YOLOv8.
- Tracking pakai ByteTrack.
- Counting dua garis virtual untuk hitung akumulasi, in, dan out.
- Estimasi kecepatan dari waktu tempuh antar garis.
- Klasifikasi kemacetan dari density + speed.
- Modul lawan arah berbasis riwayat posisi track.
- Modul insiden untuk kendaraan berhenti, pengereman mendadak, dan potensi tabrakan.
- Modul ANPR untuk baca plat nomor dengan backend OCR pluggable.
- Event logging ke `output/events/events.jsonl` plus snapshot JPG.
- Stabilisasi tampilan bounding box dengan persistence saat deteksi hilang sesaat.
- Mode GUI interaktif pakai tombol layar dan hotkey.
- Mode headless, output video, dan output report JSON.
- Test unit per modul inti di folder `tests/`.

## Cara Kerja Singkat
1. Video dibuka dari `config.py` atau argumen CLI.
2. YOLO deteksi kendaraan.
3. ByteTrack kasih tracker ID.
4. Modul counting, speed, congestion, wrong-way, incident, dan ANPR baca hasil track yang sama.
5. UI render overlay, tombol toggle, garis, label, dan event strip.
6. Event penting disimpan ke log dan snapshot.
7. Periodik, memori state dibersihkan biar stabil jalan lama.

## Entry Point
- `main.py` adalah entry point utama.
- `testing.py` adalah runner lama / eksperimen awal, bukan jalur utama.

## Struktur File
```text
ATCS-AI/
├── main.py                 # aplikasi utama real-time / headless
├── config.py               # konfigurasi kamera, model, tracker, UI, event
├── testing.py              # skrip uji lama / prototipe
├── Readme.md               # dokumentasi ringkas project
├── .gitignore              # file cache, venv, output
├── best_kendaraan.pt       # aset model kendaraan
├── best_plat.pt            # model tambahan / aset ANPR
├── models/
│   ├── yolov8n.pt          # model YOLO ringan
│   └── yolov8m.pt          # model YOLO lebih besar
├── modules/
│   ├── __init__.py
│   ├── fitur_counting.py   # counter kendaraan dan arah in/out
│   ├── fitur_kecepatan.py  # hitung speed dan average speed
│   ├── fitur_kemacetan.py  # klasifikasi status kemacetan
│   ├── fitur_lawan_arah.py # deteksi kendaraan melawan arah
│   ├── fitur_insiden.py    # deteksi berhenti, hard brake, tabrakan
│   ├── fitur_anpr.py       # lokalisasi plat + OCR backend
│   ├── fitur_stabilisasi.py# persistence bounding box
│   └── event_logger.py     # simpan log event dan snapshot
├── tests/
│   ├── test_counting.py
│   ├── test_kecepatan.py
│   ├── test_kemacetan.py
│   ├── test_lawan_arah.py
│   ├── test_insiden.py
│   ├── test_anpr.py
│   └── test_stabilisasi.py
├── samples/
│   ├── sample.mp4
│   ├── sampleuji.mp4
│   ├── sampleuji2.mp4
│   └── samplesiang.mp4
├── output/
│   └── events/
│       ├── events.jsonl
│       └── snapshot JPG event
└── .venv/                  # virtual environment lokal
```

## Library Yang Dipakai
### Runtime utama
- `opencv-python` / `cv2` untuk baca video, render overlay, writer output, dan operasi gambar.
- `numpy` untuk array, mask, perhitungan numeric.
- `ultralytics` untuk model YOLOv8.
- `supervision` untuk `Detections`, `ByteTrack`, `LineZone`, `PolygonZone`, annotator, dan smoother.

### Opsional / fallback ANPR
- `easyocr` jika tersedia.
- `pytesseract` jika tersedia.

### Standard library Python
- `os`
- `warnings`
- `argparse`
- `json`
- `collections.deque`, `collections.defaultdict`
- `datetime`

## Dependensi Yang Terlihat Dari Repo
Repo ini belum punya `requirements.txt` atau `pyproject.toml`, jadi dependensi dibaca dari import kode dan README.

Minimal paket eksternal yang dipakai:
- `ultralytics`
- `supervision`
- `opencv-python`
- `numpy`

Opsional untuk ANPR:
- `easyocr`
- `pytesseract`

## Konfigurasi Penting
### `config.py`
- `CAM_CONFIG.video_source`: sumber video, file sample, RTSP, atau webcam.
- `CAM_CONFIG.jarak_garis_meter`: jarak fisik antar garis untuk speed.
- `CAM_CONFIG.panjang_segmen_meter` dan `jumlah_lajur`: input density kemacetan.
- `CAM_CONFIG.arah_lalu_lintas`: arah normal arus kamera untuk lawan arah.
- `MODEL_CONFIG.path`: path model YOLO utama. Nilai saat ini `./best.pt`, tetapi file `best.pt` tidak ada di root repo.
- `MODEL_CONFIG.conf`: confidence threshold deteksi.
- `TRACKER_CONFIG`: parameter ByteTrack.
- `SMOOTHER_CONFIG`: parameter stabilisasi bounding box.
- `GARIS`: posisi garis virtual relatif frame.
- `KEMACETAN_CONFIG`: ambang status lancar/padat/macet.
- `LAWAN_ARAH_CONFIG`: ambang arah dan cooldown.
- `INSIDEN_CONFIG`: ambang berhenti, hard brake, dan tabrakan.
- `ANPR_CONFIG`: backend OCR, confidence, cooldown, dan ukuran minimal plat.
- `OUTPUT_CONFIG`: folder output event dan snapshot.
- `FLAGS` dan `BUTTONS`: toggle UI.

## CLI Yang Tersedia Di `main.py`
- `--source`: sumber video atau RTSP.
- `--model`: path model YOLO.
- `--conf`: threshold confidence.
- `--headless`: tanpa GUI.
- `--max-frames`: batasi frame proses.
- `--output-video`: simpan video anotasi.
- `--output-report`: simpan ringkasan JSON.
- `--no-lawan-arah`: matikan modul lawan arah.
- `--no-insiden`: matikan modul insiden.
- `--no-anpr`: matikan modul ANPR.

## Output Yang Dihasilkan
- Tampilan live GUI dengan dashboard dan tombol toggle.
- Video hasil anotasi jika `--output-video` dipakai.
- Laporan statistik JSON jika `--output-report` dipakai.
- Log event di `output/events/events.jsonl`.
- Snapshot JPG untuk event lawan arah, insiden, dan ANPR.

## Mode Interaksi
### Hotkey
- `1` = deteksi
- `2` = counting
- `3` = kecepatan
- `4` = kemacetan
- `5` = lawan arah
- `6` = insiden
- `7` = ANPR
- `q` = keluar

### UI klik
Ada tombol layar untuk toggle fitur utama tanpa stop program.

## Test Yang Ada
Folder `tests/` berisi unit test untuk:
- counting
- speed estimation
- congestion classification
- wrong-way detection
- incident detection
- ANPR localization / cleanup / cooldown
- box persistence

Test ini fokus ke logika modul, bukan pipeline video penuh.

## Catatan Teknis Penting
- `main.py` set `QT_QPA_PLATFORM=xcb` supaya OpenCV Qt aman di Wayland/Hyprland.
- `main.py` dan `testing.py` filter `FutureWarning` untuk rapikan log.
- `BoxPersistence` cuma untuk render, bukan untuk ubah hasil analitik.
- ANPR dibuat lazy supaya model OCR tidak selalu di-load saat startup.
- Event log pakai format JSONL biar gampang diproses per baris.

## Status Repo
- Struktur project sudah modular.
- Test modul inti sudah ada.
- File output runtime dan model berat masih ikut di repo lokal.
- Belum ada file lock dependency resmi seperti `requirements.txt` atau `pyproject.toml`.

## Potensi Pengembangan Lanjut
- Tambah `requirements.txt` atau `pyproject.toml` biar instalasi konsisten.
- Tambah dokumentasi setup per OS.
- Tambah test integrasi untuk `main.py` mode headless.
- Tambah sample konfigurasi untuk beberapa kamera.
- Tambah evaluasi akurasi model dan threshold per lokasi CCTV.
