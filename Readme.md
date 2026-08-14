Berikut adalah draf `README.md` yang rapi, profesional, dan komprehensif untuk proyek ATCS kamu. Kamu bisa langsung menyalinnya!

---

```markdown
# 🚦 ATCS Traffic Intelligence - CCTV Monitoring System

Sistem *Adaptive Traffic Control System* (ATCS) berbasis *Computer Vision* dan *Artificial Intelligence* (AI). Proyek ini memproses aliran video CCTV lalu lintas secara *real-time* untuk mendeteksi kendaraan, menghitung volume, mengestimasi kecepatan, dan menentukan tingkat kemacetan.

Sistem ini dioptimalkan agar ringan, modular, dan dapat berjalan secara stabil selama 24/7 (dilengkapi *memory cleanup* otomatis).

## ✨ Fitur Utama

Sistem ini memiliki 4 modul pintar yang bisa diaktifkan/dinonaktifkan (*toggle*) secara langsung *real-time* tanpa menghentikan program:

1. **📦 Deteksi & Tracking (Live Count)**: Mendeteksi jenis kendaraan (Mobil, Motor, Bus, Truk) menggunakan **YOLOv8** dan melacak pergerakannya menggunakan **ByteTrack**. Menampilkan jumlah kendaraan yang ada di dalam *frame* saat ini.
2. **🧮 Vehicle Counting (Akumulasi)**: Menghitung total kendaraan yang melintasi garis virtual (*Entry & Exit*) dua arah.
3. **⚡ Speed Estimation**: Mengkalkulasi estimasi kecepatan rata-rata kendaraan (km/jam) berdasarkan selisih waktu tempuh antar dua garis virtual.
4. **🚥 Tingkat Kemacetan (LOS)**: Menganalisis *Traffic Density* pada area *Region of Interest* (ROI) untuk menentukan status jalan: **LANCAR, RAMAI/SEDANG, PADAT,** atau **MACET TOTAL**.

## 📂 Struktur Direktori

```text
aicctv/
│
├── main.py                        # Entry point aplikasi (Jalankan file ini)
├── config.py                      # File pengaturan (Garis kamera, kalibrasi jarak, UI)
├── cctv_cihaliwung_rekam.mp4      # (Opsional) File video input testing
│
├── models/                        # 📁 Folder tempat model AI disimpan
│   └── yolov8m.pt                 # Akan diunduh otomatis jika belum ada
│
└── modules/                       # 📁 Folder logika inti (Modular)
    ├── __init__.py                
    ├── fitur_counting.py          # Logika pencacah kendaraan
    ├── fitur_kecepatan.py         # Logika hitung kecepatan & waktu tempuh
    └── fitur_kemacetan.py         # Logika hitung kepadatan jalan (Density)

```

## 🛠️ Prasyarat & Instalasi

Proyek ini menggunakan paket ekosistem Python modern. Disarankan menggunakan `uv` untuk manajemen virtual environment yang super cepat (atau bisa juga menggunakan `pip` standar).

**1. Clone atau siapkan folder proyek ini**
**2. Install dependensi library**

```bash
# Menggunakan uv (Rekomendasi)
uv pip install ultralytics supervision opencv-python numpy

# ATAU menggunakan pip biasa
pip install ultralytics supervision opencv-python numpy

```

> **Catatan untuk pengguna Linux (Wayland / Hyprland):**
> Anda tidak perlu men-setting variabel environment secara manual di terminal. File `main.py` sudah diatur sedemikian rupa (`os.environ["QT_QPA_PLATFORM"] = "xcb"`) untuk mencegah *crash* Qt/Wayland dan menyembunyikan *spam* log pada terminal.

## 🚀 Cara Menjalankan

Jalankan eksekusi melalui file `main.py`:

```bash
uv run main.py

```

*(Model `yolov8m.pt` akan terunduh otomatis ke dalam folder `models/` pada saat pertama kali dijalankan).*

## 🎮 Kontrol & Interaksi UI

Aplikasi ini dilengkapi antarmuka interaktif. Kamu bisa mematikan atau menyalakan fitur AI langsung di layar menggunakan **Klik Mouse** pada kotak menu di atas layar, atau menggunakan **Shortcut Keyboard**:

| Fitur | Klik UI (Layar) | Hotkey Keyboard | Keterangan |
| --- | --- | --- | --- |
| **Deteksi** | `DETEKSI: ON/OFF` | `1` | Menampilkan Bounding box, label, & hitungan Live. |
| **Counting** | `COUNTING: ON/OFF` | `2` | Menampilkan Garis CCTV & hitungan total Akumulasi. |
| **Kecepatan** | `KECEPATAN: ON/OFF` | `3` | Mengaktifkan estimasi rata-rata kecepatan. |
| **Kemacetan** | `KEMACETAN: ON/OFF` | `4` | Menganalisis dan menampilkan status kemacetan. |
| **Keluar** | - | `q` | Menghentikan program dengan aman. |

## ⚙️ Konfigurasi (`config.py`)

Jika kamu ingin menerapkan proyek ini ke kamera CCTV jalan yang berbeda, cukup sesuaikan parameter di dalam `config.py`:

* `video_source`: Path video atau stream RTSP (Ganti jadi `0` untuk webcam PC).
* `jarak_garis_meter`: Jarak fisik sesungguhnya (dalam meter) antara Garis 1 dan Garis 2 di lapangan. Sangat penting untuk akurasi *Speed Estimation*.
* `panjang_segmen_meter` & `jumlah_lajur`: Parameter untuk menghitung *Density* kemacetan jalan.

## 🤝 Lisensi & Kredit

* **AI Object Detection**: [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)
* **Tracking & Analytics Tools**: [Roboflow Supervision](https://github.com/roboflow/supervision)

```

```