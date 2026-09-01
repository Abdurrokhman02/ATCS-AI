# fitur_kecepatan.py
import numpy as np

MIN_KECEPATAN_KMH = 2.0     # di bawah ini dianggap data tidak valid / glitch
MAX_KECEPATAN_KMH = 200.0   # di atas ini dianggap outlier / salah deteksi


def hitung_kecepatan_kendaraan(tid, crossed_l1_i, crossed_l2_i, t_sekarang,
                               timestamp_l1, timestamp_l2, speeds_list, jarak_meter):
    """Menghitung kecepatan jika kendaraan terpantau di kedua garis (Dua Arah).

    Fungsi ini sekaligus mencatat waktu lintas Garis 1 dan Garis 2 (shared
    dengan VehicleCounter via `setdefault`, tidak bentrok).
    """
    if tid is None:
        return

    if crossed_l1_i:
        timestamp_l1.setdefault(tid, t_sekarang)
    if crossed_l2_i:
        timestamp_l2.setdefault(tid, t_sekarang)

    kecepatan = None
    if crossed_l2_i and tid in timestamp_l1 and timestamp_l2.get(tid) != timestamp_l1.get(tid):
        tt = abs(timestamp_l2[tid] - timestamp_l1[tid])
        kecepatan = (jarak_meter / tt) * 3.6 if tt > 0.2 else None
    elif crossed_l1_i and tid in timestamp_l2 and timestamp_l2.get(tid) != timestamp_l1.get(tid):
        tt = abs(timestamp_l2[tid] - timestamp_l1[tid])
        kecepatan = (jarak_meter / tt) * 3.6 if tt > 0.2 else None

    if kecepatan is not None and MIN_KECEPATAN_KMH <= kecepatan <= MAX_KECEPATAN_KMH:
        speeds_list.append(kecepatan)


def get_rata_rata_kecepatan(speeds_list, n_sample=20):
    """Menghitung rata-rata dari n sampel kecepatan terakhir."""
    return float(np.mean(speeds_list[-n_sample:])) if speeds_list else 0.0
