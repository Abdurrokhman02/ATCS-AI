# fitur_kecepatan.py
import numpy as np

def hitung_kecepatan_kendaraan(tid, crossed_l1_i, crossed_l2_i, t_sekarang, timestamp_l1, timestamp_l2, speeds_list, jarak_meter):
    """Menghitung kecepatan jika kendaraan terpantau di kedua garis (Dua Arah)."""
    
    # Arah: Garis 1 menuju Garis 2
    if crossed_l2_i and tid in timestamp_l1 and tid not in timestamp_l2:
        tt = t_sekarang - timestamp_l1[tid]
        if tt > 0.2: # Anti-glitch (filter waktu tidak logis)
            speeds_list.append((jarak_meter / tt) * 3.6)
            
    # Arah: Garis 2 menuju Garis 1
    elif crossed_l1_i and tid in timestamp_l2 and tid not in timestamp_l1:
        tt = t_sekarang - timestamp_l2[tid]
        if tt > 0.2:
            speeds_list.append((jarak_meter / tt) * 3.6)

def get_rata_rata_kecepatan(speeds_list):
    """Menghitung rata-rata dari 20 sampel kecepatan terakhir."""
    return float(np.mean(speeds_list[-20:])) if speeds_list else 0.0