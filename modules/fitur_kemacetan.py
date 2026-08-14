# fitur_kemacetan.py

def cek_status_kemacetan(jumlah_di_roi, panjang_segmen_meter, jumlah_lajur):
    """Menghitung density dan mengembalikan teks status & warna RGB."""
    
    # Konversi meter ke kilometer untuk rumus density (veh/km/lane)
    panjang_km = panjang_segmen_meter / 1000.0
    density = (jumlah_di_roi / panjang_km) / jumlah_lajur
    
    if density <= 10:
        return "LANCAR", (0, 255, 0)         # Hijau
    elif density <= 22:
        return "RAMAI / SEDANG", (0, 255, 255) # Kuning
    elif density <= 35:
        return "PADAT", (0, 165, 255)        # Oranye
    else:
        return "MACET TOTAL", (0, 0, 255)      # Merah