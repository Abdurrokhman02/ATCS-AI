# fitur_kemacetan.py

LANCAR = "LANCAR"
PADAT = "PADAT"
MACET = "MACET"

WARNA_LANCAR = (0, 255, 0)      # Hijau
WARNA_PADAT = (0, 165, 255)     # Oranye
WARNA_MACET = (0, 0, 255)       # Merah


def cek_status_kemacetan(jumlah_di_roi, panjang_segmen_meter, jumlah_lajur,
                         avg_speed_kmh, free_flow_speed_kmh, cfg=None):
    """Klasifikasi status kemacetan (LANCAR/PADAT/MACET).

    PRD 6.4: klasifikasi berdasarkan kombinasi kepadatan kendaraan (veh/km/lane)
    dan rata-rata kecepatan. Jika data kecepatan belum tersedia (0.0), klasifikasi
    dilakukan dari kepadatan saja agar tidak salah menandai MACET.
    """
    if cfg is None:
        cfg = {}

    batas_lancar = cfg.get("density_batas_lancar", 12.0)
    batas_padat = cfg.get("density_batas_padat", 25.0)
    rasio_padat = cfg.get("rasio_kecepatan_padat", 0.55)
    rasio_macet = cfg.get("rasio_kecepatan_macet", 0.30)

    panjang_km = max(panjang_segmen_meter, 1.0) / 1000.0
    density = (jumlah_di_roi / panjang_km) / max(jumlah_lajur, 1)

    # Tanpa data kecepatan -> klasifikasi density saja
    if avg_speed_kmh <= 0.0:
        if density < batas_lancar:
            return LANCAR, WARNA_LANCAR
        if density < batas_padat:
            return PADAT, WARNA_PADAT
        return MACET, WARNA_MACET

    rasio_kecepatan = avg_speed_kmh / max(free_flow_speed_kmh, 1.0)

    # Kecepatan sangat tinggi (di atas free-flow) + kepadatan tidak ekstrem -> LANCAR
    if rasio_kecepatan > 1.0 and density < batas_padat:
        return LANCAR, WARNA_LANCAR

    # Kepadatan rendah + kecepatan normal -> LANCAR
    if density < batas_lancar and rasio_kecepatan > rasio_padat:
        return LANCAR, WARNA_LANCAR

    # Kepadatan tinggi ATAU kecepatan sangat rendah -> MACET
    if density >= batas_padat or rasio_kecepatan <= rasio_macet:
        return MACET, WARNA_MACET

    # Sisanya -> PADAT
    return PADAT, WARNA_PADAT
