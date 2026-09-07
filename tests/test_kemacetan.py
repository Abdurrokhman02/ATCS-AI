# tests/test_kemacetan.py
import unittest
from modules.fitur_kemacetan import cek_status_kemacetan, LANCAR, PADAT, MACET

CFG = {
    "density_batas_lancar": 12.0,
    "density_batas_padat": 25.0,
    "rasio_kecepatan_padat": 0.55,
    "rasio_kecepatan_macet": 0.30,
}

# 500 m segmen, 2 lajur
PANJANG = 500.0
LAJUR = 2


def den(jumlah):
    return (jumlah / (PANJANG / 1000.0)) / LAJUR


class TestKemacetan(unittest.TestCase):
    def test_density_helper(self):
        self.assertEqual(den(6), 6.0)
        self.assertEqual(den(12), 12.0)
        self.assertEqual(den(25), 25.0)

    def test_lancar_density_rendah_speed_normal(self):
        # density 6 <= 12 dan rasio speed 0.875 > 0.55 -> LANCAR
        status, _ = cek_status_kemacetan(6, PANJANG, LAJUR, 35.0, 40.0, CFG)
        self.assertEqual(status, LANCAR)

    def test_lancar_speed_sangat_tinggi_moderate_density(self):
        # density 20 (moderate, < 25) + sangat tinggi (ratio > 1.0, speed > free_flow) -> LANCAR
        # speed 180 km/h, free_flow 40 -> ratio 4.5 > 1.0, density 20 < 25
        status, _ = cek_status_kemacetan(20, PANJANG, LAJUR, 180.0, 40.0, CFG)
        self.assertEqual(status, LANCAR)
        # density 24 (di batas batas_padat) + very high speed -> LANCAR
        status, _ = cek_status_kemacetan(24, PANJANG, LAJUR, 100.0, 40.0, CFG)
        self.assertEqual(status, LANCAR)
        # density 25 (tepat batas_padat) + very high speed -> MACET (density >= 25)
        status, _ = cek_status_kemacetan(25, PANJANG, LAJUR, 180.0, 40.0, CFG)
        self.assertEqual(status, MACET)

    def test_macet_density_tinggi(self):
        # density 25 >= 25 -> MACET walau speed normal
        status, _ = cek_status_kemacetan(25, PANJANG, LAJUR, 35.0, 40.0, CFG)
        self.assertEqual(status, MACET)

    def test_tanpa_data_kecepatan(self):
        # avg_speed 0 -> density saja
        self.assertEqual(cek_status_kemacetan(6, PANJANG, LAJUR, 0.0, 40.0, CFG)[0], LANCAR)
        self.assertEqual(cek_status_kemacetan(12, PANJANG, LAJUR, 0.0, 40.0, CFG)[0], PADAT)
        self.assertEqual(cek_status_kemacetan(30, PANJANG, LAJUR, 0.0, 40.0, CFG)[0], MACET)

    def test_kombinasi_speed_dan_density(self):
        # density rendah tapi kecepatan sangat rendah (rasio 0.125 <= 0.30) -> MACET
        status, _ = cek_status_kemacetan(6, PANJANG, LAJUR, 5.0, 40.0, CFG)
        self.assertEqual(status, MACET)
        # density sedang (12) + kecepatan normal -> PADAT (batas lancar butuh > rasio_padat)
        status, _ = cek_status_kemacetan(12, PANJANG, LAJUR, 35.0, 40.0, CFG)
        self.assertEqual(status, PADAT)
        # density 10, rasio 0.5 (<= 0.55) -> PADAT
        status, _ = cek_status_kemacetan(10, PANJANG, LAJUR, 20.0, 40.0, CFG)
        self.assertEqual(status, PADAT)
        # density 10, rasio 0.8 (normal, > 0.55 tapi < 1.0) -> LANCAR (density < 12, ratio > 0.55)
        status, _ = cek_status_kemacetan(10, PANJANG, LAJUR, 32.0, 40.0, CFG)
        self.assertEqual(status, LANCAR)

    def test_hanya_tiga_status(self):
        statuses = set()
        for d in range(0, 30):
            st, _ = cek_status_kemacetan(d, PANJANG, LAJUR, 30.0, 40.0, CFG)
            statuses.add(st)
        self.assertTrue(statuses.issubset({LANCAR, PADAT, MACET}))
        self.assertTrue(LANCAR in statuses and PADAT in statuses and MACET in statuses)


if __name__ == "__main__":
    unittest.main()
