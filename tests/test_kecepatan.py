# tests/test_kecepatan.py
import unittest
from modules.fitur_kecepatan import hitung_kecepatan_kendaraan, get_rata_rata_kecepatan


class TestKecepatan(unittest.TestCase):
    def test_kecepatan_dua_garis(self):
        ts1, ts2 = {}, {}
        speeds = []
        # Kendaraan lintasi Garis 1 pada t=1.0
        hitung_kecepatan_kendaraan(1, True, False, 1.0, ts1, ts2, speeds, 20.0)
        # Lintasi Garis 2 pada t=2.0 -> tt=1s, v = 20 m/1s * 3.6 = 72 km/h
        hitung_kecepatan_kendaraan(1, False, True, 2.0, ts1, ts2, speeds, 20.0)
        self.assertEqual(len(speeds), 1)
        self.assertAlmostEqual(speeds[0], 72.0, places=4)

    def test_kecepatan_arah_sebaliknya(self):
        ts1, ts2 = {}, {}
        speeds = []
        hitung_kecepatan_kendaraan(2, False, True, 1.0, ts1, ts2, speeds, 20.0)
        hitung_kecepatan_kendaraan(2, True, False, 2.0, ts1, ts2, speeds, 20.0)
        self.assertEqual(len(speeds), 1)
        self.assertAlmostEqual(speeds[0], 72.0, places=4)

    def test_waktu_tidak_logis_difilter(self):
        ts1, ts2 = {}, {}
        speeds = []
        # ts1 dicatat (mis. oleh counter) lalu lintasi L2 dengan selisih < 0.2s
        ts1[3] = 1.95
        hitung_kecepatan_kendaraan(3, False, True, 2.0, ts1, ts2, speeds, 20.0)
        self.assertEqual(len(speeds), 0)

    def test_outlier_difilter(self):
        ts1, ts2 = {}, {}
        speeds = []
        # Kecepatan absurd (> 200 km/h) dan terlalu rendah (< 2 km/h)
        ts1[4] = 1.0
        ts1[5] = 1.0
        # v = 20/0.0001*3.6 -> sangat besar -> harus difilter
        hitung_kecepatan_kendaraan(4, False, True, 1.0001, ts1, ts2, speeds, 20.0)
        self.assertEqual(len(speeds), 0)
        # v = 20/10*3.6 = 7.2 -> valid
        hitung_kecepatan_kendaraan(5, False, True, 11.0, ts1, ts2, speeds, 20.0)
        self.assertEqual(len(speeds), 1)

    def test_rata_rata(self):
        speeds = [60.0, 70.0, 80.0]
        self.assertAlmostEqual(get_rata_rata_kecepatan(speeds), 70.0)
        self.assertEqual(get_rata_rata_kecepatan([]), 0.0)


if __name__ == "__main__":
    unittest.main()
