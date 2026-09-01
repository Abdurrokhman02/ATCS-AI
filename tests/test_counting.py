# tests/test_counting.py
import unittest
from modules.fitur_counting import VehicleCounter


class TestVehicleCounter(unittest.TestCase):
    def test_satu_lintas_dihitung_sekali(self):
        c = VehicleCounter()
        # Kendaraan masuk dari atas melewati Garis 1 (c1_in)
        c.update(1, 2, True, False, False, False, 1.0)
        self.assertEqual(c.get_total_akumulasi(), 1)
        # Lalu melewati Garis 2 di frame berikutnya -> TIDAK double count
        c.update(1, 2, False, False, True, False, 2.0)
        self.assertEqual(c.get_total_akumulasi(), 1)

    def test_tidak_double_count_lintas_dua_garis(self):
        c = VehicleCounter()
        # Kendaraan dari atas: lintasi L1 (in), lalu L2 (out) -> tetap 1 kendaraan
        c.update(1, 3, True, False, False, False, 1.0)
        c.update(1, 3, False, False, True, False, 1.5)
        self.assertEqual(c.get_total_akumulasi(), 1)
        self.assertEqual(c.get_in_out(), (1, 0))  # tercatat sekali, arah "in"

    def test_arah_in_out(self):
        c = VehicleCounter()
        # Masuk dari atas: lintasi Garis 1 ke bawah -> "in"
        c.update(1, 2, True, False, False, False, 1.0)
        # Masuk dari bawah: lintasi Garis 2 ke atas -> "in"
        c.update(2, 3, False, False, False, True, 1.5)
        # Keluar ke atas: lintasi Garis 1 ke atas -> "out"
        c.update(3, 7, False, True, False, False, 2.0)
        # Keluar ke bawah: lintasi Garis 2 ke bawah -> "out"
        c.update(4, 5, False, False, True, False, 2.5)
        self.assertEqual(c.get_in_out(), (2, 2))
        self.assertEqual(c.get_total_akumulasi(), 4)
        self.assertEqual(c.class_counts[2], 1)
        self.assertEqual(c.class_counts[3], 1)
        self.assertEqual(c.class_counts[7], 1)
        self.assertEqual(c.class_counts[5], 1)

    def test_tid_none_diabaikan(self):
        c = VehicleCounter()
        c.update(None, 2, True, False, False, False, 1.0)
        self.assertEqual(c.get_total_akumulasi(), 0)

    def test_bersihkan_memori(self):
        c = VehicleCounter()
        c.update(1, 2, True, False, False, False, 1.0)
        c.bersihkan_memori(100.0)
        self.assertEqual(c.get_total_akumulasi(), 1)  # total tetap, hanya timestamp dibersihkan


if __name__ == "__main__":
    unittest.main()
