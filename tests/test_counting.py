import unittest

from modules.fitur_counting import VehicleCounter


class TestVehicleCounter(unittest.TestCase):

    def test_baru_satu_garis_belum_dihitung(self):
        c = VehicleCounter()

        # LINE1 dilewati
        c.update(1, 2, True, False, False, False, 1.0)

        self.assertEqual(c.get_total_akumulasi(), 0)
        self.assertEqual(c.get_in_out(), (0, 0))

    def test_line1_ke_line2_dihitung_satu(self):
        c = VehicleCounter()

        # Kendaraan bergerak atas -> bawah
        c.update(1, 2, True, False, False, False, 1.0)

        # Belum selesai
        self.assertEqual(c.get_total_akumulasi(), 0)

        # Melewati LINE2
        c.update(1, 2, False, False, True, False, 1.5)

        self.assertEqual(c.get_total_akumulasi(), 1)
        self.assertEqual(c.get_in_out(), (1, 1))

    def test_line2_ke_line1_dihitung_satu(self):
        c = VehicleCounter()

        # Kendaraan bergerak bawah -> atas
        c.update(1, 2, False, False, True, False, 1.0)

        # Belum selesai
        self.assertEqual(c.get_total_akumulasi(), 0)

        # Melewati LINE1
        c.update(1, 2, True, False, False, False, 1.5)

        self.assertEqual(c.get_total_akumulasi(), 1)
        self.assertEqual(c.get_in_out(), (1, 1))

    def test_tidak_double_count(self):
        c = VehicleCounter()

        # LINE2 -> LINE1
        c.update(1, 3, False, False, True, False, 1.0)
        c.update(1, 3, True, False, False, False, 1.5)

        self.assertEqual(c.get_total_akumulasi(), 1)
        self.assertEqual(c.get_in_out(), (1, 1))

        # Crossing lagi tidak boleh menambah count
        c.update(1, 3, False, False, True, False, 2.0)

        self.assertEqual(c.get_total_akumulasi(), 1)
        self.assertEqual(c.get_in_out(), (1, 1))

    def test_kendaraan_beda_arah(self):
        c = VehicleCounter()

        # ID 1: LINE1 -> LINE2
        c.update(1, 2, True, False, False, False, 1.0)
        c.update(1, 2, False, False, True, False, 1.5)

        # ID 2: LINE2 -> LINE1
        c.update(2, 3, False, False, True, False, 2.0)
        c.update(2, 3, True, False, False, False, 2.5)

        self.assertEqual(c.get_total_akumulasi(), 2)
        self.assertEqual(c.get_in_out(), (2, 2))

    def test_tid_none_diabaikan(self):
        c = VehicleCounter()

        c.update(None, 2, True, False, False, False, 1.0)

        self.assertEqual(c.get_total_akumulasi(), 0)
        self.assertEqual(c.get_in_out(), (0, 0))

    def test_bersihkan_pending(self):
        c = VehicleCounter(counted_timeout=30.0)

        # Baru lewat LINE1, belum selesai
        c.update(1, 2, True, False, False, False, 1.0)

        self.assertEqual(c.get_total_akumulasi(), 0)
        self.assertIn(1, c._pending)

        # Belum timeout
        c.bersihkan_memori(10.0)

        self.assertIn(1, c._pending)

    def test_pending_cleanup_after_timeout(self):
        c = VehicleCounter(counted_timeout=30.0)

        c.update(1, 2, True, False, False, False, 1.0)

        self.assertIn(1, c._pending)

        c.bersihkan_memori(100.0)

        self.assertNotIn(1, c._pending)

    def test_counted_cleanup_after_timeout(self):
        c = VehicleCounter(counted_timeout=30.0)

        # Selesaikan LINE1 -> LINE2
        c.update(1, 2, True, False, False, False, 1.0)
        c.update(1, 2, False, False, True, False, 2.0)

        self.assertEqual(c.get_total_akumulasi(), 1)
        self.assertIn(1, c._counted)

        c.bersihkan_memori(100.0)

        self.assertNotIn(1, c._counted)

    def test_recount_after_timeout(self):
        c = VehicleCounter(counted_timeout=30.0)

        c.update(1, 2, True, False, False, False, 1.0)
        c.update(1, 2, False, False, True, False, 2.0)

        self.assertEqual(c.get_total_akumulasi(), 1)

        c.bersihkan_memori(100.0)

        # Tracker ID yang sama boleh digunakan kembali
        c.update(1, 2, True, False, False, False, 200.0)
        c.update(1, 2, False, False, True, False, 201.0)

        self.assertEqual(c.get_total_akumulasi(), 2)

    def test_class_count(self):
        c = VehicleCounter()

        # Mobil
        c.update(1, 2, True, False, False, False, 1.0)
        c.update(1, 2, False, False, True, False, 2.0)

        # Motor
        c.update(2, 3, False, False, True, False, 3.0)
        c.update(2, 3, True, False, False, False, 4.0)

        self.assertEqual(c.class_counts[2], 1)
        self.assertEqual(c.class_counts[3], 1)
        self.assertEqual(c.get_total_akumulasi(), 2)


if __name__ == "__main__":
    unittest.main()