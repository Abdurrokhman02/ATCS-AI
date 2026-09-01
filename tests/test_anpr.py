# tests/test_anpr.py
import unittest

import cv2
import numpy as np

from modules.fitur_anpr import ANPRReader

CFG = {"min_conf": 0.3, "cooldown_detik": 60.0, "min_w_plat": 40, "min_h_plat": 14,
       "backend": "auto"}


def buat_gambar_dengan_plat():
    """Buat gambar 400x200: latar putih, plat gelap (rasio 3.3) dengan 'teks' putih."""
    img = np.full((200, 400, 3), 255, dtype=np.uint8)
    cv2.rectangle(img, (80, 60), (280, 120), (40, 40, 40), -1)   # plat 200x60, ar 3.33
    for i, ch in enumerate("B 1234"):
        x = 95 + i * 30
        cv2.putText(img, ch, (x, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
    return img


class TestANPR(unittest.TestCase):
    def test_lokalisasi_plat(self):
        reader = ANPRReader(CFG)
        img = buat_gambar_dengan_plat()
        lok = reader.lokalisasi_plat(img)
        self.assertIsNotNone(lok, "Kandidat plat harus ditemukan")
        x, y, w, h = lok
        self.assertGreaterEqual(w, 40)
        self.assertGreaterEqual(h, 14)
        ar = w / max(h, 1)
        self.assertTrue(2.0 <= ar <= 6.5, f"Aspect ratio plat tidak masuk akal: {ar:.2f}")

    def test_tidak_ada_kandidat(self):
        reader = ANPRReader(CFG)
        img = np.full((200, 400, 3), 255, dtype=np.uint8)  # polos
        self.assertIsNone(reader.lokalisasi_plat(img))

    def test_bersihkan_plat(self):
        self.assertEqual(ANPRReader._bersihkan_plat("D 1234 ABC"), "D1234ABC")
        self.assertEqual(ANPRReader._bersihkan_plat("b-4321 xy"), "B4321XY")
        self.assertEqual(ANPRReader._bersihkan_plat(""), "")

    def test_baca_plat_backend_auto(self):
        reader = ANPRReader(CFG)
        img = buat_gambar_dengan_plat()
        hasil = reader.baca_plat(img)
        self.assertIn("plat", hasil)
        self.assertIn("conf", hasil)
        self.assertIn("backend", hasil)
        self.assertIsInstance(hasil["backend"], str)

    def test_cooldown_per_kendaraan(self):
        reader = ANPRReader(CFG)
        img = buat_gambar_dengan_plat()
        hasil1 = reader.baca_untuk_kendaraan(1, img, 10.0)
        hasil2 = reader.baca_untuk_kendaraan(1, img, 10.5)   # masih dalam cooldown
        hasil3 = reader.baca_untuk_kendaraan(1, img, 71.0)   # lewat cooldown
        self.assertIsNotNone(hasil1)
        self.assertIsNone(hasil2)
        self.assertIsNotNone(hasil3)


if __name__ == "__main__":
    unittest.main()