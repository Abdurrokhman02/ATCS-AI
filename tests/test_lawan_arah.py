# tests/test_lawan_arah.py
import unittest

import numpy as np
import supervision as sv

from modules.fitur_lawan_arah import WrongWayDetector


def make_det(x1, y1, x2, y2, tid):
    return sv.Detections(
        xyxy=np.array([[x1, y1, x2, y2]], dtype=float),
        class_id=np.array([2]),
        confidence=np.array([0.9]),
        tracker_id=np.array([tid], dtype=int),
    )


class TestWrongWayDetector(unittest.TestCase):
    def test_deteksi_melawan_arah(self):
        # Arus normal: atas -> bawah (y bertambah). Kendaraan bergerak naik (y mengecil).
        det = WrongWayDetector("atas_ke_bawah", min_kecepatan_px=3.0,
                               frame_konfirmasi=5, cooldown_detik=20.0)
        flagged = 0
        events = []
        for t in range(1, 15):
            y = 500 - 10 * t
            mask, ev = det.update(make_det(100, y - 40, 160, y, 1), float(t))
            if mask[0]:
                flagged += 1
            events.extend(ev)
        self.assertGreater(flagged, 0, "Kendaraan melawan arah harus ter-flag")
        self.assertEqual(len(events), 1, "Satu alert per kendaraan dalam cooldown")

    def test_arah_normal_tidak_terflag(self):
        det = WrongWayDetector("atas_ke_bawah", min_kecepatan_px=3.0,
                               frame_konfirmasi=5, cooldown_detik=20.0)
        flagged = 0
        for t in range(1, 30):
            y = 500 + 10 * t
            mask, _ = det.update(make_det(100, y - 40, 160, y, 1), float(t))
            flagged += int(mask[0])
        self.assertEqual(flagged, 0)

    def test_gerak_terlalu_lambat_tidak_terflag(self):
        # Melawan arah tapi sangat lambat (di bawah min_kecepatan_px)
        det = WrongWayDetector("atas_ke_bawah", min_kecepatan_px=10.0,
                               frame_konfirmasi=5, cooldown_detik=20.0)
        flagged = 0
        for t in range(1, 30):
            y = 500 - 1 * t  # 1 px/frame < 10 px/s
            mask, _ = det.update(make_det(100, y - 40, 160, y, 1), float(t))
            flagged += int(mask[0])
        self.assertEqual(flagged, 0)

    def test_bersihkan_memori(self):
        det = WrongWayDetector()
        for t in range(1, 6):
            det.update(make_det(100, 500 - 10 * t, 160, 460 - 10 * t, 1), float(t))
        det.bersihkan_memori(1000.0)
        self.assertEqual(len(det._history), 0)


if __name__ == "__main__":
    unittest.main()
