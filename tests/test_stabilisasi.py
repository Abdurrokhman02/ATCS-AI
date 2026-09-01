# tests/test_stabilisasi.py
import unittest

import numpy as np
import supervision as sv

from modules.fitur_stabilisasi import BoxPersistence


def make_det(x1, y1, x2, y2, tid, conf=0.9):
    return sv.Detections(
        xyxy=np.array([[x1, y1, x2, y2]], dtype=float),
        class_id=np.array([2]),
        confidence=np.array([conf]),
        tracker_id=np.array([tid], dtype=int),
    )


class TestBoxPersistence(unittest.TestCase):
    def test_box_tetap_ada_saat_deteksi_hilang(self):
        bp = BoxPersistence(grace_frames=10, min_detections=1)
        bp.update(make_det(100, 100, 160, 140, 1), 0)
        bp.update(make_det(102, 100, 162, 140, 1), 1)
        # frame 2: deteksi hilang -> box harus tetap keluar (persisted)
        hasil = bp.update(sv.Detections.empty(), 2)
        self.assertEqual(len(hasil), 1)
        self.assertEqual(int(hasil.tracker_id[0]), 1)
        np.testing.assert_array_equal(hasil.xyxy[0], [102, 100, 162, 140])

    def test_blip_satu_frame_tidak_jadi_hantu(self):
        bp = BoxPersistence(grace_frames=10, min_detections=2)
        bp.update(make_det(100, 100, 160, 140, 1), 0)
        # Hanya terdeteksi 1 frame lalu hilang -> jangan dipersistenkan
        hasil = bp.update(sv.Detections.empty(), 1)
        self.assertEqual(len(hasil), 0)

    def test_hantu_dibersihkan_setelah_grace(self):
        bp = BoxPersistence(grace_frames=3, min_detections=1)
        bp.update(make_det(100, 100, 160, 140, 1), 0)
        bp.update(make_det(102, 100, 162, 140, 1), 1)
        for t in range(2, 8):
            hasil = bp.update(sv.Detections.empty(), t)
        # Frame 7 sudah > grace (3) sejak deteksi terakhir -> kosong
        self.assertEqual(len(hasil), 0)

    def test_decay_confidence(self):
        bp = BoxPersistence(grace_frames=10, min_detections=1, decay=0.5)
        bp.update(make_det(100, 100, 160, 140, 1, conf=0.8), 0)
        hasil = bp.update(sv.Detections.empty(), 1)
        self.assertAlmostEqual(hasil.confidence[0], 0.4)


if __name__ == "__main__":
    unittest.main()