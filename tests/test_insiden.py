# tests/test_insiden.py
import unittest

import numpy as np
import supervision as sv

from modules.fitur_insiden import IncidentDetector

CFG = {
    "ambang_berhenti_px": 2.5,
    "durasi_berhenti_detik": 8.0,
    "kecepatan_tinggi_px": 12.0,
    "kecepatan_rendah_px": 2.0,
    "frame_hard_brake": 5,
    "ambang_iou_tabrakan": 0.25,
    "min_seen_frames": 3,
    "cooldown_detik": 30.0,
}


def make_det(boxes, tids):
    return sv.Detections(
        xyxy=np.array(boxes, dtype=float),
        class_id=np.array([2] * len(boxes)),
        confidence=np.array([0.9] * len(boxes)),
        tracker_id=np.array(tids, dtype=int),
    )


class TestIncidentDetector(unittest.TestCase):
    def test_kendaraan_berhenti(self):
        det = IncidentDetector(CFG)
        events = []
        for t in range(0, 12):
            events.extend(det.update(make_det([[100, 100, 160, 140]], [1]), float(t)))
        tipe = [e["tipe"] for e in events]
        self.assertIn("Kendaraan Berhenti", tipe)

    def test_kendaraan_bergerak_normal_tidak_berhenti(self):
        det = IncidentDetector(CFG)
        events = []
        for t in range(0, 12):
            x = 100 + 20 * t
            events.extend(det.update(make_det([[x, 100, x + 60, 140]], [1]), float(t)))
        tipe = [e["tipe"] for e in events]
        self.assertNotIn("Kendaraan Berhenti", tipe)

    def test_pengereman_mendadak(self):
        det = IncidentDetector(CFG)
        events = []
        # Bergerak cepat t=0..6, lalu berhenti t=7..9
        for t in range(0, 10):
            if t <= 6:
                x = 100 + 15 * t
            else:
                x = 190
            events.extend(det.update(make_det([[x, 100, x + 60, 140]], [1]), float(t)))
        tipe = [e["tipe"] for e in events]
        self.assertIn("Pengereman Mendadak", tipe)

    def test_potensi_tabrakan(self):
        det = IncidentDetector(CFG)
        events = []
        # A berhenti di posisi tetap; B menabrak A (box overlap), lalu keduanya berhenti
        box_a = [200, 100, 260, 140]
        for t in range(0, 4):
            events.extend(det.update(make_det([box_a], [1]), float(t)))
        # B tiba di posisi menimpa A (IoU besar), keduanya berhenti
        for t in range(4, 8):
            box_b = [210, 100, 270, 140]
            events.extend(det.update(make_det([box_a, box_b], [1, 2]), float(t)))
        tipe = [e["tipe"] for e in events]
        self.assertIn("Potensi Tabrakan", tipe)

    def test_cooldown_tidak_spam(self):
        det = IncidentDetector(CFG)
        n_events = 0
        # Window 0..34 dtk < 2x cooldown (30s) -> event berhenti hanya 1x
        for t in range(0, 35):
            n_events += len(det.update(make_det([[100, 100, 160, 140]], [1]), float(t)))
        self.assertLessEqual(n_events, 1)


if __name__ == "__main__":
    unittest.main()
