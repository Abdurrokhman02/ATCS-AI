# fitur_lawan_arah.py
import numpy as np
from collections import deque


class WrongWayDetector:
    """Deteksi kendaraan yang bergerak berlawanan arah dengan arus normal.

    Menggunakan riwayat posisi anchor (bawah-tengah box) hasil tracking untuk
    menentukan arah gerak vertikal kendaraan. Jika arah gerak konsisten berlawanan
    dengan `arah_lalu_lintas` kamera selama `frame_konfirmasi`, kendaraan ditandai
    melawan arah. Alert per kendaraan dibatasi dengan cooldown.

    Arah normal dikonfigurasi di CAM_CONFIG["arah_lalu_lintas"]:
      - "atas_ke_bawah" : anchor.y membesar (bergerak turun)
      - "bawah_ke_atas" : anchor.y mengecil (bergerak naik)
    """

    def __init__(self, arah_lalu_lintas="atas_ke_bawah", min_kecepatan_px=3.0,
                 frame_konfirmasi=5, cooldown_detik=20.0):
        self.arah_lalu_lintas = arah_lalu_lintas
        self.min_kecepatan_px = min_kecepatan_px
        self.frame_konfirmasi = frame_konfirmasi
        self.cooldown_detik = cooldown_detik
        self._history = {}        # tid -> deque[(t_sekarang, y_bottom)]
        self._wrong_frames = {}   # tid -> counter frame melawan arah
        self._last_alert = {}     # tid -> waktu alert terakhir

    def _expected_sign(self):
        return 1 if self.arah_lalu_lintas == "atas_ke_bawah" else -1

    def update(self, detections, t_sekarang):
        """Proses satu frame.

        Returns:
            wrong_mask : np.ndarray bool per deteksi -> sedang melawan arah (untuk render)
            events     : list dict {tid, confidence} -> alert baru (cooldown lolos)
        """
        n = len(detections)
        wrong_mask = np.zeros(n, dtype=bool)
        events = []
        if n == 0 or detections.tracker_id is None:
            return wrong_mask, events

        for i, (tid, xyxy) in enumerate(zip(detections.tracker_id, detections.xyxy)):
            if tid is None:
                continue
            tid = int(tid)
            y_bottom = float(xyxy[3])

            hist = self._history.setdefault(tid, deque(maxlen=max(4, self.frame_konfirmasi)))
            hist.append((t_sekarang, y_bottom))

            if len(hist) < self.frame_konfirmasi:
                continue

            dy = hist[-1][1] - hist[0][1]
            dt = max(hist[-1][0] - hist[0][0], 1e-6)
            kecepatan_px = abs(dy) / dt

            if kecepatan_px < self.min_kecepatan_px:
                self._wrong_frames[tid] = max(0, self._wrong_frames.get(tid, 0) - 1)
                continue

            arah = 1 if dy > 0 else -1
            if arah != self._expected_sign():
                self._wrong_frames[tid] = self._wrong_frames.get(tid, 0) + 1
            else:
                self._wrong_frames[tid] = max(0, self._wrong_frames.get(tid, 0) - 1)

            if self._wrong_frames[tid] >= self.frame_konfirmasi:
                wrong_mask[i] = True
                if t_sekarang - self._last_alert.get(tid, -1e9) >= self.cooldown_detik:
                    self._last_alert[tid] = t_sekarang
                    confidence = min(0.95, 0.5 + 0.08 * self._wrong_frames[tid])
                    events.append({"tid": tid, "confidence": confidence})

        return wrong_mask, events

    def bersihkan_memori(self, t_sekarang, timeout=15.0):
        stale = [tid for tid, hist in self._history.items()
                 if t_sekarang - hist[-1][0] > timeout]
        for tid in stale:
            self._history.pop(tid, None)
            self._wrong_frames.pop(tid, None)
            self._last_alert.pop(tid, None)
