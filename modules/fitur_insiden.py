import numpy as np
from collections import deque

def _iou(a, b):
    """Intersection over Union dua box xyxy."""
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    if inter <= 0.0:
        return 0.0
    area_a = max(a[2] - a[0], 0.0) * max(a[3] - a[1], 0.0)
    area_b = max(b[2] - b[0], 0.0) * max(b[3] - b[1], 0.0)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


class IncidentDetector:
    """Deteksi pola anomali yang mengindikasikan potensi insiden lalu lintas.

    Menggunakan hasil detection+tracking yang sudah ada:
      1. "Kendaraan Berhenti"   : track tidak bergerak > durasi_berhenti_detik.
      2. "Pengereman Mendadak"  : kecepatan turun drastis dalam window singkat.
      3. "Potensi Tabrakan"     : dua track aktif tumpang tindih (IoU) saat berhenti.
    """

    TIPE_BERHENTI = "Kendaraan Berhenti"
    TIPE_HARD_BRAKE = "Pengereman Mendadak"
    TIPE_TABRAKAN = "Potensi Tabrakan"

    def __init__(self, cfg):
        self.ambang_berhenti_px = cfg.get("ambang_berhenti_px", 2.5)
        self.durasi_berhenti_detik = cfg.get("durasi_berhenti_detik", 8.0)
        self.kecepatan_tinggi_px = cfg.get("kecepatan_tinggi_px", 12.0)
        self.kecepatan_rendah_px = cfg.get("kecepatan_rendah_px", 2.0)
        self.frame_hard_brake = int(cfg.get("frame_hard_brake", 5))
        self.ambang_iou_tabrakan = cfg.get("ambang_iou_tabrakan", 0.25)
        self.cooldown_detik = cfg.get("cooldown_detik", 30.0)
        self.min_seen_frames = int(cfg.get("min_seen_frames", 3))

        # Buffer "Mesin Waktu" untuk menyimpan histori koordinat & frame sebelum benturan
        self.buffer_size = int(cfg.get("buffer_size", 30))
        self._frame_buffer = {}  # frame_idx -> frame
        self._box_history = {}   # tid -> list of (frame_idx, box)

        self._state = {}         # tid -> {"pos": (x, y), "t": t}
        self._hist = {}          # tid -> deque kecepatan px/detik
        self._last_event = {}    # (tid, tipe) -> t
        self._stopped = {}       # tid -> True/False
        self._stopped_since = {} # tid -> waktu mulai berhenti
        self._seen = {}          # tid -> jumlah frame terlihat

    def update(self, detections, t_sekarang, frame=None, frame_idx=0):
        """Proses satu frame. Returns list event dict."""
        events = []
        n = len(detections)

        # Update Mesin Waktu jika frame diberikan
        if frame is not None:
            self._frame_buffer[frame_idx] = frame.copy()
            if frame_idx - self.buffer_size in self._frame_buffer:
                del self._frame_buffer[frame_idx - self.buffer_size]

        if n == 0 or detections.tracker_id is None:
            return events

        tid_to_idx = {}
        for i, (tid, xyxy) in enumerate(zip(detections.tracker_id, detections.xyxy)):
            if tid is None:
                continue
            tid = int(tid)
            tid_to_idx[tid] = (i, xyxy)

            # Catat histori posisi per ID untuk fitur Mesin Waktu
            hist_box = self._box_history.setdefault(tid, [])
            hist_box.append((frame_idx, xyxy.copy()))
            if len(hist_box) > self.buffer_size:
                hist_box.pop(0)

        self._deteksi_tabrakan(tid_to_idx, t_sekarang, events, frame_idx)
        self._update_gerak(detections, t_sekarang, events)
        return events

    def _update_gerak(self, detections, t_sekarang, events):
        for tid, xyxy in zip(detections.tracker_id, detections.xyxy):
            if tid is None:
                continue
            tid = int(tid)
            self._seen[tid] = self._seen.get(tid, 0) + 1
            cukup_umur = self._seen[tid] >= self.min_seen_frames

            center = ((xyxy[0] + xyxy[2]) / 2.0, (xyxy[1] + xyxy[3]) / 2.0)
            prev = self._state.get(tid)
            self._state[tid] = {"pos": center, "t": t_sekarang}

            if prev is None:
                continue

            dt = max(t_sekarang - prev["t"], 1e-6)
            jarak = ((center[0] - prev["pos"][0]) ** 2 + (center[1] - prev["pos"][1]) ** 2) ** 0.5
            kecepatan_px = jarak / dt

            hist = self._hist.setdefault(tid, deque(maxlen=max(3, self.frame_hard_brake)))
            hist.append(kecepatan_px)

            # 1. Kendaraan berhenti
            if kecepatan_px < self.ambang_berhenti_px:
                if not self._stopped.get(tid, False):
                    self._stopped[tid] = True
                    self._stopped_since[tid] = t_sekarang
                durasi = t_sekarang - self._stopped_since[tid]
                if cukup_umur and durasi >= self.durasi_berhenti_detik:
                    conf = min(0.95, 0.45 + durasi / self.durasi_berhenti_detik * 0.3)
                    self._emit(events, tid, self.TIPE_BERHENTI, t_sekarang, conf)
            else:
                self._stopped[tid] = False

            # 2. Pengereman mendadak
            if cukup_umur and len(hist) >= self.frame_hard_brake and self._stopped.get(tid, False):
                vals = list(hist)
                awal = np.mean(vals[: max(2, self.frame_hard_brake // 2)])
                akhir = np.mean(vals[-2:])
                if awal >= self.kecepatan_tinggi_px and akhir <= self.kecepatan_rendah_px:
                    conf = min(0.9, 0.5 + (awal - akhir) / max(awal, 1.0) * 0.3)
                    self._emit(events, tid, self.TIPE_HARD_BRAKE, t_sekarang, conf)

    def _deteksi_tabrakan(self, tid_to_idx, t_sekarang, events, frame_idx):
        tids = list(tid_to_idx.keys())
        for i in range(len(tids)):
            for j in range(i + 1, len(tids)):
                tid_a, tid_b = tids[i], tids[j]
                if self._seen.get(tid_a, 0) < self.min_seen_frames or \
                        self._seen.get(tid_b, 0) < self.min_seen_frames:
                    continue
                _, box_a = tid_to_idx[tid_a]
                _, box_b = tid_to_idx[tid_b]
                iou = _iou(box_a, box_b)
                if iou < self.ambang_iou_tabrakan:
                    continue

                a_stop = self._stopped.get(tid_a, False)
                b_stop = self._stopped.get(tid_b, False)
                if not (a_stop or b_stop):
                    continue

                conf = min(0.9, 0.5 + iou)
                for tid in (tid_a, tid_b):
                    # Ambil snapshot histori sebelum tabrakan jika ada
                    pre_crash_crop = self._get_pre_crash_crop(tid)
                    self._emit(events, tid, self.TIPE_TABRAKAN, t_sekarang, conf, pre_crash_crop)

    def _get_pre_crash_crop(self, tid, padding=30):
        """Mengambil gambar crop kendaraan dari masa lalu (sebelum benturan)."""
        hist = self._box_history.get(tid)
        if not hist:
            return None
        oldest_frame_idx, oldest_box = hist[0]
        if oldest_frame_idx in self._frame_buffer:
            frame_lama = self._frame_buffer[oldest_frame_idx]
            h, w = frame_lama.shape[:2]
            x1, y1, x2, y2 = map(int, oldest_box)
            x1 = max(0, x1 - padding)
            y1 = max(0, y1 - padding)
            x2 = min(w, x2 + padding)
            y2 = min(h, y2 + padding)
            if x2 > x1 and y2 > y1:
                return frame_lama[y1:y2, x1:x2]
        return None

    def _emit(self, events, tid, tipe, t_sekarang, conf, crop_image=None):
        key = (tid, tipe)
        if t_sekarang - self._last_event.get(key, -1e9) >= self.cooldown_detik:
            self._last_event[key] = t_sekarang
            ev = {"tipe": tipe, "tid": tid, "confidence": conf, "t": t_sekarang}
            if crop_image is not None:
                ev["crop_image"] = crop_image
            events.append(ev)

    def bersihkan_memori(self, t_sekarang, timeout=20.0):
        stale = [tid for tid, st in self._state.items() if t_sekarang - st["t"] > timeout]
        for tid in stale:
            self._state.pop(tid, None)
            self._hist.pop(tid, None)
            self._stopped.pop(tid, None)
            self._stopped_since.pop(tid, None)
            self._seen.pop(tid, None)
            self._box_history.pop(tid, None)
        stale_ev = [k for k, t in self._last_event.items() if t_sekarang - t > timeout]
        for k in stale_ev:
            del self._last_event[k]