# fitur_stabilisasi.py
import numpy as np
import supervision as sv


class BoxPersistence:
    """Menahan bounding box terakhir untuk track yang sempat gagal dideteksi.

    Ketika YOLO gagal mendeteksi sebuah kendaraan pada 1..N frame (confidence
    turun, oklusi, pencahayaan), hasil tracking dari ByteTrack tidak lagi
    berisi box untuk track tsb sehingga box di layar "muncul-hilang".

    Lapisan ini menyimpan box terakhir per tracker_id dan tetap mengeluarkannya
    (dengan confidence yang didegradasi) selama `grace_frames`, sehingga
    tampilan visual bounding box menjadi kontinu.

    Diterapkan HANYA pada jalur rendering, bukan pada logika counting/speed/
    insiden yang tetap menggunakan deteksi asli agar tidak menyimpang.
    """

    def __init__(self, grace_frames=18, decay=0.6, min_detections=2):
        self.grace = grace_frames
        self.decay = decay
        self.min_detections = min_detections
        self._last = {}      # tid -> [frame_idx, xyxy, class_id, confidence, total_detection]
        self._det_count = {} # tid -> jumlah frame terdeteksi secara nyata

    @staticmethod
    def _compute_iou(box_a, box_b):
        """Intersection over Union dua box xyxy."""
        x1 = max(box_a[0], box_b[0])
        y1 = max(box_a[1], box_b[1])
        x2 = min(box_a[2], box_b[2])
        y2 = min(box_a[3], box_b[3])
        inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
        if inter <= 0.0:
            return 0.0
        area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
        area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
        union = area_a + area_b - inter
        return inter / union if union > 0 else 0.0

    def update(self, detections, frame_idx):
        """Gabungkan deteksi frame ini dengan box persisten track yang hilang.

        Hanya track yang telah terdeteksi minimal `min_detections` frame yang
        dipertahankan saat sempat hilang, agar blip deteksi 1-frame (false
        positive malam hari) tidak menjadi "hantu" di layar.

        Ghost suppression: jika box persisten overlap (IoU > 0.5, class sama)
        dengan live ByteTrack box, ghost dihapus karena kendaraan yang sama
        sudah dilacak dengan ID baru.

        Returns Detections untuk keperluan rendering.
        """
        current = {}
        if detections.tracker_id is not None:
            for i, t in enumerate(detections.tracker_id):
                t = int(t)
                self._det_count[t] = self._det_count.get(t, 0) + 1
                current[t] = (frame_idx, detections.xyxy[i].copy(),
                              detections.class_id[i], detections.confidence[i],
                              self._det_count[t])

        merged = dict(current)
        for t, (f, box, cid, conf, _) in self._last.items():
            if t in merged:
                continue
            if frame_idx - f > self.grace or self._det_count.get(t, 0) < self.min_detections:
                continue

            # Ghost suppression: cek overlap dengan live ByteTrack boxes
            suppressed = False
            for live_tid, (_, live_box, live_cid, _, _) in current.items():
                if int(live_cid) != int(cid):
                    continue
                iou = self._compute_iou(box, live_box)
                if iou > 0.5:
                    suppressed = True
                    break

            if not suppressed:
                merged[t] = (f, box, cid, conf * self.decay, self._det_count[t])

        # Prune track yang sudah lama hilang
        for t in list(self._last.keys()):
            if frame_idx - self._last[t][0] > self.grace:
                del self._last[t]
        self._last = merged

        if not merged:
            return detections

        tids = np.array(list(merged.keys()), dtype=int)
        xyxy = np.array([merged[t][1] for t in tids], dtype=float)
        cid = np.array([merged[t][2] for t in tids], dtype=int)
        conf = np.array([merged[t][3] for t in tids], dtype=float)
        return sv.Detections(xyxy=xyxy, confidence=conf, class_id=cid, tracker_id=tids)