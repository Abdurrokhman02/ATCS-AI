# event_logger.py
import json
import os
from datetime import datetime

import cv2


class EventLogger:
    """Pencatat event AI (lawan arah, insiden, ANPR) ke folder output.

    Menyimpan:
      - log baris-JSON (events.jsonl): timestamp, camera_id, event_type,
        confidence, metadata, nama snapshot.
      - snapshot gambar kejadian (jpg) bila `simpan_snapshot` = True.
    """

    def __init__(self, event_dir, simpan_snapshot=True, camera_id="CCTV"):
        self.event_dir = event_dir
        self.simpan_snapshot = simpan_snapshot
        self.camera_id = camera_id
        os.makedirs(event_dir, exist_ok=True)
        self.log_path = os.path.join(event_dir, "events.jsonl")

    @staticmethod
    def _slug(teks):
        return "".join(ch if ch.isalnum() else "_" for ch in str(teks))

    def log_event(self, event_type, confidence, frame=None, metadata=None):
        ts = datetime.now()
        ts_str = ts.isoformat(timespec="seconds")
        rec = {
            "timestamp": ts_str,
            "camera_id": self.camera_id,
            "event_type": event_type,
            "confidence": round(float(confidence), 3),
        }
        if metadata:
            rec.update({k: v for k, v in metadata.items() if v is not None})

        if frame is not None and self.simpan_snapshot:
            fname = (f"{ts.strftime('%Y%m%d_%H%M%S_%f')}_{self._slug(event_type)}.jpg")
            path = os.path.join(self.event_dir, fname)
            try:
                ok = cv2.imwrite(path, frame)
                if ok:
                    rec["snapshot"] = fname
            except Exception:
                pass

        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        return rec
