# fitur_anpr.py
import cv2
import numpy as np
from ultralytics import YOLO


class ANPRReader:
    def __init__(self, cfg):
        self.min_conf = cfg.get("min_conf", 0.3)
        self.cooldown_detik = cfg.get("cooldown_detik", 60.0)
        self.min_w_plat = cfg.get("min_w_plat", 40)
        self.min_h_plat = cfg.get("min_h_plat", 14)
        self.model_path = cfg.get("model_path", "./best_plat.pt")
        self.model_conf = cfg.get("model_conf", 0.25)
        self.backend = cfg.get("backend", "easyocr")
        self._model = None
        self._reader = None
        self._backend = None
        self._last_read = {}

    def _lazy_model(self):
        if self._model is None:
            self._model = YOLO(self.model_path)

    def _lazy_ocr(self):
        if self._backend is not None:
            return
        if self.backend in ("easyocr", "auto"):
            try:
                import easyocr
                self._reader = easyocr.Reader(["en"], gpu=False, verbose=False)
                self._backend = "easyocr"
                return
            except Exception:
                if self.backend == "easyocr":
                    self._backend = "none"
                    return
        if self.backend in ("pytesseract", "auto"):
            try:
                import pytesseract
                self._backend = "pytesseract"
                return
            except Exception:
                pass
        self._backend = "none"

    def _ocr(self, crop_plat):
        self._lazy_ocr()
        if self._backend == "easyocr":
            try:
                hasil = self._reader.readtext(crop_plat, detail=1)
                if hasil:
                    teks, conf = max((h[1], h[2]) for h in hasil if len(h[1].strip()) >= 2)
                    return teks, conf
            except Exception:
                pass
        elif self._backend == "pytesseract":
            try:
                import pytesseract
                teks = pytesseract.image_to_string(
                    crop_plat,
                    config="--psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
                )
                return teks, 0.5 if len(teks.strip()) >= 2 else 0.0
            except Exception:
                pass
        return "", 0.0

    @staticmethod
    def _bersihkan_plat(teks):
        return "".join(ch for ch in str(teks).upper() if ch.isalnum())

    def _crop_plat_yolo(self, vehicle_crop_image):
        if vehicle_crop_image is None or vehicle_crop_image.size == 0:
            return None, None
        self._lazy_model()
        result = self._model(vehicle_crop_image, conf=self.model_conf, verbose=False)[0]
        boxes = getattr(result, "boxes", None)
        if boxes is None or len(boxes) == 0:
            return None, None
        best_idx = int(np.argmax(boxes.conf.cpu().numpy()))
        conf = float(boxes.conf[best_idx].cpu().numpy())
        x1, y1, x2, y2 = boxes.xyxy[best_idx].cpu().numpy().astype(int)
        h, w = vehicle_crop_image.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        if x2 <= x1 or y2 <= y1:
            return None, None
        return vehicle_crop_image[y1:y2, x1:x2], conf

    def run_anpr_on_event(self, vehicle_crop_image):
        crop_plat, det_conf = self._crop_plat_yolo(vehicle_crop_image)
        if crop_plat is None:
            return {"plat": "", "conf": 0.0, "ocr_conf": 0.0, "det_conf": 0.0, "backend": self._backend}
        gray = cv2.cvtColor(crop_plat, cv2.COLOR_BGR2GRAY)
        teks, ocr_conf = self._ocr(gray)
        plat = self._bersihkan_plat(teks)
        conf = float(det_conf) * float(ocr_conf) if plat else 0.0
        return {
            "plat": plat,
            "conf": round(conf, 3),
            "ocr_conf": round(float(ocr_conf), 3),
            "det_conf": round(float(det_conf), 3),
            "backend": self._backend,
        }

    def baca_untuk_kendaraan(self, tid, crop, t_sekarang, camera_id="", event_type=""):
        if tid is None:
            return None
        tid = int(tid)
        if t_sekarang - self._last_read.get(tid, -1e9) < self.cooldown_detik:
            return None
        self._last_read[tid] = t_sekarang
        return self.run_anpr_on_event(crop)

    def lokalisasi_plat(self, gambar):
        if gambar is None or gambar.size == 0:
            return None
        abu = cv2.cvtColor(gambar, cv2.COLOR_BGR2GRAY)
        abu = cv2.GaussianBlur(abu, (5, 5), 0)
        thresh = cv2.adaptiveThreshold(
            abu, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, 19, 9
        )
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (12, 4))
        close = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        contours, _ = cv2.findContours(close, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        h_img, w_img = abu.shape[:2]
        best = None
        best_area = 0
        for c in contours:
            x, y, w, h = cv2.boundingRect(c)
            if w < self.min_w_plat or h < self.min_h_plat:
                continue
            ar = w / max(h, 1)
            if h > w or not (2.0 <= ar <= 6.5):
                continue
            area = w * h
            if area < 0.005 * (w_img * h_img):
                continue
            if area > best_area:
                best_area = area
                best = (x, y, w, h)
        return best

    def baca_plat(self, crop):
        return self.run_anpr_on_event(crop)
