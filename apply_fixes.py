#!/usr/bin/env python3
"""Apply fixes to fitur_anpr.py"""

with open('modules/fitur_anpr.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Fix 1: Add _lazy_cnn() call in __init__
old_init = '''        self._model = None
        self._reader = None
        self._backend = None
        self._last_read = {}'''

new_init = '''        self._model = None
        self._reader = None
        self._backend = None
        self._last_read = {}
        
        # Pre-load CNN model
        self._lazy_cnn()'''

content = content.replace(old_init, new_init)

# Fix 2: Update _crop_plat_yolo with fallback
old_crop = '''    def _crop_plat_yolo(self, vehicle_crop_image):
        if vehicle_crop_image is None or vehicle_crop_image.size == 0:
            return None, None
        self._lazy_plate_model()
        result = self._plate_model(vehicle_crop_image, conf=self.model_conf, verbose=False)[0]
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
        return vehicle_crop_image[y1:y2, x1:x2], conf'''

new_crop = '''    def _crop_plat_yolo(self, vehicle_crop_image):
        if vehicle_crop_image is None or vehicle_crop_image.size == 0:
            return None, None
        self._lazy_plate_model()
        result = self._plate_model(vehicle_crop_image, conf=self.model_conf, verbose=False)[0]
        boxes = getattr(result, "boxes", None)
        if boxes is not None and len(boxes) > 0:
            best_idx = int(np.argmax(boxes.conf.cpu().numpy()))
            conf = float(boxes.conf[best_idx].cpu().numpy())
            x1, y1, x2, y2 = boxes.xyxy[best_idx].cpu().numpy().astype(int)
            h, w = vehicle_crop_image.shape[:2]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            if x2 > x1 and y2 > y1:
                return vehicle_crop_image[y1:y2, x1:x2], conf
        
        # Fallback to OpenCV contour method (lokalisasi_plat)
        plate_box = self.lokalisasi_plat(vehicle_crop_image)
        if plate_box is not None:
            x, y, w, h = plate_box
            return vehicle_crop_image[y:y+h, x:x+w], 0.5
        
        return None, None'''

content = content.replace(old_crop, new_crop)

# Fix 3: Remove duplicate _bersihkan_plat method
old_duplicate = '''        "temporal_pending": True,
        }

    def _bersihkan_plat(self, teks):
        return "".join(ch for ch in str(teks).upper() if ch.isalnum())

    def bersihkan_memori(self, t_sekarang, timeout=120.0):'''

new_duplicate = '''        "temporal_pending": True,
        }

    def bersihkan_memori(self, t_sekarang, timeout=120.0):'''

content = content.replace(old_duplicate, new_duplicate)

# Write back
with open('modules/fitur_anpr.py', 'w', encoding='utf-8') as f:
    f.write(content)

print('All fixes applied')