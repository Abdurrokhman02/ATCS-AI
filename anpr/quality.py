"""
ANPR Quality Assessment Module
"""

import cv2
import numpy as np
from collections import defaultdict, deque
import time


class PlateQualityAssessor:
    """Assesses plate crop quality before sending to OCR"""
    
    def __init__(self, config=None):
        cfg = config or {}
        # Size thresholds (for plate crop)
        self.min_width = cfg.get('min_width', 60)
        self.min_height = cfg.get('min_height', 15)
        self.max_width = cfg.get('max_width', 500)
        self.max_height = cfg.get('max_height', 200)
        self.min_aspect_ratio = cfg.get('min_aspect_ratio', 1.5)
        self.max_aspect_ratio = cfg.get('max_aspect_ratio', 7.0)
        self.min_area_ratio = cfg.get('min_area_ratio', 0.0005)
        
        # Quality thresholds
        self.min_blur_score = cfg.get('min_blur_score', 30.0)
        self.min_brightness = cfg.get('min_brightness', 25)
        self.max_brightness = cfg.get('max_brightness', 230)
        self.min_contrast = cfg.get('min_contrast', 15)
        
    def assess(self, plate_crop, frame_shape=None, det_conf=0.0):
        """
        Assess plate crop quality.
        Returns (is_valid, quality_score, metrics_dict)
        """
        if plate_crop is None or plate_crop.size == 0:
            return False, 0.0, {'reason': 'empty_crop'}
        
        h, w = plate_crop.shape[:2]
        area = w * h
        aspect_ratio = w / max(h, 1)
        
        metrics = {
            'width': w,
            'height': h,
            'area': area,
            'aspect_ratio': aspect_ratio,
        }
        
        # Size checks
        if w < self.min_width or h < self.min_height:
            return False, 0.0, {**metrics, 'reason': 'too_small'}
        if w > self.max_width or h > self.max_height:
            return False, 0.0, {**metrics, 'reason': 'too_large'}
        if aspect_ratio < self.min_aspect_ratio or aspect_ratio > self.max_aspect_ratio:
            return False, 0.0, {**metrics, 'reason': 'bad_aspect_ratio'}
        
        # Blur detection
        gray = cv2.cvtColor(plate_crop, cv2.COLOR_BGR2GRAY)
        blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()
        metrics['blur_score'] = blur_score
        
        if blur_score < self.min_blur_score:
            return False, 0.0, {**metrics, 'reason': 'too_blurry'}
        
        # Brightness
        brightness = np.mean(plate_crop)
        metrics['brightness'] = brightness
        
        if brightness < self.min_brightness or brightness > self.max_brightness:
            return False, 0.0, {**metrics, 'reason': 'bad_brightness'}
        
        # Contrast
        gray = cv2.cvtColor(plate_crop, cv2.COLOR_BGR2GRAY)
        contrast = np.std(gray)
        metrics['contrast'] = contrast
        
        if contrast < self.min_contrast:
            return False, 0.0, {**metrics, 'reason': 'low_contrast'}
        
        # Frame area ratio
        if frame_shape:
            frame_area = frame_shape[0] * frame_shape[1]
            area_ratio = (plate_crop.shape[0] * plate_crop.shape[1]) / frame_area
            metrics['area_ratio'] = area_ratio
            if area_ratio < 0.0005:
                return False, 0.0, {**metrics, 'reason': 'too_small_in_frame'}
        
        # Quality score
        quality = self._compute_quality_score(blur_score, brightness, contrast, 0.0)
        
        return True, quality, metrics
    
    def _compute_quality_score(self, blur, brightness, contrast, det_conf=0.0):
        blur_norm = min(blur / 200.0, 1.0)
        contrast_norm = min(contrast / 80.0, 1.0)
        brightness_norm = 1.0 - abs(brightness - 127.5) / 127.5
        
        quality = (
            0.35 * blur_norm +
            0.35 * contrast_norm +
            0.15 * (1.0 - abs(brightness - 127.5) / 127.5) +
            0.15 * det_conf
        )
        
        return min(max(quality, 0.0), 1.0)


class BestFrameSelector:
    """Tracks best quality frame for each vehicle"""
    
    def __init__(self, max_history=10):
        self.max_history = max_history
        self.candidates = defaultdict(lambda: deque(maxlen=10))
        
    def add_candidate(self, tid, frame_idx, plate_crop, quality, det_conf, ocr_result=None):
        self.candidates[tid].append({
            'frame_idx': frame_idx,
            'crop': plate_crop.copy() if plate_crop is not None else None,
            'quality': quality,
            'det_conf': det_conf,
            'ocr_result': ocr_result,
            'timestamp': time.time()
        })
        
    def get_best(self, tid):
        if tid not in self.candidates or not self.candidates[tid]:
            return None
        return max(self.candidates[tid], key=lambda x: x['quality'])
    
    def clear(self, tid):
        if tid in self.candidates:
            del self.candidates[tid]


class TemporalAggregator:
    """Aggregates OCR results over time using weighted voting"""
    
    def __init__(self, window_seconds=5.0, min_votes=2):
        self.window_seconds = window_seconds
        self.min_votes = min_votes
        self.history = defaultdict(list)
        
    def add_result(self, tid, text, ocr_conf, quality, det_conf, timestamp=None):
        if timestamp is None:
            timestamp = time.time()
        
        if not text or len(text.strip()) < 2:
            return None
        
        text = text.upper().strip()
        weight = ocr_conf * quality
        
        self.history[tid].append({
            'timestamp': timestamp,
            'text': text,
            'weight': weight,
            'ocr_conf': ocr_conf,
            'quality': quality
        })
        
        cutoff = timestamp - self.window_seconds
        self.history[tid] = [h for h in self.history[tid] if h['timestamp'] > cutoff]
        
        return self._aggregate(tid)
    
    def _aggregate(self, tid):
        entries = self.history[tid]
        if len(entries) < 2:
            return None
        
        # Weighted voting per character position
        # For now, simple text-level voting
        text_votes = defaultdict(float)
        for e in entries:
            text_votes[e['text']] += e['weight']
        
        if not text_votes:
            return None
        
        best_text = max(text_votes, key=text_votes.get)
        total_weight = sum(text_votes.values())
        best_weight = text_votes[best_text]
        agg_conf = best_weight / total_weight if total_weight > 0 else 0
        
        return {
            'text': best_text,
            'confidence': agg_conf,
            'votes': dict(text_votes),
            'total_weight': total_weight
        }
    
    def get_current_best(self, tid):
        return self._aggregate(tid)
    
    def clear(self, tid):
        if tid in self.history:
            del self.history[tid]


# Character recognition function
def character_recog_cnn(model, img):
    img_roi = cv2.resize(img, (28, 28), interpolation=cv2.INTER_AREA)
    img_roi = img_roi.reshape((28, 28, 1))
    img_roi = np.expand_dims(np.array(img_roi), axis=0)
    result = model.predict(img_roi, verbose=0)
    result_idx = np.argmax(result, axis=1)[0]
    return ALPHA_DICT[int(result_idx)]


def character_recog_cnn_v2(model, img):
    img_roi = cv2.resize(img, (28, 28), interpolation=cv2.INTER_AREA)
    img_roi = img_roi.reshape((28, 28, 1))
    img_roi = np.expand_dims(np.array(img_roi), axis=0)
    result = model.predict(img_roi, verbose=0)
    result_idx = np.argmax(result, axis=1)[0]
    conf = float(np.max(result))
    return ALPHA_DICT[int(result_idx)], conf


class ANPRPipeline:
    """Complete ANPR pipeline with quality gate, best-frame selection, and temporal aggregation"""
    
    def __init__(self, plate_model_path, char_weights_path, plate_conf=0.25, config=None):
        cfg = config or {}
        
        # Load models
        self.plate_model = YOLO(plate_model_path)
        self.plate_conf = plate_conf
        
        self.cnn = CNN_Model()
        self.cnn.model.load_weights(char_weights_path)
        
        # Quality assessment
        quality_cfg = cfg.get('quality', {})
        self.quality_assessor = PlateQualityAssessor(quality_cfg)
        
        # Components
        self.best_frame_selector = BestFrameSelector()
        self.temporal_aggregator = TemporalAggregator(
            window_seconds=cfg.get('temporal_window', 5.0),
            min_votes=cfg.get('min_votes', 2)
        )
        
        # Cooldown
        self._last_anpr_time = {}
        self.cooldown_detik = cfg.get('cooldown_detik', 60.0)
        
    def process_vehicle(self, tid, vehicle_crop, frame_idx, t_sekarang, frame_shape=None):
        """Process vehicle through full ANPR pipeline"""
        if tid is None:
            return None
        tid = int(tid)
        
        # Cooldown
        if t_sekarang - self._last_anpr_time.get(tid, -1e9) < self.cooldown_detik:
            return None
        
        # Quality gate on vehicle crop
        is_valid, quality, metrics = self.quality_assessor.assess(
            vehicle_crop, frame_shape, 0.0
        )
        if not is_valid:
            return None
        
        # Detect plate within vehicle crop
        plate_crop, det_conf = self._crop_plat_yolo(vehicle_crop)
        if plate_crop is None:
            return None
        
        # Quality check on plate crop
        plate_valid, plate_quality, plate_metrics = self.quality_assessor.assess(
            plate_crop, vehicle_crop.shape, det_conf
        )
        if not plate_valid:
            return None
        
        # Run OCR on plate crop
        text, ocr_conf = self._ocr_plate(plate_crop)
        if not text or len(text.strip()) < 2:
            return None
        
        text = text.upper().strip()
        
        # Temporal aggregation
        combined_quality = quality * plate_quality
        agg_result = self.temporal_aggregator.add_result(
            tid, text, 0.9, plate_quality, det_conf
        )
        
        if agg_result and agg_result['confidence'] > 0.5:
            return agg_result
        
        return None
    
    def _crop_plat_yolo(self, vehicle_crop):
        if vehicle_crop is None or vehicle_crop.size == 0:
            return None, None
        results = self.plate_model(vehicle_crop, conf=0.25, verbose=False)[0]
        boxes = getattr(results, 'boxes', None)
        if boxes is None or len(boxes) == 0:
            return None, None
        best_idx = int(np.argmax(boxes.conf.cpu().numpy()))
        conf = float(boxes.conf[best_idx].cpu().numpy())
        x1, y1, x2, y2 = boxes.xyxy[best_idx].cpu().numpy().astype(int)
        h, w = vehicle_crop.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        if x2 <= x1 or y2 <= y1:
            return None, None
        return vehicle_crop[y1:y2, x1:x2], conf
    
    def _ocr_plate(self, plate_crop):
        # Preprocess plate crop
        gray = cv2.cvtColor(plate_crop, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # Denoise
        thresh = cv2.fastNlMeansDenoising(thresh, h=10)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2)))
        
        # Segment characters
        contours, _ = cv2.findContours(thresh, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)[:17]
        plate_h, plate_w = thresh.shape[:2]
        roi_area = plate_h * plate_w
        
        char_boxes = []
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            if h <= 0: continue
            ratio = w / float(h)
            area = w * h
            if (0.01 * (thresh.shape[0] * thresh.shape[1]) < area < 0.09 * roi_area and 0.25 < w/float(h) < 0.7):
                char_boxes.append([x, y, w, h])
        
        char_boxes.sort(key=lambda c: c[0])
        
        # Recognize
        text = ""
        for x, y, w, h in char_boxes:
            roi = thresh[y:y+h, x:x+w]
            if roi.size == 0: continue
            char, conf = character_recog_cnn_v2(self.cnn.model, roi)
            if char != 'Background':
                text += char
        
        return text, 0.9 if text else 0.0
    
    def cleanup(self, t_sekarang, timeout=120.0):
        stale = [tid for tid, t in self._last_anpr_time.items() if t_sekarang - t > timeout]
        for tid in stale:
            del self._last_anpr_time[tid]


# Import needed for ANPRPipeline
from anpr.cnn_model import CNN_Model
from ultralytics import YOLO
from collections import defaultdict

ALPHA_DICT = {
    0: "A", 1: "B", 2: "C", 3: "D", 4: "E", 5: "F", 6: "G", 7: "H",
    8: "K", 9: "L", 10: "M", 11: "N", 12: "P", 13: "R", 14: "S",
    15: "T", 16: "U", 17: "V", 18: "X", 19: "Y", 20: "Z",
    21: "0", 22: "1", 23: "2", 24: "3", 25: "4", 26: "5",
    27: "6", 28: "7", 29: "8", 30: "9", 31: "Background"
}