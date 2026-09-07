#!/usr/bin/env python3
"""
ANPR Baseline Evaluation - PHASE 3-4
Compares preprocessing methods and evaluates segmentation quality
"""

import os
import cv2
import numpy as np
from anpr.cnn_model import CNN_Model
from anpr.preprocess import preprocess
from anpr.utils_lp import ALPHA_DICT

# Load CNN
cnn = CNN_Model()
cnn.model.load_weights('./anpr/weights/weight.h5')

def character_recog_cnn_v2(model, img):
    """Wrapper that returns (char, confidence)"""
    img_roi = cv2.resize(img, (28, 28), interpolation=cv2.INTER_AREA)
    img_roi = img_roi.reshape((28, 28, 1))
    img_roi = np.expand_dims(np.array(img_roi), axis=0)
    result = model.predict(img_roi, verbose=0)
    result_idx = np.argmax(result, axis=1)[0]
    conf = float(np.max(result))
    return ALPHA_DICT[int(result_idx)], conf

def segment_characters_otsu(plate_img):
    """Segment using raw grayscale + Otsu"""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return extract_chars(thresh)

def segment_characters_adaptive(plate_img):
    """Segment using raw grayscale + Adaptive"""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    thresh = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, 19, 9
    )
    return extract_chars(thresh)

def segment_characters_repo(plate_img):
    """Segment using repository preprocess"""
    _, thresh = preprocess(img)
    return extract_chars(thresh)

def segment_characters_otsu_inv(plate_img):
    """Segment using raw grayscale + Otsu (non-inverted)"""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return extract_chars(thresh)

def extract_chars(thresh):
    """Extract character candidates from threshold image"""
    contours, _ = cv2.findContours(thresh, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:17]
    plate_h, plate_w = thresh.shape[:2]
    roi_area = plate_h * plate_w
    char_x = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if h <= 0: continue
        ratio_char = w / float(h)
        char_area = w * h
        if (0.01 * roi_area < char_area < 0.09 * roi_area and 0.25 < w/float(h) < 0.7):
            char_x.append([x, y, w, h])
    char_x = sorted(char_x, key=lambda c: c[0])
    return char_x, thresh

def recognize_chars(model, thresh, char_boxes):
    """Recognize characters from boxes"""
    results = []
    for x, y, w, h in char_boxes:
        roi = thresh[y:y+h, x:x+w]
        if roi.size == 0:
            continue
        char, conf = character_recog_cnn_v2(model, roi)
        if char != 'Background':
            results.append((x, y, w, h, char, conf))
    return results

# Load CNN
cnn = CNN_Model()
cnn.model.load_weights('./anpr/weights/weight.h5')

# Test on all plate crops
crops_dir = './output/debug_plat_crops'
crops = [f for f in os.listdir(crops_dir) if f.endswith('_UP.jpg')]

print("=" * 80)
print("ANPR BASELINE EVALUATION - PHASE 3-4")
print("=" * 80)

methods = {
    'OTSU_INV': segment_characters_otsu,
    'ADAPTIVE': segment_characters_adaptive,
    'REPO': segment_characters_repo,
    'OTSU': segment_characters_otsu_inv,
}

summary = {m: {'total_chars': 0, 'total_crops': 0, 'crops_with_chars': 0} for m in methods}

for f in sorted(os.listdir('./output/debug_plat_crops')):
    if not f.endswith('_UP.jpg'):
        continue
    
    img = cv2.imread(os.path.join('./output/debug_plat_crops', f))
    if img is None:
        continue
    
    print(f"\n{f}:")
    
    for method_name, segment_fn in methods.items():
        try:
            char_boxes, thresh = segment_fn(img)
            if not char_boxes:
                continue
            chars = recognize_chars(cnn.model, thresh, char_boxes)
            if chars:
                summary[method_name]['total_chars'] += len(chars)
                summary[method_name]['crops_with_chars'] += 1
                chars_str = ''.join([c[4] for c in chars])
                print(f"  {method_name}: {len(chars)} chars -> {chars_str}")
            else:
                print(f"  {method_name}: {len(char_boxes)} boxes, 0 recognized")
        except Exception as e:
            print(f"  {method_name}: ERROR - {e}")
    
    summary[method_name]['total_crops'] += 1

print("\n" + "=" * 80)
print("SUMMARY:")
for m in methods:
    s = summary[m]
    print(f"  {m}: {s['crops_with_chars']}/{s['total_crops']} crops with chars, {s['total_chars']} total chars")
print("=" * 80)