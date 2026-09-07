#!/usr/bin/env python3
"""
ANPR Improved Segmentation - PHASE 4-5
Implements improved preprocessing for small plate crops
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

ALPHA_DICT_REV = {v: k for k, v in ALPHA_DICT.items()}

def character_recog_cnn_v2(model, img):
    img_roi = cv2.resize(img, (28, 28), interpolation=cv2.INTER_AREA)
    img_roi = img_roi.reshape((28, 28, 1))
    img_roi = np.expand_dims(np.array(img_roi), axis=0)
    result = model.predict(img_roi, verbose=0)
    result_idx = np.argmax(result, axis=1)[0]
    conf = float(np.max(result))
    return ALPHA_DICT[int(result_idx)], conf

def segment_improved_otsu(plate_img, target_width=200):
    """
    Improved Otsu segmentation with upscaling for small plates
    """
    h, w = plate_img.shape[:2]
    
    # Upscale small plates
    if w < target_width:
        scale = target_width / w
        new_w = int(w * scale)
        new_h = int(h * scale)
        img = cv2.resize(plate_img, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
    else:
        img = plate_img
        scale = 1.0
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Denoise
    gray = cv2.fastNlMeansDenoising(gray, h=10)
    
    # Otsu threshold
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # Morphological cleanup
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=1)
    
    # Remove small noise
    kernel_open = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel_open, iterations=1)
    
    # Find contours
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    
    if hierarchy is None:
        return [], thresh
    
    plate_h, plate_w = thresh.shape[:2]
    roi_area = plate_h * plate_w
    
    char_boxes = []
    for i, cnt in enumerate(contours):
        x, y, w, h = cv2.boundingRect(cnt)
        
        if h <= 0 or w <= 0:
            continue
        
        # Check hierarchy - only external contours or holes that are char-like
        hier = hierarchy[0][i] if hierarchy is not None else [-1, -1, -1, -1]
        has_child = hier[2] != -1
        
        ratio = w / float(h)
        area = cv2.contourArea(cnt)
        
        # Filter
        if w < 3 or h < 8:
            continue
        if ratio > 1.0 or ratio < 0.15:
            continue
        if area < 10:
            continue
        if h > plate_h * 0.9:
            continue
        if w > plate_w * 0.4:
            continue
        
        # Scale back coordinates if upscaled
        if scale != 1.0:
            x = int(x / scale)
            y = int(y / scale)
            w = int(w / scale)
            h = int(h / scale)
        
        char_boxes.append([x, y, w, h])
    
    # Sort left to right
    char_boxes.sort(key=lambda c: c[0])
    
    return char_boxes, thresh

def segment_adaptive_small(plate_img, target_width=200):
    """
    Adaptive thresholding optimized for small plates
    """
    h, w = plate_img.shape[:2]
    
    if w < target_width:
        scale = target_width / w
        new_w = int(w * scale)
        new_h = int(h * scale)
        img = cv2.resize(plate_img, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
    else:
        img = plate_img
        scale = 1.0
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.fastNlMeansDenoising(gray, h=10)
    
    # Adaptive threshold with smaller block size for small plates
    block_size = 15 if w < 100 else 19
    thresh = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, block_size, 7
    )
    
    # Morphological cleanup
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    
    # Extract chars (same logic as above)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    if hierarchy is None:
        return [], thresh
    
    plate_h, plate_w = thresh.shape[:2]
    roi_area = plate_h * plate_w
    
    char_boxes = []
    for i, cnt in enumerate(contours):
        x, y, w, h = cv2.boundingRect(cnt)
        if h <= 0 or w <= 0: continue
        ratio = w / float(h)
        area = cv2.contourArea(cnt)
        if w < 3 or h < 8: continue
        if ratio > 1.0 or ratio < 0.15: continue
        if area < 10: continue
        if h > plate_h * 0.9: continue
        if w > plate_w * 0.4: continue
        if scale != 1.0:
            x, y, w, h = int(x/scale), int(y/scale), int(w/scale), int(h/scale)
        char_boxes.append([x, y, w, h])
    
    char_boxes.sort(key=lambda c: c[0])
    return char_boxes, thresh

def segment_repo_improved(plate_img):
    """Improved repo pipeline with better post-processing"""
    _, thresh = preprocess(plate_img)
    
    # Additional cleanup
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    if hierarchy is None:
        return [], thresh
    
    plate_h, plate_w = thresh.shape[:2]
    roi_area = plate_h * plate_w
    
    char_boxes = []
    for i, cnt in enumerate(contours):
        x, y, w, h = cv2.boundingRect(cnt)
        if h <= 0 or w <= 0: continue
        ratio = w / float(h)
        area = cv2.contourArea(cnt)
        if w < 3 or h < 8: continue
        if ratio > 1.0 or ratio < 0.15: continue
        if area < 10: continue
        if h > plate_h * 0.9: continue
        if w > plate_w * 0.4: continue
        char_boxes.append([x, y, w, h])
    
    char_boxes.sort(key=lambda c: c[0])
    return char_boxes, thresh

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

def character_recog_cnn_v2(model, img):
    img_roi = cv2.resize(img, (28, 28), interpolation=cv2.INTER_AREA)
    img_roi = img_roi.reshape((28, 28, 1))
    img_roi = np.expand_dims(np.array(img_roi), axis=0)
    result = model.predict(img_roi, verbose=0)
    result_idx = np.argmax(result, axis=1)[0]
    conf = float(np.max(result))
    return ALPHA_DICT[int(result_idx)], conf

# Load CNN
cnn = CNN_Model()
cnn.model.load_weights('./anpr/weights/weight.h5')

# Methods to test
methods = {
    'IMPROVED_OTSU': segment_improved_otsu,
    'IMPROVED_ADAPTIVE': segment_adaptive_small,
    'REPO_IMPROVED': segment_repo_improved,
}

# Load CNN
cnn = CNN_Model()
cnn.model.load_weights('./anpr/weights/weight.h5')

# Test
crops_dir = './output/debug_plat_crops'
crops = [f for f in os.listdir('./output/debug_plat_crops') if f.endswith('_UP.jpg')]

print("=" * 80)
print("IMPROVED SEGMENTATION EVALUATION")
print("=" * 80)

summary = {m: {'crops_with_chars': 0, 'total_chars': 0, 'total_crops': 0} for m in methods}

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
                summary[method_name]['crops_with_chars'] += 1
                summary[method_name]['total_chars'] += len(chars)
                chars_str = ''.join([c[4] for c in chars])
                print(f"  {method_name}: {len(chars)} chars -> {chars_str} (conf: {[f'{c[5]:.2f}' for c in chars]})")
        except Exception as e:
            print(f"  {method_name}: ERROR - {e}")
    
    for m in methods:
        summary[m]['total_crops'] += 1

print("\n" + "=" * 80)
print("SUMMARY:")
for m in methods:
    s = summary[m]
    print(f"  {m}: {s['crops_with_chars']}/{s['total_crops']} crops, {s['total_chars']} chars")
print("=" * 80)