import os
import cv2

from anpr.preprocess import preprocess


# ============================================================
# CONFIG
# ============================================================

INPUT_IMAGE = (
    "./output/debug_plat_crops/"
    "crop_019_frame_0273_79x37_conf_0.31_UP.jpg"
)

OUTPUT_DIR = "./output/debug_preprocess"

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# LOAD IMAGE
# ============================================================

print("=" * 60)
print("DEBUG ANPR PREPROCESSING")
print("=" * 60)

print(f"[INFO] Input : {INPUT_IMAGE}")

image = cv2.imread(INPUT_IMAGE)

if image is None:
    raise RuntimeError(
        f"Gagal membaca image:\n{INPUT_IMAGE}"
    )

original_h, original_w = image.shape[:2]

print(
    f"[INFO] Original size: "
    f"{original_w}x{original_h}"
)


# ============================================================
# SAVE ORIGINAL
# ============================================================

original_path = os.path.join(
    OUTPUT_DIR,
    "01_original.jpg"
)

cv2.imwrite(
    original_path,
    image
)

print(f"[SAVE] {original_path}")


# ============================================================
# UPSCALE ORIGINAL
# ============================================================

original_upscaled = cv2.resize(
    image,
    None,
    fx=8,
    fy=8,
    interpolation=cv2.INTER_CUBIC
)

original_upscaled_path = os.path.join(
    OUTPUT_DIR,
    "02_original_upscaled.jpg"
)

cv2.imwrite(
    original_upscaled_path,
    original_upscaled
)

print(f"[SAVE] {original_upscaled_path}")


# ============================================================
# RUN PREPROCESS
# ============================================================

print()
print("[INFO] Running preprocess()...")

img_grayscale, img_thresh = preprocess(image)

print("[INFO] preprocess() berhasil")


# ============================================================
# CHECK GRAYSCALE
# ============================================================

gray_h, gray_w = img_grayscale.shape[:2]

print(
    f"[INFO] Grayscale size: "
    f"{gray_w}x{gray_h}"
)

print(
    f"[INFO] Grayscale dtype: "
    f"{img_grayscale.dtype}"
)


# ============================================================
# SAVE GRAYSCALE
# ============================================================

gray_path = os.path.join(
    OUTPUT_DIR,
    "03_grayscale.jpg"
)

cv2.imwrite(
    gray_path,
    img_grayscale
)

print(f"[SAVE] {gray_path}")


# ============================================================
# UPSCALE GRAYSCALE
# ============================================================

gray_upscaled = cv2.resize(
    img_grayscale,
    None,
    fx=8,
    fy=8,
    interpolation=cv2.INTER_CUBIC
)

gray_upscaled_path = os.path.join(
    OUTPUT_DIR,
    "04_grayscale_upscaled.jpg"
)

cv2.imwrite(
    gray_upscaled_path,
    gray_upscaled
)

print(f"[SAVE] {gray_upscaled_path}")


# ============================================================
# SAVE THRESHOLD
# ============================================================

thresh_h, thresh_w = img_thresh.shape[:2]

print(
    f"[INFO] Threshold size: "
    f"{thresh_w}x{thresh_h}"
)

threshold_path = os.path.join(
    OUTPUT_DIR,
    "05_threshold.jpg"
)

cv2.imwrite(
    threshold_path,
    img_thresh
)

print(f"[SAVE] {threshold_path}")


# ============================================================
# UPSCALE THRESHOLD
# ============================================================

threshold_upscaled = cv2.resize(
    img_thresh,
    None,
    fx=8,
    fy=8,
    interpolation=cv2.INTER_NEAREST
)

threshold_upscaled_path = os.path.join(
    OUTPUT_DIR,
    "06_threshold_upscaled.jpg"
)

cv2.imwrite(
    threshold_upscaled_path,
    threshold_upscaled
)

print(f"[SAVE] {threshold_upscaled_path}")


# ============================================================
# EXTRA: THRESHOLD WITH COLOR
# ============================================================
#
# Biar lebih gampang dilihat secara visual.
# Tidak digunakan untuk OCR.
#

threshold_color = cv2.cvtColor(
    img_thresh,
    cv2.COLOR_GRAY2BGR
)

threshold_color_path = os.path.join(
    OUTPUT_DIR,
    "07_threshold_color.jpg"
)

cv2.imwrite(
    threshold_color_path,
    threshold_color
)

print(f"[SAVE] {threshold_color_path}")


# ============================================================
# EXTRA: ORIGINAL + THRESHOLD SIDE BY SIDE
# ============================================================

# Samakan ukuran untuk visualisasi
original_gray_bgr = cv2.cvtColor(
    img_grayscale,
    cv2.COLOR_GRAY2BGR
)

threshold_bgr = cv2.cvtColor(
    img_thresh,
    cv2.COLOR_GRAY2BGR
)

comparison = cv2.hconcat([
    original_gray_bgr,
    threshold_bgr
])

comparison_upscaled = cv2.resize(
    comparison,
    None,
    fx=8,
    fy=8,
    interpolation=cv2.INTER_NEAREST
)

comparison_path = os.path.join(
    OUTPUT_DIR,
    "08_grayscale_vs_threshold.jpg"
)

cv2.imwrite(
    comparison_path,
    comparison_upscaled
)

print(f"[SAVE] {comparison_path}")


# ============================================================
# SUMMARY
# ============================================================

print()
print("=" * 60)
print("SELESAI")
print("=" * 60)

print()
print(f"Input:")
print(f"  {INPUT_IMAGE}")

print()
print(f"Original:")
print(f"  {original_w}x{original_h}")

print()
print("Output:")
print(f"  {OUTPUT_DIR}")

print()
print("Files:")
print("  01_original.jpg")
print("  02_original_upscaled.jpg")
print("  03_grayscale.jpg")
print("  04_grayscale_upscaled.jpg")
print("  05_threshold.jpg")
print("  06_threshold_upscaled.jpg")
print("  07_threshold_color.jpg")
print("  08_grayscale_vs_threshold.jpg")

print("=" * 60)