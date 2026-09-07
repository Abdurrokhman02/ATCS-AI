import os
import cv2
import numpy as np

from anpr.preprocess import preprocess


# ============================================================
# CONFIG
# ============================================================

INPUT_IMAGE = (
    "./output/debug_plat_crops/"
    "crop_019_frame_0273_79x37_conf_0.31_UP.jpg"
)

OUTPUT_DIR = "./output/debug_segmentation"

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# CHARACTER CANDIDATE FILTER
# ============================================================

def find_character_candidates(binary, label):
    """
    Mencari contour yang berpotensi menjadi karakter plat.

    Filter digunakan seperti script sebelumnya.
    Bagian ini TIDAK diubah untuk menjaga perbandingan
    dengan hasil eksperimen sebelumnya.
    """

    h, w = binary.shape[:2]

    contours, _ = cv2.findContours(
        binary,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    candidates = []

    for contour in contours:

        x, y, cw, ch = cv2.boundingRect(contour)

        area = cv2.contourArea(contour)

        # -----------------------------------------------
        # BASIC FILTER
        # -----------------------------------------------

        if cw < 2:
            continue

        if ch < 5:
            continue

        if area < 5:
            continue

        # Karakter biasanya lebih tinggi daripada lebar
        aspect_ratio = cw / float(ch)

        if aspect_ratio > 1.2:
            continue

        # Jangan menerima blob yang hampir sebesar seluruh plate
        if ch > h * 0.95:
            continue

        if cw > w * 0.35:
            continue

        candidates.append(
            {
                "x": x,
                "y": y,
                "w": cw,
                "h": ch,
                "area": area,
                "ratio": aspect_ratio,
            }
        )

    # Sort kiri -> kanan
    candidates.sort(key=lambda c: c["x"])

    return candidates


# ============================================================
# RAW CONTOUR AUDIT
# ============================================================

def audit_all_contours(binary, label):
    """
    Audit SEMUA contour TANPA FILTER karakter.

    Tujuan:
    mengetahui apakah contour karakter sebenarnya masih
    muncul setelah threshold, atau hilang sebelum filtering.

    Tidak ada filter:
        - width
        - height
        - area
        - aspect ratio
        - ukuran relatif terhadap plate
    """

    contours, _ = cv2.findContours(
        binary,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    results = []

    for idx, contour in enumerate(contours):

        x, y, cw, ch = cv2.boundingRect(contour)

        area = cv2.contourArea(contour)

        if ch > 0:
            ratio = cw / float(ch)
        else:
            ratio = 0.0

        results.append(
            {
                "idx": idx + 1,
                "x": x,
                "y": y,
                "w": cw,
                "h": ch,
                "area": area,
                "ratio": ratio,
            }
        )

    # Sort berdasarkan posisi X
    results.sort(key=lambda c: c["x"])

    print()
    print("=" * 70)
    print(f"RAW CONTOUR AUDIT: {label}")
    print("=" * 70)

    print(f"Total contour: {len(results)}")
    print()

    if not results:
        print("Tidak ada contour.")
        return results

    for c in results:

        print(
            f"{c['idx']:02d}. "
            f"x={c['x']:02d} "
            f"y={c['y']:02d} "
            f"w={c['w']:02d} "
            f"h={c['h']:02d} "
            f"ratio={c['ratio']:.2f} "
            f"area={c['area']:.1f}"
        )

    return results


# ============================================================
# DRAW CANDIDATES
# ============================================================

def draw_candidates(image, candidates, title):

    if len(image.shape) == 2:

        output = cv2.cvtColor(
            image,
            cv2.COLOR_GRAY2BGR
        )

    else:

        output = image.copy()

    for idx, c in enumerate(candidates):

        x = c["x"]
        y = c["y"]
        w = c["w"]
        h = c["h"]

        cv2.rectangle(
            output,
            (x, y),
            (x + w, y + h),
            (0, 255, 0),
            1
        )

        cv2.putText(
            output,
            str(idx + 1),
            (x, max(10, y - 2)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.35,
            (0, 255, 0),
            1,
            cv2.LINE_AA
        )

    cv2.putText(
        output,
        f"{title} | candidates={len(candidates)}",
        (3, 12),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.35,
        (0, 0, 255),
        1,
        cv2.LINE_AA
    )

    return output


# ============================================================
# DRAW ALL CONTOURS
# ============================================================

def draw_all_contours(image, contours, title):

    if len(image.shape) == 2:

        output = cv2.cvtColor(
            image,
            cv2.COLOR_GRAY2BGR
        )

    else:

        output = image.copy()

    for c in contours:

        idx = c["idx"]

        x = c["x"]
        y = c["y"]
        w = c["w"]
        h = c["h"]

        cv2.rectangle(
            output,
            (x, y),
            (x + w, y + h),
            (0, 255, 0),
            1
        )

        cv2.putText(
            output,
            str(idx),
            (x, max(10, y - 2)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.35,
            (0, 255, 0),
            1,
            cv2.LINE_AA
        )

    cv2.putText(
        output,
        f"{title} | ALL CONTOURS={len(contours)}",
        (3, 12),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.35,
        (0, 0, 255),
        1,
        cv2.LINE_AA
    )

    return output


# ============================================================
# LOAD IMAGE
# ============================================================

print("=" * 70)
print("DEBUG CHARACTER SEGMENTATION + RAW CONTOUR AUDIT")
print("=" * 70)

print(f"[INFO] Input: {INPUT_IMAGE}")

image = cv2.imread(INPUT_IMAGE)

if image is None:

    raise RuntimeError(
        f"Gagal membaca image:\n{INPUT_IMAGE}"
    )

h, w = image.shape[:2]

print(f"[INFO] Image size: {w}x{h}")


# ============================================================
# PREPROCESS REPO
# ============================================================

print()
print("[INFO] Running repository preprocess()...")

img_grayscale, img_repo_thresh = preprocess(image)

print("[OK] preprocess() selesai")


# ============================================================
# EXPERIMENT 1
# RAW GRAYSCALE -> OTSU
# ============================================================

print()
print("[TEST 1] Raw grayscale -> Otsu")

_, thresh_otsu = cv2.threshold(
    img_grayscale,
    0,
    255,
    cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
)

candidates_otsu = find_character_candidates(
    thresh_otsu,
    "RAW GRAYSCALE + OTSU"
)

print(
    f"[RESULT] Otsu candidates: "
    f"{len(candidates_otsu)}"
)


# ============================================================
# EXPERIMENT 2
# RAW GRAYSCALE -> ADAPTIVE
# ============================================================

print()
print("[TEST 2] Raw grayscale -> Adaptive Threshold")

thresh_adaptive_raw = cv2.adaptiveThreshold(
    img_grayscale,
    255,
    cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
    cv2.THRESH_BINARY_INV,
    19,
    9
)

candidates_adaptive_raw = find_character_candidates(
    thresh_adaptive_raw,
    "RAW GRAYSCALE + ADAPTIVE"
)

print(
    f"[RESULT] Raw adaptive candidates: "
    f"{len(candidates_adaptive_raw)}"
)


# ============================================================
# EXPERIMENT 3
# REPOSITORY PIPELINE
# ============================================================

print()
print("[TEST 3] Repository preprocessing -> Adaptive Threshold")

candidates_repo = find_character_candidates(
    img_repo_thresh,
    "REPOSITORY PIPELINE"
)

print(
    f"[RESULT] Repository candidates: "
    f"{len(candidates_repo)}"
)


# ============================================================
# RAW CONTOUR AUDIT
# ============================================================

print()
print("=" * 70)
print("STARTING RAW CONTOUR AUDIT")
print("=" * 70)

all_otsu = audit_all_contours(
    thresh_otsu,
    "RAW GRAYSCALE + OTSU"
)

all_adaptive_raw = audit_all_contours(
    thresh_adaptive_raw,
    "RAW GRAYSCALE + ADAPTIVE"
)

all_repo = audit_all_contours(
    img_repo_thresh,
    "REPOSITORY PIPELINE"
)


# ============================================================
# PRINT CANDIDATE DETAILS
# ============================================================

def print_candidates(name, candidates):

    print()
    print("-" * 70)
    print(name)
    print("-" * 70)

    if not candidates:

        print("Tidak ada candidate.")
        return

    for idx, c in enumerate(candidates):

        print(
            f"{idx + 1:02d}. "
            f"x={c['x']:02d} "
            f"y={c['y']:02d} "
            f"w={c['w']:02d} "
            f"h={c['h']:02d} "
            f"ratio={c['ratio']:.2f} "
            f"area={c['area']:.1f}"
        )


print_candidates(
    "RAW GRAYSCALE + OTSU",
    candidates_otsu
)

print_candidates(
    "RAW GRAYSCALE + ADAPTIVE",
    candidates_adaptive_raw
)

print_candidates(
    "REPOSITORY PIPELINE",
    candidates_repo
)


# ============================================================
# DRAW FILTERED CANDIDATES
# ============================================================

vis_otsu = draw_candidates(
    thresh_otsu,
    candidates_otsu,
    "RAW GRAYSCALE + OTSU"
)

vis_adaptive_raw = draw_candidates(
    thresh_adaptive_raw,
    candidates_adaptive_raw,
    "RAW GRAYSCALE + ADAPTIVE"
)

vis_repo = draw_candidates(
    img_repo_thresh,
    candidates_repo,
    "REPOSITORY PIPELINE"
)


# ============================================================
# DRAW ALL CONTOURS
# ============================================================

vis_all_otsu = draw_all_contours(
    thresh_otsu,
    all_otsu,
    "RAW OTSU"
)

vis_all_adaptive = draw_all_contours(
    thresh_adaptive_raw,
    all_adaptive_raw,
    "RAW ADAPTIVE"
)

vis_all_repo = draw_all_contours(
    img_repo_thresh,
    all_repo,
    "REPO"
)


# ============================================================
# UPSCALE
# ============================================================

SCALE = 8


# Filtered candidate visualization
vis_otsu_up = cv2.resize(
    vis_otsu,
    None,
    fx=SCALE,
    fy=SCALE,
    interpolation=cv2.INTER_NEAREST
)

vis_adaptive_raw_up = cv2.resize(
    vis_adaptive_raw,
    None,
    fx=SCALE,
    fy=SCALE,
    interpolation=cv2.INTER_NEAREST
)

vis_repo_up = cv2.resize(
    vis_repo,
    None,
    fx=SCALE,
    fy=SCALE,
    interpolation=cv2.INTER_NEAREST
)


# All contour visualization
vis_all_otsu_up = cv2.resize(
    vis_all_otsu,
    None,
    fx=SCALE,
    fy=SCALE,
    interpolation=cv2.INTER_NEAREST
)

vis_all_adaptive_up = cv2.resize(
    vis_all_adaptive,
    None,
    fx=SCALE,
    fy=SCALE,
    interpolation=cv2.INTER_NEAREST
)

vis_all_repo_up = cv2.resize(
    vis_all_repo,
    None,
    fx=SCALE,
    fy=SCALE,
    interpolation=cv2.INTER_NEAREST
)


# ============================================================
# SAVE FILTERED CANDIDATE RESULTS
# ============================================================

path_otsu = os.path.join(
    OUTPUT_DIR,
    "01_otsu_candidates.jpg"
)

path_adaptive_raw = os.path.join(
    OUTPUT_DIR,
    "02_raw_adaptive_candidates.jpg"
)

path_repo = os.path.join(
    OUTPUT_DIR,
    "03_repo_candidates.jpg"
)

cv2.imwrite(
    path_otsu,
    vis_otsu_up
)

cv2.imwrite(
    path_adaptive_raw,
    vis_adaptive_raw_up
)

cv2.imwrite(
    path_repo,
    vis_repo_up
)

print()
print("[SAVE] Filtered candidate results:")
print(f"  {path_otsu}")
print(f"  {path_adaptive_raw}")
print(f"  {path_repo}")


# ============================================================
# SAVE THRESHOLD IMAGES
# ============================================================

otsu_up = cv2.resize(
    thresh_otsu,
    None,
    fx=SCALE,
    fy=SCALE,
    interpolation=cv2.INTER_NEAREST
)

adaptive_raw_up = cv2.resize(
    thresh_adaptive_raw,
    None,
    fx=SCALE,
    fy=SCALE,
    interpolation=cv2.INTER_NEAREST
)

repo_thresh_up = cv2.resize(
    img_repo_thresh,
    None,
    fx=SCALE,
    fy=SCALE,
    interpolation=cv2.INTER_NEAREST
)

cv2.imwrite(
    os.path.join(
        OUTPUT_DIR,
        "04_otsu.jpg"
    ),
    otsu_up
)

cv2.imwrite(
    os.path.join(
        OUTPUT_DIR,
        "05_raw_adaptive.jpg"
    ),
    adaptive_raw_up
)

cv2.imwrite(
    os.path.join(
        OUTPUT_DIR,
        "06_repo_threshold.jpg"
    ),
    repo_thresh_up
)


# ============================================================
# SIDE-BY-SIDE FILTERED COMPARISON
# ============================================================

comparison = cv2.hconcat(
    [
        vis_otsu_up,
        vis_adaptive_raw_up,
        vis_repo_up
    ]
)

comparison_path = os.path.join(
    OUTPUT_DIR,
    "07_SEGMENTATION_COMPARISON.jpg"
)

cv2.imwrite(
    comparison_path,
    comparison
)

print(f"[SAVE] {comparison_path}")


# ============================================================
# SAVE ALL CONTOUR AUDIT
# ============================================================

path_all_otsu = os.path.join(
    OUTPUT_DIR,
    "08_ALL_CONTOURS_OTSU.jpg"
)

path_all_adaptive = os.path.join(
    OUTPUT_DIR,
    "09_ALL_CONTOURS_RAW_ADAPTIVE.jpg"
)

path_all_repo = os.path.join(
    OUTPUT_DIR,
    "10_ALL_CONTOURS_REPO.jpg"
)

cv2.imwrite(
    path_all_otsu,
    vis_all_otsu_up
)

cv2.imwrite(
    path_all_adaptive,
    vis_all_adaptive_up
)

cv2.imwrite(
    path_all_repo,
    vis_all_repo_up
)

print()
print("[SAVE] Raw contour audit:")
print(f"  {path_all_otsu}")
print(f"  {path_all_adaptive}")
print(f"  {path_all_repo}")


# ============================================================
# SIDE-BY-SIDE RAW CONTOUR COMPARISON
# ============================================================

all_contours_comparison = cv2.hconcat(
    [
        vis_all_otsu_up,
        vis_all_adaptive_up,
        vis_all_repo_up
    ]
)

all_contours_comparison_path = os.path.join(
    OUTPUT_DIR,
    "11_ALL_CONTOURS_COMPARISON.jpg"
)

cv2.imwrite(
    all_contours_comparison_path,
    all_contours_comparison
)

print(
    f"[SAVE] {all_contours_comparison_path}"
)


# ============================================================
# FINAL SUMMARY
# ============================================================

print()
print("=" * 70)
print("FINAL SUMMARY")
print("=" * 70)

print()
print("FILTERED CHARACTER CANDIDATES:")
print(
    f"  RAW GRAYSCALE + OTSU      : "
    f"{len(candidates_otsu)}"
)

print(
    f"  RAW GRAYSCALE + ADAPTIVE  : "
    f"{len(candidates_adaptive_raw)}"
)

print(
    f"  REPOSITORY PIPELINE       : "
    f"{len(candidates_repo)}"
)

print()
print("ALL RAW CONTOURS:")
print(
    f"  RAW GRAYSCALE + OTSU      : "
    f"{len(all_otsu)}"
)

print(
    f"  RAW GRAYSCALE + ADAPTIVE  : "
    f"{len(all_adaptive_raw)}"
)

print(
    f"  REPOSITORY PIPELINE       : "
    f"{len(all_repo)}"
)

print()
print("Output directory:")
print(OUTPUT_DIR)

print()
print("FILTERED RESULTS:")
print("  01_otsu_candidates.jpg")
print("  02_raw_adaptive_candidates.jpg")
print("  03_repo_candidates.jpg")
print("  07_SEGMENTATION_COMPARISON.jpg")

print()
print("RAW CONTOUR AUDIT:")
print("  08_ALL_CONTOURS_OTSU.jpg")
print("  09_ALL_CONTOURS_RAW_ADAPTIVE.jpg")
print("  10_ALL_CONTOURS_REPO.jpg")
print("  11_ALL_CONTOURS_COMPARISON.jpg")

print()
print("PENTING:")
print("  Jangan simpulkan jumlah karakter dari FILTERED")
print("  sebelum melihat hasil RAW CONTOUR AUDIT.")

print("=" * 70)