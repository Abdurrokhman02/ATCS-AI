import os
import cv2
from paddleocr import PaddleOCR
import paddle

paddle.set_flags({
    "FLAGS_use_mkldnn": False
})

CROP_DIR = "./output/debug_plat_crops"

ocr = PaddleOCR(
    lang="en",
    use_doc_orientation_classify=False,
    use_doc_unwarping=False,
    use_textline_orientation=False,
)

files = [
    f for f in os.listdir(CROP_DIR)
    if f.lower().endswith((".jpg", ".jpeg", ".png"))
]

print(f"[INFO] Ditemukan {len(files)} crop plat")

for filename in files:
    path = os.path.join(CROP_DIR, filename)

    image = cv2.imread(path)

    if image is None:
        print(f"[WARNING] Gagal membaca: {filename}")
        continue

    result = ocr.predict(image)

    print("\n" + "=" * 60)
    print(f"FILE : {filename}")

    found = False

    for res in result:
        data = res.json

        if callable(data):
            data = data()

        if not isinstance(data, dict):
            continue

        data = data.get("res", data)

        texts = data.get("rec_texts", [])
        scores = data.get("rec_scores", [])

        for text, score in zip(texts, scores):
            print(f"OCR  : {text}")
            print(f"CONF : {float(score):.3f}")
            found = True

    if not found:
        print("OCR  : TIDAK TERBACA")

print("\n[SELESAI]")