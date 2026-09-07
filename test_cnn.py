import os
import cv2
import numpy as np
from keras.models import load_model

from anpr.cnn_model import CNN_Model
from anpr.utils_lp import ALPHA_DICT


WEIGHTS_PATH = "./anpr/weights/weight.h5"
TEST_DIR = "./output/cnn_test"


def build_model():
    cnn = CNN_Model()
    cnn.model.load_weights(WEIGHTS_PATH)
    return cnn.model


def predict_character(model, image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    if img is None:
        raise ValueError(f"Gagal membaca: {image_path}")

    original_shape = img.shape

    # Ikuti preprocessing inference repo ASLI
    img_roi = cv2.resize(
        img,
        (28, 28),
        interpolation=cv2.INTER_AREA,
    )

    img_roi = img_roi.reshape((28, 28, 1))
    img_roi = np.expand_dims(
        np.array(img_roi),
        axis=0,
    )

    result = model.predict(
        img_roi,
        verbose=0,
    )[0]

    top_indices = np.argsort(result)[::-1][:5]

    return (
        original_shape,
        result,
        top_indices,
    )


def main():
    print("=" * 70)
    print("CNN EXISTING MODEL TEST")
    print("=" * 70)

    print(f"Weights : {WEIGHTS_PATH}")
    print(f"Test dir: {TEST_DIR}")
    print()

    if not os.path.exists(WEIGHTS_PATH):
        print("ERROR: weight.h5 tidak ditemukan.")
        return

    if not os.path.exists(TEST_DIR):
        os.makedirs(TEST_DIR)
        print(f"Folder test dibuat: {TEST_DIR}")
        print()
        print("Masukkan beberapa gambar karakter ke folder tersebut.")
        print("Contoh:")
        print("  P.png")
        print("  9.png")
        print("  7.png")
        print()
        return

    image_files = [
        f
        for f in os.listdir(TEST_DIR)
        if f.lower().endswith(
            (".jpg", ".jpeg", ".png", ".bmp")
        )
    ]

    if not image_files:
        print("Belum ada gambar karakter di folder test.")
        print()
        print("Masukkan gambar karakter individual ke:")
        print(TEST_DIR)
        return

    print("Loading CNN architecture...")
    model = build_model()

    print("Model loaded successfully.")
    print()

    for filename in sorted(image_files):

        image_path = os.path.join(
            TEST_DIR,
            filename,
        )

        try:
            shape, result, top_indices = predict_character(
                model,
                image_path,
            )

            predicted_idx = int(top_indices[0])
            predicted_char = ALPHA_DICT[predicted_idx]

            print("-" * 70)
            print(f"IMAGE      : {filename}")
            print(f"ORIGINAL   : {shape[1]} x {shape[0]}")
            print()
            print(f"PREDICTION : {predicted_char}")
            print(
                f"CONFIDENCE : {result[predicted_idx] * 100:.2f}%"
            )
            print()
            print("TOP 5:")

            for rank, idx in enumerate(top_indices, start=1):
                char = ALPHA_DICT[int(idx)]
                confidence = result[int(idx)] * 100

                print(
                    f"  {rank}. "
                    f"{char:<10} "
                    f"{confidence:>7.2f}%"
                )

        except Exception as e:
            print("-" * 70)
            print(f"ERROR: {filename}")
            print(e)

    print()
    print("=" * 70)
    print("TEST SELESAI")
    print("=" * 70)


if __name__ == "__main__":
    main()