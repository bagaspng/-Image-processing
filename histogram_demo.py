import argparse
from pathlib import Path

import cv2
import numpy as np
import matplotlib.pyplot as plt


def compute_hist_grayscale(gray):
    """Histogram grayscale: 256 bin (0..255)."""
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256]).ravel()
    return hist


def compute_hist_rgb(rgb):
    """Histogram RGB per-channel (R, G, B)."""
    h_r = cv2.calcHist([rgb], [0], None, [256], [0, 256]).ravel()  # R
    h_g = cv2.calcHist([rgb], [1], None, [256], [0, 256]).ravel()  # G
    h_b = cv2.calcHist([rgb], [2], None, [256], [0, 256]).ravel()  # B
    return h_r, h_g, h_b


def compute_hist_binary(binary):
    """
    Histogram biner (idealnya dua spike: 0 dan 255).
    Tetap hitung 256 bin agar konsisten, puncak akan muncul di indeks 0 dan 255.
    """
    hist = cv2.calcHist([binary], [0], None, [256], [0, 256]).ravel()
    return hist


def to_binary(gray, thresh=None):
    """
    Thresholding ke citra biner.
    - Jika thresh=None: gunakan Otsu.
    - Jika thresh diset (0..255): gunakan nilai tersebut.
    """
    if thresh is None:
        _, bin_img = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    else:
        t = int(np.clip(thresh, 0, 255))
        _, bin_img = cv2.threshold(gray, t, 255, cv2.THRESH_BINARY)
    return bin_img


def main(image_path, thresh):
    # 1) Baca gambar (BGR), lalu konversi ke RGB & Grayscale
    bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(f"Gambar tidak ditemukan: {image_path}")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)

    # 2) Biner
    binary = to_binary(gray, thresh=thresh)

    # 3) Hitung histogram
    h_gray = compute_hist_grayscale(gray)
    h_r, h_g, h_b = compute_hist_rgb(rgb)
    h_bin = compute_hist_binary(binary)

    # 4) Tampilkan hasil
    plt.figure(figsize=(13, 10))

    # Baris 1: Gambar
    plt.subplot(3, 3, 1)
    plt.imshow(rgb)
    plt.title("RGB")
    plt.axis("off")

    plt.subplot(3, 3, 2)
    plt.imshow(gray, cmap="gray", vmin=0, vmax=255)
    plt.title("Grayscale")
    plt.axis("off")

    plt.subplot(3, 3, 3)
    plt.imshow(binary, cmap="gray", vmin=0, vmax=255)
    plt.title("Binary")
    plt.axis("off")

    # Baris 2: Histogram RGB (3 kanal dipisah)
    plt.subplot(3, 3, 4)
    plt.plot(h_r)
    plt.title("Histogram R")
    plt.xlim([0, 255])
    plt.xlabel("Intensitas")
    plt.ylabel("Frekuensi")

    plt.subplot(3, 3, 5)
    plt.plot(h_g)
    plt.title("Histogram G")
    plt.xlim([0, 255])
    plt.xlabel("Intensitas")
    plt.ylabel("Frekuensi")

    plt.subplot(3, 3, 6)
    plt.plot(h_b)
    plt.title("Histogram B")
    plt.xlim([0, 255])
    plt.xlabel("Intensitas")
    plt.ylabel("Frekuensi")

    # Baris 3: Histogram Grayscale & Binary
    plt.subplot(3, 3, 7)
    plt.plot(h_gray)
    plt.title("Histogram Grayscale")
    plt.xlim([0, 255])
    plt.xlabel("Intensitas")
    plt.ylabel("Frekuensi")

    plt.subplot(3, 3, 8)
    plt.bar(np.arange(256), h_bin, width=1.0)
    plt.title("Histogram Binary (spike di 0 dan 255)")
    plt.xlim([0, 255])
    plt.xlabel("Intensitas")
    plt.ylabel("Frekuensi")

    # Info singkat
    total_pixels = gray.size
    zeros = int(h_bin[0])
    twfiv = int(h_bin[255])
    plt.subplot(3, 3, 9)
    plt.axis("off")
    plt.text(
        0.0, 0.8,
        f"Total piksel : {total_pixels:,}\n"
        f"Binary 0 (hitam): {zeros:,}\n"
        f"Binary 255 (putih): {twfiv:,}\n"
        f"Threshold: {'Otsu' if thresh is None else thresh}",
        fontsize=10
    )

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Demo histogram untuk citra RGB, Grayscale, dan Binary."
    )
    parser.add_argument("--image", required=True, type=Path, help="Path file gambar input (jpg/png, dsb).")
    parser.add_argument(
        "--thresh",
        type=int,
        default=None,
        help="Nilai threshold 0..255 untuk biner. Kosongkan untuk Otsu."
    )
    args = parser.parse_args()
    main(args.image, args.thresh)
