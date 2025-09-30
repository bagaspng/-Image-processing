import argparse
from pathlib import Path
import cv2
import numpy as np
import matplotlib.pyplot as plt


def save_image(img, filename, cmap=None):
    """Simpan citra ke file."""
    plt.figure()
    if cmap:
        plt.imshow(img, cmap=cmap, vmin=0, vmax=255)
    else:
        plt.imshow(img)
    plt.axis("off")
    plt.title(filename)
    plt.savefig(filename, bbox_inches="tight")
    plt.close()


def save_histogram_gray_or_bin(hist_data, filename, color="k", title="Histogram"):
    """Simpan histogram grayscale/biner."""
    plt.figure()
    plt.plot(hist_data, color=color)
    plt.xlim([0, 255])
    plt.xlabel("Intensitas")
    plt.ylabel("Frekuensi")
    plt.title(title)
    plt.savefig(filename, bbox_inches="tight")
    plt.close()


def save_histogram_rgb(h_r, h_g, h_b, filename):
    """Simpan histogram RGB dalam 3 subplot (Red, Green, Blue)."""
    plt.figure(figsize=(10, 6))

    plt.subplot(3, 1, 1)
    plt.plot(h_r, color="r")
    plt.title("Histogram Red")
    plt.xlim([0, 255])

    plt.subplot(3, 1, 2)
    plt.plot(h_g, color="g")
    plt.title("Histogram Green")
    plt.xlim([0, 255])

    plt.subplot(3, 1, 3)
    plt.plot(h_b, color="b")
    plt.title("Histogram Blue")
    plt.xlim([0, 255])

    plt.tight_layout()
    plt.savefig(filename, bbox_inches="tight")
    plt.close()


def compute_hist(img, channel=None):
    """Hitung histogram (0..255)."""
    if channel is None:  # grayscale/biner
        return cv2.calcHist([img], [0], None, [256], [0, 256]).ravel()
    else:  # channel tertentu
        return cv2.calcHist([img], [channel], None, [256], [0, 256]).ravel()


def main(image_path, thresh):
    # Baca gambar
    bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)

    # Threshold biner
    if thresh is None:
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    else:
        _, binary = cv2.threshold(gray, int(thresh), 255, cv2.THRESH_BINARY)

    out_dir = Path("hasil_output")
    out_dir.mkdir(exist_ok=True)

    # Simpan RGB
    save_image(rgb, str(out_dir / "rgb_image.png"))
    h_r = compute_hist(rgb, 0)
    h_g = compute_hist(rgb, 1)
    h_b = compute_hist(rgb, 2)
    save_histogram_rgb(h_r, h_g, h_b, str(out_dir / "rgb_histogram.png"))

    # Simpan Grayscale
    save_image(gray, str(out_dir / "grayscale_image.png"), cmap="gray")
    h_gray = compute_hist(gray)
    save_histogram_gray_or_bin(h_gray, str(out_dir / "grayscale_histogram.png"),
                               color="black", title="Histogram Grayscale")

    # Simpan Binary
    save_image(binary, str(out_dir / "binary_image.png"), cmap="gray")
    h_bin = compute_hist(binary)
    save_histogram_gray_or_bin(h_bin, str(out_dir / "binary_histogram.png"),
                               color="blue", title="Histogram Binary")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pisahkan output histogram per citra")
    parser.add_argument("--image", required=True, type=Path, help="Path gambar input")
    parser.add_argument("--thresh", type=int, default=None, help="Threshold (0..255), kosong = Otsu")
    args = parser.parse_args()
    main(args.image, args.thresh)
