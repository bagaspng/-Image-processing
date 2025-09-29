# file: boundary_minmax_report.py
import os
import argparse
import cv2
import numpy as np
import matplotlib.pyplot as plt

def read_gray(path: str) -> np.ndarray:
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Gagal membaca gambar: {path}")
    return img

def max_filter(gray: np.ndarray, k: int = 3) -> np.ndarray:
    kernel = np.ones((k, k), np.uint8)
    return cv2.dilate(gray, kernel, borderType=cv2
    .BORDER_REFLECT)

def min_filter(gray: np.ndarray, k: int = 3) -> np.ndarray:
    kernel = np.ones((k, k), np.uint8)
    return cv2.erode(gray, kernel, borderType=cv2.BORDER_REFLECT)

def hist_0_255(img: np.ndarray):
    # histogram 256 bin (0..255), nilai count per intensitas
    hist, _ = np.histogram(img.ravel(), bins=256, range=(0, 256))
    return hist

def main():
    ap = argparse.ArgumentParser(description="Laporan 1 gambar: original, min, max, & histogram perbandingan")
    ap.add_argument("image", help="Path gambar grayscale/warna")
    ap.add_argument("--ksize", type=int, default=3, help="Ukuran kernel ganjil untuk min/max (default 3)")
    ap.add_argument("--dpi", type=int, default=180, help="DPI output figure (default 180)")
    args = ap.parse_args()

    if args.ksize % 2 == 0 or args.ksize < 1:
        raise ValueError("ksize harus ganjil dan > 0 (mis. 3,5,7).")

    # 1) Baca & siapkan
    gray = read_gray(args.image)
    base = os.path.splitext(os.path.basename(args.image))[0]
    out_dir = os.path.join(os.path.dirname(args.image), f"{base}_results")
    os.makedirs(out_dir, exist_ok=True)

    # 2) Proses filter min & max
    min_img = min_filter(gray, args.ksize)
    max_img = max_filter(gray, args.ksize)

    # (opsional) simpan hasil terpisah juga
    cv2.imwrite(os.path.join(out_dir, f"{base}_min{args.ksize}x{args.ksize}.png"), min_img)
    cv2.imwrite(os.path.join(out_dir, f"{base}_max{args.ksize}x{args.ksize}.png"), max_img)

    # 3) Hitung histogram
    h_orig = hist_0_255(gray)
    h_min  = hist_0_255(min_img)
    h_max  = hist_0_255(max_img)
    xbins = np.arange(256)

    # 4) Gambar figure besar: 2 baris × 3 kolom
    #   [0,0] original, [0,1] min, [0,2] max
    #   [1,0] hist (orig vs min), [1,1] hist (orig vs max), [1,2] keterangan
    fig = plt.figure(figsize=(15, 8), dpi=args.dpi)

    ax1 = plt.subplot(2, 3, 1)
    ax1.imshow(gray, cmap="gray", vmin=0, vmax=255)
    ax1.set_title("Original")
    ax1.axis("off")

    ax2 = plt.subplot(2, 3, 2)
    ax2.imshow(min_img, cmap="gray", vmin=0, vmax=255)
    ax2.set_title(f"Min filter {args.ksize}×{args.ksize}")
    ax2.axis("off")

    ax3 = plt.subplot(2, 3, 3)
    ax3.imshow(max_img, cmap="gray", vmin=0, vmax=255)
    ax3.set_title(f"Max filter {args.ksize}×{args.ksize}")
    ax3.axis("off")

    ax4 = plt.subplot(2, 3, 4)
    ax4.plot(xbins, h_orig, label="Original")
    ax4.plot(xbins, h_min,  label="Min")
    ax4.set_xlim(0, 255)
    ax4.set_title("Histogram: Original vs Min")
    ax4.set_xlabel("Intensitas (0–255)")
    ax4.set_ylabel("Jumlah piksel")
    ax4.legend()

    ax5 = plt.subplot(2, 3, 5)
    ax5.plot(xbins, h_orig, label="Original")
    ax5.plot(xbins, h_max,  label="Max")
    ax5.set_xlim(0, 255)
    ax5.set_title("Histogram: Original vs Max")
    ax5.set_xlabel("Intensitas (0–255)")
    ax5.set_ylabel("Jumlah piksel")
    ax5.legend()

    ax6 = plt.subplot(2, 3, 6)
    ax6.axis("off")
    ax6.set_title("Catatan")
    txt = [
        "Min filter (erosion): cenderung menurunkan intensitas terang,",
        "menonjolkan area gelap & menipiskan objek terang.",
        "",
        "Max filter (dilation): cenderung menaikkan intensitas,",
        "menebalkan objek terang & mengisi celah kecil.",
        "",
        "Histogram:",
        "• Kurva Min bergeser ke kiri (lebih gelap)",
        "• Kurva Max bergeser ke kanan (lebih terang)"
    ]
    ax6.text(0, 1, "\n".join(txt), va="top")

    plt.tight_layout()
    out_path = os.path.join(out_dir, f"{base}_boundary_minmax_report.png")
    fig.savefig(out_path, dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)

    print("✔️ Simpan report:", out_path)
    print("✔️ Simpan min   :", os.path.join(out_dir, f"{base}_min{args.ksize}x{args.ksize}.png"))
    print("✔️ Simpan max   :", os.path.join(out_dir, f"{base}_max{args.ksize}x{args.ksize}.png"))

if __name__ == "__main__":
    main()
