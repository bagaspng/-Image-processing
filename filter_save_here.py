# file: filter_save_here.py
import sys, os
import cv2
import numpy as np

if len(sys.argv) < 2:
    print("Usage: python filter_save_here.py <path_gambar>")
    sys.exit(1)

img_path = sys.argv[1]
print("📄 Input :", img_path)
img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
if img is None:
    raise FileNotFoundError("Gagal membaca gambar. Cek path/nama file.")

# === Filter ===
mean3   = cv2.blur(img, (3,3))
median3 = cv2.medianBlur(img, 3)
gx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
gy = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
mag = np.hypot(gx, gy); m = mag.max()
sobel_mag = (mag/m*255).astype(np.uint8) if m>0 else np.zeros_like(img, np.uint8)

# === Simpan di folder yang sama dengan gambar ===
in_dir  = os.path.dirname(img_path)
in_name = os.path.splitext(os.path.basename(img_path))[0]
out_dir = os.path.join(in_dir, f"{in_name}_results")
os.makedirs(out_dir, exist_ok=True)

def save(p, a):
    ok = cv2.imwrite(p, a)
    print(("✔️ Simpan:" if ok else "❌ Gagal:"), p)

save(os.path.join(out_dir, f"{in_name}_mean3x3.png"), mean3)
save(os.path.join(out_dir, f"{in_name}_median3x3.png"), median3)
save(os.path.join(out_dir, f"{in_name}_sobel3_mag.png"), sobel_mag)

# Bonus: diff map (biar kelihatan piksel mana yang berubah)
diff_mean   = cv2.absdiff(img, mean3)
diff_median = cv2.absdiff(img, median3)
save(os.path.join(out_dir, f"{in_name}_diff_mean.png"), diff_mean)
save(os.path.join(out_dir, f"{in_name}_diff_median.png"), diff_median)

print("✅ Selesai. Folder output:", out_dir)
