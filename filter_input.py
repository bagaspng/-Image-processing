import cv2
import numpy as np

# --- Input gambar ---
img_rgb = cv2.imread(r"D:\Documents\smt5\Pengcit\foto.png")
img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)

# --- 1. Mean Filter (blur rata-rata) ---
mean3 = cv2.blur(img_gray, (3, 3))   # kernel 3x3
mean5 = cv2.blur(img_gray, (5, 5))   # kernel 5x5

# --- 2. Median Filter ---
median3 = cv2.medianBlur(img_gray, 3)  # kernel 3x3
median5 = cv2.medianBlur(img_gray, 5)  # kernel 5x5

# --- 3. Sobel Edge Detection ---
sobelx = cv2.Sobel(img_gray, cv2.CV_64F, 1, 0, ksize=3)  # deteksi tepi arah X
sobely = cv2.Sobel(img_gray, cv2.CV_64F, 0, 1, ksize=3)  # deteksi tepi arah Y
sobel_mag = np.hypot(sobelx, sobely)                     # magnitude
sobel_mag = (sobel_mag / sobel_mag.max() * 255).astype(np.uint8)

# --- Simpan hasil ---
cv2.imwrite("mean3.png", mean3)
cv2.imwrite("median3.png", median3)
cv2.imwrite("sobel.png", sobel_mag)

print("Filter selesai → cek file mean3.png, median3.png, sobel.png")
