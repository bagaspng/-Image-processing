import numpy as np

# Citra 3x3 (grayscale)
img = np.array([
    [10, 20, 30],
    [20, 40, 60],
    [30, 60, 90]
], dtype=np.float64)

# 1) Mean & Median dari seluruh blok 3x3
mean_value = img.mean()
median_value = np.median(img)

print(f"Mean 3x3   : {mean_value:.2f}")   # harusnya 40.00
print(f"Median 3x3 : {median_value:.2f}") # harusnya 30.00

# 2) Sobel Edge (3x3) – konvolusi pada seluruh blok → satu output
sobel_x = np.array([
    [-1, 0, 1],
    [-2, 0, 2],
    [-1, 0, 1]
], dtype=np.float64)

sobel_y = np.array([
    [-1, -2, -1],
    [ 0,  0,  0],
    [ 1,  2,  1]
], dtype=np.float64)

Gx = np.sum(img * sobel_x)
Gy = np.sum(img * sobel_y)
Gmag = np.hypot(Gx, Gy)  # sqrt(Gx^2 + Gy^2)

print(f"Sobel Gx   : {Gx:.2f}")   # contoh perhitungan manual sebelumnya: 160
print(f"Sobel Gy   : {Gy:.2f}")
print(f"Sobel |G|  : {Gmag:.2f}")
