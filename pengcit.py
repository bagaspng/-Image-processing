import cv2
import matplotlib.pyplot as plt

# list gambar wajah (ganti dengan file Anda)
gambar_list = ["muka1.jpg", "muka2.jpg", "muka3.jpg"]

for i, path in enumerate(gambar_list, start=1):
    # Baca gambar dalam RGB
    img_rgb = cv2.imread(r"D:\Documents\smt5\Pengcit\Screenshot 2024-10-29 205359.png")                # default BGR
    img_rgb = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2RGB)  # konversi ke RGB

    # Konversi ke grayscale
    img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)

    # Konversi ke biner (thresholding)
    _, img_bin = cv2.threshold(img_gray, 127, 255, cv2.THRESH_BINARY)

    # Tampilkan semua versi
    plt.figure(figsize=(10,4))
    
    plt.subplot(1,3,1)
    plt.imshow(img_rgb)
    plt.title(f"Gambar {i} - RGB")
    plt.axis("off")

    plt.subplot(1,3,2)
    plt.imshow(img_gray, cmap="gray")
    plt.title(f"Gambar {i} - Grayscale")
    plt.axis("off")

    plt.subplot(1,3,3)
    plt.imshow(img_bin, cmap="gray")
    plt.title(f"Gambar {i} - Biner")
    plt.axis("off")

    plt.show()
