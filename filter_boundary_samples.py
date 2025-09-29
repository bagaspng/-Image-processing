# file: filter_boundary_samples.py
import argparse, os, cv2, numpy as np, csv

def read_gray(path):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Gagal baca gambar: {path}")
    return img

def max_filter(gray, k=3):
    return cv2.dilate(gray, np.ones((k,k), np.uint8), borderType=cv2.BORDER_REFLECT)

def min_filter(gray, k=3):
    return cv2.erode(gray,  np.ones((k,k), np.uint8), borderType=cv2.BORDER_REFLECT)

def parse_samples(s):
    if not s: return []
    pts = []
    for tok in s.split(";"):
        tok = tok.strip()
        if not tok: continue
        x,y = tok.split(",")
        pts.append((int(x), int(y)))
    return pts

def default_samples(w,h, n=9):
    # ambil titik representatif: 4 sudut, sisi tengah, dan pusat
    cand = [(0,0),(w//2,0),(w-1,0),(0,h//2),(w//2,h//2),(w-1,h//2),(0,h-1),(w//2,h-1),(w-1,h-1)]
    # pastikan unik & dalam batas
    uniq = []
    for x,y in cand:
        if 0<=x<w and 0<=y<h and (x,y) not in uniq:
            uniq.append((x,y))
    return uniq[:n]

def save_img(path, img):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    ok = cv2.imwrite(path, img)
    print(("✔️ simpan:" if ok else "❌ gagal:"), path)

def clamp_samples(samples, w, h):
    valid = []
    for x,y in samples:
        if 0<=x<w and 0<=y<h: valid.append((x,y))
    return valid

def main():
    ap = argparse.ArgumentParser(description="Filter batas (max-min) + sampel perubahan piksel")
    ap.add_argument("image", help="Path gambar (grayscale atau warna)")
    ap.add_argument("--ksize", type=int, default=3, help="Ukuran kernel (ganjil), default 3")
    ap.add_argument("--samples", default="", help='Koordinat sampel "x1,y1;x2,y2" (opsional)')
    ap.add_argument("--save-samples-csv", action="store_true", help="Simpan CSV kecil untuk sampel")
    args = ap.parse_args()

    if args.ksize % 2 == 0 or args.ksize < 1:
        raise ValueError("ksize harus ganjil dan >0 (mis. 3,5,7)")

    # 1) Baca & siapkan output dir
    gray = read_gray(args.image)
    h, w = gray.shape
    base = os.path.splitext(os.path.basename(args.image))[0]
    out_dir = os.path.join(os.path.dirname(args.image), f"{base}_results")
    os.makedirs(out_dir, exist_ok=True)
    print("📄 input :", args.image)
    print("📂 output:", out_dir)
    print("🧱 ksize :", args.ksize)

    # 2) Proses filter
    max3 = max_filter(gray, args.ksize)
    min3 = min_filter(gray, args.ksize)
    boundary = (max3.astype(np.int16) - min3.astype(np.int16)).astype(np.uint8)

    # 3) Simpan gambar hasil
    save_img(os.path.join(out_dir, f"{base}_max{args.ksize}x{args.ksize}.png"), max3)
    save_img(os.path.join(out_dir, f"{base}_min{args.ksize}x{args.ksize}.png"), min3)
    save_img(os.path.join(out_dir, f"{base}_boundary_max_minus_min.png"), boundary)

    # 4) Pilih sampel
    samples = parse_samples(args.samples)
    if not samples:
        samples = default_samples(w, h)
        print("ℹ️ samples otomatis:", "; ".join([f"{x},{y}" for x,y in samples]))
    samples = clamp_samples(samples, w, h)

    # 5) Cetak perubahan nilai pada sampel
    print("\n=== Sampel perubahan piksel (x,y) ===")
    header = f"{'x':>5} {'y':>5} {'orig':>6} {'max':>6} {'Δmax':>6} {'min':>6} {'Δmin':>6} {'bound=max-min':>14}"
    print(header)
    print("-"*len(header))
    rows = []
    for x,y in samples:
        orig = int(gray[y, x])
        mx   = int(max3[y, x])
        mn   = int(min3[y, x])
        bd   = int(boundary[y, x])
        dmx  = mx - orig
        dmn  = mn - orig
        print(f"{x:5d} {y:5d} {orig:6d} {mx:6d} {dmx:6d} {mn:6d} {dmn:6d} {bd:14d}")
        rows.append((x,y,orig,mx,dmx,mn,dmn,bd))

    # 6) (opsional) simpan CSV kecil (hanya sampel)
    if args.save_samples_csv:
        csv_path = os.path.join(out_dir, f"{base}_boundary_samples.csv")
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            wr = csv.writer(f)
            wr.writerow(["x","y","orig","max","delta_max","min","delta_min","boundary_max_minus_min"])
            wr.writerows(rows)
        print("\n✔️ CSV sampel:", csv_path)

    print("\n✅ selesai.")

if __name__ == "__main__":
    main()
