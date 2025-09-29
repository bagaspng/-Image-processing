# file: pick_strong_edges.py
import pandas as pd
import sys

csv_path = r"D:\Documents\smt5\Pengcit\foto_results\foto_boundary_pixel_changes.csv"
df = pd.read_csv(csv_path)

# 1) Top-N boundary terkuat
TOPN = 50
top = df.sort_values("boundary_max_minus_min", ascending=False).head(TOPN)
top.to_csv(csv_path.replace(".csv", f"_top{TOPN}.csv"), index=False)
print(f"Simpan TOP{TOPN} ->", csv_path.replace(".csv", f"_top{TOPN}.csv"))

# 2) (opsional) pakai ambang minimum boundary
TH = 10
strong = df[df["boundary_max_minus_min"] >= TH].sort_values("boundary_max_minus_min", ascending=False)
strong.to_csv(csv_path.replace(".csv", f"_boundary_ge_{TH}.csv"), index=False)
print(f"Simpan boundary >= {TH} ->", csv_path.replace(".csv", f"_boundary_ge_{TH}.csv"))
