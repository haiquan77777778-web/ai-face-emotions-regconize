"""
summarize_by_expr.py - xem ket qua kiem thu dieu kien THEO TUNG BIEU CAM, de so sanh cong bang.

    python summarize_by_expr.py [conditions_log.csv]

Vi sao can: neu dieu kien A quay 7 bieu cam con dieu kien B chi quay "happy" (de nhan), thi accuracy chung cua B se cao
hon A chi vi bieu cam de hon, khong phai vi dieu kien tot hon. Bang nay tach theo bieu cam nen so duoc dung.

In ra:
  1. Danh sach tung luot da quay (de biet con thieu luot nao)
  2. Bang pivot: moi dong = 1 dieu kien (detector + anh sang + goc + khoang cach + che khuat), moi cot = 1 bieu cam,
     o = ti le doan dung (so khung)
Luu:  conditions_runs.csv, conditions_by_expr.csv
"""
import os
import sys

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "core"))
HERE = os.path.join(ROOT, "results")
os.makedirs(HERE, exist_ok=True)
path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "conditions_log.csv")
df = pd.read_csv(path)
for c in ("angle", "dist"):
    df[c] = df[c].astype(str)
df["correct"] = pd.to_numeric(df["correct"], errors="coerce")
df["align"] = df["align"].astype(str)
EXPR = ["surprise", "fear", "disgust", "happy", "sad", "angry", "neutral"]


def cond_name(r):
    parts = [r["detector"], r["model"], f"sang={r['light']}", f"goc={r['angle']}", f"xa={r['dist']}m"]
    if str(r["occl"]) != "none":
        parts.append(f"che={r['occl']}")
    if r["align"] == "off":
        parts.append("KHONG can chinh")
    return " | ".join(parts)


df["cond"] = df.apply(cond_name, axis=1)
keys = ["who", "expr_true", "cond"]

runs = (df.groupby(keys, sort=False)
          .apply(lambda g: pd.Series({
              "frames": len(g),
              "detect_rate": g["face_detected"].mean(),
              "accuracy": g["correct"].mean(),
              "most_predicted": g["label_pred"].mode().iloc[0] if g["label_pred"].notna().any() else "",
          }), include_groups=False)
          .reset_index())
runs.to_csv(os.path.join(HERE, "conditions_runs.csv"), index=False, encoding="utf-8-sig")
print("=== 1. Cac luot da quay ===")
pd.set_option("display.width", 250, "display.max_colwidth", 80, "display.max_rows", 500)
print(runs.assign(detect_rate=runs["detect_rate"].map("{:.0%}".format),
                  accuracy=runs["accuracy"].map("{:.0%}".format)).to_string(index=False))

piv = (df.groupby(["cond", "expr_true"])["correct"].mean().unstack("expr_true")
         .reindex(columns=[e for e in EXPR if e in set(df["expr_true"])]))
n = df.groupby("cond").size().rename("frames")
out = piv.join(n)
out.to_csv(os.path.join(HERE, "conditions_by_expr.csv"), encoding="utf-8-sig")
show = out.copy()
for c in piv.columns:
    show[c] = show[c].map(lambda v: "" if pd.isna(v) else f"{v:.0%}")
print("\n=== 2. Ti le doan dung theo bieu cam (hang = dieu kien) ===")
print(show.to_string())
print("\nO trong = chua quay bieu cam do o dieu kien do. Chi so sanh cac dieu kien tren CUNG MOT cot.")
print("Da luu: conditions_runs.csv, conditions_by_expr.csv")
