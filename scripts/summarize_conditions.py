"""
summarize_conditions.py – tổng hợp conditions_log.csv cho báo cáo

    python summarize_conditions.py [conditions_log.csv]

Theo từng điều kiện (ánh sáng, góc mặt, khoảng cách, che khuất):
    tỉ lệ phát hiện mặt = khung có mặt / tổng khung ;  tỉ lệ đúng = khung đoán đúng / khung có mặt
    -> conditions_summary.csv, chart_<yếu tố>.png
Trên toàn bộ dữ liệu, theo từng tổ hợp detector + model (+ no-align):
    Accuracy, Precision, Recall, F1 từng lớp, macro-F1  -> metrics_overall.csv, metrics_per_class.csv
    Confusion matrix                                    -> confusion_<tổ hợp>.png
Lưu ý khi viết báo cáo: số người thử ít, biểu cảm do người thử diễn (fear/disgust khó diễn giống thật)
-> kết quả chỉ mang tính tham khảo.
"""
import os
import re
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "core"))
HERE = os.path.join(ROOT, "results")
os.makedirs(HERE, exist_ok=True)
path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "conditions_log.csv")
df = pd.read_csv(path, dtype={"angle": str, "dist": str, "light": str, "occl": str})
df["correct"] = pd.to_numeric(df["correct"], errors="coerce")
if "align" not in df.columns:
    df["align"] = "on"
df["combo"] = df["detector"] + " + " + df["model"] + \
    np.where((df["detector"] == "mediapipe") & (df["align"] == "off"), " (no-align)", "")

LABELS = ["surprise", "fear", "disgust", "happy", "sad", "angry", "neutral"]
FACTORS = {"light": "Ánh sáng", "angle": "Góc mặt (độ)", "dist": "Khoảng cách (m)", "occl": "Che khuất"}
ORDER = {"light": ["sang_deu", "binh_thuong", "thieu_sang", "nguoc_sang"],
         "angle": ["0", "30", "45"], "dist": ["0.5", "1", "2", "3"], "occl": ["none", "kinh", "khau_trang"]}

# ------------------------------------------------------------ theo điều kiện
parts = []
for f in FACTORS:
    g = df.groupby(["combo", f]).agg(frames=("face_detected", "size"),
                                     detect_rate=("face_detected", "mean"),
                                     accuracy=("correct", "mean"),
                                     frames_with_face=("correct", "count")).reset_index()
    g = g.rename(columns={f: "level"})
    g.insert(1, "factor", f)
    parts.append(g)
summary = pd.concat(parts)
summary.round(3).to_csv(os.path.join(HERE, "conditions_summary.csv"), index=False)
print(summary.round(3).to_string(index=False))

for f, title in FACTORS.items():
    sub = summary[summary.factor == f]
    if sub.level.nunique() < 2:
        continue                      # chưa đủ mức để so sánh
    order = [l for l in ORDER[f] if l in set(sub.level)] + sorted(set(sub.level) - set(ORDER[f]))
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, col, name in zip(axes, ["detect_rate", "accuracy"], ["Tỉ lệ phát hiện mặt", "Tỉ lệ đoán đúng"]):
        p = sub.pivot_table(index="level", columns="combo", values=col).reindex(order) * 100
        p.plot.bar(ax=ax, rot=0, width=0.8)
        ax.set_title(f"{name} theo {title.lower()}")
        ax.set_xlabel(title); ax.set_ylabel("%"); ax.set_ylim(0, 105)
        ax.legend(title=None, fontsize=8)
        ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(HERE, f"chart_{f}.png"), dpi=150)
    plt.close()

# ------------------------------------------------------------ Accuracy / P / R / F1 / confusion matrix
overall, per_class = [], []
for combo, g in df.groupby("combo"):
    d = g[g["label_pred"].notna() & (g["label_pred"] != "")]          # chỉ khung có phát hiện mặt
    cm = np.zeros((len(LABELS), len(LABELS)), int)                     # hàng = thật, cột = đoán
    for t, p in zip(d["expr_true"], d["label_pred"]):
        cm[LABELS.index(t), LABELS.index(p)] += 1
    f1s = []
    for i, lab in enumerate(LABELS):
        tp, sup, npred = cm[i, i], cm[i].sum(), cm[:, i].sum()
        pr = tp / npred if npred else np.nan
        rc = tp / sup if sup else np.nan
        f1 = 2 * pr * rc / (pr + rc) if pr == pr and rc == rc and (pr + rc) > 0 else (0.0 if sup else np.nan)
        if sup:
            f1s.append(f1)
        per_class.append({"combo": combo, "label": lab, "precision": pr, "recall": rc, "f1": f1, "support": int(sup)})
    overall.append({"combo": combo, "frames": len(g), "detect_rate": g["face_detected"].mean(),
                    "accuracy": np.trace(cm) / cm.sum() if cm.sum() else np.nan,
                    "macro_f1": float(np.mean(f1s)) if f1s else np.nan})
    fig, ax = plt.subplots(figsize=(6, 5.2))
    norm = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)
    ax.imshow(norm, vmin=0, vmax=1, cmap="Blues")
    ax.set_xticks(range(len(LABELS))); ax.set_xticklabels(LABELS, rotation=45, ha="right")
    ax.set_yticks(range(len(LABELS))); ax.set_yticklabels(LABELS)
    for i in range(len(LABELS)):
        for j in range(len(LABELS)):
            if cm[i, j]:
                ax.text(j, i, cm[i, j], ha="center", va="center",
                        color="white" if norm[i, j] > 0.5 else "black", fontsize=8)
    ax.set_xlabel("Dự đoán"); ax.set_ylabel("Thật"); ax.set_title(f"Confusion matrix – {combo}")
    plt.tight_layout()
    plt.savefig(os.path.join(HERE, "confusion_" + re.sub(r"[^A-Za-z0-9]+", "_", combo).strip("_") + ".png"), dpi=150)
    plt.close()

pd.DataFrame(overall).round(3).to_csv(os.path.join(HERE, "metrics_overall.csv"), index=False)
pd.DataFrame(per_class).round(3).to_csv(os.path.join(HERE, "metrics_per_class.csv"), index=False)
print("\n=== Tổng thể ===")
print(pd.DataFrame(overall).round(3).to_string(index=False))
print("\nĐã lưu conditions_summary.csv, chart_*.png, metrics_overall.csv, metrics_per_class.csv, confusion_*.png")
